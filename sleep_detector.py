import time
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import os
import torch
from collections import deque
class SleepDetector:
    def __init__(self, pose_model_path='yolo26s-pose.engine', 
                 mp_model_path='models/face_landmarker.task', 
                 pose_model_instance=None, device='cuda', lock=None):
        """
        Initialize Sleep Detector.
        
        IMPROVED: Now supports both shared and private pose models.
        - If pose_model_instance is None: loads its own private model (RECOMMENDED)
        - If pose_model_instance is provided: uses shared model (legacy)
        
        Private models provide better thread isolation.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"my device {self.device}")
        
        self.lock = lock
        
        # 1. Load YOLO Pose
        if pose_model_instance:
            # Legacy: Use shared pose model
            print("    → Using shared pose model")
            self.pose_model = pose_model_instance
        else:
            # IMPROVED: Load private pose model
            print(f"    → Loading private pose model from {pose_model_path}")
            self.pose_model = YOLO(pose_model_path)
            # TensorRT models don't support .to(device)
            if pose_model_path.endswith('.pt'):
                self.pose_model.to(self.device)
            print("    → Private pose model loaded")

        # 2. Load MediaPipe Face Landmarker
        print(f"    → Loading MediaPipe Face Landmarker from {mp_model_path}")
        if not os.path.exists(mp_model_path):
            print(f"    → Warning: {mp_model_path} not found. Face detection disabled.")
            self.detector = None
        else:
            try:
                base_options = python.BaseOptions(model_asset_path=mp_model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
                print("    → MediaPipe Face Landmarker loaded")
            except Exception as e:
                print(f"    → Error loading MediaPipe: {e}")
                self.detector = None

        # === RESEARCH-BACKED CONFIGURATION ===
        
        # EAR (Eye Aspect Ratio) Settings
        self.EAR_THRESHOLD = 0.22           # Below this = eyes closed
        self.EAR_CONSEC_FRAMES = 3          # Min consecutive frames for "closed"
        
        # PERCLOS Setting s (Percentage of Eye Closure)
        self.PERCLOS_WINDOW = 30            # Frames to calculate PERCLOS (~1 sec at 30fps)
        self.PERCLOS_DROWSY = 0.40          # 40% eyes closed = drowsy
        self.PERCLOS_SLEEP = 0.70           # 70% eyes closed = likely sleeping
        
        # Scale-Invariant Posture Thresholds (ratios of shoulder_width)
        self.HEAD_DROP_RATIO = 0.50         # 50% of shoulder width = slumped
        self.TORSO_COLLAPSE_RATIO = 0.40    # Torso < 40% of shoulder width = collapsed
        
        # Multi-Signal Scoring Weights (sum to 1.0)
        # REMOVED: Head Stillness (too short/flaky) and Head Tilt (false positives)
        self.SCORE_WEIGHTS = {
            'perclos_high': 0.40,       # Increased from 0.30
            'eyes_closed': 0.30,        # Increased from 0.25
            'head_dropped': 0.20,       # Increased from 0.15
            'sleep_posture': 0.10,      # Increased from 0.05
        }
        
        # Scoring Thresholds
        self.SCORE_DROWSY = 0.35            # Score >= 0.35 = drowsy
        self.SCORE_SLEEPING = 0.60          # Score >= 0.60 = sleeping
        
        # Temporal Smoothing
        self.SMOOTHING_WINDOW = 10          # Frames for majority voting
        
        # Resolution Gate
        self.MIN_FACE_SIZE = 48             # Skip MediaPipe if crop smaller

        # Motion Buffer
        self.MOTION_BUFFER_SIZE = 30        # Number of frames to track head position
        
        # Per-person state tracking
        self.state = {}

    def process_crop(self, crop, id_key="unknown", keypoints=None, crop_origin=(0,0), sensitivity=0.18):
        """
        Process person crop for sleep/drowsiness detection.
        
        Args:
            sensitivity: EAR threshold adjustment. Lower = less sensitive (requires more closed eyes).
                         Default lowered to 0.18 (was 0.22) to handle glasses/squinting better.
        """
        if crop.size == 0:
            return "awake", {"score": 0.0, "reason": "empty_crop"}



        current_time = time.time()

        # Initialize state for new person
        if id_key not in self.state:
            self.state[id_key] = {
                'last_seen': current_time,
                'head_positions': deque(maxlen=self.MOTION_BUFFER_SIZE),
                'ear_history': [],           # For dynamic threshold (kept as list for numpy/slicing)
                'ear_closed_history': deque(maxlen=self.PERCLOS_WINDOW),    # Boolean: was eye closed? (for PERCLOS)
                'recent_states': deque(maxlen=self.SMOOTHING_WINDOW),         # For temporal smoothing
                'last_active_time': current_time,
            }

        state = self.state[id_key]
        state['last_seen'] = current_time

        # === COLLECT SIGNALS ===
        signals = {
            'perclos_high': False,
            'eyes_closed': False,
            'head_dropped': False,
            'sleep_posture': False,
        }
        
        details = {
            'score': 0.0,
            'signals': {},
            'source': 'none',
        }

        # --- SIGNAL 1 & 2: Eye Analysis (MediaPipe) ---
        avg_ear = None
        current_threshold = sensitivity # Use passed sensitivity as base
        is_head_still = False
        face_detected = False
        is_side_profile = True # Default to TRUE (safer) to prevent calibration on back-of-head
        
        run_mediapipe = (
            self.detector is not None and 
            crop.shape[0] >= self.MIN_FACE_SIZE and 
            crop.shape[1] >= self.MIN_FACE_SIZE
        )

            
        if run_mediapipe:
            try:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                detection_result = self.detector.detect(mp_image)

                if detection_result.face_landmarks:
                    # VALIDATION: Check against YOLO Pose if available
                    # MediaPipe can hallucinate faces on hair (Back of Head).
                    # YOLO Pose is robust: If it sees shoulders but NO Nose/Eyes, it's definitely back-of-head.
                    valid_face_context = True
                    if keypoints is not None:
                        # YOLO Keypoints: 0=Nose, 1=Left Eye, 2=Right Eye
                        # Format is now [x, y, conf]
                        
                        # Check features with strict confidence > 0.5
                        n_conf = keypoints[0][2]
                        le_conf = keypoints[1][2]
                        re_conf = keypoints[2][2]
                        
                        has_nose = n_conf > 0.5
                        has_le = le_conf > 0.5
                        has_re = re_conf > 0.5
                        


                        # If we have NO facial features from YOLO, trust YOLO over MediaPipe
                        if not (has_nose or has_le or has_re):
                            valid_face_context = False
                    else:
                        # Fallback if YOLO Pose failed to associate
                        pass

                    # --- SIDE PROFILE PROTECTION (YAW CHECK) ---
                    # Only check geometry if YOLO confirms it's a valid face
                    if valid_face_context:
                        face_detected = True
                        landmarks = detection_result.face_landmarks[0]
                        
                        # Calculate horizontal ratio of nose to eyes to detect side profile
                        nose_x = landmarks[1].x
                        
                        # Better Yaw Estimation: Nose tip (1) relative to cheekbones or eye outer corners (33, 263)
                        dist_l = abs(nose_x - landmarks[33].x)
                        dist_r = abs(landmarks[263].x - nose_x)
                        
                        # Avoid division by zero
                        if dist_l > 0 and dist_r > 0:
                            yaw_ratio = min(dist_l, dist_r) / max(dist_l, dist_r)

                            # If ratio is small (<0.25), person is looking sideways
                            is_side_profile = yaw_ratio < 0.25
                        else:
                            is_side_profile = True # Invalid geometry -> Treat as side profile
                    else:
                        is_side_profile = True # Rejected by YOLO -> Treat as side profile / invalid
                    
                    if is_side_profile:
                         # Relax threshold drastically or skip eye check
                         current_threshold = 0.10 # Must be EXTREMELY closed to count in side profile
                         details['note'] = "side_profile_detected"
                         if current_time - state.get('last_side_profile_print', 0) > 2.0:
                             state['last_side_profile_print'] = current_time

                    # Track head position for stillness
                    nose_landmark = landmarks[1]
                    nose_position = (
                        nose_landmark.x * crop.shape[1], 
                        nose_landmark.y * crop.shape[0]
                    )
                    state['head_positions'].append(nose_position)

                    # Calculate EAR
                    try:
                        left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
                        right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
                        avg_ear = (left_ear + right_ear) / 2.0
                    except Exception as e:
                        avg_ear = 0.0

                    # Dynamic threshold based on person's baseline (only if frontal)
                    if not is_side_profile:
                        # Add to history
                        state['ear_history'].append(avg_ear)
                        
                        # --- CALIBRATION LOGIC (30s) ---
                        calibration_time = 30.0
                        time_since_seen = current_time - state.get('first_seen_time', current_time)
                        
                        if not state.get('is_calibrated', False):
                            # Mark first seen time if new
                            if 'first_seen_time' not in state:
                                state['first_seen_time'] = current_time
                                details['status'] = "calibrating_start"
                            else:
                                details['status'] = f"calibrating_{(calibration_time - time_since_seen):.0f}s"
                            
                            # Keep history growing during validation (limit to ~45s buffer)
                            # Keep history growing during validation (limit to ~45s buffer)
                            if len(state['ear_history']) > 1500: 
                                state['ear_history'].pop(0)

                            # PROGRESSIVE CALIBRATION FIX:
                            # 1. Don't reset history if time gap - just keep accumulating samples
                            # 2. Check total samples count instead of just time window
                            min_samples_required = 100 # Approx 3-5 seconds of valid face data
                            
                            # Fallback timeout: If seen for > 2 mins but not calibrated (e.g. constant movement)
                            max_calibration_time = 120.0 
                            
                            is_timeout = time_since_seen > max_calibration_time
                            has_enough_data = len(state['ear_history']) > min_samples_required
                            
                            # If we have enough data OR we timed out
                            if has_enough_data or is_timeout:
                                if has_enough_data:
                                    # Normal calibration
                                    baseline_ear = np.percentile(state['ear_history'], 90)
                                    calculated = baseline_ear * 0.75
                                    final_threshold = min(0.25, max(0.10, calculated))
                                    details['status'] = "calibrated_success"
                                else:
                                    # Timeout fallback - use default sensitivity
                                    final_threshold = sensitivity
                                    details['status'] = "calibrated_timeout_fallback"
                                    
                                state['personal_threshold'] = final_threshold
                                state['is_calibrated'] = True
                                # Trim history back to rolling window
                                state['ear_history'] = state['ear_history'][-300:]
                                print(f"  [SleepDetector] Calibrated {id_key}: {final_threshold:.3f} ({details['status']})")
                            
                            # During calibration, use user setting (allows manual low override)
                            current_threshold = sensitivity 
                        else:
                            # Already calibrated - use personal threshold or passed sensitivity
                            # We use the personal threshold but upper-bound it by the user's manual sensitivity setting
                            # to respect the slider in the UI if they want it VERY strict.
                            personal = state['personal_threshold']
                            current_threshold = min(sensitivity, personal)
                            
                            # Standard rolling window maintenance
                            if len(state['ear_history']) > 300:
                                state['ear_history'].pop(0)
                        
                        # Store for UI/Debug
                        details['calibrated'] = state.get('is_calibrated', False)

                    # Is eye currently closed?
                    eye_closed = avg_ear < current_threshold
                    signals['eyes_closed'] = eye_closed
                    
                    # Track for PERCLOS
                    state['ear_closed_history'].append(eye_closed)
                    
                    # Calculate PERCLOS
                    if len(state['ear_closed_history']) >= 10:  # Need minimum samples
                        closed_count = sum(state['ear_closed_history'])
                        perclos = closed_count / len(state['ear_closed_history'])
                        signals['perclos_high'] = perclos >= self.PERCLOS_DROWSY
                        details['perclos'] = perclos
                    
                    # Head stillness (REMOVED)
                    # is_head_still = False # self._is_head_still_normalized(state['head_positions'], crop.shape)
                    # signals['head_still'] = is_head_still
                    
                    details['ear'] = avg_ear
                    details['threshold'] = current_threshold
                    details['source'] = 'mediapipe'
                    
            except Exception as e:
                # print(f"MP Error: {e}")
                pass  # Fall through to posture check

        # --- SIGNAL 3, 4, 5: Posture Analysis (YOLO Pose) ---
        kpts = None
        shoulder_width = None
        
        if keypoints is not None:
            cx, cy = crop_origin
            kpts = keypoints.copy()
            kpts[:, 0] -= cx
            kpts[:, 1] -= cy
        else:
            # Fallback - run pose on crop
            # CRITICAL: Crash Fix - Use Lock if available
            if self.lock:
                with self.lock:
                    pose_results = self.pose_model.predict(crop, verbose=False, conf=0.5, device=self.device)
            else:
                pose_results = self.pose_model.predict(crop, verbose=False, conf=0.5, device=self.device)
                
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                keypoints_data = pose_results[0].keypoints.xy.cpu().numpy()
                if len(keypoints_data) > 0:
                    kpts = keypoints_data[0]

        if kpts is not None:
            posture_signals = self._check_sleep_posture_scaled(kpts, crop.shape)
            signals['head_dropped'] = posture_signals.get('head_dropped', False)
            signals['sleep_posture'] = posture_signals.get('sleep_posture', False)
            
            # Override to awake if writing detected
            if posture_signals.get('is_writing', False):
                state['last_active_time'] = current_time
                state['recent_states'].append('awake')
                return "awake", {"score": 0.0, "reason": "writing_detected", "source": "yolo-pose"}
            
            details['posture'] = posture_signals
            if details['source'] == 'none':
                details['source'] = 'yolo-pose'

        # === COMPUTE WEIGHTED SCORE ===
        score = 0.0
        active_signals = []
        for signal_name, is_active in signals.items():
            if is_active and signal_name in self.SCORE_WEIGHTS:
                score += self.SCORE_WEIGHTS[signal_name]
                active_signals.append(signal_name)
        
        details['score'] = score
        details['signals'] = signals
        details['active_signals'] = active_signals

        # === DETERMINE RAW STATE ===
        if score >= self.SCORE_SLEEPING:
            raw_state = 'sleeping'
        elif score >= self.SCORE_DROWSY:
            raw_state = 'drowsy'
        else:
            raw_state = 'awake'
            state['last_active_time'] = current_time

        # === TEMPORAL SMOOTHING ===
        state['recent_states'].append(raw_state)
        
        # Majority voting for smoothed state
        if len(state['recent_states']) >= 5:
            from collections import Counter
            state_counts = Counter(state['recent_states'])
            smoothed_state = state_counts.most_common(1)[0][0]
        else:
            smoothed_state = raw_state
        
        details['raw_state'] = raw_state
        details['smoothed_state'] = smoothed_state

        return smoothed_state, details

    def _is_head_still(self, positions):
        """Check if head movement is minimal (indicator of sleep) - legacy method."""
        if len(positions) < 10: 
            return False
        positions_array = np.array(positions)
        x_std = np.std(positions_array[:, 0])
        y_std = np.std(positions_array[:, 1])
        return (x_std + y_std) < 15.0  # Pixel-based threshold
    
    # _is_head_still_normalized REMOVED

    
    def _check_sleep_posture_scaled(self, kpts, crop_shape):
        """
        Scale-invariant posture check using ratios relative to shoulder_width.
        
        Returns dict with boolean flags for each posture signal:
        - head_dropped: nose significantly below shoulders
        - head_tilted: eyes at different heights
        - sleep_posture: shoulders visible but face not
        - is_writing: hands positioned for writing/reading
        """
        def has_pt(idx): 
            return kpts[idx][0] > 0 and kpts[idx][1] > 0
        
        result = {
            'head_dropped': False, 
            'head_tilted': False, 
            'sleep_posture': False,
            'is_writing': False,
            'details': {}
        }
        
        # Calculate reference measurements
        has_shoulders = has_pt(5) and has_pt(6)
        if not has_shoulders:
            return result  # Can't do scale-invariant checks without shoulders
            
        shoulder_width = abs(kpts[6][0] - kpts[5][0])
        if shoulder_width < 10:  # Too small to be reliable
            return result
            
        shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
        result['details']['shoulder_width'] = shoulder_width
        
        # Check if face is visible
        has_face = has_pt(0) or has_pt(1) or has_pt(2) or has_pt(3) or has_pt(4)
        
        # === SLEEP POSTURE: Shoulders visible but no face ===
        if has_shoulders and not has_face:
            # Additional check: shoulders should be near top of crop (person leaning forward)
            if shoulder_y < crop_shape[0] * 0.35:
                result['sleep_posture'] = True
                result['details']['reason'] = 'head_buried'
        
        # === HEAD DROPPED: Nose significantly below shoulder line ===
        if has_pt(0) and has_shoulders:
            nose_y = kpts[0][1]
            drop_distance = nose_y - shoulder_y
            normalized_drop = drop_distance / shoulder_width
            
            if normalized_drop > self.HEAD_DROP_RATIO:
                result['head_dropped'] = True
                result['details']['drop_ratio'] = normalized_drop
        
        # === HEAD TILTED (REMOVED) ===
        
        # === WRITING DETECTION: Hands below shoulders, head slightly forward ===
        has_wrists = has_pt(9) or has_pt(10)
        if has_shoulders and has_wrists:
            wrist_y = 0
            if has_pt(9) and has_pt(10): 
                wrist_y = max(kpts[9][1], kpts[10][1])
            elif has_pt(9): 
                wrist_y = kpts[9][1]
            else: 
                wrist_y = kpts[10][1]
            
            wrist_drop = wrist_y - shoulder_y
            normalized_wrist = wrist_drop / shoulder_width
            
            # Hands significantly below shoulders = working
            if normalized_wrist > 0.5:
                # Additional check: if nose is visible and only slightly forward, it's writing
                if has_pt(0):
                    nose_y = kpts[0][1]
                    nose_drop = (nose_y - shoulder_y) / shoulder_width
                    if 0.1 < nose_drop < 0.4:  # Slight forward lean, not collapsed
                        result['is_writing'] = True
                        result['details']['activity'] = 'writing_or_reading'
        
        return result

    def _check_sleep_posture(self, kpts, crop_shape):
        """
        Check for sleep-indicative postures using pose keypoints.
        
        Patterns detected:
        - Head buried (shoulders visible, face not)
        - Slumped forward
        - Head tilted significantly
        - Collapsed posture
        
        Also detects active postures (writing/reading).
        """
        def has_pt(idx): 
            return kpts[idx][0] > 0 and kpts[idx][1] > 0
        
        result = {'is_sleeping': False, 'is_writing': False, 'reason': None, 'details': {}}
        crop_height = crop_shape[0]
        
        has_shoulders = has_pt(5) and has_pt(6)
        has_face = has_pt(0) or has_pt(1) or has_pt(2) or has_pt(3) or has_pt(4)

        if has_shoulders and not has_face:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if shoulder_y < crop_height * 0.25:
                result['is_sleeping'] = True
                result['reason'] = "head_buried_high_shoulders"
                result['details']['shoulder_height_ratio'] = shoulder_y / crop_height
                return result
            else:
                return result
        
        if has_pt(0) and has_shoulders:
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if nose_y > shoulder_y + 30:
                result['is_sleeping'] = True
                result['reason'] = "slumped_forward"
                result['details']['slump_distance'] = nose_y - shoulder_y
                return result
        
        has_wrists = has_pt(9) or has_pt(10)
        if has_shoulders and has_wrists:
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            wrist_y = 0
            if has_pt(9) and has_pt(10): 
                wrist_y = max(kpts[9][1], kpts[10][1])
            elif has_pt(9): 
                wrist_y = kpts[9][1]
            else: 
                wrist_y = kpts[10][1]
            
            if wrist_y > shoulder_y + 80:
                result['is_writing'] = True
                result['reason'] = "hands_on_desk"
                result['details']['hand_position'] = "active"
                if has_pt(0):
                    nose_y = kpts[0][1]
                    if nose_y > shoulder_y + 10 and nose_y < shoulder_y + 50: 
                        return result
        
        if has_pt(1) and has_pt(2):
            tilt = abs(kpts[1][1] - kpts[2][1])
            if tilt > 40:
                result['is_sleeping'] = True
                result['reason'] = "head_tilted"
                result['details']['tilt_amount'] = tilt
                return result
        
        if has_shoulders and (has_pt(11) or has_pt(12)):
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            hip_y = 0
            if has_pt(11) and has_pt(12): 
                hip_y = (kpts[11][1] + kpts[12][1]) / 2
            elif has_pt(11): 
                hip_y = kpts[11][1]
            else: 
                hip_y = kpts[12][1]
            torso_length = abs(hip_y - shoulder_y)
            if torso_length < 60:
                result['is_sleeping'] = True
                result['reason'] = "collapsed_posture"
                result['details']['torso_length'] = torso_length
                return result
        
        if has_pt(0) and has_shoulders and (has_pt(7) or has_pt(8)):
            nose_y = kpts[0][1]
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            if nose_y > shoulder_y + 20 and nose_y < shoulder_y + 60:
                elbow_y = 0
                if has_pt(7) and has_pt(8): 
                    elbow_y = (kpts[7][1] + kpts[8][1]) / 2
                elif has_pt(7): 
                    elbow_y = kpts[7][1]
                else: 
                    elbow_y = kpts[8][1]
                if elbow_y > shoulder_y and elbow_y < shoulder_y + 100:
                    result['is_writing'] = True
                    result['reason'] = "reading_posture"
                    return result
        
        return result

    def _calculate_ear(self, landmarks, indices):
        """
        Calculate Eye Aspect Ratio (EAR) for drowsiness detection.
        Lower values indicate closed eyes.
        """
        def dist(i1, i2):
            p1 = landmarks[i1]
            p2 = landmarks[i2]
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        vertical_1 = dist(indices[1], indices[5])
        vertical_2 = dist(indices[2], indices[4])
        horizontal = dist(indices[0], indices[3])
        if horizontal == 0: 
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def close(self):
        """Clean up MediaPipe resources."""
        if self.detector: 
            self.detector.close()
