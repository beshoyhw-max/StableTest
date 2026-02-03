import cv2
import time
import os
import math
import datetime
from ultralytics import YOLO
import threading
from sleep_detector import SleepDetector
import torch
import numpy as np
class PhoneDetector:
    def __init__(self, model_path='yolo26s.engine', pose_model_path='yolo26s-pose.engine', 
                 output_dir="detections", 
                 phone_duration_threshold=5.0,
                 sleep_duration_threshold=10.0,
                 cooldown_seconds=120.0,
                 reset_gap=2.5,
                 model_instance=None, pose_model_instance=None, lock=None,
                 enable_face_recognition=True,
		device='cuda',
                 attendance_tracker=None,
                 shared_cooldowns=None, shared_cooldowns_lock=None):
        """
        Initialize Phone Detector with HYBRID cooldown system.
        - Recognized persons: Use shared_cooldowns (global across all cameras)
        - Unknown persons: Use local self.cooldowns (per-camera)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"my device {self.device}")
        self.last_threshold_check = 0
        self.threshold_check_interval = 5.0  # Check for threshold updates every 5s

        self.face_rec_debug_counter = 0
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load models
        # OPTIMIZATION: Split lock into detection and pose locks
        # If model_instance is None (Private Model), we DO NOT LOCK detection -> Parallel Inference!
        self.pose_lock = lock
        
        if model_instance:
            print("  → Using shared detection model (with lock)")
            self.model = model_instance
            self.detection_lock = lock
        else:
            print(f"  → Loading private detection model from {model_path}")
            self.model = YOLO(model_path, task='detect')
            # TensorRT models don't support .to(device)
            if model_path.endswith('.pt'):
                self.model.to(self.device)
            print("  → Private detection model loaded - PARALLEL INFERENCE ENABLED")
            self.detection_lock = None # Explicitly unlock private model usage
            
        print("  → Initializing sleep detector...")
        if pose_model_instance:
            self.sleep_detector = SleepDetector(pose_model_instance=pose_model_instance, lock=lock)
        else:
            self.sleep_detector = SleepDetector(pose_model_path=pose_model_path, lock=lock)
        print("  → Sleep detector initialized")

        # Initialize face recognition if enabled
        self.face_recognizer = None
        self.enable_face_recognition = enable_face_recognition
        
        if enable_face_recognition:
            try:
                from face_recognizer import FaceRecognizer
                from async_face_worker import AsyncFaceWorker
                
                self.face_recognizer = FaceRecognizer()
                self.async_face_worker = AsyncFaceWorker(self.face_recognizer)
                self.async_face_worker.start()
                
                num_people = len(self.face_recognizer.list_known_people())
                print(f"  → Face recognition ENABLED (Async Mode)")
                print(f"     • Registered people: {num_people}")
                if num_people > 0:
                    print(f"     • Names: {', '.join(self.face_recognizer.list_known_people())}")
            except Exception as e:
                print(f"  → Face recognition initialization failed: {e}")
                print(f"  → Continuing without face recognition")
                self.face_recognizer = None
                self.async_face_worker = None

        self.PHONE_CLASS_ID = 67
        self.PERSON_CLASS_ID = 0
        
        # TIME-BASED THRESHOLDS
        self.phone_duration_threshold = phone_duration_threshold
        self.sleep_duration_threshold = sleep_duration_threshold
        self.cooldown_seconds = cooldown_seconds
        self.reset_gap = reset_gap
        
        # Tracking state
        self.cooldown_seconds = cooldown_seconds
        self.reset_gap = reset_gap
        
        # Tracking state
        self.violation_timers = {}
        self.cooldowns = {}  # Local cooldowns for unknown persons (per-camera)
        self.last_display_data = []
        
        # HYBRID COOLDOWN: Shared cooldowns for recognized persons (global)
        self.shared_cooldowns = shared_cooldowns  # Shared dict from CameraManager
        self.shared_cooldowns_lock = shared_cooldowns_lock  # Thread lock
        
        # Optical Flow Tracking State
        self.prev_gray = None
        self.tracker_initialized = False
        
        # FIXED: Person identification with better caching
        self.person_identities = {}  # track_id -> name
        self.identity_confidences = {}  # track_id -> confidence
        self.identity_last_check = {}  # track_id -> timestamp
        self.identity_check_interval = 1.0  # Check every 1 second (was 2.0)
        self.identity_min_confidence = 0.40  # Minimum confidence to accept (lowered from implicit 0.363)
        
        print(f"  → Time-based thresholds configured:")
        print(f"     • Phone alert after: {phone_duration_threshold}s")
        print(f"     • Sleep alert after: {sleep_duration_threshold}s")
        print(f"     • Reset gap: {reset_gap}s")
        print(f"     • Face recognition check interval: {self.identity_check_interval}s")
        
        # New: Decay logic for identities
        self.identity_missed_counts = {}  # track_id -> consecutive misses
        self.identity_miss_threshold = 3  # Drop identity after 3 failed verifications
        
        # Attendance tracker for monitoring who left
        self.attendance_tracker = attendance_tracker
        if attendance_tracker:
            print(f"     • Attendance tracking: ENABLED")

    def process_frame(self, frame, frame_count, skip_frames=5, save_screenshots=True, 
                     conf_threshold=0.25, camera_name="Unknown", enable_sleep_detection=True,
                     sleep_sensitivity=0.18):
        """
        Process frame with FIXED face recognition display.
        """
        current_time = time.time()
        
        # Periodic cleanup
        if frame_count % 1000 == 0:
            cutoff = current_time - (self.cooldown_seconds * 2)
            self.cooldowns = {k: v for k, v in self.cooldowns.items() if v > cutoff}
            
            stale_cutoff = current_time - 10.0
            self.violation_timers = {
                k: v for k, v in self.violation_timers.items() 
                if v['last_seen'] > stale_cutoff
            }
            
            # Clean stale identities (persons not seen in 15 seconds)
            identity_cutoff = current_time - 15.0
            active_track_ids = set(t[0] for t in self.violation_timers.keys())
            self.person_identities = {
                k: v for k, v in self.person_identities.items()
                if (k in active_track_ids or 
                    self.identity_last_check.get(k, 0) > identity_cutoff)
            }
            self.identity_confidences = {
                k: v for k, v in self.identity_confidences.items()
                if k in self.person_identities
            }
            # Clean missed counts
            self.identity_missed_counts = {
                k: v for k, v in self.identity_missed_counts.items()
                if k in self.person_identities or k in active_track_ids
            }

        global_status = "safe"
        screenshot_saved_global = False

        # --- INFERENCE STEP ---
        # Convert to grayscale for Optical Flow (always needed now)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count % skip_frames == 0:
            # Measure inference time
            start_infer = time.time()
            
            # Build new detections in temp list (don't clear existing until we have new ones)
            new_display_data = []
            
            # 1. Detection + Tracking
            classes_to_track = [self.PERSON_CLASS_ID, self.PHONE_CLASS_ID]
            
            if self.detection_lock:
                with self.detection_lock:
                    results = self.model.track(frame, classes=classes_to_track, 
                                             conf=conf_threshold, persist=True, 
                                             verbose=False, imgsz=1280, device=self.device,
                                             tracker='custom_tracker.yaml')
            else:
                results = self.model.track(frame, classes=classes_to_track, 
                                         conf=conf_threshold, persist=True, 
                                         verbose=False, imgsz=1280, device=self.device,
                                         tracker='custom_tracker.yaml')
            
            person_boxes = []
            phone_boxes = []

            if len(results) > 0 and results[0].boxes:
                # DEBUG: Check if it's YOLO detecting it
                # Logic used for debug prints removed for performance

                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    coords = box.xyxy[0].cpu().numpy()

                    if cls_id == self.PERSON_CLASS_ID:
                        if box.id is not None:
                            track_id = int(box.id.item())
                            person_boxes.append((*coords, track_id))
                    elif cls_id == self.PHONE_CLASS_ID:
                        # Aspect Ratio Filter
                        w = coords[2] - coords[0]
                        h = coords[3] - coords[1]
                        aspect_ratio = w / h if h > 0 else 0
                        # print(f"DEBUG: AR: {aspect_ratio:.2f} | Conf: {float(box.conf[0].item()):.2f}")
                        
                        # Data-driven update:
                        # Pen AR seen: 0.14 - 0.33 -> Cutoff: 0.45
                        if aspect_ratio < 0.45 or aspect_ratio > 3.0:
                            continue
                            
                        # Removed Charger Filter: AR 0.8-1.5 overlaps with real phones
                        # To filter chargers, rely on Confidence Threshold in App UI

                        conf = float(box.conf[0].item())
                        phone_boxes.append((*coords, conf))

            # 2. Pose Estimation
            pose_keypoints_map = {}
            if hasattr(self.sleep_detector, 'pose_model'):
                try:
                    if self.pose_lock:
                        with self.pose_lock:
                            pose_results = self.sleep_detector.pose_model(frame, verbose=False, conf=0.5)
                    else:
                        pose_results = self.sleep_detector.pose_model(frame, verbose=False, conf=0.5)
                    
                    if len(pose_results) > 0 and pose_results[0].boxes:
                        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                        # UPDATED: Use .data to get (x, y, conf) instead of just (x, y)
                        pose_kpts = pose_results[0].keypoints.data.cpu().numpy()
                        pose_keypoints_map = self._associate_pose_to_persons(
                            person_boxes, pose_boxes, pose_kpts
                        )
                except Exception as e:
                    print(f"Pose Inference Error: {e}")

            # 3. Associate phones to persons
            phone_map = self._associate_phones_to_persons(
                person_boxes, phone_boxes, pose_keypoints_map
            )

            # 4. Face Recognition - Async Version
            if self.face_recognizer:
                # Periodically refresh threshold from shared config
                if current_time - self.last_threshold_check > self.threshold_check_interval:
                    self.face_recognizer.threshold = self.face_recognizer.get_shared_threshold()
                    self.last_threshold_check = current_time

                # A. PROCESS RESULTS from Background Worker
                for res in self.async_face_worker.get_results():
                    track_id = res['track_id']
                    person_name = res['name']
                    confidence = res['conf']
                    
                    if person_name != "Unknown" and confidence >= self.identity_min_confidence:
                        old_name = self.person_identities.get(track_id, "Unknown")
                        self.person_identities[track_id] = person_name
                        self.identity_confidences[track_id] = confidence
                        
                        # Reset missed count on success
                        self.identity_missed_counts[track_id] = 0
                        
                        if old_name != person_name:
                            print(f"  ✓ ID {track_id}: {person_name} ({confidence:.3f})")
                        
                        # Update attendance tracker with recognized person
                        # NOTE: This is safe to call here as it just updates a dict (fast)
                        if self.attendance_tracker:
                            # We need to find the current box for this track_id to invoke update_person
                            # This is a bit tricky since the result is async.
                            # We will use the 'last known box' from the current frame if available
                            current_box = None
                            for pb in person_boxes:
                                if int(pb[4]) == track_id:
                                    current_box = pb[:4]
                                    break
                            
                            if current_box is not None:
                                self.attendance_tracker.update_person(
                                    person_name, current_box, frame, camera_name
                                )
                    else:
                        # Not recognized (Unknown or low confidence)
                        # Increment missed count if we thought we knew who this was
                        if track_id in self.person_identities:
                            misses = self.identity_missed_counts.get(track_id, 0) + 1
                            self.identity_missed_counts[track_id] = misses
                            
                            if misses >= self.identity_miss_threshold:
                                removed_name = self.person_identities.pop(track_id, "Unknown")
                                self.identity_confidences.pop(track_id, None)
                                self.identity_missed_counts[track_id] = 0
                                print(f"  ✕ ID {track_id}: Identity '{removed_name}' dropped (failed {misses} verifications)")


                # B. QUEUE NEW REQUESTS
                for p_box in person_boxes:
                    x1, y1, x2, y2, track_id = p_box
                    
                    last_check = self.identity_last_check.get(track_id, 0)
                    should_check = (track_id not in self.person_identities or 
                                   current_time - last_check > self.identity_check_interval)
                    
                    if should_check:
                         self.async_face_worker.enqueue_request(frame, (x1, y1, x2, y2), track_id, current_time)
                         self.identity_last_check[track_id] = current_time

            # 5. Process each person with TIME-BASED logic
            current_detections = set()
            
            for p_box in person_boxes:
                x1, y1, x2, y2, track_id = map(int, p_box)

                # Determine current frame status
                has_phone = phone_map.get(track_id, False)
                is_sleeping = False
                
                if not has_phone and enable_sleep_detection:
                    # Check for sleep
                    h, w, _ = frame.shape
                    pad = 20
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    person_crop = frame[cy1:cy2, cx1:cx2]
                    
                    if person_crop.size > 0:
                        # FIXED: Use Person Name if available (Stable), otherwise used ID (Unstable)
                        person_name = self.person_identities.get(track_id, None)
                        if person_name:
                            sleep_key = f"{camera_name}_{person_name}"
                            # if frame_count % 30 == 0: print(f"DEBUG: Using STABLE key: {sleep_key}")
                        else:
                            sleep_key = f"{camera_name}_id_{track_id}"
                            # if frame_count % 30 == 0: print(f"DEBUG: Using ID key: {sleep_key}")
                        kpts = pose_keypoints_map.get(track_id)
                        
                        sleep_status, sleep_details = self.sleep_detector.process_crop(
                            person_crop,
                            id_key=sleep_key,
                            keypoints=kpts,
                            crop_origin=(cx1, cy1),
                            sensitivity=sleep_sensitivity
                        )
                        
                        if sleep_status in ("sleeping", "drowsy"):
                            is_sleeping = True
                
                # DEBUG: Check if phone is blocking sleep
                # elif has_phone:
                #     if frame_count % 30 == 0:
                #         print(f"DEBUG: ID {track_id} has phone -> Sleep Check SKIPPED")


                # --- TIME-BASED VIOLATION TRACKING ---
                if has_phone:
                    key = (track_id, "texting")
                    current_detections.add(key)
                    
                    if key not in self.violation_timers:
                        self.violation_timers[key] = {
                            'start_time': current_time,
                            'last_seen': current_time
                        }
                    else:
                        self.violation_timers[key]['last_seen'] = current_time
                
                if is_sleeping:
                    key = (track_id, "sleeping")
                    current_detections.add(key)
                    
                    if key not in self.violation_timers:
                        self.violation_timers[key] = {
                            'start_time': current_time,
                            'last_seen': current_time
                        }
                    else:
                        self.violation_timers[key]['last_seen'] = current_time

            # --- RESET TIMERS for violations not currently detected ---
            for key, timer_data in list(self.violation_timers.items()):
                if key not in current_detections:
                    gap = current_time - timer_data['last_seen']
                    if gap > self.reset_gap:
                        del self.violation_timers[key]

            # --- DETERMINE STATUS AND SAVE EVIDENCE ---
            for p_box in person_boxes:
                x1, y1, x2, y2, track_id = map(int, p_box)
                
                status = "safe"
                color = (0, 255, 0)
                
                # FIXED: Always show person name if available
                person_name = self.person_identities.get(track_id, None)
                person_conf = self.identity_confidences.get(track_id, 0.0)
                
                if person_name:
                    # Show name with confidence
                    label = f"{person_name} ({person_conf:.2f})"
                else:
                    label = f"ID: {track_id}"
                
                # Check texting timer
                texting_key = (track_id, "texting")
                sleeping_key = (track_id, "sleeping")
                
                texting_duration = 0.0
                sleeping_duration = 0.0
                
                if texting_key in self.violation_timers:
                    texting_duration = current_time - self.violation_timers[texting_key]['start_time']
                    
                    if texting_duration >= self.phone_duration_threshold:
                        status = "texting"
                        color = (0, 0, 255)
                        global_status = "texting"
                        label += f" | PHONE {texting_duration:.1f}s"
                    else:
                        color = (0, 165, 255)
                        label += f" | Phone {texting_duration:.1f}s/{self.phone_duration_threshold:.0f}s"
                
                elif sleeping_key in self.violation_timers:
                    sleeping_duration = current_time - self.violation_timers[sleeping_key]['start_time']
                    
                    if sleeping_duration >= self.sleep_duration_threshold:
                        status = "sleeping"
                        color = (255, 0, 0)
                        if global_status != "texting":
                            global_status = "sleeping"
                        label += f" | SLEEP {sleeping_duration:.1f}s"
                    else:
                        color = (0, 255, 255)
                        label += f" | Sleep {sleeping_duration:.1f}s/{self.sleep_duration_threshold:.0f}s"

                # --- SAVE EVIDENCE with HYBRID COOLDOWN ---
                # Recognized persons: Global cooldown (1 screenshot across ALL cameras)
                # Unknown persons: Per-camera cooldown (can capture from multiple cameras)
                if save_screenshots and status != "safe":
                    should_save = False
                    key = None
                    
                    if person_name:
                        # RECOGNIZED PERSON: Use global shared cooldowns
                        key = (person_name, status)
                        if self.shared_cooldowns is not None and self.shared_cooldowns_lock is not None:
                            with self.shared_cooldowns_lock:
                                last_save_time = self.shared_cooldowns.get(key, 0)
                                if (current_time - last_save_time) > self.cooldown_seconds:
                                    should_save = True
                                    self.shared_cooldowns[key] = current_time
                        else:
                            # Fallback to local if shared not available
                            last_save_time = self.cooldowns.get(key, 0)
                            if (current_time - last_save_time) > self.cooldown_seconds:
                                should_save = True
                                self.cooldowns[key] = current_time
                    else:
                        # UNKNOWN PERSON: Use per-camera local cooldowns
                        key = (f"id_{track_id}", status)
                        last_save_time = self.cooldowns.get(key, 0)
                        if (current_time - last_save_time) > self.cooldown_seconds:
                            should_save = True
                            self.cooldowns[key] = current_time
                    
                    if should_save:
                        type_str = "PHONE" if status == "texting" else "SLEEP"
                        duration_str = f"{texting_duration if status == 'texting' else sleeping_duration:.1f}s"
                        
                        # Use raw frame copy to only draw the violator's box
                        raw_evidence_frame = frame.copy()
                        self.save_evidence(
                            raw_evidence_frame, x1, y1, x2, y2, camera_name, 
                            type_str, track_id=track_id, duration=duration_str,
                            person_name=person_name
                        )
                        screenshot_saved_global = True
                        label += " [SAVED]"

                new_display_data.append((x1, y1, x2, y2, color, status, label))

            # Update global display data
            # NOTE: We update self.last_display_data only on inference to keep it authoritative
            # Optical flow will update it efficiently on skipped frames
            self.last_display_data = new_display_data
            
            # Update previous gray frame for next optical flow step
            self.prev_gray = gray_frame
            
            # Check for absences call REMOVED - Moved to CameraManager global loop
            # if self.attendance_tracker:
            #     self.attendance_tracker.check_absences(frame, camera_name)
            
            # Print FPS
            # fps = 1.0 / (time.time() - start_infer)
            # print(f"DEBUG: Processing FPS: {fps:.1f}")
            
        else:
            # --- SKIPPED FRAME: OPTICAL FLOW INTERPOLATION ---
            # If we have previous boxes and a previous frame, we can track movement
            if self.last_display_data and self.prev_gray is not None:
                try:
                    # Calculate Optical Flow for center points of boxes
                    # This is much faster than full inference
                    
                    # 1. Extract center points from last known boxes
                    points = []
                    boxes_idx = []
                    
                    for i, data in enumerate(self.last_display_data):
                        x1, y1, x2, y2, _, _, _ = data
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        points.append([[cx, cy]])
                        boxes_idx.append(i)
                        
                    if points:
                        p0 = np.array(points, dtype=np.float32)
                        
                        # 2. Calculate Optical Flow
                        p1, st, err = cv2.calcOpticalFlowPyrLK(
                            self.prev_gray, gray_frame, p0, None,
                            winSize=(15, 15), maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                        )
                        
                        # 3. Update Box Positions
                        updated_data = []
                        valid_points = st.flatten() == 1
                        
                        for i, (new_p, valid) in enumerate(zip(p1, valid_points)):
                            if valid:
                                idx = boxes_idx[i]
                                old_data = self.last_display_data[idx]
                                x1, y1, x2, y2, color, status, label = old_data
                                
                                # Calculate displacement (dx, dy)
                                dx = new_p[0][0] - p0[i][0][0]
                                dy = new_p[0][1] - p0[i][0][1]
                                
                                # Move box
                                nx1 = int(x1 + dx)
                                ny1 = int(y1 + dy)
                                nx2 = int(x2 + dx)
                                ny2 = int(y2 + dy)
                                
                                # Ensure within bounds
                                h, w = frame.shape[:2]
                                nx1 = max(0, min(w, nx1))
                                ny1 = max(0, min(h, ny1))
                                nx2 = max(0, min(w, nx2))
                                ny2 = max(0, min(h, ny2))
                                
                                updated_data.append((nx1, ny1, nx2, ny2, color, status, label))
                            else:
                                # Tracking failed for this point, keep old box
                                updated_data.append(self.last_display_data[boxes_idx[i]])
                        
                        self.last_display_data = updated_data
                        
                except Exception as e:
                    # print(f"Optical Flow Error: {e}")
                    pass
            
            # Update previous gray frame
            self.prev_gray = gray_frame

        # --- DRAWING ---
        # REMOVED: Drawing directly on the shared frame buffer causes race conditions (tearing)
        # The Native Viewer and Streamlit App now use draw_detections_on_frame() to draw on copies.
        
        return frame, global_status, screenshot_saved_global

    def get_detection_data(self):
        """
        Get cached detection data for drawing on any frame.
        Returns list of tuples: (x1, y1, x2, y2, color, status, label)
        """
        return self.last_display_data.copy()
    
    def draw_detections_on_frame(self, frame):
        """
        Draw cached detection boxes on any frame.
        Allows display of raw frames at 60fps with detection overlay.
        Returns the frame with boxes drawn.
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Calculate dynamic scale based on resolution (Benchmark: 1280x720)
        # If frame is larger, everything grows. If smaller, it shrinks (but clamped to minimums).
        scale_factor = max(w / 1280.0, h / 720.0)
        
        # Base parameters for 720p
        base_box_thickness = 3
        base_font_scale = 0.8  # Increased base size slightly
        base_font_thickness = 2
        
        # Apply scaling
        box_thickness = max(2, int(base_box_thickness * scale_factor))
        font_scale = max(0.6, base_font_scale * scale_factor)
        font_thick = max(1, int(base_font_thickness * scale_factor))
        
        for (x1, y1, x2, y2, color, status, label) in self.last_display_data:
            # Draw Box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)
            
            # Label Calculation
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            label_w, label_h = label_size
            
            # Dynamic Padding
            pad_x = int(6 * scale_factor)
            pad_y = int(6 * scale_factor)
            
            # Check if label fits on top, else put inside or below
            label_y_top = y1 - pad_y
            if label_y_top - label_h < 0:
                # If off-screen at top, move inside the box
                label_y_top = y1 + label_h + pad_y + box_thickness
            
            # Draw Background
            # (x1, y1 - label_h - pad_y) to (x1 + label_w + pad_x, y1)
            # Adjusted for the calculated position
            bg_p1 = (x1, label_y_top - label_h - pad_y)
            bg_p2 = (x1 + label_w + pad_x * 2, label_y_top + pad_y)
            
            cv2.rectangle(output, bg_p1, bg_p2, color, -1)
            
            # Draw Text
            cv2.putText(output, label, (x1 + pad_x, label_y_top), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
        
        return output

        return mapping

    def _associate_phones_to_persons(self, person_boxes, phone_boxes, pose_keypoints_map):
        """
        Scale-invariant geometric association with Face Overlap Filtering.
        Filters out 'phones' that are actually hands covering the face.
        """
        mapping = {}
        if not phone_boxes: 
            return mapping

        for ph in phone_boxes:
            ph_x1, ph_y1, ph_x2, ph_y2, _ = ph
            ph_cx = (ph_x1 + ph_x2) / 2
            ph_cy = (ph_y1 + ph_y2) / 2

            best_person_id = None
            min_normalized_dist = float('inf')

            for p in person_boxes:
                p_x1, p_y1, p_x2, p_y2, p_id = p
                
                person_width = p_x2 - p_x1
                person_height = p_y2 - p_y1

                pad_x = person_width * 0.4
                pad_y = person_height * 0.3
                
                if not (p_x1 - pad_x <= ph_cx <= p_x2 + pad_x and 
                       p_y1 - pad_y <= ph_cy <= p_y2 + pad_y):
                    continue

                kpts = pose_keypoints_map.get(p_id)
                raw_dist = float('inf')

                # --- 1. GEOMETRIC FILTER: IGNORE HAND-ON-FACE ---
                # Check if "phone" is centrally located on the face (likely a hand)
                if kpts is not None:
                    nose = kpts[0]      # [x, y, conf]
                    shoulder_l = kpts[5]
                    shoulder_r = kpts[6]
                    
                    # Ensure Nose is detected
                    if nose[0] > 0:
                        nose_x, nose_y = nose[:2]
                        
                        # Calculate Face Width Heuristic (distance between ears or scaled from person width)
                        # Fallback: 20% of person width
                        face_width_est = person_width * 0.20
                        
                        # Check Horizontal Alignment: Is phone "Central"?
                        # Real phone calls are at the EAR (offset from nose).
                        # Hands on face are usually overlapping the Nose/Mouth (central).
                        dist_x_nose = abs(ph_cx - nose_x)
                        
                        is_central_x = dist_x_nose < (face_width_est * 0.8) # Stricter margin means we only filter VERY central things
                        
                        # Check Vertical Alignment: Is phone "High"?
                        # Phone Texting is usually below shoulders or at chest.
                        # Hand on face is at nose level or slightly below.
                        
                        # Get Shoulder Y (higher value is lower on screen)
                        shoulder_y = float('inf')
                        if shoulder_l[0] > 0 and shoulder_r[0] > 0:
                            shoulder_y = (shoulder_l[1] + shoulder_r[1]) / 2
                        elif shoulder_l[0] > 0:
                            shoulder_y = shoulder_l[1]
                        elif shoulder_r[0] > 0:
                            shoulder_y = shoulder_r[1]
                        else:
                            # Fallback if no shoulders: use chest estimate
                            shoulder_y = p_y1 + person_height * 0.25 # Approximation
                            
                        # If phone is ABOVE the shoulders (or reasonably high)
                        is_high_y = ph_cy < shoulder_y
                        
                        if is_central_x and is_high_y:
                            # It's HIGH and CENTRAL -> Likely Hand on Face / Facepalm
                            # print(f"DEBUG: Filtered Hand-on-Face ID {p_id}. DistX: {dist_x_nose:.1f}")
                            continue
                            
                # --- 2. REGULAR ASSOCIATION ---
                if kpts is not None:
                    wrists = []
                    if kpts[9][0] > 0: 
                        wrists.append(kpts[9])
                    if kpts[10][0] > 0: 
                        wrists.append(kpts[10])

                    if wrists:
                        for w_pt in wrists:
                            d = math.hypot(ph_cx - w_pt[0], ph_cy - w_pt[1])
                            if d < raw_dist: 
                                raw_dist = d

                        eye_y = 0
                        if kpts[1][1] > 0: 
                            eye_y = kpts[1][1]
                        elif kpts[2][1] > 0: 
                            eye_y = kpts[2][1]

                        if eye_y > 0 and ph_cy < eye_y - (person_height * 0.1):
                            raw_dist += person_height * 0.5

                if raw_dist == float('inf'):
                    p_cx = (p_x1 + p_x2) / 2
                    p_chest_y = p_y1 + person_height * 0.4
                    raw_dist = math.hypot(ph_cx - p_cx, ph_cy - p_chest_y)

                normalized_dist = raw_dist / person_height

                if normalized_dist < min_normalized_dist:
                    min_normalized_dist = normalized_dist
                    best_person_id = p_id

            MAX_NORMALIZED_DISTANCE = 0.6
            
            if best_person_id is not None and min_normalized_dist < MAX_NORMALIZED_DISTANCE:
                mapping[best_person_id] = True

        return mapping

    def _associate_pose_to_persons(self, person_boxes, pose_boxes, pose_kpts):
        """Associate pose detections with tracked persons."""
        mapping = {}
        if len(person_boxes) == 0 or len(pose_boxes) == 0:
            return mapping
        
        for p_box in person_boxes:
            px1, py1, px2, py2, track_id = p_box
            p_area = (px2 - px1) * (py2 - py1)
            best_iou = 0
            best_idx = -1

            for i, (pox1, poy1, pox2, poy2) in enumerate(pose_boxes):
                ix1 = max(px1, pox1)
                iy1 = max(py1, poy1)
                ix2 = min(px2, pox2)
                iy2 = min(py2, poy2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    po_area = (pox2 - pox1) * (poy2 - poy1)
                    union_area = p_area + po_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

            if best_idx != -1 and best_iou > 0.3:
                mapping[track_id] = pose_kpts[best_idx]
                
        return mapping

    def save_evidence(self, frame, x1, y1, x2, y2, camera_name="Unknown", 
                     detection_type="PHONE", track_id=None, duration="N/A", person_name=None):
        """
        Save evidence screenshot with duration and person name metadata.
        Organizes files into folders: detections/<type>/<person_name>/
        Only draws the box for the specific violator (not all detected persons).
        """
        # Use clean copy of raw frame - only draw THIS person's box
        evidence_img = frame.copy()
        box_color = (0, 0, 255) if detection_type == "PHONE" else (255, 0, 0)
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), box_color, 3)
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(evidence_img, (0, 0), (evidence_img.shape[1], 40), (0,0,0), -1)
        
        # Build header with person name if available
        if person_name:
            header_text = f"{detection_type} {duration} | {person_name} | {camera_name} | {ts}"
        else:
            header_text = f"{detection_type} {duration} | {camera_name} | {ts}"
            if track_id is not None: 
                header_text += f" | ID: {track_id}"

        cv2.putText(evidence_img, header_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Filename format: type_camera_MM-DD-HH-MM-SS_personname.jpg
        timestamp_fn = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        safe_cam_name = "".join([c for c in camera_name 
                                if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        
        # --- FOLDER STRUCTURE: detections/<type>/<person_name>/ ---
        type_folder = detection_type.lower()  # "phone" or "sleep"
        
        if person_name:
            # Create safe person folder name
            safe_person_name = "".join([c for c in person_name 
                                       if c.isalnum() or c in ('_', '-', ' ')]).strip().replace(' ', '_')
            # Folder path: detections/phone/John_Doe/ or detections/sleep/John_Doe/
            target_folder = os.path.join(self.output_dir, type_folder, safe_person_name)
            # Filename: phone_camera1_01-25-11-02-22_John_Doe.jpg
            filename_base = f"{type_folder}_{safe_cam_name}_{timestamp_fn}_{safe_person_name}.jpg"
        else:
            # Unknown person - all go to single "Unknown" folder
            target_folder = os.path.join(self.output_dir, type_folder, "Unknown")
            # Filename: phone_camera1_01-25-11-02-22_id42.jpg
            id_str = f"id{track_id}" if track_id is not None else "unknown"
            filename_base = f"{type_folder}_{safe_cam_name}_{timestamp_fn}_{id_str}.jpg"
        
        # Create folder if it doesn't exist
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        filename = os.path.join(target_folder, filename_base)
        
        from async_image_writer import AsyncImageWriter
        AsyncImageWriter.save(filename, evidence_img)
        print(f"📸 EVIDENCE SAVED: {filename} (Duration: {duration})")
