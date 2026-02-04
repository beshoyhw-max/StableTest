import cv2
import threading
import time
import json
import os
import datetime
from detector import PhoneDetector
from ultralytics import YOLO
import torch
from attendance_tracker import AttendanceTracker
# Force TCP connection (critical for Huawei cameras and general RTSP stability)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class VideoReader:
    """
    Dedicated thread for reading frames from a video source.
    Ensures that we always have the latest frame available, discarding older ones.
    """
    def __init__(self, source, camera_name="Unknown"):
        self.source = source
        self.camera_name = camera_name
        
        self.cap = None
        self.frame = None
        self.last_read_time = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.running = False
        self.connected = False
        
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        
    def start(self):
        self.running = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def update(self):
        print(f"[{self.camera_name}] VideoReader started for source: {self.source}")
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.connected = False
                print(f"[{self.camera_name}] Connecting to source...")
                
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None

                self.cap = cv2.VideoCapture(self.source)
                
                # Check if source is a webcam (integer) or other type
                is_webcam = isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit())
                
                if is_webcam:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                
                self.fps = 30
                self.is_file = False
                if not is_webcam and isinstance(self.source, str) and not self.source.startswith("rtsp"):
                     if os.path.exists(self.source):
                         self.is_file = True
                         self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                         if self.fps <= 0: 
                             self.fps = 30
                         print(f"[{self.camera_name}] File detected. FPS: {self.fps}")

                if not self.cap.isOpened():
                    print(f"[{self.camera_name}] Connection failed. Retrying in 5s...")
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                    time.sleep(5)
                    continue
                
                print(f"[{self.camera_name}] Connected successfully.")
                self.connected = True

            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    self.last_read_time = time.time()
                    
                    # FPS Calculation
                    self.fps_frame_count += 1
                    if time.time() - self.fps_start_time > 1.0:
                        self.fps_frame_count = 0
                        self.fps_start_time = time.time()

                    # No artificial sleep - let the consumer (CameraThread) control pacing
                        
                else:
                    # Read failed - could be end of file or stream error
                    if self.is_file:
                        # For video files: loop back to beginning
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            print(f"[{self.camera_name}] Video looped to beginning")
                            continue
                        except:
                            pass
                    
                    # For streams/cameras: reconnect
                    if self.cap:
                        try:
                            self.cap.release()
                        except:
                            pass
                        self.cap = None
                    self.connected = False
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"[{self.camera_name}] Error reading frame: {e}")
                self.connected = False
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                time.sleep(1)

    def get_frame(self):
        return self.frame, self.last_read_time
        
    def is_connected(self):
        return self.connected and (time.time() - self.last_read_time < 3.0)


class CameraThread(threading.Thread):
    def __init__(self, camera_config, conf_threshold=0.25, 
                 phone_duration=5.0, sleep_duration=10.0, cooldown_duration=120.0,
                 reset_gap=2.5,  # ADDED: Configurable reset gap
                 skip_frames=5, enable_face_recognition=False,
                 shared_model=None, shared_pose_model=None, model_lock=None,
                 sleep_sensitivity=0.18, attendance_tracker=None,
                 shared_cooldowns=None, shared_cooldowns_lock=None):
        """
        Camera thread with TIME-BASED detection thresholds and face recognition.
        
        Args:
            phone_duration: Seconds of continuous phone use before alert
            sleep_duration: Seconds of continuous sleep before alert
            cooldown_duration: Seconds between evidence screenshots
            reset_gap: Seconds to wait before clearing a missing detection (buffer)
            skip_frames: Process every Nth frame (higher = faster, less accurate)
            enable_face_recognition: Enable face recognition for person identification
            shared_model: IGNORED/DEPRECATED - Now using private model for tracking
            shared_pose_model: Pre-loaded Pose model (performance optim)
            sleep_sensitivity: EAR threshold for sleep detection
            shared_cooldowns: Global cooldown dict for recognized persons
            shared_cooldowns_lock: Thread lock for shared cooldowns
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.source = camera_config['source']
        
        print(f"[{self.camera_name}] Initializing private detector with time-based thresholds...")
        print(f"  → Phone alert after: {phone_duration}s continuous use")
        print(f"  → Sleep alert after: {sleep_duration}s continuous sleep")
        print(f"  → Reset gap: {reset_gap}s")
        print(f"  → Cooldown period: {cooldown_duration}s")
        print(f"  → Face recognition: {'ENABLED' if enable_face_recognition else 'DISABLED'}")
        
        self.detector = PhoneDetector(
            model_path='yolo26s.engine',
            pose_model_path='yolo26s-pose.engine',
            model_instance=None, # Use private model for Tracking (Fixes ID mixing)
            pose_model_instance=shared_pose_model,
            lock=model_lock,
            phone_duration_threshold=phone_duration,
            sleep_duration_threshold=sleep_duration,
            cooldown_seconds=cooldown_duration,
            reset_gap=reset_gap,
            enable_face_recognition=enable_face_recognition,
	    device=self.device,
            attendance_tracker=attendance_tracker,
            shared_cooldowns=shared_cooldowns,
            shared_cooldowns_lock=shared_cooldowns_lock
        )
        print(f"[{self.camera_name}] Detector ready.")
        
        self.conf_threshold = conf_threshold
        self.phone_duration = phone_duration
        self.sleep_duration = sleep_duration
        self.cooldown_duration = cooldown_duration
        self.skip_frames = skip_frames
        self.enable_face_recognition = enable_face_recognition
        self.sleep_sensitivity = sleep_sensitivity
        
        self.running = False
        self.latest_processed_frame = None
        self.status = "safe"
        self.last_update_time = 0
        self.last_processed_timestamp = 0
        
        # Processed FPS tracking
        self._fps_frame_count = 0
        self._fps_start_time = time.time()
        self._processed_fps = 0.0
        
        self.reader = VideoReader(self.source, self.camera_name)
        
    def run(self):
        self.running = True
        print(f"[{self.camera_name}] Starting processing thread...")
        self.reader.start()
        
        frame_count = 0
        
        while self.running:
            if not self.reader.is_connected():
                self.status = "disconnected"
                time.sleep(0.5)
                continue
            
            raw_frame, timestamp = self.reader.get_frame()
            
            if raw_frame is None or timestamp == self.last_processed_timestamp:
                time.sleep(0.01)
                continue
            
            self.last_processed_timestamp = timestamp

            try:
                processed_frame, status, is_saved = self.detector.process_frame(
                    raw_frame, 
                    frame_count, 
                    skip_frames=self.skip_frames,
                    save_screenshots=True,
                    conf_threshold=self.conf_threshold,
                    camera_name=self.camera_name,
                    sleep_sensitivity=self.sleep_sensitivity
                )
                
                self.latest_processed_frame = processed_frame
                self.status = status
                self.last_update_time = time.time()
                
                # Track INFERENCE-ONLY FPS (only count frames with actual YOLO detection)
                # Inference runs when: frame_count % skip_frames == 0
                if frame_count % self.skip_frames == 0:
                    self._fps_frame_count += 1
                    elapsed = time.time() - self._fps_start_time
                    if elapsed >= 1.0:
                        self._processed_fps = self._fps_frame_count / elapsed
                        self._fps_frame_count = 0
                        self._fps_start_time = time.time()
                
            except Exception as e:
                print(f"[{self.camera_name}] Error in processing: {e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
            # Small sleep to prevent CPU spinning (UI timing handles actual display rate)
            time.sleep(0.005)

        print(f"[{self.camera_name}] Processing thread stopped.")
        self.reader.stop()

    def stop(self):
        self.running = False
        self.join(timeout=2.0)

    def get_frame(self):
        return self.latest_processed_frame

    def get_status(self):
        # Increased from 3.0 to 8.0 for slower medium models
        if time.time() - self.last_update_time > 8.0:
            return "disconnected"
        return self.status
        
    def get_fps(self):
        """Get the source FPS of the video reader."""
        if hasattr(self, 'reader') and self.reader:
            return getattr(self.reader, 'fps', 30.0)
        return 30.0
    
    def get_processed_fps(self):
        """Get the actual processed FPS (inference rate)."""
        return self._processed_fps
    
    def get_raw_frame(self):
        """
        Get the latest raw frame directly from VideoReader (no boxes).
        Use for 60fps display with overlaid detections.
        """
        if hasattr(self, 'reader') and self.reader:
            frame, _ = self.reader.get_frame()
            return frame
        return None
    
    def draw_overlay_on_frame(self, frame):
        """
        Draw cached detection boxes on any frame.
        Returns frame with detection overlay drawn.
        """
        if frame is None:
            return None
        if hasattr(self, 'detector') and self.detector:
            return self.detector.draw_detections_on_frame(frame)
        return frame
    
    def update_thresholds(self, conf=None, phone_dur=None, sleep_dur=None, cooldown=None, reset_gap=None, skip_frames=None, sleep_sensitivity=None):
        """Update detection thresholds on the fly."""
        if conf is not None:
            self.conf_threshold = conf
        if sleep_sensitivity is not None:
            self.sleep_sensitivity = sleep_sensitivity
        if phone_dur is not None:
            self.phone_duration = phone_dur
            self.detector.phone_duration_threshold = phone_dur
        if sleep_dur is not None:
            self.sleep_duration = sleep_dur
            self.detector.sleep_duration_threshold = sleep_dur
        if cooldown is not None:
            self.cooldown_duration = cooldown
            self.detector.cooldown_seconds = cooldown
        if skip_frames is not None:
            self.skip_frames = skip_frames
        if reset_gap is not None:
             self.detector.reset_gap = reset_gap


class MPCameraThread(threading.Thread):
    """
    Multiprocessing-based camera processor.
    
    Uses a separate OS process for PhoneDetector to bypass Python's GIL.
    VideoReader runs in main process (I/O bound), detector runs in child process (CPU bound).
    
    Provides the same interface as CameraThread for easy swapping.
    """
    
    def __init__(self, camera_config, conf_threshold=0.25, 
                 phone_duration=5.0, sleep_duration=10.0, cooldown_duration=120.0,
                 reset_gap=2.5, skip_frames=5, enable_face_recognition=False,
                 shared_model=None, shared_pose_model=None, model_lock=None,
                 sleep_sensitivity=0.18, attendance_tracker=None,
                 shared_cooldowns=None, shared_cooldowns_lock=None):
        """
        Initialize multiprocessing camera thread.
        Same parameters as CameraThread for compatibility.
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.source = camera_config['source']
        
        # Store config for passing to detector process
        self.detector_config = {
            'conf_threshold': conf_threshold,
            'phone_duration': phone_duration,
            'sleep_duration': sleep_duration,
            'cooldown_duration': cooldown_duration,
            'reset_gap': reset_gap,
            'skip_frames': skip_frames,
            'enable_face_recognition': enable_face_recognition,
            'sleep_sensitivity': sleep_sensitivity
        }
        
        self.conf_threshold = conf_threshold
        self.phone_duration = phone_duration
        self.sleep_duration = sleep_duration
        self.cooldown_duration = cooldown_duration
        self.skip_frames = skip_frames
        self.sleep_sensitivity = sleep_sensitivity
        
        # State
        self.running = False
        self.status = "initializing"
        self.last_update_time = time.time()
        self._processed_fps = 0.0
        
        # Cached display data for drawing
        self._display_data = []
        self._display_data_lock = threading.Lock()
        
        # Attendance tracker reference (used in main process only)
        self.attendance_tracker = attendance_tracker
        self.shared_cooldowns = shared_cooldowns
        self.shared_cooldowns_lock = shared_cooldowns_lock
        
        # Video reader (runs in this thread, I/O bound)
        self.reader = VideoReader(self.source, self.camera_name)
        
        # MP Worker (detector runs in separate process)
        from mp_detector_worker import MPDetectorWorker
        self.mp_worker = MPDetectorWorker(
            camera_name=self.camera_name,
            camera_id=self.camera_id,
            config=self.detector_config
        )
        
        print(f"[MP-{self.camera_name}] Initialized with multiprocessing detector")
    
    def run(self):
        """Main processing loop."""
        self.running = True
        print(f"[MP-{self.camera_name}] Starting processing...")
        
        # Start video reader
        self.reader.start()
        
        # Start detector process
        self.mp_worker.start()
        
        # Wait a moment for detector to initialize
        time.sleep(2.0)
        
        frame_count = 0
        last_timestamp = 0
        
        while self.running:
            # Check if detector process is alive
            if not self.mp_worker.is_alive():
                print(f"[MP-{self.camera_name}] Detector process died! Restarting...")
                self.mp_worker.start()
                time.sleep(2.0)
                continue
            
            if not self.reader.is_connected():
                self.status = "disconnected"
                time.sleep(0.5)
                continue
            
            # Get raw frame from video reader
            raw_frame, timestamp = self.reader.get_frame()
            
            if raw_frame is None or timestamp == last_timestamp:
                time.sleep(0.01)
                continue
            
            last_timestamp = timestamp
            
            # Submit frame to detector process
            self.mp_worker.submit_frame(raw_frame, frame_count)
            
            # Get latest results (non-blocking)
            result = self.mp_worker.get_result()
            
            if result:
                self.status = result.get('status', 'safe')
                self._processed_fps = result.get('fps', 0.0)
                self.last_update_time = time.time()
                
                # Cache display data for drawing
                with self._display_data_lock:
                    self._display_data = result.get('display_data', [])
            
            frame_count += 1
            time.sleep(0.005)  # Prevent CPU spinning
        
        print(f"[MP-{self.camera_name}] Stopping...")
        self.mp_worker.stop()
        self.reader.stop()
        print(f"[MP-{self.camera_name}] Stopped")
    
    def stop(self):
        """Stop the camera processing."""
        self.running = False
        self.join(timeout=5.0)
    
    def get_frame(self):
        """Get the latest processed frame (with detections drawn)."""
        raw_frame = self.get_raw_frame()
        if raw_frame is not None:
            return self.draw_overlay_on_frame(raw_frame)
        return None
    
    def get_status(self):
        """Get current detection status."""
        if time.time() - self.last_update_time > 8.0:
            return "disconnected"
        return self.status
    
    def get_fps(self):
        """Get the source FPS of the video reader."""
        if hasattr(self, 'reader') and self.reader:
            return getattr(self.reader, 'fps', 30.0)
        return 30.0
    
    def get_processed_fps(self):
        """Get the actual processed FPS (inference rate)."""
        return self._processed_fps
    
    def get_raw_frame(self):
        """Get the latest raw frame directly from VideoReader."""
        if hasattr(self, 'reader') and self.reader:
            frame, _ = self.reader.get_frame()
            return frame
        return None
    
    def draw_overlay_on_frame(self, frame):
        """Draw cached detection boxes on any frame."""
        if frame is None:
            return None
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Calculate dynamic scale
        import cv2
        scale_factor = max(w / 1280.0, h / 720.0)
        box_thickness = max(2, int(3 * scale_factor))
        font_scale = max(0.6, 0.8 * scale_factor)
        font_thick = max(1, int(2 * scale_factor))
        
        with self._display_data_lock:
            for data in self._display_data:
                if len(data) >= 7:
                    x1, y1, x2, y2, color, status, label = data[:7]
                    
                    # Draw box
                    cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thickness)
                    
                    # Draw label
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                    label_w, label_h = label_size
                    
                    pad_x = int(6 * scale_factor)
                    pad_y = int(6 * scale_factor)
                    
                    label_y_top = int(y1) - pad_y
                    if label_y_top - label_h < 0:
                        label_y_top = int(y1) + label_h + pad_y + box_thickness
                    
                    bg_p1 = (int(x1), label_y_top - label_h - pad_y)
                    bg_p2 = (int(x1) + label_w + pad_x * 2, label_y_top + pad_y)
                    
                    cv2.rectangle(output, bg_p1, bg_p2, color, -1)
                    cv2.putText(output, label, (int(x1) + pad_x, label_y_top), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
        
        return output
    
    def update_thresholds(self, conf=None, phone_dur=None, sleep_dur=None, 
                          cooldown=None, reset_gap=None, skip_frames=None, sleep_sensitivity=None):
        """Update detection thresholds on the fly."""
        if conf is not None:
            self.conf_threshold = conf
            self.mp_worker.update_conf_threshold(conf)
        if sleep_sensitivity is not None:
            self.sleep_sensitivity = sleep_sensitivity
            self.mp_worker.update_sleep_sensitivity(sleep_sensitivity)
        if phone_dur is not None:
            self.phone_duration = phone_dur
            self.mp_worker.update_phone_duration(phone_dur)
        if sleep_dur is not None:
            self.sleep_duration = sleep_dur
            self.mp_worker.update_sleep_duration(sleep_dur)
        if skip_frames is not None:
            self.skip_frames = skip_frames
            self.mp_worker.update_skip_frames(skip_frames)
        if cooldown is not None:
            self.cooldown_duration = cooldown
            self.mp_worker.update_cooldown(cooldown)
        if reset_gap is not None:
            self.mp_worker.update_reset_gap(reset_gap)


class CameraManager:
    def __init__(self, config_file="cameras.json", use_multiprocessing=False):
        """
        Camera Manager with TIME-BASED detection configuration and face recognition.
        
        Args:
            config_file: JSON file with camera configurations
            use_multiprocessing: If True, use separate processes for detectors (bypasses GIL)
                                 NOTE: Multiprocessing may not work well with Streamlit.
                                 Default is False (threading mode) for compatibility.
        """
        self.config_file = config_file
        self.cameras = {}
        self.use_multiprocessing = use_multiprocessing
        
        # Global thresholds
        self.global_conf = 0.25
        self.global_phone_duration = 5.0
        self.global_sleep_duration = 10.0
        self.global_cooldown = 120.0
        self.global_reset_gap = 2.5
        self.global_skip_frames = 5
        self.global_face_recognition = True  # Default enabled
        self.global_absence_threshold = 300.0  # Seconds (5 mins)
        
        # HYBRID COOLDOWN: Shared dict for recognized persons (global), local for unknown
        self.shared_cooldowns = {}  # key: (person_name, status) -> last_save_time
        self.shared_cooldowns_lock = threading.Lock()
        
        # Shared attendance tracker for global person monitoring
        self.attendance_tracker = AttendanceTracker(
            absence_threshold_seconds=self.global_absence_threshold,
            camera_frame_getter=self.get_camera_frame
        )
        
        print("=" * 70)
        if use_multiprocessing:
            print("Initializing Camera Manager - MULTIPROCESSING ARCHITECTURE")
            print("Each camera detector runs in its OWN PROCESS (bypasses GIL)")
        else:
            print("Initializing Camera Manager - THREADING ARCHITECTURE")
            print("Each camera detector runs in its own thread (GIL limited)")
        print("=" * 70)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_sleep_sensitivity = 0.18
        
        self.load_config_and_start()
        
        # Start FPS monitor thread
        self._fps_monitor_running = True
        self._fps_monitor_thread = threading.Thread(target=self._fps_monitor_loop, daemon=True)
        self._fps_monitor_thread.start()
        
        # Start Absence Monitor thread (Independent of FPS debug)
        self._absence_monitor_running = True
        self._absence_monitor_thread = threading.Thread(target=self._absence_monitor_loop, daemon=True)
        self._absence_monitor_thread.start()

    def load_config_and_start(self):
        if not os.path.exists(self.config_file):
            default_config = [
                {"id": 0, "name": "Webcam Main", "source": 0}
            ]
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        with open(self.config_file, 'r') as f:
            configs = json.load(f)
            
        for conf in configs:
            self.add_camera_thread(conf)

    def start_all_cameras(self):
        """Restart all cameras from config file. Used when returning from production mode."""
        # Stop any running cameras first
        for cam_id, cam in list(self.cameras.items()):
            try:
                cam.stop()
            except:
                pass
        self.cameras.clear()
        
        # Reload and start all cameras
        self.load_config_and_start()
        print("[CameraManager] All cameras restarted.")

    def add_camera_thread(self, config):
        """Add a camera thread/process with current global thresholds."""
        source = config['source']
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        config['source'] = source

        cam_id = config['id']
        if cam_id in self.cameras:
            print(f"Camera {cam_id} already running.")
            return
        
        print(f"\n{'='*60}")
        print(f"Starting Camera {cam_id}: {config['name']}")
        mode_str = "MULTIPROCESSING" if self.use_multiprocessing else "THREADING"
        print(f"Mode: {mode_str}")
        print(f"{'='*60}")
        
        # Choose thread type based on configuration
        ThreadClass = MPCameraThread if self.use_multiprocessing else CameraThread
        
        thread = ThreadClass(
            config,
            conf_threshold=self.global_conf,
            phone_duration=self.global_phone_duration,
            sleep_duration=self.global_sleep_duration,
            cooldown_duration=self.global_cooldown,
            reset_gap=self.global_reset_gap,
            skip_frames=self.global_skip_frames,
            enable_face_recognition=self.global_face_recognition,
            shared_model=None, 
            shared_pose_model=None,
            model_lock=None,
            sleep_sensitivity=self.global_sleep_sensitivity,
            attendance_tracker=self.attendance_tracker,
            shared_cooldowns=self.shared_cooldowns,
            shared_cooldowns_lock=self.shared_cooldowns_lock
        )
        thread.start()
        self.cameras[cam_id] = thread
        
        print(f"[Camera {cam_id}] Started successfully.\n")

    def add_camera(self, name, source):
        existing_ids = [c.camera_id for c in self.cameras.values()]
        new_id = max(existing_ids) + 1 if existing_ids else 0
        
        new_config = {"id": new_id, "name": name, "source": source}
        
        self.save_config_append(new_config)
        self.add_camera_thread(new_config)

    def remove_camera(self, cam_id):
        if cam_id in self.cameras:
            print(f"Removing camera {cam_id}...")
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]
            self.save_config_remove(cam_id)
            print(f"Camera {cam_id} removed.")

    def save_config_append(self, new_config):
        try:
            with open(self.config_file, 'r') as f:
                configs = json.load(f)
            configs.append(new_config)
            with open(self.config_file, 'w') as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def save_config_remove(self, cam_id):
        try:
            with open(self.config_file, 'r') as f:
                configs = json.load(f)
            configs = [c for c in configs if c['id'] != cam_id]
            with open(self.config_file, 'w') as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_active_cameras(self):
        return self.cameras

    def update_global_conf(self, conf):
        """Update confidence threshold for all cameras."""
        self.global_conf = conf
        for cam in self.cameras.values():
            cam.update_thresholds(conf=conf)
    
    def update_phone_duration(self, duration):
        """Update phone detection duration for all cameras."""
        self.global_phone_duration = duration
        for cam in self.cameras.values():
            cam.update_thresholds(phone_dur=duration)
    
    def update_sleep_duration(self, duration):
        """Update sleep detection duration for all cameras."""
        self.global_sleep_duration = duration
        for cam in self.cameras.values():
            cam.update_thresholds(sleep_dur=duration)
    
    def update_cooldown_duration(self, duration):
        """Update cooldown duration for all cameras."""
        self.global_cooldown = duration
        for cam in self.cameras.values():
            cam.update_thresholds(cooldown=duration)

    def update_reset_gap(self, duration):
        """Update reset buffer duration for all cameras."""
        self.global_reset_gap = duration
        for cam in self.cameras.values():
            cam.update_thresholds(reset_gap=duration)
    
    def update_skip_frames(self, skip_frames):
        """Update skip frames for all cameras."""
        self.global_skip_frames = skip_frames
        for cam in self.cameras.values():
            cam.update_thresholds(skip_frames=skip_frames)
            
    def update_sleep_sensitivity(self, sensitivity):
        """Update sleep sensitivity (EAR threshold) for all cameras."""
        self.global_sleep_sensitivity = sensitivity
        for cam in self.cameras.values():
            cam.update_thresholds(sleep_sensitivity=sensitivity)
    
    def enable_face_recognition(self, enabled):
        """
        Enable/disable face recognition globally.
        Note: Requires camera restart to take effect.
        """
        self.global_face_recognition = enabled
        print(f"Face recognition {'ENABLED' if enabled else 'DISABLED'} for new cameras")
        print("Note: Restart existing cameras for changes to take effect")
    
    def update_absence_threshold(self, seconds):
        """Update absence detection threshold for attendance tracking."""
        self.global_absence_threshold = seconds
        self.attendance_tracker.update_threshold(seconds)
    
    def get_absent_people(self):
        """Get list of people currently marked as absent."""
        return self.attendance_tracker.get_absent_people()
    
    def get_present_people(self):
        """Get list of people currently present."""
        return self.attendance_tracker.get_present_people()
    
    def clear_absence_alerts(self, person_name=None):
        """Clear absence alerts."""
        self.attendance_tracker.clear_absence_alerts(person_name)
    
    def _absence_monitor_loop(self):
        """Dedicated background thread for absence detection (High Precision)."""
        print("[System] Absence Monitor started (1s interval)")
        while self._absence_monitor_running:
            time.sleep(1.0)
            if self.attendance_tracker:
                self.attendance_tracker.check_absences(None, None)

    def _fps_monitor_loop(self):
        """Background thread that prints consolidated FPS for all cameras every 5 seconds."""
        while self._fps_monitor_running:
            time.sleep(5)
            
            if not self.cameras:
                continue
            
            fps_parts = []
            total_fps = 0.0
            for cam_id, cam in sorted(self.cameras.items()):
                fps = cam.get_processed_fps()
                total_fps += fps
                fps_parts.append(f"{cam.camera_name}: {fps:.1f}")
            
            fps_str = " | ".join(fps_parts)
            print(f"[FPS] {fps_str} | Total: {total_fps:.1f}")
    
    def stop_fps_monitor(self):
        """Stop the monitor threads."""
        self._fps_monitor_running = False
        self._absence_monitor_running = False

    def get_camera_frame(self, camera_name):
        """
        Get the latest raw frame from a specific camera by name.
        Used by AttendanceTracker to fetch evidence.
        """
        for cam in self.cameras.values():
            if cam.camera_name == camera_name:
                return cam.get_raw_frame()
        return None
