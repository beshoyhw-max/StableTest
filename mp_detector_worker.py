"""
Multiprocessing Detector Worker

Runs PhoneDetector in a separate OS process to bypass Python's GIL.
Uses multiprocessing.Queue for frame I/O and shared memory for large frame data.

Architecture:
- Main Process: VideoReader captures frames, sends to detector process
- Detector Process: Runs YOLO inference, sends results back
- Communication: Queue for metadata, shared memory for frame bytes
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import traceback
import os
import sys

# Windows multiprocessing support (CRITICAL for spawn method)
if sys.platform == 'win32':
    # Use 'spawn' method explicitly on Windows (default, but be explicit)
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # Already set

# Suppress TensorFlow/CUDA warnings in child process
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def detector_process_loop(
    camera_name: str,
    camera_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    control_queue: mp.Queue,
    config: dict
):
    """
    Main loop for the detector process.
    
    Receives frames from input_queue, processes with PhoneDetector,
    sends results to output_queue.
    
    Args:
        camera_name: Name of the camera (for logging)
        camera_id: Unique camera ID
        input_queue: Queue receiving (shm_name, shape, dtype, frame_count, timestamp)
        output_queue: Queue sending (status, display_data, processed_fps)
        control_queue: Queue receiving control commands ("stop", "update_conf", etc.)
        config: Dict with detector configuration
    """
    print(f"[MP-{camera_name}] Detector process starting (PID: {os.getpid()})...")
    
    # Import heavy modules inside process (they can't be pickled)
    try:
        from detector import PhoneDetector
        import torch
    except Exception as e:
        print(f"[MP-{camera_name}] FATAL: Failed to import detector: {e}")
        output_queue.put({"error": str(e)})
        return
    
    # Initialize detector with its own YOLO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[MP-{camera_name}] Initializing PhoneDetector on {device}...")
    
    try:
        detector = PhoneDetector(
            model_path='yolo26s.engine',
            pose_model_path='yolo26s-pose.engine',
            model_instance=None,  # Always load private model
            pose_model_instance=None,
            lock=None,
            phone_duration_threshold=config.get('phone_duration', 5.0),
            sleep_duration_threshold=config.get('sleep_duration', 10.0),
            cooldown_seconds=config.get('cooldown_duration', 120.0),
            reset_gap=config.get('reset_gap', 2.5),
            enable_face_recognition=config.get('enable_face_recognition', True),
            device=device,
            attendance_tracker=None,  # Can't share across processes - handle in main
            shared_cooldowns=None,    # Can't share - handle in main
            shared_cooldowns_lock=None
        )
        print(f"[MP-{camera_name}] PhoneDetector initialized successfully")
    except Exception as e:
        print(f"[MP-{camera_name}] FATAL: Failed to initialize detector: {e}")
        traceback.print_exc()
        output_queue.put({"error": str(e)})
        return
    
    # Processing state
    conf_threshold = config.get('conf_threshold', 0.25)
    skip_frames = config.get('skip_frames', 5)
    sleep_sensitivity = config.get('sleep_sensitivity', 0.18)
    
    # FPS tracking
    fps_frame_count = 0
    fps_start_time = time.time()
    processed_fps = 0.0
    
    running = True
    shm = None
    current_shm_name = None
    
    print(f"[MP-{camera_name}] Entering main processing loop...")
    
    while running:
        # Check for control commands (non-blocking)
        try:
            while not control_queue.empty():
                cmd = control_queue.get_nowait()
                if cmd.get('action') == 'stop':
                    print(f"[MP-{camera_name}] Received stop command")
                    running = False
                    break
                elif cmd.get('action') == 'update_conf':
                    conf_threshold = cmd.get('value', conf_threshold)
                elif cmd.get('action') == 'update_skip_frames':
                    skip_frames = cmd.get('value', skip_frames)
                elif cmd.get('action') == 'update_sleep_sensitivity':
                    sleep_sensitivity = cmd.get('value', sleep_sensitivity)
                elif cmd.get('action') == 'update_phone_duration':
                    detector.phone_duration_threshold = cmd.get('value', detector.phone_duration_threshold)
                elif cmd.get('action') == 'update_sleep_duration':
                    detector.sleep_duration_threshold = cmd.get('value', detector.sleep_duration_threshold)
                elif cmd.get('action') == 'update_cooldown':
                    detector.cooldown_seconds = cmd.get('value', detector.cooldown_seconds)
                elif cmd.get('action') == 'update_reset_gap':
                    detector.reset_gap = cmd.get('value', detector.reset_gap)
        except Exception as e:
            print(f"[MP-{camera_name}] Control queue error: {e}")
        
        if not running:
            break
        
        # Get frame from queue (blocking with timeout)
        try:
            frame_meta = input_queue.get(timeout=0.5)
        except:
            continue
        
        if frame_meta is None:
            continue
        
        shm_name, shape, dtype_str, frame_count, timestamp = frame_meta
        
        try:
            # Attach to shared memory if new or different
            if shm_name != current_shm_name:
                if shm is not None:
                    try:
                        shm.close()
                    except:
                        pass
                shm = shared_memory.SharedMemory(name=shm_name)
                current_shm_name = shm_name
            
            # Reconstruct frame from shared memory
            frame = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
            # CRITICAL: Make a copy! Shared memory may be overwritten by producer
            frame = frame.copy()
            
            # Process frame
            start_time = time.time()
            processed_frame, status, is_saved = detector.process_frame(
                frame,
                frame_count,
                skip_frames=skip_frames,
                save_screenshots=True,
                conf_threshold=conf_threshold,
                camera_name=camera_name,
                sleep_sensitivity=sleep_sensitivity
            )
            process_time = time.time() - start_time
            
            # FPS tracking (only count inference frames)
            if frame_count % skip_frames == 0:
                fps_frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    processed_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = time.time()
            
            # Get display data for drawing on main process
            display_data = detector.get_detection_data()
            
            # Send results back (small data only, no frame)
            try:
                output_queue.put({
                    'status': status,
                    'display_data': display_data,
                    'fps': processed_fps,
                    'frame_count': frame_count,
                    'process_time': process_time,
                    'timestamp': time.time()
                }, block=False)
            except:
                pass  # Queue full, drop result
            
        except Exception as e:
            print(f"[MP-{camera_name}] Processing error: {e}")
            traceback.print_exc()
    
    # Cleanup
    print(f"[MP-{camera_name}] Shutting down...")
    if shm is not None:
        try:
            shm.close()
        except:
            pass
    
    # Stop async workers in detector
    try:
        if hasattr(detector, 'async_sleep_worker') and detector.async_sleep_worker:
            detector.async_sleep_worker.stop()
        if hasattr(detector, 'async_face_worker') and detector.async_face_worker:
            detector.async_face_worker.stop()
    except:
        pass
    
    print(f"[MP-{camera_name}] Detector process exited cleanly")


class MPDetectorWorker:
    """
    Wrapper class that manages a detector running in a separate process.
    
    Usage:
        worker = MPDetectorWorker(camera_name, camera_id, config)
        worker.start()
        
        # In main loop:
        worker.submit_frame(frame, frame_count)
        result = worker.get_result()  # Non-blocking
        
        worker.stop()
    """
    
    def __init__(self, camera_name: str, camera_id: int, config: dict):
        self.camera_name = camera_name
        self.camera_id = camera_id
        self.config = config
        
        # Queues for communication
        # maxsize=1 ensures we always process latest frame (real-time)
        self.input_queue = mp.Queue(maxsize=2)
        self.output_queue = mp.Queue(maxsize=10)
        self.control_queue = mp.Queue(maxsize=10)
        
        # Shared memory for frame data (avoids pickle overhead)
        self.shm = None
        self.shm_size = 0
        
        # Process handle
        self.process = None
        
        # Cached results
        self.last_result = {
            'status': 'initializing',
            'display_data': [],
            'fps': 0.0
        }
    
    def start(self):
        """Start the detector process."""
        print(f"[MPWorker-{self.camera_name}] Starting detector process...")
        
        self.process = mp.Process(
            target=detector_process_loop,
            args=(
                self.camera_name,
                self.camera_id,
                self.input_queue,
                self.output_queue,
                self.control_queue,
                self.config
            ),
            daemon=True
        )
        self.process.start()
        print(f"[MPWorker-{self.camera_name}] Process started with PID: {self.process.pid}")
    
    def stop(self):
        """Stop the detector process gracefully."""
        print(f"[MPWorker-{self.camera_name}] Stopping detector process...")
        
        # Send stop command
        try:
            self.control_queue.put({'action': 'stop'}, timeout=1.0)
        except:
            pass
        
        # Wait for process to exit
        if self.process is not None:
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                print(f"[MPWorker-{self.camera_name}] Process didn't exit, terminating...")
                self.process.terminate()
                self.process.join(timeout=2.0)
        
        # Cleanup shared memory
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except:
                pass
        
        print(f"[MPWorker-{self.camera_name}] Stopped")
    
    def submit_frame(self, frame: np.ndarray, frame_count: int) -> bool:
        """
        Submit a frame for processing (non-blocking).
        
        Returns True if frame was submitted, False if queue is full.
        """
        if frame is None or self.process is None or not self.process.is_alive():
            return False
        
        try:
            # Calculate required shared memory size
            frame_bytes = frame.nbytes
            
            # Reallocate shared memory if needed
            if self.shm is None or frame_bytes > self.shm_size:
                if self.shm is not None:
                    try:
                        self.shm.close()
                        self.shm.unlink()
                    except Exception as e:
                        print(f"[MPWorker-{self.camera_name}] Warning: Shared memory cleanup failed: {e}")
                
                # Create new shared memory block
                shm_name = f"cam_{self.camera_id}_{int(time.time() * 1000)}"
                self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=frame_bytes)
                self.shm_size = frame_bytes
            
            # Copy frame to shared memory
            shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=self.shm.buf)
            np.copyto(shm_array, frame)
            
            # Send metadata (small, fast to pickle)
            meta = (
                self.shm.name,
                frame.shape,
                str(frame.dtype),
                frame_count,
                time.time()
            )
            
            # Non-blocking put with timeout
            self.input_queue.put(meta, timeout=0.01)
            return True
            
        except Exception as e:
            # Queue full or other error - drop frame (intentional for real-time)
            return False
    
    def get_result(self) -> dict:
        """
        Get latest processing result (non-blocking).
        
        Returns cached result if no new data available.
        """
        # Drain queue to get latest result
        while True:
            try:
                result = self.output_queue.get_nowait()
                if 'error' in result:
                    print(f"[MPWorker-{self.camera_name}] Error from detector: {result['error']}")
                else:
                    self.last_result = result
            except:
                break
        
        return self.last_result
    
    def update_conf_threshold(self, value: float):
        """Update confidence threshold."""
        try:
            self.control_queue.put({'action': 'update_conf', 'value': value}, timeout=0.1)
        except:
            pass
    
    def update_skip_frames(self, value: int):
        """Update skip frames."""
        try:
            self.control_queue.put({'action': 'update_skip_frames', 'value': value}, timeout=0.1)
        except:
            pass
    
    def update_sleep_sensitivity(self, value: float):
        """Update sleep sensitivity."""
        try:
            self.control_queue.put({'action': 'update_sleep_sensitivity', 'value': value}, timeout=0.1)
        except:
            pass
    
    def update_phone_duration(self, value: float):
        """Update phone duration threshold."""
        try:
            self.control_queue.put({'action': 'update_phone_duration', 'value': value}, timeout=0.1)
        except:
            pass
    
    def update_sleep_duration(self, value: float):
        """Update sleep duration threshold."""
        try:
            self.control_queue.put({'action': 'update_sleep_duration', 'value': value}, timeout=0.1)
        except Exception as e:
            print(f"[MPWorker-{self.camera_name}] Failed to send sleep_duration update: {e}")
    
    def update_cooldown(self, value: float):
        """Update cooldown duration."""
        try:
            self.control_queue.put({'action': 'update_cooldown', 'value': value}, timeout=0.1)
        except Exception as e:
            print(f"[MPWorker-{self.camera_name}] Failed to send cooldown update: {e}")
    
    def update_reset_gap(self, value: float):
        """Update reset gap duration."""
        try:
            self.control_queue.put({'action': 'update_reset_gap', 'value': value}, timeout=0.1)
        except Exception as e:
            print(f"[MPWorker-{self.camera_name}] Failed to send reset_gap update: {e}")
    
    def is_alive(self) -> bool:
        """Check if the detector process is still running."""
        return self.process is not None and self.process.is_alive()
