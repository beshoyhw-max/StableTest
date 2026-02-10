"""
Attendance Tracker Module

Tracks recognized people across all cameras and alerts when they are not seen
for a configurable time period. Captures evidence with last known position.
"""

import cv2
import numpy as np
import os
import time
import datetime
import threading


class AttendanceTracker:
    """
    Global attendance tracker that monitors when recognized people leave.
    Shared across all camera threads to provide unified presence tracking.
    """
    
    def __init__(self, absence_threshold_seconds=300.0, output_dir="detections", camera_frame_getter=None):
        """
        Initialize the attendance tracker.
        
        Args:
            absence_threshold_seconds: Seconds before a person is marked as absent
            output_dir: Base directory for saving absence evidence
        """
        self.absence_threshold = absence_threshold_seconds
        self.output_dir = output_dir
        self.camera_frame_getter = camera_frame_getter
        self.lock = threading.Lock()
        
        # Tracked people: {person_name: TrackedPerson dict}
        self.tracked_people = {}
        
        # List of absence events for UI display
        self.absence_events = []
        
        print(f"  → AttendanceTracker initialized")
        print(f"     • Absence threshold: {absence_threshold_seconds} seconds")
        self.active = True
        self.last_check_time = time.time()

    def stop(self):
        """Stop tracking and suppress new alerts (used during shutdown)."""
        self.active = False
        
    def start(self):
        """Resume tracking (used when returning from production)."""
        self.active = True
        self.last_check_time = time.time()
        # Clear old state to avoid immediate alerts
        with self.lock:
            self.tracked_people.clear()
            self.absence_events.clear()

    
    def update_person(self, person_name, box, frame, camera_name, confidence=1.0):
        """
        Called when a recognized person is detected in any camera.
        Updates their last seen time and position using a Smart Hero Shot logic.
        
        Args:
            person_name: Name of the recognized person
            box: Tuple (x1, y1, x2, y2) bounding box
            frame: Current frame (will be copied for evidence)
            camera_name: Name of the camera where person was seen
            confidence: Face recognition confidence (0.0 - 1.0)
        """
        if not self.active or not person_name or person_name == "Unknown":
            return
            
        current_time = time.time()
        
        # Calculate score: Box Area * Confidence
        # This prioritizes LARGE faces that are also High Confidence (Frontal/Clear)
        # Avoids back-of-head (low confidence) or tiny faces (low resolution)
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        
        current_score = box_area * confidence

        # Prepare resized frame (Save RAM)
        h, w = frame.shape[:2]
        scale = min(640/w, 640/h)
        if scale < 1.0:
            small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            # SCALE THE BOX TO MATCH THE RESIZED FRAME
            sx1 = int(x1 * scale)
            sy1 = int(y1 * scale)
            sx2 = int(x2 * scale)
            sy2 = int(y2 * scale)
            stored_box = (sx1, sy1, sx2, sy2)
        else:
            small_frame = frame.copy()
            stored_box = box

        
        with self.lock:
            if person_name in self.tracked_people:
                person_data = self.tracked_people[person_name]
                person_data["last_seen"] = current_time
                person_data["last_camera"] = camera_name

                # --- SMART HERO SHOT LOGIC ---
                # Goal: Keep the "Best" face (largest box) from the last 2 seconds.
                # This prevents overwriting a good face shot with a back-of-head shot 
                # immediately before they leave, but ensures evidence isn't ancient.
                
                last_best_ts = person_data.get("last_best_ts", 0)
                last_best_score = person_data.get("last_best_score", 0)
                
                # Check if the stored "best" frame is too old (> 2.0s)
                # If it's old, we MUST update it (even if the new one is worse/back of head)
                # to prove they were present recently.
                is_window_expired = (current_time - last_best_ts) > 2.0
                
                # Is this new frame BETTER than the current cached one?
                is_better_score = current_score > last_best_score
                
                if is_window_expired or is_better_score:
                    # Update the evidence frame
                    person_data["last_box"] = stored_box

                    person_data["last_frame"] = small_frame
                    person_data["last_best_score"] = current_score
                    person_data["last_best_ts"] = current_time
                    # Only update frame if we are updating the "best" logic
                
                # If they were absent and returned, reset status
                if person_data["status"] == "absent":
                    print(f"  ✓ {person_name} RETURNED (was absent)")
                    person_data["status"] = "present"
                    person_data["absence_notified"] = False
                    
                    # Remove from absence events
                    self.absence_events = [
                        e for e in self.absence_events 
                        if e["name"] != person_name
                    ]
            else:
                # First time seeing this person
                # SAFETY LIMIT: Prevent memory leaks from ghost detections (max 100 people)
                if len(self.tracked_people) >= 100:
                    # Remove oldest tracked person (FIFO)
                    oldest_person = min(self.tracked_people.keys(), 
                                      key=lambda k: self.tracked_people[k]["last_seen"])
                    del self.tracked_people[oldest_person]
                    print(f"  ⚠ AttendanceTracker limit reached. Removed oldest: {oldest_person}")

                self.tracked_people[person_name] = {
                    "last_seen": current_time,
                    "last_box": stored_box,

                    "last_frame": small_frame,
                    "last_best_score": current_score,
                    "last_best_ts": current_time,
                    "last_camera": camera_name,
                    "status": "present",
                    "absence_notified": False
                }
                print(f"  → Now tracking: {person_name}")
    
    def check_absences(self, current_frame=None, current_camera=None):
        """
        Check all tracked people for absences.
        Should be called periodically from camera processing loop.
        
        Args:
            current_frame: Current frame from any camera (for composite evidence)
            current_camera: Name of the camera providing the current frame
        """
        current_time = time.time()
        
        if not self.active:
            return
        
        with self.lock:
            # --- TIME JUMP / SLEEP DETECTION ---
            # If the last check was > 5 seconds ago (loop should run every 1s),
            # the computer likely went to sleep or process hung.
            # We must PAUSE the absence timers by shifting 'last_seen' forward.
            time_diff = current_time - self.last_check_time
            self.last_check_time = current_time
            
            if time_diff > 5.0:
                print(f"  ⚠ System time jump detected ({time_diff:.1f}s). Computer likely slept.")
                print("    → Adjusting absence timers to prevent false positive...")
                
                # Add the 'slept' duration to last_seen for everyone
                # This effectively ignores the time spent sleeping
                compensate = time_diff - 1.0 # Subtract expected 1s interval
                for data in self.tracked_people.values():
                    if data["status"] == "present":
                        data["last_seen"] += compensate
                
                # Skip detection this cycle (allow cameras to reconnect)
                return

            for person_name, data in self.tracked_people.items():
                if data["status"] == "present":
                    time_since_seen = current_time - data["last_seen"]
                    
                    if time_since_seen >= self.absence_threshold:
                        # Person has been absent for threshold time
                        if not data["absence_notified"]:
                            data["status"] = "absent"
                            data["absence_notified"] = True
                            
                            # Save evidence with composite image
                            # FIXED: ALWAYS get current frame from the SAME camera that last saw the person
                            evidence_frame = None
                            evidence_camera = data["last_camera"]
                            
                            if self.camera_frame_getter:
                                try:
                                    evidence_frame = self.camera_frame_getter(data["last_camera"])
                                    if evidence_frame is not None:
                                        print(f"  → Got current frame from {evidence_camera} for absence evidence")
                                except Exception as e:
                                    print(f"  ⚠ Failed to get frame from {data['last_camera']}: {e}")

                            self._save_absence_evidence(
                                person_name,
                                data["last_box"],
                                data["last_frame"],
                                data["last_camera"],
                                data["last_seen"],
                                evidence_frame,
                                evidence_camera,
                                time_since_seen
                            )
                            
                            # Add to absence events for UI
                            last_seen_time = datetime.datetime.fromtimestamp(
                                data["last_seen"]
                            ).strftime("%H:%M:%S")
                            
                            self.absence_events.append({
                                "name": person_name,
                                "last_camera": data["last_camera"],
                                "time": last_seen_time,
                                "duration_minutes": time_since_seen / 60
                            })
                            
                            print(f"  [ABSENT] {person_name} LEFT - not seen for {time_since_seen:.0f}s")
    
    def _save_absence_evidence(self, person_name, last_box, last_frame, last_camera,
                               last_seen_timestamp, current_frame, current_camera, duration_seconds):
        """
        Save composite screenshot: left side shows last seen, right side shows current.
        
        Args:
            person_name: Name of the absent person
            last_box: Last known bounding box (x1, y1, x2, y2)
            last_frame: Last frame where person was visible
            last_camera: Camera where last seen
            last_seen_timestamp: Unix timestamp when last seen
            current_frame: Current frame (when violation triggered)
            current_camera: Current camera name
            duration_seconds: How long they've been absent
        """
        if last_frame is None or last_box is None:
            return
        
        # Prepare last seen frame (left side)
        x1, y1, x2, y2 = map(int, last_box)
        left_img = last_frame.copy()
        
        # Draw orange box at last known position
        box_color = (0, 165, 255)  # Orange in BGR
        cv2.rectangle(left_img, (x1, y1), (x2, y2), box_color, 3)
        
        # Add "LAST SEEN" label on the box
        label = f"{person_name}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_w, label_h = label_size
        cv2.rectangle(left_img, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), box_color, -1)
        cv2.putText(left_img, label, (x1 + 2, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Prepare current frame (right side)
        if current_frame is not None:
            right_img = current_frame.copy()
        else:
            # Fallback: use last frame with warning overlay (camera may be disconnected)
            right_img = last_frame.copy()
            # Draw semi-transparent overlay
            overlay = right_img.copy()
            cv2.rectangle(overlay, (0, 0), (right_img.shape[1], right_img.shape[0]), (0, 0, 100), -1)
            right_img = cv2.addWeighted(overlay, 0.3, right_img, 0.7, 0)
            cv2.putText(right_img, "CAMERA OFFLINE", (50, right_img.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(right_img, "(Using last frame)", (50, right_img.shape[0]//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Resize both to same height for side-by-side
        target_height = 480
        left_h, left_w = left_img.shape[:2]
        right_h, right_w = right_img.shape[:2]
        
        left_scale = target_height / left_h
        right_scale = target_height / right_h
        
        left_resized = cv2.resize(left_img, (int(left_w * left_scale), target_height))
        right_resized = cv2.resize(right_img, (int(right_w * right_scale), target_height))
        
        # Create header bar (60px tall)
        header_height = 60
        total_width = left_resized.shape[1] + right_resized.shape[1]
        header = np.zeros((header_height, total_width, 3), dtype=np.uint8)
        
        # Header text
        last_seen_time = datetime.datetime.fromtimestamp(last_seen_timestamp).strftime("%H:%M:%S")
        current_time_str = datetime.datetime.now().strftime("%H:%M:%S")
        duration_str = f"{duration_seconds:.0f}s" if duration_seconds < 60 else f"{duration_seconds/60:.1f}min"
        
        header_text = f"ABSENT: {person_name} | Duration: {duration_str}"
        cv2.putText(header, header_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Create sub-headers for each panel
        left_header_height = 30
        left_header = np.zeros((left_header_height, left_resized.shape[1], 3), dtype=np.uint8)
        left_header[:] = (0, 165, 255)  # Orange
        left_text = f"LAST SEEN: {last_seen_time} | {last_camera}"
        cv2.putText(left_header, left_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        right_header = np.zeros((left_header_height, right_resized.shape[1], 3), dtype=np.uint8)
        right_header[:] = (0, 0, 200)  # Dark red
        right_text = f"NOW: {current_time_str} | {current_camera or 'Unknown'}"
        cv2.putText(right_header, right_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Stack: left panel (sub-header + image)
        left_panel = np.vstack([left_header, left_resized])
        right_panel = np.vstack([right_header, right_resized])
        
        # Combine horizontally
        combined_panels = np.hstack([left_panel, right_panel])
        
        # Stack header on top
        evidence_img = np.vstack([header, combined_panels])
        
        # Create folder structure: detections/absence/<person_name>/
        safe_person_name = "".join([c for c in person_name 
                                   if c.isalnum() or c in ('_', '-', ' ')]).strip().replace(' ', '_')
        target_folder = os.path.join(self.output_dir, "absence", safe_person_name)
        
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Filename
        timestamp_fn = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        safe_cam_name = "".join([c for c in last_camera 
                                if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        filename = f"absence_{safe_cam_name}_{timestamp_fn}_{safe_person_name}.jpg"
        filepath = os.path.join(target_folder, filename)
        
        # Use async writer to prevent blocking
        from async_image_writer import AsyncImageWriter
        AsyncImageWriter.save(filepath, evidence_img)
        print(f"[EVIDENCE] ABSENCE: {filepath}")
    
    def get_absent_people(self):
        """
        Get list of people currently marked as absent.
        Returns list of dicts with name, last_camera, time, duration_minutes.
        """
        with self.lock:
            return self.absence_events.copy()
    
    def get_present_people(self):
        """
        Get list of people currently marked as present.
        """
        with self.lock:
            return [
                name for name, data in self.tracked_people.items()
                if data["status"] == "present"
            ]
    
    def update_threshold(self, seconds):
        """
        Update the absence threshold dynamically.
        
        Args:
            seconds: New threshold in seconds
        """
        with self.lock:
            self.absence_threshold = seconds
            print(f"  → Absence threshold updated: {seconds} seconds")
    
    def clear_absence_alerts(self, person_name=None):
        """
        Clear absence alerts. If person_name provided, only clear that person.
        Otherwise clear all.
        """
        with self.lock:
            if person_name:
                self.absence_events = [
                    e for e in self.absence_events 
                    if e["name"] != person_name
                ]
                if person_name in self.tracked_people:
                    self.tracked_people[person_name]["absence_notified"] = False
            else:
                self.absence_events.clear()
                for data in self.tracked_people.values():
                    data["absence_notified"] = False
