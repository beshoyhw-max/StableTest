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
    
    def update_person(self, person_name, box, frame, camera_name):
        """
        Called when a recognized person is detected in any camera.
        Updates their last seen time and position.
        
        Args:
            person_name: Name of the recognized person
            box: Tuple (x1, y1, x2, y2) bounding box
            frame: Current frame (will be copied for evidence)
            camera_name: Name of the camera where person was seen
        """
        if not person_name or person_name == "Unknown":
            return
            
        current_time = time.time()
        
        with self.lock:
            if person_name in self.tracked_people:
                person_data = self.tracked_people[person_name]
                person_data["last_seen"] = current_time
                person_data["last_box"] = box
                person_data["last_frame"] = frame.copy()
                person_data["last_camera"] = camera_name
                
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
                self.tracked_people[person_name] = {
                    "last_seen": current_time,
                    "last_box": box,
                    "last_frame": frame.copy(),
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
        
        with self.lock:
            for person_name, data in self.tracked_people.items():
                if data["status"] == "present":
                    time_since_seen = current_time - data["last_seen"]
                    
                    if time_since_seen >= self.absence_threshold:
                        # Person has been absent for threshold time
                        if not data["absence_notified"]:
                            data["status"] = "absent"
                            data["absence_notified"] = True
                            
                            # Save evidence with composite image
                            # BUG FIX: Get current frame from the SAME camera that last saw the person
                            evidence_frame = current_frame
                            evidence_camera = current_camera
                            
                            if self.camera_frame_getter and data["last_camera"] != current_camera:
                                # Try to get frame from the correct camera
                                try:
                                    correct_frame = self.camera_frame_getter(data["last_camera"])
                                    if correct_frame is not None:
                                        evidence_frame = correct_frame
                                        evidence_camera = data["last_camera"]
                                        print(f"  → Retrieved correct frame from {evidence_camera} for absence evidence")
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
            # Fallback: use last frame with "CURRENT" text overlay
            right_img = last_frame.copy()
            cv2.putText(right_img, "NO CURRENT FRAME", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
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
