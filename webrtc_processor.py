"""
WebRTC Video Processor for Streamlit

This module provides a composite video stream that combines multiple camera feeds
into a single WebRTC stream for smooth 30fps+ playback.
"""

import cv2
import numpy as np
import av
import math
from collections import deque
import threading
import time


class CompositeVideoProcessor:
    """
    Creates a grid view of multiple camera feeds.
    Designed to work with streamlit-webrtc's video_frame_callback pattern.
    """
    
    def __init__(self, camera_manager, grid_width=1280, grid_height=720):
        """
        Args:
            camera_manager: CameraManager instance with active camera threads
            grid_width: Output composite width
            grid_height: Output composite height
        """
        self.camera_manager = camera_manager
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.last_statuses = {}
        self.lock = threading.Lock()
        
    def get_composite_frame(self):
        """
        Generate a composite frame with all camera feeds in a grid.
        Returns: numpy array (BGR format)
        """
        active_cams = self.camera_manager.get_active_cameras()
        num_cams = len(active_cams)
        
        if num_cams == 0:
            # No cameras - show placeholder
            frame = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
            cv2.putText(frame, "No Cameras Connected", (50, self.grid_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
        
        # Calculate grid layout
        cols = int(math.ceil(math.sqrt(num_cams)))
        rows = int(math.ceil(num_cams / cols))
        
        cell_width = self.grid_width // cols
        cell_height = self.grid_height // rows
        
        # Create blank composite
        composite = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        
        # Fill each cell
        for idx, (cam_id, cam_thread) in enumerate(active_cams.items()):
            row = idx // cols
            col = idx % cols
            
            x_start = col * cell_width
            y_start = row * cell_height
            
            # Get frame from camera
            frame = cam_thread.get_frame()
            status = cam_thread.get_status()
            camera_name = cam_thread.camera_name
            
            # Store status for alert checking
            with self.lock:
                self.last_statuses[cam_id] = {
                    'status': status,
                    'name': camera_name
                }
            
            if frame is not None:
                # Resize to fit cell
                resized = cv2.resize(frame, (cell_width, cell_height))
            else:
                # No signal placeholder
                resized = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                cv2.putText(resized, "No Signal", (10, cell_height // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            
            # Draw status overlay
            color = (0, 255, 0)  # Green = safe
            status_text = "SAFE"
            
            if status == "texting":
                color = (0, 0, 255)  # Red
                status_text = "PHONE DETECTED"
            elif status == "sleeping":
                color = (255, 0, 128)  # Purple
                status_text = "SLEEP DETECTED"
            elif status == "disconnected":
                color = (0, 165, 255)  # Orange
                status_text = "CONNECTING..."
            
            # Draw camera name and status
            cv2.rectangle(resized, (0, 0), (cell_width, 30), (0, 0, 0), -1)
            cv2.putText(resized, f"{camera_name}: {status_text}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw border based on status
            cv2.rectangle(resized, (0, 0), (cell_width - 1, cell_height - 1), color, 3)
            
            # Place in composite
            composite[y_start:y_start + cell_height, x_start:x_start + cell_width] = resized
        
        return composite
    
    def get_alerts(self):
        """
        Get current alert status for all cameras.
        Returns: dict with 'texting' and 'sleeping' lists of camera names
        """
        texting = []
        sleeping = []
        
        with self.lock:
            for cam_id, info in self.last_statuses.items():
                if info['status'] == 'texting':
                    texting.append(info['name'])
                elif info['status'] == 'sleeping':
                    sleeping.append(info['name'])
        
        return {'texting': texting, 'sleeping': sleeping}


def create_video_frame_callback(processor):
    """
    Factory function to create a video_frame_callback for streamlit-webrtc.
    This is used when the webrtc_streamer needs to generate frames.
    """
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        # Get composite frame from our processor
        img = processor.get_composite_frame()
        
        # Convert BGR to RGB for av.VideoFrame
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create new av.VideoFrame
        new_frame = av.VideoFrame.from_ndarray(img_rgb, format="rgb24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        
        return new_frame
    
    return callback
