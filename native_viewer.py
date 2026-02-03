"""
OpenCV Native Video Viewer

High-performance video display using OpenCV's native window.
Runs separately from Streamlit for 60fps+ playback.

Controls:
- Q / ESC: Close viewer
- D: Detection view (with boxes)
- R: Raw view (no boxes)
"""

import cv2
import numpy as np
import threading
import time
import math


class NativeVideoViewer:
    """
    OpenCV-based video viewer for high-performance display.
    Shows all cameras in a grid layout at 60fps.
    Supports dual-view mode: Detection (with boxes) or Raw (no boxes).
    """
    
    def __init__(self, camera_manager, window_name="Phone Monitor - Live View"):
        self.camera_manager = camera_manager
        self.window_name = window_name
        self.running = False
        self.thread = None
        self.grid_width = 1920
        self.grid_height = 1080
        
        # View mode: 'detection' (with boxes) or 'raw' (no boxes)
        self.view_mode = 'detection'
        
    def start(self):
        """Start the viewer in a background thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[NativeViewer] Started: {self.window_name}")
        
    def stop(self):
        """Stop the viewer."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        print(f"[NativeViewer] Stopped")
        
    def _run(self):
        """Main display loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.grid_width, self.grid_height)
        
        self.fullscreen = False
        
        while self.running:
            composite = self._create_composite()
            cv2.imshow(self.window_name, composite)
            
            # Check for keyboard input
            key = cv2.waitKey(16)  # ~60fps
            
            if key == ord('q') or key == 27:  # q or ESC
                self.running = False
                break
            elif key == ord('d') or key == ord('D'):
                self.view_mode = 'detection'
                print("[NativeViewer] Switched to DETECTION view (with boxes)")
            elif key == ord('r') or key == ord('R'):
                self.view_mode = 'raw'
                print("[NativeViewer] Switched to RAW view (no boxes)")
            elif key == ord('f') or key == ord('F'):
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
            # Check if window was closed
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                break
        
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        
    def _create_composite(self):
        """Create a composite grid of all camera feeds with status bar BELOW."""
        active_cams = self.camera_manager.get_active_cameras()
        num_cams = len(active_cams)
        
        if num_cams == 0:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "No Cameras Connected", (50, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
        
        # Get first camera's frame to determine source resolution
        first_cam = list(active_cams.values())[0]
        sample_frame = first_cam.get_frame()
        if sample_frame is not None:
            src_height, src_width = sample_frame.shape[:2]
        else:
            src_width, src_height = 1280, 720
        
        # Calculate grid layout
        if num_cams == 1:
            cols, rows = 1, 1
        elif num_cams == 2:
            cols, rows = 2, 1
        else:
            cols = int(math.ceil(math.sqrt(num_cams)))
            rows = int(math.ceil(num_cams / cols))
        
        # Grid dimensions
        grid_width = src_width * cols
        grid_height = src_height * rows
        
        # Add space for status bar at bottom
        status_bar_height = 40
        total_height = grid_height + status_bar_height
        
        cell_width = src_width
        cell_height = src_height
        
        # Create composite canvas
        composite = np.zeros((total_height, grid_width, 3), dtype=np.uint8)
        
        # Fill each cell
        for idx, (cam_id, cam_thread) in enumerate(active_cams.items()):
            row = idx // cols
            col = idx % cols
            
            x_start = col * cell_width
            y_start = row * cell_height
            
            # Get Frame
            if self.view_mode == 'detection':
                raw_frame = cam_thread.get_raw_frame()
                if raw_frame is not None:
                    frame = cam_thread.draw_overlay_on_frame(raw_frame)
                else:
                    frame = None
            else:
                frame = cam_thread.get_raw_frame()
                
            status = cam_thread.get_status()
            camera_name = cam_thread.camera_name
            
            if frame is not None:
                h, w = frame.shape[:2]
                if h != cell_height or w != cell_width:
                    resized = cv2.resize(frame, (cell_width, cell_height))
                else:
                    resized = frame
            else:
                resized = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                cv2.putText(resized, "No Signal", (10, cell_height // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            
            # Draw Status Border (colored border around video)
            color = (0, 255, 0)
            if status == "texting": color = (0, 0, 255)
            elif status == "sleeping": color = (255, 0, 128)
            elif status == "disconnected": color = (0, 165, 255)
            
            cv2.rectangle(resized, (0, 0), (cell_width - 1, cell_height - 1), color, 4)
            
            # Place video in composite
            composite[y_start:y_start + cell_height, x_start:x_start + cell_width] = resized
            
            # Draw Status Text in the Bottom Bar Area (below the video column)
            # Center the text under the video content
            bar_y = grid_height + 25
            text_x = x_start + 10
            
            status_text = f"{camera_name}: {status.upper()}"
            cv2.putText(composite, status_text, (text_x, bar_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw View Mode Instructions at far right of status bar
        mode_label = "DETECTION (D)" if self.view_mode == 'detection' else "RAW (R)"
        instr = f"Q: Quit | F: Fullscreen | {mode_label}"
        text_size, _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(composite, instr, 
                   (grid_width - text_size[0] - 10, grid_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return composite
    
    def is_running(self):
        return self.running


# Global viewer instance
_viewer = None

def launch_viewer(camera_manager):
    """Launch the native video viewer."""
    global _viewer
    if _viewer is None or not _viewer.is_running():
        _viewer = NativeVideoViewer(camera_manager)
        _viewer.start()
        return True
    return False

def stop_viewer():
    """Stop the native video viewer."""
    global _viewer
    if _viewer:
        _viewer.stop()
        _viewer = None

def is_viewer_running():
    """Check if viewer is running."""
    global _viewer
    return _viewer is not None and _viewer.is_running()
