"""
Production Viewer Launcher

Launches the production multiprocessing mode from within Streamlit.
This module handles the switch from threading to multiprocessing mode.
"""

import subprocess
import sys
import os
import time
import threading

# Track the production subprocess
_production_process = None
_saved_camera_manager = None
_recovery_thread = None

# Signal file for triggering Streamlit rerun
RERUN_SIGNAL_FILE = ".streamlit_rerun_signal"


def _start_recovery_watcher(camera_manager):
    """
    Watch for production process death and auto-restart Streamlit cameras.
    
    This handles ALL exit scenarios:
    - User presses Q
    - User presses Ctrl+C
    - User closes CMD window manually
    - Production process crashes
    """
    global _production_process, _recovery_thread
    
    def watcher():
        global _production_process
        
        while True:
            time.sleep(1.0)
            
            # Capture reference to avoid race condition (TOCTOU fix)
            proc = _production_process
            
            # Check if production died (process ended or was never started)
            if proc is None:
                break
            
            poll_result = proc.poll()
            if poll_result is not None:
                # Process has ended - trigger recovery
                print("[Recovery] Production process ended (exit code: {}), restarting cameras...".format(poll_result))
                _production_process = None
                
                time.sleep(2.0)  # Wait for GPU resources to be freed
                
                if camera_manager:
                    try:
                        camera_manager.start_all_cameras()
                        print("[Recovery] ✓ Streamlit cameras restarted successfully!")
                        
                        # Create signal file to trigger Streamlit rerun
                        from pathlib import Path
                        Path(RERUN_SIGNAL_FILE).touch()
                        print("[Recovery] Signal file created for Streamlit rerun")
                    except Exception as e:
                        print(f"[Recovery] Error restarting cameras: {e}")
                break
    
    _recovery_thread = threading.Thread(target=watcher, daemon=True)
    _recovery_thread.start()


def check_rerun_signal():
    """
    Check if a rerun signal exists (cameras were restarted in background).
    
    Returns:
        bool: True if signal exists (caller should trigger st.rerun())
    """
    from pathlib import Path
    signal_path = Path(RERUN_SIGNAL_FILE)
    
    if signal_path.exists():
        try:
            signal_path.unlink()  # Consume the signal
            return True
        except:
            pass
    return False


def launch_production_mode(camera_manager=None):
    """
    Launch production mode as a separate subprocess.
    This bypasses Streamlit's multiprocessing issues by running run_production.py externally.
    
    Args:
        camera_manager: If provided, stops its cameras to free GPU resources.
    """
    global _production_process, _saved_camera_manager
    
    if is_production_running():
        return False, "Production mode is already running"
    
    try:
        # IMPORTANT: Stop Streamlit cameras to free GPU resources
        if camera_manager:
            _saved_camera_manager = camera_manager
            print("[Production] Stopping Streamlit cameras to free GPU...")
            for cam_id, cam in camera_manager.cameras.items():
                try:
                    cam.stop()
                except:
                    pass
            # Give time for cameras to stop
            time.sleep(1.0)
        
        # Get the path to run_production.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        production_script = os.path.join(script_dir, "run_production.py")
        
        if not os.path.exists(production_script):
            return False, f"Production script not found: {production_script}"
        
        # Launch as subprocess (independent of Streamlit)
        _production_process = subprocess.Popen(
            [sys.executable, production_script],
            cwd=script_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        # Start recovery watcher - auto-restarts cameras when production exits
        _start_recovery_watcher(camera_manager)
        
        return True, f"Production mode launched (PID: {_production_process.pid}). Streamlit cameras stopped."
    
    except Exception as e:
        return False, f"Failed to launch production mode: {e}"


def stop_production_mode():
    """Stop the production mode subprocess and restart Streamlit cameras."""
    global _production_process, _saved_camera_manager
    
    if _production_process is None:
        return False, "Production mode is not running"
    
    try:
        _production_process.terminate()
        _production_process.wait(timeout=5)
        _production_process = None
    except:
        try:
            _production_process.kill()
            _production_process = None
        except:
            return False, "Failed to stop production mode"
    
    # Restart Streamlit cameras
    restart_msg = ""
    if _saved_camera_manager:
        try:
            print("[Production] Waiting for GPU to be released...")
            time.sleep(2.0)  # Give more time for GPU resources to be freed
            
            print("[Production] Restarting Streamlit cameras...")
            
            # Clear existing cameras first
            for cam_id, cam in list(_saved_camera_manager.cameras.items()):
                try:
                    cam.stop()
                except:
                    pass
            _saved_camera_manager.cameras.clear()
            
            # Reload from config file
            _saved_camera_manager.load_config_and_start()
            restart_msg = " Streamlit cameras restarted."
        except Exception as e:
            restart_msg = f" Warning: Could not restart cameras: {e}"
    
    return True, f"Production mode stopped.{restart_msg}"


def is_production_running():
    """Check if production mode is running."""
    global _production_process
    
    if _production_process is None:
        return False
    
    # Check if process is still alive
    poll = _production_process.poll()
    if poll is not None:
        # Process has ended
        _production_process = None
        return False
    
    return True

