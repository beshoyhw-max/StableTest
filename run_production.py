"""
Production Runner - Multiprocessing Enabled

This script runs the camera system with FULL multiprocessing support.
Use this for production deployment (without Streamlit UI).

For development/UI, use: streamlit run app.py (threading mode)
For production, use: python run_production.py (multiprocessing mode)

Controls:
- Q: Quit
- D: Detection view
- R: Raw view
"""

import multiprocessing as mp
import sys
import os

# CRITICAL: Windows multiprocessing guard
if __name__ == '__main__':
    # Freeze support for Windows executables
    mp.freeze_support()
    
    # Set spawn method explicitly
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)
    
    # Now safe to import and run
    from camera_manager import CameraManager
    from native_viewer import NativeVideoViewer
    from config_sync import load_thresholds, get_file_mtime
    import time
    import threading
    
    print("=" * 70)
    print("PRODUCTION MODE - MULTIPROCESSING ENABLED")
    print("Each camera detector runs in its OWN PROCESS (bypasses GIL)")
    print("=" * 70)
    print()
    print("Controls:")
    print("  Q/ESC: Quit")
    print("  D: Detection view (with boxes)")
    print("  R: Raw view")
    print()
    print("[ConfigSync] Watching for Streamlit config changes...")
    print()
    
    # Create shared cooldowns for cross-camera deduplication
    # Manager.dict() is process-safe (no additional lock needed)
    mp_manager = mp.Manager()
    mp_shared_cooldowns = mp_manager.dict()
    print("[SharedCooldowns] Created process-safe shared cooldowns dict")
    
    # Create manager with multiprocessing ENABLED
    manager = CameraManager(use_multiprocessing=True, mp_shared_cooldowns=mp_shared_cooldowns)
    
    # Apply initial thresholds from Streamlit config
    initial_config = load_thresholds()
    manager.update_global_conf(initial_config.get("conf_threshold", 0.25))
    manager.update_phone_duration(initial_config.get("phone_duration", 5.0))
    manager.update_sleep_duration(initial_config.get("sleep_duration", 10.0))
    manager.update_cooldown_duration(initial_config.get("cooldown_duration", 120.0))
    manager.update_absence_threshold(initial_config.get("absence_threshold", 300.0))
    manager.update_skip_frames(initial_config.get("skip_frames", 5))
    manager.update_sleep_sensitivity(initial_config.get("sleep_sensitivity", 0.18))
    print(f"[ConfigSync] Loaded initial thresholds from Streamlit config")
    
    # Config watcher flag
    _config_watcher_running = True
    
    def config_watcher():
        """Watch for config file changes and apply to manager."""
        global _config_watcher_running
        last_mtime = get_file_mtime()
        
        while _config_watcher_running:
            time.sleep(1.0)  # Check every second
            
            current_mtime = get_file_mtime()
            if current_mtime > last_mtime:
                last_mtime = current_mtime
                
                # Reload and apply thresholds
                config = load_thresholds()
                manager.update_global_conf(config.get("conf_threshold", 0.25))
                manager.update_phone_duration(config.get("phone_duration", 5.0))
                manager.update_sleep_duration(config.get("sleep_duration", 10.0))
                manager.update_cooldown_duration(config.get("cooldown_duration", 120.0))
                manager.update_absence_threshold(config.get("absence_threshold", 300.0))
                manager.update_skip_frames(config.get("skip_frames", 5))
                manager.update_sleep_sensitivity(config.get("sleep_sensitivity", 0.18))
                
                print(f"[ConfigSync] ✓ Config updated from Streamlit!")
    
    # Start config watcher thread
    watcher_thread = threading.Thread(target=config_watcher, daemon=True)
    watcher_thread.start()
    
    # Wait for cameras to initialize
    print("\nWaiting for cameras to initialize...")
    time.sleep(3)
    
    # Check status
    active = manager.get_active_cameras()
    print(f"\nActive cameras: {len(active)}")
    for cam_id, cam in active.items():
        print(f"  - {cam.camera_name}: {cam.get_status()}")
    
    # Launch native viewer (OpenCV window)
    print("\nLaunching native viewer...")
    viewer = NativeVideoViewer(manager)
    viewer.start()
    
    # Stop signal file (graceful IPC)
    STOP_SIGNAL_FILE = ".stop_production"
    
    # Clean up stale signal file on startup
    if os.path.exists(STOP_SIGNAL_FILE):
        try:
            os.remove(STOP_SIGNAL_FILE)
        except:
            pass
            
    # Wait for viewer to close OR stop signal
    try:
        while viewer.is_running():
            if os.path.exists(STOP_SIGNAL_FILE):
                print(f"[Production] Stop signal received via {STOP_SIGNAL_FILE}")
                try:
                    os.remove(STOP_SIGNAL_FILE)
                except:
                    pass
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    # Cleanup
    _config_watcher_running = False
    viewer.stop()
    for cam in manager.cameras.values():
        try:
            cam.stop()
        except Exception as e:
            print(f"[Production] Error stopping camera: {e}")
    
    print("Production runner exited.")

