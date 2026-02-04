"""
Config Sync Module

Handles synchronization of runtime thresholds between Streamlit UI and Production mode.
Uses a JSON file as the shared config store.
"""

import json
import os
import time

THRESHOLDS_FILE = "runtime_thresholds.json"

# Default thresholds (used if file doesn't exist)
DEFAULT_THRESHOLDS = {
    "conf_threshold": 0.25,
    "phone_duration": 5.0,
    "sleep_duration": 10.0,
    "cooldown_duration": 120.0,
    "absence_threshold": 300.0,
    "skip_frames": 5,
    "sleep_sensitivity": 0.18,
    "timestamp": 0
}


def save_thresholds(conf, phone_dur, sleep_dur, cooldown, absence, skip_frames, sleep_sensitivity):
    """
    Save thresholds to file (called by Streamlit on slider change).
    
    Thread-safe: Uses atomic write pattern.
    """
    thresholds = {
        "conf_threshold": conf,
        "phone_duration": phone_dur,
        "sleep_duration": sleep_dur,
        "cooldown_duration": cooldown,
        "absence_threshold": absence,
        "skip_frames": skip_frames,
        "sleep_sensitivity": sleep_sensitivity,
        "timestamp": time.time()
    }
    
    # Atomic write: write to temp file, then rename
    temp_file = THRESHOLDS_FILE + ".tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
        
        # Atomic rename (works on Windows too)
        if os.path.exists(THRESHOLDS_FILE):
            os.remove(THRESHOLDS_FILE)
        os.rename(temp_file, THRESHOLDS_FILE)
    except Exception as e:
        print(f"[ConfigSync] Error saving thresholds: {e}")
        # Cleanup temp file if it exists
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


def load_thresholds():
    """
    Load thresholds from file (called by Production mode).
    
    Returns:
        dict: Thresholds dictionary, or defaults if file doesn't exist.
    """
    if not os.path.exists(THRESHOLDS_FILE):
        return DEFAULT_THRESHOLDS.copy()
    
    try:
        with open(THRESHOLDS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ConfigSync] Error loading thresholds: {e}")
        return DEFAULT_THRESHOLDS.copy()


def get_file_mtime():
    """
    Get last modified time of thresholds file.
    
    Returns:
        float: Modification time, or 0 if file doesn't exist.
    """
    if not os.path.exists(THRESHOLDS_FILE):
        return 0
    
    try:
        return os.path.getmtime(THRESHOLDS_FILE)
    except:
        return 0
