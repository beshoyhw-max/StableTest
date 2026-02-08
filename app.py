import streamlit as st
import cv2
import time
import os
import glob
from camera_manager import CameraManager
from PIL import Image
import numpy as np
import io

# Set Page Config
st.set_page_config(
    page_title="Enterprise Phone Detection",
    page_icon="📢",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .alert-box {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    div[data-testid="stImage"] img {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📢 Enterprise Phone Monitor System")

# --- Load Camera Manager (Singleton) ---
@st.cache_resource
def get_camera_manager():
    return CameraManager()

try:
    manager = get_camera_manager()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ Global Controls")

# Detection Sensitivity
conf_threshold = st.sidebar.slider(
    "Detection Sensitivity", 
    0.1, 1.0, 0.25,
    help="Lower = more sensitive (more detections, more false positives)"
)
manager.update_global_conf(conf_threshold)

st.sidebar.markdown("---")

# Time-Based Thresholds
st.sidebar.subheader("📱 Phone Detection")
phone_duration = st.sidebar.slider(
    "Alert after continuous use (seconds)",
    min_value=1,
    max_value=30,
    value=5,
    step=1,
    help="Person must be using phone continuously for this duration before alert"
)
manager.update_phone_duration(phone_duration)

st.sidebar.markdown("---")

st.sidebar.subheader("😴 Sleep Detection")
sleep_duration = st.sidebar.slider(
    "Alert after sleeping (seconds)",
    min_value=3,
    max_value=60,
    value=10,
    step=1,
    help="Person must be sleeping continuously for this duration before alert"
)
manager.update_sleep_duration(sleep_duration)

sleep_sensitivity = st.sidebar.slider(
    "Eye Sensitivity (EAR)",
    min_value=0.10,
    max_value=0.30,
    value=0.18,
    step=0.01,
    help="Lower = less sensitive (requires more closing). Decrease if glasses cause false alerts."
)
manager.update_sleep_sensitivity(sleep_sensitivity)

st.sidebar.markdown("---")

st.sidebar.subheader("📸 Evidence Cooldown")
cooldown_duration = st.sidebar.slider(
    "Cooldown between screenshots (seconds)",
    min_value=30,
    max_value=300,
    value=120,
    step=10,
    help="Minimum time between saving evidence for the same person"
)
manager.update_cooldown_duration(cooldown_duration)

st.sidebar.markdown("---")

st.sidebar.subheader("🚪 Absence Detection")
absence_threshold = st.sidebar.slider(
    "Alert after absence (seconds)",
    min_value=10,
    max_value=1200,
    value=300,
    step=10,
    help="Time before a recognized person is marked as 'left' if not seen in any camera"
)
manager.update_absence_threshold(absence_threshold)

st.sidebar.markdown("---")

st.sidebar.subheader("⚡ Performance")
skip_frames = st.sidebar.slider(
    "Skip frames (process every Nth frame)",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    help="Higher = faster processing, lower accuracy. Lower = slower, more accurate."
)
manager.update_skip_frames(skip_frames)

# Sync thresholds to file for Production mode
from config_sync import save_thresholds
save_thresholds(
    conf=conf_threshold,
    phone_dur=phone_duration,
    sleep_dur=sleep_duration,
    cooldown=cooldown_duration,
    absence=absence_threshold,
    skip_frames=skip_frames,
    sleep_sensitivity=sleep_sensitivity
)

st.sidebar.markdown("---")
st.sidebar.info(f"Active Cameras: {len(manager.get_active_cameras())}")

# --- Navigation ---
page_selection = st.radio(
    "Navigate", 
    ["🔴 Live Dashboard", "📸 Evidence Log", "👤 Face Recognition", "⚙️ Configuration"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# --- Page 1: Live Dashboard ---
if page_selection == "🔴 Live Dashboard":
    active_cams = manager.get_active_cameras()
    
    # Check for any active alerts
    alert_texting = []
    alert_sleeping = []
    
    for cam in active_cams.values():
        status = cam.get_status()
        if status == "texting":
            alert_texting.append(cam.camera_name)
        elif status == "sleeping":
            alert_sleeping.append(cam.camera_name)
    
    alert_placeholder = st.empty()
    if alert_texting:
        names_str = ", ".join(alert_texting)
        alert_placeholder.markdown(
            f'<div class="alert-box">⚠️ ALERT: PHONE DETECTED IN: {names_str}</div>', 
            unsafe_allow_html=True
        )
    elif alert_sleeping:
        names_str = ", ".join(alert_sleeping)
        alert_placeholder.markdown(
            f'<div class="alert-box" style="background-color: #6a0dad;">💤 ALERT: SLEEP DETECTED IN: {names_str}</div>', 
            unsafe_allow_html=True
        )
    else:
        alert_placeholder.empty()

    # Native Viewer Launch Button
    from native_viewer import launch_viewer, is_viewer_running, stop_viewer
    
    viewer_col1, viewer_col2 = st.columns([3, 1])
    with viewer_col1:
        st.info("💡 **Tip:** For smooth 60fps video, launch the native viewer below. Streamlit preview is limited to ~15fps.")
    with viewer_col2:
        if is_viewer_running():
            if st.button("🛑 Close Native Viewer", type="secondary"):
                stop_viewer()
                st.rerun()
        else:
            if st.button("🚀 Launch 60fps Viewer", type="primary"):
                launch_viewer(manager)
                st.success("Native viewer opened! Press 'Q' to close it.")
    
    # Production Mode Button (Multiprocessing)
    from production_viewer import launch_production_mode, stop_production_mode, is_production_running
    
    prod_col1, prod_col2 = st.columns([3, 1])
    with prod_col1:
        st.warning("⚡ **Production Mode:** Runs with multiprocessing (2x faster). Opens in new window. Press 'Q' to return.")
    with prod_col2:
        if is_production_running():
            if st.button("🛑 Stop Production", type="secondary", key="stop_prod"):
                success, msg = stop_production_mode()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()
        else:
            if st.button("⚡ Launch Production", type="primary", key="launch_prod"):
                success, msg = launch_production_mode(manager)
                if success:
                    st.success(f"🚀 {msg}")
                else:
                    st.error(msg)
    
    # Who Left Panel - Separate section for absent people
    absent_people = manager.get_absent_people()
    if absent_people:
        with st.expander(f"🚪 People Who Left ({len(absent_people)})", expanded=True):
            for person in absent_people:
                st.write(f"• **{person['name']}** - Last seen: {person['last_camera']} at {person['time']}")
            if st.button("🗑️ Clear All Absence Alerts"):
                manager.clear_absence_alerts()
                st.rerun()

    # Camera Grid - Individual Views with JPEG optimization
    if not active_cams:
        st.warning("No cameras configured. Go to Configuration tab.")
    else:
        cols = st.columns(min(2, len(active_cams)))  # Max 2 columns
        
        cam_containers = {}
        cam_ids = list(active_cams.keys())
        
        for idx, cam_id in enumerate(cam_ids):
            col_idx = idx % 2
            with cols[col_idx]:
                cam = active_cams[cam_id]
                st.subheader(f"🔹 {cam.camera_name}")
                frame_view = st.empty()
                status_text = st.empty()
                
                cam_containers[cam_id] = {
                    "frame": frame_view,
                    "status": status_text,
                    "thread": cam
                }

        # Auto-Refresh Loop
        run_monitor = st.checkbox("Start Live Monitor", value=True, key="run_live_monitor")
        
        if run_monitor:
            while True:
                # Check if cameras were restarted in background (after production mode exit)
                from production_viewer import check_rerun_signal
                if check_rerun_signal():
                    st.rerun()
                
                loop_texting = []
                loop_sleeping = []
                
                for cam_id, container in list(cam_containers.items()):
                    thread = container["thread"]
                    frame = thread.get_frame()
                    status = thread.get_status()
                    
                    if status == "texting":
                        loop_texting.append(thread.camera_name)
                    elif status == "sleeping":
                        loop_sleeping.append(thread.camera_name)
                    
                    # Update Status
                    if status == "texting":
                        container["status"].markdown(":red[**📱 PHONE DETECTED**]")
                    elif status == "sleeping":
                        container["status"].markdown(":violet[**💤 SLEEP DETECTED**]")
                    elif status == "safe":
                        container["status"].markdown(":green[**✓ SAFE**]")
                    elif status == "disconnected":
                        container["status"].markdown(":orange[**⏳ CONNECTING...**]")
                    else:
                        container["status"].write(status)
                        
                    # Update Frame - Use raw frame with detection overlay for smooth display
                    raw_frame = thread.get_raw_frame()
                    if raw_frame is not None:
                        # Draw detection boxes on raw frame
                        frame = thread.draw_overlay_on_frame(raw_frame)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        container["frame"].image(frame_rgb, width="stretch")
                    else:
                        container["frame"].info("No Signal")
                
                # Update Global Alert
                if loop_texting:
                    names_str = ", ".join(loop_texting)
                    alert_placeholder.markdown(
                        f'<div class="alert-box">⚠️ ALERT: PHONE DETECTED IN: {names_str}</div>', 
                        unsafe_allow_html=True
                    )
                elif loop_sleeping:
                    names_str = ", ".join(loop_sleeping)
                    alert_placeholder.markdown(
                        f'<div class="alert-box" style="background-color: #6a0dad;">💤 ALERT: SLEEP DETECTED IN: {names_str}</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    alert_placeholder.empty()
                
                # JPEG encoding allows faster refresh
                time.sleep(0.05)  # ~20fps with JPEG encoding

# --- Page 2: Evidence Log ---
elif page_selection == "📸 Evidence Log":
    st.subheader("Infraction History")
    
    if st.button("🔄 Refresh Gallery"):
        pass
        
    image_files = glob.glob("detections/*.jpg")
    image_files.sort(key=os.path.getmtime, reverse=True)
    
    if not image_files:
        st.info("No evidence collected yet.")
    else:
        cols = st.columns(4)
        for idx, img_path in enumerate(image_files):
            with cols[idx % 4]:
                image = Image.open(img_path)
                st.image(image, width='stretch')
                st.caption(os.path.basename(img_path))

# --- Page 3: Face Recognition ---
elif page_selection == "👤 Face Recognition":
    st.header("Face Recognition Management")
    
    # Try to load face recognizer
    try:
        from face_recognizer import FaceRecognizer
        recognizer = FaceRecognizer()
        
        # Display stats
        stats = recognizer.get_database_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Registered People", stats['total_faces'])
        col2.metric("Recognition Threshold", f"{stats['threshold']:.3f}")
        col3.metric("Database Size", f"{stats['database_size_kb']:.1f} KB")
        
        st.markdown("---")
        
        # Tabs for different operations
        tab1, tab2, tab3, tab4 = st.tabs(["📋 View Registered", "➕ Register New", "📁 Bulk Import", "⚙️ Settings"])
        
        with tab1:
            st.subheader("Registered People")
            people = recognizer.list_known_people()
            
            if not people:
                st.info("No people registered yet. Use the 'Register New' tab to add people.")
            else:
                for person in people:
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"**{person}**")
                    if col2.button("Remove", key=f"remove_{person}"):
                        success, msg = recognizer.remove_person(person)
                        if success:
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(msg)
        
        with tab2:
            st.subheader("Register New Person")
            
            # Upload image
            uploaded_file = st.file_uploader("Upload a clear face photo", type=['jpg', 'jpeg', 'png'])
            person_name = st.text_input("Person Name/ID", placeholder="e.g., John_Doe or Student_001")
            
            if st.button("Register"):
                if uploaded_file is None:
                    st.error("Please upload an image")
                elif not person_name:
                    st.error("Please enter a person name")
                else:
                    # Convert uploaded file to OpenCV format
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Show preview
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=300)
                    
                    # Register
                    with st.spinner("Processing..."):
                        success, msg = recognizer.register_face(frame, person_name)
                    
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
        
        with tab3:
            st.subheader("Bulk Import from Folder")
            st.info("""
            **Folder Structure Required:**
            ```
            registered_faces/
            ├── John_Doe/
            │   └── photo.jpg
            ├── Jane_Smith/
            │   └── photo.jpg
            └── Student_001/
                └── photo.jpg
            ```
            Each person folder should contain at least one clear face photo.
            """)
            
            folder_path = st.text_input("Folder Path", value="registered_faces")
            
            if st.button("Import All"):
                if not os.path.exists(folder_path):
                    st.error(f"Folder not found: {folder_path}")
                else:
                    with st.spinner("Importing faces..."):
                        successful, failed, messages = recognizer.batch_register_from_folder(folder_path)
                    
                    st.success(f"Import complete: {successful} successful, {failed} failed")
                    
                    with st.expander("View Details"):
                        for msg in messages:
                            st.write(msg)
        with tab4:
            st.subheader("Recognition Settings")
            
            # Load current threshold from shared config
            from face_recognizer import FaceRecognizer
            current_threshold = FaceRecognizer.get_shared_threshold()
            
            new_threshold = st.slider(
                "Recognition Threshold",
                min_value=0.20,
                max_value=0.60,
                value=current_threshold,
                step=0.01,
                help="Lower = more lenient (better for distant faces, more false positives)"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Update Threshold", width='stretch'):
                    FaceRecognizer.set_shared_threshold(new_threshold)
                    st.success(f"Threshold saved: {new_threshold:.3f}")
                    st.info("⚠️ Go to Configuration → Refresh All to apply to cameras")
            
            with col2:
                if st.button("🔄 Apply Now", width='stretch'):
                    FaceRecognizer.set_shared_threshold(new_threshold)
                    
                    # Restart all cameras
                    import json
                    with open(manager.config_file, 'r') as f:
                        configs = json.load(f)
                    
                    for cam_id in list(manager.cameras.keys()):
                        manager.cameras[cam_id].stop()
                    manager.cameras.clear()
                    
                    for config in configs:
                        manager.add_camera_thread(config)
                    
                    st.success("✓ Applied and cameras restarted!")
            
            st.markdown("---")
            
            # Threshold recommendations
            st.info("""
            **Threshold Guide:**
            - **0.45-0.50**: Very strict (only exact matches)
            - **0.35-0.40**: Balanced (default)
            - **0.25-0.30**: Lenient (better for distant/angled faces)
            - **Below 0.25**: Too loose (many false matches)
            """)
            
            st.markdown("---")
            st.subheader("Face Recognition Status")
            
            face_rec_enabled = st.checkbox(
                "Enable Face Recognition for All Cameras",
                value=manager.global_face_recognition,
                help="Restart cameras after changing this"
            )
            
            if face_rec_enabled != manager.global_face_recognition:
                manager.enable_face_recognition(face_rec_enabled)
                st.warning("⚠️ Setting changed. Click 'Refresh All' in Configuration to apply.")
    except FileNotFoundError as e:
        st.error(f"Face recognition models not found!")
        st.info("""
        **Setup Required:**
        
        1. Create a `models` folder
        2. Download the required models:
        
        ```bash
        mkdir -p models
        
        # Face Detector
        wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -O models/face_detection_yunet_2023mar.onnx
        
        # Face Recognizer
        wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx -O models/face_recognition_sface_2021dec.onnx
        ```
        """)
    except Exception as e:
        st.error(f"Error initializing face recognition: {e}")

# --- Page 4: Configuration ---
elif page_selection == "⚙️ Configuration":
    st.header("Camera Management")
    
    # Top action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Active Cameras")
    
    with col2:
        if st.button("🔄 Refresh All", help="Restart all cameras", width='stretch'):
            with st.spinner("Restarting..."):
                import json
                
                # Save current configs
                with open(manager.config_file, 'r') as f:
                    configs = json.load(f)
                
                # Stop all
                for cam_id in list(manager.cameras.keys()):
                    manager.cameras[cam_id].stop()
                    time.sleep(0.1)
                
                manager.cameras.clear()
                
                # Restart all
                for config in configs:
                    manager.add_camera_thread(config)
                
                st.success("✓ Cameras restarted!")
                time.sleep(1)
                st.rerun()
    
    with col3:
        if st.button("🗑️ Remove All", width='stretch'):
            if st.session_state.get('confirm_remove', False):
                for cam_id in list(manager.cameras.keys()):
                    manager.remove_camera(cam_id)
                st.session_state['confirm_remove'] = False
                st.rerun()
            else:
                st.session_state['confirm_remove'] = True
                st.warning("Click again to confirm")
    
    # Camera list
    active_cams = manager.get_active_cameras()
    
    if not active_cams:
        st.info("No cameras configured yet.")
    else:
        for cam_id, cam in list(active_cams.items()):
            col1, col2, col3 = st.columns([1, 4, 1])
            col1.write(f"**ID {cam_id}**")
            col2.write(f"{cam.camera_name} ({cam.source})")
            if col3.button("🗑️", key=f"del_{cam_id}"):
                manager.remove_camera(cam_id)
                st.rerun()
            
    st.markdown("---")
    st.subheader("Add New Camera")
    
    with st.form("add_cam_form"):
        new_name = st.text_input("Camera Name", placeholder="e.g., Classroom A")
        new_source = st.text_input("Source (0 for webcam, rtsp://... for IP cam)", value="0")
        
        if st.form_submit_button("➕ Add Camera"):
            if new_name:
                manager.add_camera(new_name, new_source)
                st.success(f"Added {new_name}")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Name required")

    st.markdown("---")
    st.info("💡 After changing Face Recognition settings, use 'Refresh All' to apply changes")
    st.info("💡 Use '0', '1' for local webcams. Use 'rtsp://...' for IP cameras.")
