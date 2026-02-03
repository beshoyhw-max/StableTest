# Code Review & Optimization Report

## Executive Summary
This report analyzes the software performance and architecture of the Enterprise Phone Monitor System. The primary goal is to address "low FPS" issues experienced with 4-5 cameras.

**Diagnosis:** The low FPS is caused by **software-level bottlenecks**, specifically CPU-bound tasks running in the Python main execution path. While the hardware can handle the load, the software logic forces the CPU to wait for heavy operations (MediaPipe, Optical Flow) sequentially, causing GIL (Global Interpreter Lock) contention.

**Key Findings:**
1. **Sleep Detector Overkill:** MediaPipe Face Mesh is running too frequently (every frame, every person), consuming massive CPU resources.
2. **Redundant Logic:** Optical Flow calculations and image copying are adding unnecessary overhead.
3. **Face Recognition Efficiency:** The face recognizer re-runs face detection on already detected crops, doubling the inference work.

---

## File-by-File Analysis

### 1. `detector.py` (Critical)
*   **Issue:** `process_frame` calls `self.sleep_detector.process_crop()` for *every* person not holding a phone.
*   **Impact:** If 5 cameras track 2 people each = 10 MediaPipe inferences per cycle. This is the #1 cause of FPS drop.
*   **Issue:** `cv2.calcOpticalFlowPyrLK` is used for tracking between inference frames.
*   **Impact:** Optical flow is CPU-intensive in Python. With 5 concurrent threads, this causes significant GIL contention.
*   **Issue:** `self.async_face_worker.enqueue_request` performs a `frame.copy()` for every face.
*   **Impact:** Frequent memory allocation/copying of large arrays slows down the loop.

### 2. `sleep_detector.py` (Critical)
*   **Issue:** `process_crop` converts BGR to RGB and runs `self.detector.detect` (MediaPipe) immediately if the crop size is sufficient.
*   **Optimization Opportunity:** "Lazy Evaluation". We already run YOLO Pose. If the person's posture is perfect (shoulders up, head up), we don't need to check their eyelids every single frame. We can skip MediaPipe 90% of the time.

### 3. `face_recognizer.py`
*   **Issue:** `detect_faces` calls `self.detector.setInputSize` and runs a Face Detection model.
*   **Context:** This runs *after* YOLO has already found the person and cut a crop. We are detecting a face within a face crop.
*   **Impact:** Double inference. While this runs in a background thread, it limits the throughput of the face recognition system, causing "lag" in identifying people.

### 4. `camera_manager.py`
*   **Status:** Generally well-structured.
*   **Note:** The architecture creates a separate `PhoneDetector` (and thus YOLO model) for each camera. This is good for isolation but heavy on VRAM. Since you confirmed hardware is not an issue, we will keep this but optimize the *loop* inside the thread.

---

## Recommended Optimizations (Action Plan)

### Phase 1: Software FPS Boost (Immediate Implementation)
These changes will directly address the "low FPS" issue.

1.  **Lazy Sleep Detection:**
    *   Modify `SleepDetector` to use YOLO Pose as a "Gatekeeper".
    *   Only run MediaPipe (Heavy) if:
        *   YOLO Pose confidence is low (ambiguous posture).
        *   OR Head Drop ratio is suspicious.
        *   OR We haven't checked eyes in >1 second (periodic refresh).
    *   **Expected Gain:** +50-100% FPS in crowded scenes.

2.  **Optimize Optical Flow:**
    *   Disable Optical Flow by default or restrict it. Since we are using YOLO tracking (`track` mode), we might not need manual optical flow interpolation for the skipped frames if the skip rate is low (e.g., 5).
    *   Simpler interpolation (just holding the box) is often sufficient for visualization.

3.  **Reduce Memory Copies:**
    *   Optimize how crops are extracted and passed to the Async Worker.

### Phase 2: Logic Improvements
1.  **Dynamic Skip Frames:**
    *   Instead of a fixed `skip_frames=5`, measure the processing time. If the system is lagging, automatically increase skip frames to maintain real-time monitoring.

2.  **Face Recognition throughput:**
    *   Standardize input size for the Face Recognizer to avoid re-initialization overhead.

---

## Next Step Features (Roadmap)
Based on your request for development next steps:

1.  **Database Integration (SQLite/PostgreSQL):**
    *   Current `pickle` database is fragile. Move to SQLite for robust storage of face embeddings and logs.
2.  **Web Dashboard / API:**
    *   Decouple the UI from the processing. Expose an API (FastAPI) so you can have a remote React/Vue dashboard instead of Streamlit.
3.  **Notifications:**
    *   Add Discord/Slack/Email webhooks for real-time alerts.
4.  **Heatmaps:**
    *   Use the tracking data to generate heatmaps of where students sit/stand most often.
