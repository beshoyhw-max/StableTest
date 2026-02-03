
import threading
import queue
import time
import cv2

class AsyncFaceWorker:
    """
    Background worker for face recognition.
    Decouples the Face Recognition inference (heavy) from the Object Detection loop (fast).
    """

    def __init__(self, face_recognizer):
        self.face_recognizer = face_recognizer
        self.request_queue = queue.Queue(maxsize=10) # Drop older requests if busy
        self.result_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.latest_requests = {} # map track_id -> timestamp to avoid duplicate queueing

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        print("  → AsyncFaceWorker started")

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def enqueue_request(self, frame, bbox, track_id, timestamp):
        """
        Submit a face recognition request.
        Request is ignored if queue is full or if we already have a pending request for this ID (debounce).
        """
        if not self.running:
            return

        # Simple debounce: Don't queue if we just queued this ID recently (< 0.5s)
        # Note: Caller usually handles "check interval", this is a safety net for the queue
        if self.request_queue.full():
            return
            
        # Create a copy of the ROI to ensure thread safety with the image data directly
        # Instead of passing the whole frame, we crop here to save memory in queue
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Add context padding
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        
        ctx1 = max(0, x1 - pad_x)
        cty1 = max(0, y1 - pad_y)
        ctx2 = min(w, x2 + pad_x)
        cty2 = min(h, y2 + pad_y)
        
        # Crop logic
        if ctx2 > ctx1 and cty2 > cty1:
            frame_roi = frame[cty1:cty2, ctx1:ctx2].copy()
            # Adjust bbox to be relative to the roi
            adj_bbox = (x1 - ctx1, y1 - cty1, x2 - ctx1, y2 - cty1)
            
            self.request_queue.put({
                'frame': frame_roi, 
                'bbox': adj_bbox, 
                'track_id': track_id,
                'request_ts': timestamp
            })

    def get_results(self):
        """Yields all available results from the queue."""
        while not self.result_queue.empty():
            try:
                yield self.result_queue.get_nowait()
            except queue.Empty:
                break

    def _process_queue(self):
        while self.running:
            try:
                # Wait for request
                req = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            try:
                # Perform recognition
                # Note: 'frame' here is the ROI crop we made earlier
                name, conf = self.face_recognizer.recognize_face(req['frame'], req['bbox'])
                
                # Push result
                self.result_queue.put({
                    'track_id': req['track_id'],
                    'name': name,
                    'conf': conf,
                    'ts': time.time()
                })
                
            except Exception as e:
                print(f"AsyncFaceWorker Error: {e}")
            finally:
                self.request_queue.task_done()
