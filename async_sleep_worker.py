"""
Async Sleep Detection Worker

Moves heavy MediaPipe face landmark detection off the main detection loop.
Uses a cache-based pattern: main loop reads cached results, background worker updates them.
"""

import threading
import queue
import time


class AsyncSleepWorker:
    """
    Background worker for sleep/drowsiness detection.
    Decouples MediaPipe inference (CPU-heavy) from the Object Detection loop (GPU-fast).
    
    Pattern:
    - Main loop: enqueue(crop, id_key, ...) + get_sleep_status(id_key) 
    - Worker: process_crop() in background, update cache
    """

    def __init__(self, sleep_detector):
        self.sleep_detector = sleep_detector
        self.request_queue = queue.Queue(maxsize=30)
        
        # Result cache: {id_key: (status, details, timestamp)}
        self.result_cache = {}
        self.cache_lock = threading.RLock()
        
        # Track last enqueue time per key (for freshness check)
        self.last_enqueue_time = {}
        
        # Debounce: track pending requests to avoid duplicate processing
        self.pending_keys = set()
        self.pending_lock = threading.Lock()
        
        self.running = False
        self.worker_thread = None

    def start(self):
        """Start the background worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        print("  → AsyncSleepWorker started")

    def stop(self):
        """Stop the background worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

    def enqueue(self, crop, id_key, keypoints=None, crop_origin=(0, 0), sensitivity=0.18):
        """
        Submit a sleep detection request (non-blocking).
        
        Request is dropped if:
        - Queue is full after 10ms wait (backpressure with grace period)
        - Same id_key is already pending (debounce)
        """
        if not self.running:
            return False
            
        # Debounce: skip if already pending
        with self.pending_lock:
            if id_key in self.pending_keys:
                return False
            self.pending_keys.add(id_key)
        
        # Track enqueue time for freshness check
        self.last_enqueue_time[id_key] = time.time()
        
        # Create copies for thread safety
        try:
            # Use timeout instead of block=False for smoother jitter handling
            self.request_queue.put({
                'crop': crop.copy(),
                'id_key': id_key,
                'keypoints': keypoints.copy() if keypoints is not None else None,
                'crop_origin': crop_origin,
                'sensitivity': sensitivity,
                'enqueue_time': time.time()
            }, timeout=0.01)  # Wait 10ms before dropping
            return True
        except queue.Full:
            with self.pending_lock:
                self.pending_keys.discard(id_key)
            return False
        except Exception:
            with self.pending_lock:
                self.pending_keys.discard(id_key)
            return False

    def get_sleep_status(self, id_key, max_age=5.0):
        """
        Get cached sleep status for a person (instant, non-blocking).
        
        Uses cached value while new request is processing. Only returns None
        if cache is too old AND no pending request.
        """
        # Check pending status first (outside cache_lock to avoid deadlock)
        with self.pending_lock:
            is_pending = id_key in self.pending_keys
        
        with self.cache_lock:
            if id_key in self.result_cache:
                status, details, cache_ts = self.result_cache[id_key]
                
                # Check 1: Cache not too old
                if time.time() - cache_ts > max_age:
                    return None, None
                
                # Check 2: If request is pending, use cached value (don't return None)
                # Only invalidate if cache is stale AND request completed
                last_enqueue = self.last_enqueue_time.get(id_key, 0)
                if cache_ts < last_enqueue and not is_pending:
                    return None, None
                
                return status, details
        return None, None

    def get_all_statuses(self):
        """Get all cached statuses (for debugging)."""
        with self.cache_lock:
            return dict(self.result_cache)

    def clear_stale_cache(self, max_age=30.0):
        """Remove stale entries from cache."""
        now = time.time()
        with self.cache_lock:
            stale_keys = [k for k, (_, _, ts) in self.result_cache.items() if now - ts > max_age]
            for k in stale_keys:
                del self.result_cache[k]

    def _process_queue(self):
        """Background worker loop."""
        last_cleanup_time = time.time()
        request_count = 0
        
        while self.running:
            try:
                req = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                # Perform periodic cleanup even when idle
                if time.time() - last_cleanup_time > 60.0:
                    self._cleanup_stale_data()
                    last_cleanup_time = time.time()
                continue
            
            id_key = req['id_key']
            request_count += 1
            
            # === AUTOMATIC CLEANUP (prevents OOM) ===
            # Run every 100 requests to avoid overhead
            if request_count >= 100:
                request_count = 0
                self._cleanup_stale_data()
                last_cleanup_time = time.time()
            
            try:
                # Run the actual sleep detection (MediaPipe + EAR)
                status, details = self.sleep_detector.process_crop(
                    req['crop'],
                    id_key=id_key,
                    keypoints=req['keypoints'],
                    crop_origin=req['crop_origin'],
                    sensitivity=req['sensitivity']
                )
                
                # Update cache
                with self.cache_lock:
                    self.result_cache[id_key] = (status, details, time.time())
                    
            except Exception as e:
                print(f"AsyncSleepWorker Error [{id_key}]: {e}")
            finally:
                # Clear pending flag
                with self.pending_lock:
                    self.pending_keys.discard(id_key)
                self.request_queue.task_done()

    def _cleanup_stale_data(self):
        """Remove stale entries from all caches."""
        now = time.time()
        
        # Clean result_cache (entries older than 30s)
        with self.cache_lock:
            stale_keys = [k for k, (_, _, ts) in self.result_cache.items() if now - ts > 30.0]
            for k in stale_keys:
                del self.result_cache[k]
        
        # Clean last_enqueue_time (entries older than 30s)
        stale_enqueue = [k for k, ts in self.last_enqueue_time.items() if now - ts > 30.0]
        for k in stale_enqueue:
            del self.last_enqueue_time[k]
        
        if stale_keys or stale_enqueue:
            # print(f"[AsyncSleepWorker] Cleaned {len(stale_keys)} cache + {len(stale_enqueue)} enqueue entries")
            pass
