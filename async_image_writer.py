"""
Async Image Writer Utility

Offloads cv2.imwrite calls to a background thread pool to prevent 
blocking the main application loop.
"""

from concurrent.futures import ThreadPoolExecutor
import cv2
import os
import time

class AsyncImageWriter:
    """
    Non-blocking image writer using ThreadPoolExecutor.
    Uses a singleton-like pattern via class methods.
    """
    
    # Shared executor for the application
    # max_workers=2 is sufficient for disk I/O without overwhelming the system
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AsyncImageWriter")
    
    @classmethod
    def save(cls, filepath, image, quality=95):
        """
        Queue an image for asynchronous saving.
        
        Args:
            filepath: Destination path
            image: OpenCV image (numpy array)
            quality: JPEG quality (0-100), default 95
        """
        if image is None:
            return
            
        # Submit to thread pool - fire and forget
        cls._executor.submit(cls._write, filepath, image.copy(), quality)
    
    @staticmethod
    def _write(filepath, image, quality):
        """
        Actual write operation running in background thread.
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Write image
            start_time = time.time()
            success = cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if not success:
                print(f"[AsyncImageWriter] ❌ Failed to write {filepath} (cv2 returned False)")
            
            # Optional: Uncomment for debugging performance
            # elapsed = (time.time() - start_time) * 1000
            # print(f"[AsyncImageWriter] ✓ Saved {os.path.basename(filepath)} in {elapsed:.1f}ms")
            
        except Exception as e:
            print(f"[AsyncImageWriter] ❌ Error saving {filepath}: {e}")

    @classmethod
    def shutdown(cls):
        """Cleanup resources on app exit."""
        cls._executor.shutdown(wait=True)
