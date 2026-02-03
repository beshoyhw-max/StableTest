import cv2
import numpy as np
import pickle
import os
from pathlib import Path

class FaceRecognizer:
    @staticmethod
    def get_shared_threshold():
        """Get threshold from shared config file."""
        config_file = 'face_recognition_config.json'
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('threshold', 0.363)
        return 0.363
    
    @staticmethod
    def set_shared_threshold(value):
        """Set threshold in shared config file."""
        import json
        config_file = 'face_recognition_config.json'
        config = {'threshold': value}
        with open(config_file, 'w') as f:
            json.dump(config, f)
    def __init__(self, 
                 detector_model='models/face_detection_yunet_2023mar.onnx',
                 recognizer_model='models/face_recognition_sface_2021dec.onnx',
                 database_path='face_database.pkl'):
        
        self.database_path = database_path
        
        if not os.path.exists(detector_model):
            raise FileNotFoundError(
                f"Face detector model not found: {detector_model}\n"
                "Download from: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )
        
        if not os.path.exists(recognizer_model):
            raise FileNotFoundError(
                f"Face recognizer model not found: {recognizer_model}\n"
                "Download from: https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            )
        
        # Initialize detector
        self.detector = cv2.FaceDetectorYN.create(
            detector_model,
            "",
            (320, 320),
            score_threshold=0.5,
            nms_threshold=0.3
        )
        
        # Initialize recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_model, "")
        
        # Load database
        self.known_faces = self.load_database()
        
        # Recognition threshold
        self.threshold = self.get_shared_threshold()
        
        # CRITICAL: Cache for last input size to avoid unnecessary setInputSize calls
        self.last_input_size = None
        
        print(f"Face Recognizer initialized")
        print(f"  → Known faces: {len(self.known_faces['names'])}")
        print(f"  → Recognition threshold: {self.threshold}")
        
    def load_database(self):
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    db = pickle.load(f)
                print(f"Loaded face database: {len(db['names'])} faces")
                return db
            except Exception as e:
                print(f"Error loading database: {e}")
                return {'names': [], 'embeddings': []}
        return {'names': [], 'embeddings': []}
    
    def save_database(self):
        try:
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.known_faces, f)
            print(f"Saved face database: {len(self.known_faces['names'])} faces")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def detect_faces(self, frame):
        """
        FIXED: Only update input size when it actually changes.
        """
        h, w = frame.shape[:2]
        current_size = (w, h)
        
        # Only call setInputSize if size actually changed
        if current_size != self.last_input_size:
            self.detector.setInputSize(current_size)
            self.last_input_size = current_size
        
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else []
    
    def register_face(self, frame, name, bbox=None):
        """Register a new person's face."""
        
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding
            h, w = frame.shape[:2]
            pad = 30
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            search_region = frame[y1:y2, x1:x2]
            
            # Detect face in region
            faces = self.detect_faces(search_region)
            
            if len(faces) == 0:
                print(f"  → No face in bbox region, trying full frame...")
                faces = self.detect_faces(frame)
                if len(faces) == 0:
                    return False, f"No face detected for {name}"
                search_region = frame
            
        else:
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                return False, f"No face detected for {name}"
            
            search_region = frame
        
        if len(faces) > 1:
            print(f"  ⚠ Multiple faces detected ({len(faces)}), using largest")
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        
        face = faces[0]
        
        x, y, w_box, h_box = face[:4].astype(int)
        score = face[14]
        
        print(f"  → Face detected: {w_box}x{h_box}px, confidence={score:.3f}")
        
        try:
            aligned_face = self.recognizer.alignCrop(search_region, face)
            
            if aligned_face is None or aligned_face.size == 0:
                return False, "Face alignment failed"
            
            embedding = self.recognizer.feature(aligned_face)
            
            if name in self.known_faces['names']:
                idx = self.known_faces['names'].index(name)
                self.known_faces['embeddings'][idx] = embedding
                message = f"Updated face for {name}"
            else:
                self.known_faces['names'].append(name)
                self.known_faces['embeddings'].append(embedding)
                message = f"Registered new face: {name}"
            
            self.save_database()
            return True, message
            
        except Exception as e:
            return False, f"Error processing face: {str(e)}"
    
    def recognize_face(self, frame, bbox):
        """
        IMPROVED: Better recognition for distant people.
        """
        if len(self.known_faces['names']) == 0:
            return "Unknown", 0.0
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Validate bbox
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0 or roi.shape[0] < 30 or roi.shape[1] < 30:
            return "Unknown", 0.0
        
        # IMPROVED: Multiple detection attempts with different thresholds
        faces = []
        
        # Attempt 1: Standard threshold (0.5)
        faces = self.detect_faces(roi)
        
        # Attempt 2: Lower threshold for distant faces (0.3)
        if len(faces) == 0:
            old_threshold = self.detector.getScoreThreshold()
            self.detector.setScoreThreshold(0.3)
            faces = self.detect_faces(roi)
            self.detector.setScoreThreshold(old_threshold)
        
        # Attempt 3: Try on slightly larger context if roi is small
        if len(faces) == 0 and (x2 - x1) < 150:
            # Expand search area
            ctx1 = max(0, x1 - 30)
            cty1 = max(0, y1 - 30)
            ctx2 = min(w, x2 + 30)
            cty2 = min(h, y2 + 30)
            
            context_roi = frame[cty1:cty2, ctx1:ctx2]
            
            old_threshold = self.detector.getScoreThreshold()
            self.detector.setScoreThreshold(0.3)
            faces_ctx = self.detect_faces(context_roi)
            self.detector.setScoreThreshold(old_threshold)
            
            # Adjust face coordinates back to original roi space
            if len(faces_ctx) > 0:
                for face_ctx in faces_ctx:
                    adjusted_face = face_ctx.copy()
                    adjusted_face[0] = face_ctx[0] - (ctx1 - x1)
                    adjusted_face[1] = face_ctx[1] - (cty1 - y1)
                    faces.append(adjusted_face)
        
        if len(faces) == 0:
            return "Unknown", 0.0
        
        # Use largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        
        face = faces[0]
        
        # Accept smaller faces (was 30x30, now 25x25)
        face_w, face_h = int(face[2]), int(face[3])
        if face_w < 25 or face_h < 25:
            return "Unknown", 0.0
        
        try:
            # Align face
            aligned_face = self.recognizer.alignCrop(roi, face)
            
            if aligned_face is None or aligned_face.size == 0:
                return "Unknown", 0.0
            
            # Generate embedding
            query_embedding = self.recognizer.feature(aligned_face)
            
            # Compare with known faces
            best_match = "Unknown"
            best_score = 0.0
            
            for name, known_embedding in zip(
                self.known_faces['names'], 
                self.known_faces['embeddings']
            ):
                score = self.recognizer.match(
                    query_embedding, 
                    known_embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                
                if score >= self.threshold and score > best_score:
                    best_score = score
                    best_match = name
            
            return best_match, best_score
            
        except Exception as e:
            return "Unknown", 0.0

    def batch_register_from_folder(self, folder_path):
        """Register all faces from folder structure."""
        if not os.path.exists(folder_path):
            return 0, 0, [f"Folder not found: {folder_path}"]
        
        successful = 0
        failed = 0
        messages = []
        
        for person_name in sorted(os.listdir(folder_path)):
            person_dir = os.path.join(folder_path, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            registered = False
            for img_file in sorted(os.listdir(person_dir)):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(person_dir, img_file)
                
                try:
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    
                    success, message = self.register_face(frame, person_name)
                    messages.append(f"{person_name}: {message}")
                    
                    if success:
                        successful += 1
                        registered = True
                        break
                    
                except Exception as e:
                    messages.append(f"{person_name}/{img_file}: Error - {str(e)}")
            
            if not registered:
                failed += 1
                messages.append(f"{person_name}: No valid face found")
        
        return successful, failed, messages
    
    def remove_person(self, name):
        """Remove a person from the database."""
        if name not in self.known_faces['names']:
            return False, f"Person not found: {name}"
        
        idx = self.known_faces['names'].index(name)
        del self.known_faces['names'][idx]
        del self.known_faces['embeddings'][idx]
        
        self.save_database()
        return True, f"Removed {name} from database"
    
    def list_known_people(self):
        """Get list of all registered people."""
        return self.known_faces['names'].copy()
    
    def get_database_stats(self):
        """Get statistics about the face database."""
        return {
            'total_faces': len(self.known_faces['names']),
            'threshold': self.threshold,
            'database_path': self.database_path,
            'database_size_kb': os.path.getsize(self.database_path) / 1024 if os.path.exists(self.database_path) else 0
        }
    
    def update_threshold(self, new_threshold):
        """Update threshold in shared config (affects all cameras)."""
        self.threshold = max(0.0, min(1.0, new_threshold))
        self.set_shared_threshold(self.threshold)
        print(f"Recognition threshold updated to {self.threshold} (saved to shared config)")
