#!/usr/bin/env python3
"""
Basic Face Recognition System
Uses only OpenCV built-in face detection - No external ML packages required
"""

import cv2
import numpy as np
import sqlite3
import os
import json
import time
from datetime import datetime
from pathlib import Path
import hashlib
from typing import List, Dict, Tuple, Optional

class BasicFaceDetector:
    """Basic face detector using OpenCV's Haar cascades"""
    
    def __init__(self):
        # Load pre-trained face detection model
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Warning: Could not load face cascade. Using basic detection.")
            self.face_cascade = None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame using Haar cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.face_cascade is None:
            # Fallback: simple edge detection
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            faces = []
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    faces.append((x, y, w, h))
            return faces
        
        # Use Haar cascade
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]

class BasicFaceRecognizer:
    """Basic face recognizer using image hashing"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.known_faces = {}  # face_hash -> person_id
    
    def get_image_hash(self, face_img: np.ndarray) -> str:
        """Generate a simple hash for face image"""
        # Resize to standard size
        face_img = cv2.resize(face_img, (64, 64))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Simple hash based on average pixel values
        avg_pixel = np.mean(gray)
        hash_value = hashlib.md5(gray.tobytes()).hexdigest()[:16]
        
        return f"{hash_value}_{int(avg_pixel)}"
    
    def calculate_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes"""
        if hash1 == hash2:
            return 1.0
        
        # Simple similarity based on hash prefix
        common_prefix = 0
        for i in range(min(len(hash1), len(hash2))):
            if hash1[i] == hash2[i]:
                common_prefix += 1
            else:
                break
        
        return common_prefix / max(len(hash1), len(hash2))
    
    def recognize_face(self, face_img: np.ndarray) -> Optional[str]:
        """Recognize face and return person_id if found"""
        face_hash = self.get_image_hash(face_img)
        
        for known_hash, person_id in self.known_faces.items():
            similarity = self.calculate_similarity(face_hash, known_hash)
            if similarity >= self.similarity_threshold:
                return person_id
        
        return None
    
    def add_face(self, face_img: np.ndarray, person_id: str):
        """Add a new face to the database"""
        face_hash = self.get_image_hash(face_img)
        self.known_faces[face_hash] = person_id

class BasicTracker:
    """Simple tracker using bounding box overlap"""
    
    def __init__(self, max_disappeared: int = 30):
        self.next_id = 1
        self.objects = {}  # id -> (bbox, disappeared_count)
        self.max_disappeared = max_disappeared
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """Update tracker with new detections"""
        # If no objects being tracked, register all detections
        if len(self.objects) == 0:
            for bbox in detections:
                self.objects[self.next_id] = (bbox, 0)
                self.next_id += 1
            return {obj_id: bbox for obj_id, (bbox, _) in self.objects.items()}
        
        # Calculate overlaps between existing objects and new detections
        object_ids = list(self.objects.keys())
        object_bboxes = [self.objects[obj_id][0] for obj_id in object_ids]
        
        # Simple overlap calculation
        overlaps = []
        for i, obj_bbox in enumerate(object_bboxes):
            for j, det_bbox in enumerate(detections):
                overlap = self._calculate_overlap(obj_bbox, det_bbox)
                if overlap > 0.3:  # 30% overlap threshold
                    overlaps.append((object_ids[i], j, overlap))
        
        # Update existing objects
        used_detections = set()
        for obj_id, det_idx, _ in overlaps:
            self.objects[obj_id] = (detections[det_idx], 0)
            used_detections.add(det_idx)
        
        # Increment disappeared count for unused objects
        for obj_id in object_ids:
            if obj_id not in [overlap[0] for overlap in overlaps]:
                bbox, disappeared = self.objects[obj_id]
                self.objects[obj_id] = (bbox, disappeared + 1)
        
        # Remove objects that have disappeared too long
        self.objects = {obj_id: (bbox, disappeared) 
                       for obj_id, (bbox, disappeared) in self.objects.items()
                       if disappeared < self.max_disappeared}
        
        # Register new detections
        for i, bbox in enumerate(detections):
            if i not in used_detections:
                self.objects[self.next_id] = (bbox, 0)
                self.next_id += 1
        
        return {obj_id: bbox for obj_id, (bbox, _) in self.objects.items()}
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0

class BasicDatabase:
    """Simple SQLite database for storing visitor data"""
    
    def __init__(self, db_path: str = "visitors_basic.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Register datetime adapter for Python 3.12+ compatibility
        def adapt_datetime(dt):
            return dt.isoformat()
        
        def convert_datetime(s):
            return datetime.fromisoformat(s.decode())
        
        sqlite3.register_adapter(datetime, adapt_datetime)
        sqlite3.register_converter("TIMESTAMP", convert_datetime)
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                duration_seconds INTEGER,
                face_hash TEXT,
                image_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def clear_visitors(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM visitors')
        conn.commit()
        conn.close()
    
    def log_entry(self, person_id: str, face_hash: str, image_path: str = None) -> int:
        """Log a visitor entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO visitors (person_id, entry_time, face_hash, image_path)
            VALUES (?, ?, ?, ?)
        ''', (person_id, datetime.now(), face_hash, image_path))
        
        visitor_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return visitor_id
    
    def log_exit(self, person_id: str, duration_seconds: int = None):
        """Log a visitor exit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First, find the most recent entry for this person that hasn't been updated
        cursor.execute('''
            SELECT id FROM visitors 
            WHERE person_id = ? AND exit_time IS NULL
            ORDER BY entry_time DESC LIMIT 1
        ''', (person_id,))
        
        result = cursor.fetchone()
        if result:
            visitor_id = result[0]
            
            if duration_seconds is None:
                cursor.execute('''
                    UPDATE visitors 
                    SET exit_time = ?, duration_seconds = 
                        (julianday(?) - julianday(entry_time)) * 86400
                    WHERE id = ?
                ''', (datetime.now(), datetime.now(), visitor_id))
            else:
                cursor.execute('''
                    UPDATE visitors 
                    SET exit_time = ?, duration_seconds = ?
                    WHERE id = ?
                ''', (datetime.now(), duration_seconds, visitor_id))
        
        conn.commit()
        conn.close()
    
    def get_visitor_count(self) -> int:
        """Get total number of unique visitors"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(DISTINCT person_id) FROM visitors')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def get_recent_visitors(self, limit: int = 10) -> List[Dict]:
        """Get recent visitor entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT person_id, entry_time, exit_time, duration_seconds
            FROM visitors
            ORDER BY entry_time DESC
            LIMIT ?
        ''', (limit,))
        
        visitors = []
        for row in cursor.fetchall():
            visitors.append({
                'person_id': row[0],
                'entry_time': row[1],
                'exit_time': row[2],
                'duration_seconds': row[3]
            })
        
        conn.close()
        return visitors

class BasicLogger:
    """Simple logger for visitor events"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_file = self.log_dir / f"visitor_log_{datetime.now().strftime('%Y%m%d')}.txt"
    
    def log_event(self, event_type: str, person_id: str, details: str = ""):
        """Log an event"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {event_type}: {person_id} - {details}"
        
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

class BasicFaceRecognitionSystem:
    """Basic face recognition system using only OpenCV"""
    
    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.detector = BasicFaceDetector()
        self.recognizer = BasicFaceRecognizer(self.config.get('similarity_threshold', 0.8))
        self.tracker = BasicTracker(self.config.get('max_disappeared', 30))
        self.database = BasicDatabase(self.config.get('database_path', 'visitors_basic.db'))
        self.database.clear_visitors()  # <-- Add this line
        self.logger = BasicLogger(self.config.get('log_dir', 'logs'))
        
        # State tracking
        self.active_visitors = {}  # person_id -> entry_time
        self.frame_count = 0
        self.unique_visitor_ids = set()
        
        # Create directories
        Path(self.config.get('image_dir', 'visitor_images')).mkdir(exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'video_source': 0,
            'frame_skip': 5,
            'confidence_threshold': 0.5,
            'similarity_threshold': 0.8,
            'max_faces': 10,
            'max_disappeared': 30,
            'database_path': 'visitors_basic.db',
            'log_dir': 'logs',
            'image_dir': 'visitor_images',
            'display_fps': True,
            'save_images': True
        }
    
    def save_face_image(self, face_img: np.ndarray, person_id: str, event_type: str) -> str:
        """Save face image to disk"""
        if not self.config.get('save_images', True):
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_id}_{event_type}_{timestamp}.jpg"
        filepath = Path(self.config.get('image_dir', 'visitor_images')) / filename
        
        cv2.imwrite(str(filepath), face_img)
        return str(filepath)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.config.get('frame_skip', 5) != 0:
            return frame
        
        # Detect faces
        faces = self.detector.detect_faces(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(faces)
        
        # Process each tracked face
        for obj_id, (x, y, w, h) in tracked_objects.items():
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            
            # Recognize face
            person_id = self.recognizer.recognize_face(face_img)
            
            if person_id is None:
                # New face detected
                person_id = f"visitor_{obj_id}"
                self.recognizer.add_face(face_img, person_id)

            # Only log entry if this person_id has never been seen before
            if person_id not in self.unique_visitor_ids:
                self.unique_visitor_ids.add(person_id)
                if person_id not in self.active_visitors:
                    self.active_visitors[person_id] = time.time()
                    face_hash = self.recognizer.get_image_hash(face_img)
                    image_path = self.save_face_image(face_img, person_id, "entry")
                    self.database.log_entry(person_id, face_hash, image_path)
                    self.logger.log_event("ENTRY", person_id, f"New unique visitor detected (ID: {obj_id})")
            
            # Add label
            label = f"{person_id} (ID: {obj_id})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check for visitors who have left
        current_time = time.time()
        left_visitors = []
        for person_id, entry_time in self.active_visitors.items():
            if current_time - entry_time > 5:  # 5 seconds threshold
                duration = int(current_time - entry_time)
                self.database.log_exit(person_id, duration)
                self.logger.log_event("EXIT", person_id, f"Duration: {duration}s")
                left_visitors.append(person_id)
        
        for person_id in left_visitors:
            del self.active_visitors[person_id]
        
        return frame
    
    def run(self):
        """Run the face recognition system"""
        print("🚀 Starting Basic Face Recognition System")
        print("=" * 50)
        print(f"Video source: {self.config.get('video_source', 0)}")
        print(f"Frame skip: {self.config.get('frame_skip', 5)}")
        print(f"Press 'q' to quit, 's' to show stats")
        
        # Open video capture
        cap = cv2.VideoCapture(r"C:\Users\mohanraj\Desktop\Face recognition\video.mp4")
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):  # fallback if FPS is not available
            fps = 25  # default to 25 FPS
        delay = int(1000 / fps)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps:.1f} FPS")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Add FPS counter
                if self.config.get('display_fps', True):
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add visitor count
                visitor_count = self.database.get_visitor_count()
                cv2.putText(processed_frame, f"Visitors: {visitor_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Basic Face Recognition System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.show_stats()
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Log final stats
            final_count = self.database.get_visitor_count()
            print(f"\n📊 Final Statistics:")
            print(f"Total unique visitors: {final_count}")
            print(f"Session duration: {time.time() - start_time:.1f} seconds")
    
    def show_stats(self):
        """Show current statistics"""
        visitor_count = self.database.get_visitor_count()
        recent_visitors = self.database.get_recent_visitors(5)
        
        print(f"\n📊 Current Statistics:")
        print(f"Total unique visitors: {visitor_count}")
        print(f"Active visitors: {len(self.active_visitors)}")
        print(f"Recent visitors:")
        
        for visitor in recent_visitors:
            entry_time = visitor['entry_time']
            exit_time = visitor['exit_time'] or "Active"
            duration = visitor['duration_seconds'] or "N/A"
            print(f"  {visitor['person_id']}: {entry_time} - {exit_time} ({duration}s)")

def main():
    """Main function"""
    system = BasicFaceRecognitionSystem()
    system.run()

if __name__ == "__main__":
    main() 