import cv2
import json
import logging
import time
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import hashlib

from database import VisitorDatabase
from detector import FaceDetector
from tracker import FaceTracker
from logger import EventLogger

class SimpleFaceRecognizer:
    """Simple face recognition using image hashing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = config['recognition']['similarity_threshold']
        
        # In-memory cache for face hashes
        self.face_cache: Dict[str, str] = {}
        self.face_id_counter = 0
    
    def generate_face_hash(self, face_image: np.ndarray) -> str:
        """Generate a simple hash for face image."""
        try:
            # Resize to standard size
            resized = cv2.resize(face_image, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Create hash based on pixel distribution
            hash_values = []
            for i in range(0, 64, 8):
                for j in range(0, 64, 8):
                    block = gray[i:i+8, j:j+8]
                    hash_values.append(float(np.mean(block)))
            
            # Convert to hash string
            hash_str = hashlib.md5(str(hash_values).encode()).hexdigest()
            return hash_str
            
        except Exception as e:
            self.logger.error(f"Face hash generation failed: {e}")
            return ""
    
    def calculate_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two face hashes."""
        try:
            if not hash1 or not hash2:
                return 0.0
            
            if hash1 == hash2:
                return 1.0
            
            # Calculate Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            max_distance = len(hash1)
            similarity = 1.0 - (distance / max_distance)
            
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def recognize_face(self, face_hash: str, known_hashes: Dict[str, str]) -> Tuple[Optional[str], float]:
        """Recognize a face by comparing hashes."""
        try:
            best_match_id = None
            best_similarity = 0.0
            
            for face_id, known_hash in known_hashes.items():
                similarity = self.calculate_similarity(face_hash, known_hash)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = face_id
            
            return best_match_id, best_similarity
            
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def add_face_to_cache(self, face_id: str, face_hash: str):
        """Add face hash to cache."""
        self.face_cache[face_id] = face_hash
    
    def get_face_from_cache(self, face_id: str) -> Optional[str]:
        """Get face hash from cache."""
        return self.face_cache.get(face_id)
    
    def generate_face_id(self) -> str:
        """Generate unique face ID."""
        self.face_id_counter += 1
        return f"face_{self.face_id_counter}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def get_cache_size(self) -> int:
        """Get cache size."""
        return len(self.face_cache)

class SimpleFaceRecognitionVisitorCounter:
    """Simplified face recognition visitor counter without InsightFace."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.database = None
        self.detector = None
        self.recognizer = None
        self.tracker = None
        self.logger_component = None
        
        # Video capture
        self.cap = None
        self.frame_skip = self.config['video']['frame_skip']
        self.frame_count = 0
        
        # Application state
        self.running = False
        self.paused = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'unique_visitors': 0,
            'start_time': None,
            'processing_fps': 0
        }
        
        # Initialize all components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Failed to load config: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config['logging']['log_level'].upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('visitor_counter.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing simplified face recognition visitor counter...")
            
            # Initialize database
            self.database = VisitorDatabase(self.config['database']['path'])
            self.logger.info("Database initialized")
            
            # Initialize face detector
            self.detector = FaceDetector(self.config)
            self.logger.info("Face detector initialized")
            
            # Initialize simple face recognizer
            self.recognizer = SimpleFaceRecognizer(self.config)
            self.logger.info("Simple face recognizer initialized")
            
            # Initialize face tracker
            self.tracker = FaceTracker(self.config)
            self.logger.info("Face tracker initialized")
            
            # Initialize event logger
            self.logger_component = EventLogger(self.config, self.database)
            self.logger.info("Event logger initialized")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start_video_capture(self, source: str = None):
        """Start video capture from specified source."""
        try:
            if source is None:
                source = self.config['video']['source']
            
            # Try to convert source to int if it's a number
            try:
                source = int(source)
            except ValueError:
                pass  # Keep as string for file paths or RTSP URLs
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                raise Exception(f"Failed to open video source: {source}")
            
            # Set video properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video']['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config['video']['fps'])
            
            self.logger.info(f"Video capture started from source: {source}")
            
        except Exception as e:
            self.logger.error(f"Failed to start video capture: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the face recognition pipeline."""
        try:
            # Detect faces
            detections = self.detector.detect_faces(frame)
            self.stats['faces_detected'] += len(detections)
            
            # Update trackers
            tracked_faces = self.tracker.update_trackers(frame, detections)
            
            # Process each tracked face
            for face_id, bbox in tracked_faces.items():
                self._process_tracked_face(frame, face_id, bbox)
            
            # Draw tracking information
            frame = self.tracker.draw_tracking_info(frame, tracked_faces)
            
            # Draw detection boxes
            frame = self.detector.draw_detections(frame, detections)
            
            # Draw statistics
            frame = self._draw_statistics(frame)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            return frame
    
    def _process_tracked_face(self, frame: np.ndarray, face_id: str, bbox: Tuple[int, int, int, int]):
        """Process a tracked face for recognition and logging."""
        try:
            # Extract face region
            face_region = self.detector.extract_face_region(frame, bbox)
            
            if face_region is None:
                return
            
            # Generate face hash
            face_hash = self.recognizer.generate_face_hash(face_region)
            
            if not face_hash:
                return
            
            # Check if face is already known
            known_hash = self.recognizer.get_face_from_cache(face_id)
            
            if known_hash is None:
                # New face detected
                recognized_id, similarity = self.recognizer.recognize_face(
                    face_hash, self.recognizer.face_cache
                )
                
                if recognized_id is None:
                    # Completely new face
                    new_face_id = self.recognizer.generate_face_id()
                    self.recognizer.add_face_to_cache(new_face_id, face_hash)
                    
                    # Create a simple embedding for database (using hash as bytes)
                    embedding = np.frombuffer(face_hash.encode(), dtype=np.uint8).astype(np.float32)
                    
                    # Add to database
                    is_new_visitor = self.database.add_visitor(new_face_id, embedding)
                    
                    if is_new_visitor:
                        self.stats['unique_visitors'] += 1
                    
                    # Log entry
                    face_image_bytes = self.logger_component.save_face_image(
                        face_region, new_face_id, 'entry'
                    )
                    self.logger_component.log_entry(
                        new_face_id, face_image_bytes, 
                        location=(bbox[0], bbox[1])
                    )
                    
                    self.logger.info(f"New face detected and registered: {new_face_id}")
                    
                else:
                    # Face recognized from previous session
                    self.recognizer.add_face_to_cache(face_id, face_hash)
                    self.logger.debug(f"Face recognized from previous session: {recognized_id}")
            
            self.stats['faces_recognized'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to process tracked face {face_id}: {e}")
    
    def _draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """Draw statistics on the frame."""
        try:
            # Calculate FPS
            if self.stats['start_time'] is not None:
                elapsed_time = time.time() - self.stats['start_time']
                if elapsed_time > 0:
                    self.stats['processing_fps'] = self.stats['frames_processed'] / elapsed_time
            
            # Draw statistics
            stats_text = [
                f"FPS: {self.stats['processing_fps']:.1f}",
                f"Frames: {self.stats['frames_processed']}",
                f"Faces Detected: {self.stats['faces_detected']}",
                f"Faces Recognized: {self.stats['faces_recognized']}",
                f"Unique Visitors: {self.stats['unique_visitors']}",
                f"Active Trackers: {self.tracker.get_tracker_count()}",
                f"Cached Faces: {self.recognizer.get_cache_size()}",
                f"Mode: Simple Hash"
            ]
            
            y_offset = 30
            for text in stats_text:
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Failed to draw statistics: {e}")
            return frame
    
    def run(self, display: bool = True):
        """Main application loop."""
        try:
            self.running = True
            self.stats['start_time'] = time.time()
            
            self.logger.info("Starting simplified face recognition visitor counter...")
            
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("Failed to read frame, trying to reconnect...")
                    time.sleep(1)
                    continue
                
                # Process every nth frame based on frame_skip
                if self.frame_count % self.frame_skip == 0:
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    self.stats['frames_processed'] += 1
                    
                    # Display frame
                    if display:
                        cv2.imshow('Simple Face Recognition Visitor Counter', processed_frame)
                        
                        # Handle key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('p'):
                            self.paused = not self.paused
                            self.logger.info("Paused" if self.paused else "Resumed")
                        elif key == ord('s'):
                            self._save_screenshot(processed_frame)
                
                self.frame_count += 1
                
                # Limit processing rate
                time.sleep(1.0 / self.config['video']['fps'])
            
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
        finally:
            self.stop()
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save a screenshot of the current frame."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.logger.info(f"Screenshot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save screenshot: {e}")
    
    def stop(self):
        """Stop the application and cleanup resources."""
        try:
            self.running = False
            
            # Flush events
            if self.logger_component:
                self.logger_component.flush_events()
            
            # Release video capture
            if self.cap:
                self.cap.release()
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_statistics()
            
            self.logger.info("Application stopped")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def _print_final_statistics(self):
        """Print final application statistics."""
        try:
            print("\n" + "="*50)
            print("FINAL STATISTICS (Simple Mode)")
            print("="*50)
            
            # Application stats
            if self.stats['start_time'] is not None:
                total_time = time.time() - self.stats['start_time']
                print(f"Total Runtime: {total_time:.2f} seconds")
                print(f"Average FPS: {self.stats['processing_fps']:.2f}")
            
            print(f"Frames Processed: {self.stats['frames_processed']}")
            print(f"Faces Detected: {self.stats['faces_detected']}")
            print(f"Faces Recognized: {self.stats['faces_recognized']}")
            print(f"Unique Visitors: {self.stats['unique_visitors']}")
            
            # Database stats
            total_visitors = self.database.get_visitor_count()
            print(f"Total Visitors in Database: {total_visitors}")
            
            # Logging stats
            logging_stats = self.logger_component.get_logging_stats()
            print(f"Total Entries Logged: {logging_stats['total_entries']}")
            print(f"Total Exits Logged: {logging_stats['total_exits']}")
            print(f"Images Saved: {logging_stats['total_images_saved']}")
            
            print("="*50)
            
        except Exception as e:
            self.logger.error(f"Failed to print final statistics: {e}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Face Recognition Visitor Counter")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--source", help="Video source (camera index, file path, or RTSP URL)")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    
    args = parser.parse_args()
    
    try:
        # Create and start application
        app = SimpleFaceRecognitionVisitorCounter(args.config)
        
        # Start video capture
        app.start_video_capture(args.source)
        
        # Run application
        app.run(display=not args.no_display)
        
    except Exception as e:
        print(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 