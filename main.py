import cv2
import json
import logging
import time
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from database import VisitorDatabase
from detector import FaceDetector
from recognizer import FaceRecognizer
from tracker import FaceTracker
from logger import EventLogger

class FaceRecognitionVisitorCounter:
    """Main application class for face recognition visitor counter."""
    
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
            self.logger.info("Initializing face recognition visitor counter...")
            
            # Initialize database
            self.database = VisitorDatabase(self.config['database']['path'])
            self.logger.info("Database initialized")
            
            # Initialize face detector
            self.detector = FaceDetector(self.config)
            self.logger.info("Face detector initialized")
            
            # Initialize face recognizer
            self.recognizer = FaceRecognizer(self.config)
            self.logger.info("Face recognizer initialized")
            
            # Initialize face tracker
            self.tracker = FaceTracker(self.config)
            self.logger.info("Face tracker initialized")
            
            # Initialize event logger
            self.logger_component = EventLogger(self.config, self.database)
            self.logger.info("Event logger initialized")
            
            # Load existing faces from database
            self._load_existing_faces()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _load_existing_faces(self):
        """Load existing faces from database into recognizer cache."""
        try:
            visitors = self.database.get_all_visitors()
            
            for visitor in visitors:
                face_id = visitor['face_id']
                embedding = self.database.get_visitor_embedding(face_id)
                
                if embedding is not None:
                    self.recognizer.add_face_to_cache(face_id, embedding)
            
            self.logger.info(f"Loaded {len(visitors)} existing faces from database")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing faces: {e}")
    
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
        """
        Process a single frame through the face recognition pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with annotations
        """
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
            
            # Generate embedding
            embedding = self.recognizer.generate_embedding(face_region)
            
            if embedding is None:
                return
            
            # Check if face is already known
            known_embedding = self.recognizer.get_face_from_cache(face_id)
            
            if known_embedding is None:
                # New face detected
                recognized_id, similarity = self.recognizer.recognize_face(
                    embedding, self.recognizer.embedding_cache
                )
                
                if recognized_id is None:
                    # Completely new face
                    new_face_id = self.recognizer.generate_face_id()
                    self.recognizer.add_face_to_cache(new_face_id, embedding)
                    
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
                    self.recognizer.add_face_to_cache(face_id, embedding)
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
                f"Cached Faces: {self.recognizer.get_cache_size()}"
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
            
            self.logger.info("Starting face recognition visitor counter...")
            
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
                        cv2.imshow('Face Recognition Visitor Counter', processed_frame)
                        
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
            print("FINAL STATISTICS")
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
            
            # Component stats
            detector_stats = self.detector.get_detection_stats()
            recognizer_stats = self.recognizer.get_recognition_stats()
            tracker_stats = self.tracker.get_tracking_stats()
            
            print(f"\nDetection Model: {detector_stats['model']}")
            print(f"Recognition Model: {recognizer_stats['model']}")
            print(f"Active Trackers: {tracker_stats['active_trackers']}")
            
            print("="*50)
            
        except Exception as e:
            self.logger.error(f"Failed to print final statistics: {e}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition Visitor Counter")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--source", help="Video source (camera index, file path, or RTSP URL)")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    
    args = parser.parse_args()
    
    try:
        # Create and start application
        app = FaceRecognitionVisitorCounter(args.config)
        
        # Start video capture
        app.start_video_capture(args.source)
        
        # Run application
        app.run(display=not args.no_display)
        
    except Exception as e:
        print(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 