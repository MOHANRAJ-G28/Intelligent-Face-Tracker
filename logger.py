import os
import cv2
import logging
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import threading
from collections import deque
import numpy as np

class EventLogger:
    """Event logger for face recognition system."""
    
    def __init__(self, config: dict, database):
        self.config = config
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.save_images = config['logging']['save_images']
        self.flush_interval = config['logging']['flush_interval']
        self.max_log_size = config['logging']['max_log_size']
        
        # Create log directories
        self.logs_dir = "logs"
        self.entries_dir = os.path.join(self.logs_dir, "entries")
        self.exits_dir = os.path.join(self.logs_dir, "exits")
        self._create_directories()
        
        # Event queue for batch processing
        self.event_queue = deque(maxlen=self.max_log_size)
        self.queue_lock = threading.Lock()
        
        # Statistics
        self.total_entries = 0
        self.total_exits = 0
        self.total_images_saved = 0
        
        # Start background flush thread
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
    
    def _create_directories(self):
        """Create necessary log directories."""
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.entries_dir, exist_ok=True)
            os.makedirs(self.exits_dir, exist_ok=True)
            
            # Create date-based subdirectories
            today = datetime.now().strftime("%Y-%m-%d")
            os.makedirs(os.path.join(self.entries_dir, today), exist_ok=True)
            os.makedirs(os.path.join(self.exits_dir, today), exist_ok=True)
            
            self.logger.info("Log directories created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create log directories: {e}")
            raise
    
    def log_entry(self, face_id: str, face_image: Optional[bytes], 
                  confidence: float = None, location: Tuple[int, int] = None) -> bool:
        """
        Log a face entry event.
        
        Args:
            face_id: Unique identifier for the face
            face_image: Face image as bytes (optional)
            confidence: Detection confidence
            location: Face location (x, y)
            
        Returns:
            True if logging successful, False otherwise
        """
        try:
            timestamp = datetime.now()
            event_data = {
                'type': 'entry',
                'face_id': face_id,
                'timestamp': timestamp.isoformat(),
                'confidence': confidence,
                'location': location,
                'image': face_image
            }
            
            # Add to queue
            with self.queue_lock:
                self.event_queue.append(event_data)
            
            # Save image if provided
            image_path = None
            if face_image is not None and self.save_images:
                image_path = self._save_entry_image(face_id, face_image, timestamp)
            
            # Log to database
            self.database.log_event(face_id, 'entry', image_path, confidence, location)
            
            self.total_entries += 1
            self.logger.info(f"Entry logged for face {face_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log entry for {face_id}: {e}")
            return False
    
    def log_exit(self, face_id: str, face_image: Optional[bytes], 
                 confidence: float = None, location: Tuple[int, int] = None) -> bool:
        """
        Log a face exit event.
        
        Args:
            face_id: Unique identifier for the face
            face_image: Face image as bytes (optional)
            confidence: Detection confidence
            location: Face location (x, y)
            
        Returns:
            True if logging successful, False otherwise
        """
        try:
            timestamp = datetime.now()
            event_data = {
                'type': 'exit',
                'face_id': face_id,
                'timestamp': timestamp.isoformat(),
                'confidence': confidence,
                'location': location,
                'image': face_image
            }
            
            # Add to queue
            with self.queue_lock:
                self.event_queue.append(event_data)
            
            # Save image if provided
            image_path = None
            if face_image is not None and self.save_images:
                image_path = self._save_exit_image(face_id, face_image, timestamp)
            
            # Log to database
            self.database.log_event(face_id, 'exit', image_path, confidence, location)
            
            self.total_exits += 1
            self.logger.info(f"Exit logged for face {face_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log exit for {face_id}: {e}")
            return False
    
    def _save_entry_image(self, face_id: str, face_image: bytes, timestamp: datetime) -> Optional[str]:
        """Save entry face image to disk."""
        try:
            # Create filename
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H-%M-%S-%f")[:-3]  # Include milliseconds
            filename = f"{face_id}_entry_{time_str}.jpg"
            
            # Create directory path
            dir_path = os.path.join(self.entries_dir, date_str)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save image
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'wb') as f:
                f.write(face_image)
            
            self.total_images_saved += 1
            self.logger.debug(f"Entry image saved: {file_path}")
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save entry image: {e}")
            return None
    
    def _save_exit_image(self, face_id: str, face_image: bytes, timestamp: datetime) -> Optional[str]:
        """Save exit face image to disk."""
        try:
            # Create filename
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H-%M-%S-%f")[:-3]  # Include milliseconds
            filename = f"{face_id}_exit_{time_str}.jpg"
            
            # Create directory path
            dir_path = os.path.join(self.exits_dir, date_str)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save image
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'wb') as f:
                f.write(face_image)
            
            self.total_images_saved += 1
            self.logger.debug(f"Exit image saved: {file_path}")
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save exit image: {e}")
            return None
    
    def _flush_worker(self):
        """Background worker to flush events to disk."""
        import time
        
        while True:
            try:
                time.sleep(self.flush_interval)
                self.flush_events()
            except Exception as e:
                self.logger.error(f"Flush worker error: {e}")
    
    def flush_events(self):
        """Flush queued events to disk."""
        try:
            with self.queue_lock:
                if not self.event_queue:
                    return
                
                # Get events to flush
                events_to_flush = list(self.event_queue)
                self.event_queue.clear()
            
            # Write events to log file
            log_file = os.path.join(self.logs_dir, "events.log")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                for event in events_to_flush:
                    # Remove image data for log file
                    log_event = event.copy()
                    if 'image' in log_event:
                        del log_event['image']
                    
                    f.write(json.dumps(log_event) + '\n')
            
            self.logger.debug(f"Flushed {len(events_to_flush)} events to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to flush events: {e}")
    
    def get_logging_stats(self) -> Dict:
        """Get logging statistics."""
        return {
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "total_images_saved": self.total_images_saved,
            "queued_events": len(self.event_queue),
            "save_images": self.save_images,
            "flush_interval": self.flush_interval
        }
    
    def get_recent_events(self, limit: int = 10) -> list:
        """Get recent events from the queue."""
        with self.queue_lock:
            return list(self.event_queue)[-limit:]
    
    def clear_queue(self):
        """Clear the event queue."""
        with self.queue_lock:
            self.event_queue.clear()
        self.logger.info("Event queue cleared")
    
    def save_face_image(self, face_image: np.ndarray, face_id: str, event_type: str) -> Optional[bytes]:
        """
        Convert face image to bytes for storage.
        
        Args:
            face_image: Face image as numpy array
            face_id: Face identifier
            event_type: Type of event ('entry' or 'exit')
            
        Returns:
            Image as bytes or None if conversion fails
        """
        try:
            # Encode image to JPEG
            success, encoded_image = cv2.imencode('.jpg', face_image)
            
            if success:
                return encoded_image.tobytes()
            else:
                self.logger.error(f"Failed to encode image for {face_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to convert image to bytes: {e}")
            return None
    
    def get_daily_summary(self, date: str = None) -> Dict:
        """Get daily logging summary."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Get database stats
            db_stats = self.database.get_daily_stats(date)
            
            # Count images in directories
            entry_images = 0
            exit_images = 0
            
            entry_dir = os.path.join(self.entries_dir, date)
            exit_dir = os.path.join(self.exits_dir, date)
            
            if os.path.exists(entry_dir):
                entry_images = len([f for f in os.listdir(entry_dir) if f.endswith('.jpg')])
            
            if os.path.exists(exit_dir):
                exit_images = len([f for f in os.listdir(exit_dir) if f.endswith('.jpg')])
            
            return {
                "date": date,
                "database_stats": db_stats,
                "entry_images": entry_images,
                "exit_images": exit_images,
                "total_images": entry_images + exit_images
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get daily summary: {e}")
            return {"date": date, "error": str(e)} 