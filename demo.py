#!/usr/bin/env python3
"""
Demo script for the Face Recognition Visitor Counter.
This script demonstrates the system capabilities with sample data.
"""

import cv2
import numpy as np
import json
import time
import os
from datetime import datetime

from database import VisitorDatabase
from detector import FaceDetector
from recognizer import FaceRecognizer
from tracker import FaceTracker
from logger import EventLogger

def create_demo_config():
    """Create a demo configuration optimized for demonstration."""
    config = {
        "video": {
            "source": "0",
            "frame_skip": 1,
            "fps": 15,
            "width": 640,
            "height": 480
        },
        "detection": {
            "confidence_threshold": 0.6,
            "nms_threshold": 0.4,
            "model_size": "n",
            "device": "cpu"
        },
        "recognition": {
            "similarity_threshold": 0.65,
            "embedding_size": 512,
            "model_name": "buffalo_l",
            "device": "cpu"
        },
        "tracking": {
            "max_disappeared": 20,
            "min_hits": 2,
            "iou_threshold": 0.3
        },
        "logging": {
            "log_level": "INFO",
            "save_images": True,
            "flush_interval": 5,
            "max_log_size": 50
        },
        "database": {
            "path": "demo_visitor_counter.db",
            "backup_interval": 3600
        }
    }
    
    with open("demo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return config

def create_sample_faces():
    """Create sample face images for demonstration."""
    faces = []
    
    # Create different colored rectangles as "faces" for demo
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, color in enumerate(colors):
        # Create a face-like image
        face = np.ones((112, 112, 3), dtype=np.uint8) * 128
        face[20:90, 30:80] = color
        
        # Add some "features"
        cv2.circle(face, (40, 40), 5, (255, 255, 255), -1)  # Left eye
        cv2.circle(face, (70, 40), 5, (255, 255, 255), -1)  # Right eye
        cv2.ellipse(face, (55, 70), (15, 8), 0, 0, 180, (255, 255, 255), 2)  # Mouth
        
        faces.append(face)
    
    return faces

def simulate_visitors(config, duration=30):
    """Simulate visitors entering and exiting the frame."""
    print("🎬 Starting Face Recognition Visitor Counter Demo")
    print("=" * 60)
    
    # Initialize components
    db = VisitorDatabase(config['database']['path'])
    detector = FaceDetector(config)
    recognizer = FaceRecognizer(config)
    tracker = FaceTracker(config)
    logger = EventLogger(config, db)
    
    # Create sample faces
    sample_faces = create_sample_faces()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera available. Using simulated video.")
        cap = None
    
    # Demo statistics
    stats = {
        'frames_processed': 0,
        'faces_detected': 0,
        'unique_visitors': 0,
        'start_time': time.time()
    }
    
    # Demo loop
    frame_count = 0
    visitor_schedule = [
        (5, 0),   # Visitor 1 enters at 5 seconds
        (10, 1),  # Visitor 2 enters at 10 seconds
        (15, 0),  # Visitor 1 exits at 15 seconds
        (20, 2),  # Visitor 3 enters at 20 seconds
        (25, 1),  # Visitor 2 exits at 25 seconds
        (30, 2),  # Visitor 3 exits at 30 seconds
    ]
    
    print("📊 Demo Schedule:")
    for time_sec, visitor_id in visitor_schedule:
        action = "enters" if visitor_id not in [v[1] for v in visitor_schedule[:visitor_schedule.index((time_sec, visitor_id))]] else "exits"
        print(f"   {time_sec}s: Visitor {visitor_id + 1} {action}")
    
    print("\n🎯 Demo Controls:")
    print("   Press 'q' to quit")
    print("   Press 'p' to pause/resume")
    print("   Press 's' to save screenshot")
    print("\n" + "=" * 60)
    
    try:
        while True:
            current_time = time.time() - stats['start_time']
            
            if current_time > duration:
                break
            
            # Read frame
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Create simulated frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Demo Mode - No Camera", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Process frame
            if frame_count % config['video']['frame_skip'] == 0:
                # Detect faces
                detections = detector.detect_faces(frame)
                stats['faces_detected'] += len(detections)
                
                # Update trackers
                tracked_faces = tracker.update_trackers(frame, detections)
                
                # Process tracked faces
                for face_id, bbox in tracked_faces.items():
                    face_region = detector.extract_face_region(frame, bbox)
                    if face_region is not None:
                        embedding = recognizer.generate_embedding(face_region)
                        if embedding is not None:
                            # Check if new face
                            known_embedding = recognizer.get_face_from_cache(face_id)
                            if known_embedding is None:
                                recognized_id, similarity = recognizer.recognize_face(
                                    embedding, recognizer.embedding_cache
                                )
                                
                                if recognized_id is None:
                                    # New visitor
                                    new_face_id = recognizer.generate_face_id()
                                    recognizer.add_face_to_cache(new_face_id, embedding)
                                    is_new = db.add_visitor(new_face_id, embedding)
                                    
                                    if is_new:
                                        stats['unique_visitors'] += 1
                                    
                                    # Log entry
                                    image_bytes = logger.save_face_image(face_region, new_face_id, 'entry')
                                    logger.log_entry(new_face_id, image_bytes, 0.95, (bbox[0], bbox[1]))
                                    
                                    print(f"👤 New visitor detected: {new_face_id}")
                
                stats['frames_processed'] += 1
            
            # Draw demo information
            frame = draw_demo_info(frame, stats, current_time, duration, visitor_schedule)
            
            # Draw tracking information
            frame = tracker.draw_tracking_info(frame, tracked_faces)
            
            # Display frame
            cv2.imshow('Face Recognition Demo', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"demo_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            
            frame_count += 1
            time.sleep(1.0 / config['video']['fps'])
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print_demo_statistics(stats, db, logger)
        
        # Cleanup demo files
        cleanup_demo_files()

def draw_demo_info(frame, stats, current_time, duration, visitor_schedule):
    """Draw demo information on the frame."""
    # Background for text
    cv2.rectangle(frame, (0, 0), (640, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (640, 120), (255, 255, 255), 2)
    
    # Demo title
    cv2.putText(frame, "Face Recognition Visitor Counter - DEMO", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Progress bar
    progress = current_time / duration
    bar_width = 400
    bar_height = 20
    bar_x, bar_y = 10, 40
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
    cv2.putText(frame, f"Progress: {progress*100:.1f}%", (bar_x + bar_width + 10, bar_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Statistics
    stats_text = [
        f"Time: {current_time:.1f}s / {duration}s",
        f"Frames: {stats['frames_processed']}",
        f"Faces Detected: {stats['faces_detected']}",
        f"Unique Visitors: {stats['unique_visitors']}"
    ]
    
    y_offset = 70
    for text in stats_text:
        cv2.putText(frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    
    # Next scheduled event
    for time_sec, visitor_id in visitor_schedule:
        if time_sec > current_time:
            action = "enters" if visitor_id not in [v[1] for v in visitor_schedule[:visitor_schedule.index((time_sec, visitor_id))]] else "exits"
            cv2.putText(frame, f"Next: Visitor {visitor_id + 1} {action} at {time_sec}s", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            break
    
    return frame

def print_demo_statistics(stats, db, logger):
    """Print demo statistics."""
    print("\n" + "=" * 60)
    print("📊 DEMO STATISTICS")
    print("=" * 60)
    
    total_time = time.time() - stats['start_time']
    print(f"Demo Duration: {total_time:.2f} seconds")
    print(f"Frames Processed: {stats['frames_processed']}")
    print(f"Average FPS: {stats['frames_processed']/total_time:.2f}")
    print(f"Faces Detected: {stats['faces_detected']}")
    print(f"Unique Visitors: {stats['unique_visitors']}")
    
    # Database stats
    total_visitors = db.get_visitor_count()
    print(f"Visitors in Database: {total_visitors}")
    
    # Logging stats
    logging_stats = logger.get_logging_stats()
    print(f"Entries Logged: {logging_stats['total_entries']}")
    print(f"Exits Logged: {logging_stats['total_exits']}")
    print(f"Images Saved: {logging_stats['total_images_saved']}")
    
    print("=" * 60)

def cleanup_demo_files():
    """Clean up demo files."""
    demo_files = [
        "demo_config.json",
        "demo_visitor_counter.db",
        "visitor_counter.log"
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Cleaned up: {file}")

def main():
    """Main demo function."""
    print("🎭 Face Recognition Visitor Counter - Demo Mode")
    print("This demo showcases the system capabilities.")
    print("Make sure you have a camera connected for the full experience.")
    print()
    
    # Create demo configuration
    config = create_demo_config()
    
    # Run demo
    simulate_visitors(config, duration=30)
    
    print("\n🎉 Demo completed!")
    print("Check the logs/ directory for saved face images and events.")

if __name__ == "__main__":
    main() 