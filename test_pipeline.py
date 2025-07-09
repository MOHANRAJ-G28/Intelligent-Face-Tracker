#!/usr/bin/env python3
"""
Test script for the face recognition visitor counter pipeline.
This script tests all components individually and as a complete system.
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from datetime import datetime

# Import our modules
from database import VisitorDatabase
from detector import FaceDetector
from recognizer import FaceRecognizer
from tracker import FaceTracker
from logger import EventLogger

def create_test_config():
    """Create a test configuration."""
    config = {
        "video": {
            "source": "0",
            "frame_skip": 1,
            "fps": 30,
            "width": 640,
            "height": 480
        },
        "detection": {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "model_size": "n",
            "device": "cpu"
        },
        "recognition": {
            "similarity_threshold": 0.6,
            "embedding_size": 512,
            "model_name": "buffalo_l",
            "device": "cpu"
        },
        "tracking": {
            "max_disappeared": 30,
            "min_hits": 3,
            "iou_threshold": 0.3
        },
        "logging": {
            "log_level": "INFO",
            "save_images": True,
            "flush_interval": 5,
            "max_log_size": 50
        },
        "database": {
            "path": "test_visitor_counter.db",
            "backup_interval": 3600
        }
    }
    
    with open("test_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return config

def test_database():
    """Test database functionality."""
    print("Testing Database...")
    
    try:
        # Initialize database
        db = VisitorDatabase("test_visitor_counter.db")
        
        # Test adding visitor
        test_embedding = np.random.rand(512).astype(np.float32)
        success = db.add_visitor("test_face_1", test_embedding)
        assert success, "Failed to add visitor"
        
        # Test retrieving embedding
        retrieved_embedding = db.get_visitor_embedding("test_face_1")
        assert retrieved_embedding is not None, "Failed to retrieve embedding"
        assert np.allclose(test_embedding, retrieved_embedding), "Embedding mismatch"
        
        # Test visitor count
        count = db.get_visitor_count()
        assert count == 1, f"Expected 1 visitor, got {count}"
        
        # Test logging event
        success = db.log_event("test_face_1", "entry", "test_image.jpg", 0.95, (100, 100))
        assert success, "Failed to log event"
        
        print("✅ Database tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_detector():
    """Test face detector."""
    print("Testing Face Detector...")
    
    try:
        config = create_test_config()
        detector = FaceDetector(config)
        
        # Create a test image with a simple pattern
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection (may not find faces in random image, but should not crash)
        detections = detector.detect_faces(test_image)
        assert isinstance(detections, list), "Detections should be a list"
        
        # Test face extraction
        if len(detections) > 0:
            bbox = detections[0][:4]
            face_region = detector.extract_face_region(test_image, bbox)
            assert face_region is not None, "Failed to extract face region"
        
        print("✅ Face detector tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Face detector test failed: {e}")
        return False

def test_recognizer():
    """Test face recognizer."""
    print("Testing Face Recognizer...")
    
    try:
        config = create_test_config()
        recognizer = FaceRecognizer(config)
        
        # Create test face image
        test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # Test embedding generation
        embedding = recognizer.generate_embedding(test_face)
        if embedding is not None:
            assert embedding.shape[0] == 512, f"Expected 512-dim embedding, got {embedding.shape[0]}"
        
        # Test similarity calculation
        if embedding is not None:
            embedding2 = np.random.rand(512).astype(np.float32)
            similarity = recognizer.calculate_similarity(embedding, embedding2)
            assert 0 <= similarity <= 1, f"Similarity should be between 0 and 1, got {similarity}"
        
        # Test face ID generation
        face_id = recognizer.generate_face_id()
        assert isinstance(face_id, str), "Face ID should be a string"
        assert len(face_id) > 0, "Face ID should not be empty"
        
        print("✅ Face recognizer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Face recognizer test failed: {e}")
        return False

def test_tracker():
    """Test face tracker."""
    print("Testing Face Tracker...")
    
    try:
        config = create_test_config()
        tracker = FaceTracker(config)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test tracker initialization
        bbox = (100, 100, 50, 50)  # x, y, width, height
        success = tracker.initialize_tracker("test_face_1", bbox, test_frame)
        assert success, "Failed to initialize tracker"
        
        # Test tracker update
        detections = [(110, 110, 160, 160, 0.9)]  # x1, y1, x2, y2, confidence
        tracked_faces = tracker.update_trackers(test_frame, detections)
        assert isinstance(tracked_faces, dict), "Tracked faces should be a dictionary"
        
        # Test tracker count
        count = tracker.get_tracker_count()
        assert count >= 0, "Tracker count should be non-negative"
        
        print("✅ Face tracker tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Face tracker test failed: {e}")
        return False

def test_logger():
    """Test event logger."""
    print("Testing Event Logger...")
    
    try:
        config = create_test_config()
        db = VisitorDatabase("test_visitor_counter.db")
        logger = EventLogger(config, db)
        
        # Test entry logging
        test_image_bytes = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8).tobytes()
        success = logger.log_entry("test_face_1", test_image_bytes, 0.95, (100, 100))
        assert success, "Failed to log entry"
        
        # Test exit logging
        success = logger.log_exit("test_face_1", test_image_bytes, 0.95, (100, 100))
        assert success, "Failed to log exit"
        
        # Test statistics
        stats = logger.get_logging_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert stats['total_entries'] >= 1, "Should have at least one entry"
        assert stats['total_exits'] >= 1, "Should have at least one exit"
        
        print("✅ Event logger tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Event logger test failed: {e}")
        return False

def test_integration():
    """Test integration of all components."""
    print("Testing Integration...")
    
    try:
        config = create_test_config()
        
        # Initialize all components
        db = VisitorDatabase("test_visitor_counter.db")
        detector = FaceDetector(config)
        recognizer = FaceRecognizer(config)
        tracker = FaceTracker(config)
        logger = EventLogger(config, db)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simulate pipeline
        detections = detector.detect_faces(test_frame)
        tracked_faces = tracker.update_trackers(test_frame, detections)
        
        # Process tracked faces
        for face_id, bbox in tracked_faces.items():
            face_region = detector.extract_face_region(test_frame, bbox)
            if face_region is not None:
                embedding = recognizer.generate_embedding(face_region)
                if embedding is not None:
                    recognizer.add_face_to_cache(face_id, embedding)
                    db.add_visitor(face_id, embedding)
                    
                    # Log entry
                    image_bytes = logger.save_face_image(face_region, face_id, 'entry')
                    logger.log_entry(face_id, image_bytes, 0.95, (bbox[0], bbox[1]))
        
        print("✅ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        "test_config.json",
        "test_visitor_counter.db",
        "visitor_counter.log"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Face Recognition Visitor Counter - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Database", test_database),
        ("Face Detector", test_detector),
        ("Face Recognizer", test_recognizer),
        ("Face Tracker", test_tracker),
        ("Event Logger", test_logger),
        ("Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        cleanup_test_files()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        cleanup_test_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        cleanup_test_files()
        sys.exit(1) 