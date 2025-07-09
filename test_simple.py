#!/usr/bin/env python3
"""
Simple test script that doesn't require InsightFace.
This is a fallback for Python 3.13 compatibility issues.
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from datetime import datetime

# Import our modules (excluding recognizer for now)
from database import VisitorDatabase
from detector import FaceDetector
from tracker import FaceTracker
from logger import EventLogger

def create_simple_config():
    """Create a simple configuration for testing."""
    config = {
        "video": {
            "source": "0",
            "frame_skip": 2,
            "fps": 15,
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
            "path": "simple_test.db",
            "backup_interval": 3600
        }
    }
    
    with open("simple_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return config

def test_basic_components():
    """Test basic components without face recognition."""
    print("🧪 Testing Basic Components (No Face Recognition)")
    print("=" * 60)
    
    try:
        config = create_simple_config()
        
        # Test database
        print("Testing Database...")
        db = VisitorDatabase(config['database']['path'])
        print("✅ Database initialized")
        
        # Test detector
        print("Testing Face Detector...")
        detector = FaceDetector(config)
        print("✅ Face detector initialized")
        
        # Test tracker
        print("Testing Face Tracker...")
        tracker = FaceTracker(config)
        print("✅ Face tracker initialized")
        
        # Test logger
        print("Testing Event Logger...")
        logger = EventLogger(config, db)
        print("✅ Event logger initialized")
        
        # Test video capture
        print("Testing Video Capture...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera accessible")
            ret, frame = cap.read()
            if ret:
                print("✅ Frame capture successful")
                
                # Test face detection
                detections = detector.detect_faces(frame)
                print(f"✅ Face detection: {len(detections)} faces found")
                
                # Test tracking
                tracked_faces = tracker.update_trackers(frame, detections)
                print(f"✅ Face tracking: {len(tracked_faces)} faces tracked")
                
            cap.release()
        else:
            print("⚠️ Camera not accessible (using simulated frame)")
            
            # Create test frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Test Frame", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Test face detection
            detections = detector.detect_faces(frame)
            print(f"✅ Face detection: {len(detections)} faces found")
            
            # Test tracking
            tracked_faces = tracker.update_trackers(frame, detections)
            print(f"✅ Face tracking: {len(tracked_faces)} faces tracked")
        
        print("\n🎉 Basic component tests passed!")
        print("Note: Face recognition requires InsightFace which may not be compatible with Python 3.13")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def cleanup_simple_files():
    """Clean up simple test files."""
    simple_files = [
        "simple_config.json",
        "simple_test.db",
        "visitor_counter.log"
    ]
    
    for file in simple_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

def main():
    """Main test function."""
    print("🎭 Simple Face Recognition Test (Python 3.13 Compatible)")
    print("This test verifies basic functionality without InsightFace.")
    print()
    
    try:
        success = test_basic_components()
        cleanup_simple_files()
        
        if success:
            print("\n✅ Basic system is working!")
            print("To use full face recognition, try installing InsightFace manually:")
            print("   pip install insightface")
            print("   Or use Python 3.11/3.12 for better compatibility")
        else:
            print("\n❌ Basic system test failed")
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        cleanup_simple_files()

if __name__ == "__main__":
    main() 