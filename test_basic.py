#!/usr/bin/env python3
"""
Test script for basic face recognition system
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

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        
        import numpy as np
        print(f"✓ NumPy imported successfully (version: {np.__version__})")
        
        import sqlite3
        print("✓ SQLite3 imported successfully")
        
        from PIL import Image
        print("✓ PIL imported successfully")
        
        import imagehash
        print("✓ ImageHash imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_face_detection():
    """Test basic face detection"""
    print("\nTesting face detection...")
    try:
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print("⚠️  Could not load face cascade")
            return False
        
        print("✓ Face cascade loaded successfully")
        
        # Create a test image (simple colored rectangle)
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        test_image[:] = (100, 100, 100)  # Gray background
        
        # Test detection (should not find faces in this image)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        print(f"✓ Face detection test completed (found {len(faces)} faces)")
        return True
        
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\nTesting database...")
    try:
        # Test database connection
        db_path = "test_visitors.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL
            )
        ''')
        
        # Insert test data
        cursor.execute('''
            INSERT INTO test_visitors (person_id, entry_time)
            VALUES (?, ?)
        ''', ("test_person", datetime.now()))
        
        # Query test data
        cursor.execute('SELECT COUNT(*) FROM test_visitors')
        count = cursor.fetchone()[0]
        
        conn.commit()
        conn.close()
        
        # Clean up
        os.remove(db_path)
        
        print(f"✓ Database test completed (inserted and queried {count} records)")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_image_processing():
    """Test basic image processing"""
    print("\nTesting image processing...")
    try:
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = (255, 0, 0)  # Blue image
        
        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(test_image, (50, 50))
        
        # Test hash generation
        hash_value = hashlib.md5(test_image.tobytes()).hexdigest()[:16]
        
        print(f"✓ Image processing test completed (hash: {hash_value})")
        return True
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Basic Face Recognition System - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_face_detection,
        test_database,
        test_image_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The system is ready to use.")
        print("\n🚀 You can now run:")
        print("   python main_basic.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 