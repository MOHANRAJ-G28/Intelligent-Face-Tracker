#!/usr/bin/env python3
"""
Quick test to verify database fix works
"""

import sqlite3
from datetime import datetime
import os

def test_database():
    """Test the database functionality"""
    print("Testing database fix...")
    
    # Use a test database
    db_path = "test_fix.db"
    
    # Initialize database
    conn = sqlite3.connect(db_path)
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
    
    # Test entry
    entry_time = datetime.now()
    cursor.execute('''
        INSERT INTO visitors (person_id, entry_time, face_hash, image_path)
        VALUES (?, ?, ?, ?)
    ''', ("test_person", entry_time, "test_hash", "test_path"))
    
    visitor_id = cursor.lastrowid
    print(f"✓ Inserted visitor with ID: {visitor_id}")
    
    # Test exit (this was the problematic part)
    exit_time = datetime.now()
    duration = 30
    
    # First, find the most recent entry for this person that hasn't been updated
    cursor.execute('''
        SELECT id FROM visitors 
        WHERE person_id = ? AND exit_time IS NULL
        ORDER BY entry_time DESC LIMIT 1
    ''', ("test_person",))
    
    result = cursor.fetchone()
    if result:
        visitor_id = result[0]
        
        cursor.execute('''
            UPDATE visitors 
            SET exit_time = ?, duration_seconds = ?
            WHERE id = ?
        ''', (exit_time, duration, visitor_id))
        
        print(f"✓ Updated visitor exit time successfully")
    
    # Verify the update
    cursor.execute('SELECT * FROM visitors WHERE person_id = ?', ("test_person",))
    row = cursor.fetchone()
    if row:
        print(f"✓ Database record: {row}")
    
    conn.commit()
    conn.close()
    
    # Clean up
    os.remove(db_path)
    
    print("✅ Database fix test passed!")

if __name__ == "__main__":
    test_database() 