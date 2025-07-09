import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

class VisitorDatabase:
    """Database handler for storing visitor information and events."""
    
    def __init__(self, db_path: str = "visitor_counter.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create visitors table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS visitors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_id TEXT UNIQUE NOT NULL,
                        embedding BLOB,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        visit_count INTEGER DEFAULT 1,
                        total_duration INTEGER DEFAULT 0
                    )
                ''')
                
                # Create events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        image_path TEXT,
                        confidence REAL,
                        location_x INTEGER,
                        location_y INTEGER,
                        FOREIGN KEY (face_id) REFERENCES visitors (face_id)
                    )
                ''')
                
                # Create daily_stats table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE UNIQUE NOT NULL,
                        total_visitors INTEGER DEFAULT 0,
                        unique_visitors INTEGER DEFAULT 0,
                        total_entries INTEGER DEFAULT 0,
                        total_exits INTEGER DEFAULT 0
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def add_visitor(self, face_id: str, embedding: np.ndarray) -> bool:
        """Add a new visitor to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert embedding to bytes for storage
                embedding_bytes = embedding.tobytes()
                
                cursor.execute('''
                    INSERT OR IGNORE INTO visitors (face_id, embedding, first_seen, last_seen)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''', (face_id, embedding_bytes))
                
                if cursor.rowcount > 0:
                    self.logger.info(f"New visitor added: {face_id}")
                    return True
                else:
                    # Update last_seen for existing visitor
                    cursor.execute('''
                        UPDATE visitors 
                        SET last_seen = CURRENT_TIMESTAMP, visit_count = visit_count + 1
                        WHERE face_id = ?
                    ''', (face_id,))
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to add visitor {face_id}: {e}")
            return False
    
    def get_visitor_embedding(self, face_id: str) -> Optional[np.ndarray]:
        """Retrieve visitor embedding from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT embedding FROM visitors WHERE face_id = ?', (face_id,))
                result = cursor.fetchone()
                
                if result:
                    embedding_bytes = result[0]
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    return embedding
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get embedding for {face_id}: {e}")
            return None
    
    def log_event(self, face_id: str, event_type: str, image_path: str = None, 
                  confidence: float = None, location: Tuple[int, int] = None) -> bool:
        """Log an event (entry/exit) to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                location_x, location_y = location if location else (None, None)
                
                cursor.execute('''
                    INSERT INTO events (face_id, event_type, image_path, confidence, location_x, location_y)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (face_id, event_type, image_path, confidence, location_x, location_y))
                
                conn.commit()
                self.logger.info(f"Event logged: {event_type} for {face_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to log event for {face_id}: {e}")
            return False
    
    def get_all_visitors(self) -> List[Dict]:
        """Get all visitors from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT face_id, first_seen, last_seen, visit_count, total_duration
                    FROM visitors
                    ORDER BY last_seen DESC
                ''')
                
                visitors = []
                for row in cursor.fetchall():
                    visitors.append({
                        'face_id': row[0],
                        'first_seen': row[1],
                        'last_seen': row[2],
                        'visit_count': row[3],
                        'total_duration': row[4]
                    })
                
                return visitors
                
        except Exception as e:
            self.logger.error(f"Failed to get visitors: {e}")
            return []
    
    def get_visitor_count(self) -> int:
        """Get total number of unique visitors."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM visitors')
                return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Failed to get visitor count: {e}")
            return 0
    
    def update_daily_stats(self, date: str, total_visitors: int, unique_visitors: int, 
                          total_entries: int, total_exits: int) -> bool:
        """Update daily statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_stats 
                    (date, total_visitors, unique_visitors, total_entries, total_exits)
                    VALUES (?, ?, ?, ?, ?)
                ''', (date, total_visitors, unique_visitors, total_entries, total_exits))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update daily stats: {e}")
            return False
    
    def get_daily_stats(self, date: str) -> Optional[Dict]:
        """Get daily statistics for a specific date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT total_visitors, unique_visitors, total_entries, total_exits
                    FROM daily_stats WHERE date = ?
                ''', (date,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'total_visitors': result[0],
                        'unique_visitors': result[1],
                        'total_entries': result[2],
                        'total_exits': result[3]
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get daily stats: {e}")
            return None
    
    def cleanup_old_events(self, days: int = 30) -> bool:
        """Clean up old events to prevent database bloat."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM events 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
                self.logger.info(f"Cleaned up {deleted_count} old events")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old events: {e}")
            return False 