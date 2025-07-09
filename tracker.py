import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

class FaceTracker:
    """Face tracking using OpenCV trackers."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.max_disappeared = config['tracking']['max_disappeared']
        self.min_hits = config['tracking']['min_hits']
        self.iou_threshold = config['tracking']['iou_threshold']
        
        # Tracking state
        self.trackers: Dict[str, cv2.Tracker] = {}
        self.tracked_faces: Dict[str, Dict] = {}
        self.disappeared_count: Dict[str, int] = defaultdict(int)
        self.next_face_id = 1
        
        # Statistics
        self.total_tracked = 0
        self.total_lost = 0
    
    def initialize_tracker(self, face_id: str, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> bool:
        """
        Initialize a tracker for a new face.
        
        Args:
            face_id: Unique identifier for the face
            bbox: Bounding box (x, y, width, height)
            frame: Current frame
            
        Returns:
            True if tracker initialization successful, False otherwise
        """
        try:
            # Create OpenCV tracker (using KCF as default)
            tracker = cv2.TrackerKCF_create()
            
            # Initialize tracker with bounding box
            success = tracker.init(frame, bbox)
            
            if success:
                self.trackers[face_id] = tracker
                self.tracked_faces[face_id] = {
                    'bbox': bbox,
                    'last_seen': time.time(),
                    'hits': 1,
                    'total_frames': 1
                }
                self.disappeared_count[face_id] = 0
                self.logger.debug(f"Initialized tracker for face {face_id}")
                return True
            else:
                self.logger.warning(f"Failed to initialize tracker for face {face_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Tracker initialization failed for face {face_id}: {e}")
            return False
    
    def update_trackers(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Update all trackers and match with new detections.
        
        Args:
            frame: Current frame
            detections: List of new detections (x1, y1, x2, y2, confidence)
            
        Returns:
            Dictionary of tracked faces {face_id: bbox}
        """
        try:
            # Update existing trackers
            active_trackers = {}
            
            for face_id, tracker in list(self.trackers.items()):
                success, bbox = tracker.update(frame)
                
                if success:
                    # Convert bbox format from (x, y, w, h) to (x1, y1, x2, y2)
                    x, y, w, h = bbox
                    bbox_xyxy = (int(x), int(y), int(x + w), int(y + h))
                    
                    active_trackers[face_id] = bbox_xyxy
                    self.tracked_faces[face_id]['bbox'] = bbox_xyxy
                    self.tracked_faces[face_id]['last_seen'] = time.time()
                    self.tracked_faces[face_id]['hits'] += 1
                    self.tracked_faces[face_id]['total_frames'] += 1
                    self.disappeared_count[face_id] = 0
                    
                else:
                    # Tracker failed, increment disappeared count
                    self.disappeared_count[face_id] += 1
                    
                    if self.disappeared_count[face_id] > self.max_disappeared:
                        # Remove lost tracker
                        self._remove_tracker(face_id)
                        self.logger.info(f"Removed lost tracker for face {face_id}")
            
            # Match new detections with existing trackers
            matched_detections = self._match_detections_to_trackers(detections, active_trackers)
            
            # Initialize trackers for unmatched detections
            for detection in detections:
                if not any(detection in matched for matched in matched_detections.values()):
                    self._initialize_tracker_for_detection(detection, frame)
            
            return active_trackers
            
        except Exception as e:
            self.logger.error(f"Tracker update failed: {e}")
            return {}
    
    def _match_detections_to_trackers(self, detections: List[Tuple[int, int, int, int, float]], 
                                    active_trackers: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, Tuple[int, int, int, int, float]]:
        """
        Match new detections with existing trackers using IoU.
        
        Args:
            detections: List of new detections
            active_trackers: Dictionary of active trackers
            
        Returns:
            Dictionary of matched detections
        """
        matched = {}
        
        for face_id, tracker_bbox in active_trackers.items():
            best_iou = 0
            best_detection = None
            
            for detection in detections:
                iou = self._calculate_iou(tracker_bbox, detection[:4])
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_detection = detection
            
            if best_detection:
                matched[face_id] = best_detection
        
        return matched
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _initialize_tracker_for_detection(self, detection: Tuple[int, int, int, int, float], frame: np.ndarray):
        """Initialize a new tracker for an unmatched detection."""
        try:
            x1, y1, x2, y2, confidence = detection
            
            # Convert to (x, y, width, height) format
            bbox = (x1, y1, x2 - x1, y2 - y1)
            
            # Generate new face ID
            face_id = f"face_{self.next_face_id}"
            self.next_face_id += 1
            
            # Initialize tracker
            if self.initialize_tracker(face_id, bbox, frame):
                self.total_tracked += 1
                self.logger.info(f"Started tracking new face: {face_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tracker for detection: {e}")
    
    def _remove_tracker(self, face_id: str):
        """Remove a tracker and clean up associated data."""
        try:
            if face_id in self.trackers:
                del self.trackers[face_id]
            if face_id in self.tracked_faces:
                del self.tracked_faces[face_id]
            if face_id in self.disappeared_count:
                del self.disappeared_count[face_id]
            
            self.total_lost += 1
            
        except Exception as e:
            self.logger.error(f"Failed to remove tracker {face_id}: {e}")
    
    def get_tracked_faces(self) -> Dict[str, Dict]:
        """Get information about all tracked faces."""
        return self.tracked_faces.copy()
    
    def get_tracker_count(self) -> int:
        """Get the number of active trackers."""
        return len(self.trackers)
    
    def is_face_tracked(self, face_id: str) -> bool:
        """Check if a face is currently being tracked."""
        return face_id in self.trackers
    
    def get_face_tracking_info(self, face_id: str) -> Optional[Dict]:
        """Get tracking information for a specific face."""
        return self.tracked_faces.get(face_id)
    
    def draw_tracking_info(self, frame: np.ndarray, tracked_faces: Dict[str, Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw tracking information on the frame.
        
        Args:
            frame: Input frame
            tracked_faces: Dictionary of tracked faces
            
        Returns:
            Frame with tracking information drawn
        """
        frame_copy = frame.copy()
        
        for face_id, bbox in tracked_faces.items():
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw face ID
            label = f"ID: {face_id}"
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw tracking info
            if face_id in self.tracked_faces:
                info = self.tracked_faces[face_id]
                hits_text = f"Hits: {info['hits']}"
                cv2.putText(frame_copy, hits_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return frame_copy
    
    def get_tracking_stats(self) -> dict:
        """Get tracking statistics."""
        return {
            "active_trackers": len(self.trackers),
            "total_tracked": self.total_tracked,
            "total_lost": self.total_lost,
            "max_disappeared": self.max_disappeared,
            "min_hits": self.min_hits,
            "iou_threshold": self.iou_threshold
        }
    
    def reset_tracking(self):
        """Reset all tracking state."""
        self.trackers.clear()
        self.tracked_faces.clear()
        self.disappeared_count.clear()
        self.next_face_id = 1
        self.total_tracked = 0
        self.total_lost = 0
        self.logger.info("Tracking state reset") 