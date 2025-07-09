import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from ultralytics import YOLO
import torch

class FaceDetector:
    """Face detection using YOLOv8."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = config['detection']['device']
        self.confidence_threshold = config['detection']['confidence_threshold']
        self.nms_threshold = config['detection']['nms_threshold']
        self.model_size = config['detection']['model_size']
        
        # Initialize YOLO model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 face detection model."""
        try:
            # Use YOLOv8n-face model for face detection
            model_name = f"yolov8{self.model_size}-face.pt"
            
            # Try to load from local path first, then download if not available
            try:
                self.model = YOLO(model_name)
                self.logger.info(f"Loaded local model: {model_name}")
            except:
                # Download and load the model
                self.model = YOLO(f"keremberke/yolov8{self.model_size}-face-detection")
                self.logger.info(f"Downloaded and loaded model: {model_name}")
            
            # Set device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")
                self.logger.info("Using CUDA for face detection")
            else:
                self.model.to("cpu")
                self.logger.info("Using CPU for face detection")
                
        except Exception as e:
            self.logger.error(f"Failed to load face detection model: {e}")
            raise
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of tuples (x1, y1, x2, y2, confidence) for detected faces
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            if len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Get coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        detections.append((x1, y1, x2, y2, confidence))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_face_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face region from frame using bounding box.
        
        Args:
            frame: Input image frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped face image or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Extract face region
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size > 0:
                return face_region
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Face extraction failed: {e}")
            return None
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw detection boxes on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detections (x1, y1, x2, y2, confidence)
            
        Returns:
            Frame with detection boxes drawn
        """
        frame_copy = frame.copy()
        
        for x1, y1, x2, y2, confidence in detections:
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy
    
    def get_detection_stats(self) -> dict:
        """Get detection statistics."""
        return {
            "model": f"YOLOv8{self.model_size}-face",
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold
        } 