import cv2
import numpy as np
import logging
import uuid
from typing import Dict, List, Optional, Tuple
import hashlib

class SimpleFaceRecognizer:
    """Simplified face recognition using basic image hashing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = config['recognition']['similarity_threshold']
        
        # In-memory cache for face hashes
        self.face_cache: Dict[str, str] = {}
        self.face_id_counter = 0
    
    def generate_face_hash(self, face_image: np.ndarray) -> str:
        """
        Generate a simple hash for face image using average pixel values.
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Face hash string
        """
        try:
            # Resize to standard size
            resized = cv2.resize(face_image, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate average pixel value
            avg_pixel = float(np.mean(gray))
            
            # Create hash based on pixel distribution
            hash_values = []
            for i in range(0, 64, 8):
                for j in range(0, 64, 8):
                    block = gray[i:i+8, j:j+8]
                    hash_values.append(float(np.mean(block)))
            
            # Convert to hash string
            hash_str = hashlib.md5(str(hash_values).encode()).hexdigest()
            
            return hash_str
            
        except Exception as e:
            self.logger.error(f"Face hash generation failed: {e}")
            return ""
    
    def calculate_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity between two face hashes.
        
        Args:
            hash1: First face hash
            hash2: Second face hash
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            if not hash1 or not hash2:
                return 0.0
            
            # Simple hash comparison
            if hash1 == hash2:
                return 1.0
            
            # Calculate Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            max_distance = len(hash1)
            similarity = 1.0 - (distance / max_distance)
            
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def recognize_face(self, face_hash: str, known_hashes: Dict[str, str]) -> Tuple[Optional[str], float]:
        """
        Recognize a face by comparing its hash with known hashes.
        
        Args:
            face_hash: Face hash to recognize
            known_hashes: Dictionary of known face hashes {face_id: hash}
            
        Returns:
            Tuple of (face_id, similarity_score) or (None, 0.0) if no match
        """
        try:
            best_match_id = None
            best_similarity = 0.0
            
            for face_id, known_hash in known_hashes.items():
                similarity = self.calculate_similarity(face_hash, known_hash)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = face_id
            
            return best_match_id, best_similarity
            
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def add_face_to_cache(self, face_id: str, face_hash: str):
        """Add face hash to in-memory cache."""
        try:
            self.face_cache[face_id] = face_hash
            self.logger.debug(f"Added face {face_id} to cache")
        except Exception as e:
            self.logger.error(f"Failed to add face to cache: {e}")
    
    def remove_face_from_cache(self, face_id: str):
        """Remove face hash from in-memory cache."""
        try:
            if face_id in self.face_cache:
                del self.face_cache[face_id]
                self.logger.debug(f"Removed face {face_id} from cache")
        except Exception as e:
            self.logger.error(f"Failed to remove face from cache: {e}")
    
    def get_face_from_cache(self, face_id: str) -> Optional[str]:
        """Get face hash from in-memory cache."""
        return self.face_cache.get(face_id)
    
    def generate_face_id(self) -> str:
        """Generate a unique face ID."""
        self.face_id_counter += 1
        return f"face_{self.face_id_counter}_{uuid.uuid4().hex[:8]}"
    
    def preprocess_face(self, face_image: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """
        Preprocess face image for recognition.
        
        Args:
            face_image: Input face image
            target_size: Target size for preprocessing
            
        Returns:
            Preprocessed face image or None if preprocessing fails
        """
        try:
            # Resize to target size
            resized = cv2.resize(face_image, target_size)
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Face preprocessing failed: {e}")
            return None
    
    def get_recognition_stats(self) -> dict:
        """Get recognition statistics."""
        return {
            "model": "SimpleHash",
            "similarity_threshold": self.similarity_threshold,
            "cached_faces": len(self.face_cache)
        }
    
    def clear_cache(self):
        """Clear the face cache."""
        self.face_cache.clear()
        self.logger.info("Face cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the number of cached faces."""
        return len(self.face_cache) 