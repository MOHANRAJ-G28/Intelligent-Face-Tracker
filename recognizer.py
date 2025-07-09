import cv2
import numpy as np
import logging
import uuid
from typing import Dict, List, Optional, Tuple
from insightface.app import FaceAnalysis
import onnxruntime

class FaceRecognizer:
    """Face recognition using InsightFace (ArcFace)."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = config['recognition']['similarity_threshold']
        self.embedding_size = config['recognition']['embedding_size']
        self.model_name = config['recognition']['model_name']
        self.device = config['recognition']['device']
        
        # Initialize InsightFace app
        self.app = None
        self._load_model()
        
        # In-memory cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.face_id_counter = 0
    
    def _load_model(self):
        """Load InsightFace model for face recognition."""
        try:
            # Configure InsightFace app
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            )
            
            # Prepare the model
            self.app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
            
            self.logger.info(f"Loaded InsightFace model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load InsightFace model: {e}")
            raise
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from face image.
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Face embedding vector or None if generation fails
        """
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get face embedding
            faces = self.app.get(face_rgb)
            
            if len(faces) > 0:
                # Get the first (and should be only) face
                face = faces[0]
                embedding = face.embedding
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
            else:
                self.logger.warning("No face detected for embedding generation")
                return None
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Ensure embeddings are normalized
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def recognize_face(self, embedding: np.ndarray, known_embeddings: Dict[str, np.ndarray]) -> Tuple[Optional[str], float]:
        """
        Recognize a face by comparing its embedding with known embeddings.
        
        Args:
            embedding: Face embedding to recognize
            known_embeddings: Dictionary of known face embeddings {face_id: embedding}
            
        Returns:
            Tuple of (face_id, similarity_score) or (None, 0.0) if no match
        """
        try:
            best_match_id = None
            best_similarity = 0.0
            
            for face_id, known_embedding in known_embeddings.items():
                similarity = self.calculate_similarity(embedding, known_embedding)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = face_id
            
            return best_match_id, best_similarity
            
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def add_face_to_cache(self, face_id: str, embedding: np.ndarray):
        """Add face embedding to in-memory cache."""
        try:
            self.embedding_cache[face_id] = embedding
            self.logger.debug(f"Added face {face_id} to cache")
        except Exception as e:
            self.logger.error(f"Failed to add face to cache: {e}")
    
    def remove_face_from_cache(self, face_id: str):
        """Remove face embedding from in-memory cache."""
        try:
            if face_id in self.embedding_cache:
                del self.embedding_cache[face_id]
                self.logger.debug(f"Removed face {face_id} from cache")
        except Exception as e:
            self.logger.error(f"Failed to remove face from cache: {e}")
    
    def get_face_from_cache(self, face_id: str) -> Optional[np.ndarray]:
        """Get face embedding from in-memory cache."""
        return self.embedding_cache.get(face_id)
    
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
            "model": f"InsightFace-{self.model_name}",
            "embedding_size": self.embedding_size,
            "similarity_threshold": self.similarity_threshold,
            "device": self.device,
            "cached_faces": len(self.embedding_cache)
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self.embedding_cache) 