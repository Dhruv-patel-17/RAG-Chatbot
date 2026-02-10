# embedding_manager.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingManager:
    """Handles document embedding generation"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(
                f"Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Embedding model not loaded")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )
        return embeddings
