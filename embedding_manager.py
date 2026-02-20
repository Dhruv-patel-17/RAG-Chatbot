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

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Supports batching for large number of chunks.
        Returns a numpy array of shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Embedding model not loaded")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"⚠️ Failed to embed batch {i}-{i+len(batch_texts)}: {e}")
                # fallback: zero embeddings
                dim = self.model.get_sentence_embedding_dimension()
                all_embeddings.extend([np.zeros(dim)] * len(batch_texts))

        print(f"Generated embeddings for {len(all_embeddings)} texts")
        return np.array(all_embeddings)
