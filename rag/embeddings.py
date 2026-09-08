# rag/embeddings.py
# Generates dense vector embeddings for text chunks using a local,
# open-source sentence-transformers model.  No API key required.
#
# Model used: 'all-MiniLM-L6-v2'
#   - Small (80 MB), fast, runs on CPU
#   - 384-dimensional embeddings
#   - Good general-purpose semantic similarity
#
# The model is downloaded automatically on first run and cached by
# the sentence-transformers / HuggingFace libraries in ~/.cache.

from typing import List
import numpy as np

# sentence-transformers is imported lazily so the app can load even
# if the library has not yet been installed (error shown at query time).
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False


class EmbeddingModel:
    """Wraps a sentence-transformers model for text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Name of the sentence-transformers model to use.
                        Defaults to 'all-MiniLM-L6-v2'.
        """
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        self.model_name = model_name
        print(f"[EmbeddingModel] Loading model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)
        print(f"[EmbeddingModel] Model loaded successfully.")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            NumPy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit-normalise for cosine similarity via dot product
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a single query string.

        Args:
            query: The query text.

        Returns:
            1-D NumPy array of shape (embedding_dim,).
        """
        return self.embed([query])[0]
