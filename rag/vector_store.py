"""
rag/vector_store.py
-------------------
Lightweight in-memory vector store using TF-IDF cosine similarity.
No external vector database is required — everything runs locally.

⚠️  DEMO PROJECT — IBM AICTE Internship 2026
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Chunk:
    """A single text chunk with its source file name."""

    def __init__(self, text: str, source: str, chunk_id: int) -> None:
        self.text = text
        self.source = source          # e.g. "knowledge_base/fees.txt"
        self.chunk_id = chunk_id

    def __repr__(self) -> str:  # pragma: no cover
        return f"Chunk(id={self.chunk_id}, source={self.source!r}, len={len(self.text)})"


class TFIDFVectorStore:
    """
    Stores document chunks as TF-IDF vectors and returns the top-k most
    similar chunks for a given query.
    """

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None           # sparse matrix: (n_chunks, n_features)
        self._is_built: bool = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add a list of Chunk objects to the store."""
        self._chunks.extend(chunks)
        self._is_built = False        # invalidate the index

    def build(self) -> None:
        """Fit the TF-IDF vectorizer on all stored chunks."""
        if not self._chunks:
            raise ValueError("No chunks have been added to the vector store.")
        texts = [c.text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),       # unigrams + bigrams for richer matching
            max_features=20_000,
            sublinear_tf=True,        # apply log normalization to TF
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        self._is_built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search_all(self, query: str) -> List[Tuple[Chunk, float]]:
        """
        Return all chunks with non-zero similarity scores.
        Used by the retriever for topic-based source prioritization.
        """
        if not self._is_built:
            raise RuntimeError(
                "Vector store is not built yet. Call build() first."
            )

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(
            query_vec,
            self._matrix
        ).flatten()

        results: List[Tuple[Chunk, float]] = []

        for idx, score in enumerate(scores):
            score = float(score)

            if score > 0.0:
                results.append(
                    (self._chunks[idx], score)
                )

        return results


