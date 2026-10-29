"""Rank chunks against a query with cosine similarity in NumPy."""

import numpy as np


def cosine_scores(query: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector (d,) and every row of chunks (n, d)."""
    query = query / np.linalg.norm(query)
    chunks = chunks / np.linalg.norm(chunks, axis=1, keepdims=True)
    return chunks @ query  # normalized vectors: cosine similarity is just a dot product


def top_k(scores: np.ndarray, k: int = 3) -> list[int]:
    """Indices of the k highest scores, best first."""
    k = min(k, len(scores))
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]
