"""The retrieval pipeline: chunk → embed → score → top-k."""

from dataclasses import dataclass

import numpy as np

from .chunking import chunk_text
from .retrieve import cosine_scores, top_k

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional sentence embeddings


@dataclass
class Hit:
    index: int
    score: float
    text: str


class Retriever:
    def __init__(self, text: str, size: int = 100, overlap: int = 40, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer  # imported lazily: slow to load

            model = SentenceTransformer(MODEL_NAME)
        self.model = model
        self.chunks = chunk_text(text, size, overlap)
        # one batched call instead of encoding chunk by chunk
        self.embeddings = np.asarray(self.model.encode(self.chunks))

    def search(self, question: str, k: int = 3) -> list[Hit]:
        query = np.asarray(self.model.encode(question))
        scores = cosine_scores(query, self.embeddings)
        return [Hit(i, float(scores[i]), self.chunks[i]) for i in top_k(scores, k)]
