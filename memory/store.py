
"""Long‑term memory storage and retrieval."""
from pathlib import Path
from typing import List
import numpy as np
import faiss
import json

class MemoryStore:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatL2(512)
        self.metadata = []

    def add(self, embedding: np.ndarray, meta: dict):
        self.index.add(embedding.reshape(1, -1))
        self.metadata.append(meta)

    def search(self, query: np.ndarray, k: int = 5) -> List[dict]:
        D, I = self.index.search(query.reshape(1, -1), k)
        return [self.metadata[i] for i in I[0] if i < len(self.metadata)]
