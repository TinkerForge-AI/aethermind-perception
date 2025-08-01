
"""Retrieve memories for reflection."""
import random
from typing import List
from memory.store import MemoryStore

class ReplayEngine:
    def __init__(self, store: MemoryStore):
        self.store = store

    def sample(self, k: int = 5) -> List[dict]:
        return random.sample(self.store.metadata, k=min(k, len(self.store.metadata)))
