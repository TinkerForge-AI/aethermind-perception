
"""Generates internal narration for a memory."""
from memory.valence_scorer import Valence
from datetime import datetime

class ThoughtGenerator:
    def narrate(self, memory_meta: dict) -> str:
        val = Valence(memory_meta["valence"]).name.lower()
        ts = memory_meta.get("timestamp", "unknown time")
        return f"[{ts}] I experienced a {val} moment."
