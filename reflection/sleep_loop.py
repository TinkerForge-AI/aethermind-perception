
"""Coordinates nightly reflection."""
from pathlib import Path
from datetime import datetime
from memory.chunker import Chunker
from memory.encoder import Encoder
from memory.valence_scorer import ValenceScorer
from memory.store import MemoryStore

class SleepLoop:
    def __init__(self, store_dir: Path):
        store_dir.mkdir(parents=True, exist_ok=True)
        self.store = MemoryStore(store_dir / "index.faiss")
        self.chunker = Chunker()
        self.encoder = Encoder()
        self.scorer = ValenceScorer()

    def process_session(self, video: Path, audio: Path):
        for chunk in self.chunker.chunk_session(video, audio):
            emb = self.encoder.encode(chunk.video_path, chunk.audio_path)
            val = int(self.scorer.predict(emb))
            meta = {
                "start": chunk.start_time,
                "end": chunk.end_time,
                "valence": val,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.store.add(emb, meta)
