import json
import os
import time
from pathlib import Path

class SessionLogger:
    def __init__(self, session_id=None, output_dir="sessions"):
        if session_id is None:
            session_id = time.strftime("session_%Y%m%d_%H%M%S")
        self.session_id = session_id
        self.output_dir = Path(output_dir) / session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "session.json"

        self.data = {
            "session_id": self.session_id,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "chunks": [],
            "notes": ""
        }

    def log_chunk(self, start, end, video_path, audio_path, valence="unknown", annotations=None):
        chunk_entry = {
            "start": start,
            "end": end,
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "valence": valence,
            "source": "perception",
            "annotations": annotations or {}
        }
        self.data["chunks"].append(chunk_entry)

    def set_notes(self, note_text):
        self.data["notes"] = note_text

    def finalize(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Session log saved to: {self.log_path}")