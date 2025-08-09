# event_seed_emitter.py
import os, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = {"major": 1, "minor": 0}

def monotonic_s() -> float:
    return time.monotonic_ns() / 1_000_000_000.0

def wall_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time_ns()%1_000_000_000):09d}Z"

def media_key(path: str) -> str:
    return os.path.basename(path)

def make_event_uid(session_id: str, video_path: str, start: float, end: float) -> str:
    return f"{session_id}|{media_key(video_path)}|{start:.3f}-{end:.3f}"

@dataclass
class EventSeed:
    event_uid: str
    session_id: str
    schema_version: Dict[str, int]
    created_at: str
    start: float
    end: float
    source: str
    video_path: str
    audio_path: str
    actions: List[Dict[str, Any]]
    sync: Dict[str, Any]
    video_dyn: Dict[str, Any]
    audio_dyn: Dict[str, Any]
    system: Dict[str, Any]
    action_window: Dict[str, int]
    decision_trace: Optional[Dict[str, Any]] = None
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

def emit_event_seed(session_id, video_path, audio_path, actions, sync, video_dyn, audio_dyn, system,
                    pre_ms=300, post_ms=500, decision_trace=None, duration_s=2.0):
    start = monotonic_s(); end = start + float(duration_s)
    seed = EventSeed(
        event_uid=make_event_uid(session_id, video_path, start, end),
        session_id=session_id, schema_version=SCHEMA_VERSION, created_at=wall_iso(),
        start=start, end=end, source="perception",
        video_path=video_path, audio_path=audio_path,
        actions=actions, sync=sync, video_dyn=video_dyn, audio_dyn=audio_dyn,
        system=system, action_window={"pre_ms": int(pre_ms), "post_ms": int(post_ms)},
        decision_trace=decision_trace,
    )
    _validate(seed); return seed.to_dict()

def _validate(seed: EventSeed) -> None:
    assert seed.end >= seed.start, "end < start"
    for k in ("av_ms","ai_ms","vi_ms"): assert k in seed.sync, f"sync missing {k}"
    assert "fps" in seed.system, "system.fps missing"
