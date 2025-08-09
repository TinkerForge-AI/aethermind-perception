import re
import numpy as np

# Import the pipeline and modules under test
from aethermind_perception.aethermind_pipeline import process_event
from aethermind_perception.event_seed_emitter import emit_event_seed

# ---------- helpers ----------

def _monotonic_fixed(val=100.0):
    # Return a function to monkeypatch monotonic_s in event_seed_emitter
    def _fn():
        return float(val)
    return _fn

# ---------- tests ----------

def test_emit_event_seed_monotonic_and_uid(monkeypatch, tmp_path):
    # Monkeypatch monotonic_s so start/end are deterministic
    import aethermind_perception.event_seed_emitter as ese
    monkeypatch.setattr(ese, "monotonic_s", _monotonic_fixed(100.0))

    session_id = "session_20250805_162657"
    video_path = "/some/path/chunks/session_video.mp4"
    audio_path = "/some/path/audio.wav"

    actions = []
    sync = {"av_ms": 0.0, "ai_ms": 0.0, "vi_ms": 0.0, "beacon_id": None, "estimated": False}
    video_dyn = {"flow_mean": 0.0, "flow_std": 0.0, "cut_prob": 0.0, "frame_idx_start": 0, "frame_idx_end": 0}
    audio_dyn = {"rms_frames": [0.0, 0.0], "sr": 16000}
    system = {"fps": 60, "dropped_frames": 0, "window_focused": True, "resolution": [1920,1080], "codec":"h264", "crf": None}

    seed = emit_event_seed(
        session_id=session_id,
        video_path=video_path,
        audio_path=audio_path,
        actions=actions,
        sync=sync,
        video_dyn=video_dyn,
        audio_dyn=audio_dyn,
        system=system,
        duration_s=2.0,
    )

    # 1) monotonic timestamps
    assert seed["start"] == 100.0
    assert seed["end"] == 102.0
    assert seed["end"] >= seed["start"]

    # 2) UID format: session|basename|start-end with 3 decimals
    uid = seed["event_uid"]
    assert "session_20250805_162657|session_video.mp4|100.000-102.000" == uid

    # 3) required sync keys present
    for k in ("av_ms","ai_ms","vi_ms","beacon_id","estimated"):
        assert k in seed["sync"]

    # 4) system contains fps
    assert "fps" in seed["system"]


def test_process_actions_mouse_norm_and_semantics():
    from aethermind_perception.input_semantics_mapper import process_actions

    # Raw actions with multiple keys and mouse buttons; out-of-bounds mouse gets clipped to [0,1]
    raw_actions = [
        {"ts": 1.0, "keys": ["Q","W"], "mouse":{"position":[2500,-100], "buttons":{"left": False, "right": False, "middle": False}}},
        {"ts": 1.1, "keys": [], "mouse":{"position":[960,540], "buttons":{"left": True, "right": False, "middle": False}}},
    ]
    processed = process_actions(raw_actions, resolution=(1920,1080))

    # 1) first action: keys include W -> maps to move_forward; mouse_norm clipped to [0,1]
    a0 = processed[0]["semantic"]
    mx0, my0 = a0["mouse_norm"]
    assert a0["action"] == "move_forward"
    assert a0["valid_for_game"] is True
    assert 0.0 <= mx0 <= 1.0
    assert 0.0 <= my0 <= 1.0

    # 2) second action: no keys, left mouse down -> click_primary
    a1 = processed[1]["semantic"]
    assert a1["action"] == "click_primary"
    assert a1["valid_for_game"] is True
    mx1, my1 = a1["mouse_norm"]
    assert 0.0 <= mx1 <= 1.0
    assert 0.0 <= my1 <= 1.0


def test_pipeline_process_event_end_to_end(monkeypatch):
    # Monkeypatch monotonic so the UID is deterministic
    import aethermind_perception.event_seed_emitter as ese
    monkeypatch.setattr(ese, "monotonic_s", _monotonic_fixed(200.0))

    # Monkeypatch audio RMS to avoid filesystem I/O
    import aethermind_perception.dynamics_computation as dyn
    monkeypatch.setattr(dyn, "compute_audio_rms", lambda path: {"rms_frames":[0.1,0.2,0.3], "sr":16000})

    session_id = "session_20250805_162657"
    audio_path = "fake.wav"
    video_path = "chunks/session_video.mp4"

    # Tiny synthetic video: 5 black frames (no motion, but function should handle)
    video_frames = [np.zeros((480,640,3), dtype=np.uint8) for _ in range(5)]

    # Mix of keys and mouse
    raw_actions = [
        {"ts": 0.0, "keys":["I"], "mouse":{"position":[0,0], "buttons":{"left": False, "right": False, "middle": False}}},
        {"ts": 0.2, "keys":[], "mouse":{"position":[640,480], "buttons":{"left": True, "right": False, "middle": False}}},
    ]

    out = process_event(
        session_id=session_id,
        video_frames=video_frames,
        audio_path=audio_path,
        raw_actions=raw_actions,
        video_path=video_path,
        resolution=(640,480),
    )

    # Basic shape checks
    assert out["event_uid"] == "session_20250805_162657|session_video.mp4|200.000-202.000"
    assert out["source"] == "perception"
    assert "actions" in out and len(out["actions"]) == 2
    assert "sync" in out and all(k in out["sync"] for k in ("av_ms","ai_ms","vi_ms"))
    assert "video_dyn" in out and "flow_mean" in out["video_dyn"]
    assert "audio_dyn" in out and "rms_frames" in out["audio_dyn"]
    assert "system" in out and "fps" in out["system"]

    # mouse_norm in [0,1]
    for a in out["actions"]:
        mx, my = a["semantic"]["mouse_norm"]
        assert 0.0 <= mx <= 1.0
        assert 0.0 <= my <= 1.0
