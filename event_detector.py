import numpy as np
import wave
import contextlib
import cv2
from pathlib import Path
import json

def compute_video_motion(video_path):
    """Return a simple motion score: mean frame-to-frame difference."""
    cap = cv2.VideoCapture(str(video_path))
    ret, prev = cap.read()
    diffs = []
    while ret:
        ret, frame = cap.read()
        if not ret: break
        diff = cv2.absdiff(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        diffs.append(np.mean(diff))
        prev = frame
    cap.release()
    return float(np.mean(diffs)) if diffs else 0.0

def compute_audio_energy(audio_path):
    """Return RMS energy of the audio chunk."""
    with contextlib.closing(wave.open(str(audio_path),'r')) as wf:
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        return float(np.sqrt(np.mean(data**2))) if data.size else 0.0

def detect_events(session_dir, output_path=None):
    """
    - Load sessions/.../session.json to find all chunks
    - For each chunk, compute motion & energy
    - Add an 'event_score' = normalized combination
    - Write out sessions/.../session_events.json
    """
    session_dir = Path(session_dir)
    manifest = json.loads((session_dir / "session.json").read_text())
    events = []
    motion_vals, energy_vals = [], []

    # First pass: gather raw metrics
    for c in manifest["chunks"]:
        mp = session_dir / Path(c["video_path"]).name
        ap = session_dir / Path(c["audio_path"]).name
        m = compute_video_motion(mp)
        e = compute_audio_energy(ap)
        motion_vals.append(m)
        energy_vals.append(e)
        events.append({**c, "raw_motion": m, "raw_energy": e})

    # Normalize and score
    m_min, m_max = min(motion_vals), max(motion_vals)
    e_min, e_max = min(energy_vals), max(energy_vals)
    for ev in events:
        m = (ev["raw_motion"] - m_min) / (m_max - m_min + 1e-8)
        e = (ev["raw_energy"] - e_min) / (e_max - e_min + 1e-8)
        ev["event_score"] = float(0.6*m + 0.4*e)
        ev["is_event"] = ev["event_score"] > 0.3  # threshold as example

    # Output
    out = output_path or (session_dir / "session_events.json")
    (session_dir / out.name).write_text(json.dumps(events, indent=2))
    return events
