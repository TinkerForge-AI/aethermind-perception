import numpy as np
import wave
import contextlib
import cv2
from pathlib import Path
import json

def compute_video_motion(video_path):
    """
    Return a simple motion score: mean frame-to-frame difference.
    - For each pair of consecutive frames, compute the mean absolute pixel difference (grayscale).
    - The final score is the mean of all these frame-to-frame differences.
    - Higher values indicate more motion.
    """
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
    """
    Return RMS (root mean square) energy of the audio chunk.
    - Reads all audio samples as int16.
    - Computes sqrt(mean(sample^2)) over the chunk.
    - Higher values indicate louder audio.
    """
    with contextlib.closing(wave.open(str(audio_path),'r')) as wf:
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        return float(np.sqrt(np.mean(data**2))) if data.size else 0.0

def detect_events(session_dir, output_path=None):
    """
    Event detection pipeline:
    - Loads session.json to get all chunk metadata.
    - For each chunk:
        - Computes video motion (see compute_video_motion).
        - Computes audio energy (see compute_audio_energy).
    - Normalizes motion and energy across all chunks to [0, 1] range.
    - Calculates 'event_score' as a weighted sum: 0.6*motion + 0.4*energy.
    - Sets 'is_event' True if event_score > 0.3 (example threshold).
    - Writes results to session_events.json.
    """
    session_dir = Path(session_dir)
    manifest = json.loads((session_dir / "session.json").read_text())
    events = []
    motion_vals, energy_vals = [], []

    # First pass: gather raw metrics for each chunk
    for c in manifest["chunks"]:
        mp = session_dir / Path(c["video_path"]).name
        ap = session_dir / Path(c["audio_path"]).name
        # Compute mean frame-to-frame motion for this chunk
        m = compute_video_motion(mp)
        # Compute RMS audio energy for this chunk
        e = compute_audio_energy(ap)
        motion_vals.append(m)
        energy_vals.append(e)
        # Store raw metrics in the event record
        events.append({**c, "raw_motion": m, "raw_energy": e})

    # Normalize and score
    m_min, m_max = min(motion_vals), max(motion_vals)
    e_min, e_max = min(energy_vals), max(energy_vals)
    for ev in events:
        # Normalize motion and energy to [0, 1] across all chunks
        m = (ev["raw_motion"] - m_min) / (m_max - m_min + 1e-8)
        e = (ev["raw_energy"] - e_min) / (e_max - e_min + 1e-8)
        # Weighted event score: 60% motion, 40% energy
        ev["event_score"] = float(0.6*m + 0.4*e)
        # Mark as event if score exceeds threshold (example: 0.3)
        ev["is_event"] = ev["event_score"] > 0.3

    # Output results to session_events.json
    out = output_path or (session_dir / "session_events.json")
    (session_dir / out.name).write_text(json.dumps(events, indent=2))
    return events
