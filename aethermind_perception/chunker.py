import json
import subprocess
from pathlib import Path
from moviepy import VideoFileClip
import cv2
from aethermind_pipeline import process_event
from input_semantics_mapper import process_actions
from sync_and_health import capture_health_metrics

# python3 chunker.py --folder 
ACTION_LOG = "aethermind-input/data/SESSION_NAME/actions.jsonl"

def read_video_frames_sparse(path: str, max_frames: int = 12):
    """
    Read up to max_frames, spaced across the clip. Returns (frames, (width,height), fps).
    Frames are BGR uint8 for OpenCV-based flow.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], (1920, 1080), 30  # sensible defaults
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    if total <= 0:
        cap.release()
        return [], (w, h), fps

    # pick indices roughly evenly spaced
    take = min(max_frames, total)
    idxs = [int(i * (total - 1) / max(1, take - 1)) for i in range(take)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames, (w, h), fps

def load_actions(action_log_path: str, video_start_ts: float = None):
    """
    Load actions from JSONL, convert epoch‐ms timestamps to seconds
    relative to the start of the video.
    If video_start_ts is provided, subtract it (in seconds). 
    """
    actions = []
    with open(action_log_path, "r") as f:
        for line in f:
            raw = json.loads(line)
            # raw["time"] is epoch seconds with fraction
            ts = raw["time"]
            if video_start_ts is not None:
                ts = ts - video_start_ts
            entry = {
                "ts": ts,
                "keys": raw.get("keys", []),
                "mouse": raw.get("mouse", {}),
            }
            actions.append(entry)
    return actions

def split_media_ffmpeg(input_path: Path, start: float, end: float, out_path: Path, is_video: bool):
    """
    Use ffmpeg to slice video or audio.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", str(start),
        "-to", str(end),
    ]
    if is_video:
        cmd += ["-c", "copy", "-an", str(out_path)]
    else:
        cmd += ["-c:a", "pcm_s16le", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def deduplicate_actions(actions):
    """
    Collapse consecutive identical actions (ignoring 'ts') into a single entry.
    Keeps the first occurrence of each run of identical actions.
    """
    if not actions:
        return []
    def action_state(a):
        # Return a tuple of the action state, excluding 'ts'
        return (
            tuple(a.get("keys", [])),
            tuple(a.get("mouse", {}).get("position", [])),
            tuple(sorted(a.get("mouse", {}).get("buttons", {}).items())),
            tuple(a.get("mouse", {}).get("scroll", []))
        )
    deduped = [actions[0]]
    last_state = action_state(actions[0])
    for action in actions[1:]:
        if action_state(action) != last_state:
            deduped.append(action)
            last_state = action_state(action)
    return deduped

def chunk_video_audio_with_actions(
    video_path: str,
    audio_path: str,
    action_log_path: str = ACTION_LOG,
    chunk_duration: float = 2.0,
    output_dir: str = "chunks",
    video_start_ts: float = None
):
    """
    Splits video/audio into synchronized chunks and bundles in-window actions.
    Returns list of chunk dicts with actions.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load and normalize action timestamps
    actions = load_actions(action_log_path, video_start_ts)
    print(f"[DEBUG] Loaded {len(actions)} actions from {action_log_path}")
    if actions:
        print(f"[DEBUG] First 5 action timestamps: {[a['ts'] for a in actions[:5]]}")
        print(f"[DEBUG] Last 5 action timestamps: {[a['ts'] for a in actions[-5:]]}")

    video = VideoFileClip(str(video_path))
    total_dur = video.duration


    chunks = []
    idx = 0
    # If video_start_ts is provided, chunk times are absolute epoch seconds
    if video_start_ts is not None:
        abs_start = video_start_ts
    else:
        abs_start = 0.0
    abs_end = abs_start + total_dur
    cur_start = abs_start
    while cur_start < abs_end:
        cur_end = min(cur_start + chunk_duration, abs_end)
        base = f"chunk_{idx:04d}"
        vid_out = out_dir / f"{base}.mp4"
        aud_out = out_dir / f"{base}.wav"

        # 1) write video chunk (no audio)
        rel_start = cur_start - abs_start
        rel_end = cur_end - abs_start
        video.subclipped(rel_start, rel_end).write_videofile(
            str(vid_out), codec="libx264", audio=False, logger=None
        )

        # 2) extract audio chunk
        split_media_ffmpeg(Path(audio_path), rel_start, rel_end, aud_out, is_video=False)

        # 3) collect actions within [cur_start, cur_end)
        window_actions = [
            a for a in actions
            if cur_start <= (a["ts"] + video_start_ts) < cur_end
        ]

        window_actions = deduplicate_actions(window_actions)

        # Ensure action ts are absolute (your existing logic)
        if video_start_ts is not None:
            for action in window_actions:
                action["ts"] = action["ts"] + video_start_ts

        # --- NEW: build EventSeed for this chunk ---
        # Derive a session_id from the output session folder name (e.g., "session_20250805_162657")
        session_id = out_dir.name

        # Read sparse frames + get resolution/fps for health metrics & mouse normalization
        frames, (w, h), fps = read_video_frames_sparse(str(vid_out), max_frames=12)

        # Pack health metrics (the pipeline will also compute dynamics; this adds provenance)
        health = capture_health_metrics(fps=int(fps), dropped_frames=0, focused=True, resolution=(w, h), codec="h264", crf=None)

        # Let the pipeline convert raw actions → semantic actions (includes mouse_norm)
        semantic_actions = process_actions(window_actions, resolution=(w, h))

        # Call the pipeline: this computes video_dyn (flow) + audio_dyn (RMS) and emits a seed
        seed = process_event(
            session_id=session_id,
            video_frames=frames,                 # sparse frames are fine for flow stats
            audio_path=str(aud_out),
            raw_actions=semantic_actions,        # already semantic-ized is fine; pipeline tolerates this
            video_path=str(vid_out),
            resolution=(w, h),
        )

        # Persist seeds incrementally (jsonl), and/or collect them for the session manifest
        seeds_path = out_dir / "seeds.jsonl"
        with open(seeds_path, "a") as sf:
            sf.write(json.dumps(seed) + "\n")

        print(f"[DEBUG] Emitted EventSeed → {seed['event_uid']}")

        # --- keep your existing chunk record for downstream tools ---
        chunks.append({
            "start": cur_start,
            "end": cur_end,
            "video_path": str(vid_out),
            "audio_path": str(aud_out),
            "valence": "unknown",
            "source": "perception",
            "annotations": {},
            "raw_motion": None,
            "raw_energy": None,
            "event_score": None,
            "is_event": False,
            "actions": window_actions
        })

        cur_start = cur_end
        idx += 1

    video.close()
    return chunks


if __name__ == "__main__":
    import argparse, json, datetime, shutil, sys, re, time as time_mod

    parser = argparse.ArgumentParser(
        description="Aethermind Perception Pipeline: chunk video/audio/actions and run event detection."
    )
    parser.add_argument("--folder", type=str, help="Path to folder containing input .mp4/.wav (and optionally actions.jsonl and session.json)")
    parser.add_argument("--video", type=str, help="Path to .mp4 (overrides --folder)")
    parser.add_argument("--audio", type=str, help="Path to .wav (overrides --folder)")
    parser.add_argument("--actions", type=str, help="Path to actions.jsonl (overrides --folder)")
    parser.add_argument("--duration", type=float, default=2.0, help="Chunk length (s)")
    parser.add_argument("--out", default="chunks", help="Main output directory")
    parser.add_argument("--video_start_ts", type=float, default=None,
                        help="Epoch seconds of video start (for timestamp normalization)")
    args = parser.parse_args()

    # Determine input files
    session_json = None
    session_start_ts = None
    if args.folder:
        folder = Path(args.folder)
        files = list(folder.iterdir())
        video = args.video or next((str(f) for f in files if f.suffix.lower() == ".mp4"), None)
        audio = args.audio or next((str(f) for f in files if f.suffix.lower() == ".wav"), None)
        actions = args.actions or next((str(f) for f in files if f.name.endswith("actions.jsonl")), None)
        session_json_path = next((f for f in files if f.name == "session.json"), None)
        if session_json_path:
            with open(session_json_path, "r") as sjf:
                session_json = json.load(sjf)
            # Parse start_time string to epoch seconds
            start_time_str = session_json.get("start_time")
            if start_time_str:
                # Example: "2025-08-01_20-12-27"
                dt_match = re.match(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", start_time_str)
                if dt_match:
                    dt_obj = datetime.datetime(
                        int(dt_match.group(1)), int(dt_match.group(2)), int(dt_match.group(3)),
                        int(dt_match.group(4)), int(dt_match.group(5)), int(dt_match.group(6))
                    )
                    session_start_ts = int(dt_obj.replace(tzinfo=datetime.timezone.utc).timestamp())
        if not video or not audio:
            print("[ERROR] Could not find .mp4 and .wav in folder or via arguments.")
            sys.exit(1)
    else:
        video = args.video
        audio = args.audio
        actions = args.actions
        if not video or not audio:
            print("[ERROR] Must provide --video and --audio if --folder is not used.")
            sys.exit(1)

    # Prefer CLI --video_start_ts, else session.json, else None
    video_start_ts = args.video_start_ts if args.video_start_ts is not None else session_start_ts
    # Apply 4-hour offset to correct for UTC/local mismatch if session_start_ts is used
    if video_start_ts is not None and args.video_start_ts is None:
        print("[INFO] Applying +4 hour (14400s) offset to session start time for UTC/local correction.")
        video_start_ts += 14400 # this might need to be adjusted as there's an issue with absolute time between session.json and the actions.jsonl timestamps

    # Create timestamped output subfolder
    dt_str = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / dt_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy input files for provenance
    shutil.copy(video, out_dir / Path(video).name)
    shutil.copy(audio, out_dir / Path(audio).name)
    if actions:
        shutil.copy(actions, out_dir / Path(actions).name)
    if session_json:
        shutil.copy(str(session_json_path), out_dir / Path(session_json_path).name)

    # Use copied files as input
    video_in = str(out_dir / Path(video).name)
    audio_in = str(out_dir / Path(audio).name)
    actions_in = str(out_dir / Path(actions).name) if actions else ACTION_LOG

    # Run chunking
    chunks = chunk_video_audio_with_actions(
        video_in,
        audio_in,
        actions_in,
        args.duration,
        str(out_dir),
        video_start_ts
    )

    # No need to adjust chunk start/end here; already absolute in chunk_video_audio_with_actions

    # Write relative manifest for reference
    rel_path = out_dir / "chunks.json"
    with open(rel_path, "w") as mf:
        json.dump(chunks, mf, indent=2)
    print(f"Generated {len(chunks)} chunks → {rel_path}")

    # Build session.json for event_detector
    session_manifest = {}
    if session_json:
        session_manifest.update(session_json)
    session_manifest['chunks'] = chunks
    session_manifest['seeds_path'] = str(out_dir / "seeds.jsonl")
    session_path = out_dir / "session.json"
    with open(session_path, "w") as sf:
        json.dump(session_manifest, sf, indent=2)
    print(f"Wrote session manifest with {len(chunks)} chunks → {session_path}")

    # Run event detection if available
    try:
        from event_detector import detect_events
        print(f"Running event detection on {out_dir} ...")
        events = detect_events(str(out_dir))
        print(f"Event detection complete. {len(events)} events detected.")
    except ImportError:
        print("[WARN] event_detector not found, skipping event detection.")
