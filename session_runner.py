

import argparse
import time
import os
from chunker import chunk_video_audio
from logger import SessionLogger

def run_session_from_folder(folder_path, chunk_duration=2.0):
    print(f"Starting Aethermind Perception Session from folder: {folder_path}")
    # Find .mp4 and .wav files
    files = os.listdir(folder_path)
    video_files = [f for f in files if f.lower().endswith('.mp4')]
    audio_files = [f for f in files if f.lower().endswith('.wav')]
    if not video_files or not audio_files:
        raise FileNotFoundError("No .mp4 or .wav files found in the provided folder.")
    video_path = os.path.join(folder_path, video_files[0])
    audio_path = os.path.join(folder_path, audio_files[0])

    logger = SessionLogger()
    chunks = chunk_video_audio(
        video_path=video_path,
        audio_path=audio_path,
        chunk_duration=chunk_duration,
        output_dir=str(logger.output_dir)
    )
    for chunk in chunks:
        logger.log_chunk(
            start=chunk['start'],
            end=chunk['end'],
            video_path=chunk['video_path'],
            audio_path=chunk['audio_path'],
            valence="unknown"
        )
    logger.set_notes("Initial perception run with chunking and logging only.")
    logger.finalize()
    # Run event detection after logging chunks
    from event_detector import detect_events
    print(f"Running event detection on session directory: {logger.output_dir}")
    events = detect_events(logger.output_dir)
    print(f"Event detection complete. {len(events)} events detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a perception session from a folder containing .mp4 and .wav files.")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing input video and audio files")
    parser.add_argument("--chunk_duration", type=float, default=2.0, help="Duration of each chunk in seconds")

    args = parser.parse_args()

    run_session_from_folder(
        folder_path=args.folder,
        chunk_duration=args.chunk_duration
    )