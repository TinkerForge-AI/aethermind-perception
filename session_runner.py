
import argparse
import time
from chunker import chunk_video_audio
from logger import SessionLogger

def run_session(video_path, audio_path, chunk_duration=2.0):
    print("Starting Aethermind Perception Session...")
    logger = SessionLogger()

    # Call your chunker and get back a list of chunk info
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a perception session.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--chunk_duration", type=float, default=2.0, help="Duration of each chunk in seconds")

    args = parser.parse_args()

    run_session(
        video_path=args.video,
        audio_path=args.audio,
        chunk_duration=args.chunk_duration
    )