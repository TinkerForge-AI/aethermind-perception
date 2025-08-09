
"""CLI entrypoint."""
from pathlib import Path
# from reflection.sleep_loop import SleepLoop
import argparse

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("./data"))

    args = parser.parse_args()
    # sl = SleepLoop(args.store)
    # sl.process_session(args.video, args.audio)
    print("Sleep loop completed.")

if __name__ == "__main__":
    cli()
