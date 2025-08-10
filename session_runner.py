import subprocess

def run_full_pipeline(session_folder):
    # 1. Chunking
    subprocess.run(["python3", "chunker.py", "--folder", session_folder], check=True)
    # 2. Event Detection
    subprocess.run(["python3", "event_detector.py", session_folder], check=True)
    # 3. Vectorization
    subprocess.run(["python3", "stream_to_vectors.py", "--folder", session_folder], check=True)
    # 4. Merge vectors into events
    subprocess.run([
        "python3", "add_vectors_to_events.py",
        "--chunks", f"{session_folder}/session_events.json",
        "--vectors", f"{session_folder}/vector_windows.jsonl",
        "--output", f"{session_folder}/session_events_with_vectors.json"
    ], check=True)

# Usage:
# run_full_pipeline("path/to/session_folder")