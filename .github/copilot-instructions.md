# Copilot Instructions for Aethermind Perception

## Project Overview
Aethermind Perception is a cognitive pipeline for processing multimodal session data (video, audio, actions, thoughts) from an embodied agent. The system segments experiences, scores valence, compresses memory, and prepares unified outputs for downstream cognitive modules.

## Architecture & Data Flow
- **Major Components:**
  - `chunker.py`: Segments session data into temporal chunks, aligns actions, ensures all timestamps are absolute epoch times.
  - `event_detector.py`: Detects events within chunks, outputs event metadata.
  - `stream_to_vectors.py`: Vectorizes video/audio/actions per window, outputs vectors with absolute epoch timestamps.
  - `add_vectors_to_events.py`: Merges vectors into chunk/event objects based on time windows; supports CLI usage.
  - `session_runner.py`: Orchestrates the pipeline, running chunking, event detection, vectorization, and merging in sequence.
- **Data Files:**
  - `session_events.json`: Contains chunk/event data with absolute timestamps.
  - `vector_windows.jsonl`: Contains vector data with `t` (timestamp).
  - Output: Unified JSON for downstream interpretation.

## Developer Workflows
- **Run Full Pipeline:**
  ```bash
  python3 session_runner.py path/to/session_folder
  ```
- **Manual Merging (CLI):**
  ```bash
  python add_vectors_to_events.py --chunks session_events.json --vectors vector_windows.jsonl --output session_events_with_vectors.json
  ```
- **Testing:**
  - Tests are in `tests/` (e.g., `test_event_detector.py`).
  - Run with pytest or unittest as appropriate.

## Project-Specific Conventions
- **Timestamps:** All chunk, event, and vector timestamps are absolute epoch times (float, seconds since epoch).
- **Chunk/Event Structure:**
  - Chunks/events are lists or dicts with `start`, `end`, and may include `actions`, `vectors`, etc.
  - Scripts must handle both list and dict session structures.
- **Deduplication:** Chunker ensures no duplicate actions per chunk.
- **CLI Support:** Key scripts (`add_vectors_to_events.py`) use argparse for flexible input/output paths.

## Integration Points & Dependencies
- **External:**
  - Video: OpenCV
  - Audio: soundfile
  - Vectors: numpy
- **Session Data:**
  - Input sessions are folders with chunked media and metadata.
  - Output is a unified JSON for downstream cognitive modules.

## Patterns & Examples
- **Chunk Alignment:** See `chunker.py` for action alignment and deduplication logic.
- **Vector Merging:** See `add_vectors_to_events.py` for merging vectors into chunk objects by time window.
- **Pipeline Orchestration:** See `session_runner.py` for end-to-end automation.

## Key Files & Directories
- `chunker.py`, `event_detector.py`, `stream_to_vectors.py`, `add_vectors_to_events.py`, `session_runner.py`
- `sessions/`: Contains session folders with chunked media and metadata.
- `tests/`: Contains unit tests for core modules.

---
_If any section is unclear or incomplete, please provide feedback for iterative improvement._
