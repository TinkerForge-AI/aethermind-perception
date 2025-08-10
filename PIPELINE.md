# Aethermind Perception Pipeline: Successful Run Documentation

## Overview
This document describes the successful execution of the Aethermind Perception pipeline for multimodal session data processing. The pipeline segments, annotates, vectorizes, and merges session data into a unified output for downstream cognitive modules.

---

## Pipeline Steps

### 1. Chunking
- **Script:** `chunker.py`
- **Function:** Segments session data (video, audio, actions) into temporal chunks.
- **Details:**
  - All chunk `start` and `end` times are absolute epoch seconds.
  - Actions are aligned and deduplicated per chunk.
  - Output: `chunks.json` in the session output folder.

### 2. Event Detection
- **Script:** `event_detector.py`
- **Function:** Detects and scores events within each chunk.
- **Details:**
  - Annotates chunks with event metadata (valence, scores, etc.).
  - Output: `session_events.json` with enriched chunk/event objects.

### 3. Vectorization
- **Script:** `stream_to_vectors.py`
- **Function:** Computes vector embeddings for each chunk window.
- **Details:**
  - Vectors are generated for video, audio, and actions.
  - Each vector entry includes a `t` field (absolute epoch timestamp).
  - Output: `vector_windows.jsonl` in the session output folder.

### 4. Merging Vectors
- **Script:** `add_vectors_to_events.py`
- **Function:** Merges vectors into corresponding chunk/event objects.
- **Details:**
  - For each chunk, vectors with `t` in `[start, end)` are added to a `vectors` array.
  - Output: `session_events_with_vectors.json` (unified event+vector data).

### 5. Orchestration
- **Script:** `session_runner.py`
- **Function:** Runs the full pipeline in sequence.
- **Usage:**
  ```bash
  python3 session_runner.py path/to/session_folder