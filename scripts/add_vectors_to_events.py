
import argparse
import json

def add_vectors_to_chunks(session_events_path, vector_windows_path, output_path):
    # Load session_events.json
    with open(session_events_path, "r") as f:
        session = json.load(f)

    # Load all vectors from vector_windows.jsonl
    vectors = []
    with open(vector_windows_path, "r") as f:
        for line in f:
            vectors.append(json.loads(line))


    # Support both list and dict session structures
    chunks = session if isinstance(session, list) else session.get("chunks", [])
    for chunk in chunks:
        start = chunk["start"]
        end = chunk["end"]
        chunk_vectors = [v for v in vectors if start <= v["t"] + 14400 < end] # another 14400 offset from UTC >> local time
        chunk["vectors"] = chunk_vectors

    # If session is a dict, update its chunks
    if isinstance(session, dict):
        session["chunks"] = chunks
    else:
        session = chunks


    # Save the new session file
    with open(output_path, "w") as f:
        json.dump(session, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge vectors into chunk objects based on time windows.")
    parser.add_argument("--chunks", required=True, help="Path to session_events.json or chunks.json")
    parser.add_argument("--vectors", required=True, help="Path to vector_windows.jsonl")
    parser.add_argument("--output", required=True, help="Path to output merged JSON file")
    args = parser.parse_args()
    add_vectors_to_chunks(args.chunks, args.vectors, args.output)