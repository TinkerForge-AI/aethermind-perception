import argparse
import os
import json
import math
import datetime
import cv2
import soundfile as sf
import numpy as np

"""
Breaks down a scene into `vector_windows.jsonl` in the `aethermind-input` folder specified and represents the following: 
a sequence of half-second “snapshots” of your scene, each reduced to a compact 9-dimensional feature vector. Here’s exactly what each field means:

    t = the timestamp (in seconds since the epoch) at the midpoint of that 0.5 s window.

    x = a length-9 list broken down as:

        Video embedding (3 numbers):

            x[0] = mean red intensity over all frames in the window

            x[1] = mean green intensity

            x[2] = mean blue intensity
            These are stand-ins for a real visual embedding (e.g. CLIP); they capture the overall “brightness/color” of the scene.

        Audio embedding (2 numbers):

            x[3] = mean audio amplitude over the 0.5 s (loudness)

            x[4] = standard deviation of that amplitude (how much variability/noise there is)

        Action summary (4 numbers):

            x[5] = number of left-mouse clicks

            x[6] = number of right-mouse clicks

            x[7] = total key-press count

            x[8] = total mouse movement distance (in pixels)
    """

def embed_video(frames):
    """
    Stub: Replace with actual video embedding, e.g. CLIP.
    `frames` is a list of numpy arrays.
    Return a 1D numpy vector.
    """
    if not frames:
        return np.zeros(3)  # adjust size as needed
    # Example: mean RGB values across frames
    stack = np.stack([f.mean(axis=(0,1)) for f in frames])
    return stack.mean(axis=0)

def embed_audio(audio_segment, sample_rate):
    """
    Stub: Replace with actual audio embedding, e.g. VGGish.
    `audio_segment` is a numpy array of shape (n_samples,).
    Return a 1D numpy vector.
    """
    if audio_segment.size == 0:
        return np.zeros(2)
    return np.array([audio_segment.mean(), audio_segment.std()])

def summarize_actions(actions, start_ts, end_ts):
    """
    Summarize action events in [start_ts, end_ts].
    `actions` is a list of dicts with 'time', 'keys', 'mouse'.
    Return a fixed-length numpy vector.
    """
    # Filter actions in window
    window_actions = [a for a in actions if start_ts <= a['time'] < end_ts]
    left_clicks = sum(1 for a in window_actions if a['mouse']['buttons']['left'])
    right_clicks = sum(1 for a in window_actions if a['mouse']['buttons']['right'])
    key_presses = sum(len(a['keys']) for a in window_actions)
    # mouse movement distance
    mouse_dist = 0.0
    for prev, a in zip(window_actions, window_actions[1:]):
        prev_pos = np.array(prev['mouse']['position'])
        curr_pos = np.array(a['mouse']['position'])
        mouse_dist += np.linalg.norm(curr_pos - prev_pos)
    return np.array([left_clicks, right_clicks, key_presses, mouse_dist])

def load_actions(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def main(folder):
    # Load session metadata
    with open(os.path.join(folder, 'session.json'), 'r') as f:
        session = json.load(f)
    # Convert start_time string to epoch seconds if needed
    start_time_val = session['start_time']
    if isinstance(start_time_val, str):
        dt = datetime.datetime.strptime(start_time_val, "%Y-%m-%d_%H-%M-%S")
        start_time = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    else:
        start_time = start_time_val
    # Duration field
    duration = session.get('actual_duration', session.get('duration'))
    if duration is None:
        raise ValueError("Session JSON must include 'actual_duration' or 'duration'.")

    # Open video and audio
    video_path = os.path.join(folder, 'screen.mp4')
    audio_path = os.path.join(folder, 'audio.wav')
    cap = cv2.VideoCapture(video_path)
    audio, sr = sf.read(audio_path)

    # Load actions
    actions = load_actions(os.path.join(folder, 'actions.jsonl'))

    # Define window parameters
    window_size = 0.5  # seconds
    num_windows = math.ceil(duration / window_size)

    # Prepare output
    out_path = os.path.join(folder, 'vector_windows.jsonl')
    with open(out_path, 'w') as out_file:
        for i in range(num_windows):
            t0 = start_time + i * window_size
            t1 = min(start_time + (i + 1) * window_size, start_time + duration)

            # Video frames in [t0, t1)
            frames = []
            cap.set(cv2.CAP_PROP_POS_MSEC, (t0 - start_time) * 1000)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 + start_time
                if ts >= t1:
                    break
                frames.append(frame)

            # Audio samples in [t0, t1)
            a0 = int((t0 - start_time) * sr)
            a1 = int((t1 - start_time) * sr)
            audio_seg = audio[a0:a1] if a1 > a0 else np.array([])

            # Compute embeddings
            v_emb = embed_video(frames)
            a_emb = embed_audio(audio_seg, sr)
            c_feat = summarize_actions(actions, t0, t1)

            # Fuse into single vector
            x = np.concatenate([v_emb, a_emb, c_feat]).tolist()
            record = {'t': (t0 + t1) / 2.0, 'x': x}

            # Write JSON line
            out_file.write(json.dumps(record) + '\n')

            # Progress update
            if (i + 1) % max(1, num_windows // 10) == 0 or (i + 1) == num_windows:
                print(f"Processed window {i+1}/{num_windows} ({(i+1)/num_windows*100:.1f}%)")

    cap.release()
    print(f"Vector windows saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Stream input folder into VectorWindow objects"
    )
    parser.add_argument('--folder', required=True, help="Path to input folder")
    args = parser.parse_args()
    main(args.folder)
