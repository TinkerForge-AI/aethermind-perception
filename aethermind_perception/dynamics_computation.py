# dynamics_computation.py
import numpy as np, cv2, librosa
def compute_optical_flow(video_frames, frame_idx_start=0):
    if len(video_frames) < 2:
        return {"flow_mean": 0.0, "flow_std": 0.0, "cut_prob": 0.0,
                "frame_idx_start": frame_idx_start, "frame_idx_end": frame_idx_start}
    mags = []
    prev = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(video_frames)):
        nxt = cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        mags.append(mag.mean()); prev = nxt
    return {"flow_mean": float(np.mean(mags)), "flow_std": float(np.std(mags)),
            "cut_prob": 0.0, "frame_idx_start": frame_idx_start,
            "frame_idx_end": frame_idx_start + len(video_frames)-1}

def compute_audio_rms(audio_path, frame_len=2048, hop_len=1024):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    return {"rms_frames": rms.astype(float).tolist(), "sr": int(sr)}
