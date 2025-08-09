# sync_and_health.py
def capture_sync_metrics():
    return {"av_ms": 0.0, "ai_ms": 0.0, "vi_ms": 0.0, "beacon_id": None, "estimated": False}

def capture_health_metrics(fps=60, dropped_frames=0, focused=True, resolution=(1920,1080),
                           codec="h264", crf=None):
    return {"fps": fps, "dropped_frames": dropped_frames, "window_focused": focused,
            "resolution": list(resolution), "codec": codec, "crf": crf}
