# aethermind_pipeline.py
from event_seed_emitter import emit_event_seed
from input_semantics_mapper import process_actions
from sync_and_health import capture_sync_metrics, capture_health_metrics
from dynamics_computation import compute_optical_flow, compute_audio_rms

def process_event(session_id, video_frames, audio_path, raw_actions, video_path, resolution=(1920,1080)):
    semantic_actions = process_actions(raw_actions, resolution=resolution)
    sync_metrics = capture_sync_metrics()
    health_metrics = capture_health_metrics(resolution=resolution)
    video_dyn = compute_optical_flow(video_frames, frame_idx_start=0)
    audio_dyn = compute_audio_rms(audio_path)
    return emit_event_seed(
        session_id=session_id, video_path=video_path, audio_path=audio_path,
        actions=semantic_actions, sync=sync_metrics, video_dyn=video_dyn,
        audio_dyn=audio_dyn, system=health_metrics
    )
