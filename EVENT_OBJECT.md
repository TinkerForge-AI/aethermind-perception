# Base Event Object example:

[
  {
    "start": 12.5,
    "end": 17.5,
    "video_path": "chunks/chunk_0003.mp4",
    "audio_path": "chunks/chunk_0003.wav",
    "valence": "unknown",
    "source": "perception",
    "annotations": {},
    "raw_motion": 0.24,
    "raw_energy": 0.68,
    "event_score": 0.45,
    "is_event": false,
    "actions": [
        { "type": "mouse_move", "ts": 13.04, "x": 125, "y": 300 },
        { "type": "key_press",  "ts": 16.88, "key": "Enter" }
    ]
    }
  // ... more chunk event objects ...
]

# Minimal "VectorWindow" schema

{
  "t": 1754093547.5,              // mid-point timestamp
  "video_emb": [0.12, -0.03, …],  // d_v floats
  "audio_emb": [ 1.23, 0.98, …],  // d_a floats
  "action_feat": {                // a small dict of scalars
    "left_clicks": 0,
    "right_clicks": 1,
    "mouse_dx": 20,
    "mouse_dy": -15,
    "key_presses": 1
  }
}
