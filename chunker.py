
import os
from moviepy import VideoFileClip
from pathlib import Path

def chunk_video_audio(video_path, audio_path, chunk_duration=2.0, output_dir="chunks"):
    video = VideoFileClip(video_path)
    total_duration = video.duration
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    start = 0.0
    idx = 0

    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        chunk_filename_base = f"chunk_{idx:04d}"

        video_chunk_path = output_dir / f"{chunk_filename_base}.mp4"
        audio_chunk_path = output_dir / f"{chunk_filename_base}.wav"

        # Extract video segment and write to file (includes audio)
        video.subclipped(start, end).write_videofile(
            str(video_chunk_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(output_dir / f"{chunk_filename_base}_temp-audio.m4a"),
            remove_temp=True,
        )

        # Extract audio only
        video.subclipped(start, end).audio.write_audiofile(
            str(audio_chunk_path),
        )

        chunks.append({
            "start": start,
            "end": end,
            "video_path": str(video_chunk_path),
            "audio_path": str(audio_chunk_path)
        })

        start = end
        idx += 1

    video.close()
    return chunks