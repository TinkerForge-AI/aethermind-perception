import subprocess
from pathlib import Path
from moviepy import VideoFileClip

def chunk_video_audio(video_path, audio_path, chunk_duration=2.0, output_dir="chunks"):
    """
    Split a video and its corresponding WAV file into synchronized chunks.
    Video chunks are written without audio; audio chunks are extracted via ffmpeg.
    Returns a list of dicts with 'start', 'end', 'video_path', and 'audio_path'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video = VideoFileClip(str(video_path))
    total_duration = video.duration

    chunks = []
    start = 0.0
    idx = 0
    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        base = f"chunk_{idx:04d}"
        video_out = out_dir / f"{base}.mp4"
        audio_out = out_dir / f"{base}.wav"

        # 1) write video chunk (no audio)
        video.subclipped(start, end).write_videofile(
            str(video_out), codec="libx264", audio=False
        )

        # 2) extract audio chunk via ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start), "-to", str(end),
            "-c:a", "pcm_s16le", str(audio_out)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        chunks.append({
            "start": start,
            "end": end,
            "video_path": str(video_out),
            "audio_path": str(audio_out)
        })

        start = end
        idx += 1

    video.close()
    return chunks