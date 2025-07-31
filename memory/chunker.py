
"""Temporal chunking of raw session data."""
from dataclasses import dataclass
from pathlib import Path
from typing import List
from moviepy.editor import VideoFileClip, AudioFileClip
import os

@dataclass
class Chunk:
    video_path: Path
    audio_path: Path
    start_time: float
    end_time: float

class Chunker:
    """Split a recorded session into fixed-length overlapping clips."""
    def __init__(self, window_sec: float = 2.0, stride_sec: float = 1.0, output_dir: Path = Path("./clips")):
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def chunk_session(self, video_file: Path, audio_file: Path) -> List[Chunk]:
        video_clip = VideoFileClip(str(video_file))
        audio_clip = AudioFileClip(str(audio_file))
        duration = min(video_clip.duration, audio_clip.duration)

        chunks = []
        t = 0.0
        idx = 0
        while t + self.window_sec <= duration:
            start = t
            end = t + self.window_sec

            chunk_video_path = self.output_dir / f"chunk_{idx:04d}_video.mp4"
            chunk_audio_path = self.output_dir / f"chunk_{idx:04d}_audio.wav"

            video_subclip = video_clip.subclip(start, end)
            audio_subclip = audio_clip.subclip(start, end)

            video_subclip.write_videofile(str(chunk_video_path), audio=False, codec="libx264", logger=None)
            audio_subclip.write_audiofile(str(chunk_audio_path), logger=None)

            chunks.append(Chunk(
                video_path=chunk_video_path,
                audio_path=chunk_audio_path,
                start_time=start,
                end_time=end
            ))

            t += self.stride_sec
            idx += 1

        video_clip.close()
        audio_clip.close()
        return chunks