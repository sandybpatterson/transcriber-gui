"""Video -> audio -> timestamped text, decoupled from the UI.

`TranscriptionWorker` is a QObject meant to be moved to a QThread by the
caller; everything it depends on (`extract_audio`, `transcribe_audio`) is a
plain function usable and testable on its own.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3-turbo"]

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


def find_ffmpeg() -> str:
    """Return a path to an ffmpeg binary, downloading a portable one on first use."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: str, output_dir: Optional[str] = None) -> str:
    """Extract a mono 16kHz WAV audio track from a video file via ffmpeg."""
    video = Path(video_path)
    out_dir = Path(output_dir) if output_dir else video.parent
    audio_path = out_dir / f"{video.stem}.wav"

    cmd = [
        find_ffmpeg(), "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{result.stderr}")
    return str(audio_path)


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    on_segment: Optional[Callable[[Segment], None]] = None,
) -> list[Segment]:
    """Run faster-whisper on an audio file, returning timestamped segments."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    raw_segments, _info = model.transcribe(audio_path, language=language, vad_filter=True)

    segments: list[Segment] = []
    for raw in raw_segments:
        segment = Segment(start=raw.start, end=raw.end, text=raw.text.strip())
        segments.append(segment)
        if on_segment:
            on_segment(segment)
    return segments


class TranscriptionWorker(QObject):
    """Runs extract -> transcribe -> (optional) diarize on a background thread."""

    progress = Signal(str)
    segment_ready = Signal(object)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        video_path: str,
        model_size: str = "base",
        language: Optional[str] = None,
        diarize: bool = False,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.video_path = video_path
        self.model_size = model_size
        self.language = language
        self.diarize = diarize
        self.hf_token = hf_token

    def run(self) -> None:
        try:
            self.progress.emit("Extracting audio...")
            audio_path = extract_audio(self.video_path)

            self.progress.emit("Transcribing...")
            segments = transcribe_audio(
                audio_path,
                model_size=self.model_size,
                language=self.language,
                on_segment=lambda seg: self.segment_ready.emit(seg),
            )

            if self.diarize:
                self.progress.emit("Identifying speakers...")
                from core.diarization import assign_speakers, diarize_audio

                speaker_segments = diarize_audio(audio_path, hf_token=self.hf_token)
                segments = assign_speakers(segments, speaker_segments)

            self.progress.emit("Done.")
            self.finished.emit(segments)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))
