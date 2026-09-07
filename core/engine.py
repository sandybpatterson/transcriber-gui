"""Video -> audio -> timestamped text, decoupled from the UI.

`TranscriptionWorker` is a QObject meant to be moved to a QThread by the
caller; everything it depends on (`extract_audio`, `transcribe_audio`) is a
plain function usable and testable on its own.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3-turbo"]

# Speed/accuracy hint shown next to each option in the UI -- "tiny" and
# "large-v3-turbo" mean nothing to someone who's never used Whisper before.
MODEL_SIZE_LABELS = {
    "tiny": "tiny (fastest)",
    "base": "base (fast)",
    "small": "small (balanced)",
    "medium": "medium (accurate)",
    "large-v3-turbo": "large-v3-turbo (most accurate)",
}

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


class TranscriptionCancelled(Exception):
    """Raised internally when a user cancels a running transcription."""


def find_ffmpeg() -> str:
    """Return a path to an ffmpeg binary, downloading a portable one on first use."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def is_url(path: str) -> bool:
    """True if `path` looks like a URL (YouTube, etc.) rather than a local file."""
    return urlparse(path).scheme in ("http", "https")


def fetch_video_title(url: str) -> str:
    """Looks up a video's title without downloading anything -- used to
    pre-fill a suggested filename before a batch actually runs."""
    import yt_dlp

    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("title") or url


def download_audio(
    url: str,
    output_dir: str,
    on_progress: Optional[Callable[[str], None]] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
    on_title: Optional[Callable[[str], None]] = None,
) -> str:
    """Fetch only the audio-only stream from a URL (YouTube, etc.) as a WAV file.

    Never downloads the video track: yt-dlp picks an audio-only format when
    one is offered (true for virtually all YouTube videos), so this is a
    small, fast fetch compared to grabbing the full video.
    """
    import yt_dlp

    def hook(d: dict) -> None:
        if check_cancelled:
            check_cancelled()
        if on_progress and d.get("status") == "downloading":
            pct = d.get("_percent_str", "").strip()
            on_progress(f"Downloading audio...{(' ' + pct) if pct else ''}")

    # No FFmpegExtractAudio postprocessor here: yt-dlp's postprocessor also
    # shells out to ffprobe (to inspect the downloaded codec first), which
    # imageio-ffmpeg doesn't bundle -- it only provides ffmpeg. Downloading
    # the raw audio-only stream as-is and converting it with our own
    # extract_audio() (ffmpeg only, already used for local video files)
    # sidesteps that dependency entirely.
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(Path(output_dir) / "%(id)s.%(ext)s"),
        "ffmpeg_location": str(Path(find_ffmpeg()).parent),
        "progress_hooks": [hook],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))
    if on_title:
        on_title(info.get("title") or info.get("id") or downloaded_path.stem)
    if not downloaded_path.exists():
        raise RuntimeError(f"Expected downloaded audio at {downloaded_path}, but it wasn't created.")
    return extract_audio(str(downloaded_path), output_dir=output_dir)


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
    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{result.stderr}")
    return str(audio_path)


def _wav_duration_seconds(path: str) -> float:
    """Total duration of a WAV file, for estimating transcription progress."""
    import wave

    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        return wf.getnframes() / float(rate) if rate else 0.0


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    on_segment: Optional[Callable[[Segment], None]] = None,
) -> list[Segment]:
    """Run faster-whisper on an audio file, returning timestamped segments."""
    # First run downloads model weights from Hugging Face Hub. Left
    # unbounded, a dead or blocked connection hangs here indefinitely with
    # no way for the app to recover -- there's no cancellation checkpoint
    # reachable inside a third-party blocking HTTP call. Bounding it here
    # means a bad connection fails with a catchable exception in a fixed
    # window instead. Respects a value the user has already set.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

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
    title_resolved = Signal(str)
    segment_progress = Signal(int, float, float)  # percent, seconds done, total seconds
    segment_ready = Signal(object)
    finished = Signal(list)
    failed = Signal(str)
    cancelled = Signal()

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
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Ask the pipeline to stop at its next checkpoint (cooperative, not immediate)."""
        self._cancel_requested = True

    def _check_cancelled(self) -> None:
        if self._cancel_requested:
            raise TranscriptionCancelled()

    def run(self) -> None:
        download_dir: Optional[tempfile.TemporaryDirectory] = None
        extracted_audio_path: Optional[str] = None
        logger.info(
            "Transcription starting: video_path=%s model=%s language=%s diarize=%s",
            self.video_path, self.model_size, self.language, self.diarize,
        )
        try:
            if is_url(self.video_path):
                self.progress.emit("Downloading audio...")
                download_dir = tempfile.TemporaryDirectory(prefix="transcribergui_")
                audio_path = download_audio(
                    self.video_path,
                    output_dir=download_dir.name,
                    on_progress=self.progress.emit,
                    check_cancelled=self._check_cancelled,
                    on_title=self.title_resolved.emit,
                )
            else:
                self.title_resolved.emit(Path(self.video_path).stem)
                self.progress.emit("Extracting audio...")
                audio_path = extract_audio(self.video_path)
                # This WAV is a byproduct with no further use once transcribed --
                # clean it up rather than leaving it next to the source video
                # forever (matters at the volume of a batch of podcast episodes).
                extracted_audio_path = audio_path
            self._check_cancelled()

            self.progress.emit("Transcribing...")
            total_duration = _wav_duration_seconds(audio_path)

            def on_segment(seg: Segment) -> None:
                self.segment_ready.emit(seg)
                if total_duration > 0:
                    percent = max(0, min(100, int(seg.end / total_duration * 100)))
                    self.segment_progress.emit(percent, seg.end, total_duration)
                self._check_cancelled()

            segments = transcribe_audio(
                audio_path,
                model_size=self.model_size,
                language=self.language,
                on_segment=on_segment,
            )
            self._check_cancelled()
            logger.info("Transcription produced %d segment(s)", len(segments))

            if self.diarize:
                # Without a token, pyannote's gated-model download fails
                # authentication -- but that can surface as a confusing
                # stall rather than a clear error depending on how the
                # underlying HTTP client retries. Checked here, before
                # starting, so the failure is immediate and explains itself
                # instead of leaving the run looking hung.
                if not (self.hf_token or os.environ.get("HF_TOKEN")):
                    raise RuntimeError(
                        "Speaker identification needs a free Hugging Face token. "
                        "Add one in the app (next to the diarization checkbox) and try again."
                    )
                logger.info("Starting diarization")
                self.progress.emit("Identifying speakers...")
                from core.diarization import assign_speakers, diarize_audio

                speaker_segments = diarize_audio(audio_path, hf_token=self.hf_token, on_progress=self.progress.emit)
                segments = assign_speakers(segments, speaker_segments)
                self._check_cancelled()
                logger.info("Diarization found %d speaker turn(s)", len(speaker_segments))

            self.progress.emit("Done.")
            logger.info("Transcription finished successfully")
            self.finished.emit(segments)
        except TranscriptionCancelled:
            logger.info("Transcription cancelled by user")
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            logger.exception("Transcription failed")
            self.failed.emit(str(exc))
        finally:
            if download_dir is not None:
                download_dir.cleanup()
            if extracted_audio_path is not None:
                try:
                    os.remove(extracted_audio_path)
                except OSError:
                    pass
