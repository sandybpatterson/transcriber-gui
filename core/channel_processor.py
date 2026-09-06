"""Background worker that checks watched channels and transcribes new videos.

Mirrors TranscriptionWorker's cooperative-cancellation design (see engine.py)
but drives a queue of videos across possibly-several channels instead of a
single file, and writes each result straight to an output folder rather
than feeding a live results table.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from core.channel_watcher import find_new_videos, list_channels, mark_seen
from core.engine import TranscriptionCancelled, download_audio, transcribe_audio
from exporters import export_txt


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def _slugify(name: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return slug[:80] or "transcript"


class ChannelCheckWorker(QObject):
    log = Signal(str)
    finished = Signal(int)  # count of videos successfully processed
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        output_dir: str,
        model_size: str = "base",
        language: Optional[str] = None,
        diarize: bool = False,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.model_size = model_size
        self.language = language
        self.diarize = diarize
        self.hf_token = hf_token
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _check_cancelled(self) -> None:
        if self._cancel_requested:
            raise TranscriptionCancelled()

    def run(self) -> None:
        try:
            if self.diarize and not (self.hf_token or os.environ.get("HF_TOKEN")):
                raise RuntimeError(
                    "Speaker identification needs a free Hugging Face token. "
                    "Add one in the main window and try again."
                )
            channels = list_channels()
            if not channels:
                self.log.emit("No channels are being watched yet.")
                self.finished.emit(0)
                return

            queue: list[tuple[str, dict]] = []
            for url, info in channels.items():
                self.log.emit(f"Checking {info['name']}...")
                new_videos = find_new_videos(url)
                self._check_cancelled()
                if new_videos:
                    self.log.emit(f"  {len(new_videos)} new video(s).")
                    queue.extend((url, video) for video in new_videos)
                else:
                    self.log.emit("  Nothing new.")

            if not queue:
                self.log.emit("No new videos found across any watched channel.")
                self.finished.emit(0)
                return

            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            processed = 0
            for channel_url, video in queue:
                self._check_cancelled()
                self.log.emit(f"Processing: {video['title']}")
                try:
                    self._process_one(video)
                    mark_seen(channel_url, video["id"])
                    processed += 1
                except TranscriptionCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - log and move on to the next video
                    self.log.emit(f"  Failed ({exc}) -- will retry next check.")

            self.log.emit(f"Done. {processed} of {len(queue)} new video(s) transcribed.")
            self.finished.emit(processed)
        except TranscriptionCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))

    def _process_one(self, video: dict) -> None:
        with tempfile.TemporaryDirectory(prefix="transcribergui_channel_") as download_dir:
            self.log.emit("  Downloading audio...")
            audio_path = download_audio(
                video["url"],
                output_dir=download_dir,
                check_cancelled=self._check_cancelled,
            )
            self._check_cancelled()

            self.log.emit("  Transcribing...")
            segments = transcribe_audio(
                audio_path,
                model_size=self.model_size,
                language=self.language,
            )
            self._check_cancelled()

            if self.diarize:
                self.log.emit("  Identifying speakers...")
                from core.diarization import assign_speakers, diarize_audio

                speaker_segments = diarize_audio(audio_path, hf_token=self.hf_token)
                segments = assign_speakers(segments, speaker_segments)
                self._check_cancelled()

        out_path = _unique_path(Path(self.output_dir), _slugify(video["title"]), ".txt")
        export_txt(segments, str(out_path))
        self.log.emit(f"  Saved: {out_path}")
