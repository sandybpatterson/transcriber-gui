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
from urllib.parse import parse_qs, urlparse

from PySide6.QtCore import QObject, QSettings, Signal

logger = logging.getLogger(__name__)

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# A sentinel "model size" meaning "try the video's existing captions
# first" (YouTube's, or any other site yt-dlp can pull them from, e.g.
# Facebook) rather than an actual Whisper model -- listed first
# since it's the fastest option when it works. Falls back to a real model
# (FALLBACK_MODEL_SIZE) if the video has no captions, isn't a URL, or
# diarization is on (captions carry no speaker info, so that combination
# always needs real audio regardless of this setting).
USE_YOUTUBE_CAPTIONS = "youtube_captions"
FALLBACK_MODEL_SIZE = "base"

MODEL_SIZES = [USE_YOUTUBE_CAPTIONS, "tiny", "base", "small", "medium", "large-v3-turbo"]

# Speed/accuracy hint shown next to each option in the UI -- "tiny" and
# "large-v3-turbo" mean nothing to someone who's never used Whisper before.
MODEL_SIZE_LABELS = {
    USE_YOUTUBE_CAPTIONS: "Use the video's own captions if available (fastest)",
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
    """True if `path` looks like a URL (YouTube, Facebook, etc.) rather than a local file."""
    return urlparse(path).scheme in ("http", "https")


def url_fallback_name(url: str) -> str:
    """A filename-ish stand-in for a URL whose real title isn't known yet.

    Taking everything after the last '/' (the original approach) works for
    youtu.be/ID or facebook.com/reel/ID, but turns the very common
    youtube.com/watch?v=ID and facebook.com/watch/?v=ID shapes into
    "watch?v=ID" or "?v=ID" -- the `v` query parameter is the real ID.
    """
    parsed = urlparse(url)
    video_id = parse_qs(parsed.query).get("v")
    if video_id and video_id[0]:
        return video_id[0]
    return parsed.path.rstrip("/").rsplit("/", 1)[-1] or parsed.netloc or url


# Where yt-dlp should borrow sign-in cookies from, so videos that are only
# visible when logged in (most Facebook videos outside public Pages:
# friends-only posts, groups, many reels) can be fetched. Stored in
# QSettings as "cookies_source" ("" / "firefox" / "file") plus
# "cookies_file". Firefox is the only browser offered on purpose: since
# Chrome 127 (and Edge, which shares its engine), Windows encrypts their
# cookies with an app-bound key yt-dlp can't read, so offering them would
# just be a setting that silently fails. Anyone on Chrome/Edge can export
# a cookies.txt instead.
COOKIES_NONE = ""
COOKIES_FIREFOX = "firefox"
COOKIES_FILE = "file"


def ytdlp_auth_opts() -> dict:
    """yt-dlp options for the user's chosen sign-in cookies, if any.

    Read fresh from QSettings on every call (QSettings is safe to use from
    worker threads) so a change in the main window applies to the very
    next fetch without threading the setting through every caller.
    """
    settings = QSettings("TranscriberGUI", "TranscriberGUI")
    source = settings.value("cookies_source", COOKIES_NONE, type=str)
    if source == COOKIES_FIREFOX:
        return {"cookiesfrombrowser": ("firefox",)}
    if source == COOKIES_FILE:
        cookie_file = settings.value("cookies_file", "", type=str)
        if cookie_file:
            return {"cookiefile": cookie_file}
    return {}


def _is_facebook(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "fb.watch" or host.endswith("facebook.com")


def explain_download_error(url: str, exc: Exception) -> str:
    """yt-dlp's own error text, plus a plain-English next step when the
    likely cause is a Facebook video that needs a signed-in session --
    yt-dlp's raw message for that case ("Cannot parse data", "login
    required", ...) doesn't say what to actually do about it."""
    message = str(exc)
    if not _is_facebook(url):
        return message
    lowered = message.lower()
    login_hints = ("login", "log in", "logged", "cookies", "cannot parse data", "unsupported url", "private")
    if any(hint in lowered for hint in login_hints):
        if ytdlp_auth_opts():
            message += (
                "\n\nThis Facebook video still couldn't be read with your sign-in cookies. "
                "Make sure you're logged in to Facebook in that browser (or re-export "
                "cookies.txt) and that your account can actually see this video."
            )
        else:
            message += (
                "\n\nThis Facebook video probably needs you to be signed in. Log in to "
                "Facebook in Firefox, then set \"Sign in for private videos\" to Firefox "
                "(or point it at an exported cookies.txt file) and try again."
            )
    return message


def fetch_video_title(url: str) -> str:
    """Looks up a video's title without downloading anything -- used to
    pre-fill a suggested filename before a batch actually runs."""
    import yt_dlp

    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True, **ytdlp_auth_opts()}
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
    """Fetch only the audio-only stream from a URL (YouTube, Facebook, etc.) as a WAV file.

    Avoids the video track where it can: yt-dlp picks an audio-only format
    when one is offered (true for virtually all YouTube videos and most
    Facebook ones), so this is a small, fast fetch compared to grabbing the
    full video. `bestaudio/best` still falls back to a combined stream for
    the odd video with no separate audio, which extract_audio() handles.
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
        **ytdlp_auth_opts(),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = Path(ydl.prepare_filename(info))
    except TranscriptionCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with a clearer message
        # yt-dlp can wrap an exception raised from a progress hook in its
        # own DownloadError -- a user cancel must still surface as a cancel.
        if check_cancelled:
            check_cancelled()
        raise RuntimeError(explain_download_error(url, exc)) from exc
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


class _LogForwardingHandler(logging.Handler):
    """Forwards a logger's records to an on_progress callback, so an
    otherwise-silent internal library step shows up as visible status
    text instead of a dead gap in the UI."""

    def __init__(self, on_progress: Callable[[str], None], prefix: str):
        super().__init__()
        self._on_progress = on_progress
        self._prefix = prefix

    def emit(self, record: logging.LogRecord) -> None:
        self._on_progress(f"{self._prefix} ({self.format(record)})")


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    on_segment: Optional[Callable[[Segment], None]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[Segment]:
    """Run faster-whisper on an audio file, returning timestamped segments."""
    # First run downloads model weights from Hugging Face Hub. Left
    # unbounded, a dead or blocked connection hangs here indefinitely with
    # no way for the app to recover -- there's no cancellation checkpoint
    # reachable inside a third-party blocking HTTP call. Bounding it here
    # means a bad connection fails with a catchable exception in a fixed
    # window instead. Respects a value the user has already set.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

    if on_progress:
        on_progress("Transcribing... (loading model)")

    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="auto", compute_type="auto")

    # model.transcribe() returns a lazy generator: loading the model
    # weights, running voice-activity detection over the *entire* file,
    # and detecting the spoken language all happen up front, the moment
    # iteration starts below -- before the very first transcript line is
    # available. For a long episode with no feedback during that stretch,
    # this reads as a hang: confirmed live, a user watching it couldn't
    # tell it apart from one. faster_whisper already logs each of those
    # steps ("Processing audio with duration...", "VAD filter removed...",
    # "Detected language...") -- forwarding them to on_progress surfaces
    # real, specific movement instead of a static "Transcribing..." the
    # whole time.
    fw_logger = logging.getLogger("faster_whisper")
    handler: Optional[_LogForwardingHandler] = None
    previous_level = fw_logger.level
    if on_progress:
        on_progress("Transcribing... (analyzing audio)")
        handler = _LogForwardingHandler(on_progress, "Transcribing...")
        fw_logger.addHandler(handler)
        # A handler attached directly to this logger still only receives
        # what the logger's own *effective* level lets through -- which,
        # with no level of its own (the default), is inherited from its
        # nearest ancestor. Confirmed directly: standalone, with no other
        # logging configured, that resolves to root's default WARNING,
        # silently dropping these INFO messages before they ever reach
        # this handler. Setting it explicitly makes this work regardless
        # of whatever the app's global logging setup happens to be.
        fw_logger.setLevel(logging.INFO)

    try:
        raw_segments, _info = model.transcribe(audio_path, language=language, vad_filter=True)

        segments: list[Segment] = []
        for raw in raw_segments:
            segment = Segment(start=raw.start, end=raw.end, text=raw.text.strip())
            segments.append(segment)
            if on_segment:
                on_segment(segment)
        return segments
    finally:
        if handler is not None:
            fw_logger.removeHandler(handler)
            fw_logger.setLevel(previous_level)


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
            effective_model_size = self.model_size
            if self.model_size == USE_YOUTUBE_CAPTIONS:
                if self.diarize:
                    # Captions carry no speaker info at all -- this
                    # combination always needs real audio regardless of
                    # the captions preference.
                    self.progress.emit(
                        "Speaker identification needs real audio -- transcribing instead of using captions."
                    )
                    effective_model_size = FALLBACK_MODEL_SIZE
                elif not is_url(self.video_path):
                    self.progress.emit("Captions aren't available for local files -- transcribing instead.")
                    effective_model_size = FALLBACK_MODEL_SIZE
                else:
                    self.progress.emit("Checking for existing captions...")
                    from core.captions import fetch_captions

                    caption_segments = fetch_captions(
                        self.video_path, language=self.language, on_title=self.title_resolved.emit,
                    )
                    self._check_cancelled()
                    if caption_segments:
                        logger.info("Using %d caption segment(s) instead of transcribing", len(caption_segments))
                        for seg in caption_segments:
                            self.segment_ready.emit(seg)
                            self._check_cancelled()
                        self.progress.emit("Done (used the video's existing captions).")
                        self.finished.emit(caption_segments)
                        return
                    self.progress.emit("No captions found -- transcribing instead.")
                    effective_model_size = FALLBACK_MODEL_SIZE

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
                model_size=effective_model_size,
                language=self.language,
                on_segment=on_segment,
                on_progress=self.progress.emit,
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
