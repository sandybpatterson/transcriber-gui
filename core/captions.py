"""Pull a YouTube video's existing captions instead of running Whisper.

Manual (creator-uploaded) captions are usually high quality; auto-
generated ones vary but are often still usable and are essentially free
to check for. Either way this is metadata + a small text file -- no
audio or video download at all -- so checking first costs almost nothing
compared to a full transcription.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional

from core.engine import Segment

logger = logging.getLogger(__name__)

# VTT uses '.' before milliseconds, SRT uses ',' -- accepting either makes
# one parser handle both formats, since yt-dlp doesn't always honor a
# requested format conversion (observed directly: asking for srt without
# a working ffmpeg location can silently come back as vtt instead).
_TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[.,](\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def fetch_captions(
    url: str,
    language: Optional[str] = None,
    on_title: Optional[Callable[[str], None]] = None,
) -> Optional[list[Segment]]:
    """Returns existing caption segments for `url`, or None if it has no
    captions in the requested language at all (falling back to actually
    transcribing is the caller's job in that case)."""
    import yt_dlp

    lang = language or "en"
    with tempfile.TemporaryDirectory(prefix="transcribergui_captions_") as tmp_dir:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [lang, "en"],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmp_dir) / "%(id)s.%(ext)s"),
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:  # noqa: BLE001 - no captions is a normal, expected outcome
            logger.info("Couldn't fetch captions for %s (%s) -- falling back to transcribing", url, exc)
            return None

        if on_title:
            on_title(info.get("title") or url)

        caption_files = list(Path(tmp_dir).glob("*.vtt")) + list(Path(tmp_dir).glob("*.srt"))
        if not caption_files:
            logger.info("No captions found for %s", url)
            return None

        path = caption_files[0]
        logger.info("Using captions from %s", path.name)
        segments = _parse_captions(path)
        return segments or None


def _parse_time(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _clean_text(text: str) -> str:
    return " ".join(_TAG_RE.sub("", text).split())


def _parse_captions(path: Path) -> list[Segment]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    segments: list[Segment] = []
    last_text: Optional[str] = None
    i = 0
    while i < len(lines):
        match = _TIME_RE.search(lines[i])
        if match:
            groups = match.groups()
            start = _parse_time(*groups[0:4])
            end = _parse_time(*groups[4:8])
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i])
                i += 1
            text = _clean_text(" ".join(text_lines))
            # YouTube's "rolling" auto-caption style repeats the same
            # cue text across consecutive entries as it grows on screen --
            # collapsing an exact repeat of the immediately previous cue
            # handles the common case without needing a full text diff.
            if text and text != last_text:
                segments.append(Segment(start=start, end=end, text=text))
                last_text = text
        i += 1
    return segments
