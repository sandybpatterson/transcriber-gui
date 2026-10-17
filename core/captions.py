"""Pull a video's existing captions (YouTube, Facebook, etc.) instead of running Whisper.

Manual (creator-uploaded) captions are usually high quality; auto-
generated ones vary but are often still usable and are essentially free
to check for. Either way this is metadata + a small text file -- no
audio or video download at all -- so checking first costs almost nothing
compared to a full transcription.
"""
from __future__ import annotations

import html
import logging
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional

from core.engine import Segment, ytdlp_auth_opts

logger = logging.getLogger(__name__)

# A circuit breaker, not an expected outcome: no real video's spoken
# lines should ever number in the thousands. This exists specifically
# because a parsing miss against one caption style already produced over
# 8000 bloated segments for a single ~2-hour video -- the fix for that
# exact case is in _parse_captions, but this catches the same failure
# mode for any caption quirk not yet seen, falling back to transcribing
# instead of flooding the UI (and, worse, autosave -- which rewrites the
# whole growing file on every single segment) with thousands of entries.
MAX_REASONABLE_SEGMENTS = 3000

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
            # YouTube keys caption tracks by plain codes ("en"); Facebook
            # uses locales ("en_US"). The underscore patterns (yt-dlp
            # treats each entry as a regex) catch Facebook's without
            # widening YouTube's, whose regional variants use hyphens.
            "subtitleslangs": [lang, "en", f"{lang}_.*", "en_.*"],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmp_dir) / "%(id)s.%(ext)s"),
            "noplaylist": True,
            **ytdlp_auth_opts(),
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:  # noqa: BLE001 - no captions is a normal, expected outcome
            logger.info("Couldn't fetch captions for %s (%s) -- falling back to transcribing", url, exc)
            return None

        if on_title:
            on_title(info.get("title") or url)

        # Sorted so a video with several matching tracks always yields the
        # same one, rather than whatever order the filesystem lists them.
        caption_files = sorted(Path(tmp_dir).glob("*.vtt")) + sorted(Path(tmp_dir).glob("*.srt"))
        if not caption_files:
            logger.info("No captions found for %s", url)
            return None

        path = caption_files[0]
        segments = _parse_captions(path)
        if segments and len(segments) > MAX_REASONABLE_SEGMENTS:
            logger.warning(
                "Parsed %d segments from %s -- treating as a parsing failure, not real captions",
                len(segments), path.name,
            )
            return None
        logger.info("Using %d caption segment(s) from %s", len(segments) if segments else 0, path.name)
        return segments or None


def _parse_time(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _clean_text(text: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub("", text)).split())


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
            # YouTube's "rolling" auto-caption style repeats the *previous*
            # cue's full line again as leading context above each new
            # growing line within the same cue block -- confirmed directly
            # against a real 2-hour video, where joining every physical
            # line in a cue (the original approach) accumulated the whole
            # prior sentence into every subsequent segment, producing over
            # 8000 bloated, heavily-overlapping "segments" for what should
            # have been a few hundred real lines and made the UI look
            # hung trying to emit and autosave all of them. Only the LAST
            # physical line of a multi-line cue is the newly-revealed
            # text; the leading line(s) are always a repeat of what the
            # previous cue already contributed.
            text = _clean_text(text_lines[-1]) if text_lines else ""
            # The short "transition" cue YouTube emits between growing
            # lines (a sliver of a cue, often ~10ms) repeats that same
            # last line verbatim with nothing new -- collapsing an exact
            # repeat of the immediately previous cue's text drops those.
            if text and text != last_text:
                segments.append(Segment(start=start, end=end, text=text))
                last_text = text
        i += 1
    return _merge_into_sentences(segments)


def _merge_into_sentences(fragments: list[Segment], max_fragments: int = 20, max_duration: float = 20.0) -> list[Segment]:
    """Auto-captions break lines roughly every 1-3 seconds for live
    display, not at sentence boundaries -- confirmed directly, a real
    2-hour video's cleaned fragments were still 4000+ entries, each a
    partial phrase like "the scary open secret in the AI industry" /
    "right now is that it's possible that". Merges consecutive fragments
    up through the next sentence-ending punctuation, producing segments
    closer to how Whisper itself naturally segments speech. Capped by
    fragment count and duration in case punctuation is sparse or missing
    entirely, so one run-on stretch can't merge into an unreasonably huge
    segment."""
    sentence_end = re.compile(r"[.?!][\"')\]]?$")
    merged: list[Segment] = []
    buffer_start: Optional[float] = None
    buffer_end: Optional[float] = None
    buffer_parts: list[str] = []

    def flush() -> None:
        if buffer_parts:
            merged.append(Segment(start=buffer_start, end=buffer_end, text=" ".join(buffer_parts)))

    for frag in fragments:
        if buffer_start is None:
            buffer_start = frag.start
        buffer_end = frag.end
        buffer_parts.append(frag.text)
        too_long = len(buffer_parts) >= max_fragments or (buffer_end - buffer_start) >= max_duration
        if sentence_end.search(frag.text.strip()) or too_long:
            flush()
            buffer_start, buffer_end, buffer_parts = None, None, []
    flush()
    return merged
