"""Tracks watched YouTube channels and detects newly-published videos.

Uses yt-dlp's flat extraction (metadata only, no downloads) to list a
channel's videos cheaply, and a small JSON registry to remember which
videos have already been seen so repeat checks only surface new ones.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Only the most recent videos are fetched per check -- channels list
# newest-first, and a long-running show can have 1000+ episodes in its
# back catalog that there's no need to page through just to find new ones.
RECENT_VIDEOS_TO_CHECK = 30

REGISTRY_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / "TranscriberGUI" / "channels.json"


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"channels": {}}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"channels": {}}


def _save_registry(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _seen_videos(channel: dict) -> list[dict]:
    """Normalizes both the current schema ("seen_videos": [{"id","title"}])
    and the older one ("seen_ids": [str, ...]) into the richer form, so a
    registry written by an earlier version of the app still reads fine."""
    if "seen_videos" in channel:
        return channel["seen_videos"]
    return [{"id": vid, "title": vid} for vid in channel.get("seen_ids", [])]


def list_channels() -> dict:
    """channel_url -> {"name": str, "seen_videos": [{"id", "title"}, ...]}"""
    registry = _load_registry()
    for channel in registry["channels"].values():
        channel["seen_videos"] = _seen_videos(channel)
        channel.pop("seen_ids", None)
    return registry["channels"]


def get_channel_videos(channel_url: str) -> list[dict]:
    """The videos currently recorded as seen for this channel (baseline
    plus anything processed since) -- each with "id", "title", "url"."""
    channel = list_channels().get(channel_url)
    if channel is None:
        return []
    return [
        {"id": v["id"], "title": v.get("title") or v["id"], "url": f"https://www.youtube.com/watch?v={v['id']}"}
        for v in channel["seen_videos"]
    ]


_CHANNEL_TABS = ("/videos", "/shorts", "/streams", "/playlists", "/community", "/featured")


def normalize_channel_url(url: str) -> str:
    """A bare channel URL (https://youtube.com/@name, /channel/UC..., etc.)
    resolves in yt-dlp's flat listing to its tab-selection page -- Videos,
    Live, and Shorts each as a *nested* playlist entry reusing the
    channel's own id, not an actual video list. Confirmed directly: this
    is exactly what produced "3 videos" that were really just those 3
    tabs. Appending /videos when no tab or specific path is already
    present forces it straight to the uploads list instead."""
    url = url.rstrip("/")
    if any(url.endswith(tab) for tab in _CHANNEL_TABS):
        return url
    if "watch?v=" in url or "/playlist" in url or "list=" in url:
        return url  # not a bare channel URL at all -- leave it alone
    return url + "/videos"


def _fetch_recent_videos(channel_url: str) -> tuple[str, list[dict]]:
    """Returns (channel_name, videos) -- videos newest-first, metadata only."""
    import yt_dlp

    ydl_opts = {
        "extract_flat": "in_playlist",
        "playlistend": RECENT_VIDEOS_TO_CHECK,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(normalize_channel_url(channel_url), download=False)
    name = info.get("title") or channel_url
    for suffix in (" - Videos", " - Live", " - Shorts"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    # Defense in depth alongside normalize_channel_url(): a nested tab
    # pointer (Videos/Live/Shorts) comes back with _type "playlist" and
    # reuses the channel's own id instead of being a real video -- skip
    # any that still slip through.
    entries = [e for e in (info.get("entries") or []) if e.get("id") and e.get("_type") != "playlist"]
    logger.info("%s: yt-dlp flat-listing returned %d entries (cap %d)", channel_url, len(entries), RECENT_VIDEOS_TO_CHECK)
    return name, entries


def add_channel(channel_url: str) -> dict:
    """Fetches the channel's current videos and baselines all of them as
    already-seen -- only videos published after this point count as new."""
    channel_url = normalize_channel_url(channel_url)
    logger.info("Adding channel: %s", channel_url)
    name, entries = _fetch_recent_videos(channel_url)
    registry = _load_registry()
    registry["channels"][channel_url] = {
        "name": name,
        "seen_videos": [{"id": e["id"], "title": e.get("title") or e["id"]} for e in entries],
    }
    _save_registry(registry)
    logger.info("Added channel %r (%s) with %d baseline video(s)", name, channel_url, len(entries))
    return registry["channels"][channel_url]


def remove_channel(channel_url: str) -> None:
    logger.info("Removing channel: %s", channel_url)
    registry = _load_registry()
    registry["channels"].pop(channel_url, None)
    _save_registry(registry)


def find_new_videos(channel_url: str) -> list[dict]:
    """New videos (not yet marked seen), oldest-first so they process in
    upload order. Each dict has "id", "title", and "url"."""
    registry = _load_registry()
    channel = registry["channels"].get(channel_url)
    if channel is None:
        return []
    seen = {v["id"] for v in _seen_videos(channel)}
    _, entries = _fetch_recent_videos(channel_url)
    new_entries = [e for e in entries if e["id"] not in seen]
    new_entries.reverse()
    logger.info("%s: %d new video(s) found", channel_url, len(new_entries))
    return [
        {"id": e["id"], "title": e.get("title") or e["id"], "url": f"https://www.youtube.com/watch?v={e['id']}"}
        for e in new_entries
    ]


def mark_seen(channel_url: str, video_id: str, title: str = "") -> None:
    registry = _load_registry()
    channel = registry["channels"].get(channel_url)
    if channel is None:
        return
    videos = _seen_videos(channel)
    if video_id not in {v["id"] for v in videos}:
        videos.append({"id": video_id, "title": title or video_id})
        channel["seen_videos"] = videos
        channel.pop("seen_ids", None)
        _save_registry(registry)
