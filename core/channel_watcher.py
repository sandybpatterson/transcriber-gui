"""Tracks watched YouTube channels and detects newly-published videos.

Uses yt-dlp's flat extraction (metadata only, no downloads) to list a
channel's videos cheaply, and a small JSON registry to remember which
video ids have already been seen so repeat checks only surface new ones.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

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


def list_channels() -> dict:
    """channel_url -> {"name": str, "seen_ids": [str, ...]}"""
    return _load_registry()["channels"]


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
        info = ydl.extract_info(channel_url, download=False)
    name = info.get("title") or channel_url
    entries = [e for e in (info.get("entries") or []) if e.get("id")]
    return name, entries


def add_channel(channel_url: str) -> dict:
    """Fetches the channel's current videos and baselines all of them as
    already-seen -- only videos published after this point count as new."""
    name, entries = _fetch_recent_videos(channel_url)
    registry = _load_registry()
    registry["channels"][channel_url] = {
        "name": name,
        "seen_ids": [e["id"] for e in entries],
    }
    _save_registry(registry)
    return registry["channels"][channel_url]


def remove_channel(channel_url: str) -> None:
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
    seen = set(channel["seen_ids"])
    _, entries = _fetch_recent_videos(channel_url)
    new_entries = [e for e in entries if e["id"] not in seen]
    new_entries.reverse()
    return [
        {"id": e["id"], "title": e.get("title") or e["id"], "url": f"https://www.youtube.com/watch?v={e['id']}"}
        for e in new_entries
    ]


def mark_seen(channel_url: str, video_id: str) -> None:
    registry = _load_registry()
    channel = registry["channels"].get(channel_url)
    if channel is None:
        return
    if video_id not in channel["seen_ids"]:
        channel["seen_ids"].append(video_id)
        _save_registry(registry)
