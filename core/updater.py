"""Self-update from GitHub: check for new commits, pull them, restart.

The app runs straight from a git clone (the desktop shortcut points at
this folder's venv), so "updating" is just a fast-forward `git pull` of
origin -- no installer, no download step. Every git call runs without a
console window, since the app itself is launched with pythonw.exe and a
flashing black box per call would look like something went wrong.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class UpdateError(Exception):
    """An update step failed in a way the user should be told about."""


@dataclass
class UpdateInfo:
    current: str
    latest: str
    # One "abc1234 Commit subject" line per new commit, newest first.
    commits: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.commits)


def _git(*args: str, timeout: int = 60) -> str:
    git = shutil.which("git")
    if git is None:
        raise UpdateError("Git isn't installed (or isn't on PATH), so Scribe can't update itself.")
    try:
        result = subprocess.run(
            [git, *args], cwd=APP_DIR, capture_output=True, text=True,
            timeout=timeout, creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(f"git {args[0]} timed out -- check the internet connection.") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise UpdateError(f"git {' '.join(args)} failed:\n{detail}")
    return result.stdout.strip()


def current_version() -> str:
    """Short commit hash of what's running, or "" if it can't be read."""
    try:
        return _git("rev-parse", "--short", "HEAD", timeout=10)
    except UpdateError:
        return ""


def _upstream() -> str:
    try:
        return _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", timeout=10)
    except UpdateError:
        return "origin/main"


def check_for_update() -> UpdateInfo:
    _git("fetch", "--quiet", "origin")
    upstream = _upstream()
    log = _git("log", "--format=%h %s", f"HEAD..{upstream}")
    return UpdateInfo(
        current=_git("rev-parse", "--short", "HEAD"),
        latest=_git("rev-parse", "--short", upstream),
        commits=[line for line in log.splitlines() if line.strip()],
    )


def apply_update() -> bool:
    """Fast-forward to the latest commit. Returns True if dependencies were
    (re)installed because requirements.txt changed."""
    # Refuse rather than merge or stash: a local edit here means someone is
    # working on the code in this folder, and an update should never be
    # the thing that tangles that work up.
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise UpdateError(
            "This copy of Scribe has local code changes that aren't on GitHub yet.\n"
            "Commit and push them (or undo them) first, then update."
        )
    before = _git("rev-parse", "HEAD")
    _git("pull", "--ff-only", "--quiet", timeout=120)
    changed = _git("diff", "--name-only", before, "HEAD").splitlines()
    logger.info("Updated %s -> %s (%d files)", before[:7], current_version(), len(changed))
    if "requirements.txt" in changed:
        _install_requirements()
        return True
    return False


def _console_python() -> str:
    # sys.executable is pythonw.exe when launched from the shortcut; pip
    # works with either, but python.exe next to it is the conventional one.
    exe = Path(sys.executable)
    candidate = exe.with_name("python.exe")
    return str(candidate if candidate.exists() else exe)


def _install_requirements() -> None:
    try:
        result = subprocess.run(
            [_console_python(), "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=APP_DIR, capture_output=True, text=True, timeout=1800,
            creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise UpdateError("Installing the new requirements took too long and was stopped.") from exc
    if result.returncode != 0:
        raise UpdateError(
            "The code updated, but installing its new requirements failed:\n"
            + (result.stderr or result.stdout).strip()[-1500:]
        )


def restart() -> None:
    """Launch a fresh copy of the app. The caller quits the current one."""
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        [sys.executable, str(APP_DIR / "main.py")], cwd=APP_DIR,
        creationflags=flags, close_fds=True, env=os.environ.copy(),
    )


class UpdateWorker(QObject):
    """Runs a check or an update off the GUI thread (fetch/pull can take
    several seconds, and pip much longer)."""

    checked = Signal(object)   # UpdateInfo
    applied = Signal(bool)     # requirements were reinstalled
    failed = Signal(str)

    def __init__(self, mode: str):
        super().__init__()
        self._mode = mode

    def run(self) -> None:
        try:
            if self._mode == "check":
                self.checked.emit(check_for_update())
            else:
                self.applied.emit(apply_update())
        except UpdateError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - never let an update crash the app
            logger.exception("Update %s failed", self._mode)
            self.failed.emit(f"Unexpected error: {exc}")
