"""File logging for TranscriberGUI.

The app runs via pythonw.exe (no console window), so without this, any
error or hang is completely invisible after the fact -- there's nowhere
to look. Call setup_logging() once at startup; every other module just
does `logger = logging.getLogger(__name__)` and logs normally.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

LOG_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "TranscriberGUI"
LOG_PATH = LOG_DIR / "app.log"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Uncaught exceptions would otherwise vanish silently under pythonw
    # (no console to print a traceback to) -- log them instead.
    def log_uncaught(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught
    root.info("=== TranscriberGUI starting (log: %s) ===", LOG_PATH)
