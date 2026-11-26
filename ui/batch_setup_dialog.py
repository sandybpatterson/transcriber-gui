"""Setup dialog for batch-transcribing multiple pasted URLs.

Confirms a filename per video and an output folder up front, then hands
the confirmed queue off to MainWindow to actually run -- one at a time,
through its own single-video pipeline (same progress bar, results table,
autosave) rather than a second parallel one in here.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QSettings, QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from core.engine import expand_playlist_url, fetch_video_info, is_url, url_fallback_name

DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "Batch Transcripts"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return slug[:80] or "transcript"


def _format_duration(seconds: float) -> str:
    """Seconds -> "1:23:45" / "12:34", or an em dash when unknown."""
    if seconds is None or seconds < 0:
        return "—"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


class TitleLookupWorker(QObject):
    """Expands playlist links into their videos, drops duplicate URLs,
    then fetches each video's real title and length (metadata only, no
    download) so the filename column can be pre-filled with something
    meaningful instead of a raw URL fragment."""

    queue_ready = Signal(list)  # final URL list, after expansion + dedupe
    progress = Signal(int, int)  # current (1-based), total
    title_found = Signal(int, str, float)  # row index, title, duration seconds (-1 if unknown)
    finished = Signal()

    def __init__(self, urls: list[str]):
        super().__init__()
        self.urls = urls

    def run(self) -> None:
        expanded: list[str] = []
        seen: set[str] = set()
        for url in self.urls:
            try:
                candidates = expand_playlist_url(url)
            except Exception:  # noqa: BLE001 - a playlist that can't be read is one video
                candidates = [url]
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    expanded.append(candidate)
        self.queue_ready.emit(expanded)

        total = len(expanded)
        for i, url in enumerate(expanded):
            self.progress.emit(i + 1, total)
            try:
                title, duration = fetch_video_info(url)
            except Exception:  # noqa: BLE001 - fall back to the URL itself as a last resort
                title, duration = url, -1
            self.title_found.emit(i, title, duration if duration is not None else -1)
        self.finished.emit()


class BatchSetupDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Transcribe Multiple URLs")
        self.resize(680, 580)
        self._settings = QSettings("TranscriberGUI", "TranscriberGUI")
        self.queue: list[dict] = []  # filled in on accept: [{"url", "filename"}, ...]
        self.formats: list[str] = [".txt"]  # filled in on accept
        self.output_dir: Optional[Path] = None
        self._urls: list[str] = []
        self._worker: Optional[TitleLookupWorker] = None
        self._thread: Optional[QThread] = None
        # Set when Start is pressed before the title lookup has run (or
        # while it's still running): the batch starts by itself once the
        # titles are in, instead of making the user press Start twice.
        self._start_when_ready = False
        # Set when the URL box changes while a lookup is already running:
        # the lookup in flight is for stale text, so a fresh one starts
        # once it finishes instead of leaving the table out of date.
        self._lookup_dirty = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Paste one video URL per line (a YouTube playlist link adds all its videos):"))
        self.url_box = QTextEdit()
        self.url_box.setPlaceholderText(
            "https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=..."
        )
        # Titles look themselves up shortly after pasting, so the
        # "Look up titles" button is a manual re-check, not a mandatory
        # step before Start works.
        self._lookup_timer = QTimer(self)
        self._lookup_timer.setSingleShot(True)
        self._lookup_timer.setInterval(800)
        self._lookup_timer.timeout.connect(self._start_lookup)
        self.url_box.textChanged.connect(self._on_urls_edited)
        layout.addWidget(self.url_box, stretch=1)

        lookup_row = QHBoxLayout()
        self.lookup_button = QPushButton("Look up titles")
        self.lookup_button.clicked.connect(self._start_lookup)
        lookup_row.addWidget(self.lookup_button)
        self.lookup_status_label = QLabel("")
        lookup_row.addWidget(self.lookup_status_label, stretch=1)
        layout.addLayout(lookup_row)

        layout.addWidget(QLabel("Confirm the file name for each video before starting:"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Video", "Length", "File name"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.table, stretch=2)

        formats_row = QHBoxLayout()
        formats_row.addWidget(QLabel("Save each transcript as:"))
        self.format_txt = QCheckBox(".txt")
        self.format_txt.setChecked(True)
        self.format_srt = QCheckBox(".srt")
        self.format_docx = QCheckBox(".docx")
        for box in (self.format_txt, self.format_srt, self.format_docx):
            formats_row.addWidget(box)
        formats_row.addStretch(1)
        layout.addLayout(formats_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Save transcripts to:"))
        self.output_dir_label = QLabel(self._output_dir_str())
        self.output_dir_label.setWordWrap(True)
        output_row.addWidget(self.output_dir_label, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(browse_button)
        layout.addLayout(output_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        self.start_button = QPushButton("Start Batch")
        self.start_button.clicked.connect(self._on_start)
        buttons.addButton(self.start_button, QDialogButtonBox.ButtonRole.AcceptRole)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Output folder
    # ------------------------------------------------------------------
    def _output_dir_str(self) -> str:
        return self._settings.value("batch_output_dir", str(DEFAULT_OUTPUT_DIR), type=str)

    def _browse_output_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", self._output_dir_str())
        if chosen:
            self._settings.setValue("batch_output_dir", chosen)
            self.output_dir_label.setText(chosen)

    # ------------------------------------------------------------------
    # Title lookup
    # ------------------------------------------------------------------
    def _on_urls_edited(self) -> None:
        # Restart the countdown on every keystroke; the lookup only fires
        # once pasting/typing has paused, and never while one is running.
        if self._thread is not None:
            self._lookup_dirty = True
            return
        self._lookup_timer.start()

    def _pasted_urls(self) -> list[str]:
        raw_lines = [line.strip() for line in self.url_box.toPlainText().splitlines()]
        return [line for line in raw_lines if line]

    def _start_lookup(self) -> None:
        if self._thread is not None:
            return
        self._lookup_timer.stop()
        urls = self._pasted_urls()
        invalid = [u for u in urls if not is_url(u)]
        if invalid:
            QMessageBox.warning(
                self, "Not all valid URLs",
                "These lines don't look like valid URLs:\n\n" + "\n".join(invalid[:10]),
            )
            return
        if not urls:
            if self._start_when_ready:
                self._start_when_ready = False
                QMessageBox.information(self, "Nothing to run", "Paste at least one URL first.")
            return

        self._urls = urls
        self._build_table(urls)

        self.lookup_button.setEnabled(False)
        self.lookup_status_label.setText(f"Looking up 0/{len(urls)}...")

        self._worker = TitleLookupWorker(urls)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        worker = self._worker
        thread = self._thread

        # Real bound methods, not closures -- see main_window.py's
        # _start_transcription for the detailed reasoning (a real,
        # previously-shipped crash from getting this wrong).
        thread.started.connect(worker.run)
        worker.queue_ready.connect(self._on_queue_ready)
        worker.progress.connect(self._on_lookup_progress)
        worker.title_found.connect(self._on_title_found)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._cleanup_lookup_thread)
        thread.start()

    def _build_table(self, urls: list[str]) -> None:
        """One row per URL, with a fallback file name until its real title
        arrives. Rebuilding here (rather than appending) is what lets a
        playlist expansion or a dropped duplicate reshape the list."""
        self.table.setRowCount(len(urls))
        for row, url in enumerate(urls):
            video_item = QTableWidgetItem(url)
            video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, video_item)
            length_item = QTableWidgetItem("…")
            length_item.setFlags(length_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, length_item)
            fallback_name = url_fallback_name(url)
            self.table.setItem(row, 2, QTableWidgetItem(_slugify(fallback_name)))

    def _on_queue_ready(self, urls: list[str]) -> None:
        if self.sender() is not self._worker:
            return
        # A playlist link expands to many rows (or a repeated URL collapses
        # to one) -- rebuild the table to match what will actually run.
        if urls != self._urls:
            self._urls = urls
            self._build_table(urls)
            self.lookup_status_label.setText(f"Found {len(urls)} video(s), looking up titles...")

    def _on_lookup_progress(self, current: int, total: int) -> None:
        if self.sender() is not self._worker:
            return
        self.lookup_status_label.setText(f"Looking up {current}/{total}...")

    def _on_title_found(self, row: int, title: str, duration: float) -> None:
        if self.sender() is not self._worker:
            return
        video_item = self.table.item(row, 0)
        if video_item is not None:
            video_item.setText(title)
        length_item = self.table.item(row, 1)
        if length_item is not None:
            length_item.setText(_format_duration(duration))
        self.table.setItem(row, 2, QTableWidgetItem(_slugify(title)))

    def _cleanup_lookup_thread(self) -> None:
        if self.sender() is not self._thread:
            return
        self.lookup_button.setEnabled(True)
        self.lookup_status_label.setText("Done.")
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        if self._lookup_dirty:
            # The URL box changed mid-lookup: run a fresh lookup for the
            # current text first. _start_when_ready stays set, so the
            # batch still starts by itself once that lookup finishes.
            self._lookup_dirty = False
            self._lookup_timer.start()
        elif self._start_when_ready:
            self._start_when_ready = False
            self._on_start()

    def closeEvent(self, event) -> None:
        # A lookup is cheap/short-lived (metadata only), but still: don't
        # leave a background thread running invisibly if the dialog closes
        # mid-lookup.
        self._lookup_timer.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.terminate()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Confirm and hand off
    # ------------------------------------------------------------------
    def _selected_formats(self) -> list[str]:
        formats = []
        for box, ext in (
            (self.format_txt, ".txt"),
            (self.format_srt, ".srt"),
            (self.format_docx, ".docx"),
        ):
            if box.isChecked():
                formats.append(ext)
        return formats or [".txt"]

    def _on_start(self) -> None:
        if self._thread is not None:
            # Titles are still being looked up -- start the batch the
            # moment they're in, instead of asking for a second click.
            self._start_when_ready = True
            return
        if self.table.rowCount() == 0:
            if self._pasted_urls():
                self._start_lookup()
                self._start_when_ready = True
                return
            QMessageBox.information(self, "Nothing to run", "Paste and look up at least one video first.")
            return

        queue = []
        for row in range(self.table.rowCount()):
            video_item = self.table.item(row, 0)
            title = video_item.text().strip() if video_item else ""
            filename_item = self.table.item(row, 2)
            filename = (filename_item.text().strip() if filename_item else "") or f"video_{row + 1}"
            queue.append({"url": self._urls[row], "filename": _slugify(filename), "title": title})

        self.queue = queue
        self.formats = self._selected_formats()
        self.output_dir = Path(self._output_dir_str())
        self.accept()
