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

from PySide6.QtCore import QObject, QSettings, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from core.engine import fetch_video_title, is_url

DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "Batch Transcripts"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return slug[:80] or "transcript"


class TitleLookupWorker(QObject):
    """Fetches each URL's real title (metadata only, no download) so the
    filename column can be pre-filled with something meaningful instead
    of a raw URL fragment."""

    progress = Signal(int, int)  # current (1-based), total
    title_found = Signal(int, str)  # row index, title
    finished = Signal()

    def __init__(self, urls: list[str]):
        super().__init__()
        self.urls = urls

    def run(self) -> None:
        total = len(self.urls)
        for i, url in enumerate(self.urls):
            self.progress.emit(i + 1, total)
            try:
                title = fetch_video_title(url)
            except Exception:  # noqa: BLE001 - fall back to the URL itself as a last resort
                title = url
            self.title_found.emit(i, title)
        self.finished.emit()


class BatchSetupDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Transcribe Multiple URLs")
        self.resize(640, 560)
        self._settings = QSettings("TranscriberGUI", "TranscriberGUI")
        self.queue: list[dict] = []  # filled in on accept: [{"url", "filename"}, ...]
        self.output_dir: Optional[Path] = None
        self._urls: list[str] = []
        self._worker: Optional[TitleLookupWorker] = None
        self._thread: Optional[QThread] = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Paste one video URL per line:"))
        self.url_box = QTextEdit()
        self.url_box.setPlaceholderText(
            "https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=..."
        )
        layout.addWidget(self.url_box, stretch=1)

        lookup_row = QHBoxLayout()
        self.lookup_button = QPushButton("Look up titles")
        self.lookup_button.clicked.connect(self._start_lookup)
        lookup_row.addWidget(self.lookup_button)
        self.lookup_status_label = QLabel("")
        lookup_row.addWidget(self.lookup_status_label, stretch=1)
        layout.addLayout(lookup_row)

        layout.addWidget(QLabel("Confirm the file name for each video before starting:"))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Video", "File name"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=2)

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
    def _start_lookup(self) -> None:
        raw_lines = [line.strip() for line in self.url_box.toPlainText().splitlines()]
        urls = [line for line in raw_lines if line]
        invalid = [u for u in urls if not is_url(u)]
        if invalid:
            QMessageBox.warning(
                self, "Not all valid URLs",
                "These lines don't look like valid URLs:\n\n" + "\n".join(invalid[:10]),
            )
            return
        if not urls:
            QMessageBox.information(self, "Nothing to look up", "Paste at least one URL first.")
            return

        self._urls = urls
        self.table.setRowCount(len(urls))
        for row, url in enumerate(urls):
            video_item = QTableWidgetItem(url)
            video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, video_item)
            fallback_name = url.rstrip("/").rsplit("/", 1)[-1]
            self.table.setItem(row, 1, QTableWidgetItem(_slugify(fallback_name)))

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
        worker.progress.connect(self._on_lookup_progress)
        worker.title_found.connect(self._on_title_found)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._cleanup_lookup_thread)
        thread.start()

    def _on_lookup_progress(self, current: int, total: int) -> None:
        if self.sender() is not self._worker:
            return
        self.lookup_status_label.setText(f"Looking up {current}/{total}...")

    def _on_title_found(self, row: int, title: str) -> None:
        if self.sender() is not self._worker:
            return
        video_item = self.table.item(row, 0)
        if video_item is not None:
            video_item.setText(title)
        self.table.setItem(row, 1, QTableWidgetItem(_slugify(title)))

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

    def closeEvent(self, event) -> None:
        # A lookup is cheap/short-lived (metadata only), but still: don't
        # leave a background thread running invisibly if the dialog closes
        # mid-lookup.
        if self._thread is not None and self._thread.isRunning():
            self._thread.terminate()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Confirm and hand off
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if self.table.rowCount() == 0:
            QMessageBox.information(self, "Nothing to run", "Paste and look up at least one video first.")
            return

        queue = []
        for row in range(self.table.rowCount()):
            filename_item = self.table.item(row, 1)
            filename = (filename_item.text().strip() if filename_item else "") or f"video_{row + 1}"
            queue.append({"url": self._urls[row], "filename": _slugify(filename)})

        self.queue = queue
        self.output_dir = Path(self._output_dir_str())
        self.accept()
