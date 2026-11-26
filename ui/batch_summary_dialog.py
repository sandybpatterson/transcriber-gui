"""End-of-batch summary: what finished, what was skipped, what failed --
and a way back in for the ones that didn't make it.

Shown when a batch finishes (or is cancelled part-way): one row per video
with its outcome, an "Open output folder" button, and a "Retry remaining"
button that re-runs only the videos that failed or never ran -- the ones
already saved are left alone, so a cancelled batch resumes instead of
starting over.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

# Result dicts carried from MainWindow: {"title", "status", "detail"}.
# status is one of "done" / "skipped" / "failed" / "cancelled".
STATUS_LABELS = {
    "done": "Done",
    "skipped": "Skipped — already saved",
    "failed": "Failed",
    "cancelled": "Not run",
}


class BatchSummaryDialog(QDialog):
    def __init__(self, results: list[dict], output_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch finished")
        self.resize(680, 420)
        self.retry_requested = False
        self._results = results
        self._output_dir = output_dir
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        done = sum(1 for r in self._results if r["status"] == "done")
        skipped = sum(1 for r in self._results if r["status"] == "skipped")
        failed = sum(1 for r in self._results if r["status"] in ("failed", "cancelled"))
        summary = f"{done} saved"
        if skipped:
            summary += f", {skipped} already saved"
        if failed:
            summary += f", {failed} didn't finish"
        layout.addWidget(QLabel(f"{summary} — {self._output_dir}"))

        self.table = QTableWidget(len(self._results), 3)
        self.table.setHorizontalHeaderLabels(["Video", "Status", "Details"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        for row, result in enumerate(self._results):
            title_item = QTableWidgetItem(result["title"])
            title_item.setFlags(title_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, title_item)
            status_item = QTableWidgetItem(STATUS_LABELS.get(result["status"], result["status"]))
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, status_item)
            detail_item = QTableWidgetItem(result.get("detail") or "")
            detail_item.setFlags(detail_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, detail_item)
        layout.addWidget(self.table, stretch=1)

        buttons_row = QHBoxLayout()
        open_button = QPushButton("Open output folder")
        open_button.clicked.connect(self._open_output_folder)
        buttons_row.addWidget(open_button)
        buttons_row.addStretch(1)
        self.retry_button = QPushButton("Retry remaining")
        self.retry_button.clicked.connect(self._on_retry)
        self.retry_button.setEnabled(failed > 0)
        buttons_row.addWidget(self.retry_button)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons_row.addWidget(buttons)
        layout.addLayout(buttons_row)

    def _open_output_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_dir)))

    def _on_retry(self) -> None:
        self.retry_requested = True
        self.accept()
