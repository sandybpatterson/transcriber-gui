"""Scrollable, searchable, editable panel showing timestamped transcript segments."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.engine import Segment


def _format_hms(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ResultsView(QWidget):
    """Timestamped transcript table with search and inline text editing."""

    COLUMNS = ("Start", "Speaker", "Text")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._segments: list[Segment] = []

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search transcript...")
        self.search_box.textChanged.connect(self._filter_rows)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_item_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(self.search_box)
        layout.addWidget(self.table)

    def clear(self) -> None:
        self._segments = []
        self.table.setRowCount(0)

    def add_segment(self, segment: Segment) -> None:
        self._segments.append(segment)
        self._append_row(segment)

    def set_segments(self, segments: list[Segment]) -> None:
        self.clear()
        for segment in segments:
            self.add_segment(segment)

    def segments(self) -> list[Segment]:
        return list(self._segments)

    def _append_row(self, segment: Segment) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        start_item = QTableWidgetItem(_format_hms(segment.start))
        start_item.setFlags(start_item.flags() & ~Qt.ItemIsEditable)

        self.table.blockSignals(True)
        self.table.setItem(row, 0, start_item)
        self.table.setItem(row, 1, QTableWidgetItem(segment.speaker or ""))
        self.table.setItem(row, 2, QTableWidgetItem(segment.text))
        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if row >= len(self._segments):
            return
        segment = self._segments[row]
        if item.column() == 1:
            segment.speaker = item.text() or None
        elif item.column() == 2:
            segment.text = item.text()

    def _filter_rows(self, query: str) -> None:
        query = query.lower()
        for row in range(self.table.rowCount()):
            text_item = self.table.item(row, 2)
            speaker_item = self.table.item(row, 1)
            haystack = f"{text_item.text() if text_item else ''} {speaker_item.text() if speaker_item else ''}".lower()
            self.table.setRowHidden(row, query not in haystack)
