"""Shows the videos a watched channel is currently tracking as "seen",
and lets the user pick one to manually re-run through transcription --
useful right after adding a channel, when its whole baseline was marked
seen but never actually transcribed."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.channel_watcher import get_channel_videos


class ChannelVideosDialog(QDialog):
    def __init__(self, parent, channel_url: str, channel_name: str):
        super().__init__(parent)
        self.setWindowTitle(f"Videos tracked: {channel_name}")
        self.resize(480, 420)
        self.selected_video: Optional[dict] = None

        videos = get_channel_videos(channel_url)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"{len(videos)} video(s) this channel is currently tracking as "
            "already seen (so they won't be picked up automatically). Pick "
            "one and click \"Re-run\" to transcribe it manually anyway."
        ))

        self.list_widget = QListWidget()
        for video in videos:
            item = QListWidgetItem(video["title"])
            item.setData(Qt.ItemDataRole.UserRole, video)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, stretch=1)

        button_row = QHBoxLayout()
        rerun_button = QPushButton("Re-run selected video")
        rerun_button.clicked.connect(self._on_rerun)
        button_row.addWidget(rerun_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _on_rerun(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "No video selected", "Select a video from the list first.")
            return
        self.selected_video = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
