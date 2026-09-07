"""Dialog for watching YouTube channels and auto-transcribing new episodes."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.channel_processor import ChannelCheckWorker
from core.channel_watcher import add_channel, list_channels, remove_channel
from ui.channel_videos_dialog import ChannelVideosDialog

DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "YouTube Channel Transcripts"


class ChannelManagerDialog(QDialog):
    """Non-modal: stays open (and keeps checking) while the main window is
    still usable for one-off transcriptions. Kept alive by MainWindow
    holding a persistent reference rather than recreating it each time --
    closing this window only hides it, so a background check in progress
    can keep safely delivering signals to it either way."""

    # Emitted with a video's URL when the user picks one to manually
    # re-run from the video browser -- MainWindow handles actually running
    # it, through the exact same single-video pipeline (progress bar,
    # results table, autosave, speaker rename, export) a pasted URL gets,
    # rather than this dialog running a second parallel copy of it.
    video_rerun_requested = Signal(str)

    def __init__(
        self,
        parent,
        model_size: str,
        language: Optional[str],
        diarize: bool,
        hf_token: Optional[str],
    ):
        super().__init__(parent)
        self.setWindowTitle("YouTube Channel Watcher")
        self.resize(560, 480)
        self._model_size = model_size
        self._language = language
        self._diarize = diarize
        self._hf_token = hf_token
        self._settings = QSettings("TranscriberGUI", "TranscriberGUI")
        self._worker: Optional[ChannelCheckWorker] = None
        self._thread: Optional[QThread] = None

        self._build_ui()
        self._reload_channel_list()

    def update_transcription_settings(
        self, model_size: str, language: Optional[str], diarize: bool, hf_token: Optional[str]
    ) -> None:
        """Called each time the dialog is (re)opened, so it always uses
        whatever model/language/diarization settings are currently chosen
        in the main window rather than whatever was picked when it was
        first created."""
        self._model_size = model_size
        self._language = language
        self._diarize = diarize
        self._hf_token = hf_token

    def closeEvent(self, event) -> None:
        # Closing this window previously left an active channel check
        # running invisibly in the background -- confirmed, there was no
        # override here at all before this. terminate() immediately
        # rather than the graceful "cooperative cancel, then force-stop
        # after a delay" dance used for the visible Cancel button: the
        # user closing the window wants everything stopped now, and a
        # delayed QTimer might not even get a chance to fire once the
        # window (and possibly the whole app, if this was the last one)
        # is gone.
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.request_cancel()
            self._thread.terminate()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- management view: channel list, add/remove, output folder --
        # hidden while a check/re-run is actively in progress, so the
        # progress view below isn't competing for attention with controls
        # that don't apply mid-run.
        self.management_panel = QWidget()
        management_layout = QVBoxLayout(self.management_panel)
        management_layout.setContentsMargins(0, 0, 0, 0)

        intro = QLabel(
            "Adding a channel records its current videos as a baseline -- only "
            "videos published after that point will ever be transcribed "
            "automatically, so a long-running show's back catalog is never "
            "processed all at once. Double-click a channel to see exactly "
            "which videos it's tracking, and optionally re-run one of them."
        )
        intro.setWordWrap(True)
        management_layout.addWidget(intro)

        self.channel_list = QListWidget()
        self.channel_list.itemDoubleClicked.connect(self._open_channel_videos)
        management_layout.addWidget(self.channel_list, stretch=1)

        channel_row = QHBoxLayout()
        add_button = QPushButton("Add channel...")
        add_button.clicked.connect(self._add_channel)
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_channel)
        channel_row.addWidget(add_button)
        channel_row.addWidget(remove_button)
        management_layout.addLayout(channel_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Save transcripts to:"))
        self.output_dir_label = QLabel(self._output_dir_str())
        self.output_dir_label.setWordWrap(True)
        output_row.addWidget(self.output_dir_label, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(browse_button)
        management_layout.addLayout(output_row)

        layout.addWidget(self.management_panel)

        action_row = QHBoxLayout()
        self.check_button = QPushButton("Check for new episodes now")
        # Not a direct connection: QPushButton.clicked emits a bool, which
        # would otherwise land in _start_check's explicit_queue parameter
        # (confirmed directly) now that it has one.
        self.check_button.clicked.connect(lambda: self._start_check())
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_check)
        action_row.addWidget(self.check_button)
        action_row.addWidget(self.cancel_button)
        layout.addLayout(action_row)

        # --- progress view: takes over as the dialog's main focus while
        # something is actually running ---
        self.progress_status_label = QLabel("")
        self.progress_status_label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        self.progress_status_label.setWordWrap(True)
        self.progress_status_label.setVisible(False)
        layout.addWidget(self.progress_status_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Channel list management
    # ------------------------------------------------------------------
    def _output_dir_str(self) -> str:
        return self._settings.value("channel_output_dir", str(DEFAULT_OUTPUT_DIR), type=str)

    def _browse_output_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", self._output_dir_str())
        if chosen:
            self._settings.setValue("channel_output_dir", chosen)
            self.output_dir_label.setText(chosen)

    def _reload_channel_list(self) -> None:
        self.channel_list.clear()
        for url, info in list_channels().items():
            item = QListWidgetItem(f"{info['name']}  ({len(info['seen_videos'])} tracked)")
            item.setData(Qt.ItemDataRole.UserRole, url)
            self.channel_list.addItem(item)

    def _open_channel_videos(self, item: QListWidgetItem) -> None:
        url = item.data(Qt.ItemDataRole.UserRole)
        name = list_channels().get(url, {}).get("name", url)
        dialog = ChannelVideosDialog(self, url, name)
        if dialog.exec() and dialog.selected_video is not None:
            # Re-running a video hands off to the main window's own
            # single-video pipeline (same progress bar, results table,
            # autosave, everything a pasted URL gets) rather than running
            # a second parallel copy of it in here -- so this whole
            # window's job is done once a video is picked.
            self.video_rerun_requested.emit(dialog.selected_video["url"])
            self.close()

    def _add_channel(self) -> None:
        url, ok = QInputDialog.getText(
            self, "Add channel", "Channel URL (e.g. https://www.youtube.com/@channelname):"
        )
        url = url.strip()
        if not ok or not url:
            return
        self.setEnabled(False)
        try:
            record = add_channel(url)
        except Exception as exc:  # noqa: BLE001 - show whatever yt-dlp/network failure occurred
            QMessageBox.critical(self, "Couldn't add channel", str(exc))
            return
        finally:
            self.setEnabled(True)
        self._reload_channel_list()
        QMessageBox.information(
            self,
            "Channel added",
            f"Now watching \"{record['name']}\". Its current {len(record['seen_videos'])} "
            "video(s) are the baseline -- only videos published after now will be transcribed.",
        )

    def _remove_channel(self) -> None:
        item = self.channel_list.currentItem()
        if item is None:
            return
        remove_channel(item.data(Qt.ItemDataRole.UserRole))
        self._reload_channel_list()

    # ------------------------------------------------------------------
    # Checking / processing
    # ------------------------------------------------------------------
    def _start_check(self) -> None:
        self.check_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.log_view.clear()
        self.management_panel.setVisible(False)
        self.progress_status_label.setVisible(True)
        self.progress_status_label.setText("Starting...")

        self._worker = ChannelCheckWorker(
            output_dir=self._output_dir_str(),
            model_size=self._model_size,
            language=self._language,
            diarize=self._diarize,
            hf_token=self._hf_token,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        worker = self._worker
        thread = self._thread

        # Real bound methods connected directly (not wrapped in a closure)
        # so Qt's auto-connection logic can detect these signals cross
        # threads and marshal them onto the GUI thread -- see the detailed
        # note in main_window.py's _start_transcription for why that
        # distinction matters (a real, previously-shipped crash).
        thread.started.connect(worker.run)
        worker.log.connect(self._on_log)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._cleanup_thread)

        thread.start()

    def _cancel_check(self) -> None:
        if self._worker is None:
            return
        self.cancel_button.setEnabled(False)
        self._on_log("Cancelling after the current video finishes...")
        self._worker.request_cancel()

    def _on_log(self, message: str) -> None:
        self.log_view.append(message)
        # The status label mirrors whatever the log's latest line is --
        # that's the one thing worth being big and hard to miss while
        # this is running; the full log stays available below for detail.
        self.progress_status_label.setText(message.strip())

    def _on_finished(self, count: int) -> None:
        if self.sender() is not self._worker:
            return
        self.progress_status_label.setText(f"Done. {count} video(s) transcribed.")
        self._reload_channel_list()

    def _on_failed(self, message: str) -> None:
        if self.sender() is not self._worker:
            return
        self._on_log(f"Failed: {message}")

    def _on_cancelled(self) -> None:
        if self.sender() is not self._worker:
            return
        self._on_log("Cancelled.")
        self._reload_channel_list()

    def _cleanup_thread(self) -> None:
        if self.sender() is not self._thread:
            return
        self.check_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)
        self.management_panel.setVisible(True)
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
