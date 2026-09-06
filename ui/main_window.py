"""Main application window: file picker, model/language options, transcription, export."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.engine import LANGUAGES, MODEL_SIZES, TranscriptionWorker
from exporters import export_docx, export_srt, export_txt
from ui.results_view import ResultsView

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")


class DropZone(QLabel):
    """A label that accepts a dragged-and-dropped video file."""

    def __init__(self, on_file_dropped, parent=None):
        super().__init__(parent)
        self._on_file_dropped = on_file_dropped
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; padding: 20px; }"
        )
        self.setText("Drag and drop a video file here\n(or use Browse below)")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(VIDEO_EXTENSIONS):
                self._on_file_dropped(path)
                return
        QMessageBox.warning(
            self, "Unsupported file",
            f"Please drop a video file ({', '.join(VIDEO_EXTENSIONS)}).",
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TranscriberGUI")
        self.resize(900, 700)

        self.video_path: Optional[str] = None
        self._worker: Optional[TranscriptionWorker] = None
        self._thread: Optional[QThread] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        self.drop_zone = DropZone(self._set_video_path)
        layout.addWidget(self.drop_zone)

        browse_row = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_file)
        browse_row.addWidget(self.file_label, stretch=1)
        browse_row.addWidget(browse_button)
        layout.addLayout(browse_row)

        options_row = QHBoxLayout()

        model_col = QVBoxLayout()
        model_col.addWidget(QLabel("Model size"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_SIZES)
        self.model_combo.setCurrentText("base")
        model_col.addWidget(self.model_combo)
        options_row.addLayout(model_col)

        language_col = QVBoxLayout()
        language_col.addWidget(QLabel("Language"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(LANGUAGES.keys())
        language_col.addWidget(self.language_combo)
        options_row.addLayout(language_col)

        layout.addLayout(options_row)

        self.diarize_checkbox = QCheckBox("Identify speakers (diarization)")
        layout.addWidget(self.diarize_checkbox)

        start_row = QHBoxLayout()
        self.start_button = QPushButton("Start transcription")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_transcription)
        start_row.addWidget(self.start_button, stretch=1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_transcription)
        start_row.addWidget(self.cancel_button)
        layout.addLayout(start_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.results_view = ResultsView()
        layout.addWidget(self.results_view, stretch=1)

        export_row = QHBoxLayout()
        self.export_txt_button = QPushButton("Export .txt")
        self.export_srt_button = QPushButton("Export .srt")
        self.export_docx_button = QPushButton("Export .docx")
        for button, handler in (
            (self.export_txt_button, self._export_txt),
            (self.export_srt_button, self._export_srt),
            (self.export_docx_button, self._export_docx),
        ):
            button.setEnabled(False)
            button.clicked.connect(handler)
            export_row.addWidget(button)
        layout.addLayout(export_row)

        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------
    def _browse_file(self) -> None:
        pattern = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video file", "", f"Video files ({pattern})"
        )
        if path:
            self._set_video_path(path)

    def _set_video_path(self, path: str) -> None:
        self.video_path = path
        self.file_label.setText(Path(path).name)
        self.start_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def _start_transcription(self) -> None:
        if not self.video_path:
            return

        self.results_view.clear()
        self._set_export_enabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Starting...")

        language = LANGUAGES.get(self.language_combo.currentText())

        self._worker = TranscriptionWorker(
            video_path=self.video_path,
            model_size=self.model_combo.currentText(),
            language=language,
            diarize=self.diarize_checkbox.isChecked(),
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        # A force-stopped worker's OS thread can't reliably be disconnected
        # either (see _force_stop_if_still_running) — disconnect() blocks the
        # same way wait() does against a thread with no cancellation point.
        # So instead of relying on disconnecting a stale worker, every handler
        # that touches UI state is guarded: it checks the emitting worker is
        # still *the* current one before doing anything. A signal from an
        # abandoned worker that fires late is then simply inert, with no
        # blocking call required to make it so.
        worker = self._worker
        thread = self._thread

        def guarded(handler):
            def wrapper(*args):
                if self._worker is worker:
                    handler(*args)

            return wrapper

        thread.started.connect(worker.run)
        worker.progress.connect(guarded(self.status_label.setText))
        worker.segment_ready.connect(guarded(self.results_view.add_segment))
        worker.finished.connect(guarded(self._on_transcription_finished))
        worker.failed.connect(guarded(self._on_transcription_failed))
        worker.cancelled.connect(guarded(self._on_transcription_cancelled))
        # These three only tell the worker's own thread to quit -- harmless
        # even if stale, and bound to this specific thread regardless.
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(guarded(self._cleanup_thread))

        thread.start()

    def _cancel_transcription(self) -> None:
        if self._worker is None or self._thread is None:
            return
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling...")
        self._worker.request_cancel()

        # The worker checks for cancellation between pipeline steps, but it can
        # also be stuck in a single blocking call (a model download with a dead
        # connection, a very slow ffmpeg run) that never reaches a checkpoint.
        # Give it a couple of seconds to stop cleanly, then force it — but only
        # this specific run, in case a new one has since been started.
        watched_thread = self._thread
        watched_worker = self._worker
        QTimer.singleShot(
            2500, lambda: self._force_stop_if_still_running(watched_thread, watched_worker)
        )

    def _force_stop_if_still_running(self, watched_thread: QThread, watched_worker: TranscriptionWorker) -> None:
        if self._thread is not watched_thread or not watched_thread.isRunning():
            return

        # terminate() asks the OS to kill the thread, but a thread with no
        # cancellation point (a tight compute loop, some native calls) may
        # never actually respond to it. Critically, QThread.wait() is *not*
        # reliably bounded by its own timeout in that case either — measured
        # directly, wait(1000) can simply never return. So this never blocks
        # the GUI thread waiting for confirmation: terminate(), detach right
        # away, and let a non-blocking poll finish the cleanup later if the
        # OS thread ever actually dies.
        watched_thread.terminate()

        # No disconnect() call here on purpose: it carries the exact same
        # risk as wait() above -- confirmed directly, it can block just as
        # indefinitely against a thread with no cancellation point. Safety
        # against a late signal from this abandoned worker instead comes
        # from the identity guard set up in _start_transcription, which
        # needs no blocking call at all.
        #
        # Known residual limitation: if the worker thread is truly never
        # yielding the GIL at all (a tight pure-Python or non-releasing
        # native loop with no I/O), even this UI update can be delayed
        # behind it -- an inherent constraint of CPython's single GIL, not
        # something fixable from the application side short of moving
        # transcription into a separate process. In practice this doesn't
        # come up: ffmpeg extraction is subprocess/I/O-bound and faster-
        # whisper's native inference releases the GIL during computation
        # the same way blocking I/O does, so real hangs behave like the
        # network-stall / slow-I/O case this mechanism was built for.
        self.status_label.setText(
            "Cancelled — the background task could not be confirmed stopped "
            "and may still be finishing silently. Safe to start another one."
        )
        self._set_export_enabled(bool(self.results_view.segments()))
        self._detach_thread(delete=False)
        self._poll_for_stale_thread_cleanup(watched_thread, watched_worker, attempts_left=30)

    def _poll_for_stale_thread_cleanup(self, thread: QThread, worker: TranscriptionWorker, attempts_left: int) -> None:
        """Best-effort deferred cleanup for a thread abandoned by force-stop.

        Never blocks: only ever polls isRunning(), which is safe, unlike wait().
        """
        if not thread.isRunning():
            thread.deleteLater()
            worker.deleteLater()
            return
        if attempts_left <= 0:
            return  # Never died. Leaked for the life of the process; nothing more we can safely do.
        QTimer.singleShot(
            1000, lambda: self._poll_for_stale_thread_cleanup(thread, worker, attempts_left - 1)
        )

    def _cleanup_thread(self) -> None:
        self._detach_thread(delete=True)

    def _detach_thread(self, delete: bool) -> None:
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)
        self.start_button.setEnabled(True)
        # Only delete the Qt objects once we know the thread actually stopped —
        # deleting a QThread that is still executing is undefined behavior.
        # If it never stopped, drop the Python references and let it leak
        # rather than risk a crash or another hang.
        if delete:
            if self._worker is not None:
                self._worker.deleteLater()
            if self._thread is not None:
                self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def _on_transcription_finished(self, segments) -> None:
        self.results_view.set_segments(segments)
        self.status_label.setText(f"Done — {len(segments)} segments.")
        self._set_export_enabled(bool(segments))

    def _on_transcription_failed(self, message: str) -> None:
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "Transcription failed", message)

    def _on_transcription_cancelled(self) -> None:
        self.status_label.setText("Cancelled.")
        self._set_export_enabled(bool(self.results_view.segments()))

    def _set_export_enabled(self, enabled: bool) -> None:
        self.export_txt_button.setEnabled(enabled)
        self.export_srt_button.setEnabled(enabled)
        self.export_docx_button.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _default_export_path(self, suffix: str) -> str:
        stem = Path(self.video_path).stem if self.video_path else "transcript"
        return str(Path(self.video_path or ".").parent / f"{stem}{suffix}")

    def _export_txt(self) -> None:
        self._export(export_txt, "Text files (*.txt)", ".txt")

    def _export_srt(self) -> None:
        self._export(export_srt, "Subtitle files (*.srt)", ".srt")

    def _export_docx(self) -> None:
        self._export(export_docx, "Word documents (*.docx)", ".docx")

    def _export(self, export_fn, file_filter: str, suffix: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export transcript", self._default_export_path(suffix), file_filter
        )
        if not path:
            return
        try:
            export_fn(self.results_view.segments(), path)
            self.status_label.setText(f"Exported to {path}")
        except Exception as exc:  # noqa: BLE001 - show any export failure to the user
            QMessageBox.critical(self, "Export failed", str(exc))
