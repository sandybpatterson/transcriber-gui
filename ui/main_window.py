"""Main application window: file picker, model/language options, transcription, export."""
from __future__ import annotations

import re
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
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.engine import LANGUAGES, MODEL_SIZES, TranscriptionWorker, is_url
from exporters import export_docx, export_srt, export_txt
from ui.results_view import ResultsView

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")
AUTOSAVE_DIR = Path.home() / "Documents" / "TranscriberGUI Autosaves"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return slug[:80] or "transcript"


def _fallback_name(video_path: str) -> str:
    return video_path.rstrip("/").rsplit("/", 1)[-1] if is_url(video_path) else Path(video_path).stem


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """`stem.suffix`, or `stem (2).suffix` etc. if that name is already taken."""
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def _format_hms(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


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
        self._autosave_path: Optional[Path] = None

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

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("Or paste a video URL (YouTube, etc.):"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.url_input.returnPressed.connect(self._use_url)
        url_row.addWidget(self.url_input, stretch=1)
        use_url_button = QPushButton("Use URL")
        use_url_button.clicked.connect(self._use_url)
        url_row.addWidget(use_url_button)
        layout.addLayout(url_row)

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

    def _use_url(self) -> None:
        url = self.url_input.text().strip()
        if not url:
            return
        if not is_url(url):
            QMessageBox.warning(self, "Not a URL", "That doesn't look like a valid http(s) URL.")
            return
        self._set_video_path(url)

    def _set_video_path(self, path: str) -> None:
        self.video_path = path
        self.file_label.setText(path if is_url(path) else Path(path).name)
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

        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        self._autosave_path = _unique_path(
            AUTOSAVE_DIR, _slugify(_fallback_name(self.video_path)), ".txt"
        )

        self._worker = TranscriptionWorker(
            video_path=self.video_path,
            model_size=self.model_combo.currentText(),
            language=language,
            diarize=self.diarize_checkbox.isChecked(),
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        worker = self._worker
        thread = self._thread

        # Every handler below is a real bound method of `self` (a QObject
        # living on the GUI thread) rather than a wrapping closure -- that's
        # what lets Qt's auto-connection logic detect these signals cross
        # threads and marshal them onto the GUI thread automatically. A
        # bare closure has no attached QObject for Qt to inspect, so it
        # can't tell -- and silently runs the handler (and whatever widgets
        # it touches) directly on the worker's background thread instead.
        # That's a real crash found by running this for real: mutating
        # QWidgets off the GUI thread eventually segfaults, consistently
        # right after the burst of updates near the end of a transcription.
        # Wrapping in `type=Qt.ConnectionType.QueuedConnection` does NOT
        # fix this for a bare closure either -- confirmed directly, Qt still
        # needs a real QObject receiver to queue against.
        #
        # A force-stopped worker's OS thread can't reliably be disconnected
        # (see _force_stop_if_still_running) -- disconnect() blocks the same
        # way wait() does against a thread with no cancellation point. So
        # instead of relying on disconnecting a stale worker, every handler
        # checks self.sender() against the worker/thread that's still
        # current before doing anything -- a signal from an abandoned
        # worker that fires late is then simply inert, no blocking call
        # required.
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_phase_message)
        worker.title_resolved.connect(self._on_title_resolved)
        worker.segment_progress.connect(self._on_segment_progress)
        worker.segment_ready.connect(self._on_segment_ready)
        worker.finished.connect(self._on_transcription_finished)
        worker.failed.connect(self._on_transcription_failed)
        worker.cancelled.connect(self._on_transcription_cancelled)
        # These three only tell the worker's own thread to quit -- harmless
        # even if stale, and bound to this specific thread regardless.
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._cleanup_thread)

        thread.start()

    def _on_title_resolved(self, title: str) -> None:
        if self.sender() is not self._worker:
            return
        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        self._autosave_path = _unique_path(AUTOSAVE_DIR, _slugify(title), ".txt")

    def _on_segment_ready(self, segment) -> None:
        if self.sender() is not self._worker:
            return
        self.results_view.add_segment(segment)
        # Re-writes the whole file each time rather than appending a line --
        # simpler and reuses export_txt as-is, and transcripts are small
        # enough that this stays cheap. Never let a write failure (e.g. a
        # full disk) take down the transcription itself.
        if self._autosave_path is None:
            return
        try:
            export_txt(self.results_view.segments(), str(self._autosave_path))
        except OSError:
            pass

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
            f"{self._autosave_note()}"
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
        if self.sender() is not self._thread:
            return
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

    def _on_phase_message(self, message: str) -> None:
        if self.sender() is not self._worker:
            return
        # Reset to indeterminate for phases with no known progress (extraction,
        # diarization) -- _on_segment_progress switches it to determinate once
        # transcription starts reporting real percentages.
        self.progress_bar.setRange(0, 0)
        self.status_label.setText(message)

    def _on_segment_progress(self, percent: int, seconds_done: float, total_seconds: float) -> None:
        if self.sender() is not self._worker:
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(percent)
        self.status_label.setText(
            f"Transcribing... {percent}% "
            f"({_format_hms(seconds_done)} / {_format_hms(total_seconds)})"
        )

    def _on_transcription_finished(self, segments) -> None:
        if self.sender() is not self._worker:
            return
        self.results_view.set_segments(segments)
        self.status_label.setText(f"Done — {len(segments)} segments.")
        self._set_export_enabled(bool(segments))

    def _on_transcription_failed(self, message: str) -> None:
        if self.sender() is not self._worker:
            return
        self.status_label.setText("Failed.")
        note = self._autosave_note()
        QMessageBox.critical(self, "Transcription failed", f"{message}{note}")

    def _on_transcription_cancelled(self) -> None:
        if self.sender() is not self._worker:
            return
        self.status_label.setText(f"Cancelled.{self._autosave_note()}")
        self._set_export_enabled(bool(self.results_view.segments()))

    def _autosave_note(self) -> str:
        if self._autosave_path is not None and self.results_view.segments():
            return f"\n\nWhatever was transcribed so far was auto-saved to:\n{self._autosave_path}"
        return ""

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
