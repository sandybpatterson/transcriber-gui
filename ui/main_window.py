"""Main application window: file picker, model/language options, transcription, export."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSettings, QThread, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon
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

from core.diarization import verification_key
from core.engine import (
    COOKIES_FILE,
    COOKIES_FIREFOX,
    COOKIES_NONE,
    LANGUAGES,
    MODEL_SIZE_LABELS,
    MODEL_SIZES,
    TranscriptionWorker,
    is_url,
    url_fallback_name,
)
from exporters import export_docx, export_srt, export_txt
from ui.channel_manager_dialog import ChannelManagerDialog
from ui.diarization_setup_wizard import DiarizationSetupWizard
from ui.results_view import ResultsView
from ui.speaker_rename_dialog import SpeakerRenameDialog

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")
AUTOSAVE_DIR = Path.home() / "Documents" / "TranscriberGUI Autosaves"
ICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "icon.ico"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return slug[:80] or "transcript"


def _fallback_name(video_path: str) -> str:
    return url_fallback_name(video_path) if is_url(video_path) else Path(video_path).stem


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
        self.setWindowTitle("Scribe")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(900, 700)

        self.video_path: Optional[str] = None
        self._worker: Optional[TranscriptionWorker] = None
        self._thread: Optional[QThread] = None
        self._autosave_path: Optional[Path] = None
        # The video's real resolved title (from title_resolved), used for
        # the manual Export buttons' suggested filename -- without this,
        # a URL source fell back to Path(url).stem, producing garbage
        # like "watch?v=xyz" instead of the actual episode title.
        self._current_title: Optional[str] = None
        self._settings = QSettings("TranscriberGUI", "TranscriberGUI")
        self._channel_dialog: Optional[ChannelManagerDialog] = None

        # Batch state: set by _start_batch(), consumed as each item's
        # worker finishes/fails/cancels. None means "not running a batch"
        # -- every batch-aware handler below checks this first so the
        # normal single-video flow (the common case) stays completely
        # unaffected when it's None.
        self._batch_queue: Optional[list[dict]] = None
        self._batch_index: int = 0
        self._batch_output_dir: Optional[Path] = None
        self._batch_failures: list[str] = []

        self._build_ui()

    def closeEvent(self, event) -> None:
        # Confirmed: with no override here, closing the window while a
        # transcription was running left the background thread executing
        # invisibly -- the process doesn't stop just because its window
        # did. terminate() immediately rather than the graceful
        # "cooperative cancel, then force-stop after a delay" dance used
        # for the visible Cancel button: the user closing the window
        # wants everything stopped now, and a delayed QTimer might not
        # even get a chance to fire once the last window (and, by Qt's
        # default quitOnLastWindowClose, the whole app) is gone.
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.request_cancel()
            self._thread.terminate()
        # The channel watcher window is independent and may have its own
        # job running -- closing it here triggers its own closeEvent,
        # which stops that the same way.
        if self._channel_dialog is not None:
            self._channel_dialog.close()
        super().closeEvent(event)

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
        url_row.addWidget(QLabel("Or paste a video URL (YouTube, Facebook, etc.):"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.url_input.returnPressed.connect(self._use_url)
        url_row.addWidget(self.url_input, stretch=1)
        use_url_button = QPushButton("Use URL")
        use_url_button.clicked.connect(self._use_url)
        url_row.addWidget(use_url_button)
        layout.addLayout(url_row)

        channels_row = QHBoxLayout()
        channels_row.addStretch(1)
        batch_button = QPushButton("Transcribe multiple URLs...")
        batch_button.clicked.connect(self._open_batch_setup)
        channels_row.addWidget(batch_button)
        watch_channels_button = QPushButton("Watch YouTube Channels...")
        watch_channels_button.clicked.connect(self._open_channel_manager)
        channels_row.addWidget(watch_channels_button)
        layout.addLayout(channels_row)

        options_row = QHBoxLayout()

        model_col = QVBoxLayout()
        model_col.addWidget(QLabel("Transcription quality"))
        self.model_combo = QComboBox()
        for size in MODEL_SIZES:
            self.model_combo.addItem(MODEL_SIZE_LABELS.get(size, size), size)
        self.model_combo.setCurrentIndex(MODEL_SIZES.index("base"))
        model_col.addWidget(self.model_combo)
        options_row.addLayout(model_col)

        language_col = QVBoxLayout()
        language_col.addWidget(QLabel("Language"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(LANGUAGES.keys())
        language_col.addWidget(self.language_combo)
        options_row.addLayout(language_col)

        signin_col = QVBoxLayout()
        signin_col.addWidget(QLabel("Sign in for private videos"))
        self.cookies_combo = QComboBox()
        self.cookies_combo.addItem("Don't sign in", COOKIES_NONE)
        self.cookies_combo.addItem("Use my Firefox login", COOKIES_FIREFOX)
        self.cookies_combo.addItem("Use a cookies.txt file...", COOKIES_FILE)
        self.cookies_combo.setToolTip(
            "Most Facebook videos outside public Pages (friends-only posts, groups,\n"
            "many reels) can only be fetched while signed in. Firefox's login is read\n"
            "directly; Chrome and Edge lock theirs, so export a cookies.txt instead."
        )
        self._select_cookies_source(self._settings.value("cookies_source", COOKIES_NONE, type=str))
        self.cookies_combo.activated.connect(self._on_cookies_source_chosen)
        signin_col.addWidget(self.cookies_combo)
        options_row.addLayout(signin_col)

        layout.addLayout(options_row)

        self.diarize_checkbox = QCheckBox("Identify speakers (diarization)")
        layout.addWidget(self.diarize_checkbox)

        self.diarize_panel = QWidget()
        diarize_layout = QHBoxLayout(self.diarize_panel)
        diarize_layout.setContentsMargins(20, 0, 0, 8)

        # --- setup state: shown until the token is verified against every
        # required repo -- opens the step-by-step wizard rather than an
        # inline wall of instructions ---
        self.diarize_setup_widget = QWidget()
        setup_layout = QHBoxLayout(self.diarize_setup_widget)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_button = QPushButton("Set up speaker identification...")
        setup_button.clicked.connect(self._open_diarization_setup)
        setup_layout.addWidget(setup_button)
        setup_layout.addStretch(1)
        diarize_layout.addWidget(self.diarize_setup_widget)

        # --- verified state: shown once every required repo is confirmed ---
        self.diarize_verified_widget = QWidget()
        verified_layout = QHBoxLayout(self.diarize_verified_widget)
        verified_layout.setContentsMargins(0, 0, 0, 0)
        verified_layout.addWidget(QLabel("Speaker identification ready."))
        change_token_button = QPushButton("Change...")
        change_token_button.clicked.connect(self._open_diarization_setup)
        verified_layout.addWidget(change_token_button)
        verified_layout.addStretch(1)
        diarize_layout.addWidget(self.diarize_verified_widget)

        layout.addWidget(self.diarize_panel)
        self.diarize_panel.setVisible(False)
        self.diarize_checkbox.toggled.connect(self._on_diarize_toggled)

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

    def _select_cookies_source(self, source: str) -> None:
        index = self.cookies_combo.findData(source)
        self.cookies_combo.setCurrentIndex(index if index >= 0 else 0)

    def _on_cookies_source_chosen(self, _index: int) -> None:
        source = self.cookies_combo.currentData()
        if source == COOKIES_FILE:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select an exported cookies.txt file", "", "Cookie files (*.txt);;All files (*)"
            )
            if not path:
                # Backing out of the picker means "never mind" -- keep
                # whatever was in effect before rather than a file option
                # with no file behind it.
                self._select_cookies_source(self._settings.value("cookies_source", COOKIES_NONE, type=str))
                return
            self._settings.setValue("cookies_file", path)
        self._settings.setValue("cookies_source", source)

    def _get_hf_token(self) -> str:
        return self._settings.value("hf_token", "", type=str)

    def _on_diarize_toggled(self, checked: bool) -> None:
        self.diarize_panel.setVisible(checked)
        if checked:
            self._refresh_diarize_setup_state()

    def _refresh_diarize_setup_state(self) -> None:
        # "Verified" means this exact token already passed a real access
        # check against the current required-repos list -- the app adding
        # a newly-discovered repo falls back to setup state automatically,
        # since that hasn't been (re)tested yet.
        token = self._get_hf_token()
        verified_key = self._settings.value("hf_token_verified_key", "", type=str)
        is_verified = bool(token) and verification_key(token) == verified_key
        self.diarize_setup_widget.setVisible(not is_verified)
        self.diarize_verified_widget.setVisible(is_verified)

    def _open_diarization_setup(self) -> None:
        wizard = DiarizationSetupWizard(self, existing_token=self._get_hf_token())
        wizard.setup_completed.connect(lambda _token: self._refresh_diarize_setup_state())
        wizard.exec()

    def _open_channel_manager(self) -> None:
        language = LANGUAGES.get(self.language_combo.currentText())
        diarize = self.diarize_checkbox.isChecked()
        hf_token = self._get_hf_token() or None
        if self._channel_dialog is None:
            self._channel_dialog = ChannelManagerDialog(
                self, self.model_combo.currentData(), language, diarize, hf_token
            )
            self._channel_dialog.video_rerun_requested.connect(self._rerun_video_from_channel)
        else:
            self._channel_dialog.update_transcription_settings(
                self.model_combo.currentData(), language, diarize, hf_token
            )
        self._channel_dialog.show()
        self._channel_dialog.raise_()
        self._channel_dialog.activateWindow()

    # ------------------------------------------------------------------
    # Batch (multiple pasted URLs)
    # ------------------------------------------------------------------
    def _open_batch_setup(self) -> None:
        from ui.batch_setup_dialog import BatchSetupDialog

        dialog = BatchSetupDialog(self)
        if dialog.exec() and dialog.queue:
            self._start_batch(dialog.queue, dialog.output_dir)

    def _start_batch(self, queue: list[dict], output_dir: Path) -> None:
        if self._worker is not None:
            QMessageBox.warning(
                self, "Already running",
                "A transcription is already in progress. Wait for it to finish, or cancel it, first.",
            )
            return
        self._batch_queue = queue
        self._batch_index = 0
        self._batch_output_dir = output_dir
        self._batch_failures = []
        self._run_current_batch_item()

    def _run_current_batch_item(self) -> None:
        item = self._batch_queue[self._batch_index]
        self.url_input.setText(item["url"])
        self._use_url()
        self._start_transcription()

    def _batch_progress_text(self) -> str:
        return f"({self._batch_index + 1}/{len(self._batch_queue)})"

    def _export_current_batch_item(self, segments) -> None:
        item = self._batch_queue[self._batch_index]
        out_path = _unique_path(self._batch_output_dir, Path(item["filename"]).stem, ".txt")
        try:
            self._batch_output_dir.mkdir(parents=True, exist_ok=True)
            export_txt(segments, str(out_path))
        except OSError as exc:
            self._batch_failures.append(f"{item['filename']} (save failed: {exc})")

    def _advance_or_finish_batch(self) -> None:
        self._batch_index += 1
        if self._batch_index >= len(self._batch_queue):
            self._finish_batch()
            return
        self._run_current_batch_item()

    def _finish_batch(self) -> None:
        total = len(self._batch_queue)
        failed = len(self._batch_failures)
        summary = f"Batch complete: {total - failed}/{total} saved to {self._batch_output_dir}"
        if self._batch_failures:
            summary += "\nFailed: " + "; ".join(self._batch_failures)
        self.status_label.setText(summary)
        self._batch_queue = None
        self._batch_index = 0
        self._batch_output_dir = None
        self._batch_failures = []

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def _rerun_video_from_channel(self, url: str) -> None:
        """Handles "Re-run" from the channel video browser: loads the URL
        into the main window exactly as if it had been pasted, and runs it
        through this window's own pipeline rather than a second parallel
        one -- same progress bar, results table, autosave, everything."""
        self.raise_()
        self.activateWindow()
        self.url_input.setText(url)
        self._use_url()
        self._start_transcription()

    def _start_transcription(self) -> None:
        if not self.video_path:
            return
        if self._worker is not None:
            QMessageBox.warning(
                self, "Already running",
                "A transcription is already in progress. Wait for it to finish, or cancel it, first.",
            )
            return

        self.results_view.clear()
        self._set_export_enabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Starting...")
        self._current_title = None

        language = LANGUAGES.get(self.language_combo.currentText())

        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        self._autosave_path = _unique_path(
            AUTOSAVE_DIR, _slugify(_fallback_name(self.video_path)), ".txt"
        )

        self._worker = TranscriptionWorker(
            video_path=self.video_path,
            model_size=self.model_combo.currentData(),
            language=language,
            diarize=self.diarize_checkbox.isChecked(),
            hf_token=self._get_hf_token() or None,
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
        self._current_title = title
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
        # By this point self._worker/_thread are already None (set just
        # above), so the next item's _start_transcription() won't trip
        # the "already running" guard. A cancelled batch already cleared
        # _batch_queue back in _on_transcription_cancelled, so this only
        # fires here for a completed or failed item -- both continue.
        if self._batch_queue is not None:
            self._advance_or_finish_batch()

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
        if self._batch_queue is not None:
            message = f"{message} {self._batch_progress_text()}"
        self.status_label.setText(message)

    def _on_segment_progress(self, percent: int, seconds_done: float, total_seconds: float) -> None:
        if self.sender() is not self._worker:
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(percent)
        text = (
            f"Transcribing... {percent}% "
            f"({_format_hms(seconds_done)} / {_format_hms(total_seconds)})"
        )
        if self._batch_queue is not None:
            text = f"{text} {self._batch_progress_text()}"
        self.status_label.setText(text)

    def _on_transcription_finished(self, segments) -> None:
        if self.sender() is not self._worker:
            return
        self.results_view.set_segments(segments)
        if self._batch_queue is not None:
            # No interactive speaker-rename in batch mode -- the whole
            # point is walking away while several videos process
            # unattended, not babysitting a rename prompt per video.
            self._export_current_batch_item(segments)
            self._set_export_enabled(bool(segments))
            return
        self._maybe_rename_speakers(segments)
        self.status_label.setText(f"Done — {len(segments)} segments.")
        self._set_export_enabled(bool(segments))

    def _maybe_rename_speakers(self, segments) -> None:
        examples: dict[str, list[str]] = {}
        for segment in segments:
            if not segment.speaker:
                continue
            bucket = examples.setdefault(segment.speaker, [])
            if len(bucket) < 2:
                text = segment.text.strip()
                bucket.append(text if len(text) <= 80 else text[:77] + "...")
        if not examples:
            return

        dialog = SpeakerRenameDialog(examples, self)
        accepted = dialog.exec()
        if not accepted:
            return  # Cancel: leave the original SPEAKER_NN labels as transcribed.

        rename_map = dialog.rename_map()
        for segment in segments:
            if segment.speaker in rename_map:
                segment.speaker = rename_map[segment.speaker]

        if self._autosave_path is not None:
            try:
                export_txt(segments, str(self._autosave_path))
            except OSError:
                pass

    def _on_transcription_failed(self, message: str) -> None:
        if self.sender() is not self._worker:
            return
        if self._batch_queue is not None:
            # One bad link (private video, deleted, typo) shouldn't stop
            # the rest of the batch -- note it and keep going, same
            # philosophy as the channel watcher's per-video handling.
            item = self._batch_queue[self._batch_index]
            self._batch_failures.append(f"{item['filename']} ({message})")
            return
        self.status_label.setText("Failed.")
        note = self._autosave_note()
        QMessageBox.critical(self, "Transcription failed", f"{message}{note}")

    def _on_transcription_cancelled(self) -> None:
        if self.sender() is not self._worker:
            return
        if self._batch_queue is not None:
            done = self._batch_index
            total = len(self._batch_queue)
            self.status_label.setText(f"Batch cancelled after {done}/{total} video(s).")
            self._batch_queue = None
            self._batch_index = 0
            self._batch_output_dir = None
            self._batch_failures = []
            self._set_export_enabled(bool(self.results_view.segments()))
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
        # A URL source has no meaningful parent directory to default to,
        # and Path(url).stem produces garbage like "watch?v=xyz" -- not
        # the actual episode title. Use the real resolved title once
        # known, saving into Documents, matching what autosave already
        # does correctly for the same case.
        if self.video_path and is_url(self.video_path):
            stem = _slugify(self._current_title or _fallback_name(self.video_path))
            return str(Path.home() / "Documents" / f"{stem}{suffix}")
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
