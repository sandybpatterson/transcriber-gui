"""Main application window: file picker, model/language options, transcription, export."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread
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

        self.start_button = QPushButton("Start transcription")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_transcription)
        layout.addWidget(self.start_button)

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

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.segment_ready.connect(self.results_view.add_segment)
        self._worker.finished.connect(self._on_transcription_finished)
        self._worker.failed.connect(self._on_transcription_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _cleanup_thread(self) -> None:
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
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
