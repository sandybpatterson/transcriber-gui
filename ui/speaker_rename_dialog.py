"""Dialog to assign real names to detected SPEAKER_NN labels before saving."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class SpeakerRenameDialog(QDialog):
    """Lets the user rename each detected speaker label to a real name.

    There's no audio playback in this app, so each speaker is shown next
    to a couple of lines of what they actually said -- the only basis
    available here for telling voices apart.
    """

    def __init__(self, speaker_examples: dict[str, list[str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Name the speakers")
        self.setMinimumWidth(420)
        self._inputs: dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Diarization found these speakers. Enter a name for each "
            "(leave blank to keep the original label)."
        ))

        form = QFormLayout()
        for speaker, examples in speaker_examples.items():
            preview_text = "  /  ".join(f'"{example}"' for example in examples) or "(no speech captured)"
            preview = QLabel(preview_text)
            preview.setWordWrap(True)
            preview.setStyleSheet("color: gray; font-style: italic;")

            field = QWidget()
            field_layout = QVBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 6)
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(speaker)
            field_layout.addWidget(preview)
            field_layout.addWidget(line_edit)

            form.addRow(speaker, field)
            self._inputs[speaker] = line_edit
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def rename_map(self) -> dict[str, str]:
        """Original label -> new name (falls back to the original label if left blank)."""
        return {
            speaker: (line_edit.text().strip() or speaker)
            for speaker, line_edit in self._inputs.items()
        }
