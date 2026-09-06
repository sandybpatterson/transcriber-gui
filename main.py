"""Entry point for TranscriberGUI."""
import os
import sys

# Found live, by reading a user's actual app.log: downloading
# pyannote/speaker-diarization-3.1's config.yaml raised "'NoneType' object
# has no attribute 'write'" *after* the HEAD and GET requests both
# succeeded (200 OK) -- meaning access and the network transfer were
# both fine, and the failure was purely in huggingface_hub's newer Xet
# storage download path (this repo shows a Xet badge on its Files tab).
# A plain non-Xet download of a public file (bert-base-uncased's
# config.json) worked with no issue in the same environment, confirming
# the traditional HTTP path is reliable here. Must be set before
# huggingface_hub's constants module is first imported anywhere (it reads
# this env var at import time), hence set here at the very top of the
# entry point.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from PySide6.QtWidgets import QApplication

from core.app_logging import setup_logging
from ui.main_window import MainWindow


def main() -> int:
    setup_logging()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
