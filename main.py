"""Entry point for TranscriberGUI."""
import os
import sys

# The real cause of a "'NoneType' object has no attribute 'write'" crash
# during the HF token check, found via the full traceback in app.log: a
# tqdm progress bar (from inside huggingface_hub's hf_hub_download) tries
# to print download progress to sys.stderr -- which pythonw.exe sets to
# None, not just a closed/redirected stream, since there's no console at
# all. This isn't specific to that one call site: *any* dependency that
# tries to print() or write a progress bar under pythonw hits the same
# crash. Command-line testing during development never caught this,
# since python.exe (used for those tests) has a real console and thus a
# real, non-None stderr -- only pythonw.exe (what the app actually
# launches as) is affected. Redirecting to a null stream here, before
# anything else runs, fixes the whole class of bug at once rather than
# chasing individual libraries that print.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

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
