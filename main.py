"""Entry point for TranscriberGUI."""
import sys

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
