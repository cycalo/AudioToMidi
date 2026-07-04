"""Application entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

# Ensure repo root is on sys.path when running as `python app/main.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    """Launch the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if os.environ.get("AUDIOTOMIDI_SELFTEST") == "1":
        QTimer.singleShot(0, app.quit)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
