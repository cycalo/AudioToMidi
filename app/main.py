"""Application entry point."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

# Ensure repo root is on sys.path when running as `python app/main.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui.main_window import MainWindow  # noqa: E402

STEM_TEMP_PREFIX = "audiotomidi_"


def cleanup_orphaned_stem_dirs(
    temp_root: Optional[Path] = None,
    *,
    log: Callable[[str], None] = print,
) -> Tuple[int, int]:
    """Remove leftover ``audiotomidi_*`` temp dirs from a prior crashed session.

    A live instance only ever needs stems created during its own session; anything
    already on disk at launch is orphaned. Individual delete failures (permissions,
    folder still locked) are logged and skipped so startup always continues.

    Returns:
        ``(found, removed)`` counts of matching directories seen and deleted.
    """
    root = temp_root if temp_root is not None else Path(tempfile.gettempdir())
    found = 0
    removed = 0

    try:
        entries = list(root.iterdir())
    except OSError as exc:
        log(f"startup cleanup: could not scan {root} ({exc})")
        return 0, 0

    for entry in entries:
        if not entry.is_dir() or not entry.name.startswith(STEM_TEMP_PREFIX):
            continue
        found += 1
        try:
            shutil.rmtree(entry)
            removed += 1
        except OSError as exc:
            log(f"startup cleanup: skipped {entry} ({exc})")

    if found:
        log(
            f"startup cleanup: removed {removed}/{found} orphaned "
            f"{STEM_TEMP_PREFIX}* folder(s)"
        )
    return found, removed


def main() -> int:
    """Launch the application."""
    cleanup_orphaned_stem_dirs()
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.controller.cleanup)
    window.show()

    if os.environ.get("AUDIOTOMIDI_SELFTEST") == "1":
        QTimer.singleShot(0, app.quit)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
