"""Application entry point."""

from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess
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

from app.log_buffer import get_log_buffer, install_logging  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

STEM_TEMP_PREFIX = "audiotomidi_"


def ensure_stdio() -> None:
    """Provide writable stdout/stderr for PyInstaller windowed builds.

    With ``console=False`` (``runw.exe``), Windows leaves ``sys.stdout`` and
    ``sys.stderr`` as ``None``. Demucs/tqdm progress and ``print(..., file=sys.stderr)``
    then fail with ``'NoneType' object has no attribute 'write'``.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: SIM115


def suppress_windows_console_windows() -> None:
    """Reduce console flashes from child processes in frozen Windows builds.

    Torch/Demucs may spawn helpers via ``subprocess``. OR-ing
    ``CREATE_NO_WINDOW`` into ``Popen`` creation flags keeps those children
    headless when the parent is a windowed (``console=False``) PyInstaller exe.
    """
    if sys.platform != "win32":
        return
    if not getattr(sys, "frozen", False):
        return
    try:
        flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
        _orig_popen_init = subprocess.Popen.__init__

        def _popen_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | flags
            return _orig_popen_init(self, *args, **kwargs)

        subprocess.Popen.__init__ = _popen_init  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001 - never block app launch
        get_log_buffer().append("console suppression: skipped (unavailable)")


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
    multiprocessing.freeze_support()
    ensure_stdio()
    suppress_windows_console_windows()
    install_logging()
    cleanup_orphaned_stem_dirs(log=get_log_buffer().append)
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.controller.cleanup)
    window.show()

    if os.environ.get("AUDIOTOMIDI_SELFTEST") == "1":
        QTimer.singleShot(0, app.quit)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
