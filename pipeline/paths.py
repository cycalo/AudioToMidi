"""Resolve the application resource root for dev and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the directory that contains ``mappings/``, ``Preview Kit/``, etc.

    - Dev / ``python app/main.py``: repository root.
    - PyInstaller frozen: ``sys._MEIPASS`` (onedir ``_internal`` folder).
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    # pipeline/paths.py -> pipeline/ -> repo root
    return Path(__file__).resolve().parent.parent
