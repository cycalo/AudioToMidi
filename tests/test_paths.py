"""Tests for resource-root resolution (dev vs frozen)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.paths import repo_root  # noqa: E402
from pipeline.remap import available_profiles  # noqa: E402


def test_repo_root_in_dev_contains_mappings() -> None:
    root = repo_root()
    assert (root / "mappings").is_dir()
    assert (root / "Preview Kit" / "kit.json").is_file()


def test_repo_root_frozen_uses_meipass(tmp_path: Path) -> None:
    mappings = tmp_path / "mappings"
    mappings.mkdir()
    (mappings / "ggd.json").write_text(
        '{"plugin":"GetGood Drums","confidence":"high","map":{"36":36}}',
        encoding="utf-8",
    )
    with patch.object(sys, "frozen", True, create=True), patch.object(
        sys, "_MEIPASS", str(tmp_path), create=True
    ):
        assert repo_root() == tmp_path


def test_available_profiles_nonempty_in_dev() -> None:
    profiles = available_profiles()
    assert "ggd" in profiles
    assert len(profiles) >= 7
