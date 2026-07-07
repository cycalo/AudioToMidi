"""Startup cleanup for orphaned audiotomidi_* temp stem directories."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import STEM_TEMP_PREFIX, cleanup_orphaned_stem_dirs  # noqa: E402


def test_orphaned_audiotomidi_folder_removed_on_cleanup(tmp_path: Path) -> None:
    orphan = tmp_path / f"{STEM_TEMP_PREFIX}deadbeef"
    orphan.mkdir()
    (orphan / "kick.wav").write_bytes(b"wav")

    found, removed = cleanup_orphaned_stem_dirs(tmp_path, log=lambda _msg: None)

    assert found == 1
    assert removed == 1
    assert not orphan.exists()


def test_non_audiotomidi_temp_folder_left_untouched(tmp_path: Path) -> None:
    other = tmp_path / "some_other_app_xyz"
    other.mkdir()
    (other / "data.txt").write_text("keep", encoding="utf-8")

    found, removed = cleanup_orphaned_stem_dirs(tmp_path, log=lambda _msg: None)

    assert found == 0
    assert removed == 0
    assert other.is_dir()
    assert (other / "data.txt").read_text(encoding="utf-8") == "keep"


def test_cleanup_skips_undeletable_folder_without_raising(tmp_path: Path) -> None:
    orphan_ok = tmp_path / f"{STEM_TEMP_PREFIX}ok"
    orphan_ok.mkdir()
    orphan_bad = tmp_path / f"{STEM_TEMP_PREFIX}locked"
    orphan_bad.mkdir()

    logs: list[str] = []
    real_rmtree = __import__("shutil").rmtree

    def rmtree_with_failure(path, *args, **kwargs):
        if Path(path) == orphan_bad:
            raise PermissionError("folder locked")
        real_rmtree(path, *args, **kwargs)

    with patch("app.main.shutil.rmtree", side_effect=rmtree_with_failure):
        found, removed = cleanup_orphaned_stem_dirs(tmp_path, log=logs.append)

    assert found == 2
    assert removed == 1
    assert not orphan_ok.exists()
    assert orphan_bad.is_dir()
    assert any("skipped" in msg for msg in logs)
