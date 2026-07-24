"""Tests for the in-memory debug log ring buffer."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.log_buffer import LogBuffer  # noqa: E402


def test_log_buffer_appends_and_joins() -> None:
    buf = LogBuffer(max_lines=10)
    buf.append("one")
    buf.append("two")
    assert buf.lines() == ["one", "two"]
    assert "one\ntwo" in buf.text()


def test_log_buffer_rings_when_over_capacity() -> None:
    buf = LogBuffer(max_lines=3)
    for i in range(5):
        buf.append(f"line-{i}")
    assert buf.lines() == ["line-2", "line-3", "line-4"]


def test_log_buffer_clear() -> None:
    buf = LogBuffer(max_lines=10)
    buf.append("x")
    buf.clear()
    assert buf.lines() == []
    assert buf.text() == ""


def test_log_buffer_save(tmp_path: Path) -> None:
    buf = LogBuffer(max_lines=10)
    buf.append("hello")
    buf.append("world")
    out = tmp_path / "debug.txt"
    buf.save(out)
    assert out.read_text(encoding="utf-8") == "hello\nworld\n"
