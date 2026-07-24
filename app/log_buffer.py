"""In-memory ring buffer for GUI debug logs."""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Iterable, List, Optional


DEFAULT_MAX_LINES = 2000


class LogBuffer:
    """Thread-safe ring buffer of log lines for the debug dialog."""

    def __init__(self, max_lines: int = DEFAULT_MAX_LINES) -> None:
        self._max_lines = max(1, int(max_lines))
        self._lines: Deque[str] = deque(maxlen=self._max_lines)
        self._lock = threading.Lock()
        self._listeners: List[Callable[[str], None]] = []
        self._generation = 0

    def append(self, message: str) -> None:
        text = str(message).rstrip("\n")
        if not text:
            return
        new_lines = text.splitlines() or [text]
        with self._lock:
            for line in new_lines:
                self._lines.append(line)
            listeners = list(self._listeners)
        for line in new_lines:
            for listener in listeners:
                try:
                    listener(line)
                except Exception:  # noqa: BLE001 - never break logging for UI
                    pass

    def lines(self) -> List[str]:
        with self._lock:
            return list(self._lines)

    def text(self) -> str:
        return "\n".join(self.lines())

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._generation += 1

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def save(self, path: Path | str) -> None:
        content = self.text()
        Path(path).write_text(content + ("\n" if content else ""), encoding="utf-8")

    def add_listener(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass


_GLOBAL: Optional[LogBuffer] = None


def get_log_buffer() -> LogBuffer:
    """Return the process-wide log buffer, creating it on first use."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = LogBuffer()
    return _GLOBAL


class BufferLogHandler(logging.Handler):
    """Forward standard logging records into a ``LogBuffer``."""

    def __init__(self, buffer: LogBuffer) -> None:
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append(self.format(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)


def install_logging(buffer: Optional[LogBuffer] = None, *, level: int = logging.INFO) -> LogBuffer:
    """Attach a buffer handler to the root logger and return the buffer."""
    buf = buffer if buffer is not None else get_log_buffer()
    root = logging.getLogger()
    for existing in root.handlers:
        if isinstance(existing, BufferLogHandler) and existing._buffer is buf:
            return buf
    handler = BufferLogHandler(buf)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    handler.setLevel(level)
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    return buf


def log_lines(messages: Iterable[str], buffer: Optional[LogBuffer] = None) -> None:
    buf = buffer if buffer is not None else get_log_buffer()
    for message in messages:
        buf.append(message)
