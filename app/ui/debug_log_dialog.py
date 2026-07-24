"""Help → View debug log dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.log_buffer import LogBuffer, get_log_buffer


class DebugLogDialog(QDialog):
    """Live view of the in-memory debug log with Save / Clear.

    Polls the buffer on a timer so worker-thread log appends stay thread-safe.
    """

    def __init__(self, parent: QWidget | None = None, *, buffer: LogBuffer | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Debug log")
        self.resize(720, 420)
        self._buffer = buffer if buffer is not None else get_log_buffer()
        self._seen = 0
        self._generation = self._buffer.generation()

        layout = QVBoxLayout(self)
        self.view = QPlainTextEdit(self)
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.view.setFont(font)
        layout.addWidget(self.view)

        buttons = QHBoxLayout()
        self.save_btn = QPushButton("Save log…", self)
        self.clear_btn = QPushButton("Clear", self)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close_btn = box.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        buttons.addWidget(box)
        layout.addLayout(buttons)

        self.save_btn.clicked.connect(self._on_save)
        self.clear_btn.clicked.connect(self._on_clear)

        lines = self._buffer.lines()
        self.view.setPlainText("\n".join(lines))
        self._seen = len(lines)
        self._scroll_to_end()

        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self) -> None:
        generation = self._buffer.generation()
        lines = self._buffer.lines()
        if generation != self._generation:
            self._generation = generation
            self.view.setPlainText("\n".join(lines))
            self._seen = len(lines)
            self._scroll_to_end()
            return
        if len(lines) > self._seen:
            chunk = "\n".join(lines[self._seen :])
            self.view.appendPlainText(chunk)
            self._seen = len(lines)
            self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.view.setTextCursor(cursor)

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save debug log", "audiotomidi-debug.txt", "Text files (*.txt)"
        )
        if not path:
            return
        try:
            self._buffer.save(Path(path))
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _on_clear(self) -> None:
        self._buffer.clear()
        self.view.clear()
        self._seen = 0
        self._generation = self._buffer.generation()
