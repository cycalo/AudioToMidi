"""Main application window."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    """Primary window for the Drum Stem to MIDI application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Drum Stem to MIDI")
        self.resize(800, 600)

        central = QWidget(self)
        layout = QVBoxLayout(central)

        label = QLabel("Drum Stem to MIDI", central)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        self.setCentralWidget(central)
