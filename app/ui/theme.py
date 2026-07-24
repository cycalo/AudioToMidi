"""HitMap visual theme — colors and global Qt stylesheet."""

from __future__ import annotations

# Graphite + teal pro-audio palette (see design spec).
COLOR_SURFACE = "#1a1d24"
COLOR_SURFACE_RAISED = "#22262f"
COLOR_STAGE = "#0f1218"
COLOR_BORDER = "#2a3140"
COLOR_ACCENT = "#14b8a6"
COLOR_ACCENT_HOVER = "#2dd4bf"
COLOR_ACCENT_TEXT = "#042f2e"
COLOR_SECONDARY = "#0ea5e9"
COLOR_SECONDARY_HOVER = "#38bdf8"
COLOR_SECONDARY_TEXT = "#0c4a6e"
COLOR_TEXT = "#e2e8f0"
COLOR_TEXT_SECONDARY = "#94a3b8"
COLOR_TEXT_MUTED = "#64748b"
COLOR_WARNING = "#fbbf24"
COLOR_WAVEFORM = "#5eead4"
COLOR_PLAYHEAD = "#f8fafc"

FONT_UI = '"Segoe UI", "Helvetica Neue", sans-serif'
FONT_DISPLAY = 'Georgia, "Palatino Linotype", "Times New Roman", serif'


def app_stylesheet() -> str:
    """Return the global QSS applied to the QApplication."""
    return f"""
    QWidget {{
        background-color: {COLOR_SURFACE};
        color: {COLOR_TEXT};
        font-family: {FONT_UI};
        font-size: 13px;
    }}
    QMainWindow {{
        background-color: {COLOR_SURFACE};
    }}
    QLabel {{
        background: transparent;
        color: {COLOR_TEXT_SECONDARY};
    }}
    QLabel#brandTitle {{
        color: {COLOR_WAVEFORM};
        font-family: {FONT_DISPLAY};
        font-size: 28px;
        font-weight: 400;
        letter-spacing: 0.5px;
    }}
    QLabel#brandTagline {{
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
        padding-bottom: 4px;
    }}
    QLabel#sessionMeta {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: 12px;
        padding-bottom: 8px;
    }}
    QLabel#sectionLabel {{
        color: {COLOR_TEXT_MUTED};
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding-top: 10px;
        padding-bottom: 4px;
    }}
    QLabel#statusLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: 12px;
    }}
    QLabel#warningLabel {{
        color: {COLOR_WARNING};
        font-size: 11px;
    }}
    QLabel#emptyHintTitle {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: 15px;
    }}
    QLabel#emptyHintSub {{
        color: {COLOR_TEXT_MUTED};
        font-size: 12px;
    }}
    QLineEdit, QComboBox, QDoubleSpinBox {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 6px 10px;
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_ACCENT};
        selection-color: {COLOR_ACCENT_TEXT};
        min-height: 18px;
    }}
    QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled {{
        color: {COLOR_TEXT_MUTED};
        background-color: #181b22;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: center right;
        width: 28px;
        border-left: 1px solid {COLOR_BORDER};
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
        background-color: #2a3140;
    }}
    QComboBox::down-arrow {{
        width: 0;
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {COLOR_WAVEFORM};
    }}
    QComboBox#previewCombo {{
        padding-right: 4px;
    }}
    QLabel#comboHintLabel {{
        color: {COLOR_TEXT_MUTED};
        font-size: 11px;
        padding-bottom: 2px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        selection-background-color: {COLOR_ACCENT};
        selection-color: {COLOR_ACCENT_TEXT};
        color: {COLOR_TEXT};
        outline: none;
    }}
    QPushButton {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 8px 12px;
        color: {COLOR_TEXT};
        font-weight: 600;
    }}
    QPushButton:hover:!disabled {{
        background-color: #2c3340;
        border-color: #3d4658;
    }}
    QPushButton:disabled {{
        color: {COLOR_TEXT_MUTED};
        background-color: #181b22;
    }}
    QPushButton#primaryButton {{
        background-color: {COLOR_ACCENT};
        border: none;
        color: {COLOR_ACCENT_TEXT};
        font-weight: 700;
        padding: 10px 14px;
    }}
    QPushButton#primaryButton:hover:!disabled {{
        background-color: {COLOR_ACCENT_HOVER};
    }}
    QPushButton#primaryButton:disabled {{
        background-color: #0f766e;
        color: #134e4a;
    }}
    QPushButton#secondaryButton {{
        background-color: {COLOR_SECONDARY};
        border: none;
        color: {COLOR_SECONDARY_TEXT};
        font-weight: 700;
        padding: 10px 14px;
    }}
    QPushButton#secondaryButton:hover:!disabled {{
        background-color: {COLOR_SECONDARY_HOVER};
    }}
    QPushButton#secondaryButton:disabled {{
        background-color: #0369a1;
        color: #0c4a6e;
    }}
    QPushButton#ghostButton {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        color: {COLOR_TEXT};
        font-weight: 600;
    }}
    QPushButton#ghostButton:hover:!disabled {{
        background-color: #2c3340;
        border-color: #3d4658;
    }}
    QPushButton#transportButton {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_WAVEFORM};
        color: {COLOR_TEXT};
        font-weight: 700;
        padding: 10px 14px;
    }}
    QPushButton#transportButton:hover:!disabled {{
        background-color: #2c3340;
        border-color: {COLOR_ACCENT_HOVER};
        color: {COLOR_WAVEFORM};
    }}
    QPushButton#transportButton:disabled {{
        background-color: #181b22;
        border: 1px solid {COLOR_BORDER};
        color: {COLOR_TEXT_MUTED};
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: {COLOR_SURFACE_RAISED};
        border-radius: 3px;
        border: 1px solid {COLOR_BORDER};
    }}
    QSlider::handle:horizontal {{
        width: 14px;
        height: 14px;
        margin: -5px 0;
        background: {COLOR_WAVEFORM};
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{
        background: {COLOR_ACCENT};
        border-radius: 3px;
    }}
    QProgressBar {{
        background-color: {COLOR_SURFACE_RAISED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        text-align: center;
        color: {COLOR_TEXT};
        min-height: 16px;
    }}
    QProgressBar::chunk {{
        background-color: {COLOR_ACCENT};
        border-radius: 5px;
    }}
    QFrame#railPanel {{
        background-color: {COLOR_SURFACE};
        border-right: 1px solid {COLOR_BORDER};
    }}
    QFrame#stagePanel {{
        background-color: {COLOR_STAGE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    """
