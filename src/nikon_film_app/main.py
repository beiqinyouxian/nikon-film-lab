import sys

from PySide6 import QtGui, QtWidgets
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QByteArray
import base64
import os

from nikon_film_app.ui.main_window import MainWindow


def _pick_ui_font() -> QtGui.QFont:
    """Prefer clean CJK UI fonts; fall back gracefully cross-platform."""
    preferred = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "Segoe UI",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial",
    ]
    db = QtGui.QFontDatabase()
    available = set(db.families())
    for name in preferred:
        if name in available:
            font = QtGui.QFont(name)
            font.setPointSize(10)
            font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
            return font
    font = QtGui.QFont()
    font.setPointSize(10)
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    return font


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Nikon Film Lab")
    app.setFont(_pick_ui_font())
    # Dark theme stylesheet (Photoshop/Lightroom 风格)
    app.setStyleSheet("""
        QWidget { background-color: #1e1e1e; color: #d0d0d0; }
        QGroupBox {
          border: 1px solid #333333; border-radius: 6px; margin-top: 10px;
          padding-top: 8px; background-color: #222222;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #e0e0e0; }
        QLabel { color: #cfcfcf; }
        QSlider::groove:horizontal { height: 4px; background: #3a3a3a; border-radius: 2px; }
        QSlider::handle:horizontal {
          background: #4f8cff; width: 12px; margin: -5px 0; border-radius: 6px; border: 1px solid #1a1a1a;
        }
        QSlider::sub-page:horizontal { background: #4f8cff; }
        QPushButton {
          background-color: #2b2b2b; border: 1px solid #3a3a3a; padding: 6px 10px; border-radius: 4px;
        }
        QPushButton:hover { background-color: #333333; }
        QPushButton:pressed { background-color: #2a2a2a; }
        QComboBox, QLineEdit {
          background-color: #262626; border: 1px solid #3a3a3a; padding: 4px 6px; border-radius: 4px;
        }
        QProgressBar { background: #2b2b2b; border: 1px solid #3a3a3a; border-radius: 4px; text-align: center; color: #cfcfcf; }
        QProgressBar::chunk { background-color: #4f8cff; }
        QScrollArea { background-color: #1e1e1e; border: none; }
    """)
    # Load window icon from assets or fallback
    def _load_app_icon() -> QIcon:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        assets_dir = os.path.join(base_dir, "..", "assets")
        png_path = os.path.abspath(os.path.join(assets_dir, "app_icon.png"))
        if os.path.exists(png_path):
            return QIcon(png_path)
        # fallback: light in center on dark
        pix = QPixmap(64, 64)
        pix.fill(QtGui.QColor("#1e1e1e"))
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        g = QtGui.QRadialGradient(pix.rect().center(), 22)
        g.setColorAt(0.0, QtGui.QColor("#ffb74d"))
        g.setColorAt(1.0, QtGui.QColor(30,30,30,0))
        p.fillRect(pix.rect(), g)
        p.setPen(QtGui.QPen(QtGui.QColor("#444"), 6))
        p.drawRoundedRect(6, 6, 52, 52, 10, 10)
        p.end()
        return QIcon(pix)
    app.setWindowIcon(_load_app_icon())
    window = MainWindow()
    # Dark preview background
    try:
        window.preview_label.setStyleSheet("background-color: #111111;")
    except Exception:
        pass
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
