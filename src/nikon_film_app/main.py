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
    # Dark theme stylesheet（偏中性灰，接近 Lightroom 面板灰）
    app.setStyleSheet("""
        QWidget { background-color: #3e3e3e; color: #e2e2e2; }
        QGroupBox {
          border: 1px solid #505050; border-radius: 6px; margin-top: 10px;
          padding-top: 8px; background-color: #4a4a4a;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #f0f0f0; }
        QLabel { color: #ededed; }
        QLineEdit, QComboBox, QAbstractSpinBox, QTextEdit {
          background-color: #565656; border: 1px solid #626262; padding: 4px 6px; border-radius: 4px; color: #f0f0f0;
        }
        QSlider::groove:horizontal { height: 5px; background: #5a5a5a; border-radius: 3px; }
        QSlider::handle:horizontal {
          background: #4f8cff; width: 12px; margin: -6px 0; border-radius: 6px; border: 1px solid #1f1f1f;
        }
        QSlider::sub-page:horizontal { background: #4f8cff; }
        QPushButton {
          background-color: #555555; border: 1px solid #6a6a6a; padding: 6px 10px; border-radius: 4px; color: #f3f3f3;
        }
        QPushButton:hover { background-color: #5f5f5f; }
        QPushButton:pressed { background-color: #515151; }
        QProgressBar { background: #515151; border: 1px solid #6a6a6a; border-radius: 4px; text-align: center; color: #f0f0f0; }
        QProgressBar::chunk { background-color: #4f8cff; }
        QScrollArea { background-color: #3e3e3e; border: none; }
        /* 现代化滚动条 */
        QScrollBar:vertical {
          background: #4a4a4a; width: 10px; margin: 6px 0 6px 0; border-radius: 5px;
        }
        QScrollBar::handle:vertical {
          background: #6a6a6a; min-height: 24px; border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover { background: #7a7a7a; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
          background: #4a4a4a; height: 10px; margin: 0 6px 0 6px; border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
          background: #6a6a6a; min-width: 24px; border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover { background: #7a7a7a; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
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
    # 各平台统一的略暗预览背景（比面板略暗，但不是纯黑）
    try:
        window.preview_label.setStyleSheet("background-color: #2d2d2d;")
    except Exception:
        pass
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
