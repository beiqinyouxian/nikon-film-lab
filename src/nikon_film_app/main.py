import sys

from PySide6 import QtGui, QtWidgets

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
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
