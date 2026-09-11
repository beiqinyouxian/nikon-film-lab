import sys
from PySide6 import QtWidgets
from nikon_film_app.ui.main_window import MainWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Nikon Film Lab")
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

