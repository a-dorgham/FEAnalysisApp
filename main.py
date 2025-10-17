import sys, os
import qdarkstyle
from PyQt6.QtWidgets import QApplication
from src.gui.windows.main_window import MainWindow

# ---------------------------------------------
# MAIN APPLICATION ENTRY POINT
# ---------------------------------------------

def main() -> None:
    """
    The main entry point of the application.
    This function initializes the QApplication, creates an instance of the MainWindow,
    displays it, and starts the application's event loop. 
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['QTWEBENGINE_DICTIONARIES_PATH'] = '/tmp'
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()