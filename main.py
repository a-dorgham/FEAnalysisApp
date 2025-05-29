import sys, os
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

# ---------------------------------------------
# MAIN APPLICATION ENTRY POINT
# ---------------------------------------------

def main() -> None:
    """
    The main entry point of the application.
    This function initializes the QApplication, creates an instance of the MainWindow,
    displays it, and starts the application's event loop. It also clears the terminal
    before launching the GUI.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['QTWEBENGINE_DICTIONARIES_PATH'] = '/tmp'
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()