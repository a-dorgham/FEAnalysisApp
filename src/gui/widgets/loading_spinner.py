from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout
from PyQt6.QtGui import QMovie, QGuiApplication
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
import os, sys


class LoadingSpinner(QDialog):
    """
    A custom QDialog class that displays a GIF animation as a loading spinner.
    This spinner is frameless, has a translucent background, is modal, and
    includes a fade-in animation for a smoother user experience.
    - `movie` (QMovie): The QMovie object responsible for playing the GIF animation.
    - `label` (QLabel): The QLabel widget that displays the QMovie.
    - `fade_animation` (QPropertyAnimation): The animation object for fading in the spinner.
    """

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self, parent: QDialog = None) -> None:
        """
        Initializes the `LoadingSpinner` dialog.
        Sets up the dialog's window flags for a frameless and translucent appearance,
        loads the spinner GIF, configures the QLabel to display it, and sets up
        a fade-in animation.
        - `parent` (`QDialog`, optional): The parent widget of the spinner. Defaults to `None`.
        - `FileNotFoundError`: If the spinner GIF file cannot be found at the specified path.
        """
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)

        # ---------------------------------------------
        # GIF LOADING
        # ---------------------------------------------

        spinner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "icons", "main_window/spinner.gif")

        if not os.path.exists(spinner_path):
            raise FileNotFoundError(f"Spinner GIF not found: {spinner_path}")
        self.movie = QMovie(spinner_path)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMovie(self.movie)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.resize(100, 100)

        # ---------------------------------------------
        # ANIMATION SETUP
        # ---------------------------------------------

        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(500)
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    # ---------------------------------------------
    # CONTROL METHODS
    # ---------------------------------------------

    def start(self) -> None:
        """
        Starts the loading spinner animation and displays the dialog.
        This method centers the spinner on the screen, starts the GIF animation,
        makes the dialog visible, and initiates the fade-in effect.
        """
        self.center_on_screen()
        self.movie.start()
        self.show()
        self.fade_animation.start()


    def stop(self) -> None:
        """
        Stops the loading spinner animation and closes the dialog.
        This method halts the GIF animation and accepts the dialog, effectively closing it.
        """
        self.movie.stop()
        self.accept()


    def center_on_screen(self) -> None:
        """
        Centers the spinner dialog on the primary screen.
        This method calculates the center point of the primary screen's available geometry
        and moves the dialog's frame to align its center with the screen's center.
        """
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        screen_center = screen_geometry.center()
        dialog_rect = self.frameGeometry()
        dialog_rect.moveCenter(screen_center)
        self.move(dialog_rect.topLeft())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    spinner = LoadingSpinner()
    spinner.start()
    QTimer.singleShot(5000, spinner.stop)
    sys.exit(app.exec())