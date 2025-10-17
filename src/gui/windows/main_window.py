from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QPushButton, QLabel, QSizePolicy, QGridLayout
)
from PyQt6.QtGui import QPixmap, QIcon, QCloseEvent
from PyQt6.QtCore import Qt, QTimer
import os, sys
import functools
import traceback
from src.config import StructureConfig
from src.utils.http import HTTPServerThread
from src.gui.windows.analysis_window import AnalysisWindow
from src.constants import theme


class MainWindow(QWidget):
    """
    The `MainWindow` class serves as the main application window, allowing users to select a
    modelling system. It initializes an HTTP server in a separate thread, sets up the user
    interface with various structural analysis options represented by buttons, and manages
    the opening of the `AnalysisWindow` based on user selection.
    - `structure_type` (str | None): Stores the type of structure selected by the user (e.g., "Beam", "Truss").
    - `dofs_per_node` (int | None): Stores the degrees of freedom per node for the selected structure type.
    - `analysis_window` (AnalysisWindow | None): Holds an instance of the `AnalysisWindow` once it's opened.
    - `http_thread` (HTTPServerThread | None): Manages the HTTP server thread for the application.
    """

    # ---------------------------------------------
    # Class variables
    # ---------------------------------------------

    structure_type: str | None = None
    dofs_per_node: int | None = None
    analysis_window: AnalysisWindow | None = None
    http_thread: HTTPServerThread | None = None

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self) -> None:
        """
        Initializes the `MainWindow` class.
        This constructor sets up the main window, starts an HTTP server in a separate thread,
        and configures the user interface for selecting a modelling system.
        """
        super().__init__()
        MainWindow.http_thread = HTTPServerThread()
        MainWindow.http_thread.start()
        self.setup_main_window()
        self.center_window()

    # ---------------------------------------------
    # UI SETUP METHODS
    # ---------------------------------------------

    def setup_main_window(self) -> None:
        """
        Sets up the main window's title, geometry, and layout.
        This method configures the window title, its initial size and position,
        and arranges the main layout including the title label and the grid
        of selection buttons.
        """
        self.setWindowTitle('FEAnalysisApp v1.0 - Main Window')
        self.setGeometry(400, 300, 600, 300)
        layout = QVBoxLayout()
        title = QLabel('Available Systems')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        button_layout = QGridLayout()
        images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "icons", "main_window")
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"Icons folder not found: {images_folder}")
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for (structure, (row, col)) in zip(StructureConfig.load_config("structure").keys(), positions):
            image_file = f"{os.path.join(images_folder,structure)}.png"
            self.create_button(button_layout, structure, image_file, row, col)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        QApplication.instance().setStyleSheet(self._load_stylesheet(mode = theme))

    def center_window(self) -> None:
        """
        Centers the main window on the primary screen.
        This method calculates the center point of the screen and moves the window
        to that position, ensuring it's displayed centrally upon launch.
        """
        qt_rectangle = self.frameGeometry()
        center_point = QApplication.primaryScreen().geometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())


    def create_button(self, layout: QGridLayout, label_text: str, image_path: str, row: int, col: int) -> None:
        """
        Helper function to create a button with an image and a label, and add it to the layout.
        - `layout` (`QGridLayout`): The grid layout to which the button and label will be added.
        - `label_text` (str): The text to be displayed below the button and used for selection.
        - `image_path` (str): The file path to the image icon for the button.
        - `row` (int): The row index in the grid layout for the button.
        - `col` (int): The column index in the grid layout for the button.
        """
        button_layout = QVBoxLayout()
        button = QPushButton()
        button.setMinimumSize(150, 150)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        button.clicked.connect(functools.partial(self.open_new_window, label_text))
        pixmap = QPixmap(image_path).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(pixmap.rect().size())
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(button)
        button_layout.addWidget(label)
        layout.addLayout(button_layout, row, col)

    # ---------------------------------------------
    # STYLESHEET LOADING
    # ---------------------------------------------
    def _load_stylesheet(self, mode: str = "dark") -> str:
        """
        Loads the CSS stylesheet for the widget from a file.
        Includes paths to SVG icons within the stylesheet.

        Args:
            mode (str): The theme mode. Options are: "light", "dark".

        Returns:
            str: The loaded stylesheet content.
        """
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if mode == "light":
                styles_file = os.path.join(base_dir, "..", "..", "..", "assets", "styles", "light_mode.qss")

            else:
                styles_file = os.path.join(base_dir, "..", "..", "..", "assets", "styles", "dark_mode.qss")

            up_arrow_file: str = os.path.join(base_dir, "..", "..", "..", "assets", "icons", "analysis_window", "up_arrow.svg")
            down_arrow_file: str = os.path.join(base_dir, "..", "..", "..", "assets", "icons", "analysis_window", "down_arrow.svg")

            if not os.path.exists(styles_file):
                raise FileNotFoundError(f"Stylesheet file not found: {styles_file}")

            with open(styles_file, "r") as f:
                css: str = f.read()

            css = css.replace("__UP_ARROW_PATH__", up_arrow_file.replace(os.sep, "/"))
            css = css.replace(
                "__DOWN_ARROW_PATH__", down_arrow_file.replace(os.sep, "/")
            )
            print("Stylesheet was loaded successfully.")
            return css
        except FileNotFoundError as e:
            print(f"Error loading stylesheet from {styles_file}. \n{str(e)}\n{traceback.format_exc(-1)}")
            return ""
        except Exception as e:
            print(f"Error loading stylesheet from {styles_file}")
            return ""

    # ---------------------------------------------
    # WINDOW MANAGEMENT
    # ---------------------------------------------

    def open_new_window(self, label: str) -> None:
        """
        Opens the analysis window and hides the main window, based on the selected label.
        For specific labels (e.g., "3D_Solid", "2D_Plane"), this method does nothing.
        Otherwise, it sets the `structure_type` and `dofs_per_node` class variables
        and then displays the `AnalysisWindow`.
        - `label` (str): The text label associated with the clicked button, indicating
          the selected modelling system.
        """

        if label in {"3D_Solid", "2D_Plane"}:
            return

        def complete_loading() -> None:
            """
            Inner function to complete the loading process after a short delay.
            Sets the structure type and degrees of freedom per node, then
            initializes and shows the `AnalysisWindow`.
            """
            MainWindow.structure_type = label
            MainWindow.dofs_per_node = StructureConfig.load_config("structure")[label]["dofs_per_node"]
            MainWindow.analysis_window = AnalysisWindow(self)
            MainWindow.analysis_window.show()
        QTimer.singleShot(1, complete_loading)

    def _stop_http_server(self) -> None:
        """Stops the HTTP server thread."""
        if hasattr(self, 'http_thread') and self.http_thread.isRunning():
            self.http_thread.stop()
            self.http_thread.quit()
            if not self.http_thread.wait(1000):  
                print("HTTP thread did not terminate.")
            else:
                print("HTTP server stopped successfully.")

    def closeEvent(self, event:QCloseEvent) -> None:
        """
        Handles the close event for the main window.
        This method ensures that the HTTP server thread is stopped gracefully
        before the application closes.
        - `event` (`QCloseEvent`): The close event triggered when the window is being closed.
        """
        self._stop_http_server()
        event.accept()
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())