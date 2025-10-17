from PyQt6.QtCore import QTimer
from src.gui.setup_docks import SetupDocks, QMainWindow
from src.gui.widgets.loading_spinner import LoadingSpinner


class AnalysisWindow(SetupDocks):
    """
    The `AnalysisWindow` class extends `SetupDocks` to create a dedicated window for analysis
    within the application. It initializes the window with a title, a predefined size,
    and incorporates a loading spinner for visual feedback during startup.
    - `is_minimized` (bool): A flag indicating whether the window is currently minimized.
    """

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self, main_window:QMainWindow = None) -> None:
        """
        Initializes the `AnalysisWindow`.
        This constructor sets up the main window, applies a title, sets the initial size,
        and displays a loading spinner for a brief period to indicate activity.
        - `main_window` (`QMainWindow`, optional): A reference to the main application window.
          Defaults to `None`.
        """
        super().__init__(main_window=main_window)
        self.setWindowTitle("FEAnalysisApp v1.0 - Analysis Window")
        self.resize(1200, 800)
        self.is_minimized = False

        # ---------------------------------------------
        # LOADING SPINNER
        # ---------------------------------------------

        spinner = LoadingSpinner(self)
        spinner.start()
        QTimer.singleShot(2000, spinner.stop)
        print("##### FEAnalysisApp initialized, and its dependencies were inherited properly. #####")