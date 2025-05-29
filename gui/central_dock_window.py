from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTableWidget, QMainWindow, QDockWidget, QTabWidget, QSlider)
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView
from gui.results_viewer import ResultsViewer
from utils.utils_classes import SilentWebEnginePage


class CentralDockWindow(QWidget):
    """
    `CentralDockWindow` is a QMainWindow subclass designed to display and manage
    the results of structural analysis, including plots and detailed information.
    It integrates with `ResultsViewer` to visualize data and provides interactive
    controls for post-processing, such as animation of deformations and
    display of stiffness matrices, displacements, stresses, and forces.
    - `current_canvas` (`QWebEngineView | None`): A reference to the currently active
      QWebEngineView used for plotting.
    - `imported_data` (dict): A dictionary containing the imported structural analysis data.
    - `result_plotting` (ResultsViewer): An instance of the `ResultsViewer` class
      responsible for generating and updating plots.
    - `increasings_slider` (bool): A flag to control the animation direction.
    - `central_dock_widget` (QDockWidget): The main dock widget holding all
      post-processing content.
    - `tab_widget` (QTabWidget): A tabbed interface to switch between different
      result views (Geometry, Stiffness-Matrix, Displacements, Stresses, Forces, Information, File).
    - `frame_canvas` (QWebEngineView): Canvas for displaying the structural frame geometry.
    - `stiffness_tab` (QWidget): Tab for displaying the stiffness matrix.
    - `stiffness_tab_layout` (QVBoxLayout): Layout for the stiffness matrix tab.
    - `stiffness_table` (QTableWidget): Table to display the stiffness matrix.
    - `deformation_canvas` (QWebEngineView): Canvas for displaying structural deformations.
    - `force_canvas` (QWebEngineView): Canvas for displaying internal forces.
    - `stress_canvas` (QWebEngineView): Canvas for displaying internal stresses.
    - `info_tab` (QWidget): Tab for displaying solver reports and information.
    - `info_tab_layout` (QVBoxLayout): Layout for the information tab.
    - `info_canvas` (QWebEngineView): Canvas for displaying solver reports.
    - `file_tab` (QWidget): Tab for displaying input file details.
    - `file_tab_layout` (QVBoxLayout): Layout for the file tab.
    - `file_canvas` (QWebEngineView): Canvas for displaying input file content.
    - `force_tab_index` (int): The index of the "Forces" tab in the tab widget.
    - `animation_direction` (int): Controls the direction of the deformation animation (1 for forward, -1 for backward).
    - `current_frame_index` (int): The current frame index in the animation sequence.
    - `scale_slider` (QSlider): Slider to control the animation frame/scale factor.
    - `animation_timer` (QTimer): Timer to control the animation playback.
    - `play_pause_button` (QPushButton): Button to play or pause the animation.
    """
    current_canvas: QWebEngineView | None = None

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self, imported_data: dict, parent: QWidget = None) -> None:
        """
        Initializes the `CentralDockWindow`.
        - `imported_data` (dict): A dictionary containing the data imported
          from the structural analysis, used for plotting results.
        - `parent` (`QWidget`, optional): The parent widget. Defaults to `None`.
        """
        super().__init__(parent)
        self.imported_data = imported_data
        self.result_plotting = ResultsViewer(imported_data=self.imported_data, CentralDockWindow=self)
        self.increasings_slider = True
        self.setWindowTitle("FEAnalysisApp v1.0")
        self.setGeometry(100, 100, 1000, 600)
        layout = QVBoxLayout()
        layout.addLayout(self.create_post_process_layout())
        self.setLayout(layout)
        ResultsViewer.plot_truss()

    # ---------------------------------------------
    # LAYOUT CREATION
    # ---------------------------------------------

    def create_post_process_layout(self) -> QVBoxLayout:
        """
        Creates the main layout for the post-processing section, including
        tabbed views for different analysis results.
        - `QVBoxLayout`: The vertical box layout containing all post-processing tabs.
        """
        post_process_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        frame_tab = QWidget()
        frame_tab_layout = QVBoxLayout()
        self.frame_canvas = QWebEngineView()
        self.frame_canvas.setPage(SilentWebEnginePage(self.frame_canvas))
        html_content = ResultsViewer.generate_html_with_plotly()
        self.frame_canvas.setHtml(html_content)
        self.frame_canvas.plotly_figure = ResultsViewer._create_plotly_figure()
        frame_tab_layout.addWidget(self.frame_canvas)
        frame_tab.setLayout(frame_tab_layout)
        self.stiffness_tab = QWidget()
        self.stiffness_tab_layout = QVBoxLayout()
        self.stiffness_table = QTableWidget()
        self.stiffness_tab_layout.addWidget(self.stiffness_table)
        self.stiffness_tab.setLayout(self.stiffness_tab_layout)
        displacement_tab = QWidget()
        displacement_tab_layout = QVBoxLayout(displacement_tab)
        displacement_tab_layout.setContentsMargins(20, 10, 20, 0)
        displacement_tab_layout.setSpacing(0)
        self.deformation_canvas = QWebEngineView()
        self.deformation_canvas.setPage(SilentWebEnginePage(self.deformation_canvas))
        html_content = ResultsViewer.generate_html_with_plotly()
        self.deformation_canvas.setHtml(html_content)
        self.deformation_canvas.plotly_figure = ResultsViewer._create_plotly_figure()
        canvas_container = QWidget()
        canvas_container_layout = QVBoxLayout(canvas_container)
        canvas_container_layout.setContentsMargins(0, 0, 0, 0)
        canvas_container_layout.setSpacing(0)
        canvas_container_layout.addWidget(self.deformation_canvas)
        visualization_controls_layout = self.create_visualization_controls()
        controls_container = QWidget()
        controls_container.setLayout(visualization_controls_layout)
        controls_container.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        controls_container.setFixedHeight(100)
        overlay_layout = QVBoxLayout()
        overlay_layout.setContentsMargins(0, 0, 0, 20)
        overlay_layout.setSpacing(0)
        overlay_layout.addStretch()
        overlay_layout.addWidget(controls_container)
        canvas_container_layout.addLayout(overlay_layout)
        displacement_tab_layout.addWidget(canvas_container)
        displacement_tab.setLayout(displacement_tab_layout)
        force_tab = QWidget()
        force_tab_layout = QVBoxLayout()
        self.force_canvas = QWebEngineView()
        html_content = ResultsViewer.generate_html_with_plotly()
        self.force_canvas.setHtml(html_content)
        self.force_canvas.plotly_figure = ResultsViewer._create_plotly_figure()
        force_tab_layout.addWidget(self.force_canvas)
        force_tab.setLayout(force_tab_layout)
        stress_tab = QWidget()
        stress_tab_layout = QVBoxLayout()
        self.stress_canvas = QWebEngineView()
        html_content = ResultsViewer.generate_html_with_plotly()
        self.stress_canvas.setHtml(html_content)
        self.stress_canvas.plotly_figure = ResultsViewer._create_plotly_figure()
        stress_tab_layout.addWidget(self.stress_canvas)
        stress_tab.setLayout(stress_tab_layout)
        self.info_tab = QWidget()
        self.info_tab_layout = QVBoxLayout()
        self.info_canvas = QWebEngineView()
        self.info_canvas.setHtml("<html><body><p>Solver report...</p></body></html>")
        self.info_tab_layout.addWidget(self.info_canvas)
        self.info_tab.setLayout(self.info_tab_layout)
        self.file_tab = QWidget()
        self.file_tab_layout = QVBoxLayout()
        self.file_canvas = QWebEngineView()
        self.file_canvas.setHtml("<html><body><p>Input file...</p></body></html>")
        self.file_tab_layout.addWidget(self.file_canvas)
        self.file_tab.setLayout(self.file_tab_layout)
        self.tab_widget.addTab(self.file_tab, "File")
        self.tab_widget.addTab(frame_tab, "Geometry")
        self.tab_widget.addTab(self.stiffness_tab, "Stiffness-Matrix")
        self.tab_widget.addTab(displacement_tab, "Displacements")
        self.tab_widget.addTab(stress_tab, "Stresses")
        self.force_tab_index = self.tab_widget.addTab(force_tab, "Forces")
        self.tab_widget.addTab(self.info_tab, "Information")
        self.current_canvas = self.frame_canvas
        self.tab_widget.currentChanged.connect(ResultsViewer._on_tab_changed)
        self.tab_widget.setCurrentIndex(1)
        self.current_canvas = self.frame_canvas
        post_process_layout.addWidget(self.tab_widget)

        if self.imported_data['structure_info']['element_type'] in ['2D_Truss', '3D_Truss']:
            self.tab_widget.setTabVisible(self.force_tab_index, False)
        return post_process_layout

    def create_visualization_controls(self) -> QHBoxLayout:
        """
        Creates the layout for animation and visualization controls,
        including a slider for animation frames and play/pause button.
        - `QHBoxLayout`: The horizontal box layout containing the visualization controls.
        """
        self.animation_direction = 1
        self.current_frame_index = 0
        ResultsViewer.precompute_animation_frames()
        total_frames = len(ResultsViewer.animation_frames)
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(int(total_frames))
        self.scale_slider.setValue(1)


        def toggle_animation() -> None:
            """Toggles the animation between play and pause states."""

            if self.animation_timer.isActive():
                self.animation_timer.stop()
                self.play_pause_button.setText("Play")
            else:
                total_frames = len(ResultsViewer.animation_frames)
                self.scale_slider.setMaximum(int(total_frames))
                slider_value = self.scale_slider.value()
                self.current_frame_index = slider_value - 1
                self.animation_direction = 1
                self.animation_timer.start(100)
                self.play_pause_button.setText("Pause")


        def update_animation() -> None:
            """Updates the animation frame based on the current direction and frame index."""

            if not hasattr(ResultsViewer, "animation_frames"):
                return

            total_frames = len(ResultsViewer.animation_frames)
            self.current_frame_index += self.animation_direction

            if self.current_frame_index >= total_frames:
                self.current_frame_index = total_frames - 2
                self.animation_direction = -1
            elif self.current_frame_index < 0:
                self.current_frame_index = 1
                self.animation_direction = 1
            scale_value, nodes = ResultsViewer.animation_frames[self.current_frame_index]
            ResultsViewer.plot_deformation_with_nodes(nodes, scale_value)
            self.scale_slider.blockSignals(True)
            self.scale_slider.setValue(self.current_frame_index + 1)
            self.scale_slider.blockSignals(False)


        def scale_changed() -> None:
            """Handles changes in the animation scale slider, updating the displayed deformation."""
            index = self.scale_slider.value() - 1

            if not hasattr(ResultsViewer, "animation_frames"):
                ResultsViewer.precompute_animation_frames()

            if 0 <= index < len(ResultsViewer.animation_frames):
                scale_value, nodes = ResultsViewer.animation_frames[index]
                ResultsViewer.plot_deformation_with_nodes(nodes, scale_value)
        self.scale_slider.valueChanged.connect(scale_changed)
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(update_animation)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.setStyleSheet("background-color: #E5ECF6; color: #989898;")
        self.play_pause_button.clicked.connect(toggle_animation)
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.scale_slider)
        controls_layout.addWidget(self.play_pause_button)
        return controls_layout


    # ---------------------------------------------
    # EVENT HANDLERS
    # ---------------------------------------------

    def resizeEvent(self, event:QResizeEvent) -> None:
        """
        Handles the resize event for the main window, ensuring that the
        current plot canvas is resized accordingly.
        - `event` (`QResizeEvent`): The resize event object.
        """
        super().resizeEvent(event)

        if hasattr(self, 'current_canvas') and self.current_canvas:
            self.current_canvas.page().runJavaScript("window.dispatchEvent(new Event('resize'));")


    def toggle_play_pause(self, stop: bool = False) -> None:
        """
        Toggles the play/pause state of the animation or stops it.
        - `stop` (bool): If `True`, stops the animation. Otherwise, it toggles
          the play/pause state using the play/pause button's click event.
          Defaults to `False`.
        """

        if stop:
            self.animation_timer.stop()
        else:
            self.play_pause_button.click()