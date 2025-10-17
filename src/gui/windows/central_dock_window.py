from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTableWidget, QMainWindow, QDockWidget, QTabWidget, 
                             QSlider, QGridLayout, QSizePolicy)
from PyQt6.QtGui import QResizeEvent, QIcon, QPixmap, QPainter
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtSvg import QSvgRenderer
import plotly.io as pio
from src.gui.viewers.results import ResultsViewer
from src.utils.classes import SilentWebEnginePage
from src.constants import theme
import os


class CentralDockWindow(QWidget):
    """
    `CentralDockWindow` is a QMainWindow subclass designed to display and manage
    the results of structural analysis, including plots and detailed information.
    """
    current_canvas: QWebEngineView | None = None

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self, imported_data: dict, parent: QWidget = None) -> None:
        """
        Initializes the `CentralDockWindow`.
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
    # ICON MANAGEMENT
    # ---------------------------------------------

    def create_colored_icon(self, svg_path: str, color: str, size: QSize = QSize(16, 16)) -> QIcon:
        """
        Creates a QIcon from SVG with specified color.
        
        Args:
            svg_path: Path to the SVG file
            color: Color in hex format (#RRGGBB) or named color
            size: Size of the icon
            
        Returns:
            QIcon: The colored icon
        """
        # Read SVG content
        try:
            with open(svg_path, 'r', encoding='utf-8') as file:
                svg_content = file.read()
        except FileNotFoundError:
            print(f"Warning: SVG file not found at {svg_path}")
            pixmap = QPixmap(size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(0, 0, size.width(), size.height(), Qt.GlobalColor.gray)
            painter.end()
            return QIcon(pixmap)
        
        colored_svg = svg_content.replace('currentColor', color)
        colored_svg = colored_svg.replace('fill="currentColor"', f'fill="{color}"')
        colored_svg = colored_svg.replace('stroke="currentColor"', f'stroke="{color}"')
        
        renderer = QSvgRenderer()
        renderer.load(colored_svg.encode('utf-8'))
        
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        
        return QIcon(pixmap)

    def get_icon_color(self) -> str:
        """
        Returns the appropriate icon color based on theme.
        
        Returns:
            str: Color in hex format
        """
        return "#ffffff" if theme == "dark" else "#333333"

    def get_icon_path(self, icon_name: str) -> str:
        """
        Returns the full path to an icon file.
        
        Args:
            icon_name: Name of the icon file
            
        Returns:
            str: Full path to the icon file
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icons_dir = os.path.join(base_dir, "..", "..", "..", "assets", "icons", "analysis_window")
        return os.path.join(icons_dir, f"{icon_name}.svg")

    def setup_button_with_icon(self, button: QPushButton, icon_name: str, tooltip: str = "") -> None:
        """
        Sets up a button with an SVG icon.
        
        Args:
            button: The button to setup
            icon_name: Name of the icon file (without .svg extension)
            tooltip: Tooltip text for the button
        """
        icon_path = self.get_icon_path(icon_name)
        icon_color = self.get_icon_color()
        icon = self.create_colored_icon(icon_path, icon_color, QSize(20, 20))
        
        button.setIcon(icon)
        button.setIconSize(QSize(16, 16))
        
        if tooltip:
            button.setToolTip(tooltip)
        
        button.setText("")

    # ---------------------------------------------
    # LAYOUT CREATION
    # ---------------------------------------------

    def create_post_process_layout(self) -> QVBoxLayout:
        """
        Creates the main layout for the post-processing section.
        """
        bg_color = (
            Qt.GlobalColor.white if theme == "light" else Qt.GlobalColor.black
        )
        pio.templates.default = "plotly_dark" if theme == "dark" else "plotly_white"
        post_process_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        
        # Frame Tab (Geometry)
        frame_tab = QWidget()
        frame_tab_layout = QVBoxLayout()
        frame_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_canvas = QWebEngineView()
        self.frame_canvas.setPage(SilentWebEnginePage(self.frame_canvas, bg_color))
        html_content = ResultsViewer.generate_html_with_plotly()
        self.frame_canvas.setHtml(html_content)
        self.frame_canvas.plotly_figure = ResultsViewer._create_plotly_figure()
        frame_tab_layout.addWidget(self.frame_canvas)
        frame_tab.setLayout(frame_tab_layout)
        
        # Stiffness Matrix Tab
        self.stiffness_tab = QWidget()
        self.stiffness_tab_layout = QVBoxLayout()
        self.stiffness_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.stiffness_table = QTableWidget()
        self.stiffness_tab_layout.addWidget(self.stiffness_table)
        self.stiffness_tab.setLayout(self.stiffness_tab_layout)
        
        # Displacement Tab
        displacement_tab = QWidget()
        displacement_tab_layout = QGridLayout(displacement_tab)
        displacement_tab_layout.setContentsMargins(0, 0, 0, 0)
        displacement_tab_layout.setSpacing(0)
        
        self.deformation_canvas = QWebEngineView()
        self.deformation_canvas.setPage(SilentWebEnginePage(self.deformation_canvas, bg_color))
        self.deformation_canvas.setHtml("")
        self.deformation_canvas.plotly_figure = ResultsViewer._create_plotly_figure()
        
        displacement_tab_layout.addWidget(self.deformation_canvas, 0, 0)      
        floating_controls = self.create_floating_controls_panel()     
        displacement_tab_layout.addWidget(floating_controls, 0, 0, 
                                         Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        
        # Force Tab
        force_tab = QWidget()
        force_tab_layout = QVBoxLayout()
        force_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.force_canvas = QWebEngineView()
        self.force_canvas.setPage(SilentWebEnginePage(self.force_canvas, bg_color))
        self.force_canvas.setHtml("")
        self.force_canvas.plotly_figure = ResultsViewer._create_plotly_figure()
        force_tab_layout.addWidget(self.force_canvas)
        force_tab.setLayout(force_tab_layout)
        
        # Stress Tab
        stress_tab = QWidget()
        stress_tab_layout = QVBoxLayout()
        stress_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.stress_canvas = QWebEngineView()
        self.stress_canvas.setPage(SilentWebEnginePage(self.stress_canvas, bg_color))
        self.stress_canvas.setHtml("")
        self.stress_canvas.plotly_figure = ResultsViewer._create_plotly_figure()
        stress_tab_layout.addWidget(self.stress_canvas)
        stress_tab.setLayout(stress_tab_layout)
        
        # Information Tab
        self.info_tab = QWidget()
        self.info_tab_layout = QVBoxLayout()
        self.info_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.info_canvas = QWebEngineView()
        self.info_canvas.setHtml("<html><body><p>Solver report...</p></body></html>")
        self.info_tab_layout.addWidget(self.info_canvas)
        self.info_tab.setLayout(self.info_tab_layout)
        
        # File Tab
        self.file_tab = QWidget()
        self.file_tab_layout = QVBoxLayout()
        self.file_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.file_canvas = QWebEngineView()
        self.file_canvas.setPage(SilentWebEnginePage(self.file_canvas, bg_color))
        self.file_canvas.setHtml("<html><body><p>Input file...</p></body></html>")
        self.file_tab_layout.addWidget(self.file_canvas)
        self.file_tab.setLayout(self.file_tab_layout)
        
        # Add all tabs to tab widget
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
        Creates the layout for animation and visualization controls with SVG icons.
        """
        self.animation_direction = 1
        self.current_frame_index = 0
        ResultsViewer.precompute_animation_frames()
        total_frames = len(ResultsViewer.animation_frames)
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(int(total_frames))
        self.scale_slider.setValue(1)
        self.scale_slider.setFixedWidth(200)

        self.play_button = QPushButton()
        self.pause_button = QPushButton()
        self.stop_button = QPushButton()
        
        self.setup_button_with_icon(self.play_button, "play", "Play animation")
        self.setup_button_with_icon(self.pause_button, "pause", "Pause animation")
        self.setup_button_with_icon(self.stop_button, "stop", "Stop animation")
        
        if theme == "dark":
            button_style = """
                QPushButton {
                    background-color: rgba(70, 70, 70, 230);
                    border: 1px solid #555555;
                    border-radius: 6px;
                    min-width: 36px;
                    min-height: 36px;
                    max-width: 36px;
                    max-height: 36px;
                }
                QPushButton:hover {
                    background-color: rgba(90, 90, 90, 250);
                    border: 1px solid #666666;
                }
                QPushButton:pressed {
                    background-color: rgba(50, 50, 50, 230);
                    border: 1px solid #444444;
                }
                QPushButton:disabled {
                    background-color: rgba(40, 40, 40, 180);
                    border: 1px solid #333333;
                }
            """
        else:
            button_style = """
                QPushButton {
                    background-color: rgba(220, 220, 220, 230);
                    border: 1px solid #aaaaaa;
                    border-radius: 6px;
                    min-width: 36px;
                    min-height: 36px;
                    max-width: 36px;
                    max-height: 36px;
                }
                QPushButton:hover {
                    background-color: rgba(200, 200, 200, 250);
                    border: 1px solid #999999;
                }
                QPushButton:pressed {
                    background-color: rgba(180, 180, 180, 230);
                    border: 1px solid #888888;
                }
                QPushButton:disabled {
                    background-color: rgba(240, 240, 240, 180);
                    border: 1px solid #cccccc;
                }
            """
        
        for button in [self.play_button, self.pause_button, self.stop_button]:
            button.setFixedSize(36, 36)
            button.setStyleSheet(button_style)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setAutoDefault(False)
            button.setDefault(False)

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)

        self.play_button.clicked.connect(self.play_animation)
        self.pause_button.clicked.connect(self.pause_animation)
        self.stop_button.clicked.connect(self.stop_animation)
        self.scale_slider.valueChanged.connect(self.scale_changed)

        self.pause_button.setEnabled(False)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addSpacing(15)
        controls_layout.addWidget(self.scale_slider)
        controls_layout.addStretch()
        controls_layout.setContentsMargins(15, 8, 15, 8)
        controls_layout.setSpacing(8)

        return controls_layout

    def create_floating_controls_panel(self) -> QWidget:
        """
        Creates a truly floating panel for visualization controls that overlays the canvas.
        """
        floating_panel = QWidget()
        
        if theme == "dark":
            panel_style = """
                QWidget {
                    background-color: rgba(40, 40, 40, 230);
                    border: 1px solid #555555;
                    border-radius: 12px;
                    padding: 0px;
                }
            """
        else:
            panel_style = """
                QWidget {
                    background-color: rgba(240, 240, 240, 230);
                    border: 1px solid #cccccc;
                    border-radius: 12px;
                    padding: 0px;
                }
            """
        
        floating_panel.setStyleSheet(panel_style)
        
        floating_panel.setFixedHeight(50)
        floating_panel.setMinimumWidth(350)
        floating_panel.setMaximumWidth(450)
        floating_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        controls_layout = self.create_visualization_controls()
        floating_panel.setLayout(controls_layout)
        
        return floating_panel

    # ---------------------------------------------
    # ANIMATION CONTROL METHODS
    # ---------------------------------------------

    def play_animation(self) -> None:
        """Starts the animation."""
        if not hasattr(ResultsViewer, "animation_frames"):
            ResultsViewer.precompute_animation_frames()
            
        total_frames = len(ResultsViewer.animation_frames)
        self.scale_slider.setMaximum(int(total_frames))
        slider_value = self.scale_slider.value()
        self.current_frame_index = slider_value - 1
        self.animation_direction = 1
        self.animation_timer.start(100)
        
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)

    def pause_animation(self) -> None:
        """Pauses the animation."""
        self.animation_timer.stop()
        
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def stop_animation(self) -> None:
        """Stops the animation and resets to first frame."""
        self.animation_timer.stop()
        self.scale_slider.setValue(1)
        self.current_frame_index = 0
        
        if hasattr(ResultsViewer, "animation_frames") and ResultsViewer.animation_frames:
            scale_value, nodes = ResultsViewer.animation_frames[0]
            ResultsViewer.plot_deformation_with_nodes(nodes, scale_value)
        
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def update_animation(self) -> None:
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

    def scale_changed(self) -> None:
        """Handles changes in the animation scale slider."""
        index = self.scale_slider.value() - 1

        if not hasattr(ResultsViewer, "animation_frames"):
            ResultsViewer.precompute_animation_frames()

        if 0 <= index < len(ResultsViewer.animation_frames):
            scale_value, nodes = ResultsViewer.animation_frames[index]
            ResultsViewer.plot_deformation_with_nodes(nodes, scale_value)

    # ---------------------------------------------
    # EVENT HANDLERS
    # ---------------------------------------------

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Handles the resize event for the main window.
        """
        super().resizeEvent(event)

        if hasattr(self, 'current_canvas') and self.current_canvas:
            self.current_canvas.page().runJavaScript("window.dispatchEvent(new Event('resize'));")

    def closeEvent(self, event) -> None:
        """
        Handles the close event to ensure proper cleanup of resources.
        """
        if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
            self.animation_timer.stop()
        super().closeEvent(event)