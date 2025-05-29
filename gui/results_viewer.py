from PyQt6.QtWidgets import QTableWidgetItem, QHeaderView
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import json
from plotly.io import to_json
from typing import Dict, Any, Tuple, Optional, List, Union, Callable
from collections import defaultdict


class ResultsViewer:
    """
    A class responsible for visualizing structural analysis results using Plotly.
    This class handles the display of geometry, deformations, forces, and stresses

    for truss and frame structures. It integrates with a PyQt6-based main window
    to manage different display canvases and provides interactive features like
    hover annotations and dynamic plot resizing.
    - imported_data (Optional[Dict[str, Any]]): Stores the raw imported structural data,
        including nodes, elements, loads, and structure information.
    - MainWindow (Optional[Any]): Reference to the main application window.
    - ProcessingWindow (Optional[Any]): Reference to a processing window.
    - ProcessingWindowFunctions (Optional[Any]): Functions related to the processing window.
    - UnitsHandling (Optional[Any]): Utility for handling unit conversions.
    - CentralDockWindow (Optional[Any]): Reference to the central dock window containing canvases.
    - current_canvas (Optional[Any]): The currently active Plotly canvas (e.g., frame_canvas, deformation_canvas).
    - Defaults (Dict[str, Any]): A dictionary storing default settings, such as structure type (2D/3D).
    - deformation_fig_cache (Dict[float, go.Figure]): A cache for deformed figures,
        keyed by scale factor, to optimize rendering.
    - annotation (Optional[Any]): A Plotly annotation object, often used for hover text.
    - ax_main (Optional[Any]): Main matplotlib axes, if used (though Plotly is primary).
    - updatePlot (Optional[Callable[[Dict[str, Any]], None]]): A JavaScript function reference
        to update the Plotly plot in the QWebEngineView without reloading.
    - html_content (Optional[str]): The HTML content generated for the Plotly canvas.
    - animation_frames (Optional[List[Tuple[float, Dict[int, Tuple[float, float, Optional[float], Any, Any]]]]]):
        Precomputed frames for deformation animation, each containing a scale factor
        and the corresponding node positions.
    - FRAME_COUNT (int): The number of frames to generate for animations.
    """
    imported_data: Optional[Dict[str, Any]] = None
    MainWindow: Optional[Any] = None
    ProcessingWindow: Optional[Any] = None
    ProcessingWindowFunctions: Optional[Any] = None
    UnitsHandling: Optional[Any] = None
    CentralDockWindow: Optional[Any] = None
    current_canvas: Optional[Any] = None
    Defaults: Dict[str, Any] = {}
    deformation_fig_cache: Dict[float, go.Figure] = {}
    annotation: Optional[Any] = None
    ax_main: Optional[Any] = None
    updatePlot: Optional[Callable[[Dict[str, Any]], None]] = None
    html_content: Optional[str] = None
    animation_frames: Optional[List[Tuple[float, Dict[int, Tuple[float, float, Optional[float], Any, Any]]]]] = None
    FRAME_COUNT: int = 10

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self, imported_data: Dict[str, Any], CentralDockWindow: Any) -> None:
        """
        Initializes the ResultsViewer.
        Args:
            imported_data (Dict[str, Any]): The structural data imported from the analysis.
            CentralDockWindow (Any): The central dock widget of the main application window.
        """
        super().__init__()
        ResultsViewer.CentralDockWindow = CentralDockWindow
        ResultsViewer.imported_data = imported_data.copy()
        ResultsViewer.initialize_defaults()
        ResultsViewer._create_plotly_figure()

    # ---------------------------------------------
    # INITIALIZATION & DEFAULTS
    # ---------------------------------------------

    @staticmethod
    def initialize_defaults() -> None:
        """
        Initializes default settings based on the imported structure data.
        Sets 'structure_type' (e.g., "2D", "3D") and 'is_3d' boolean in `ResultsViewer.Defaults`.
        """
        structure_type: str = ResultsViewer.imported_data['structure_info']['dimension']
        ResultsViewer.Defaults['structure_type'] = structure_type
        ResultsViewer.Defaults['is_3d'] = "3D" == structure_type
  
    # ---------------------------------------------
    # SOLUTION VISUALIZATION
    # ---------------------------------------------

    @staticmethod
    def plot_solution(solution: Dict[str, Any], solution_valid: bool = True) -> None:
        """Plots the structural solution, including deformation, forces, and stresses.
        This method visualizes the results of a structural analysis. If the solution is valid,
        it displays the global stiffness matrix with computed displacements, plots the deformed
        structure, and, for certain element types, visualizes element forces and stresses.
        Args:
            solution (Dict[str, Any]): A dictionary containing all solution data, including:
                - 'solution_data' (Dict[str, Any]): General data about the solution.
                - 'forces' (Dict[int, Dict[str, float]]): Nodal forces.
                - 'global_stiffness_matrix' (np.ndarray): The global stiffness matrix.
                - 'displacements' (Dict[int, Dict[str, float]]): Nodal displacements.
                - 'elements_forces' (Dict[str, Any]): Forces within individual elements.
            solution_valid (bool, optional): Indicates whether the solution is valid for plotting.
                                             Defaults to True.
        """
        solution_data: Dict[str, Any] = solution['solution_data']
        F: Dict[int, Dict[str, float]] = solution['forces']
        K_global: np.ndarray = solution['global_stiffness_matrix']
        D: Dict[int, Dict[str, float]] = solution['displacements']
        elements_forces: Dict[str, Any] = solution['elements_forces']   
        ResultsViewer.imported_data = solution_data.copy()

        if solution_valid:

            # ---------------------------------------------
            # POPULATE STIFFNESS MATRIX 
            # ---------------------------------------------

            ResultsViewer.display_stiffness_matrix_with_solution(F, K_global, D, solution=True)

            # ---------------------------------------------
            # PLOT DEFORMATION
            # ---------------------------------------------

            ResultsViewer.plot_deformation()

            if ResultsViewer.imported_data['structure_info']['element_type'] in ['3D_Frame', '2D_Beam']:

                # ---------------------------------------------
                # PLOT FORCES
                # ---------------------------------------------

                ResultsViewer.plot_forces(elements_forces)

            # ---------------------------------------------
            # PLOT STRESSES
            # ---------------------------------------------

            ResultsViewer.plot_stresses(elements_forces = elements_forces)
        else:
            ResultsViewer.display_stiffness_matrix_with_solution(F, K_global, D, solution=False)

    # ---------------------------------------------
    # PLOTTING FUNCTIONS
    # ---------------------------------------------

    @staticmethod
    def plot_truss(
        frame_canvas: Optional[Any] = None,
        Spring: bool = False,
        Truss: bool = False,
        Frame: bool = False,
        deformed: bool = False
    ) -> go.Figure:
        """
        Plots the initial truss or frame structure.
        This method clears existing plot data, precomputes animation frames,
        and then renders elements, element numbers, XYZ triad, and loads.
        It also sets up consistent axis limits for smooth transitions during
        deformation animations.
        Args:
            frame_canvas (Optional[Any]): The canvas object where the plot will be displayed.
                                          If None, it defaults to ResultsViewer.CentralDockWindow.frame_canvas.
            Spring (bool): Not currently used, but can be extended to show springs.
            Truss (bool): Not currently used, but can be extended to show trusses.
            Frame (bool): Not currently used, but can be extended to show frames.
            deformed (bool): If True, plots the deformed shape. Currently set to False.
        Returns:
            plotly.graph_objects.Figure: The Plotly figure object with the plotted truss/frame.
        """
        fig_canvas: Any = ResultsViewer.CentralDockWindow.frame_canvas
        fig: go.Figure = fig_canvas.plotly_figure
        fig.data = []
        ResultsViewer.precompute_animation_frames()
        scale_value: float
        nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]]
        scale_value, nodes = ResultsViewer.animation_frames[0]
        fig = ResultsViewer._plot_elements(fig, nodes, Frame=True, deformed=False)
        fig = ResultsViewer._add_elements_numbers(fig, nodes, Frame=True, deformed=False)
        fig = ResultsViewer.add_xyz_triad(fig, nodes)
        fig = ResultsViewer._plot_distributed_loads(fig, nodes)
        fig = ResultsViewer._plot_concentrated_loads(fig, nodes)
        min_x: float
        max_x: float
        min_y: float
        max_y: float
        min_z: Optional[float]
        max_z: Optional[float]
        min_x, max_x, min_y, max_y, min_z, max_z = ResultsViewer.animation_axis_limits
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[min_x, max_x]),
                yaxis=dict(range=[min_y, max_y]),
                zaxis=dict(range=[min_z, max_z]) if min_z is not None else {}
            )
        )
        fig = ResultsViewer._set_equal_axes(fig, nodes)
        ResultsViewer._update_plotly_canvas(fig_canvas, fig)
        ResultsViewer.deformation_fig_cache[-1] = fig
        return fig


    @staticmethod
    def clear_fig_data(fig: go.Figure) -> go.Figure:
        """
        Clears all traces (data) and non-camera/layout-essential settings
        from a Plotly figure, preserving the camera view and main layout settings.
        Args:
            fig (go.Figure): The Plotly figure to clear.
        Returns:
            go.Figure: The cleared Plotly figure.
        """
        fig.data = []
        fig.layout.annotations = []
        fig.layout.shapes = []
        fig.layout.images = []
        fig.layout.sliders = []
        fig.layout.updatemenus = []
        fig.layout.ternary = {}
        fig.layout.polar = {}
        fig.layout.mapbox = {}
        fig.layout.scene = {}
        return fig


    @staticmethod
    def plot_deformation(scale_multiplier: float = 1.0) -> None:
        """
        Plots the deformed shape of the structure.
        This method precomputes animation frames and then calls
        `plot_deformation_with_nodes` to render the deformed shape.
        The resulting figure is then set to the deformation canvas.
        Args:
            scale_multiplier (float): A factor to scale the displacements for visualization.
                                      Defaults to 1.0 (actual displacements).
        """
        ResultsViewer.precompute_animation_frames()
        scale_value: float
        nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]]
        scale_value, nodes = ResultsViewer.animation_frames[0]
        fig: go.Figure = ResultsViewer.plot_deformation_with_nodes(nodes, scale_value)
        fig_canvas: Any = ResultsViewer.CentralDockWindow.deformation_canvas
        fig_canvas.plotly_figure = fig

    @staticmethod
    def plot_deformation_with_nodes(
        nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]],
        scale_value: float
    ) -> go.Figure:
        """
        Plots the deformed shape of the structure using provided node positions.
        This helper method is used by `plot_deformation` and animation logic.
        It clears the figure, plots elements with deformation, adds element numbers,
        XYZ triad, distributed loads, and concentrated loads. It also sets up
        consistent axis limits.
        Args:
            nodes (Dict[int, Tuple[float, float, Optional[float], Any, Any]]):
                A dictionary where keys are node numbers and values are tuples
                (x, y, z, force_data, displacement_data) representing the deformed node positions.
            scale_value (float): The scale factor applied to the displacements for this plot.
        Returns:
            go.Figure: The Plotly figure object with the deformed structure.
        """
        fig_canvas: Any = ResultsViewer.CentralDockWindow.deformation_canvas
        fig: go.Figure = fig_canvas.plotly_figure
        frame_fig: go.Figure = ResultsViewer.deformation_fig_cache[-1] 
        fig.data = []
        fig = ResultsViewer._plot_elements(fig, nodes, Frame=True, deformed=True)
        fig = ResultsViewer._add_elements_numbers(fig, nodes, Frame=True, deformed=True)
        fig = ResultsViewer.add_xyz_triad(fig, nodes)
        fig = ResultsViewer._plot_distributed_loads(fig, nodes)
        fig = ResultsViewer._plot_concentrated_loads(fig, nodes)
        min_x: float
        max_x: float
        min_y: float
        max_y: float
        min_z: Optional[float]
        max_z: Optional[float]
        min_x, max_x, min_y, max_y, min_z, max_z = ResultsViewer.animation_axis_limits
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[min_x, max_x]),
                yaxis=dict(range=[min_y, max_y]),
                zaxis=dict(range=[min_z, max_z]) if min_z is not None else {}
            )
        )
        fig = ResultsViewer._setup_plotly_figure(fig)
        fig = ResultsViewer._set_equal_axes(fig, nodes)
        ResultsViewer._update_plotly_canvas(fig_canvas, fig)
        return fig


    @staticmethod
    def plot_forces(elements_forces: Dict[str, Any] = {}) -> None:
        """
        Plots the structure along with internal forces (e.g., shear, moment)
        or axial forces for truss elements.
        This method leverages `plot_force_shear_moment` to display internal
        force diagrams.
        Args:
            elements_forces (Dict[str, Any]): A dictionary containing force data for elements.
        """
        fig_canvas: Any = ResultsViewer.CentralDockWindow.force_canvas
        fig: go.Figure = fig_canvas.plotly_figure
        fig = ResultsViewer.plot_force_shear_moment(elements_forces)
        ResultsViewer._update_plotly_canvas(fig_canvas, fig)

    @staticmethod
    def plot_stresses(elements_forces: Dict[str, Any] = {}) -> None:
        """
        Plots the structure along with stress visualization.
        This method plots elements with color mapping based on stress values.
        It checks for 3D_Solid and 3D_Heat elements and returns early if found,
        as stress plotting for these types might be handled differently.
        Args:
            elements_forces (Dict[str, Any]): A dictionary containing stress data for elements.
        """
        structure_type: str = ResultsViewer.imported_data['structure_info']['element_type']

        if structure_type in {"3D_Solid", "3D_Heat"}:
            return

        Frame: bool = True
        deformed: bool = False
        fig_canvas: Any = ResultsViewer.CentralDockWindow.stress_canvas
        fig: go.Figure = ResultsViewer._create_plotly_figure()
        nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]] = ResultsViewer._get_node_positions(deformed)
        fig = ResultsViewer._plot_elements(fig, nodes, Frame, deformed, radius=0.2, elements_forces=elements_forces)
        fig = ResultsViewer._add_elements_numbers(fig, nodes, Frame, deformed, radius=0.2) 
        fig = ResultsViewer.add_xyz_triad(fig, nodes)        
        fig = ResultsViewer._plot_distributed_loads(fig, nodes, arrows_per_length=5)       
        fig = ResultsViewer._plot_concentrated_loads(fig, nodes)      
        fig = ResultsViewer._set_equal_axes(fig, nodes)     
        ResultsViewer.deformation_fig_cache[-1] = fig
        ResultsViewer._update_plotly_canvas(fig_canvas, fig)
        fig_canvas.plotly_figure = fig

    @staticmethod
    def on_hover(trace: go.Trace, points: Any, hover_annotation: Any) -> None:
        """
        Displays node or element data on hover.
        This method is intended to be a callback for Plotly hover events.
        It identifies whether the hovered point corresponds to a node or an element
        and updates the provided `hover_annotation` with relevant information
        (e.g., node number, force, displacement; or element number, E, A).
        Args:
            trace (go.Trace): The Plotly trace object that was hovered over.
            points (Any): An object containing information about the hovered points.
                          `points.point_inds` is a list of indices of the hovered points.
            hover_annotation (Any): A Plotly annotation object to be updated and displayed.
        """

        if points.point_inds:
            index: int = points.point_inds[0]
            x: float
            y: float
            x, y = trace.x[index], trace.y[index]

            for node_num, data in ResultsViewer.imported_data['nodes'].items():

                if np.isclose([x], data['X'])[0] and np.isclose([y], data['Y'])[0]:
                    hover_annotation.text = f"Node {node_num}\nForce: {data['force']}\nDisp: {data['displacement']}"
                    hover_annotation.x, hover_annotation.y = x, y
                    hover_annotation.visible = True
                    return

            for elem_num, elem_data in ResultsViewer.imported_data['elements'].items():
                node1: int
                node2: int
                node1, node2 = elem_data['node1'], elem_data['node2']
                x1: float
                y1: float
                x2: float
                y2: float
                x1, y1 = ResultsViewer.imported_data['nodes'][node1]['X'], ResultsViewer.imported_data['nodes'][node1]['Y']
                x2, y2 = ResultsViewer.imported_data['nodes'][node2]['X'], ResultsViewer.imported_data['nodes'][node2]['Y']
                midpoint_x: float
                midpoint_y: float
                midpoint_x, midpoint_y = (x1 + x2) / 2, (y1 + y2) / 2

                if np.isclose([x], [midpoint_x])[0] and np.isclose([y], [midpoint_y])[0]:
                    hover_annotation.text = f"Element {elem_num}\nE: {elem_data['E']}\nA: {elem_data['A']}"
                    hover_annotation.x, hover_annotation.y = midpoint_x, midpoint_y
                    hover_annotation.visible = True
                    return

        hover_annotation.visible = False

    @staticmethod
    def set_axis_transparency(fig: go.Figure, alpha: float = 0.5) -> go.Figure:
        """
        Sets transparency for axis elements and plot background in a Plotly figure.
        Args:
            fig (go.Figure): The Plotly figure to modify.
            alpha (float): The transparency level (0.0 for fully transparent, 1.0 for opaque).
        Returns:
            go.Figure: The modified Plotly figure.
        """
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, title_font=dict(color=f'rgba(0,0,0,{alpha})')),
            yaxis=dict(showgrid=False, zeroline=False, title_font=dict(color=f'rgba(0,0,0,{alpha})')),
            plot_bgcolor=f'rgba(255,255,255,{alpha})'
        )
        return fig

    # ---------------------------------------------
    # PLOTLY FIGURE UTILITIES
    # ---------------------------------------------

    @staticmethod
    def _create_plotly_figure(fig: Optional[go.Figure] = None) -> go.Figure:
        """
        Creates a new empty Plotly figure or initializes an existing one.
        This method ensures the figure is configured for dynamic resizing and
        applies default layout settings by calling `_setup_plotly_figure`.
        Args:
            fig (Optional[go.Figure]): An existing Plotly figure to modify. If None, a new figure is created.
        Returns:
            go.Figure: The initialized or modified Plotly figure.
        """

        if fig is None:
            fig = go.Figure()
        fig = ResultsViewer._setup_plotly_figure(fig)
        return fig


    @staticmethod
    def _setup_plotly_figure(fig: go.Figure) -> go.Figure:
        """
        Configures a Plotly figure with default layout settings for structural plots.
        This includes enabling autosizing, hiding axes, setting a transparent background,
        and configuring camera settings for 2D or 3D views based on the structure type.
        Args:
            fig (go.Figure): The Plotly figure to configure.
        Returns:
            go.Figure: The configured Plotly figure.
        """
        fig.update_layout(
            autosize=True,
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, showticklabels=False, zeroline=False),
                zaxis=dict(visible=False, showgrid=False, showticklabels=False, zeroline=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        is_3d: bool = ResultsViewer.Defaults['is_3d']
        y_limits: Optional[Tuple[float, float]] = (-0.3, 0.3) if not is_3d else None  

        if is_3d:  
            fig.update_layout(
                scene=dict(
                    aspectmode="auto",
                    camera=dict(
                        eye=dict(x=1.25, y=1.25, z=1.25),
                        up=dict(x=0, y=1, z=0)
                    ),
                )
            )        
        else:  
            fig.update_layout(
                scene=dict(
                    aspectmode="auto",
                    camera=dict(
                        eye=dict(x=0, y=0, z=2),
                        up=dict(x=0, y=1, z=0)
                    ),
                    yaxis=dict(range=y_limits),
                    zaxis=dict(range=y_limits),
                )
            )
        return fig

    # ---------------------------------------------
    # HTML & CANVAS MANAGEMENT
    # ---------------------------------------------

    @staticmethod
    def generate_html_with_plotly(fig: Optional[go.Figure] = None, plot_id: str = "plotly-3Dplot", server_url: str = "http://localhost:8000") -> str:
        """
        Generates an HTML string containing a Plotly plot that dynamically resizes
        and loads the Plotly JavaScript library from a local HTTP server.

        This HTML includes JavaScript functions for updating the plot while
        preserving camera views, zooming to fit, and handling window resizing.

        Args:
            fig (Optional[go.Figure]): The Plotly figure to embed in the HTML. If None, an empty figure is used.
            plot_id (str): The HTML `id` attribute for the plot container div.
            server_url (str): The base URL from which the Plotly JavaScript library will be loaded.

        Returns:
            str: The complete HTML string for embedding the Plotly plot.
        """
        plotly_script: str = f'<script src="{server_url}/plotly-3.0.1.min.js"></script>'
        json_data: str = to_json(fig) if fig else '{"data": [], "layout": {}}'

        html_content: str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            {plotly_script}  <style>
                html, body, #container {{
                    width: 100%;
        height: 100%;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                }}
                #plot-container {{
                    width: 100%;
        height: 100%;
                }}
                canvas {{
                    will-read-frequently: true;
        }}                
            </style>
        </head>
        <body>
            <div id="container">
                <div id="{plot_id}" style="width: 100%; height: 100%;"></div>
            </div>
          
            <script>
                window.ResultsViewer = window.ResultsViewer ||
        {{}};
                var canvas = document.createElement('canvas');
                var ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
        document.addEventListener("DOMContentLoaded", function() {{
                    var fig = {json_data};  // Load initial figure
                    var graphDiv = document.getElementById('{plot_id}');
                    Plotly.newPlot(graphDiv, fig.data, fig.layout);
                    window.plotLoaded = true;

                    // Store a global reference to the plot
                    window.graph3d = graphDiv;

                    // Listen for camera updates when user rotates the scene
                    graphDiv.on('plotly_relayout', function(eventData) {{
   
                        if (eventData["scene.camera"]) {{
                            window.currentCamera = eventData["scene.camera"];  // Store latest camera position
                            console.log("Updated Camera Data:", window.currentCamera);
          
                        }}
                    }});

                    // Function to update the plot while preserving camera view
                    window.ResultsViewer.updatePlot = function (newData) {{
             
                        if (window.plotLoaded) {{
                            if (window.currentCamera) {{
                                newData.layout.scene.camera = window.currentCamera;
        // Preserve the latest camera
                            }}
                            Plotly.react(graphDiv, newData.data, newData.layout);
        }}
                    }};
        // Function to Zoom to Fit content
                    window.zoomToFit = function () {{
                        if (window.graph3d) {{
                            Plotly.relayout(window.graph3d, {{
               
                                'scene.xaxis.autorange': true,
                                'scene.yaxis.autorange': true,
                                'scene.zaxis.autorange': true
                
                            }});
                            console.log("Zoom to Fit Applied.");
        }}
                    }};
        // Function to dynamically resize the plot when the window is resized
                    window.resizePlot = function() {{
                        if (window.graph3d) {{
                            Plotly.relayout(window.graph3d, {{ autosize: true }}).then(() => {{
      
                                console.log("Plot resized successfully.");
                            }});
        }}
                    }};
        // Attach resize event listener
                    window.addEventListener("resize", window.resizePlot);
        }});
            </script>
        </body>
        </html>
        """
        ResultsViewer.html_content = html_content
        return html_content


    @staticmethod
    def _update_plotly_canvas(frame_canvas: Any, fig: go.Figure) -> None:
        """
        Updates the Plotly canvas embedded in a QWebEngineView.
        For the first update, it sets the full HTML content. Subsequent updates
        use JavaScript injection to update the Plotly figure data while attempting
        to preserve the current camera view.
        Args:
            frame_canvas (Any): The QWebEngineView (or similar) instance used as a canvas.
            fig (go.Figure): The Plotly figure object to display.
        """

        if not hasattr(frame_canvas, "initialized") or not frame_canvas.initialized:
            html_content: str = ResultsViewer.generate_html_with_plotly(fig)
            frame_canvas.setHtml(html_content)
            frame_canvas.initialized = True
        else:
            capture_camera_script: str = """
                (function() {
                    return window.currentCamera ? JSON.stringify(window.currentCamera) : null;

                })();
        """


            def apply_update(camera_data: str) -> None:
                """
                Inner function to apply plot updates after capturing camera data.
                """

                if camera_data and camera_data != "null":

                    try:
                        camera_json: Dict[str, Any] = json.loads(camera_data)
                        fig.update_layout(scene=dict(camera=camera_json))

                    except json.JSONDecodeError:
                        print("Invalid camera data format received:", camera_data)
                json_data_with_camera: str = to_json(fig)
                update_script: str = f"""
                setTimeout(function() {{

                    if (typeof window.ResultsViewer.updatePlot === 'function') {{
                        window.ResultsViewer.updatePlot({json_data_with_camera});
                    }} else {{
                        console.error('window.ResultsViewer.updatePlot is not yet defined.');
                    }}
                }}, 0);
        """
                frame_canvas.page().runJavaScript(update_script)
            frame_canvas.page().runJavaScript(capture_camera_script, apply_update)
            update_script: str = f"""
            setTimeout(function() {{

                if (typeof window.zoomToFit === 'function') {{
                    window.zoomToFit();
                }} else {{
                    console.error('window.zoomToFit is not yet 
        defined.');
                }}
            }}, 100);
        """                

    @staticmethod
    def _on_tab_changed(index: int) -> None:
        """
        Updates the `current_canvas` class attribute based on the selected tab in the UI.
        This method is typically connected to the `currentChanged` signal of a QTabWidget.
        Args:
            index (int): The index of the newly selected tab.
        """

        if ResultsViewer.CentralDockWindow:
            tab_widget: Any = ResultsViewer.CentralDockWindow.tab_widget
            tab_name: str = tab_widget.tabText(index)

            if tab_name == "File":
                ResultsViewer.current_canvas = ResultsViewer.CentralDockWindow.file_canvas

            if tab_name == "Geometry":
                ResultsViewer.current_canvas = ResultsViewer.CentralDockWindow.frame_canvas
            elif tab_name == "Stiffness-Matrix":
                ResultsViewer.current_canvas = ResultsViewer.CentralDockWindow.stiffness_table
            elif tab_name == "Displacements":
                ResultsViewer.current_canvas = ResultsViewer.CentralDockWindow.deformation_canvas
            elif tab_name == "Forces":
                ResultsViewer.current_canvas = ResultsViewer.CentralDockWindow.force_canvas
            elif tab_name == "Stresses":
                ResultsViewer.current_canvas = ResultsViewer.CentralDockWindow.stress_canvas
            elif tab_name == "Information":
                ResultsViewer.current_canvas = ResultsViewer.CentralDockWindow.info_canvas
            ResultsViewer.CentralDockWindow.current_canvas =  ResultsViewer.current_canvas    
 
    # ---------------------------------------------
    # NODE & DEFORMATION UTILITIES
    # ---------------------------------------------

    @staticmethod
    def _get_node_positions(
        scale_multiplier: float = 1.0,
        deformed: bool = False,
        displacements_override: Optional[Dict[int, List[float]]] = None
    ) -> Dict[int, Tuple[float, float, Optional[float], Any, Any]]:
        """
        Calculates the positions of nodes, optionally including deformation.
        Args:
            scale_multiplier (float): Factor to scale displacements. Defaults to 1.0.
            deformed (bool): If True, apply displacements to original node positions.
            displacements_override (Optional[Dict[int, List[float]]]):
                A dictionary of displacement lists keyed by node number. If provided,
                these displacements are used instead of `data['displacement']`.
        Returns:
            Dict[int, Tuple[float, float, Optional[float], Any, Any]]:
                A dictionary where keys are node numbers and values are tuples
                (x, y, z, force_data, displacement_data) of node positions
                (z is Optional[float] for 2D structures).
        """
        is_3D: bool = ResultsViewer.Defaults['is_3d']
        nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]] = {}

        for node_num, data in ResultsViewer.imported_data['nodes'].items():
            disp: List[float] = displacements_override[node_num] if displacements_override else data['displacement']
            dx: float = disp[0] if not np.isnan(disp[0]) else 0
            dy: float = disp[1] if not np.isnan(disp[1]) else 0
            dz: Optional[float] = disp[2] if is_3D and not np.isnan(disp[2]) else (0 if is_3D else None)
            x: float = data['X'] + (scale_multiplier * dx if deformed else 0)
            y: float = data['Y'] + (scale_multiplier * dy if deformed else 0)
            z: Optional[float] = data['Z'] + (scale_multiplier * dz if deformed and is_3D else 0) if is_3D else None
            nodes[node_num] = (x, y, z, data['force'], disp)
        return nodes


    @staticmethod
    def compute_auto_scale_factor() -> float:
        """
        Compute a deformation scale factor so deformations are visible but not exaggerated.
        This method calculates a suitable scale factor based on the maximum displacement
        and the overall span of the structure. It prevents excessively large or small
        deformation visualizations.
        Returns:
            float: The computed auto-scale factor for deformations.
        """
        all_coords: List[Dict[str, Any]] = [n for n in ResultsViewer.imported_data['nodes'].values()]
        all_displacements: List[List[float]] = [n['displacement'] for n in all_coords if n['displacement'] is not None]
        max_disp: float = max(
            (np.linalg.norm(d[:3]) for d in all_displacements if not np.any(np.isnan(d[:3]))),
            default=0.0 
        )

        if all_coords:
            all_coords_arr: np.ndarray = np.array([[n['X'], n['Y'], n['Z']] if ResultsViewer.Defaults['is_3d']
                                    else [n['X'], n['Y']] for n in all_coords])
            max_span: float = np.max(all_coords_arr) - np.min(all_coords_arr)
        else:
            max_disp = 0.0

        if max_disp == 0.0:
            return 1.0

        return min(0.5 * max_span / max_disp, 1e23)


    @staticmethod
    def precompute_animation_frames() -> None:
        """
        Precomputes a series of frames for deformation animation.
        This method calculates node positions for various scaled deformation states
        and stores them, along with global axis limits, in `ResultsViewer.animation_frames`
        and `ResultsViewer.animation_axis_limits`. This optimizes animation playback.
        """
        scale_factor: float = ResultsViewer.compute_auto_scale_factor()
        ResultsViewer.FRAME_COUNT = 10
        is_3d: bool = ResultsViewer.Defaults['is_3d']
        forward_frames: List[Tuple[float, Dict[int, Tuple[float, float, Optional[float], Any, Any]]]] = []
        min_x: float = float("inf")
        max_x: float = float("-inf")
        min_y: float = float("inf")
        max_y: float = float("-inf")

        if is_3d:
            min_z: Optional[float] = float("inf") 
            max_z: Optional[float] = float("-inf")
        else:
            min_z, max_z = None, None

        for i in range(1, ResultsViewer.FRAME_COUNT + 1):
            scale: float = (i / ResultsViewer.FRAME_COUNT) * scale_factor
            nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]] = ResultsViewer._get_node_positions(scale_multiplier=scale, deformed=True)
            forward_frames.append((scale, nodes))
            x_vals: List[float] = [node[0] for node in nodes.values()]
            y_vals: List[float] = [node[1] for node in nodes.values()]
            z_vals: List[float] = [node[2] for node in nodes.values() if node[2] is not None]

            if x_vals and y_vals:
                min_x, max_x = min(min_x, min(x_vals)), max(max_x, max(x_vals))
                min_y, max_y = min(min_y, min(y_vals)), max(max_y, max(y_vals))

            if z_vals:

                if min_z is None:
                    min_z, max_z = min(z_vals), max(z_vals)
                else:
                    min_z, max_z = min(min_z, min(z_vals)), max(max_z, max(z_vals))
        ResultsViewer.animation_frames = forward_frames
        ResultsViewer.animation_axis_limits = (min_x, max_x, min_y, max_y, min_z, max_z)
 
    # ---------------------------------------------
    # ELEMENT RENDERING & GEOMETRY
    # ---------------------------------------------

    @staticmethod
    def _add_elements_numbers(
        fig: go.Figure,
        nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]],
        Frame: bool,
        deformed: bool = False,
        radius: float = 0.1
    ) -> go.Figure:
        """
        Adds element numbers as annotations to the Plotly figure.
        The position of the number is calculated to be near the midpoint of the element.
        Text size is dynamically adjusted based on element length.
        Args:
            fig (go.Figure): The Plotly figure to add annotations to.
            nodes (Dict[int, Tuple[float, float, Optional[float], Any, Any]]):
                A dictionary of node positions (deformed or undeformed).
            Frame (bool): Indicates if it's a frame structure (influences behavior like force plotting).
            deformed (bool): If True, elements are assumed to be in their deformed state.
            radius (float): The approximate radius of the element for offsetting text.
        Returns:
            go.Figure: The Plotly figure with element number annotations added.
        """
        color: str = "blue" if deformed else "#e3dddb"

        if not nodes:
            return fig

        if not ResultsViewer.imported_data['elements']:
            return fig

        element_lengths: List[float] = []

        for elem in ResultsViewer.imported_data['elements'].values():
            ni: int
            nj: int
            ni, nj = elem["node1"], elem["node2"]

            if ni in nodes and nj in nodes:
                x1: float
                y1: float
                z1: Optional[float]
                x2: float
                y2: float
                z2: Optional[float]
                x1, y1, z1 = nodes[ni][:3]
                x2, y2, z2 = nodes[nj][:3]
                z1, z2 = (0 if z is None else z for z in (z1, z2))
                length: float = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

                if length > 0:
                    element_lengths.append(length)

        if not element_lengths:
            print("Error: No valid elements with length > 0 found!")
            return fig

        min_length: float = min(element_lengths)
        text_size: float = max(10, 12 * (min_length / np.mean(element_lengths)))

        for elem_num, elem in ResultsViewer.imported_data['elements'].items():
            ni: int
            nj: int
            ni, nj = elem["node1"], elem["node2"]

            if ni not in nodes or nj not in nodes:
                continue
            x1, y1, z1 = nodes[ni][:3]
            x2, y2, z2 = nodes[nj][:3]
            z1, z2 = (0 if z is None else z for z in (z1, z2))
            length: float = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

            if length == 0:
                continue
            vx: float
            vy: float
            vz: float
            vx, vy, vz = (x2 - x1) / length, (y2 - y1) / length, (z2 - z1) / length
            shift_factor: float = radius * 1.5
            mid_x: float
            mid_y: float
            mid_z: float
            mid_x, mid_y, mid_z = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
            text_x: float = mid_x
            text_y: float = mid_y - shift_factor * vy
            text_z: float = mid_z - shift_factor * vz
            fig.add_trace(go.Scatter3d(
                x=[text_x], y=[text_y], z=[text_z],
                mode="text",
                text=[str(elem_num)],
                textfont=dict(size=text_size, color="black"),
                showlegend=False
            ))
        return fig


    @staticmethod
    def _plot_elements(fig: go.Figure, nodes: Dict[int, Tuple[float, float, float, Any, List[float]]],
                       Frame: Any, deformed: bool = False, radius: float = 0.1,
                       elements_forces: Dict[int, Dict[str, Any]] = None,
                       shiny: bool = True, shadow_effect: bool = True,
                       edge_visibility: bool = False, edge_color: str = 'black',
                       edge_width: float = 5.5) -> go.Figure:
        """
        Plots 3D representations of structural elements (beams) in a Plotly figure.
        This function visualizes beam elements based on their cross-sectional properties
        and node coordinates. It supports various cross-section types, and can display
        elements in their deformed state or with stress/displacement intensity.
        Optional features include shiny rendering, shadow effects, and visible edges.
        Args:
            fig (go.Figure): The Plotly figure object to which the elements will be added.
            nodes (Dict[int, Tuple[float, float, float, Any, List[float]]]): A dictionary
                where keys are node IDs and values are tuples containing (x, y, z, _, displacement_list).
            Frame (Any): A placeholder argument, not directly used in the provided logic.
            deformed (bool, optional): If True, elements are plotted in their deformed
                configuration (requires displacement data in `nodes`). Defaults to False.
            radius (float, optional): Default radius for circular elements if dimensions
                are not fully specified. Defaults to 0.1.
            elements_forces (Dict[int, Dict[str, Any]], optional): A dictionary containing
                element forces or stresses. If provided, element colors will be based on
                'von_mises' stress. Defaults to None.
            shiny (bool, optional): If True, applies a shinier lighting effect to the meshes.
                Defaults to True.
            shadow_effect (bool, optional): If True, adds a shadow-like effect based on
                Z-position to enhance 3D perception. Defaults to True.
            edge_visibility (bool, optional): If True, renders black lines along the outer
                edges of the elements. Defaults to False.
            edge_color (str, optional): The color of the edge lines if `edge_visibility` is True.
                Defaults to 'black'.
            edge_width (float, optional): The width of the edge lines if `edge_visibility` is True.
                Defaults to 5.5.
        Returns:
            go.Figure: The modified Plotly figure object with the added beam elements.
        """
        
        # ---------------------------------------------
        # HELPER FUNCTIONS - DIMENSION & GEOMETRY
        # ---------------------------------------------

        def create_cylinder(radius_dims: Dict[str, float], height: float = 1.0, segments: int = 16) -> Tuple[np.ndarray, np.ndarray]:
            """
            Creates vertices and faces for a solid cylinder.
            Args:
                radius_dims (Dict[str, float]): Dictionary containing 'D' for diameter.
                height (float, optional): Height of the cylinder. Defaults to 1.0.
                segments (int, optional): Number of segments for the circular base. Defaults to 16.
            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing:
                    - verts (np.ndarray): (N, 3) array of vertices.
                    - faces (np.ndarray): (M, 3) array of triangular face indices.
            """
            radius_val = radius_dims['D'] / 2
            theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
            x = radius_val * np.cos(theta)
            y = radius_val * np.sin(theta)
            z0 = np.zeros_like(x)
            z1 = np.full_like(x, height)
            verts = np.stack([np.concatenate([x, x]),
                            np.concatenate([y, y]),
                            np.concatenate([z0, z1])], axis=1)
            faces = []

            for i in range(segments):
                ni = (i + 1) % segments
                faces.append([i, ni, i + segments])
                faces.append([ni, ni + segments, i + segments])
            return verts, np.array(faces)

        def create_hollow_cylinder(dims: Dict[str, float], height: float = 1.0, segments: int = 16) -> Tuple[np.ndarray, np.ndarray]:
            """
            Creates vertices and faces for a hollow cylinder.
            Args:
                dims (Dict[str, float]): Dictionary containing 'D' for outer diameter and 'd' for inner diameter.
                height (float, optional): Height of the cylinder. Defaults to 1.0.
                segments (int, optional): Number of segments for the circular base. Defaults to 16.
            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing:
                    - verts (np.ndarray): (N, 3) array of vertices.
                    - faces (np.ndarray): (M, 3) array of triangular face indices.
            """
            outer_radius = dims['D'] / 2
            inner_radius = dims['d'] / 2
            theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
            x_out = outer_radius * np.cos(theta)
            y_out = outer_radius * np.sin(theta)
            x_in = inner_radius * np.cos(theta)
            y_in = inner_radius * np.sin(theta)
            z0 = np.zeros(segments)
            z1 = np.ones(segments) * height
            verts = np.stack([
                np.concatenate([x_out, x_out, x_in, x_in]),
                np.concatenate([y_out, y_out, y_in, y_in]),
                np.concatenate([z0, z1, z0, z1])
            ], axis=1)
            faces = []

            for i in range(segments):
                ni = (i + 1) % segments
                faces.append([i, ni, i + segments])
                faces.append([ni, ni + segments, i + segments])
                faces.append([i + 2 * segments, ni + 2 * segments, i + 3 * segments])
                faces.append([ni + 2 * segments, ni + 3 * segments, i + 3 * segments])
            return verts, np.array(faces)

        def create_rectangular_prism(dims: Dict[str, float], height: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
            """
            Creates vertices and faces for a solid rectangular prism.
            Args:
                dims (Dict[str, float]): Dictionary containing 'B' (width), 'H' (height), and 'angle' (rotation).
                height (float, optional): Length of the beam along the Z-axis. Defaults to 1.0.
            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing:
                    - verts (np.ndarray): (N, 3) array of vertices.
                    - faces (np.ndarray): (M, 3) array of triangular face indices.
            """
            B = dims['B'] / 2
            H = dims['H'] / 2
            angle_deg = dims.get('angle', 0.0)
            base = np.array([
                [-B, -H], [B, -H], [B, H], [-B, H]
            ])
            z0 = 0
            z1 = height
            verts = np.vstack([
                np.hstack([base, np.full((4, 1), z0)]),
                np.hstack([base, np.full((4, 1), z1)])
            ])
            faces = np.array([
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
                [4, 5, 6], [4, 6, 7],
                [3, 2, 1], [3, 1, 0]
            ])
            angle_rad = np.radians(angle_deg)
            R = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad),  np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            verts = verts @ R.T
            return verts, faces

        def create_hollow_rect_prism(dims: Dict[str, float], height: float = 1.0, rotate_center: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
            """
            Creates a hollow rectangular prism (completely open at both ends),
            with optional rotation of the cross-section around a specified center.
            Args:
                dims (Dict[str, float]): Dictionary with keys 'B', 'H' (outer dimensions),
                    'b', 'h' (inner dimensions), and 'angle' (rotation).
                height (float, optional): Length of the beam along the Z-axis. Defaults to 1.0.
                rotate_center (Tuple[float, float], optional): Tuple (x, y) specifying the
                    center of rotation. Defaults to (0.0, 0.0).
            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing:
                    - verts (np.ndarray): (N, 3) array of vertices.
                    - faces (np.ndarray): (M, 3) array of triangular face indices.
            """
            B, H = dims['B'] / 2, dims['H'] / 2
            b, h = dims['b'] / 2, dims['h'] / 2
            angle_deg = dims.get('angle', 0.0)
            outer = np.array([
                [-B, -H], [ B, -H], [ B,  H], [-B,  H]
            ])
            inner = np.array([
                [ b, -h], [-b, -h], [-b,  h], [ b,  h]
            ])
            angle_rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)


            def rotate(coords: np.ndarray, center: np.ndarray) -> np.ndarray:
                """Applies 2D rotation to coordinates around a center."""
                coords_shifted = coords - center
                x, y = coords_shifted[:, 0], coords_shifted[:, 1]
                rotated = np.column_stack([cos_a * x - sin_a * y, sin_a * x + cos_a * y])
                return rotated + center

            outer = rotate(outer, np.array(rotate_center))
            inner = rotate(inner, np.array(rotate_center))


            def extrude(points: np.ndarray, z: float) -> np.ndarray:
                """Extrudes 2D points to 3D at a given Z-coordinate."""
                return np.hstack([points, np.full((points.shape[0], 1), z)])

            outer_bottom = extrude(outer, 0.0)
            outer_top = extrude(outer, height)
            inner_bottom = extrude(inner, 0.0)
            inner_top = extrude(inner, height)
            verts = np.vstack([outer_bottom, outer_top, inner_bottom, inner_top])
            ob, ot = 0, 4
            ib, it = 8, 12
            faces = []

            for i in range(4):
                ni = (i + 1) % 4
                faces.append([ob + i, ob + ni, ot + ni])
                faces.append([ob + i, ot + ni, ot + i])

            for i in range(4):
                ni = (i + 1) % 4
                faces.append([ib + ni, ib + i, it + i])
                faces.append([ib + ni, it + i, it + ni])
            return verts, np.array(faces)

        def create_ibeam(dims: Dict[str, float], height: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
            """
            Creates vertices and faces for an I-beam section.
            Args:
                dims (Dict[str, float]): Dictionary containing 'B' (flange width), 'H' (total height),
                    'tw' (web thickness), 'tf' (flange thickness), and 'angle' (rotation).
                height (float, optional): Length of the beam along the Z-axis. Defaults to 1.0.
            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing:
                    - verts (np.ndarray): (N, 3) array of vertices.
                    - faces (np.ndarray): (M, 3) array of triangular face indices.
            """
            b = dims['B']
            h = dims['H']
            tw = dims['tw']
            tf = dims['tf']
            angle_deg = float(dims.get('angle', 0.0))
            top_dims = {"B": b, "H": tf, "angle": 0}
            web_dims = {"B": tw, "H": h - 2 * tf, "angle": 0}
            bot_dims = {"B": b, "H": tf, "angle": 0}
            top = create_rectangular_prism(top_dims, height)
            web = create_rectangular_prism(web_dims, height)
            bot = create_rectangular_prism(bot_dims, height)
            top[0][:, 1] += (h/2 - tf/2)
            web[0][:, 1] += 0
            bot[0][:, 1] -= (h/2 - tf/2)
            verts = np.vstack([top[0], web[0], bot[0]])
            offset = [0, len(top[0]), len(top[0]) + len(web[0])]
            faces = np.vstack([
                top[1],
                web[1] + offset[1],
                bot[1] + offset[2],
            ])
            angle_rad = np.radians(angle_deg)
            R = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad),  np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            verts = verts @ R.T
            return verts, faces

        def create_cbeam(dims: Dict[str, float], height: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
            """
            Creates vertices and faces for a C-beam (channel) section.
            Args:
                dims (Dict[str, float]): Dictionary containing 'B' (flange width), 'H' (total height),
                    'tw' (web thickness), 'tf' (flange thickness), and 'angle' (rotation).
                height (float, optional): Length of the beam along the Z-axis. Defaults to 1.0.
            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing:
                    - verts (np.ndarray): (N, 3) array of vertices.
                    - faces (np.ndarray): (M, 3) array of triangular face indices.
            """
            B = dims['B']
            H = dims['H']
            tw = dims['tw']
            tf = dims['tf']
            angle_deg = float(dims.get('angle', 0.0))
            top_dims = {"B": B, "H": tf, "angle": 0}
            bot_dims = {"B": B, "H": tf, "angle": 0}
            web_dims = {"B": tw, "H": H - 2 * tf, "angle": 0}
            top = create_rectangular_prism(top_dims, height)
            bot = create_rectangular_prism(bot_dims, height)
            web = create_rectangular_prism(web_dims, height)
            top[0][:, 1] += (H/2 - tf/2)
            bot[0][:, 1] -= (H/2 - tf/2)
            web[0][:, 0] -= (B/2 - tw/2)
            verts = np.vstack([top[0], web[0], bot[0]])
            offsets = [0, len(top[0]), len(top[0]) + len(web[0])]
            faces = np.vstack([
                top[1],
                web[1] + offsets[1],
                bot[1] + offsets[2],
            ])
            angle_rad = np.radians(angle_deg)
            R = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad),  np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            verts = verts @ R.T
            return verts, faces

        def create_lbeam(dims: Dict[str, float], height: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
            """
            Creates vertices and faces for an L-beam (angle) section.
            Args:
                dims (Dict[str, float]): Dictionary containing 'B' (flange width), 'H' (total height),
                    'tw' (web thickness), 'tf' (flange thickness), and 'angle' (rotation).
                height (float, optional): Length of the beam along the Z-axis. Defaults to 1.0.
            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing:
                    - verts (np.ndarray): (N, 3) array of vertices.
                    - faces (np.ndarray): (M, 3) array of triangular face indices.
            """
            B = dims['B']
            H = dims['H']
            tw = dims['tw']
            tf = dims['tf']
            angle_deg = dims.get('angle', 0.0)
            flange_dims = {"B": B, "H": tf, "angle": 0.0}
            flange_verts, flange_faces = create_rectangular_prism(flange_dims, height)
            web_dims = {"B": tw, "H": H - tf, "angle": 0.0}
            web_verts, web_faces = create_rectangular_prism(web_dims, height)
            flange_verts[:, 1] -= (H - tf) / 2
            web_verts[:, 0] -= (B - tw) / 2
            web_verts[:, 1] += tf / 2
            verts = np.vstack([flange_verts, web_verts])
            face_offset = len(flange_verts)
            faces = np.vstack([flange_faces, web_faces + face_offset])

            if angle_deg != 0.0:
                angle_rad = np.radians(angle_deg)
                R = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad), 0],
                    [np.sin(angle_rad),  np.cos(angle_rad), 0],
                    [0, 0, 1]
                ])
                centroid = np.mean(verts[:, :2], axis=0)
                verts[:, :2] -= centroid
                verts = (R @ verts.T).T
                verts[:, :2] += centroid
            return verts, faces

        # ---------------------------------------------
        # GLOBAL UTILITIES
        # ---------------------------------------------

        def find_cmax() -> float:
            """
            Calculates the maximum displacement magnitude or von Mises stress across all elements.
            This value is used for color scaling in the plot.
            Returns:
                float: The maximum displacement magnitude or von Mises stress. Returns 1.0 if no
                       elements are found or values are near zero to prevent division by zero.
            """
            element_values = []

            for elem_num, elem in ResultsViewer.imported_data['elements'].items():

                if elem_num == 0:
                    continue
                ni, nj = elem["node1"], elem["node2"]

                if ni not in nodes or nj not in nodes:
                    continue

                if elements_forces:

                    if elem_num in elements_forces and 'stresses' in elements_forces[elem_num] and 'von_mises' in elements_forces[elem_num]['stresses']:
                        elem_val = elements_forces[elem_num]['stresses']['von_mises']
                    else:
                        elem_val = 1.0
                else:
                    _, _, _, _, disp1 = nodes[ni]
                    _, _, _, _, disp2 = nodes[nj]
                    disp_mag1 = np.sqrt(np.nansum(np.array(disp1)**2)) if disp1 else 0.0
                    disp_mag2 = np.sqrt(np.nansum(np.array(disp2)**2)) if disp2 else 0.0
                    elem_val = (disp_mag1 + disp_mag2) / 2.0
                element_values.append(elem_val)
            global_cmax = max(element_values) if element_values else 1.0

            if abs(global_cmax) < 1e-10:
                global_cmax = 1.0
            return global_cmax

        def extract_outer_edges(faces: np.ndarray, verts: np.ndarray) -> List[Tuple[int, int]]:
            """
            Identifies and extracts the outer edges of a 3D mesh.
            Args:
                faces (np.ndarray): (M, 3) array of triangular face indices.
                verts (np.ndarray): (N, 3) array of vertices.
            Returns:
                List[Tuple[int, int]]: A list of tuples, where each tuple represents an
                    outer edge as (vertex_idx1, vertex_idx2).
            """
            edge_faces = defaultdict(list)

            for face_idx, face in enumerate(faces):
                edges = [
                    frozenset({face[0], face[1]}),
                    frozenset({face[1], face[2]}),
                    frozenset({face[2], face[0]})
                ]

                for edge in edges:
                    edge_faces[edge].append(face_idx)

            if len(verts) > 20 and 'Circular' in ResultsViewer.imported_data['elements'][1].get('section_code', ''):
                 radial_edges = set()

                 for edge, face_ids in edge_faces.items():
                     v1, v2 = tuple(edge)
                     z1, z2 = verts[v1,2], verts[v2,2]

                     if abs(z1 - z2) > 1e-6 and len(face_ids) > 1:
                         radial_edges.add(edge)

                 for edge in radial_edges:
                     edge_faces[edge] = [edge_faces[edge][0]]
            outer_edges = [tuple(e) for e, fids in edge_faces.items() if len(fids) == 1]
            return outer_edges

        def create_edge_lines(verts: np.ndarray, edges: List[Tuple[int, int]]) -> Tuple[List[float], List[float], List[float]]:
            """
            Generates coordinate lists suitable for Plotly Scatter3d 'lines' mode
            from vertices and edge indices.
            Args:
                verts (np.ndarray): (N, 3) array of vertices.
                edges (List[Tuple[int, int]]): A list of tuples, each representing an edge
                    as (vertex_idx1, vertex_idx2).
            Returns:
                Tuple[List[float], List[float], List[float]]: Tuples of X, Y, and Z coordinates

                    for the edge lines, with None inserted to break lines between segments.
            """
            edge_x, edge_y, edge_z = [], [], []

            for v1_idx, v2_idx in edges:
                edge_x.extend([verts[v1_idx, 0], verts[v2_idx, 0], None])
                edge_y.extend([verts[v1_idx, 1], verts[v2_idx, 1], None])
                edge_z.extend([verts[v1_idx, 2], verts[v2_idx, 2], None])
            return edge_x, edge_y, edge_z

        # ---------------------------------------------
        # MESH TRANSFORMATION & PLOTTING CONFIGURATION
        # ---------------------------------------------
        
        shape_generators = {
            'Solid_Circular': lambda dims: create_cylinder(radius_dims=dims),
            'Hollow_Circular': lambda dims: create_hollow_cylinder(dims=dims),
            'Solid_Rectangular': lambda dims: create_rectangular_prism(dims=dims),
            'Hollow_Rectangular': lambda dims: create_hollow_rect_prism(dims=dims),
            'I_Beam': lambda dims: create_ibeam(dims=dims),
            'C_Beam': lambda dims: create_cbeam(dims=dims),
            'L_Beam': lambda dims: create_lbeam(dims=dims),
        }


        def transform_mesh(verts: np.ndarray, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> np.ndarray:
            """
            Transforms a mesh from a local coordinate system (aligned with Z-axis)
            to the global coordinates defined by start and end points.
            Args:
                verts (np.ndarray): (N, 3) array of vertices in the local system.
                start (Tuple[float, float, float]): Global (x, y, z) coordinates of the start node.
                end (Tuple[float, float, float]): Global (x, y, z) coordinates of the end node.
            Returns:
                np.ndarray: (N, 3) array of transformed vertices in the global system.
            """
            vec = np.array(end) - np.array(start)
            length = np.linalg.norm(vec)

            if length == 0:
                return None

            direction = vec / length
            z_axis = np.array([0, 0, 1])
            v = np.cross(z_axis, direction)
            c = np.dot(z_axis, direction)

            if np.linalg.norm(v) < 1e-8:

                if c > 0:
                    R = np.eye(3)
                else:
                    R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            else:
                skew = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                R = np.eye(3) + skew + skew @ skew * ((1 - c) / (np.linalg.norm(v) ** 2))
            transformed_verts = verts.copy()
            transformed_verts[:, 2] *= length
            return (R @ transformed_verts.T).T + np.array(start)

        colorscale = [
            [0.0, "blue"], [0.2, "cyan"], [0.4, "lime"],
            [0.6, "yellow"], [0.8, "orange"], [1.0, "red"]
        ]

        if shiny:
            lighting_params = {
                'ambient': 0.3, 'diffuse': 0.8, 'fresnel': 0.2,
                'specular': 0.7, 'roughness': 0.1, 'facenormalsepsilon': 0
            }
            opacity = 0.5
        else:
            lighting_params = {
                'ambient': 0.5, 'diffuse': 0.9, 'fresnel': 0.1,
                'specular': 0.2, 'roughness': 0.5, 'facenormalsepsilon': 0.1
            }
            opacity = 0.4
        imported_elements = ResultsViewer.imported_data['elements']
        cross_sections = ResultsViewer.imported_data['cross_sections']
        cmax = find_cmax()

        if elements_forces:
            scale_factor = 1.0
        else:
            scale_factor = ResultsViewer.CentralDockWindow.scale_slider.value()
        scale_factor_max = ResultsViewer.CentralDockWindow.scale_slider.maximum()
 
        # ---------------------------------------------
        # ITERATE AND PLOT ELEMENTS
        # ---------------------------------------------

        for eid, elem in imported_elements.items():

            if eid == 0:
                continue
            ni, nj = elem.get('node1'), elem.get('node2')
            sec_code = elem.get('section_code')

            if ni not in nodes or nj not in nodes or sec_code not in cross_sections:
                continue
            x1, y1, z1_orig, _, disp1 = nodes[ni]
            x2, y2, z2_orig, _, disp2 = nodes[nj]
            z1 = 0.0 if z1_orig is None else z1_orig
            z2 = 0.0 if z2_orig is None else z2_orig
            elem_color_value: float = 0.0

            if elements_forces:

                if eid in elements_forces and 'stresses' in elements_forces[eid] and 'von_mises' in elements_forces[eid]['stresses']:
                    disp_mag1 = elements_forces[eid]['stresses']['von_mises']
                    disp_mag2 = disp_mag1
            else:
                disp_mag1 = np.sqrt(np.nansum(np.array(disp1)**2)) if disp1 else 0.0
                disp_mag2 = np.sqrt(np.nansum(np.array(disp2)**2)) if disp2 else 0.0
            elem_color_value = (disp_mag1 + disp_mag2) / 2.0
            section = cross_sections[sec_code]
            shape_type = section.get('type')
            dims = section.get('dimensions')

            if shape_type not in shape_generators:
                print(f"Warning: Unsupported shape type '{shape_type}' for element {eid}. Skipping.")
                continue

            try:
                verts, faces = shape_generators[shape_type](dims)

            except Exception as e:
                print(f"Error generating {shape_type} mesh for element {eid}: {e}. Skipping.")
                continue
            transformed_verts = transform_mesh(verts, (x1, y1, z1), (x2, y2, z2))

            if transformed_verts is None:
                continue
            local_i, local_j, local_k = faces.T
            local_x, local_y, local_z = transformed_verts.T
            local_color_values = np.full(faces.shape[0], disp_mag1 * scale_factor)

            if elements_forces:
                intensity_values = [v / cmax for v in local_color_values]
            else:
                intensity_values = [v / (scale_factor_max*cmax) for v in local_color_values]

            if shadow_effect:

                if len(local_z) > 0:
                    min_z, max_z = min(local_z), max(local_z)

                    if max_z != min_z:
                        z_normalized = [(z - min_z)/(max_z - min_z) for z in local_z]
                        intensity_values = [c * (0.7 + 0.3*z) for c, z in zip(intensity_values, z_normalized)]
            hovertext = f"Element: {eid}"

            if elements_forces:
                hovertext += f"<br>Von Mises Stress: {elem_color_value:.3f}"

            if deformed:
                hovertext += f"<br>Displacement: {elem_color_value:.3e}"
            fig.add_trace(go.Mesh3d(
                x=local_x,
                y=local_y,
                z=local_z,
                i=local_i,
                j=local_j,
                k=local_k,
                intensity=intensity_values,
                hoverinfo='text',
                text=[hovertext] * len(local_i), 
                colorscale='rainbow',
                cmin=0,
                cmax=1, 
                opacity=opacity,
                lighting=lighting_params,
                lightposition={'x':0, 'y':0, 'z':100},
                flatshading=False,
                showscale=False
            ))

            if edge_visibility:

                try:
                    outer_edges = extract_outer_edges(faces, transformed_verts)

                    if outer_edges:
                        edge_x, edge_y, edge_z = create_edge_lines(transformed_verts, outer_edges)
                        fig.add_trace(go.Scatter3d(
                            x=edge_x,
                            y=edge_y,
                            z=edge_z,
                            mode='lines',
                            line=dict(
                                color=edge_color,
                                width=edge_width
                            ),
                            hoverinfo='none',
                            showlegend=False
                        ))

                except Exception as e:
                    print(f"Edge generation failed for element {eid}: {e}")
        return fig

    # ---------------------------------------------
    # FORCE/MOMENT/DISPLACEMENT DIAGRAMS
    # ---------------------------------------------

    @staticmethod
    def plot_force_shear_moment(elements_forces: Dict[int, Dict[str, Any]]) -> go.Figure:
        """
        Generates interactive plots for shear force, bending moment, and displacement
        diagrams for structural elements.
        This function iterates through each element, extracts its nodal forces and
        displacements, and then calculates the shear force, bending moment, and
        displacement values along the element's length. These values are then
        plotted using Plotly to create a comprehensive visualization of the
        structural analysis results.
        Args:
            elements_forces (Dict[int, Dict[str, Any]]): A dictionary where keys are
                element IDs (int) and values are dictionaries containing element
                results, specifically 'nodal_forces' (a list or array of forces
                and moments at the nodes) and 'nodal_displacements' (a list or
                array of displacements at the nodes).
                Expected structure for 'nodal_forces':
                [Fx1, Fy1, Mz1, Fx2, Fy2, Mz2]
                Expected structure for 'nodal_displacements':
                [Ux1, Uy1, Rz1, Ux2, Uy2, Rz2]
        Returns:
            go.Figure: A Plotly Figure object containing three subplots:
                       Shear Force Diagram, Bending Moment Diagram, and
                       Displacement Diagram.
        """
        
        # ---------------------------------------------
        # DATA INITIALIZATION
        # ---------------------------------------------

        nodes: Dict[int, Dict[str, Any]] = ResultsViewer.imported_data['nodes']
        elements: Dict[int, Dict[str, Any]] = ResultsViewer.imported_data['elements']
        x_coords: List[float] = []
        shear_forces: List[float] = []
        bending_moments: List[float] = []
        displacements: List[float] = []

        # ---------------------------------------------
        # ELEMENT PROCESSING LOOP
        # ---------------------------------------------

        for elem_id, elem_data in elements.items():

            if elem_id == 0:
                continue
            node1_id: int = elem_data['node1']
            node2_id: int = elem_data['node2']
            x1: float = nodes[node1_id]['X']
            y1: float = nodes[node1_id]['Y']
            x2: float = nodes[node2_id]['X']
            y2: float = nodes[node2_id]['Y']
            L: float = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            D1: float = nodes[node1_id]['displacement'][1]
            D2: float = nodes[node2_id]['displacement'][1]
            elem_result: Dict[str, Any] = elements_forces[elem_id]
            V1: float = -elem_result['nodal_forces'][1]
            M1: float = -elem_result['nodal_forces'][2]
            V2: float = elem_result['nodal_forces'][4]
            M2: float = elem_result['nodal_forces'][5]
            num_points: int = 20
            x_points: np.ndarray = np.linspace(x1, x2, num_points)
            shear: np.ndarray = np.linspace(V1, V2, num_points)
            moment: np.ndarray = M1 + (M2 - M1) * (x_points - x1) / (L + 1e-9)
            disp: np.ndarray = np.linspace(D1, D2, num_points)
            x_coords.extend(x_points.tolist())
            shear_forces.extend(shear.tolist())
            bending_moments.extend(moment.tolist())
            displacements.extend(disp.tolist())

        # ---------------------------------------------
        # PLOT GENERATION
        # ---------------------------------------------

        fig: go.Figure = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(
                "Shear Force Diagram",
                "Bending Moment Diagram",
                "Displacement Diagram"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_coords, y=shear_forces,
                mode='lines', name='Shear Force',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        fig.update_yaxes(title_text="F (kN)", row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=x_coords, y=bending_moments,
                mode='lines', name='Bending Moment',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="M (kN·m)", row=2, col=1)
        fig.add_trace(
            go.Scatter(
                x=x_coords, y=displacements,
                mode='lines', name='Displacement',
                line=dict(color='green', width=2)
            ),
            row=3, col=1
        )
        fig.update_yaxes(title_text="D (mm)", row=3, col=1)

        # ---------------------------------------------
        # LAYOUT CONFIGURATION
        # ---------------------------------------------

        fig.update_layout(
            margin=dict(l=10, r=10, b=30, t=30),
            showlegend=False,
        )

        if x_coords:
            min_x: float = min(x_coords)
            max_x: float = max(x_coords)

            for i in range(1, 4):
                fig.add_shape(
                    type="line",
                    x0=min_x, y0=0,
                    x1=max_x, y1=0,
                    line=dict(color="black", width=1, dash="dot"),
                    row=i, col=1
                )
        else:
            print("Warning: No element data processed. Plots may be empty.")
        return fig


    # ---------------------------------------------
    # GEOMETRY & LOAD VISUALIZATIONS (PLACEHOLDERS)
    # ---------------------------------------------

    @staticmethod
    def add_xyz_triad(fig: go.Figure, nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]]) -> go.Figure:
        """
        Adds an XYZ triad (coordinate axes) to the Plotly figure.
        This is a placeholder function and needs to be implemented to actually
        draw the axes.
        Args:
            fig (go.Figure): The Plotly figure to add the triad to.
            nodes (Dict[int, Tuple[float, float, Optional[float], Any, Any]]):
                Node data, used to determine triad position/size.
        Returns:
            go.Figure: The Plotly figure with the XYZ triad.
        """
        origin=(0, 0, 0)
        axis_length=1.0
        ox, oy, oz = origin
        x_end = (ox + axis_length, oy, oz)
        y_end = (ox, oy + axis_length, oz)
        z_end = (ox, oy, oz + axis_length)
        fig.add_trace(go.Scatter3d(x=[ox, x_end[0]], y=[oy, x_end[1]], z=[oz, x_end[2]],
                                   mode='lines', line=dict(color='red', width=5), name="X-axis"))
        fig.add_trace(go.Scatter3d(x=[ox, y_end[0]], y=[oy, y_end[1]], z=[oz, y_end[2]],
                                   mode='lines', line=dict(color='green', width=5), name="Y-axis"))
        fig.add_trace(go.Scatter3d(x=[ox, z_end[0]], y=[oy, z_end[1]], z=[oz, z_end[2]],
                                   mode='lines', line=dict(color='blue', width=5), name="Z-axis"))
        fig.add_trace(go.Cone(x=[x_end[0]], y=[x_end[1]], z=[x_end[2]], 
                              u=[1], v=[0], w=[0], sizemode="absolute", sizeref=0.5, colorscale="Blues", showscale=False))
        fig.add_trace(go.Cone(x=[y_end[0]], y=[y_end[1]], z=[y_end[2]], 
                              u=[0], v=[1], w=[0], sizemode="absolute", sizeref=0.5, colorscale="Blues", showscale=False))
        fig.add_trace(go.Cone(x=[z_end[0]], y=[z_end[1]], z=[z_end[2]], 
                              u=[0], v=[0], w=[1], sizemode="absolute", sizeref=0.5, colorscale="Blues", showscale=False))
        fig.add_trace(go.Scatter3d(x=[x_end[0]], y=[x_end[1]], z=[x_end[2]], 
                                   mode='text', text=['X'], textposition="top center", textfont=dict(size=16, color='red')))
        fig.add_trace(go.Scatter3d(x=[y_end[0]], y=[y_end[1]], z=[y_end[2]], 
                                   mode='text', text=['Y'], textposition="top center", textfont=dict(size=16, color='green')))
        fig.add_trace(go.Scatter3d(x=[z_end[0]], y=[z_end[1]], z=[z_end[2]], 
                                   mode='text', text=['Z'], textposition="top center", textfont=dict(size=16, color='blue')))
        return fig


    @staticmethod
    def _plot_distributed_loads(fig: go.Figure, nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]], arrows_per_length: int = 5) -> go.Figure:
        """
        Plots distributed loads on the structure.
        This is a placeholder function and needs to be implemented to actually
        draw distributed loads.
        Args:
            fig (go.Figure): The Plotly figure to add loads to.
            nodes (Dict[int, Tuple[float, float, Optional[float], Any, Any]]): Node data.
            arrows_per_length (int): Desired number of arrows per unit length for distributed loads.
        Returns:
            go.Figure: The Plotly figure with distributed loads (if implemented).
        """
        arrows_per_length=5
        include_labels=False
        elements = ResultsViewer.imported_data['elements']
        distributed_loads = ResultsViewer.imported_data['distributed_loads']
        structure_properties = ResultsViewer.imported_data['structure_info']
        force_units  = ResultsViewer.imported_data['saved_units']['Force (Fx,Fy,Fz)']
        length_units  = ResultsViewer.imported_data['saved_units']['Position (X,Y,Z)']
        dist_load_units = f"{force_units}/{length_units}"
        directions = {
            '+Global_X': np.array([1, 0, 0]), '-Global_X': np.array([-1, 0, 0]),
            '+Global_Y': np.array([0, 1, 0]), '-Global_Y': np.array([0, -1, 0]),
            '+Global_Z': np.array([0, 0, 1]), '-Global_Z': np.array([0, 0, -1]),
        }
        load_colorscales = {
            'Uniform': 'Blues',
            'Trapezoidal': 'Greens',
            'Rectangular': 'Oranges',
            'Triangular': 'Reds',
            'Equation': 'Purples'
        }
        all_magnitudes = []

        for load in distributed_loads.values():
            params = load['parameters']

            if isinstance(params, (int, float, list, tuple)):  
                all_magnitudes.extend(np.abs(params) if isinstance(params, (list, tuple)) else [abs(params)])

        if all_magnitudes:
            magnitude_min, magnitude_max = min(all_magnitudes), max(all_magnitudes)
        else:
            magnitude_min, magnitude_max = 2, 2

        for elem, load in distributed_loads.items():

            if elem not in elements or elem == 0:
                continue
            node1, node2 = elements[elem]['node1'], elements[elem]['node2']
            section_code = ResultsViewer.imported_data['elements'][elem]['section_code']

            if section_code in ResultsViewer.imported_data['cross_sections']:
                dim = list(ResultsViewer.imported_data['cross_sections'][section_code]['dimensions'].values())[0]
            else:
                dim = 1.0
            x1, y1, z1 = nodes[node1][:3]  
            z1 = z1 if z1 is not None else 0.0  
            x2, y2, z2 = nodes[node2][:3]  
            z2 = z2 if z2 is not None else 0.0  
            load_type = load['type']
            params = load['parameters']
            direction = directions.get(load['direction'], None)

            if direction is None:
                continue
            element_length = np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])
            num_arrows = max(2, int(arrows_per_length * element_length))
            arrow_positions = np.linspace(0, 1, num_arrows)

            if load_type == 'Uniform':

                if not isinstance(params, (int, float)):
                    params = params[0]
                magnitudes = np.full(num_arrows, params)
                mag = params
            elif load_type in ['Trapezoidal', 'Triangular']:
                q1, q2 = params
                mag = (q1-q2)/element_length
                magnitudes = np.interp(arrow_positions, [0, 1], [q1, q2])
            elif load_type == 'Rectangular':
                q_start, q_end = params
                mag = (q_start-q_end)/element_length
                magnitudes = np.interp(arrow_positions, [0, 1], [q_start, q_end])
            elif load_type == 'Triangular2':
                q_max, x_peak = params
                mag = (q_max)/element_length
                magnitudes = np.interp(arrow_positions, [0, x_peak, 1], [0, 1.5*q_max, 0])
            elif load_type == 'Equation':
                equation = params
                magnitudes = np.array([eval(equation, {'x': pos}) for pos in arrow_positions])
                mag = (magnitudes[0]-magnitudes[-1])/element_length
            else:
                continue
            hovertext = f"{mag:.3g} {dist_load_units} @ E{elem}"
            min_threshold = 0.2
            denominator = max(magnitude_max - magnitude_min, 1e-6)
            normalized_magnitudes = (magnitudes - magnitude_min) / denominator
            normalized_magnitudes = min_threshold + (1 - min_threshold) * normalized_magnitudes
            colorscale = load_colorscales.get(load_type, 'Viridis')

            for i, pos in enumerate(arrow_positions):
                x_arrow = x1 + pos * (x2 - x1)
                y_arrow = y1 + pos * (y2 - y1)
                z_arrow = z1 + pos * (z2 - z1)
                arrow_length = 0.5 * abs(magnitudes[i]) / magnitude_max  
                cone_size = 0.2 * abs(magnitudes[i]) / magnitude_max  
                tail_length = 0.7 * abs(magnitudes[i]) / magnitude_max  
                dx, dy, dz = direction * arrow_length /3 +  direction * dim /2
                cone_base_x = x_arrow - dx  
                cone_base_y = y_arrow - dy 
                cone_base_z = z_arrow - dz 
                tail_start_x, tail_start_y, tail_start_z = cone_base_x, cone_base_y, cone_base_z
                tail_end_x = tail_start_x - direction[0] * tail_length
                tail_end_y = tail_start_y - direction[1] * tail_length
                tail_end_z = tail_start_z - direction[2] * tail_length
                clamped_value = max(0.001, min(0.999, normalized_magnitudes[i]))
                arrow_color = pc.sample_colorscale(colorscale, clamped_value)[0]
                fig.add_trace(go.Scatter3d(
                    x=[tail_start_x, tail_end_x],  
                    y=[tail_start_y, tail_end_y],  
                    z=[tail_start_z, tail_end_z],  
                    mode="lines",
                    line=dict(color=arrow_color, width=5),
                    text=[hovertext]* 2,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False
                ))
                fig.add_trace(go.Cone(
                    x=[x_arrow-dx], y=[y_arrow-dy], z=[z_arrow-dz],  
                    u=[dx], v=[dy], w=[dz],  
                    sizemode="absolute", sizeref=cone_size, 
                    colorscale=colorscale, showscale=False,
                    text=[hovertext]* len([x_arrow-dx]),
                    hovertemplate="%{text}<extra></extra>", 
                ))

            if include_labels:
                fig.add_trace(go.Scatter3d(
                    x=[x_arrow], y=[y_arrow], z=[z_arrow + 0.2],
                    mode='text',
                    textposition="top center",
                    text=[hovertext]* len([x_arrow-dx]),
                    showlegend=False
                ))
        return fig


    @staticmethod
    def _get_cone_size_from_fig(fig: go.Figure) -> float:
        """
        Extracts the sizeref (cone size) from an existing Plotly figure with cone traces.
        Parameters:
            fig (go.Figure): The Plotly figure containing cones.
        Returns:
            float: The sizeref value of the first detected cone, or a default value.
        """

        for trace in fig.data:

            if trace.type == "cone" and hasattr(trace, "sizeref") and trace.name == "distributed_load":
                return trace.sizeref  

        return 0.5  


    @staticmethod
    def _plot_concentrated_loads(fig: go.Figure, nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]]) -> go.Figure:
        """
        Plots concentrated loads (point loads) on the structure.
        This is a placeholder function and needs to be implemented to actually
        draw concentrated loads.
        Args:
            fig (go.Figure): The Plotly figure to add loads to.
            nodes (Dict[int, Tuple[float, float, Optional[float], Any, Any]]): Node data.
        Returns:
            go.Figure: The Plotly figure with concentrated loads (if implemented).
        """
        structure_properties = ResultsViewer.imported_data['structure_info']
        force_units  = ResultsViewer.imported_data['saved_units']['Force (Fx,Fy,Fz)']
        length_units  = ResultsViewer.imported_data['saved_units']['Position (X,Y,Z)']
        moment_units = f"{force_units}.{length_units}"
        force_labels = structure_properties["force_labels"]
        dofs = structure_properties["dofs_per_node"]
        dimension = structure_properties["dimension"]
        force_scale = 0.2
        moment_scale = 0.2
        moment_directions = {"Mx": "red", "My": "blue", "Mz": "green"}
        cone_size = ResultsViewer._get_cone_size_from_fig(fig)
        all_magnitudes = []

        for node_data in nodes.values():
            forces, displacements = node_data[3], node_data[4]
            valid_forces = [abs(f) for f in forces if not np.isnan(f)]
            valid_displacements = [abs(d) for d in displacements if not np.isnan(d)]
            all_magnitudes.extend(valid_forces)
            all_magnitudes.extend(valid_displacements)
        magnitude_min, magnitude_max = (min(all_magnitudes), max(all_magnitudes)) if all_magnitudes else (2, 2)
        min_threshold = 0.3

        for node_id, node_data in nodes.items():
            x, y, z = node_data[:3] if dimension == "3D" else (*node_data[:2], 0)
            force_values = node_data[3]

            if len(force_values) != len(force_labels):
                print(f"Warning: Node {node_id} has an incorrect force format: {force_values}")
                continue

            for i, (label, force) in enumerate(zip(force_labels, force_values)):

                if np.isnan(force) or abs(force) < 1e-10 or label.startswith("M"):
                    continue
                hovertext = f"{force:.3g} {force_units} @ N{node_id}"
                direction = np.array([0, 0, 0])
                direction[i] = np.sign(force)
                magnitude = abs(force)
                normalized_magnitude = (magnitude - magnitude_min) / max(magnitude_max - magnitude_min, 1e-6)
                normalized_magnitude = min_threshold + (1 - min_threshold) * normalized_magnitude
                arrow_length = 0.5
                cone_size = 0.2
                tail_length = 0.7
                dx, dy, dz = direction * arrow_length / 1.8
                cone_base_x = x - dx  
                cone_base_y = y - dy  
                cone_base_z = z - dz  
                tail_start_x, tail_start_y, tail_start_z = cone_base_x, cone_base_y, cone_base_z
                tail_end_x = tail_start_x - direction[0] * tail_length
                tail_end_y = tail_start_y - direction[1] * tail_length
                tail_end_z = tail_start_z - direction[2] * tail_length
                colorscale = 'Reds'
                clamped_value = max(0.001, min(0.999, normalized_magnitude))
                arrow_color = pc.sample_colorscale(colorscale, clamped_value)[0]
                fig.add_trace(go.Scatter3d(
                    x=[tail_start_x, tail_end_x],  
                    y=[tail_start_y, tail_end_y],  
                    z=[tail_start_z, tail_end_z],  
                    mode="lines",
                    line=dict(color=arrow_color, width=5),  
                    showlegend=False,
                    text=[hovertext]* 2,
                    hovertemplate="%{text}<extra></extra>"
                ))
                fig.add_trace(go.Cone(
                    x=[x - dx], y=[y - dy], z=[z - dz],  
                    u=[dx], v=[dy], w=[dz],  
                    sizemode="absolute", sizeref=cone_size, 
                    colorscale=colorscale, showscale=False,
                    text=[hovertext],
                    hovertemplate="%{text}<extra></extra>"
                ))

            for i, (label, moment) in enumerate(zip(force_labels, force_values)):

                if "M" in label and not (np.isnan(moment) or abs(moment) < 1E-10):
                    color = moment_directions.get(label, "black")
                    moment_sign = 1 if moment > 0 else -1
                    theta = np.linspace(np.pi, 0, 20) if moment_sign < 0 else np.linspace(0, np.pi, 20)
                    arc_x = (x + moment_scale * np.cos(theta)).tolist()
                    arc_y = (y + moment_scale * np.sin(theta)).tolist()
                    arc_z = [z] * len(arc_x)
                    cone_x, cone_y, cone_z = arc_x[-1], arc_y[-1], arc_z[-1]
                    last_theta = theta[-1]
                    dx = -np.sin(last_theta)
                    dy = np.cos(last_theta)
                    dz = 0

                    if moment_sign < 0:
                        dx, dy = -dx, -dy
                    hovertext = f"{label}: {moment:.3g} {moment_units} at Node {node_id}"
                    fig.add_trace(go.Scatter3d(
                        x=arc_x, y=arc_y, z=arc_z,
                        mode="lines",
                        line=dict(color=color, width=5),
                        text=[hovertext]*len(arc_x),
                        hovertemplate="%{text}<extra></extra>"
                    ))
                    fig.add_trace(go.Cone(
                        x=[cone_x], y=[cone_y], z=[cone_z],
                        u=[dx], v=[dy], w=[dz],
                        sizemode="absolute", 
                        sizeref=cone_size,
                        colorscale=[[0, color], [1, color]], 
                        showscale=False,
                        text=[hovertext]*len([cone_x]),
                        hovertemplate="%{text}<extra></extra>"
                    ))
        return fig


    @staticmethod
    def _set_equal_axes(fig: go.Figure, nodes: Dict[int, Tuple[float, float, Optional[float], Any, Any]]) -> go.Figure:
        """
        Sets equal scaling for axes in the Plotly figure to maintain aspect ratio.
        This is a placeholder function and needs to be implemented to actually
        ensure equal axes.
        Args:
            fig (go.Figure): The Plotly figure to modify.
            nodes (Dict[int, Tuple[float, float, Optional[float], Any, Any]]): Node data.
        Returns:
            go.Figure: The Plotly figure with equalized axes (if implemented).
        """
        extra_margin=0.0
        x_vals, y_vals, z_vals = [], [], []

        for trace in fig.data:

            if hasattr(trace, 'x') and hasattr(trace, 'y') and hasattr(trace, 'z'):
                x_vals.extend([v for v in trace.x if v is not None])
                y_vals.extend([v for v in trace.y if v is not None])
                z_vals.extend([v for v in trace.z if v is not None])

        if not x_vals or not y_vals or not z_vals:
            return fig

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        margin = max_range * extra_margin
        half_range = max_range / 2 + margin
        x_limits = [x_mid - half_range, x_mid + half_range]
        y_limits = [y_mid - half_range, y_mid + half_range]
        z_limits = [z_mid - half_range, z_mid + half_range]
        fig.update_layout(scene=dict(
            xaxis=dict(range=x_limits),
            yaxis=dict(range=y_limits),
            zaxis=dict(range=z_limits),
            aspectmode='cube'
        ))
        return fig

# ---------------------------------------------
# Stiffness Matrix Display
# ---------------------------------------------

    @staticmethod
    def display_stiffness_matrix_with_solution(F: Dict[int, Dict[str, float]], 
                                                K_global: np.ndarray, 
                                                D: Dict[int, Dict[str, float]], 
                                                solution: bool = True) -> None:
        """Displays the global stiffness equation (F = K * D) in a table format.
        This method populates a PyQt5 QTableWidget with the force vector (F),
        the global stiffness matrix (K_global), and the displacement vector (D).
        It handles both dense and sparse stiffness matrices and applies color-coding

        for better readability.
        Args:
            F (Dict[int, Dict[str, float]]): A dictionary where keys are node numbers and values
                                            are dictionaries of force components (e.g., {'Fx': 100.0}).
            K_global (np.ndarray): The global stiffness matrix, which can be dense or sparse (e.g., scipy.sparse matrix).
            D (Dict[int, Dict[str, float]]): A dictionary where keys are node numbers and values
                                            are dictionaries of displacement components (e.g., {'u': 0.05}).
            solution (bool, optional): If True, displays computed numerical values for forces and displacements.
                                    If False, displays labels for forces and generic values for displacements.
                                    Defaults to True.
        """
        table = ResultsViewer.CentralDockWindow.stiffness_table
        rows, cols = K_global.shape
        total_cols = cols + 4
        table.setRowCount(rows)
        table.setColumnCount(total_cols)
        node_dofs: list[str] = ResultsViewer.imported_data["structure_info"]["force_labels"]
        num_nodes: int = rows // len(node_dofs)
        f_labels: list[str] = [f"{dof}{node+1}" for node in range(num_nodes) for dof in node_dofs]
        table.setVerticalHeaderLabels(f_labels)
        table.setHorizontalHeaderLabels(["F", "="] + f_labels + ["*", "D"])
 
        # ---------------------------------------------
        # SET COLUMN WIDTHS
        # ---------------------------------------------

        equal_sign_width: int = 30
        standard_column_width: int = 80
        table.setColumnWidth(0, standard_column_width)
        table.setColumnWidth(1, equal_sign_width)

        for j in range(2, cols + 2):
            table.setColumnWidth(j, standard_column_width)
        table.setColumnWidth(cols + 2, equal_sign_width)
        table.setColumnWidth(cols + 3, standard_column_width)

        # ---------------------------------------------
        # PREPARE FLAT F AND D VECTORS
        # ---------------------------------------------

        force_components: list[str] = ResultsViewer.imported_data["structure_info"]["force_labels"]
        displacement_components: list[str] = ResultsViewer.imported_data["structure_info"]["displacement_labels"]
        flat_F: list[float] = []

        for node in sorted(F.keys()):

            for comp in force_components:
                flat_F.append(F[node].get(comp, 0.0))
        flat_D: list[float] = []

        for node in sorted(D.keys()):

            for comp in displacement_components:
                flat_D.append(D[node].get(comp, 0.0))

        # ---------------------------------------------
        # FILL F, '=', AND '*' COLUMNS
        # ---------------------------------------------

        for i, label in enumerate(f_labels):
            f_value: float = flat_F[i]

            if solution is True:
                f_display: str = f"{f_value:.3g}"
            elif isinstance(f_value, (int, float)):
                f_display: str = f"{f_value:.3g}"
            else:
                f_display: str = f"{label}"
            f_item = QTableWidgetItem(f_display)
            f_item.setFlags(f_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, 0, f_item)

            # ---------------------------------------------
            # FILL '=' SIGN
            # ---------------------------------------------

            equal_item = QTableWidgetItem("=")
            equal_item.setFlags(equal_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            equal_item.setBackground(QColor(12, 12, 12))
            table.setItem(i, 1, equal_item)

            # ---------------------------------------------
            # FILL '*' SIGN
            # ---------------------------------------------

            multiply_item = QTableWidgetItem("*")
            multiply_item.setFlags(multiply_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            multiply_item.setBackground(QColor(12, 12, 12))
            table.setItem(i, cols + 2, multiply_item)

        # ---------------------------------------------
        # HANDLE DENSE OR SPARSE MATRIX
        # ---------------------------------------------

        if hasattr(K_global, "tocoo"):
            K_global_coo = K_global.tocoo()
            max_value: float = K_global_coo.data.max() if K_global_coo.data.size > 0 else 1.0
            min_value: float = K_global_coo.data.min() if K_global_coo.data.size > 0 else 0.0
            k_entries: Dict[Tuple[int, int], float] = {(i, j): v for i, j, v in zip(K_global_coo.row, K_global_coo.col, K_global_coo.data)}
        else:
            max_value: float = np.max(K_global) if K_global.size > 0 else 1.0
            min_value: float = np.min(K_global) if K_global.size > 0 else 0.0
            k_entries: Dict[Tuple[int, int], float] = {(i, j): K_global[i, j] for i in range(rows) for j in range(cols)}

        # ---------------------------------------------
        # FILL STIFFNESS MATRIX VALUES
        # ---------------------------------------------

        for i in range(rows):

            for j in range(cols):
                value: float = k_entries.get((i, j), 0.0)
                item = QTableWidgetItem(f"{value:.2f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                # ---------------------------------------------
                # COLORING
                # ---------------------------------------------

                if max_value != min_value:
                    intensity: float = (value - min_value) / (max_value - min_value)
                else:
                    intensity: float = 1.0
                red: int = int(255 * (1 - intensity))
                blue: int = int(255 * (1 - intensity))
                item.setBackground(QColor(red, 255, blue))

                if i == j:
                    item.setBackground(QColor(139, 89, 89))
                elif value == 0:
                    item.setBackground(QColor(220, 220, 220))
                table.setItem(i, j + 2, item)

        # ---------------------------------------------
        # FILL D COLUMN
        # ---------------------------------------------

        for i in range(rows):

            if solution:
                d_item = QTableWidgetItem(f"{flat_D[i]:.3g}")
            else:
                d_item = QTableWidgetItem(str(flat_D[i]))
            d_item.setFlags(d_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, cols + 3, d_item)

        # ---------------------------------------------
        # RESIZE COLUMNS
        # ---------------------------------------------

        for col in range(table.columnCount()):
            table.resizeColumnToContents(col)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)