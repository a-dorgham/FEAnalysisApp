import sys
import json
import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QDockWidget,
    QLineEdit, QScrollArea, QPushButton, QLabel, QSizePolicy, QToolBar,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QComboBox, QFormLayout, 
    QMessageBox, QSizePolicy, QTableWidget, QTableWidgetItem,
    QFileDialog, QProgressDialog, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer, QMarginsF, QByteArray, QPoint
from PyQt6.QtGui import (QIcon, QAction, QPixmap, 
                         QPageLayout, QPageSize)
from PyQt6.QtWebEngineWidgets import QWebEngineView
import typing 
from typing import Dict, Any, Optional, List, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from utils.utils_classes import OutputStream
from utils.file_io import FileIO
from gui.left_dock_window import LeftDockWindow
from gui.central_dock_window import CentralDockWindow
from gui.units_handling import UnitsHandling
from gui.results_viewer import ResultsViewer
from core.process_imported_structure import ProcessImportedStructure


class SetupDocks(QMainWindow):
    """
    The `SetupDocks` class is the main window class for the FEAnalysisApp application.
    It inherits from `QMainWindow` and is responsible for setting up the main UI layout,
    including the central display area, menu bar, toolbar, and various dockable widgets

    for model information, messages, and input/output.
    It manages the loading of default files, handles user interactions with the toolbar,
    and facilitates the display and export of structural analysis data.
    - `imported_data` (dict | None): A class-level variable to store the imported
      structural data, accessible across different parts of the application. Initialized to `None`.
    """
    imported_data: dict | None = None

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self, main_window: QMainWindow = None) -> None:
        """
        Initializes the `SetupDocks` main window.
        - `main_window` (QMainWindow, optional): A reference to the parent main window if `SetupDocks`
          is part of a larger application. Defaults to `None`.
        """
        super().__init__()
        self.is_minimized: bool = False
        self.file_html: str = "" 
        self.main_window = main_window
        self.load_defaults()
        self.setWindowTitle("FEAnalysisApp v1.0")
        self.resize(1200, 800)
        self.create_docks()

    # ---------------------------------------------
    # DATA LOADING & INITIALIZATION
    # ---------------------------------------------

    def load_defaults(self, file_path: str = None, get_file_path: bool = False) -> bool:
        """
        Loads default or user-specified input files for the FEA model.
        This method is called during initialization to set up the initial state
        of the application with a predefined structure. It also handles unit conversion

        for the loaded data.
        - `file_path` (str, optional): The explicit path to a file to load. If `None`,
          a default example file is loaded based on the `structure_type` from `main_window`.
          Defaults to `None`.
        - `get_file_path` (bool, optional): If `True`, a file dialog will be opened for
          the user to select a file, overriding `file_path`. Defaults to `False`.
        """
        response = self.load_default_files(file_path, get_file_path)

        if self.imported_data and response:
            self.imported_data['saved_units'] = self.imported_data['units'].copy()
            UnitsHandling(self.imported_data, left_dock_window=LeftDockWindow)
        return response

    def load_default_files(self, file_path: str = None, get_file_path: bool = False) -> bool:
        """
        Loads a default example file based on the selected structure type, or a user-specified file.
        It utilizes the `FileIO.toolbar_open` method to handle file loading and parsing.
        After successful loading, it initializes `ProcessImportedStructure` to process the data.
        - `file_path` (str, optional): The explicit path to the file to load. Defaults to `None`.
        - `get_file_path` (bool, optional): If `True`, a file dialog will be displayed
          to allow the user to select a file. Defaults to `False`.
        """
        structure_type: str = self.main_window.structure_type if self.main_window else "2D_Beam"

        if not file_path and not get_file_path:
            file_map: dict[str, str] = {
                "2D_Truss": "2D_Truss.txt",
                "3D_Truss": "3D_Truss.txt",
                "2D_Beam": "2D_Beam.txt",
                "3D_Frame": "3D_Frame_ex50.txt"
            }
            file_mapping: str | None = file_map.get(structure_type)
            file_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples/")
            file_path = os.path.join(file_dir, file_mapping) if file_mapping else None

        if file_path or get_file_path:
            imported_data, self.file_html = FileIO.toolbar_open(self, file_path, structure_type)

            if imported_data and self.file_html:
                self.imported_data = imported_data.copy()
                ProcessImportedStructure(structure_type=structure_type, imported_data=self.imported_data)
                return True

        return False

    # ---------------------------------------------
    # UI SETUP
    # ---------------------------------------------

    def create_docks(self) -> None:
        """
        Creates and arranges the main dockable widgets within the `QMainWindow`.
        This includes the central dock, menu bar, toolbar, and the bottom message dock.
        It also redirects `sys.stdout` to the message box for console output.
        """
        self.create_central_dock()
        self.central_dock_window.file_canvas.setHtml(self.file_html)
        self.create_MenuBar()
        self.create_ToolBar()
        self.create_bottom_dock()
        self.output_stream: OutputStream = OutputStream(self.logging_console)
        sys.stdout = self.output_stream
        LeftDockWindow(imported_data=self.imported_data, output_stream=self.output_stream, central_dock=self.central_dock_window, parent=self)
        self.create_left_dock()


    def create_MenuBar(self) -> None:
        """
        Creates the application's menu bar with "File", "Edit", and "Help" menus.
        """
        menuBar: typing.Type[QToolBar] = self.menuBar()
        fileMenu: QAction = menuBar.addMenu("&File")
        editMenu: QAction = menuBar.addMenu("&Edit")
        helpMenu: QAction = menuBar.addMenu(QIcon("./icons/analysis_window/plus_icon.png"), "&Help")
        self.setMenuBar(menuBar)


    def create_ToolBar(self) -> None:
        """
        Creates the application's toolbar with actions like New, Open, Save, Solve, Export, Units, and Default View.
        Each action is associated with an icon and a specific function.
        """
        toolBar: QToolBar = QToolBar("Main Toolbar", self)


        def add_toolbar_action(toolBar: QToolBar, icon_path: str, text: str, tooltip: str, icon_size: int, icon_function: typing.Callable) -> None:
            """
            Helper function to create and add an action to the toolbar with a specified icon size.
            - `toolBar` (QToolBar): The toolbar to add the action to.
            - `icon_path` (str): The file path to the icon image.
            - `text` (str): The text to display for the action.
            - `tooltip` (str): The tooltip text for the action.
            - `icon_size` (int): The desired size (width and height) for the icon.
            - `icon_function` (typing.Callable): The function to be called when the action is triggered.
            """
            action: QAction = QAction(QIcon(QPixmap(icon_path).scaled(icon_size, icon_size)), text, self)
            action.setToolTip(tooltip)
            action.triggered.connect(icon_function)
            toolBar.addAction(action)
        left_spacer: QWidget = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolBar.addWidget(left_spacer)
        folder: str = "../icons/analysis_window/"
        actions: list[tuple[str, str, str, int, typing.Callable]] = [
            ("new_icon.png", "New", "New File", 35, self.toolbar_new),
            ("open_icon.png", "Open", "Open File", 35, self.toolbar_open),
            ("save_icon.png", "Save", "Save Structure", 30-2, self.toolbar_save),
            ("solve.png", "Exit", "Solve Structure", 30+2, self.toolbar_solve),
            ("export_icon.png", "Export", "Export View/Report", 30, self.toolbar_export),
            ("units_icon.png", "Units", "Set Units", 30-2, self.toolbar_units),
            ("default_view_icon.png", "Default View", "Reset Views", 30, self.restore_default_view)
        ]

        for icon_path, text, tooltip, icon_size, icon_function in actions:
            add_toolbar_action(toolBar, f'{folder}{icon_path}', text, tooltip, icon_size, icon_function)
        right_spacer: QWidget = QWidget()
        right_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolBar.addWidget(right_spacer)
        self.addToolBar(toolBar)


    def create_central_dock(self) -> None:
        """
        Creates the central dock widget, which hosts the main display area for the FEA model.
        It instantiates `CentralDockWindow` and sets its central widget as the main window's central widget.
        """
        self.central_dock_window: CentralDockWindow = CentralDockWindow(imported_data=self.imported_data, parent=self)
        self.central_dock_widget: QWidget = self.central_dock_window.centralWidget()
        self.setCentralWidget(self.central_dock_widget)


    def create_left_dock(self) -> None:
        """
        Creates and adds the left dock widgets to the main window.
        This method utilizes the static references within `LeftDockWindow` to add
        its boundary and details docks to the left dock area.
        """

        if self.imported_data is None:
            print("Warning: imported_data is None. LeftDockWindow may not function properly.")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, LeftDockWindow.boundary_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, LeftDockWindow.details_dock)


    def create_right_dock(self) -> None:
        """
        Creates the right dock widget for displaying model inspection and editing capabilities.
        This dock includes tree-like navigation for node, element, boundary condition,
        mesh statistics, and solver status information.
        """
        self.right_dock: QDockWidget = QDockWidget("Model Inspector", self)
        self.right_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        tree_widget: QTreeWidget = QTreeWidget()
        tree_widget.setHeaderHidden(True)
        node_info_item: QTreeWidgetItem = QTreeWidgetItem(tree_widget, ["Node Information"])
        selected_node_label: QLabel = QLabel("Selected Node: None")
        node_coordinates_label: QLabel = QLabel("Coordinates: (0, 0, 0)")
        element_info_item: QTreeWidgetItem = QTreeWidgetItem(tree_widget, ["Element Information"])
        selected_element_label: QLabel = QLabel("Selected Element: None")
        element_type_label: QLabel = QLabel("Type: -")
        bc_info_item: QTreeWidgetItem = QTreeWidgetItem(tree_widget, ["Boundary Conditions"])
        applied_loads_label: QLabel = QLabel("Applied Loads: 0")
        applied_constraints_label: QLabel = QLabel("Applied Constraints: 0")
        mesh_stats_item: QTreeWidgetItem = QTreeWidgetItem(tree_widget, ["Mesh Statistics"])
        num_nodes_label: QLabel = QLabel("Nodes: 0")
        num_elements_label: QLabel = QLabel("Elements: 0")
        solver_status_item: QTreeWidgetItem = QTreeWidgetItem(tree_widget, ["Solution Status"])
        solver_progress_label: QLabel = QLabel("Progress: Not Started")
        computation_time_label: QLabel = QLabel("Computation Time: 0s")
        inspector_widget: QWidget = QWidget()
        inspector_layout: QVBoxLayout = QVBoxLayout(inspector_widget)
        inspector_layout.addWidget(selected_node_label)
        inspector_layout.addWidget(node_coordinates_label)
        inspector_layout.addWidget(selected_element_label)
        inspector_layout.addWidget(element_type_label)
        inspector_layout.addWidget(applied_loads_label)
        inspector_layout.addWidget(applied_constraints_label)
        inspector_layout.addWidget(num_nodes_label)
        inspector_layout.addWidget(num_elements_label)
        inspector_layout.addWidget(solver_progress_label)
        inspector_layout.addWidget(computation_time_label)
        inspector_layout.addStretch()
        inspector_widget.setLayout(inspector_layout)
        scroll_area: QScrollArea = QScrollArea()
        scroll_area.setWidget(tree_widget)
        scroll_area.setWidgetResizable(True)
        self.right_dock.setWidget(scroll_area)
        self.right_dock.setWidget(inspector_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)


    def create_bottom_dock(self) -> None:
        """
        Creates the bottom dock widget, which serves as a message box to display
        application logs, errors, and other output.
        """
        self.bottom_dock: QDockWidget = QDockWidget("Message Box", self)
        self.bottom_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.logging_console: QTextEdit = QTextEdit(self)
        self.logging_console.setReadOnly(True)
        dock_widget: QWidget = QWidget()
        layout: QVBoxLayout = QVBoxLayout(dock_widget)
        layout.addWidget(self.logging_console)
        layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_dock.setWidget(dock_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)
        self.resizeDocks([self.bottom_dock], [100], Qt.Orientation.Vertical)


    def toggle_dock(self) -> None:
        """
        Toggles the floating state of the bottom message dock.
        If the dock is currently floating, it will be re-docked.
        If it's docked, it will become floating.
        """

        if self.bottom_dock.isFloating():
            self.bottom_dock.setFloating(False)
        else:
            self.bottom_dock.setFloating(True)
            self.bottom_dock.setMaximumHeight(500)


    def toggle_minimize_maximize(self) -> None:
        """
        Toggles the minimized/maximized state of the bottom message dock.
        When minimized, the dock collapses to show only its title bar.
        When maximized, it restores to its previous height.
        """

        if self.is_minimized:
            self.bottom_dock.setMaximumHeight(400)
            QTimer.singleShot(0, lambda: self.resizeDocks([self.bottom_dock], [self.previous_size], Qt.Orientation.Vertical))
            self.is_minimized = False
        else:
            self.previous_size: int = self.bottom_dock.height()
            title_bar_height: int = self.bottom_dock.titleBarWidget().height()
            self.bottom_dock.setMaximumHeight(25)
            self.is_minimized = True

    # ---------------------------------------------
    # TOOLBAR ACTIONS
    # ---------------------------------------------

    def toolbar_new(self) -> None:
        """
        Handles the "New File" action from the toolbar.
        It prompts the user with a warning about data loss and, if confirmed,
        resets the `imported_data` to a default state, preserving the structure type,
        and updates the Left Dock's tree view.
        """
        result: bool = FileIO.warning_message(
            self,
            "Warning",
            "Are you sure?",
            "Opening a new file will erase the current structure and all analysis results. Do you want to continue?",
            "This action cannot be undone. Please ensure you have saved any important data before proceeding."
        )

        if result:
            structure_info: dict = self.imported_data['structure_info']
            self.imported_data = FileIO.initialize_default_data()
            self.imported_data['structure_info'] = structure_info
            LeftDockWindow.tree_widget.update_tree(self.imported_data)


    def toolbar_open(self) -> None:
        """
        Handles the "Open File" action from the toolbar.
        It triggers the loading of a new file (via a file dialog) and
        updates the Left Dock's tree view and the central display area
        with the content of the newly opened file.
        """
        response = self.load_defaults(get_file_path=True)

        if response:
            self.reset_post_process_tabs()
            LeftDockWindow.tree_widget.update_tree(self.imported_data)

            if self.file_html:
                self.central_dock_window.file_canvas.setHtml(self.file_html)


    def toolbar_save(self) -> None:
        """
        Handles the "Save Structure" action from the toolbar.
        It opens a file dialog for the user to select a save location and filename,
        then calls `FileIO.save_structure_to_file` to write the current
        `imported_data` to the specified file.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Structure Information",
            "",
            "TXT Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        if not file_path.endswith('.txt'):
            file_path += '.txt'
        FileIO.save_structure_to_file(self.imported_data, file_path)


    def toolbar_export(self) -> None:
        """
        Handles the "Export" action from the toolbar.
        Currently, it triggers the `export_to_pdf` method to generate a PDF report.
        """
        self.export_to_pdf()


    def toolbar_solve(self) -> None:
        """
        Handles the "Solve Structure" action from the toolbar.
        It calls the `_main_solver` method of the `LeftDockWindow.tree_widget`
        to initiate the structural analysis.
        """
        LeftDockWindow.tree_widget._main_solver()


    def toolbar_units(self) -> None:
        """
        Handles the "Set Units" action from the toolbar.
        It opens the unit selection window, allowing the user to change the
        application's units. The currently saved units are passed to the window.
        """
        UnitsHandling.unit_selection_window(parent=self, selected_units=self.imported_data['saved_units'])


    def reset_post_process_tabs(self, tabs_to_reset: Optional[List[str]] = None) -> None:
        """
        Resets content of specified post-processing tabs to their default state.
        If no tabs are specified, all known tabs will be reset.
        Args:
            tabs_to_reset (Optional[List[str]]): List of tab names to reset. Resets all if None.
        """
        all_tabs = {
            "File": self.central_dock_window.file_canvas,
            "Stiffness-Matrix": self.central_dock_window.stiffness_table,
            "Displacements": self.central_dock_window.deformation_canvas,
            "Stresses": self.central_dock_window.stress_canvas,
            "Forces": self.central_dock_window.force_canvas,
            "Information": self.central_dock_window.info_canvas
        }

        if tabs_to_reset is None:
            tabs_to_reset = list(all_tabs.keys())

        for tab_name in tabs_to_reset:
            widget = all_tabs.get(tab_name)

            if isinstance(widget, QWebEngineView):
                html_content = (
                    ResultsViewer.generate_html_with_plotly()

                    if tab_name != "File" and tab_name != "Information"
                    else "<html><body><p>Empty</p></body></html>"
                )
                widget.setHtml(html_content)

                if tab_name != "File" and tab_name != "Information":
                    widget.plotly_figure = ResultsViewer._create_plotly_figure()
            elif isinstance(widget, QTableWidget):
                widget.clear()
                widget.setRowCount(0)
                widget.setColumnCount(0)


    def restore_default_view(self) -> None:
        """
        Restores the default layout of the docks within the main window.
        It shows and re-adds the left, right (if any), bottom docks, and the central widget
        to their default positions.
        """
        print("Default View action triggered")
        LeftDockWindow.boundary_dock.show()
        LeftDockWindow.details_dock.show()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, LeftDockWindow.boundary_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, LeftDockWindow.details_dock)
        self.bottom_dock.show()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)
        self.central_dock_widget.show()
        self.setCentralWidget(self.central_dock_widget)
        print("Views reset to default.")

    # ---------------------------------------------
    # PDF EXPORT FUNCTIONS
    # ---------------------------------------------

    def export_to_pdf(self, scale_factor: float = 1.0) -> None:
        """
        Exports the content of the currently displayed central widget to a PDF file.
        It handles both `QWebEngineView` (for HTML reports) and `QTableWidget` (for matrices),
        scaling the content to fit an A4 page with specified margins.
        A progress dialog is shown during the PDF generation.
        - `scale_factor` (float, optional): A scaling factor to apply to the content
          before printing to PDF. Defaults to `1.0`.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Report",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )

        if not file_path:
            return

        if not file_path.endswith('.pdf'):
            file_path += '.pdf'
        progress: QProgressDialog = QProgressDialog("Generating PDF...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        progress.setValue(0)
        page_layout: QPageLayout = QPageLayout(
            QPageSize(QPageSize.PageSizeId.A4),
            QPageLayout.Orientation.Portrait,
            QMarginsF(10, 10, 10, 10),
            QPageLayout.Unit.Millimeter
        )

        def handle_pdf_generated(pdf_data: QByteArray) -> None:
            """
            Callback function executed after PDF data is generated by `printToPdf`.
            It saves the PDF data to the specified file and updates the progress.
            - `pdf_data` (QByteArray): The raw PDF data.
            """

            try:
                with open(file_path, 'wb') as f:
                    f.write(pdf_data)
                progress.setValue(100)
                print(f"Export Complete. PDF saved to: {file_path}")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to save PDF:\n{str(e)}",
                    QMessageBox.StandardButton.Ok
                )
            finally:
                progress.close()
        widget: QWidget = self.central_dock_window.current_canvas

        if isinstance(widget, QWebEngineView):
            js: str = """
            (function() {
                var oldStyle = document.getElementById('print-scale-style');

                if (oldStyle) oldStyle.remove();
                var newPrintStyle = document.createElement('style');
                newPrintStyle.id = 'print-scale-style';
                newPrintStyle.media = 'print';
                newPrintStyle.innerHTML = `

                    @page {
                        size: A4;
                        margin: 10mm 10mm 10mm 10mm;
                    }
                    body {
                        transform: scale(%f);
                        transform-origin: top left;
                        width: %f%%;
                        margin: 0 !important;
                        padding: 0 !important;
                        line-height: 1; 
                    }
                    html, body {
                        height: %f%%;
                        overflow: visible !important;
                    }
                    .avoid-break {
                        page-break-inside: avoid;
                    }
                `;
                document.head.appendChild(newPrintStyle);
                return true;

            })();
            """ % (scale_factor, 100/scale_factor, 100/scale_factor)
            def on_js_completed(success: bool) -> None:
                """
                Callback executed after JavaScript injection.
                If successful, it prints the web page to PDF.
                - `success` (bool): True if JavaScript execution was successful.
                """

                if success:
                    page_layout_print: QPageLayout = QPageLayout(
                        QPageSize(QPageSize.PageSizeId.A4),
                        QPageLayout.Orientation.Portrait,
                        QMarginsF(10, 10, 10, 10),
                        QPageLayout.Unit.Millimeter
                    )
                    widget.page().printToPdf(
                        lambda pdf_data: self._finalize_pdf(pdf_data, file_path, progress, widget),
                        pageLayout=page_layout_print
                    )
            widget.page().runJavaScript(js, on_js_completed)
            progress.canceled.connect(lambda: widget.stop() if hasattr(widget, 'stop') else None)
        elif isinstance(widget, QTableWidget):

            try:
                temp_web_view: QWebEngineView = QWebEngineView()
                html: str = self._convert_table_to_html(widget)
                temp_web_view.setHtml(html)


                def on_load_finished(ok: bool) -> None:
                    """
                    Callback executed when the temporary web view finishes loading HTML.
                    It then scales the table content using JavaScript and prints to PDF.
                    - `ok` (bool): True if HTML loading was successful.
                    """

                    if ok:
                        page_width_mm: float = 297 - 20
                        mm_to_px: float = 3.78
                        js_scale: str = """
                        // Get the table and its container
                        const table = document.querySelector('table');
                        const container = document.querySelector('.table-container');
                        // Calculate available space in pixels
                        const pageWidthPx = %f * %f;
                        const tableWidthPx = table.offsetWidth;
                        // Calculate scale factor (with 5%% margin for safety)
                        const scale = Math.min(1, (pageWidthPx * 0.95) / tableWidthPx);
                        // Apply scaling transformation
                        table.style.transform = `scale(${scale})`;
                        table.style.transformOrigin = 'top left';
                        // Adjust container dimensions to match scaled table for proper PDF rendering
                        container.style.width = `${tableWidthPx * scale}px`;
                        container.style.height = `${table.offsetHeight * scale}px`;
                        container.style.overflow = 'hidden';
                        // Return the calculated scale factor to Python
                        scale;
                        """ % (page_width_mm, mm_to_px)


                        def on_js_finished(scale_factor_js: float) -> None:
                            """
                            Callback executed after JavaScript scaling is applied.
                            It then prints the scaled content to PDF.
                            - `scale_factor_js` (float): The actual scale factor applied by JavaScript.
                            """

                            if scale_factor_js:
                                page_layout_table: QPageLayout = QPageLayout(
                                    QPageSize(QPageSize.PageSizeId.A4),
                                    QPageLayout.Orientation.Landscape,
                                    QMarginsF(10, 10, 10, 10),
                                    QPageLayout.Unit.Millimeter
                                )
                                temp_web_view.page().printToPdf(
                                    handle_pdf_generated,
                                    pageLayout=page_layout_table
                                )
                        temp_web_view.page().runJavaScript(js_scale, on_js_finished)
                temp_web_view.loadFinished.connect(on_load_finished)
                progress.canceled.connect(temp_web_view.stop)

            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "Export Failed", f"Error: {str(e)}")
        else:
            progress.close()
            QMessageBox.warning(
                self,
                "Export Failed",
                "No exportable content found",
                QMessageBox.StandardButton.Ok
            )


    def _finalize_pdf(self, pdf_data: QByteArray, file_path: str, progress: QProgressDialog, widget: QWebEngineView) -> None:
        """
        Finalizes the PDF export process by saving the generated PDF data to a file
        and performing cleanup (e.g., removing injected JavaScript styles).
        - `pdf_data` (QByteArray): The raw PDF data generated by `printToPdf`.
        - `file_path` (str): The full path to the output PDF file.
        - `progress` (QProgressDialog): The progress dialog to update and close.
        - `widget` (QWebEngineView): The web engine view where the content was rendered.
        """

        try:
            with open(file_path, 'wb') as f:
                f.write(pdf_data)
            clean_js: str = """
            (function() {
                var elem = document.getElementById('print-scale-style');

                if (elem) elem.remove();
                return true;

            })();
            """
            widget.page().runJavaScript(clean_js)
            print(f"Export Complete. PDF saved to: {file_path}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to save PDF:\n{str(e)}",
                QMessageBox.StandardButton.Ok
            )
        finally:
            progress.close()


    def _convert_table_to_html(self, table: QTableWidget) -> str:
        """
        Converts a `QTableWidget`'s content into an HTML table string,
        including basic CSS for styling and responsiveness, suitable for `QWebEngineView`.
        - `table` (QTableWidget): The `QTableWidget` instance to convert.
        - `str`: An HTML string representing the table.
        """
        css: str = """
        <style>
            body {
                margin: 0;
                padding: 10px;
                font-family: Arial, sans-serif;
            }
            .table-container {
                width: max-content;
                max-width: 100%;
                margin: 0 auto;
                overflow: hidden;
            }
            table {
                border-collapse: collapse;
                width: auto;
                table-layout: auto;
            }
            th, td {
                padding: 6px 10px;
                border: 1px solid
                text-align: center;
                white-space: nowrap;
            }
            th {
                background-color:
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color:
            }
            .matrix-title {
                text-align: center;
                font-size: 14pt;
                margin: 10px 0;
                font-weight: bold;
            }
        </style>
        """
        rows: int = table.rowCount()
        cols: int = table.columnCount()
        html: str = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Stiffness Matrix</title>
            {css}
        </head>
        <body>
            <div class="matrix-title">Stiffness Matrix</div>
            <div class="table-container">
                <table>
                    <thead><tr>
        """

        for col in range(cols):
            header: QTableWidgetItem | None = table.horizontalHeaderItem(col)
            header_text: str = header.text() if header else f"Col {col+1}"
            html += f"<th>{header_text}</th>"
        html += "</tr></thead><tbody>"

        for row in range(rows):
            html += "<tr>"

            for col in range(cols):
                item: QTableWidgetItem | None = table.item(row, col)
                text: str = item.text() if item else ""
                html += f"<td>{text}</td>"
            html += "</tr>"
        html += """
                </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        return html

    def qtable_to_html(qtable: QTableWidget) -> str:
        """
        Static method to convert a `QTableWidget`'s content into a basic HTML table string.
        This is a simpler version compared to `_convert_table_to_html` and might be for internal use.
        - `qtable` (QTableWidget): The `QTableWidget` instance to convert.
        - `str`: A basic HTML string representation of the table.
        """
        model = qtable.model()
        html: str = """<html>
        <head>
        <style>
        table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
        }
        </style>
        </head>"""
        html += "<table><thead>"
        html += "<tr>"

        for c in range(model.columnCount()):
            html += "<th>{}</th>".format(model.headerData(c, Qt.Horizontal))
        html += "</tr></thead>"
        html += "<tbody>"

        for r in range(model.rowCount()):
            html += "<tr>"

            for c in range(model.columnCount()):
                html += "<td>{}</td>".format(model.index(r, c).data() or "")
            html += "</tr>"
        html += "</tbody></table>"
        return html

    def add_stiffness_matrix_table(self, pdf_canvas: typing.Any) -> None:
        """
        Adds the stiffness matrix table (from `self.stiffness_table`) to a ReportLab PDF canvas.
        This method formats the table data and applies ReportLab styling for PDF rendering.
        - `pdf_canvas` (typing.Any): The ReportLab PDF canvas object to draw on.
        """
        num_rows: int = self.stiffness_table.rowCount()
        num_cols: int = self.stiffness_table.columnCount()
        data: list[list[str]] = [[''] * num_cols for _ in range(num_rows + 1)]

        for col in range(num_cols):
            header_item: QTableWidgetItem | None = self.stiffness_table.horizontalHeaderItem(col)

            if header_item:
                data[0][col] = header_item.text()
            else:
                data[0][col] = f"Column {col + 1}"

        for row in range(num_rows):

            for col in range(num_cols):
                item: QTableWidgetItem | None = self.stiffness_table.item(row, col)

                if item:
                    data[row + 1][col] = item.text()
                else:
                    data[row + 1][col] = ""
        table: Table = Table(data)
        style: TableStyle = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ])
        table.setStyle(style)
        available_width: float = A4[0] - 200
        available_height: float = A4[1] - 200
        table_width, table_height = table.wrap(available_width, available_height)
        x: float = (A4[0] - table_width) / 2
        y: float = available_height - table_height
        table.drawOn(pdf_canvas, x, y)

    # ---------------------------------------------
    # VIEW MANAGEMENT
    # ---------------------------------------------

    def restore_default_view(self) -> None:
        """
        Restores the application's dock layout to its default configuration.
        This includes showing and repositioning the left, right (if applicable),
        and bottom docks, and setting the central widget.
        """
        LeftDockWindow.boundary_dock.show()
        LeftDockWindow.details_dock.show()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, LeftDockWindow.boundary_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, LeftDockWindow.details_dock)
        self.bottom_dock.show()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)
        LeftDockWindow.boundary_dock.setFloating(False)
        LeftDockWindow.details_dock.setFloating(False)                
        self.bottom_dock.setFloating(False)
        print("Views reset to default.")

    # ---------------------------------------------
    # DOCK CREATION
    # ---------------------------------------------

    def create_central_dock(self) -> None:
        """
        Creates and sets the central dock widget of the `QMainWindow`.
        This dock typically houses the primary content or view of the application.
        """
        self.central_dock_window = CentralDockWindow(imported_data=self.imported_data, parent=self)
        self.setCentralWidget(self.central_dock_window)


    def create_left_dock(self) -> None:
        """
        Creates and adds the left dock widgets (`boundary_dock` and `details_dock`)
        to the left dock area of the `QMainWindow`.
        Warning:
            If `self.imported_data` is `None`, a warning will be printed as
            `LeftDockWindow` might depend on this data.
        """

        if self.imported_data is None:
            print("Warning: imported_data is None. LeftDockWindow may not function properly.")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, LeftDockWindow.boundary_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, LeftDockWindow.details_dock)


    def create_right_dock(self) -> None:
        """
        Creates the right dock widget, designated as the "Model Inspector".
        This dock displays various information about the model, including
        node information, element information, boundary conditions, mesh statistics,
        and solver status.
        """
        self.right_dock = QDockWidget("Model Inspector", self)
        self.right_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        tree_widget = QTreeWidget()
        tree_widget.setHeaderHidden(True)
        node_info_item = QTreeWidgetItem(tree_widget, ["Node Information"])
        selected_node_label: QLabel = QLabel("Selected Node: None")
        node_coordinates_label: QLabel = QLabel("Coordinates: (0, 0, 0)")
        element_info_item = QTreeWidgetItem(tree_widget, ["Element Information"])
        selected_element_label: QLabel = QLabel("Selected Element: None")
        element_type_label: QLabel = QLabel("Type: -")
        bc_info_item = QTreeWidgetItem(tree_widget, ["Boundary Conditions"])
        applied_loads_label: QLabel = QLabel("Applied Loads: 0")
        applied_constraints_label: QLabel = QLabel("Applied Constraints: 0")
        mesh_stats_item = QTreeWidgetItem(tree_widget, ["Mesh Statistics"])
        num_nodes_label: QLabel = QLabel("Nodes: 0")
        num_elements_label: QLabel = QLabel("Elements: 0")
        solver_status_item = QTreeWidgetItem(tree_widget, ["Solution Status"])
        solver_progress_label: QLabel = QLabel("Progress: Not Started")
        computation_time_label: QLabel = QLabel("Computation Time: 0s")
        inspector_widget = QWidget()
        inspector_layout = QVBoxLayout(inspector_widget)
        inspector_layout.addWidget(selected_node_label)
        inspector_layout.addWidget(node_coordinates_label)
        inspector_layout.addWidget(selected_element_label)
        inspector_layout.addWidget(element_type_label)
        inspector_layout.addWidget(applied_loads_label)
        inspector_layout.addWidget(applied_constraints_label)
        inspector_layout.addWidget(num_nodes_label)
        inspector_layout.addWidget(num_elements_label)
        inspector_layout.addWidget(solver_progress_label)
        inspector_layout.addWidget(computation_time_label)
        inspector_layout.addStretch()
        inspector_widget.setLayout(inspector_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(tree_widget)
        scroll_area.setWidgetResizable(True)
        self.right_dock.setWidget(scroll_area)
        self.right_dock.setWidget(inspector_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)


    def create_bottom_dock(self) -> None:
        """
        Creates the bottom dock widget, designated as the "Logging Console".
        This dock provides a read-only area for displaying application messages
        and logs.
        """

        def clear_bottom_dock_content():
            """
            Clears the content of the QTextEdit inside the bottom dock.
            """
            self.logging_console.clear()
            print("### Logging Console content cleared. ###\n")

        def show_bottom_dock_context_menu(pos: QPoint):
            """
            Gets the default context menu from QTextEdit, adds a 'Clear Content' action,
            and then displays it.
            """
            default_menu = self.logging_console.createStandardContextMenu()
            default_menu.addSeparator()
            clear_action = default_menu.addAction("Clear Console")
            clear_action.triggered.connect(clear_bottom_dock_content)
            default_menu.exec(self.logging_console.mapToGlobal(pos))
        self.bottom_dock = QDockWidget("Logging Console", self)
        self.bottom_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.logging_console = QTextEdit(self)
        self.logging_console.setReadOnly(True)
        self.logging_console.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.logging_console.customContextMenuRequested.connect(show_bottom_dock_context_menu)
        dock_widget = QWidget()
        layout = QVBoxLayout(dock_widget)
        layout.addWidget(self.logging_console)
        layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_dock.setWidget(dock_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)
        self.resizeDocks([self.bottom_dock], [100], Qt.Orientation.Vertical)
 
    # ---------------------------------------------
    # DOCK BEHAVIOR
    # ---------------------------------------------

    def toggle_dock(self) -> None:
        """
        Toggles the floating state of the bottom dock. If the dock is currently
        floating, it will be re-docked. If it's docked, it will be set to
        a floating state with a maximum height.
        """

        if self.bottom_dock.isFloating():
            self.bottom_dock.setFloating(False)
        else:
            self.bottom_dock.setFloating(True)
            self.bottom_dock.setMaximumHeight(500)


    def toggle_minimize_maximize(self) -> None:
        """
        Toggles the minimized/maximized state of the bottom dock.
        When minimized, the dock's height is reduced to its title bar height.
        When maximized, it restores its previous height.
        """

        if self.is_minimized:
            self.bottom_dock.setMaximumHeight(400)
            QTimer.singleShot(0, lambda: self.resizeDocks([self.bottom_dock], [self.previous_size], Qt.Orientation.Vertical))
            self.is_minimized = False
        else:
            self.previous_size = self.bottom_dock.height()
            title_bar_height: int = self.bottom_dock.titleBarWidget().height()
            self.bottom_dock.setMaximumHeight(25)
            self.is_minimized = True

    # ---------------------------------------------
    # WIDGET CREATION
    # ---------------------------------------------

    def create_materials_widget(self) -> QWidget:
        """
        Creates and returns a `QWidget` designed for selecting and editing material properties.
        Material data is loaded from and saved to a JSON file (`../data/material_library.json`).
        The widget includes a dropdown for material selection, input fields for properties,
        and a save button.
        Returns:
            QWidget: The widget containing the material selection and editing interface.
        """
        material_file: str = "../data/material_library.json"
        materials_data: dict = {}

        if os.path.exists(material_file):
            with open(material_file, "r") as file:
                materials_data = json.load(file)
        else:
            materials_data = {}
        materials_widget = QWidget()
        materials_layout = QVBoxLayout(materials_widget)
        material_dropdown = QComboBox()
        material_dropdown.addItem("➕ Add New Material")
        material_dropdown.addItems(materials_data.keys())
        materials_layout.addWidget(material_dropdown)
        properties_form = QFormLayout()
        property_fields: dict[str, QLineEdit] = {}
        new_material_name = QLineEdit()
        properties_form.addRow("Material Name:", new_material_name)


        def update_material_properties(index: int) -> None:
            """
            Updates the displayed material properties in the form when the
            material selection in the dropdown changes.
            Args:
                index (int): The index of the currently selected item in the dropdown.
            """
            selected_material: str = material_dropdown.currentText()

            for key in list(property_fields.keys()):
                properties_form.removeRow(property_fields[key])
            property_fields.clear()

            if selected_material == "➕ Add New Material":
                material_data = materials_data.get(self.previous_material, {}) if self.previous_material else {}
                new_material_name.setText("Name")
                new_material_name.setEnabled(True)
            else:
                material_data = materials_data.get(selected_material, {})
                new_material_name.setText(selected_material)
                new_material_name.setEnabled(False)

            if selected_material != "➕ Add New Material":
                self.previous_material = selected_material

            for key, value in material_data.items():

                if key == "Unit":
                    continue
                unit: str = material_data.get("Unit", {}).get(key, "")
                field = QLineEdit(str(value))
                property_fields[key] = field
                properties_form.addRow(f"{key} ({unit}):", field)
        save_button = QPushButton("Save Material")


        def save_material() -> None:
            """
            Saves or updates the material properties to the JSON file.
            If the material name is new, it will be added; otherwise, existing
            properties will be updated.
            """
            name: str = new_material_name.text().strip()

            if not name:
                QMessageBox.warning(None, "Error", "Material name cannot be empty.")
                return

            material_data: dict[str, str] = {key: field.text().strip() for key, field in property_fields.items()}

            if name in materials_data:
                material_data["Unit"] = materials_data[name].get("Unit", {})
            elif self.previous_material and self.previous_material in materials_data:
                material_data["Unit"] = materials_data[self.previous_material].get("Unit", {})
            materials_data[name] = material_data
            with open(material_file, "w") as file:
                json.dump(materials_data, file, indent=4)

            if material_dropdown.findText(name) == -1:
                material_dropdown.addItem(name)
            material_dropdown.setCurrentText(name)
            QMessageBox.information(None, "Success", "Material saved successfully.")
        save_button.clicked.connect(save_material)
        materials_layout.addLayout(properties_form)
        materials_layout.addWidget(save_button)
        materials_layout.addStretch()
        material_dropdown.currentIndexChanged.connect(update_material_properties)
        material_dropdown.currentTextChanged.connect(update_material_properties)

        if material_dropdown.count() > 1:
            material_dropdown.setCurrentIndex(1)
            update_material_properties(1)
        else:
            update_material_properties(0)
        return materials_widget

    # ---------------------------------------------
    # STATIC METHODS
    # ---------------------------------------------

    @staticmethod
    def populate_boundary_conditions(tree_widget: QTreeWidget, data: dict) -> QTreeWidget:
        """
        Static Method: Populates a PyQt6 `QTreeWidget` with boundary conditions (loads and constraints)
        extracted from a provided data dictionary. The boundary conditions are categorized
        into loads (concentrated and distributed) and constraints (fixed, pinned, roller).
        Args:
            tree_widget (QTreeWidget): The tree widget to populate.
            data (dict): A dictionary containing boundary condition data, expected to have keys
                         like "concentrated_loads", "distributed_loads", and "nodal_displacements".
        Returns:
            QTreeWidget: The populated tree widget.
        """
        boundary_conditions_item = QTreeWidgetItem(tree_widget, ["Boundary Conditions"])
        boundary_conditions_loads_item = QTreeWidgetItem(boundary_conditions_item, ["Loads"])
        concentrated_loads_item = QTreeWidgetItem(boundary_conditions_loads_item, ["Concentrated Loads"])

        for node, forces in data.get("concentrated_loads", {}).items():
            force_text: str = ", ".join([f"{v:.2f}" if not np.isnan(v) else "Free" for v in forces])
            QTreeWidgetItem(concentrated_loads_item, [f"Node {node}: ({force_text})"])
        distributed_loads_item = QTreeWidgetItem(boundary_conditions_loads_item, ["Distributed Loads"])

        for element, load in data.get("distributed_loads", {}).items():
            load_type: str = load["type"]
            direction: str = load["direction"]
            params: str = ", ".join(map(str, load["parameters"])) if isinstance(load["parameters"], (list, tuple)) else str(load["parameters"])
            QTreeWidgetItem(distributed_loads_item, [f"Element {element}: {load_type}, {direction}, ({params})"])
        boundary_conditions_constraints_item = QTreeWidgetItem(boundary_conditions_item, ["Constraints"])
        fixed_constraints_item = QTreeWidgetItem(boundary_conditions_constraints_item, ["Fixed"])
        pinned_constraints_item = QTreeWidgetItem(boundary_conditions_constraints_item, ["Pinned"])
        roller_constraints_item = QTreeWidgetItem(boundary_conditions_constraints_item, ["Roller"])

        for node, displacements in data.get("nodal_displacements", {}).items():
            disp_text: str = ", ".join([f"{v:.2f}" if not np.isnan(v) else "Free" for v in displacements])

            if all(not np.isnan(d) for d in displacements):
                QTreeWidgetItem(fixed_constraints_item, [f"Node {node}: ({disp_text})"])
            elif list(displacements).count(np.nan) == 1:
                QTreeWidgetItem(pinned_constraints_item, [f"Node {node}: ({disp_text})"])
            else:
                QTreeWidgetItem(roller_constraints_item, [f"Node {node}: ({disp_text})"])
        tree_widget.expandAll()
        return tree_widget

class BoundaryConditionEditor(QWidget):
    """
    The `BoundaryConditionEditor` class provides a widget for editing and managing
    various types of boundary conditions in a structural analysis context. It
    features dedicated tables for concentrated loads, distributed loads, and
    nodal constraints, allowing users to input and modify these conditions.
    The class supports different structure types (e.g., "3D_Frame") to adapt
    to varying degrees of freedom and force/displacement labels.
    """


    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the BoundaryConditionEditor widget.
        Args:
            parent (Optional[QWidget]): The parent widget of this editor.
        """
        super().__init__(parent)
        self.structure_types: Dict[str, Dict] = {}
        self.structure_type: str = ""
        self.force_labels: List[str] = []
        self.concentrated_table: Optional[QTableWidget] = None
        self.distributed_table: Optional[QTableWidget] = None
        self.constraints_table: Optional[QTableWidget] = None
 
    # ---------------------------------------------
    # LAYOUT AND TABLE CREATION
    # ---------------------------------------------

    def get_bounday_conditions_layout(self, data: Dict, structure_type: str = "3D_Frame") -> QVBoxLayout:
        """
        Configures and returns the main layout for the boundary conditions editor.
        This includes setting up the structure type, force labels, and initializing
        the tables for concentrated loads, distributed loads, and constraints.
        Args:
            data (Dict): A dictionary containing existing boundary condition data
                         to pre-populate the tables.
            structure_type (str): The type of structure (e.g., "3D_Frame") which
                                  determines the degrees of freedom and labels.
        Returns:
            QVBoxLayout: The vertical layout containing all boundary condition tables.
        """
        self.structure_types = {
            "3D_Frame": {
                "dofs_per_node": 6,
                "force_labels": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
                "displacement_labels": ["u", "v", "w", "ɸx", "ɸy", "ɸz"],
            }
        }
        self.structure_type = structure_type
        self.force_labels = self.structure_types[structure_type]["force_labels"]
        layout = QVBoxLayout()
        self.concentrated_table = self.create_table("Concentrated Loads", ["Node#"] + self.force_labels + ["Actions"])
        layout.addWidget(self.concentrated_table)
        self.distributed_table = self.create_table("Distributed Loads", ["Element#", "Type"] + self.force_labels + ["Actions"], dropdown_col=1)
        layout.addWidget(self.distributed_table)
        self.constraints_table = self.create_table("Constraints", ["Node#", "Type"] + self.force_labels + ["Actions"], dropdown_col=1)
        layout.addWidget(self.constraints_table)
        self.setLayout(layout)
        self.populate_boundary_conditions(self.concentrated_table, self.distributed_table, self.constraints_table, data, self.structure_type)
        return layout

    def create_table(self, title: str, headers: List[str], dropdown_col: Optional[int] = None) -> QTableWidget:
        """
        Creates and configures an editable `QTableWidget` with the specified title and headers.
        Optionally adds dropdown combo boxes to a specified column for certain table types.
        Args:
            title (str): The title of the table (e.g., "Concentrated Loads"). Used for logic
                         to determine dropdown options.
            headers (List[str]): A list of strings to be used as column headers for the table.
            dropdown_col (Optional[int]): The column index where a `QComboBox` should be added
                                          to each row, if applicable. Defaults to `None`.
        Returns:
            QTableWidget: The newly created and configured table widget.
        """
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(5)

        if dropdown_col is not None:

            for row in range(table.rowCount()):
                combo = QComboBox()

                if title == "Distributed Loads":
                    combo.addItems(["Uniform", "Triangular", "Trapezoidal", "Equation"])
                elif title == "Constraints":
                    combo.addItems(["Fixed", "Pinned", "Roller"])
                table.setCellWidget(row, dropdown_col, combo)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        return table

    # ---------------------------------------------
    # POPULATION AND ROW MANAGEMENT
    # ---------------------------------------------

    @staticmethod
    def populate_boundary_conditions(
        concentrated_table: QTableWidget,
        distributed_table: QTableWidget,
        constraints_table: QTableWidget,
        data: Dict,
        structure_type: str = "3D_Frame"
    ) -> Tuple[QTableWidget, QTableWidget, QTableWidget]:
        """
        Static Method: Populates the provided `QTableWidget` instances with boundary condition data
        (concentrated loads, distributed loads, and nodal displacements). It adapts the
        table population based on the specified `structure_type` and includes buttons for row removal.
        Args:
            concentrated_table (QTableWidget): The table widget for concentrated loads.
            distributed_table (QTableWidget): The table widget for distributed loads.
            constraints_table (QTableWidget): The table widget for nodal constraints.
            data (Dict): A dictionary containing the boundary condition data. Expected keys are
                         "concentrated_loads", "distributed_loads", and "nodal_displacements".
            structure_type (str): The type of structure, used to determine force/displacement labels.
        Returns:
            Tuple[QTableWidget, QTableWidget, QTableWidget]: The three populated table widgets.
        """
        structure_types: Dict[str, Dict] = {
            "3D_Frame": {
                "dofs_per_node": 6,
                "force_labels": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
                "displacement_labels": ["u", "v", "w", "ɸx", "ɸy", "ɸz"],
            }
        }
        concentrated_loads_data: Dict[str, List[float]] = data.get("concentrated_loads", {})
        concentrated_table.setRowCount(len(concentrated_loads_data))

        for row, (node, forces) in enumerate(concentrated_loads_data.items()):
            concentrated_table.setItem(row, 0, QTableWidgetItem(str(node)))

            for col, value in enumerate(forces):
                item = QTableWidgetItem(f"{value:.2f}" if not np.isnan(value) else "Free")
                concentrated_table.setItem(row, col + 1, item)
            remove_button = QPushButton("✖")
            remove_button.clicked.connect(lambda checked, r=row: BoundaryConditionEditor.remove_row(concentrated_table, r))
            concentrated_table.setCellWidget(row, concentrated_table.columnCount() - 1, remove_button)
        distributed_loads_data: Dict[str, Dict] = data.get("distributed_loads", {})
        distributed_table.setRowCount(len(distributed_loads_data))

        for row, (element, load) in enumerate(distributed_loads_data.items()):
            distributed_table.setItem(row, 0, QTableWidgetItem(str(element)))
            combo = QComboBox()
            combo.addItems(["Uniform", "Triangular", "Trapezoidal", "Equation"])
            combo.setCurrentText(load.get("type", "Uniform"))
            distributed_table.setCellWidget(row, 1, combo)
            parameters: List[float] = load.get("parameters", [])

            if not isinstance(parameters, (list, tuple)):
                parameters = [parameters]

            for col, value in enumerate(parameters):
                distributed_table.setItem(row, col + 2, QTableWidgetItem(f"{value}"))
            remove_button = QPushButton("✖")
            remove_button.clicked.connect(lambda checked, r=row: BoundaryConditionEditor.remove_row(distributed_table, r))
            distributed_table.setCellWidget(row, distributed_table.columnCount() - 1, remove_button)
        nodal_displacements_data: Dict[str, List[float]] = data.get("nodal_displacements", {})
        constraints_table.setRowCount(len(nodal_displacements_data))

        for row, (node, displacements) in enumerate(nodal_displacements_data.items()):
            constraints_table.setItem(row, 0, QTableWidgetItem(str(node)))
            combo = QComboBox()
            num_nan = sum(1 for val in displacements if np.isnan(val))
            constraint_type: str

            if num_nan == 0:
                constraint_type = "Fixed"
            elif num_nan == 1:
                constraint_type = "Pinned"
            else:
                constraint_type = "Roller"
            combo.addItems(["Fixed", "Pinned", "Roller"])
            combo.setCurrentText(constraint_type)
            constraints_table.setCellWidget(row, 1, combo)

            for col, value in enumerate(displacements):
                constraints_table.setItem(row, col + 2, QTableWidgetItem(f"{value:.2f}" if not np.isnan(value) else "Free"))
            remove_button = QPushButton("✖")
            remove_button.clicked.connect(lambda checked, r=row: BoundaryConditionEditor.remove_row(constraints_table, r))
            constraints_table.setCellWidget(row, constraints_table.columnCount() - 1, remove_button)
        BoundaryConditionEditor.auto_resize_columns(concentrated_table)
        BoundaryConditionEditor.auto_resize_columns(distributed_table)
        BoundaryConditionEditor.auto_resize_columns(constraints_table)
        return concentrated_table, distributed_table, constraints_table


    @staticmethod
    def auto_resize_columns(table: QTableWidget) -> None:
        """
        Static Method: Adjusts the columns and rows of a `QTableWidget` to fit their content.
        Args:
            table (QTableWidget): The table widget whose columns and rows need resizing.
        """
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

    @staticmethod
    def add_row(table: QTableWidget) -> None:
        """
        Static Method: Dynamically adds a new empty row to the given `QTableWidget`.
        A "✖" button is added to the last column of the new row to allow for removal.
        Args:
            table (QTableWidget): The table widget to which a new row will be added.
        """
        row_count: int = table.rowCount()
        table.insertRow(row_count)
        remove_button = QPushButton("✖")
        remove_button.clicked.connect(lambda checked, r=row_count: BoundaryConditionEditor.remove_row(table, r))
        table.setCellWidget(row_count, table.columnCount() - 1, remove_button)

    @staticmethod
    def remove_row(table: QTableWidget, row: int) -> None:
        """
        Static Method: Removes a specified row from the given `QTableWidget`.
        Args:
            table (QTableWidget): The table widget from which the row will be removed.
            row (int): The index of the row to be removed.
        """
        table.removeRow(row)


class MaterialWidget(QWidget):
    """
    The `MaterialWidget` class provides a user interface for managing a material library.
    It allows users to select existing materials, view and edit their properties,
    add new materials, and save changes to a JSON file.
    """


    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the MaterialWidget.
        Args:
            parent (Optional[QWidget]): The parent widget of this editor.
        """
        super().__init__(parent)
        self.material_file: str = "../data/material_library.json"
        self.materials_data: Dict[str, Any] = {}
        self.material_dropdown: Optional[QComboBox] = None
        self.properties_form: Optional[QFormLayout] = None
        self.property_fields: Dict[str, QLineEdit] = {}
        self.new_material_name: Optional[QLineEdit] = None
        self.save_button: Optional[QPushButton] = None
 
    # ---------------------------------------------
    # WIDGET SETUP
    # ---------------------------------------------

    def create_materials_widget(self) -> QWidget:
        """
        Creates and configures the main widget for material management.
        This includes the material selection dropdown, a form for displaying
        and editing material properties, and a save button.
        Returns:
            QWidget: The fully configured materials widget.
        """
        self.load_materials()
        materials_widget = QWidget()
        materials_widget.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        materials_layout = QVBoxLayout(materials_widget)
        self.material_dropdown = QComboBox()
        self.material_dropdown.addItem("➕ Add New Material")
        self.material_dropdown.addItems(self.materials_data.keys())
        materials_layout.addWidget(self.material_dropdown)
        self.properties_form = QFormLayout()
        self.property_fields = {}
        self.new_material_name = QLineEdit()
        self.properties_form.addRow("Material Name:", self.new_material_name)
        materials_layout.addLayout(self.properties_form)
        self.save_button = QPushButton("Save Material")
        self.save_button.clicked.connect(self.save_material)
        materials_layout.addWidget(self.save_button)
        self.material_dropdown.currentIndexChanged.connect(self.update_material_properties)
        self.material_dropdown.currentTextChanged.connect(self.update_material_properties)
        self.update_material_properties(0)
        self.properties_form.update()
        materials_widget.update()
        return materials_widget

    # ---------------------------------------------
    # DATA HANDLING
    # ---------------------------------------------

    def load_materials(self) -> None:
        """
        Loads material data from the specified JSON file (`self.material_file`).
        If the file does not exist, an empty dictionary is initialized for materials data.
        """

        if os.path.exists(self.material_file):
            with open(self.material_file, "r") as file:
                self.materials_data = json.load(file)
        else:
            self.materials_data = {}


    def update_material_properties(self, index: int) -> None:
        """
        Updates the displayed material properties in the form based on the
        currently selected item in the material dropdown. Handles both
        existing material selection and the "Add New Material" option.
        Args:
            index (int): The index of the selected item in the dropdown. This
                         parameter is provided by the `currentIndexChanged` signal,
                         but its value is not directly used for logic as `currentText()`
                         is preferred for clarity.
        """
        selected_material: str = self.material_dropdown.currentText()

        for key in list(self.property_fields.keys()):
            self.properties_form.removeRow(self.property_fields[key])
        self.property_fields.clear()

        if selected_material == "➕ Add New Material":
            self.new_material_name.setText("")
            self.new_material_name.setEnabled(True)
            material_data: Dict[str, Any] = {}
        else:
            self.new_material_name.setText(selected_material)
            self.new_material_name.setEnabled(False)
            material_data = self.materials_data.get(selected_material, {})

        for key, value in material_data.items():

            if key == "Unit":
                continue
            unit: str = material_data.get("Unit", {}).get(key, "")
            field = QLineEdit(str(value))
            self.property_fields[key] = field
            self.properties_form.addRow(f"{key} ({unit}):", field)


    def save_material(self) -> None:
        """
        Saves the current material's properties to the JSON file.
        This function handles both adding new materials and updating existing ones.
        It performs basic validation to ensure the material name is not empty.
        """
        name: str = self.new_material_name.text().strip()

        if not name:
            QMessageBox.warning(self, "Error", "Material name cannot be empty.")
            return

        material_data: Dict[str, str] = {key: field.text().strip() for key, field in self.property_fields.items()}

        if name in self.materials_data and "Unit" in self.materials_data[name]:
            material_data["Unit"] = self.materials_data[name]["Unit"]
        self.materials_data[name] = material_data
        with open(self.material_file, "w") as file:
            json.dump(self.materials_data, file, indent=4)

        if self.material_dropdown.findText(name) == -1:
            self.material_dropdown.addItem(name)
        self.material_dropdown.setCurrentText(name)
        QMessageBox.information(self, "Success", "Material saved successfully.")

    # ---------------------------------------------
    # EVENT HANDLING
    # ---------------------------------------------

    def closeEvent(self, event: Any) -> None:
        """
        Overrides the `closeEvent` method to restore standard output/error
        streams before the widget closes. This is crucial if `sys.stdout`
        or `sys.stderr` have been redirected within the application.
        Args:
            event (Any): The close event object.
        """
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        super().closeEvent(event)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = SetupDocks()
    window.show()
    sys.exit(app.exec())