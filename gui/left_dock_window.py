from PyQt6.QtWidgets import (
    QDockWidget, QWidget,
    QVBoxLayout, QLabel, QMainWindow,
    QTreeWidget, QTreeWidgetItem, QLineEdit, QPushButton,
    QScrollArea, QComboBox, QHBoxLayout,
    QSizePolicy, QMessageBox,
    QCheckBox, QDoubleSpinBox
)
from PyQt6.QtGui import QFont, QColor, QDoubleValidator, QIntValidator
from PyQt6.QtCore import Qt, QSignalBlocker
from gui.units_handling import UnitsHandling
from gui.tree_widget import TreeWidget
from utils.utils_classes import NaNValidator
import json, gc
import numpy as np
import re
import copy
from typing import Dict, Any, Optional, Tuple, List


class LeftDockWindow(QWidget):
    """
    LeftDockWindow is a dockable panel that displays and manages project data,
    such as nodes, elements, loads, and material properties within the GUI.
    It provides a tree-based navigation panel on the left side of the application,
    allowing users to select and interact with individual components of their
    finite element model. It connects the UI elements to the underlying data
    structure and enables editing, selection, and postprocessing views.
    Attributes:
        boundary_dock (QDockWidget): Upper left dock for tree widget display.
        details_dock (QDockWidget): Lower left dock for showing item-specific details.
        tree_widget (TreeWidgetWithContextMenu): Tree view of model hierarchy.
        imported_data (dict): The model's imported configuration and definition data.
        central_dock (QWidget): Reference to the central dock widget.
        output_stream (io.TextIOWrapper): Output stream for logs or messages.
        solver_settings (dict): Dictionary storing solver configuration parameters.
        mesh_settings (dict): Dictionary storing mesh configuration parameters.
    Methods:
        __init__(...): Constructor to initialize the LeftDockWindow.
        create_left_dock_window(...): Sets up dock widgets and layout.
        create_details_tree_widget(): Initializes and styles the details panel.
        initialize_last_selected_cross_sections(): Remembers last-used cross-sections.
        initialize_last_selected_materials(): Remembers last-used materials.
        populate_tree(): Populates the tree widget from imported data.
        clear_tree_widget_items(): Clears detail tree content.
        clear_tree_widget_item(...): Recursively clears tree widget items.
        display_selected_item_info(...): Displays item details based on selection.
        select_tab_by_label(...): Activates tab in central dock by name.
        display_material_info(...): Displays editable material information.
        update_dictionary_key_ordered(...): Renames a dictionary key preserving order.
        display_solver_info(): Displays solver configuration options.
        display_mesh_info(): Displays mesh-related options.
        display_postprocessing_info(): Displays postprocessing visualization options.
        display_node_info(...): Displays node data and constraints.
        get_constraint_type(...): Returns string description of constraint type.
        update_imported_data(): Updates displacement/load info in imported_data.
        add_tree_item(...): Adds an entry to the QTreeWidget.
        find_child_item(...): Helper to find child item by text.
        remove_tree_item(...): Removes an item from the QTreeWidget.
        display_element_info(...): Displays details about an element.
        display_distributed_load_info(...): Displays distributed load information.
    """

    # ---------------------------------------------
    # CLASS-WIDE STATIC VARIABLES
    # ---------------------------------------------

    boundary_dock: Optional[QDockWidget] = None
    details_dock: Optional[QDockWidget] = None
    tree_widget: Optional[QTreeWidget] = None

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self, imported_data: Dict[str, Any], central_dock: QWidget, output_stream: Any, parent: Optional[QWidget] = None) -> None:
        """
        Initializes the LeftDockWindow.
        Args:
            imported_data (Dict[str, Any]): A dictionary containing all the
                imported structural data (nodes, elements, materials, etc.).
            central_dock (QWidget): A reference to the central dock widget,
                used for interacting with other parts of the GUI, such as
                switching tabs in the results viewer.
            output_stream (Any): An object or function used for logging
                or displaying output messages.
            parent (Optional[QWidget]): The parent widget of this dock window.
                Defaults to None.
        """
        super().__init__(parent)
        self.imported_data: Dict[str, Any] = imported_data
        self.imported_data_bk: Dict[str, Any] = copy.deepcopy(imported_data)
        self.central_dock: QWidget = central_dock
        self.output_stream: Any = output_stream
        self.solver_settings: Dict[str, Any] = {}
        self.mesh_settings: Dict[str, Any] = {}
        self.last_selected_cross_sections: Dict[str, str] = {}
        self.last_selected_materials: Dict[str, str] = {}
        self.setup_docks: QMainWindow = parent
        self.create_left_dock_window(parent)

    # ---------------------------------------------
    # WINDOW CREATION & INITIALIZATION
    # ---------------------------------------------

    def create_left_dock_window(self, parent: Optional[QWidget]) -> None:
        """
        Creates and sets up the layout and widgets for the left dock window.
        This includes the main tree widget for navigation and the details
        panel for displaying item properties.
        Args:
            parent (Optional[QWidget]): The parent widget of this dock window.
        """
        self.temp_cross_sections: Dict[str, Any] = {}
        self.temp_materials: Dict[str, Any] = {}
        self.new_generated_codes: set[str] = set()

        # ---------------------------------------------
        # MAIN LAYOUT
        # ---------------------------------------------

        main_layout: QVBoxLayout = QVBoxLayout(self)

        # ---------------------------------------------
        # TREE WIDGET FOR NAVIGATION
        # ---------------------------------------------

        self.tree_widget: TreeWidget = TreeWidget(self.imported_data, self)
        self.tree_widget.setHeaderLabel("Geometry, Loads & Constraints")
        self.tree_widget.setIndentation(10)
        LeftDockWindow.tree_widget = self.tree_widget 

        # ---------------------------------------------
        # LEFT UPPER DOCK: BOUNDARY CONDITIONS & LOADS
        # ---------------------------------------------

        self.boundary_dock = QDockWidget("Imported Geometry", parent)
        self.boundary_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.boundary_dock.setWidget(self.tree_widget)

        # ---------------------------------------------
        # LEFT LOWER DOCK: ELEMENT AND NODE DETAILS
        # ---------------------------------------------

        self.details_dock = QDockWidget("Item Details", parent)
        self.details_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.details_widget: QWidget = QWidget()
        self.details_layout: QVBoxLayout = QVBoxLayout(self.details_widget)
        self.details_dock.setWidget(self.details_widget)

        # ---------------------------------------------
        # INFORMATION DISPLAY (EDITABLE FIELDS)
        # ---------------------------------------------

        self.details_label: QLabel = QLabel("\nSelect an item to view details\n")
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_layout.addWidget(self.details_label)
        self.create_details_tree_widget()

        # ---------------------------------------------
        # POPULATE TREE AND ADD INTERACTIONS
        # ---------------------------------------------

        self.populate_tree()
        self.tree_widget.itemClicked.connect(self.display_selected_item_info)

        # ---------------------------------------------
        # ADD DOCKS TO MAIN LAYOUT
        # ---------------------------------------------

        main_layout.addWidget(self.boundary_dock)
        main_layout.addWidget(self.details_dock)
        LeftDockWindow.boundary_dock = self.boundary_dock  
        LeftDockWindow.details_dock = self.details_dock    
        self.initialize_last_selected_cross_sections()
        self.initialize_last_selected_materials() 


    def create_details_tree_widget(self) -> None:
        """
        Creates and configures the QTreeWidget used for displaying detailed
        information of selected items. This tree widget is placed inside a
        QScrollArea to ensure content is scrollable if it exceeds visible bounds.
        """
        self.details_tree: QTreeWidget = QTreeWidget()
        self.details_tree.setHeaderHidden(True)
        font: QFont = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setIndentation(5)  
        self.details_tree.setRootIsDecorated(True)
        self.details_tree.setIndentation(10)
        self.details_tree.setAlternatingRowColors(False)
        self.details_tree.setHeaderHidden(True)
        self.details_tree.setStyleSheet("""
            QTreeWidget {
                background-color: white;
                border: none;
                font-size: 10pt;
            }
            QTreeWidget::item {
                padding: 4px;
                color: black; /* Ensure default text color is black */
            }
            QTreeWidget::item:selected {
                background-color: #d0e3ff;
                color: black; /* Force text color to black on selection */
            }
            QTreeWidget::item:hover {
                background-color: #f0faff;
            }
        """)
        self.details_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area: QScrollArea = QScrollArea()
        self.scroll_area.setWidget(self.details_tree)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.details_layout.addWidget(self.scroll_area)
        self.details_layout.setContentsMargins(0, 0, 0, 0)

    # ---------------------------------------------
    # DATA INITIALIZATION METHODS
    # ---------------------------------------------

    def initialize_last_selected_cross_sections(self) -> None:
        """
        Initializes `last_selected_cross_sections` dictionary.
        If 'cross_sections' exist in `self.imported_data` and is a dictionary,
        it populates `last_selected_cross_sections` where keys and values are
        the same as the cross-section codes. Otherwise, it initializes an empty dictionary.
        """

        if "cross_sections" in self.imported_data and isinstance(self.imported_data["cross_sections"], dict):
            self.last_selected_cross_sections = {key: key for key in self.imported_data["cross_sections"].keys()}
        else:
            self.last_selected_cross_sections = {}


    def initialize_last_selected_materials(self) -> None:
        """
        Initializes `last_selected_materials` dictionary.
        If 'materials' exist in `self.imported_data` and is a dictionary,
        it populates `last_selected_materials` where keys and values are
        the same as the material codes. Otherwise, it initializes an empty dictionary.
        """

        if "materials" in self.imported_data and isinstance(self.imported_data["materials"], dict):
            self.last_selected_materials = {key: key for key in self.imported_data["materials"].keys()}
        else:
            self.last_selected_materials = {}

    # ---------------------------------------------
    # ITEM DISPLAY METHODS
    # ---------------------------------------------

    def display_selected_item_info(self, item_id: int = 1, item_data_dict: Optional[Dict[str, Any]] = None, new_constraint: bool = False) -> None:
        """
        Displays detailed information for the selected item in the `details_tree`.
        This method acts as a dispatcher, calling specific display methods based
        on the type of the selected item.
        Args:
            item_id (int): The id of currently selected tree widget item.
            item_data_dict (Optional[Dict[str, Any]]): Optional dictionary
                containing item data. If None, data is extracted from the `item`.
                Defaults to None.
            new_constraint (bool): A flag indicating if a new constraint is being
                added. Defaults to False.
        """
        selected_items: list[QTreeWidgetItem] = self.tree_widget.selectedItems()

        if not selected_items:
            self.clear_tree_widget_items()
            return

        item_data: Optional[Dict[str, Any]] = selected_items[0].data(0, Qt.ItemDataRole.UserRole)

        if not item_data:
            self.clear_tree_widget_items()
            return

        item_type: str = item_data["type"]
        item_id: Any = item_data.get("id", 0)

        if item_id == 0 and item_type in ["nodal_displacement", "concentrated_load"]:
            new_constraint = True
        self.clear_tree_widget_items()

        if item_type == "node":
            self.display_node_info(node_id=item_id, node_data=item_data_dict, show_node_id=False, enable_node_id=False, enable_constraint_type=True, new_constraint=new_constraint)
            self.details_label.setHidden(True)
        elif item_type == "concentrated_load":
            self.display_node_info(node_id=item_id, node_data=item_data_dict, show_node_id=True, enable_constraint_type=True, new_constraint=new_constraint)
            self.details_label.setHidden(True)
        elif item_type == "nodal_displacement":
            self.display_node_info(node_id=item_id, node_data=item_data_dict, show_node_id=True, new_constraint=new_constraint)
            self.details_label.setHidden(True)
        elif item_type == "element":
            elem_id: Any = item_data["id"]
            self.display_element_info(elem_id)
            self.details_label.setHidden(True)
        elif item_type == "material":
            material_code: Optional[str] = item_data["code"]
            self.display_material_info(material_code)
            self.details_label.setHidden(True)
        elif item_type == "cross_section":
            section_code: Optional[str] = item_data["code"]
            self.display_cross_section_info(section_code)
            self.details_label.setHidden(True)
        elif item_type == "distributed_load":
            elem_id: Any = item_data["id"]
            self.display_distributed_load_info(elem_id)
            self.details_label.setHidden(True)
        elif item_type == "mesh":
            self.display_mesh_info()
            self.details_label.setHidden(True)
        elif item_type == "solver":
            self.display_solver_info()
            self.details_label.setHidden(True)
        elif item_type == "post_processing":
            self.display_postprocessing_info()
            self.select_tab_by_label(label=item_data["label"])
            self.details_label.setHidden(True)

    # ----------------------------------------------------------------------
    # NODE DISPLAY & HANDLING
    # ----------------------------------------------------------------------

    def display_node_info(self, node_id: int = 1, node_data: Optional[Dict[str, Any]] = None,
                          show_node_id: bool = False, enable_node_id: bool = True,
                          new_constraint: bool = False, show_constraint_type: bool = True,
                          enable_constraint_type: bool = True, previous_node_id: Optional[int] = None) -> None:
        """
        Displays detailed information for a given node in the `details_tree` QTreeWidget.
        This includes coordinates, forces, and displacements, and allows for their
        modification. It also provides options to change the node ID and constraint type.
        Args:
            node_id (int): The ID of the node to display information for. Defaults to 1.
            node_data (Optional[Dict[str, Any]]): A dictionary containing the node's data.
                                                    If None, it will be fetched from `self.imported_data`.
            show_node_id (bool): If True, a dropdown to change the node ID will be displayed.
            enable_node_id (bool): If True, the node ID dropdown will be enabled.
            new_constraint (bool): If True, indicates that a new constraint is being added.
            show_constraint_type (bool): If True, a dropdown to change the constraint type will be displayed.
            enable_constraint_type (bool): If True, the constraint type dropdown will be enabled.
            previous_node_id (Optional[int]): The ID of the previously selected node, used for updates.
        """
        self.details_tree.clear()
        self.details_tree.setColumnCount(3)
        self.details_tree.setHeaderLabels(["Property", "Value", "Unit"])
        # ---------------------------------------------
        # UTILITY: Widget Creators and Handlers
        # ---------------------------------------------

        def create_line_edit(value: Any, callback: Optional[callable] = None,
                             validator: Optional[NaNValidator] = None) -> QLineEdit:
            """
            Creates a QLineEdit with an optional validator and a callback function

            for `editingFinished` signal.
            Args:
                value (Any): The initial value for the QLineEdit. Can be `np.nan`.
                callback (Optional[callable]): A function to be called when editing finishes.
                                               It will receive the parsed new value.
                validator (Optional[NaNValidator]): A validator to apply to the QLineEdit.
            Returns:
                QLineEdit: The configured QLineEdit widget.
            """

            if isinstance(value, float) and np.isnan(value):
                value_str = "nan"
            else:
                value_str = str(value)
            line_edit = QLineEdit(value_str)

            if validator:
                line_edit.setValidator(validator)

            if callback:
                line_edit.editingFinished.connect(lambda le=line_edit: handle_editing_finished(le, callback))
            return line_edit

        def handle_editing_finished(line_edit: QLineEdit, callback: callable) -> None:
            """
            Handles the `editingFinished` signal of a QLineEdit.
            Parses the text and calls the provided callback.
            Args:
                line_edit (QLineEdit): The QLineEdit that emitted the signal.
                callback (callable): The function to call with the parsed value.
            """
            text = line_edit.text()

            if text.lower() == "nan" or text.strip() == "":
                callback(np.nan)
            else:

                try:
                    float_value = float(text)
                    callback(float_value)

                except ValueError as e:
                    print(f"Invalid value entered: {e}")


        def wrap_widget(widget: QWidget) -> QWidget:
            """
            Wraps a given widget in a QHBoxLayout to remove extra margins
            and control its stretching behavior.
            Args:
                widget (QWidget): The widget to be wrapped.
            Returns:
                QWidget: A container widget holding the wrapped widget.
            """
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            layout.setStretch(0, 1)
            return container

        def update_and_refresh(node_id: int, bc_type: str) -> None:
            """
            Saves current values, updates the tree widget, hides the details label,
            and refreshes the node information display.
            Args:
                node_id (int): The ID of the node to refresh.
                bc_type (str): The type of boundary condition (e.g., "Nodal Displacements").
            """

            if self.tree_widget:
                self.tree_widget.update_tree(self.imported_data)
                self.details_label.setHidden(True)
                self.display_node_info(node_id=node_id, show_node_id=True)
                self.tree_widget.set_current_tree_item("Boundary Conditions", bc_type, f"Node {node_id}")


        def on_node_id_change(new_text: str) -> None:
            """
            Handles changes in the Node ID dropdown. If the node ID is changed,
            it attempts to transfer the existing boundary conditions or loads
            to the new node ID and updates the display.
            Args:
                new_text (str): The new node ID as a string.
            """
            selected_items = self.tree_widget.selectedItems() if self.tree_widget else []

            if not selected_items:
                return

            item_data = selected_items[0].data(0, Qt.ItemDataRole.UserRole)

            if not item_data:
                return

            load_type = item_data["type"]
            current_id = item_data.get("id", 0)
            is_new_constraint = (current_id == 0)

            if load_type == "nodal_displacement":
                load_key = "displacement"
                bc_category = "Nodal Displacements"
            elif load_type == "concentrated_load":
                load_key = "force"
                bc_category = "Nodal Forces"
            else:
                return

            try:
                new_id = int(float(new_text))
                nodes = self.imported_data['nodes']

                if new_id not in nodes:
                    QMessageBox.warning(self, "Invalid Node ID",
                                        f"Node ID {new_id} does not exist in the structure. Please create it first.")
                    self.details_tree.findChild(QComboBox).setCurrentText(str(current_id))
                    return

                if current_id != 0:

                    if load_key in nodes[current_id]:
                        nodes[new_id][load_key] = nodes[current_id][load_key]

                        if load_key == "displacement":
                            dofs_per_node = len(nodes[current_id][load_key])
                            nodes[current_id][load_key] = tuple(np.nan for _ in range(dofs_per_node))
                        else:
                            dofs_per_node = len(nodes[current_id][load_key])
                            nodes[current_id][load_key] = tuple(np.zeros(dofs_per_node))

                if is_new_constraint:

                    if load_key == "displacement" and load_key not in nodes[new_id]:
                        displacement = [np.nan] * dofs_per_node
                        displacement[0] = 0.0  
                        nodes[new_id][load_key] = tuple(displacement)
                        self.tree_widget.set_current_tree_item("Boundary Conditions", bc_category, f"Node {new_id}")
                    elif load_key == "force" and load_key not in nodes[new_id]:
                        force = [0.0] * dofs_per_node
                        force[0] = 1.0  
                        nodes[new_id][load_key] = tuple(force)
                    del nodes[0]
                self.imported_data['nodes'] = nodes
                self.update_imported_data()
                update_and_refresh(new_id, bc_category)

            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for Node ID.")

            except KeyError:
                QMessageBox.warning(self, "Node Not Found", f"Node ID {new_text} does not exist in the structure.")

        # ---------------------------------------------
        # COLORS AND FONTS
        # ---------------------------------------------

        color_heading = QColor("#e1ecf4")
        color_items = QColor("white")
        font = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setContentsMargins(5, 5, 5, 5)
        nodes = self.imported_data.get("nodes", {})
        units_dict = self.imported_data.get("saved_units", {})
        node_data = nodes.get(node_id)

        if not node_data:
            QTreeWidgetItem(self.details_tree, [f"Node {node_id} not found", "", ""])
            return


        # ---------------------------------------------
        # TOP-LEVEL ITEM: Node ID
        # ---------------------------------------------

        node_item = QTreeWidgetItem(self.details_tree, [f"Node {node_id}", "", ""])
        node_font = QFont("Arial", 12, QFont.Weight.Bold)
        node_item.setFont(0, node_font)
        node_item.setFirstColumnSpanned(True)
        node_item.setBackground(0, QColor("#e1ecf4"))
        self.details_tree.addTopLevelItem(node_item)

        # ---------------------------------------------
        # NODE ID DROPDOWN
        # ---------------------------------------------

        if show_node_id:
            nodes_item = QTreeWidgetItem(node_item, ["Node ID", "", ""])
            nodes_item.setBackground(0, color_items)
            nodes_item.setBackground(1, color_items)
            nodes_item.setBackground(2, color_items)
            nodes_combo = QComboBox()
            node_ids = [str(node_id) for node_id in self.imported_data["nodes"].keys()]
            nodes_combo.addItems(node_ids)
            current_node = str(node_id)
            nodes_combo.setCurrentText(current_node)
            nodes_combo.currentTextChanged.connect(on_node_id_change)
            nodes_combo.setEnabled(enable_node_id)
            self.details_tree.setItemWidget(nodes_item, 1, wrap_widget(nodes_combo))
            node_item.addChild(nodes_item)

        # ---------------------------------------------
        # CONSTRAINT TYPE DROPDOWN
        # ---------------------------------------------

        if show_constraint_type:
            constraint_item = QTreeWidgetItem(node_item, ["Constraint Type", "", ""])
            constraint_item.setBackground(0, color_items)
            constraint_item.setBackground(1, color_items)
            constraint_item.setBackground(2, color_items)
            constraints_combo = QComboBox()
            boundary_types = ["Fixed", "Pinned", "Roller(x)", "Roller(y)", "Roller(z)", "Free", "Custom"]
            constraints_combo.addItems(boundary_types)
            constraint = self.get_constraint_type(node_data, self.imported_data["structure_info"])
            constraints_combo.setCurrentText(constraint)


            def update_constraint_type(text: str) -> None:
                """
                Updates the displacement boundary conditions for the current node
                based on the selected constraint type.
                Args:
                    text (str): The selected constraint type (e.g., "Fixed", "Pinned").
                """
                structure_dim = self.imported_data["structure_info"].get("dimension", "3D")
                structure_elem_type = self.imported_data["structure_info"].get("element_type", "")
                num_dofs = self.imported_data['structure_info']['dofs_per_node']
                new_displacement = [np.nan] * num_dofs

                if text == "Fixed":
                    new_displacement = [0] * num_dofs
                elif text == "Pinned":

                    if structure_dim == "2D":
                        new_displacement[0:2] = [0, 0]

                        if num_dofs == 3 and structure_elem_type == "2D_Beam":
                            new_displacement[2] = np.nan
                    elif structure_dim == "3D":
                        new_displacement[0:3] = [0, 0, 0]
                        new_displacement[3:6] = [np.nan, np.nan, np.nan]
                elif "Roller" in text:

                    if text == "Roller(x)":
                        new_displacement[0] = 0
                    elif text == "Roller(y)":
                        new_displacement[1] = 0
                    elif text == "Roller(z)":
                        new_displacement[2] = 0
                elif text == "Free":
                    new_displacement = [np.nan] * num_dofs
                elif text == "Custom":
                    current_disp = self.imported_data["nodes"][node_id].get("displacement")

                    if current_disp and not all(np.isnan(val) for val in current_disp):
                        new_displacement = list(current_disp)
                    else:
                        new_displacement = [np.nan] * num_dofs

                if len(new_displacement) > num_dofs:
                    new_displacement = new_displacement[:num_dofs]
                elif len(new_displacement) < num_dofs:
                    new_displacement.extend([np.nan] * (num_dofs - len(new_displacement)))
                self.imported_data["nodes"][node_id]["displacement"] = tuple(new_displacement)
                self.update_imported_data()

                if self.tree_widget:
                    self.tree_widget.update_tree(self.imported_data)
                self.details_label.setHidden(True)
                self.display_node_info(node_id=node_id, show_node_id=show_node_id, show_constraint_type=True)
            constraints_combo.currentTextChanged.connect(update_constraint_type)
            constraints_combo.setEnabled(enable_constraint_type)
            self.details_tree.setItemWidget(constraint_item, 1, wrap_widget(constraints_combo))
            node_item.addChild(constraint_item)

        # ---------------------------------------------
        # COORDINATES AND FORCES GROUPS
        # ---------------------------------------------

        def add_group(heading: str, labels: list[str], values: list[Any], unit_key: str, data_key: str) -> None:
            """
            Adds a group of properties (e.g., Coordinates, Forces, Displacements)
            to the tree, with editable line edits for each value.
            Args:
                heading (str): The header text for the group.
                labels (list[str]): A list of labels for each property within the group (e.g., "X", "Y").
                values (list[Any]): A list of corresponding values for each property.
                unit_key (str): The key to retrieve the unit from `units_dict`.
                data_key (str): The key indicating the type of data ("coordinate", "force", "displacement").
            """
            group_item = QTreeWidgetItem(node_item, [heading, "", ""])
            group_item.setBackground(0, color_heading)
            group_item.setFirstColumnSpanned(True)
            unit = units_dict.get(unit_key, "")

            for i, (label, val) in enumerate(zip(labels, values)):
                display_val = "---" if np.isnan(val) else f"{val:.4g}"
                item = QTreeWidgetItem(group_item, [label, display_val, unit])
                item.setBackground(0, color_items)
                item.setBackground(1, color_items)
                item.setBackground(2, color_items)
                le = create_line_edit(
                    val,
                    callback=lambda new_val, k_type=data_key, idx=i: on_node_property_changed(
                        node_id, k_type, idx, new_val),
                    validator=NaNValidator()
                )
                self.details_tree.setItemWidget(item, 1, wrap_widget(le))
                group_item.addChild(item)


        def on_node_property_changed(node_id: int, key_type: str, index: int, new_value: float) -> None:
            """
            Handles changes to node properties (coordinates, forces, or displacements).
            Updates the `self.imported_data` accordingly and refreshes the tree.
            Args:
                node_id (int): The ID of the node being modified.
                key_type (str): The type of property being changed ("coordinate", "force", or "displacement").
                index (int): The index of the specific property within its group (e.g., 0 for X, 1 for Y).
                new_value (float): The new value for the property.
            """

            try:
                nodes = self.imported_data["nodes"]
                current_node_data = nodes.get(node_id)

                if not current_node_data:
                    return

                if key_type == "coordinate":

                    if index == 0:
                        current_node_data["X"] = new_value
                    elif index == 1:
                        current_node_data["Y"] = new_value
                    elif index == 2:
                        current_node_data["Z"] = new_value
                elif key_type == "force":
                    force_labels = self.imported_data["structure_info"]["force_labels"]
                    forces = list(current_node_data.get("force", [np.nan] * len(force_labels)))

                    if index < len(forces):
                        forces[index] = new_value
                    current_node_data["force"] = tuple(forces)
                    self.imported_data['concentrated_loads'][node_id] = current_node_data["force"]
                elif key_type == "displacement":
                    disp_labels = self.imported_data["structure_info"]["displacement_labels"]
                    displacements = list(current_node_data.get("displacement", [np.nan] * len(disp_labels)))

                    if index < len(displacements):
                        displacements[index] = new_value
                    current_node_data["displacement"] = tuple(displacements)
                    self.imported_data['nodal_displacements'][node_id] = current_node_data["displacement"]
                self.imported_data["nodes"][node_id] = current_node_data
                self.update_imported_data()

                if self.tree_widget:
                    self.tree_widget.update_tree(self.imported_data)

            except ValueError as e:
                print(f"Error updating node {key_type}: {e}")

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        coords = [node_data.get("X", np.nan), node_data.get("Y", np.nan),
                  node_data.get("Z", np.nan)]
        add_group("Coordinates", ["X", "Y", "Z"], coords, "Position (X,Y,Z)",
                  "coordinate")
        force_labels = self.imported_data["structure_info"]["force_labels"]
        add_group("Forces", force_labels,
                  list(node_data.get("force", [np.nan] * len(force_labels))),
                  "Force (Fx,Fy,Fz)", "force")
        disp_labels = self.imported_data["structure_info"]["displacement_labels"]
        add_group("Displacements", disp_labels,
                  list(node_data.get("displacement", [np.nan] * len(disp_labels))),
                  "Displacement (Dx,Dy,Dz)", "displacement")
        self.details_tree.expandAll()

    # ---------------------------------------------
    # ELEMENT DISPLAY & HANDLING
    # ---------------------------------------------

    def display_element_info(self, elem_id: int) -> None:
        """
        Displays element information in the `details_tree` QTreeWidget.
        Args:
            elem_id (int): The ID of the element whose information is to be displayed.
        """
        self.block_item_changed = True
        self.details_tree.clear()
        self.details_tree.setColumnCount(3)
        self.details_tree.setHeaderLabels(["Property", "Value", "Unit"])


        def create_line_edit(value: Any, callback: Optional[callable] = None,
                             validator: Optional[Any] = None) -> QLineEdit:
            """
            Creates a QLineEdit with an optional validator.
            Args:
                value (Any): The initial value to display in the QLineEdit.
                callback (Optional[callable]): A function to call when editing finishes.
                                               Defaults to None.
                validator (Optional[Any]): A validator object (e.g., QIntValidator, QDoubleValidator).
                                           Defaults to None.
            Returns:
                QLineEdit: The created QLineEdit widget.
            """
            line_edit = QLineEdit(str(value))

            if validator:
                line_edit.setValidator(validator)

            if callback:
                line_edit.editingFinished.connect(
                    lambda le=line_edit: handle_editing_finished(le, callback))
            return line_edit

        def handle_editing_finished(line_edit: QLineEdit, callback: callable) -> None:
            """
            Handles the `editingFinished` signal of a QLineEdit.
            Args:
                line_edit (QLineEdit): The QLineEdit that emitted the signal.
                callback (callable): The callback function to execute with the new value.
            """

            try:
                text: str = line_edit.text()

                if not text:
                    return

                if line_edit.validator() is not None:
                    callback(int(text))
                else:
                    callback(text)

            except ValueError as e:
                print(f"Invalid value entered: {e}")


        def wrap_widget(widget: QWidget) -> QWidget:
            """
            Wraps a widget in a layout to remove extra spacing.
            Args:
                widget (QWidget): The widget to wrap.
            Returns:
                QWidget: A container widget with the wrapped widget.
            """
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            layout.setStretch(0, 1)
            return container

        # ---------------------------------------------
        # COLORS AND STYLING
        # ---------------------------------------------

        color_heading: QColor = QColor("#e1ecf4")
        color_items: QColor = QColor("white")
        font: QFont = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setContentsMargins(5, 5, 5, 5)
        elem_data: Dict[str, Any] = self.imported_data["elements"][elem_id]
        units: Dict[str, str] = self.imported_data["saved_units"]
        element_item: QTreeWidgetItem = QTreeWidgetItem(self.details_tree, [f"Element {elem_id}"])
        element_font: QFont = QFont("Arial", 12, QFont.Weight.Bold)
        element_item.setFont(0, element_font)
        element_item.setFirstColumnSpanned(True)
        element_item.setBackground(0, QColor("#e1ecf4"))
        self.details_tree.addTopLevelItem(element_item)

        # ---------------------------------------------
        # SECTION CODE COMBO BOX ROW
        # ---------------------------------------------

        section_code: str = elem_data.get("section_code", "")
        section_item: QTreeWidgetItem = QTreeWidgetItem(element_item, ["Section Code", "", ""])
        section_combo: QComboBox = QComboBox()
        section_combo.addItems(self.imported_data["cross_sections"].keys())
        section_combo.setCurrentText(section_code)


        def update_section_code(text: str) -> None:
            """
            Updates the section code for the current element.
            """
            self.imported_data["elements"][elem_id]["section_code"] = text
            self.tree_widget.update_tree(self.imported_data)
        section_combo.currentTextChanged.connect(update_section_code)
        self.details_tree.setItemWidget(section_item, 1, wrap_widget(section_combo))

        # ---------------------------------------------
        # MATERIAL CODE COMBO BOX ROW
        # ---------------------------------------------

        material_code: str = elem_data.get("material_code", "")
        material_item: QTreeWidgetItem = QTreeWidgetItem(element_item, ["Material Code", "", ""])
        material_combo: QComboBox = QComboBox()
        material_combo.addItems(self.imported_data["materials"].keys())
        material_combo.setCurrentText(material_code)


        def update_material_code(text: str) -> None:
            """
            Updates the material code for the current element.
            """
            self.imported_data["elements"][elem_id]["material_code"] = text
            self.tree_widget.update_tree(self.imported_data)
        material_combo.currentTextChanged.connect(update_material_code)
        self.details_tree.setItemWidget(material_item, 1, wrap_widget(material_combo))

        # ---------------------------------------------
        # PROPERTY UNITS
        # ---------------------------------------------

        property_units: Dict[str, str] = {
            "node1": "",
            "node2": "",
            "v": "",
            "E": units.get("Modulus (E,G)", ""),
            "G": units.get("Modulus (E,G)", ""),
            "A": units.get("Cross-Sectional Area (A)", ""),
            "J": units.get("Moment of Inertia (Iy,Iz,J)", ""),
            "Iy": units.get("Moment of Inertia (Iy,Iz,J)", ""),
            "Iz": units.get("Moment of Inertia (Iy,Iz,J)", ""),
            "length": units.get("Position (X,Y,Z)", ""),
            "angle": "degrees"
        }


        def add_group(heading: str, properties: Dict[str, Any], parent_item: QTreeWidgetItem) -> None:
            """
            Adds a group of properties to the tree.
            Args:
                heading (str): The heading for the property group.
                properties (Dict[str, Any]): A dictionary of property names and their values.
                parent_item (QTreeWidgetItem): The parent item to which this group will be added.
            """
            group: QTreeWidgetItem = QTreeWidgetItem(parent_item, [heading])
            group.setBackground(0, color_heading)
            group.setFirstColumnSpanned(True)

            for key, value in properties.items():

                if isinstance(value, (tuple, list, dict)):
                    continue
                else:
                    unit: str = property_units.get(key, "")

                    if key in ("node1", "node2"):
                        item: QTreeWidgetItem = QTreeWidgetItem(group, [key, str(int(value)), unit])
                    else:
                        item = QTreeWidgetItem(group, [key, f"{float(value):.4g}", unit])
                    item.setBackground(0, color_items)
                    item.setBackground(1, color_items)
                    item.setBackground(2, color_items)

                    if key in ("node1", "node2"):
                        line_edit: QLineEdit = create_line_edit(
                            value,
                            callback=lambda val, k=key: on_element_property_changed(elem_id, k, val),
                            validator=QIntValidator()
                        )
                        self.details_tree.setItemWidget(item, 1, wrap_widget(line_edit))
                    else:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)


        def on_element_property_changed(elem_id: int, key: str, new_value: Any) -> None:
            """
            Handles changes to element properties.
            Args:
                elem_id (int): The ID of the element being updated.
                key (str): The key of the property being changed.
                new_value (Any): The new value for the property.
            """

            try:
                self.imported_data["elements"][elem_id][key] = new_value
                self.tree_widget.update_tree(self.imported_data)

            except ValueError as e:
                print(f"Error updating element property {key}: {e}")

        # ---------------------------------------------
        # ADD ELEMENT INFO GROUPS
        # ---------------------------------------------

        add_group("Connectivity", {
            "node1": elem_data.get("node1", ""),
            "node2": elem_data.get("node2", "")
        }, element_item)
        add_group("Section / Material Properties", {
            "E": elem_data.get("E", ""),
            "G": elem_data.get("G", ""),
            "v": elem_data.get("v", ""),
            "A": elem_data.get("A", ""),
            "J": elem_data.get("J", ""),
            "Iy": elem_data.get("Iy", ""),
            "Iz": elem_data.get("Iz", "")
        }, element_item)
        add_group("Geometry", {
            "length": elem_data.get("length", ""),
            "angle": elem_data.get("angle", 0.0)
        }, element_item)
        self.block_item_changed = False
        self.details_tree.expandAll()

    # ---------------------------------------------
    # CROSS-SECTION DISPLAY & HANDLING
    # ---------------------------------------------

    def display_cross_section_info(self, section_code: Optional[str] = None, add_section: bool = False) -> None:
        """
        Displays cross-section details in the `details_tree` QTreeWidget.
        This function populates the `details_tree` with information about a
        specific cross-section, including its type, dimensions, and allows

        for interactive editing of these properties. It also handles the
        addition of new cross-sections.
        Args:
            section_code (Optional[str]): The unique identifier of the cross-section
                                          to display. If None, and cross-sections exist,
                                          the first available section will be displayed.
            add_section (bool): If True, a new default cross-section ("Solid_Circular")
                                will be added to the imported data and then displayed.
        """

        # ---------------------------------------------
        # HELPER FUNCTIONS
        # ---------------------------------------------

        def create_line_edit(value: Any, param: str, callback: Optional[callable] = None) -> QLineEdit:
            """
            Creates a QLineEdit pre-filled with a value and optionally connects
            it to a callback function for `editingFinished` signal.
            Args:
                value (Any): The initial value to display in the QLineEdit.
                param (str): The parameter name associated with this QLineEdit.
                             Used in the callback to identify which parameter is being changed.
                callback (Optional[callable]): A function to be called when editing
                                               of the line edit is finished.
                                               The callback should accept `section_code`, `param`,
                                               and the new `value` (as a string).
            Returns:
                QLineEdit: The configured QLineEdit widget.
            """
            line_edit = QLineEdit(str(value))
            line_edit.setValidator(QDoubleValidator())

            if callback:
                line_edit.editingFinished.connect(lambda sc=section_code, p=param, le=line_edit: callback(sc, p, le.text()))
            return line_edit

        def wrap_widget(widget: QWidget) -> QWidget:
            """
            Wraps a given widget in a QHBoxLayout within a QWidget.
            This is often used to ensure proper layout and stretching behavior
            when embedding widgets directly into a QTreeWidget item.
            Args:
                widget (QWidget): The widget to be wrapped.
            Returns:
                QWidget: A container widget with the input widget laid out.
            """
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            layout.setStretch(0, 1)
            return container

        def on_dimension_change(section_code: str, param: str, value: str) -> None:
            """
            Handles changes to dimension values within the cross-section details.
            This function updates the `imported_data` dictionary with the new
            dimension value and then refreshes the main `tree_widget` to reflect
            the changes.
            Args:
                section_code (str): The unique identifier of the cross-section.
                param (str): The specific dimension parameter (e.g., "D", "B", "H").
                value (str): The new value of the dimension, as a string.
            """
            cross_sections = self.imported_data.get("cross_sections", {})

            if section_code in cross_sections:
                cross_sections[section_code]["dimensions"][param] = float(value)
                self.imported_data["cross_sections"] = cross_sections

                if self.tree_widget:
                    self.tree_widget.update_tree(self.imported_data)


        def update_dimensions_tree(section_code: str, section_type: str, dimensions_group: QTreeWidgetItem) -> None:
            """
            Updates the dimensions section of the details tree for a given cross-section.
            This function clears existing dimension items and populates them based on
            the section type and its associated dimensions, allowing for in-place editing.
            Args:
                section_code (str): The unique identifier of the cross-section.
                section_type (str): The type of the cross-section (e.g., "Solid_Circular").
                dimensions_group (QTreeWidgetItem): The parent tree widget item under which
                                                    dimension details will be displayed.
            """
            dimensions_group.takeChildren()
            param_units = self.imported_data['saved_units']["Position (X,Y,Z)"]
            default_value = 0.1
            section_params: Dict[str, list[str]] = {
                "Solid_Circular": ["D"],
                "Hollow_Circular": ["D", "d"],
                "Solid_Rectangular": ["B", "H", "angle"],
                "Hollow_Rectangular": ["B", "H", "b", "h", "angle"],
                "I_Beam": ["B", "H", "tf", "tw", "angle"],
                "C_Beam": ["B", "H", "tf", "tw", "angle"],
                "L_Beam": ["B", "H", "tf", "tw", "angle"]
            }
            param_list: list[str] = section_params.get(section_type, [])
            cross_sections = self.imported_data.get("cross_sections", {})
            section_data = cross_sections.get(section_code, {})
            raw_dimensions = section_data.get("dimensions", {})

            if isinstance(raw_dimensions, dict) and raw_dimensions:
                dimensions = raw_dimensions
            elif isinstance(raw_dimensions, tuple) and raw_dimensions:
                dimensions = dict(zip(param_list, raw_dimensions))
            elif isinstance(raw_dimensions, (int, float)) and raw_dimensions and param_list:
                dimensions = {param_list[0]: raw_dimensions}
            else:
                dimensions = {param: default_value for param in param_list}
            self.imported_data.setdefault("cross_sections", {}).setdefault(section_code, {})["dimensions"] = copy.deepcopy(dimensions)

            for param in param_list:
                value = dimensions.get(param, default_value)
                unit = "°" if param == "angle" else param_units
                item = QTreeWidgetItem(dimensions_group, [param, str(value), unit])
                item.setBackground(0, QColor("white"))
                item.setBackground(1, QColor("white"))
                item.setBackground(2, QColor("white"))
                dimensions_group.addChild(item)
                line_edit = create_line_edit(value, param, callback=lambda sc=section_code, p=param, v=value: on_dimension_change(sc, p, v))
                self.details_tree.setItemWidget(item, 1, wrap_widget(line_edit))


        def update_section_type(section_type: str) -> None:
            """
            Updates the type of a cross-section based on user selection in the combo box.
            This method renames the section code if a section with the new type already
            exists, updates the `imported_data` accordingly, and then refreshes the
            main tree widget and the cross-section info display.
            Args:
                section_type (str): The newly selected section type (e.g., "Solid_Circular").
            """
            old_code: str = self.sender().property("section_code")
            old_type: str = self.imported_data["cross_sections"][old_code]["type"]
            base_code: str = section_type
            count: int = 1
            new_code: str = f"{base_code}_{count}"

            while new_code in self.imported_data['cross_sections'].keys():
                count += 1
                new_code = f"{base_code}_{count}"
            update_dimensions_tree(new_code, section_type, dimensions_group)
            self.imported_data["cross_sections"][new_code]["type"] = section_type
            new_sections: Dict[str, Any] = self.update_dictionary_key_ordered(self.imported_data["cross_sections"], old_code, new_code)
            self.imported_data["cross_sections"] = new_sections.copy()

            if self.tree_widget:
                self.tree_widget.update_tree(self.imported_data, item_text=new_code)
        
        # ---------------------------------------------
        # INITIAL SETUP OF DETAILS TREE
        # ---------------------------------------------

        self.details_tree.clear()
        self.details_tree.setColumnCount(3)
        self.details_tree.setHeaderLabels(["Property", "Value", "Unit"])
        color_heading = QColor("#e1ecf4")
        color_normal = QColor("white")
        color_items = QColor("#e1ecf4")
        font = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setContentsMargins(5, 5, 5, 5)
        cross_sections: Dict[str, Any] = self.imported_data.get("cross_sections", {})

        if section_code not in cross_sections:
            return

        section_data: Dict[str, Any] = cross_sections[section_code]

        # ---------------------------------------------
        # TOP-LEVEL SECTION ITEM
        # ---------------------------------------------

        section_group = QTreeWidgetItem(self.details_tree)
        section_font = QFont("Arial", 12, QFont.Weight.Bold)
        section_group.setFont(0, section_font)
        section_group.setFirstColumnSpanned(True)
        section_group.setText(0, f"{section_code}")
        section_group.setBackground(0, color_heading)
        self.details_tree.addTopLevelItem(section_group)

        # ---------------------------------------------
        # SECTION TYPE DROPDOWN
        # ---------------------------------------------

        type_item = QTreeWidgetItem(["Section Type", "", ""])
        type_item.setBackground(0, color_normal)
        type_item.setBackground(1, color_normal)
        type_item.setBackground(2, color_normal)
        self.section_type_combo = QComboBox()
        section_types: list[str] = [
            "Solid_Circular", "Hollow_Circular", "Solid_Rectangular",
            "Hollow_Rectangular", "I_Beam", "C_Beam", "L_Beam"
        ]
        self.section_type_combo.addItems(section_types)
        self.section_type_combo.setCurrentText(section_data.get("type", "Solid_Circular"))
        self.section_type_combo.setProperty("section_code", section_code)
        self.section_type_combo.currentTextChanged.connect(update_section_type)
        section_group.addChild(type_item)
        self.details_tree.setItemWidget(type_item, 1, self.section_type_combo)

        # ---------------------------------------------
        # DIMENSIONS GROUP
        # ---------------------------------------------

        dimensions_group = QTreeWidgetItem(["Dimensions", "", ""])
        dimensions_group.setBackground(0, color_items)
        dimensions_group.setBackground(1, color_items)
        dimensions_group.setBackground(2, color_items)
        section_group.addChild(dimensions_group)
        update_dimensions_tree(section_code, section_data["type"], dimensions_group)
        self.details_tree.expandAll()

    # ---------------------------------------------
    # MATERIAL DISPLAY & HANDLING
    # ---------------------------------------------

    def display_material_info(self, material_code: Optional[str] = None, add_material: bool = False) -> Optional[str]:
        """
        Displays material details in the `details_tree` QTreeWidget.
        Allows viewing existing material properties and adding new materials
        from a library.
        Args:
            material_code (Optional[str]): The code of the material to display.
                If None, and `add_material` is True, a new material is added.
                If None and no materials exist, the function returns.
                Defaults to None.
            add_material (bool): If True, a new material from the library is
                added and displayed. Defaults to False.
        Returns:
            Optional[str]: The code of the displayed or newly added material,
                or None if no material could be displayed/added.
        """

        # ---------------------------------------------
        # HELPER FUNCTIONS
        # ---------------------------------------------

        def create_line_edit(value: Any, callback: Optional[callable] = None) -> QLineEdit:
            """
            Creates a QLineEdit pre-filled with a value and optionally connects
            its `editingFinished` signal to a callback.
            """
            line_edit: QLineEdit = QLineEdit(str(value))
            line_edit.setValidator(QDoubleValidator())

            if callback:
                line_edit.editingFinished.connect(
                    lambda le=line_edit: handle_editing_finished(le, callback))
            return line_edit

        def handle_editing_finished(line_edit: QLineEdit, callback: callable) -> None:
            """
            Callback function to handle `editingFinished` signal from QLineEdit.
            Converts the text to a float and calls the provided callback.
            """

            try:
                float_value: float = float(line_edit.text())
                callback(float_value)

            except ValueError:
                print("Invalid float value entered")


        def wrap_widget(widget: QWidget) -> QWidget:
            """
            Wraps a given widget in a QHBoxLayout to ensure proper alignment
            and stretching within a QTreeWidget item.
            """
            container: QWidget = QWidget()
            layout: QHBoxLayout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            layout.setStretch(0, 1)
            return container

        def update_element_material_code(elements: Dict[Any, Dict[str, Any]], code: str, new_code: str) -> Dict[Any, Dict[str, Any]]:
            """
            Static Method:
            Updates the 'material_code' for elements in the given dictionary.
            Args:
                elements (Dict[Any, Dict[str, Any]]): Dictionary of elements.
                code (str): The old material code to be replaced.
                new_code (str): The new material code.
            Returns:
                Dict[Any, Dict[str, Any]]: A new dictionary with updated elements.
            """
            updated_elements: Dict[Any, Dict[str, Any]] = {}

            for elem_id, elem_data in elements.items():

                if elem_data.get('material_code') == code:
                    updated_elements[elem_id] = {**elem_data, 'material_code': new_code}
                else:
                    updated_elements[elem_id] = elem_data
            return updated_elements

        def on_material_property_changed(material_code: str, prop_name: str, new_value: float) -> None:
            """
            Slot:
            Handles changes to individual material properties from the QLineEdit.
            Updates the internal `imported_data` structure and refreshes the tree widget.
            Args:
                material_code (str): The code of the material being edited.
                prop_name (str): The name of the property that was changed.
                new_value (float): The new numerical value of the property.
            """

            try:
                self.imported_data["materials"][material_code]["properties"][prop_name] = new_value
                self.tree_widget.update_tree(self.imported_data)

            except ValueError as e:
                print(f"Error updating material property: {e}")

            except KeyError as e:
                print(f"Error: Material code '{material_code}' or property '{prop_name}' not found: {e}")

        # ---------------------------------------------
        # CLEAR AND CONFIGURE DETAILS TREE
        # ---------------------------------------------

        self.details_tree.clear()
        self.details_tree.setColumnCount(3)
        self.details_tree.setHeaderLabels(["Property", "Value", "Unit"])

        # ---------------------------------------------
        # COLORS AND STYLING
        # ---------------------------------------------

        color_heading: QColor = QColor("#e1ecf4")
        color_items: QColor = QColor("white")
        font: QFont = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setContentsMargins(5, 5, 5, 5)

        # ---------------------------------------------
        # LOAD MATERIAL LIBRARY
        # ---------------------------------------------

        material_file: str = "../data/material_library.json" 

        try:
            with open(material_file, "r") as file:
                materials_data: Dict[str, Any] = json.load(file)

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Material library file not found: {material_file}")
            return None

        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", f"Error decoding material library file: {material_file}")
            return None

        target_units: Dict[str, str] = {
            "Modulus (E,G)": self.imported_data['saved_units'].get("Modulus (E,G)", "Pa"),
            "Density (ρ)": self.imported_data['saved_units'].get("Density (ρ)", "kg/m³")
        }
        materials_data = UnitsHandling.convert_material_units(materials_data, target_units)

        # ---------------------------------------------
        # ADD NEW MATERIAL LOGIC
        # ---------------------------------------------

        if add_material:

            if not materials_data:
                QMessageBox.warning(self, "Warning", "No materials available in the library to add.")
                return None

            selected_type: str = list(materials_data.keys())[0]
            selected_data: Dict[str, Any] = materials_data[selected_type]
            selected_data_without_units: Dict[str, Any] = selected_data.copy()
            selected_data_without_units.pop("Unit", None)
            base_code: str = selected_type.replace(" ", "_")
            count: int = 1
            material_code = f"{base_code}_{count}"
            existing_material_codes: set[str] = set(self.imported_data.get("materials", {}).keys()) | set(materials_data.keys())

            while material_code in existing_material_codes:
                count += 1
                material_code = f"{base_code}_{count}"
            self.imported_data.setdefault("materials", {})[material_code] = {
                "type": selected_type,
                "properties": selected_data_without_units
            }

        # ---------------------------------------------
        # VALIDATE MATERIAL CODE AND RETRIEVE DATA
        # ---------------------------------------------

        if not self.imported_data.get("materials"):
            return None

        if material_code is None or material_code not in self.imported_data["materials"]:
            material_code = next(iter(self.imported_data["materials"]))

            if material_code is None:
                return None

        material_data: Dict[str, Any] = self.imported_data["materials"][material_code]
        material_type: str = material_data["type"]
        material_properties: Dict[str, Any] = material_data["properties"]
        global_units: Dict[str, str] = self.imported_data.get("saved_units", {})
        material_unit_from_type: Dict[str, str] = materials_data.get(material_type, {}).get("Unit", {})
        unit_mapping: Dict[str, str] = {
            "v": "v",
            "E": "Modulus (E,G)",
            "density": "Density (ρ)",
            "G": "Modulus (E,G)",
            "force_per_length": "Force/Length (F/L)",
            "A": "Cross-Sectional Area (A)",
            "J": "Moment of Inertia (Iy,Iz,J)",
            "Iy": "Moment of Inertia (Iy,Iz,J)",
            "Iz": "Moment of Inertia (Iy,Iz,J)",
            "length": "Position (X,Y,Z)",
            "angle": "angle"
        }
        material_units: Dict[str, str] = {
            key: global_units.get(global_key, material_unit_from_type.get(key, ""))

            for key, global_key in unit_mapping.items()
        }

        # ---------------------------------------------
        # TOP-LEVEL ITEM: Material Code
        # ---------------------------------------------

        material_group: QTreeWidgetItem = QTreeWidgetItem(self.details_tree)
        material_font: QFont = QFont("Arial", 12, QFont.Weight.Bold)
        material_group.setFont(0, material_font)
        material_group.setFirstColumnSpanned(True)
        material_group.setText(0, f"{material_code}")
        material_group.setBackground(0, color_heading)
        self.details_tree.addTopLevelItem(material_group)

        # ---------------------------------------------
        # TYPE ROW (Combo box)
        # ---------------------------------------------

        type_item: QTreeWidgetItem = QTreeWidgetItem(material_group)
        type_item.setText(0, "Type")
        type_item.setText(2, "")
        type_item.setBackground(0, color_items)
        type_item.setBackground(1, color_items)
        combo: QComboBox = QComboBox()
        combo.addItem("+Save to JSON")
        combo.addItems(materials_data.keys())
        combo.setCurrentText(material_type)
        combo.setProperty("material_code", material_code)


        def update_material_type(new_type: str) -> None:
            """
            Slot:
            Updates the material type when the QComboBox selection changes.
            Handles saving to JSON and updating existing material codes across elements.
            Args:
                new_type (str): The newly selected material type from the combo box.
            """

            if not new_type:
                return

            code: str = combo.property("material_code")

            if new_type == "+Save to JSON":
                data: Dict[str, Any] = self.imported_data["materials"][code]
                new_type_for_save: str = data["type"]
                saved: Dict[str, Any] = materials_data.get(new_type_for_save, {})
                saved.update(data["properties"])
                materials_data[new_type_for_save] = {
                    **data["properties"],
                    "Unit": material_units
                }

                try:
                    with open(material_file, "w") as f:
                        json.dump(materials_data, f, indent=4)
                    QMessageBox.information(self, "Material Saved", f"Material '{new_type_for_save}' saved to library.")

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save material to library: {e}")
                combo.setCurrentText(material_type)
                return

            self.imported_data["materials"][code]["type"] = new_type
            default_data: Dict[str, Any] = materials_data[new_type].copy()
            default_data.pop("Unit", None)
            self.imported_data["materials"][code]["properties"] = default_data
            base_code_new: str = new_type.replace(" ", "_")
            count_new: int = 1
            new_code_candidate: str = f"{base_code_new}_{count_new}"
            match = re.match(rf"^{re.escape(base_code_new)}_(\d+)$", new_type)

            if match:
                count_new = int(match.group(1))

            while new_code_candidate in self.imported_data['materials'].keys() or new_code_candidate in materials_data.keys():
                count_new += 1
                new_code_candidate = f"{base_code_new}_{count_new}"
            new_code: str = new_code_candidate
            self.imported_data["materials"] = self.update_dictionary_key_ordered(
                self.imported_data["materials"], code, new_code)
            self.imported_data["elements"] = update_element_material_code(
                self.imported_data["elements"], code, new_code)
            self.tree_widget.update_tree(self.imported_data)
            self.display_material_info(material_code=new_code)
            self.tree_widget.set_current_tree_item(parent_text="Materials", item_initial_text=new_code)
        combo.currentTextChanged.connect(update_material_type)
        self.details_tree.setItemWidget(type_item, 1, combo)
        material_group.addChild(type_item)

        # ---------------------------------------------
        # PROPERTY ROWS (Editable)
        # ---------------------------------------------

        for prop, val in material_properties.items():
            child: QTreeWidgetItem = QTreeWidgetItem(material_group)
            child.setText(0, prop)
            child.setText(1, str(val))
            child.setText(2, material_units.get(prop, ""))
            child.setFlags(child.flags() | Qt.ItemFlag.ItemIsEditable)
            child.setBackground(0, color_items)
            child.setBackground(1, color_items)
            child.setBackground(2, color_items)
            le: QLineEdit = create_line_edit(val,
                                callback=lambda v, p=prop, c=material_code: on_material_property_changed(c, p, v))
            self.details_tree.setItemWidget(child, 1, wrap_widget(le))
            material_group.addChild(child)
        self.details_tree.expandAll()
        return material_code

    # ---------------------------------------------
    # DISTRIBUTED LOAD DISPLAY & HANDLING
    # ---------------------------------------------

    def display_distributed_load_info(self, elem_id: int, elem_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Displays distributed load information in the `details_tree` QTreeWidget.
        Args:
            elem_id (int): The ID of the element associated with the distributed load.
            elem_data (Optional[Dict[str, Any]]): Pre-fetched element data. If None, it will be retrieved
                                                 from `self.imported_data`. Defaults to None.
        """
        self.details_tree.clear()
        self.details_tree.setColumnCount(3)
        self.details_tree.setHeaderLabels(["Property", "Value", "Unit"])

        # ---------------------------------------------
        # COLORS AND STYLING
        # ---------------------------------------------

        color_heading: QColor = QColor("#e1ecf4")
        color_items: QColor = QColor("white")
        font: QFont = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setContentsMargins(5, 5, 5, 5)

        try:
            self.details_tree.itemChanged.disconnect()

        except TypeError:
            pass


        # ---------------------------------------------
        # GET ELEMENT DATA
        # ---------------------------------------------

        if elem_data is None:

            if elem_id not in self.imported_data['distributed_loads']:
                return

            elem_data = self.imported_data['distributed_loads'][elem_id]
        self.selected_elemet_id = elem_id
        load_type: str = elem_data['type']
        load_direction: str = elem_data['direction']
        load_values: Any = elem_data['parameters']
        units: Dict[str, str] = self.imported_data['saved_units']

        # ---------------------------------------------
        # UNIT SETTINGS
        # ---------------------------------------------

        force_unit: str = units.get("Force (Fx,Fy,Fz)", "kN")
        length_unit: str = units.get("Position (X,Y,Z)", "m")
        force_per_length_unit: str = units.get("Force/Length (F/L)", f"{force_unit}/{length_unit}")

        # ---------------------------------------------
        # UTILITY: Create widgets
        # ---------------------------------------------

        def create_dropdown(items: list[str], current: str, callback: callable) -> QComboBox:
            """
            Creates a QComboBox (dropdown) widget.
            Args:
                items (list[str]): A list of string items to populate the dropdown.
                current (str): The currently selected item.
                callback (callable): A function to call when the selected item changes.
            Returns:
                QComboBox: The created QComboBox widget.
            """
            combo = QComboBox()
            combo.addItems(items)
            combo.setCurrentText(current)
            combo.currentTextChanged.connect(callback)
            return combo

        def create_line_edit(value: Any, callback: Optional[callable] = None) -> QLineEdit:
            """
            Creates a QLineEdit for numerical input with a QDoubleValidator.
            Args:
                value (Any): The initial value to display in the QLineEdit.
                callback (Optional[callable]): A function to call when editing finishes.
                                               Defaults to None.
            Returns:
                QLineEdit: The created QLineEdit widget.
            """
            line_edit = QLineEdit(str(float(value)))
            line_edit.setValidator(QDoubleValidator())

            if callback:
                line_edit.editingFinished.connect(lambda le=line_edit: callback(le.text()))
            return line_edit

        def wrap_widget(widget: QWidget) -> QWidget:
            """
            Wraps a widget in a layout to remove extra spacing.
            Args:
                widget (QWidget): The widget to wrap.
            Returns:
                QWidget: A container widget with the wrapped widget.
            """
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            layout.setStretch(0, 1)
            return container

        def update_and_refresh() -> None:
            """
            Saves current values and refreshes the display of the distributed load info.
            """
            self.tree_widget.update_tree(self.imported_data)
            self.display_distributed_load_info(self.selected_elemet_id)


        def on_element_id_change(new_text: str) -> None:
            """
            Handles changes to the associated element ID for the distributed load.
            Args:
                new_text (str): The new element ID as a string.
            """

            try:
                new_id: int = int(float(new_text))

                if new_id in self.imported_data['elements']:
                    new_dict = self.update_dictionary_key_ordered(self.imported_data['distributed_loads'], self.selected_elemet_id, new_id)
                    self.imported_data['distributed_loads'] = new_dict.copy()
                    self.selected_elemet_id = new_id
                    update_and_refresh()

            except ValueError:
                pass

        def on_type_change(new_type: str) -> None:
            """
            Handles changes to the distributed load type.
            Args:
                new_type (str): The new load type (e.g., "Uniform", "Equation").
            """

            if not new_type:
                return

            self.imported_data['distributed_loads'][self.selected_elemet_id]['type'] = new_type

            if new_type == "Uniform":
                self.imported_data['distributed_loads'][self.selected_elemet_id]['parameters'] = (1.0,)
            elif new_type == "Rectangular":
                self.imported_data['distributed_loads'][self.selected_elemet_id]['parameters'] = (1.0, 1.0)
            elif new_type == "Triangular":
                self.imported_data['distributed_loads'][self.selected_elemet_id]['parameters'] = (1.0, 1.0)
            elif new_type == "Trapezoidal":
                self.imported_data['distributed_loads'][self.selected_elemet_id]['parameters'] = (1.0, 1.0)
            elif new_type == "Equation":
                self.imported_data['distributed_loads'][self.selected_elemet_id]['parameters'] = "1.0 * x"
            update_and_refresh()


        def on_direction_change(new_direction: str) -> None:
            """
            Handles changes to the distributed load direction.
            Args:
                new_direction (str): The new load direction (e.g., "+Global_X").
            """

            if not new_direction:
                return

            self.imported_data['distributed_loads'][self.selected_elemet_id]['direction'] = new_direction
            update_and_refresh()


        def on_value_change(new_value: str, property_name: str) -> None:
            """
            Handles changes to the distributed load parameter values.
            Args:
                new_value (str): The new value as a string.
                property_name (str): The name of the property being changed (e.g., "Magnitude", "Equation").
            """

            if not new_value:
                return

            load_type_local: str = self.imported_data['distributed_loads'][self.selected_elemet_id]['type']
            current_params: list[Any] = list(self.imported_data['distributed_loads'][self.selected_elemet_id]['parameters'])

            try:

                if load_type_local == "Equation":
                    self.imported_data['distributed_loads'][self.selected_elemet_id]['parameters'] = new_value
                else:
                    index: int = -1
                    param_labels: Dict[str, list[str]] = {
                        "Uniform": ["Magnitude"],
                        "Rectangular": ["Start Magnitude", "End Magnitude"],
                        "Triangular": ["Start Mag.", "End Mag."],
                        "Trapezoidal": ["Start Mag.", "End Mag."],
                    }
                    labels: list[str] = param_labels.get(load_type_local, [])

                    if property_name in labels:
                        index = labels.index(property_name)

                    if index != -1:
                        current_params[index] = float(new_value)
                        self.imported_data['distributed_loads'][self.selected_elemet_id]['parameters'] = tuple(current_params)

            except ValueError as e:
                print(f"Error updating loads: {e}")

        # ---------------------------------------------
        # GROUP HEADER: Distributed Load Properties
        # ---------------------------------------------

        load_group: QTreeWidgetItem = QTreeWidgetItem(self.details_tree)
        load_font: QFont = QFont("Arial", 12, QFont.Weight.Bold)
        load_group.setFont(0, load_font)
        load_group.setFirstColumnSpanned(True)
        load_group.setText(0, f"{load_type} (E{elem_id})")
        load_group.setBackground(0, color_heading)
        self.details_tree.addTopLevelItem(load_group)

        # ---------------------------------------------
        # PROPERTY ROWS: Dropdowns and Editable Values
        # ---------------------------------------------
        
        elems_ids: list[str] = [str(eid) for eid in self.imported_data["elements"].keys() if eid != 0]
        current_element: str = str(elem_id)
        element_combo: QComboBox = create_dropdown(elems_ids, current_element, on_element_id_change)
        item_elem_id: QTreeWidgetItem = QTreeWidgetItem(["Element ID", "", ""])
        item_elem_id.setBackground(0, color_items)
        item_elem_id.setBackground(1, color_items)
        item_elem_id.setBackground(2, color_items)
        load_group.addChild(item_elem_id)
        self.details_tree.setItemWidget(item_elem_id, 1, element_combo)
        load_types: list[str] = ["Uniform", "Rectangular", "Triangular", "Trapezoidal", "Equation"]
        type_combo: QComboBox = create_dropdown(load_types, load_type, on_type_change)
        item_load_type: QTreeWidgetItem = QTreeWidgetItem(["Load Type", "", ""])
        item_load_type.setBackground(0, color_items)
        item_load_type.setBackground(1, color_items)
        item_load_type.setBackground(2, color_items)
        load_group.addChild(item_load_type)
        self.details_tree.setItemWidget(item_load_type, 1, type_combo)
        directions: list[str] = ["+Global_X", "-Global_X", "+Global_Y", "-Global_Y", "+Global_Z", "-Global_Z"]
        direction_combo: QComboBox = create_dropdown(directions, load_direction, on_direction_change)
        item_direction: QTreeWidgetItem = QTreeWidgetItem(["Load Direction", "", ""])
        item_direction.setBackground(0, color_items)
        item_direction.setBackground(1, color_items)
        item_direction.setBackground(2, color_items)
        load_group.addChild(item_direction)
        self.details_tree.setItemWidget(item_direction, 1, direction_combo)
        param_labels: Dict[str, list[str]] = {
            "Uniform": ["Magnitude"],
            "Rectangular": ["Start Magnitude", "End Magnitude"],
            "Triangular": ["Start Mag.", "End Mag."],
            "Trapezoidal": ["Start Mag.", "End Mag."],
            "Equation": ["Equation"]
        }
        units_labels: Dict[str, list[str]] = {
            "Uniform": [force_per_length_unit],
            "Rectangular": [force_per_length_unit] * 2,
            "Triangular": [force_per_length_unit, force_per_length_unit],
            "Trapezoidal": [force_per_length_unit, force_per_length_unit],
            "Equation": [force_per_length_unit]
        }
        labels: list[str] = param_labels.get(load_type, [])
        units_list: list[str] = units_labels.get(load_type, [])

        if load_type == "Equation":
            val_str: str = load_values if isinstance(load_values, str) else str(load_values)
            line_edit: QLineEdit = create_line_edit(val_str, lambda text: on_value_change(text, "Equation"))
            item_eq: QTreeWidgetItem = QTreeWidgetItem(["Equation", val_str, force_per_length_unit])
            item_eq.setBackground(0, color_items)
            item_eq.setBackground(1, color_items)
            item_eq.setBackground(2, color_items)
            load_group.addChild(item_eq)
            self.details_tree.setItemWidget(item_eq, 1, wrap_widget(line_edit))
        else:

            for i, label in enumerate(labels):

                if isinstance(load_values, (tuple, list)):
                    val: float = float(load_values[i]) if i < len(load_values) else 0.0
                elif isinstance(load_values, (int, float, str)):
                    val = float(load_values) if isinstance(load_values, (int, float)) else 1.0
                else:
                    val = 1.0
                line_edit: QLineEdit = create_line_edit(val, lambda text, current_label=label: on_value_change(text, current_label))
                item_param: QTreeWidgetItem = QTreeWidgetItem([label, str(val), units_list[i]])
                item_param.setBackground(0, color_items)
                item_param.setBackground(1, color_items)
                item_param.setBackground(2, color_items)
                load_group.addChild(item_param)
                self.details_tree.setItemWidget(item_param, 1, wrap_widget(line_edit))
        self.details_tree.expandAll()
    
    # ----------------------------------------------------------------------
    # MESH SETTINGS
    # ----------------------------------------------------------------------

    def display_mesh_info(self) -> None:
        """
        Populates the `details_tree` with mesh-related information and settings.
        This includes options for element type, element order, mesh density,
        meshing algorithm, refinement factor, non-conforming mesh, and auto re-mesh.
        Users can modify these settings through various input widgets like
        QComboBox, QLineEdit, QDoubleSpinBox, and QCheckBox.
        The changes are saved to `self.mesh_settings`.
        """
        self.details_tree.clear()
        self.details_tree.setColumnCount(2)
        self.details_tree.setHeaderLabels(["Property", "Value"])

        if not hasattr(self, 'mesh_settings'):
            self.mesh_settings = {}
        # ---------------------------------------------
        # COLORS AND STYLING
        # ---------------------------------------------

        color_heading = QColor("#e1ecf4")
        color_items = QColor("white")
        font = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setContentsMargins(5, 5, 5, 5)

        # ---------------------------------------------
        # UTILITY: Widget Creators (Helper Functions)
        # ---------------------------------------------

        def create_dropdown(items: list[str], current: str) -> QComboBox:
            """
            Creates and returns a QComboBox populated with the given items,
            setting the current text.
            Args:
                items (list[str]): A list of strings to populate the dropdown.
                current (str): The initial text to display in the dropdown.
            Returns:
                QComboBox: The configured QComboBox widget.
            """
            combo = QComboBox()
            combo.addItems(items)
            combo.setCurrentText(current)
            return combo

        def create_grouped_dropdown(groups: Dict[str, list[str]], current: str) -> QComboBox:
            """
            Creates a QComboBox with grouped items. Group names are displayed
            as disabled headers.
            Args:
                groups (Dict[str, list[str]]): A dictionary where keys are group names
                                                and values are lists of items within that group.
                current (str): The initial text to display in the dropdown.
            Returns:
                QComboBox: The configured QComboBox widget.
            """
            combo = QComboBox()

            for group_name, items in groups.items():
                combo.addItem("── " + group_name + " ──")
                index = combo.count() - 1
                combo.model().item(index).setEnabled(False)

                for item in items:
                    combo.addItem(item)

            if current in [item for sublist in groups.values() for item in sublist]:
                combo.setCurrentText(current)
            return combo

        def create_line_edit(value: Any) -> QLineEdit:
            """
            Creates and returns a QLineEdit with the given value,
            and a QDoubleValidator.
            Args:
                value (Any): The initial value for the QLineEdit.
            Returns:
                QLineEdit: The configured QLineEdit widget.
            """
            line_edit = QLineEdit(str(value))
            line_edit.setValidator(QDoubleValidator())
            return line_edit

        def create_double_spinbox(value: float, step: float) -> QDoubleSpinBox:
            """
            Creates and returns a QDoubleSpinBox with a specified range,
            single step, and initial value.
            Args:
                value (float): The initial value of the spinbox.
                step (float): The step size for incrementing/decrementing.
            Returns:
                QDoubleSpinBox: The configured QDoubleSpinBox widget.
            """
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.0, 1000.0)
            spinbox.setSingleStep(step)
            spinbox.setValue(value)
            return spinbox

        def create_checkbox(checked: bool) -> QCheckBox:
            """
            Creates and returns a QCheckBox with a specified checked state.
            Args:
                checked (bool): The initial checked state of the checkbox.
            Returns:
                QCheckBox: The configured QCheckBox widget.
            """
            cb = QCheckBox()
            cb.setChecked(checked)
            return cb

        def wrap_widget(widget: QWidget) -> QWidget:
            """
            Wraps a given widget in a QHBoxLayout to remove extra margins
            and control its stretching behavior.
            Args:
                widget (QWidget): The widget to be wrapped.
            Returns:
                QWidget: A container widget holding the wrapped widget.
            """
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            layout.setStretch(0, 1)
            return container

        def save_mesh_setting(key: str, value: Any) -> None:
            """
            Saves a mesh setting to the `self.mesh_settings` dictionary.
            Args:
                key (str): The key (name) of the mesh setting.
                value (Any): The value to be saved for the given key.
            """

            if not hasattr(self, 'mesh_settings'):
                self.mesh_settings = {}
            self.mesh_settings[key] = value

        # ---------------------------------------------
        # GROUP HEADER: Mesh Settings
        # ---------------------------------------------

        mesh_group = QTreeWidgetItem(self.details_tree)
        mesh_font = QFont("Arial", 12, QFont.Weight.Bold)
        mesh_group.setFont(0, mesh_font)
        mesh_group.setFirstColumnSpanned(True)
        mesh_group.setText(0, "Mesh Settings")
        mesh_group.setBackground(0, color_heading)
        self.details_tree.addTopLevelItem(mesh_group)

        # ---------------------------------------------
        # ELEMENT TYPES: Detailed Element Options
        # ---------------------------------------------

        structure_type = self.imported_data.get("structure_info", {}).get("element_type", "")
        element_categories = {
            "1D Elements": ["Bar", "Beam", "Frame"],
            "2D Elements": ["Quadrilateral Dominant", "Triangular Dominant", "Mixed Quad/Tri"],
            "3D Elements": ["Tetrahedrons", "Hexahedrons", "Mixed Tet/Hex"]
        }
        structure_to_groups = {
            "2D_Truss": ["1D Elements"],
            "3D_Truss": ["1D Elements"],
            "2D_Beam": ["1D Elements"],
            "3D_Frame": ["1D Elements"],
            "2D_Solid": ["2D Elements"],
            "3D_Solid": ["3D Elements"]
        }
        allowed_groups = structure_to_groups.get(structure_type, [])
        allowed_elements = element_categories.get(allowed_groups[0], []) if allowed_groups else []
        filtered_groups = {k: v for k, v in element_categories.items() if k in allowed_groups}
        default_selection = self.mesh_settings.get("element_type")

        if default_selection is None:

            for category, elements in element_categories.items():

                for e in elements:

                    if e in allowed_elements:
                        default_selection = e
                        break

                if default_selection:
                    break

        if structure_type in ["2D_Truss", "3D_Truss"]:
            default_text = "Bar"
        elif structure_type == "2D_Beam":
            default_text = "Beam"
        elif structure_type == "3D_Frame":
            default_text = "Frame"
        elif default_selection in allowed_elements:
            default_text = default_selection
        elif allowed_elements:
            default_text = allowed_elements[0]
        else:
            default_text = ""

        # ---------------------------------------------
        # PROPERTY ROWS: Editable values
        # ---------------------------------------------

        settings_data = [
            ("Element Type", create_grouped_dropdown(filtered_groups, default_text), "element_type"),
            ("Element Order", ["Linear", "Quadratic"], "element_order"),
            ("Mesh Density", 10, "mesh_density"),
            ("Mesh Type", ["Automatic", "Structured", "Free"], "algorithm"),
            ("Refinement Factor", 1.0, "refinement_factor"),
            ("Non-Conforming Mesh", False, "non_conforming"),
            ("Auto Re-mesh", True, "auto_remesh"),
        ]

        for name, data, key in settings_data:
            item = QTreeWidgetItem()
            item.setText(0, name)
            item.setBackground(0, color_items)
            item.setBackground(1, color_items)
            mesh_group.addChild(item)
            stored_value = self.mesh_settings.get(key)
            current_value = None
            widget = None

            if isinstance(data, QComboBox):
                widget = data

                if stored_value is not None and stored_value in [widget.itemText(i) for i in range(widget.count())]:
                    widget.setCurrentText(stored_value)
                    current_value = stored_value
                else:
                    current_value = default_text
                widget.currentTextChanged.connect(lambda val, k=key: save_mesh_setting(k, val))
            elif isinstance(data, list):
                default_value = data[0]
                current_value = stored_value if stored_value is not None else default_value
                widget = create_dropdown(data, current_value)
                widget.currentTextChanged.connect(lambda val, k=key: save_mesh_setting(k, val))
            elif key == "refinement_factor":
                default_value = data
                current_value = stored_value if stored_value is not None else default_value
                widget = create_double_spinbox(current_value, 1)
                widget.valueChanged.connect(lambda val, k=key: save_mesh_setting(k, val))
            elif key in ["non_conforming", "auto_remesh"]:
                default_value = data
                current_value = stored_value if stored_value is not None else default_value
                widget = create_checkbox(current_value)
                widget.stateChanged.connect(lambda state, k=key: save_mesh_setting(k, bool(state)))
            elif isinstance(data, (int, float)):
                default_value = data
                current_value = stored_value if stored_value is not None else default_value
                widget = create_line_edit(current_value)
                widget.editingFinished.connect(lambda k=key, w=widget: save_mesh_setting(k, w.text()))

            if widget:
                self.details_tree.setItemWidget(item, 1, wrap_widget(widget))
                save_mesh_setting(key, current_value)
        self.details_tree.expandAll()

    # ---------------------------------------------
    # SOLVER SETTINGS
    # ---------------------------------------------

    def display_solver_info(self) -> None:
        """
        Displays solver settings in the `details_tree` QTreeWidget.
        Allows users to configure solver parameters such as solver type,
        tolerance, max iterations, sparse matrix usage, and large deflection analysis.
        Includes a 'Solve' button to initiate the structural analysis.
        """
        self.details_tree.clear()
        self.details_tree.setColumnCount(2)
        self.details_tree.setHeaderLabels(["Property", "Value"])

        # ---------------------------------------------
        # COLORS AND STYLING
        # ---------------------------------------------

        color_heading: QColor = QColor("#e1ecf4")
        color_items: QColor = QColor("white")
        font: QFont = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setContentsMargins(5, 5, 5, 5)

        # ---------------------------------------------
        # UTILITY: Create widgets
        # ---------------------------------------------

        def create_dropdown(items: list[str], current: str) -> QComboBox:
            """
            Static Method:
            Creates and returns a QComboBox populated with `items`,
            with `current` as the initially selected text.
            """
            combo: QComboBox = QComboBox()
            combo.addItems(items)
            combo.setCurrentText(current)
            return combo

        def create_line_edit(value: Any) -> QLineEdit:
            """
            Static Method:
            Creates and returns a QLineEdit with a `QDoubleValidator`
            and initial `value`.
            """
            line_edit: QLineEdit = QLineEdit(str(value))
            line_edit.setValidator(QDoubleValidator())
            return line_edit

        def create_checkbox(checked: bool) -> QCheckBox:
            """
            Static Method:
            Creates and returns a QCheckBox with its checked state set.
            """
            cb: QCheckBox = QCheckBox()
            cb.setChecked(checked)
            return cb

        def create_button(text: str = "Button") -> QPushButton:
            """
            Static Method:
            Creates and returns a styled QPushButton with the given text.
            """
            button: QPushButton = QPushButton(text)
            button.setContentsMargins(0, 0, 0, 0)
            button.setMinimumHeight(24)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            button.setStyleSheet("""
                QPushButton {
                    padding: 6px 3px;
                    border: 1px solid
                    border-radius: 5px;
                    background-color: #f7f7f7;
                    color: #000000;
                    font-size: 11px;
                    font-weight: bold;
                    text-align: center;
                }
                QPushButton:hover {
                    background-color: #c1d7df;
                }
                QPushButton:pressed {
                    background-color: #9fd6eb;
                }
            """)
            return button

        def wrap_widget(widget: QWidget) -> QWidget:
            """
            Static Method:
            Wraps a given widget in a QHBoxLayout within a QWidget

            for proper layout within a QTreeWidget item.
            """
            container: QWidget = QWidget()
            layout: QHBoxLayout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            layout.setStretch(0, 1)
            return container

        def save_solver_setting(key: str, value: Any) -> None:
            """
            Slot:
            Saves a solver setting to the `self.solver_settings` dictionary.
            Args:
                key (str): The key for the solver setting (e.g., "tolerance", "solver_type").
                value (Any): The value to be saved for the setting.
            """

            if key in ["tolerance", "max_iterations"]:

                try:

                    if key == "tolerance":
                        self.solver_settings[key] = float(value)
                    elif key == "max_iterations":
                        self.solver_settings[key] = int(value)

                except ValueError:
                    print(f"Invalid input for {key}: {value}. Keeping previous value.")
            else:
                self.solver_settings[key] = value
            print(f"Solver setting '{key}' updated to: {self.solver_settings[key]}")
        solver_group: QTreeWidgetItem = QTreeWidgetItem(self.details_tree)
        solver_font: QFont = QFont("Arial", 12, QFont.Weight.Bold)
        solver_group.setFont(0, solver_font)
        solver_group.setFirstColumnSpanned(True)
        solver_group.setText(0, "Solver Settings")
        solver_group.setBackground(0, color_heading)
        self.details_tree.addTopLevelItem(solver_group)
        settings_data: list[Tuple[str, Any, str]] = [
            ("Solver Type", ["Direct", "Iterative"], "solver_type"),
            ("Tolerance", 1e-6, "tolerance"),
            ("Max Iterations", 100, "max_iterations"),
            ("Sparse Matrix", True, "sparse_matrix"),
            ("Large Deflection", False, "large_deflection"),
            ("Solve", {'button'}, "solve"),
        ]

        for name, default_value, key in settings_data:
            item: QTreeWidgetItem = QTreeWidgetItem()
            item.setText(0, name)
            item.setBackground(0, color_items)
            item.setBackground(1, color_items)
            solver_group.addChild(item)
            stored_value: Any = self.solver_settings.get(key)
            widget: Optional[QWidget] = None

            if isinstance(default_value, list):
                current_value: str = stored_value if stored_value is not None else default_value[0]
                widget = create_dropdown(default_value, current_value)
                widget.currentTextChanged.connect(lambda val, k=key: save_solver_setting(k, val))
            elif isinstance(default_value, bool):
                current_value_bool: bool = stored_value if stored_value is not None else default_value
                widget = create_checkbox(current_value_bool)
                widget.stateChanged.connect(lambda state, k=key: save_solver_setting(k, bool(state)))
            elif isinstance(default_value, (int, float)):
                current_value_numeric: Any = stored_value if stored_value is not None else default_value
                widget = create_line_edit(current_value_numeric)
                widget.editingFinished.connect(lambda k=key, w=widget: save_solver_setting(k, type(default_value)(w.text())))
            elif key == "solve":
                widget = create_button("Solve")
                widget.clicked.connect(self.tree_widget._main_solver)

            if widget:
                self.details_tree.setItemWidget(item, 1, wrap_widget(widget))
        self.details_tree.expandAll()

    # ----------------------------------------------------------------------
    # POST-PROCESSING SETTINGS
    # ----------------------------------------------------------------------

    def display_postprocessing_info(self) -> None:
        """
        Populates the `details_tree` with post-processing related information and settings.
        This includes options for display type (Contour, Deformed Shape, Vector Field),
        result to display (Displacement, Force, Stress), scale factor,
        showing undeformed mesh, auto update on mesh change, and default tab.
        Users can modify these settings through various input widgets like
        QComboBox, QLineEdit, and QCheckBox.
        The changes are saved to `self.post_settings`.
        """
        self.details_tree.clear()
        self.details_tree.setColumnCount(2)
        self.details_tree.setHeaderLabels(["Property", "Value"])
        
        # ---------------------------------------------
        # COLORS AND FONTS
        # ---------------------------------------------

        color_heading = QColor("#e1ecf4")
        color_items = QColor("white")
        font = QFont("Arial", 11)
        self.details_tree.setFont(font)
        self.details_tree.setContentsMargins(5, 5, 5, 5)

        # ---------------------------------------------
        # UTILITY: Widget Creators (Helper Functions)
        # ---------------------------------------------

        def create_dropdown(items: list[str], current: str) -> QComboBox:
            """
            Creates and returns a QComboBox populated with the given items,
            setting the current text.
            Args:
                items (list[str]): A list of strings to populate the dropdown.
                current (str): The initial text to display in the dropdown.
            Returns:
                QComboBox: The configured QComboBox widget.
            """
            combo = QComboBox()
            combo.addItems(items)
            combo.setCurrentText(current)
            return combo

        def create_line_edit(value: Any) -> QLineEdit:
            """
            Creates and returns a QLineEdit with the given value,
            and a QDoubleValidator for a range of 0.0 to 1e4 with 3 decimal places.
            Args:
                value (Any): The initial value for the QLineEdit.
            Returns:
                QLineEdit: The configured QLineEdit widget.
            """
            line_edit = QLineEdit(str(value))
            line_edit.setValidator(QDoubleValidator(0.0, 1e4, 3))
            return line_edit

        def create_checkbox(checked: bool) -> QCheckBox:
            """
            Creates and returns a QCheckBox with a specified checked state.
            Args:
                checked (bool): The initial checked state of the checkbox.
            Returns:
                QCheckBox: The configured QCheckBox widget.
            """
            cb = QCheckBox()
            cb.setChecked(checked)
            return cb

        def wrap_widget(widget: QWidget) -> QWidget:
            """
            Wraps a given widget in a QHBoxLayout to remove extra margins
            and control its stretching behavior.
            Args:
                widget (QWidget): The widget to be wrapped.
            Returns:
                QWidget: A container widget holding the wrapped widget.
            """
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            layout.setStretch(0, 1)
            return container

        def save_post_setting(key: str, value: Any) -> None:
            """
            Saves a post-processing setting to the `self.post_settings` dictionary.
            Args:
                key (str): The key (name) of the post-processing setting.
                value (Any): The value to be saved for the given key.
            """

            if not hasattr(self, 'post_settings'):
                self.post_settings = {}
            self.post_settings[key] = value

        # ---------------------------------------------
        # GROUP HEADER: Post-Processing Settings
        # ---------------------------------------------

        group = QTreeWidgetItem(self.details_tree)
        postprocess_font = QFont("Arial", 12, QFont.Weight.Bold)
        group.setFont(0, postprocess_font)
        group.setFirstColumnSpanned(True)
        group.setText(0, "Post-Processing Settings")
        group.setBackground(0, color_heading)
        self.details_tree.addTopLevelItem(group)

        # ---------------------------------------------
        # PROPERTY ROWS: Editable values
        # ---------------------------------------------

        settings = [
            ("Display Type", create_dropdown(["Contour", "Deformed Shape", "Vector Field"], "Contour"), "", "display_type"),
            ("Result to Display", create_dropdown(["Displacement", "Force", "Stress"], "Displacement"), "", "result_type"),
            ("Scale Factor", create_line_edit(100.0), "x", "scale_factor"),
            ("Show Undeformed Mesh", create_checkbox(True), "", "show_undeformed"),
            ("Auto Update on Mesh Change", create_checkbox(True), "", "auto_update"),
            ("Default Tab", create_dropdown(["Geometry", "Stiffness-Matrix", "Displacements", "Forces", "Stresses"], "Displacements"), "", "default_tab"),
        ]

        for name, widget, unit, key in settings:
            item = QTreeWidgetItem()
            item.setText(0, name)
            item.setBackground(0, color_items)
            item.setBackground(1, color_items)
            group.addChild(item)
            self.details_tree.setItemWidget(item, 1, wrap_widget(widget))

            if isinstance(widget, QComboBox):

                if key == "default_tab":
                    widget.currentTextChanged.connect(lambda label: self.select_tab_by_label(label))
                widget.currentTextChanged.connect(lambda val, k=key: save_post_setting(k, val))
            elif isinstance(widget, QLineEdit):
                widget.editingFinished.connect(lambda k=key, w=widget: save_post_setting(k, w.text()))
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(lambda state, k=key: save_post_setting(k, bool(state)))
        self.details_tree.expandAll()

    @staticmethod
    def get_constraint_type(node_data: Dict[str, Any], structure_info: Dict[str, Any]) -> str:
        """
        Determines the constraint type (e.g., "Fixed", "Pinned", "Roller(x)", "Free", "Custom")

        for a given node based on its displacement boundary conditions and the
        overall structure's element type and dimension.
        Args:
            node_data (Dict[str, Any]): The node data dictionary, containing 'displacement'.
            structure_info (Dict[str, Any]): Information about the structure, including
                                              'element_type' and 'dimension'.
        Returns:
            str: The identified constraint type as a string.
        """
        displacement_raw = node_data.get("displacement", [np.nan] * 6)
        displacement = list(displacement_raw) if isinstance(displacement_raw, (list, tuple)) else [np.nan] * 6
        element_type = structure_info.get("element_type", "General")
        dimension = structure_info.get("dimension")
        num_dofs_structure = 6

        if dimension == "2D":

            if element_type in ["2D_Truss"]:
                num_dofs_structure = 2
                displacement = displacement[:2] + [np.nan]*(6-2)
            elif element_type in ["2D_Beam", "2D_Plane"]:
                num_dofs_structure = 3
                displacement = displacement[:3] + [np.nan]*(6-3)
        elif dimension == "3D":

            if element_type == "3D_Truss":
                num_dofs_structure = 3
                displacement = displacement[:3] + [np.nan]*(6-3)
            elif element_type in ["3D_Frame", "3D_Solid"]:
                num_dofs_structure = 6

        if len(displacement) < 6:
            displacement.extend([np.nan] * (6 - len(displacement)))
        elif len(displacement) > 6:
            displacement = displacement[:6]


        def is_fixed_dofs(dofs_indices: list[int]) -> bool:
            return all(not np.isnan(displacement[i]) and displacement[i] == 0 for i in dofs_indices)

        def is_free_dofs(dofs_indices: list[int]) -> bool:
            return all(np.isnan(displacement[i]) for i in dofs_indices)

        if is_free_dofs(list(range(6))):
            return "Free"

        if is_fixed_dofs(list(range(6))):
            return "Fixed"

        if dimension == "2D":

            if element_type == "2D_Beam":

                if is_fixed_dofs([0, 1]) and is_free_dofs([2]):
                    return "Pinned"

                elif is_fixed_dofs([0]) and is_free_dofs([1, 2]):
                    return "Roller(x)"

                elif is_fixed_dofs([1]) and is_free_dofs([0, 2]):
                    return "Roller(y)"

            elif element_type == "2D_Truss":

                if is_fixed_dofs([0, 1]) and is_free_dofs([2,3,4,5]):
                    return "Pinned"

                elif is_fixed_dofs([0]) and is_free_dofs([1,2,3,4,5]):
                    return "Roller(x)"

                elif is_fixed_dofs([1]) and is_free_dofs([0,2,3,4,5]):
                    return "Roller(y)"

            elif element_type == "2D_Solid":

                if is_fixed_dofs([0]) and is_free_dofs([1,2,3,4,5]):
                    return "Roller(x)"

                elif is_fixed_dofs([1]) and is_free_dofs([0,2,3,4,5]):
                    return "Roller(y)"

                elif is_fixed_dofs([2]) and is_free_dofs([0,1,3,4,5]):
                    return "Roller(z)"

        elif dimension == "3D":

            if element_type == "3D_Frame":

                if is_fixed_dofs([0, 1, 2]) and is_free_dofs([3, 4, 5]):
                    return "Pinned"

                elif is_fixed_dofs([0]) and is_free_dofs([1, 2, 3, 4, 5]):
                    return "Roller(x)"

                elif is_fixed_dofs([1]) and is_free_dofs([0, 2, 3, 4, 5]):
                    return "Roller(y)"

                elif is_fixed_dofs([2]) and is_free_dofs([0, 1, 3, 4, 5]):
                    return "Roller(z)"

            elif element_type == "3D_Truss":

                if is_fixed_dofs([0, 1, 2]) and is_free_dofs([3,4,5]):
                    return "Fixed"

                elif is_fixed_dofs([0, 1]) and is_free_dofs([2,3,4,5]):
                    return "Pinned"

                elif is_fixed_dofs([0]) and is_free_dofs([1, 2, 3, 4, 5]):
                    return "Roller(x)"

                elif is_fixed_dofs([1]) and is_free_dofs([0, 2, 3, 4, 5]):
                    return "Roller(y)"

                elif is_fixed_dofs([2]) and is_free_dofs([0, 1, 3, 4, 5]):
                    return "Roller(z)"

            elif element_type == "3D_Solid":

                if is_fixed_dofs([0]) and is_free_dofs([1, 2, 3, 4, 5]):
                    return "Roller(x)"

                elif is_fixed_dofs([1]) and is_free_dofs([0, 2, 3, 4, 5]):
                    return "Roller(y)"

                elif is_fixed_dofs([2]) and is_free_dofs([0, 1, 3, 4, 5]):
                    return "Roller(z)"

        return "Custom"

    def update_imported_data(self) -> None:
        """
        Updates the `nodal_displacements` and `concentrated_loads` dictionaries
        within `self.imported_data` based on the current values stored in
        `self.imported_data['nodes']`. This ensures consistency between different
        representations of boundary conditions and loads.
        """
        displacement_nodes_to_keep = self.imported_data['nodal_displacements'].keys()
        force_nodes_to_keep = self.imported_data['concentrated_loads'].keys()
        self.imported_data['nodal_displacements'] = {}
        self.imported_data['concentrated_loads'] = {}
        nodes_to_remove = []

        for node_id, node_data in list(self.imported_data['nodes'].items()):
            displacement = node_data.get('displacement', ())
            force = node_data.get('force', ())
            has_coordinates = 'X' in node_data or 'Y' in node_data or 'Z' in node_data

            if not has_coordinates:

                if node_id == 0:
                    nodes_to_remove.append(node_id)
                    continue

            if any(not np.isnan(val) for val in displacement):
                self.imported_data['nodal_displacements'][node_id] = displacement

            if any(not np.isnan(val) and val != 0 for val in force):
                self.imported_data['concentrated_loads'][node_id] = force

        for node_id in nodes_to_remove:
            del self.imported_data['nodes'][node_id]


    def update_imported_data0000(self) -> None:
        """
        Updates the `nodal_displacements` and `concentrated_loads` dictionaries
        within `self.imported_data` based on the current values stored in
        `self.imported_data['nodes']`. This ensures consistency between different
        representations of boundary conditions and loads.
        Existing keys in these dictionaries are preserved unless overwritten by new data.
        """
        updated_displacements = self.imported_data.get('nodal_displacements', {}).copy()
        updated_forces = self.imported_data.get('concentrated_loads', {}).copy()
        nodes_to_remove = []

        for node_id, node_data in list(self.imported_data['nodes'].items()):
            displacement = node_data.get('displacement', ())
            force = node_data.get('force', ())

            if node_id == 0:
                nodes_to_remove.append(node_id)
                continue

            if any(not np.isnan(val) for val in displacement):
                updated_displacements[node_id] = displacement

            if any(not np.isnan(val) and val != 0 for val in force):
                updated_forces[node_id] = force
        self.imported_data['nodal_displacements'] = updated_displacements
        self.imported_data['concentrated_loads'] = updated_forces

        for node_id in nodes_to_remove:
            del self.imported_data['nodes'][node_id]

    # ---------------------------------------------
    # TREE WIDGET MANAGEMENT
    # ---------------------------------------------

    def populate_tree(self) -> None:
        """
        Populates the main `tree_widget` with data from `self.imported_data`.
        This method delegates the actual population logic to the `TreeWidget` class.
        """
        self.tree_widget.populate_tree(self.imported_data)


    def clear_tree_widget_items(self) -> None:
        """
        Clears all items from the `details_tree` QTreeWidget and also
        clears any QComboBoxes or QLineEdits that might be present in the items.
        This handles clearing the *contents* of the widgets within the tree,
        not just the tree items themselves.
        """
        items: List[QTreeWidgetItem] = [
            self.details_tree.topLevelItem(i)

            for i in range(self.details_tree.topLevelItemCount())
        ]

        for item in items:
            self.clear_tree_widget_item(item, self.details_tree)
        self.details_tree.clear()
        self.details_label.setHidden(False)
        gc.collect() 


    def clear_tree_widget_item(self, item: QTreeWidgetItem, tree_widget: QTreeWidget) -> None:
        """
        Recursively clears a QTreeWidgetItem and any QComboBoxes or QLineEdits within it.
        Args:
            item (QTreeWidgetItem): The QTreeWidgetItem to clear.
            tree_widget (QTreeWidget): The QTreeWidget containing the item.
        """

        for column in range(tree_widget.columnCount()):

            try:
                widget: Optional[QWidget] = tree_widget.itemWidget(item, column)

                if widget:
                    with QSignalBlocker(widget):

                        if isinstance(widget, QComboBox):
                            widget.clear()
                            widget.setCurrentIndex(-1)
                        elif isinstance(widget, QLineEdit):
                            widget.clear()
                    tree_widget.removeItemWidget(item, column)
                    widget.deleteLater()

            except RuntimeError as e:
                print(f"Error accessing widget: {e}")
                continue
        children: List[QTreeWidgetItem] = [item.child(i) for i in range(item.childCount())]

        for child_item in children:
            self.clear_tree_widget_item(child_item, tree_widget)


    def add_tree_item(self, index: int = 0, item_text: str = "", parent_text: Optional[str] = None,
                      item_type: str = "Node", section_code: str = "", withoutIndex: bool = False) -> None:
        """
        Adds an item to the QTreeWidget under a specified parent and selects it.
        Args:
            index (int): The node or element number to include in the item text
                         (e.g., 2 in 'Node 2: XXX'). Defaults to 0.
            item_text (str): The full text to display for the new tree item.
                             Defaults to "".
            parent_text (Optional[str]): The parent item text under which the new item should be added.
                                         If None, the item is added at the top level. Defaults to None.
            item_type (str): The type of item being added. Must be one of
                             "Node", "Element", "Cross Section", or "Material".
                             Defaults to "Node".
            section_code (str): The section code associated with the item, used for Cross Sections
                                and Materials. Defaults to "".
            withoutIndex (bool): If True, the `index` is not prepended to the `item_text`.
                                 Defaults to False.
        Raises:
            ValueError: If `item_type` is not one of the allowed values.
        """

        if item_type not in ["Node", "Element", "Cross Section", "Material"]:
            raise ValueError("item_type must be 'Node', 'Element', 'Cross Section', or 'Material'")

        if withoutIndex:
            target_text = item_text
        else:
            target_text = f"{item_type} {index}: {item_text}"
        new_item: Optional[QTreeWidgetItem] = None
        load_type: Optional[str] = None

        if parent_text:
            # ---------------------------------------------
            # DETERMINE TOP-LEVEL PARENT
            # ---------------------------------------------

            if parent_text in ["Nodes", "Elements", "Cross Sections"]:
                top_level_parent_text = "Geometry"
            elif parent_text in ["Nodal Forces", "Nodal Displacements", "Distributed Loads"]:
                top_level_parent_text = "Boundary Conditions"
            else:
                top_level_parent_text = parent_text

            if top_level_parent_text:

                # ---------------------------------------------
                # SEARCH FOR TOP-LEVEL PARENT
                # ---------------------------------------------

                for i in range(self.tree_widget.topLevelItemCount()):
                    top_level_item: QTreeWidgetItem = self.tree_widget.topLevelItem(i)

                    if top_level_item.text(0) == top_level_parent_text:

                        # ---------------------------------------------
                        # SEARCH FOR SPECIFIC PARENT UNDER TOP-LEVEL PARENT
                        # ---------------------------------------------

                        parent_item: Optional[QTreeWidgetItem] = self.find_child_item(top_level_item, parent_text)

                        if parent_item is None:
                            parent_item = top_level_item
                        new_item = QTreeWidgetItem(parent_item)
                        new_item.setText(0, target_text)

                        # ---------------------------------------------
                        # SET ITEM DATA BASED ON PARENT TEXT
                        # ---------------------------------------------

                        if parent_text == "Nodes":
                            load_type = 'node'
                        elif parent_text == "Elements":
                            load_type = 'element'
                        elif parent_text == "Cross Sections":
                            load_type = 'cross_section'
                        elif parent_text == "Nodal Forces":
                            load_type = 'concentrated_load'
                        elif parent_text == "Distributed Loads":
                            load_type = 'distributed_load'
                        elif parent_text == "Materials":
                            load_type = 'material'
                        item_data: Dict[str, Any] = {"type": load_type, "id": index}

                        if parent_text in ["Cross Sections", "Materials"]:
                            item_data["code"] = section_code
                        new_item.setData(0, Qt.ItemDataRole.UserRole, item_data)
                        parent_item.addChild(new_item)
                        break
            else:
                print(f"Warning: Could not determine top-level parent for '{parent_text}'. Adding to top level.")
                new_item = QTreeWidgetItem(self.tree_widget)
                new_item.setText(0, target_text)
                self.tree_widget.addTopLevelItem(new_item)

        if new_item:

            # ---------------------------------------------
            # ENSURE NEWLY ADDED ITEM IS SELECTED
            # ---------------------------------------------

            self.tree_widget.setCurrentItem(new_item)


    def find_child_item(self, parent_item: QTreeWidgetItem, child_text: str) -> Optional[QTreeWidgetItem]:
        """
        Recursively finds a child item with the specified text under a parent item.
        Args:
            parent_item (QTreeWidgetItem): The parent item to start the search from.
            child_text (str): The text of the child item to find.
        Returns:
            Optional[QTreeWidgetItem]: The found child item, or None if not found.
        """

        for i in range(parent_item.childCount()):
            child_item: QTreeWidgetItem = parent_item.child(i)

            if child_item.text(0) == child_text:
                return child_item

            found_item: Optional[QTreeWidgetItem] = self.find_child_item(child_item, child_text)

            if found_item:
                return found_item

        return None

    def remove_tree_item(self, index: int = 0, item_text: str = "", parent_text: Optional[str] = None,
                         item_type: str = "Node", withoutIndex: bool = False) -> None:
        """
        Removes an item from the QTreeWidget based on similar parameters to add_tree_item.
        Args:
            index (int): The node or element number to search for (e.g., 2 in 'Node 2: XXX' or 'Element 2: XXX').
                         Defaults to 0.
            item_text (str): The specific text part to match (excluding the index prefix if withoutIndex is False).
                             Defaults to "".
            parent_text (Optional[str]): The parent item text to restrict the search within. Defaults to None.
            item_type (str): Either "Node", "Element", or "Material" to specify what to search for.
                             Defaults to "Node".
            withoutIndex (bool): If True, search for the exact `item_text`; otherwise, search for
                                 "{item_type} {index}: {item_text}". Defaults to False.
        Raises:
            ValueError: If `item_type` is not "Node", "Element", or "Material".
        """

        if item_type not in ["Node", "Element", "Material"]:
            raise ValueError("item_type must be either 'Node' or 'Element'")

        if withoutIndex:
            target_text: str = item_text
        else:
            target_text = f"{item_type} {index}:"

        if parent_text:

            # ---------------------------------------------
            # DETERMINE TOP-LEVEL PARENT
            # ---------------------------------------------

            if parent_text in ["Nodes", "Elements", "Cross Sections"]:
                top_level_parent_text = "Geometry"
            elif parent_text in ["Nodal Forces", "Nodal Displacements", "Distributed Loads"]:
                top_level_parent_text = "Boundary Conditions"
            else:
                top_level_parent_text = parent_text

            if top_level_parent_text:

                # ---------------------------------------------
                # SEARCH FOR TOP-LEVEL PARENT
                # ---------------------------------------------

                for i in range(self.tree_widget.topLevelItemCount()):
                    top_level_item: QTreeWidgetItem = self.tree_widget.topLevelItem(i)

                    if top_level_item.text(0) == top_level_parent_text:

                        # ---------------------------------------------
                        # SEARCH FOR SPECIFIC PARENT UNDER TOP-LEVEL PARENT
                        # ---------------------------------------------

                        parent_item: Optional[QTreeWidgetItem] = self.find_child_item(top_level_item, parent_text)

                        if parent_item:

                            # ---------------------------------------------
                            # SEARCH FOR AND REMOVE ITEM WITHIN THIS PARENT
                            # ---------------------------------------------

                            for j in range(parent_item.childCount()):
                                child_item: QTreeWidgetItem = parent_item.child(j)

                                if child_item.text(0).startswith(target_text):
                                    parent_item.removeChild(child_item)
                                    del child_item
                                    return

                        return

                return

        else:

            # ---------------------------------------------
            # SEARCH THE ENTIRE TREE (ALL PARENTS & CHILDREN)
            # ---------------------------------------------

            for i in range(self.tree_widget.topLevelItemCount()):
                top_level_item = self.tree_widget.topLevelItem(i)

                for j in range(top_level_item.childCount()):
                    child_item = top_level_item.child(j)

                    if child_item.text(0).startswith(target_text):
                        top_level_item.removeChild(child_item)
                        del child_item
                        return

    def update_dimensions_tree(self, section_code: str, section_type: str) -> None:
        """
        Updates the dimensions displayed in the `dimensions_group` within the details tree.
        Args:
            section_code (str): The code of the cross-section.
            section_type (str): The type of the cross-section (e.g., "Solid_Circular").
        """

        # ---------------------------------------------
        # CLEAR OLD DIMENSION ITEMS
        # ---------------------------------------------

        self.dimensions_group.takeChildren()
        param_units: str = self.imported_data['saved_units']["Position (X,Y,Z)"]

        # ---------------------------------------------
        # DEFINE SECTION PARAMETERS BY TYPE
        # ---------------------------------------------

        section_params: Dict[str, list[str]] = {
            "Solid_Circular": ["D"],
            "Hollow_Circular": ["D", "d"],
            "Solid_Rectangular": ["B", "H", "angle"],
            "Hollow_Rectangular": ["B", "H", "b", "h", "angle"],
            "I_Beam": ["B", "H", "tf", "tw", "angle"],
            "C_Beam": ["B", "H", "tf", "tw", "angle"],
            "L_Beam": ["B", "H", "tf", "tw", "angle"]
        }
        raw_dimensions: Any = self.imported_data["cross_sections"][section_code].get("dimensions", {})

        # ---------------------------------------------
        # NORMALIZE DIMENSIONS TO A DICTIONARY
        # ---------------------------------------------

        param_list: list[str] = section_params.get(section_type, [])

        if isinstance(raw_dimensions, dict):
            dimensions: Dict[str, Any] = raw_dimensions
        elif isinstance(raw_dimensions, (tuple, list)):
            dimensions = dict(zip(param_list, raw_dimensions))
        elif isinstance(raw_dimensions, (int, float)) and param_list:
            dimensions = {param_list[0]: raw_dimensions}
        else:
            dimensions = {}
        self.imported_data["cross_sections"][section_code]["dimensions"] = copy.deepcopy(dimensions)

        # ---------------------------------------------
        # ADD ITEMS TO TREE
        # ---------------------------------------------

        for param in param_list:
            value: Any = dimensions.get(param, "")

            if param == "angle":
                unit: str = "°"
            else:
                unit = param_units
            item: QTreeWidgetItem = QTreeWidgetItem(self.dimensions_group, [param, str(value), unit])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)

    # ---------------------------------------------
    # UTILITY METHODS
    # ---------------------------------------------

    def update_dictionary_key_ordered(self, data: Dict[Any, Any], old_key: Any, new_key: Any) -> Dict[Any, Any]:
        """
        Static Method:
        Updates a dictionary key while preserving the original insertion order.
        Args:
            data (Dict[Any, Any]): The input dictionary.
            old_key (Any): The key to be replaced.
            new_key (Any): The new key.
        Returns:
            Dict[Any, Any]: A new dictionary with the updated key, maintaining order.
        """

        if old_key not in data:
            return data

        new_data: Dict[Any, Any] = {}

        for key, value in data.items():

            if key == old_key:
                new_data[new_key] = value
            else:
                new_data[key] = value
        return new_data

    def select_tab_by_label(self, label: str) -> None:
        """
        Selects a specific tab in the central dock's QTabWidget based on its label.
        Args:
            label (str): The text label of the tab to be selected.
        Returns:
            None  
        """
        label_to_widget: Dict[str, QWidget] = {
            "Geometry": self.central_dock.frame_canvas.parent(),
            "Stiffness-Matrix": self.central_dock.stiffness_tab,
            "Displacements": self.central_dock.deformation_canvas.parent().parent(),
            "Forces": self.central_dock.force_canvas.parent(),
            "Stresses": self.central_dock.stress_canvas.parent(),
            "Information": self.central_dock.info_canvas.parent(),
        }

        for i in range(self.central_dock.tab_widget.count()):
            widget: QWidget = self.central_dock.tab_widget.widget(i)

            if label in label_to_widget:
                target_widget: QWidget = label_to_widget[label]

                if target_widget == widget:
                    self.central_dock.tab_widget.setCurrentIndex(i)
                    break


    def update_boundary_condition(self, node_id: str, selected_constraint: str) -> Dict[str, Any]:
        """
        Updates the boundary conditions (force and displacement) for a specific node.
        This method retrieves the appropriate force and displacement values based
        on the `selected_constraint` type and updates the corresponding node data
        within `self.imported_data`.
        Args:
            node_id (str): The unique identifier of the node whose boundary conditions are to be updated.
            selected_constraint (str): The type of boundary constraint selected (e.g., "Fixed", "Pinned").
        Returns:
            Dict[str, Any]: The updated node data dictionary.
        """
        node_data: Dict[str, Any] = self.imported_data['nodes'][node_id]
        structure_info: Dict[str, Any] = self.imported_data['structure_info']
        updated_values: Dict[str, Tuple[float, ...]] = self.get_constraint_values(selected_constraint, structure_info)
        node_data['force'] = updated_values['force']
        node_data['displacement'] = updated_values['displacement']
        return node_data


    @staticmethod
    def get_constraint_values(constraint_type: str, structure_info: Dict[str, Any]) -> Dict[str, Tuple[float, ...]]:
        """
        Static Method: Returns the expected force and displacement values for a given constraint type.
        This static method determines the default force and displacement values
        (e.g., 0.0 for fixed, NaN for free) based on the specified
        `constraint_type` and the `structure_info` (which provides force and
        displacement labels).
        Args:
            constraint_type (str): The type of boundary constraint (e.g., "Fixed", "Pinned", "Roller(u)").
            structure_info (Dict[str, Any]): The structure information including force and displacement labels.
        Returns:
            Dict[str, Tuple[float, ...]]: A dictionary with 'force' and 'displacement' tuples,
                                          where each tuple contains float or numpy.nan values.
        """
        force_labels: list[str] = structure_info['force_labels']
        displacement_labels: list[str] = structure_info['displacement_labels']
        force_values: Tuple[float, ...] = tuple(np.nan for _ in force_labels)
        displacement_values: Tuple[float, ...] = tuple(np.nan for _ in displacement_labels)
        translational_labels: list[str] = [d for f, d in zip(force_labels, displacement_labels) if 'M' not in f]
        rotational_labels: list[str] = [d for f, d in zip(force_labels, displacement_labels) if 'M' in f]

        if constraint_type == "Fixed":
            displacement_values = tuple(0.0 for _ in displacement_labels)
        elif constraint_type == "Pinned":
            displacement_values = tuple(0.0 if label in {'u', 'v', 'w'} else np.nan for label in displacement_labels)
            force_values = tuple(0.0 if label in {'Mx', 'My', 'Mz'} else np.nan for label in force_labels)
        elif constraint_type.startswith("Roller"):
            roller_map: Dict[str, str] = {"Roller(x)": "u", "Roller(y)": "v", "Roller(z)": "w"}
            roller_dof: Optional[str] = roller_map.get(constraint_type)
            roller_displacement: Tuple[float, ...] = tuple(np.nan if d == roller_dof else 0.0 for d in translational_labels)
            displacement_values = roller_displacement + tuple(np.nan for _ in displacement_labels[len(translational_labels):])
            force_values = tuple(0.0 if label in {'Mx', 'My', 'Mz'} else np.nan for label in force_labels)
        elif constraint_type == "Free":
            force_values = tuple(0.0 for _ in force_labels)
        elif constraint_type == "Custom":
            pass

        return {'force': force_values, 'displacement': displacement_values}

    def get_node_data(self, constraint_type: str) -> Dict[str, Any]:
        """
        Returns a dictionary representing default node data, including coordinates
        and initial boundary condition values based on the `constraint_type`.
        Args:
            constraint_type (str): The type of boundary constraint to apply
                                   (e.g., "Fixed", "Pinned", "Free").
        Returns:
            Dict[str, Any]: A dictionary containing default 'X', 'Y', 'Z' coordinates
                            (if 3D) and 'force' and 'displacement' boundary condition values.
        """
        structure_info: Dict[str, Any] = self.imported_data["structure_info"]
        is_3d: bool = structure_info["dimension"] == "3D"
        constraint_values: Dict[str, Tuple[float, ...]] = self.get_constraint_values(constraint_type, structure_info)
        node_data: Dict[str, Any]

        if is_3d:
            node_data = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, **constraint_values}
        else:
            node_data = {'X': 0.0, 'Y': 0.0, **constraint_values}
        return node_data