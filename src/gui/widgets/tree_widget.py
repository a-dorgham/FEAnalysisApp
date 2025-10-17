from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QMenu
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QPoint
from src.core.solvers.solvers import MainSolver
from src.gui.viewers.results import ResultsViewer
from src.core.structure_io import ProcessImportedStructure
import numpy as np
import re
import copy
from typing import Dict, Any, Optional, Tuple, Set, Union, List


class TreeWidget(QTreeWidget):
    """
    A custom QTreeWidget that provides a right-click context menu with various actions

    for managing structural analysis data, including nodes, elements, materials,
    cross-sections, boundary conditions, and solver/post-processing options.
    This widget interacts with `imported_data` (a dictionary holding structural model data)
    and `LeftDockWindow` for displaying detailed information and managing the UI.
    """


    def __init__(self, imported_data: Dict[str, Any], left_dock_window: QDockWidget, parent: Optional[QWidget] = None):
        """
        Initializes the TreeWidget.
        Args:
            imported_data (Dict[str, Any]): A dictionary containing the imported
                                             structural analysis data (nodes, elements, etc.).
            left_dock_window (QDockWidget): Reference to the LeftDockWindow for UI interactions.
            parent (Optional[QWidget]): The parent widget of this tree widget.
        """
        super().__init__(parent)

        # ---------------------------------------------
        # WIDGET SETUP
        # ---------------------------------------------
        self.setIndentation(20)         
        self.setItemsExpandable(True)    
        self.setRootIsDecorated(True)   
        self.customContextMenuRequested.connect(self.show_context_menu)

        # ---------------------------------------------
        # DATA AND REFERENCES
        # ---------------------------------------------

        self.imported_data: Dict[str, Any] = imported_data
        self.left_dock_window: QDockWidget = left_dock_window
        self.setup_docks = self.left_dock_window.setup_docks


    def populate_tree(self, imported_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Populates the tree widget with structural analysis data. Clears existing items
        and rebuilds the tree based on the provided or stored `imported_data`.
        Args:
            imported_data (Optional[Dict[str, Any]]): New structural data to populate the tree.
                                                      If None, uses the instance's `imported_data`.
            node_id_old (Optional[int]): An optional old node ID, used for selection
                                         restoration (deprecated).
        """
        self.clear()

        if imported_data:
            self.imported_data = imported_data

        # ---------------------------------------------
        # CREATE MAIN CATEGORIES
        # ---------------------------------------------

        geometry_item: QTreeWidgetItem = QTreeWidgetItem(self, ["Geometry"])
        materials_item: QTreeWidgetItem = QTreeWidgetItem(self, ["Materials"])
        materials_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "materials"})
        boundary_conditions_item: QTreeWidgetItem = QTreeWidgetItem(self, ["Boundary Conditions"])
        meshing_item: QTreeWidgetItem = QTreeWidgetItem(self, ["Meshing"])
        solver_item: QTreeWidgetItem = QTreeWidgetItem(self, ["Solver"])
        post_processing_item: QTreeWidgetItem = QTreeWidgetItem(self, ["Post-Processing"])

        # ---------------------------------------------
        # ADD NODES TO GEOMETRY
        # ---------------------------------------------

        nodes_item: QTreeWidgetItem = QTreeWidgetItem(geometry_item, ["Nodes"])
        nodes_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "nodes"})

        for node_id, node_data in self.imported_data['nodes'].items():

            if node_id == 0:
                continue
            node_text: str = f"Node {node_id}: ({', '.join([str(node_data.get(coord, '')) for coord in ['X', 'Y', 'Z']])})"
            node_item: QTreeWidgetItem = QTreeWidgetItem(nodes_item, [node_text])
            node_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "node", "id": node_id})

        # ---------------------------------------------
        # ADD ELEMENTS TO GEOMETRY
        # ---------------------------------------------

        elements_item: QTreeWidgetItem = QTreeWidgetItem(geometry_item, ["Elements"])
        elements_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "elements"})

        for elem_id, elem_data in self.imported_data['elements'].items():

            if elem_id == 0:
                continue
            section_code: str = elem_data["section_code"]
            material_code: str = elem_data["material_code"]
            element_text: str = f"Element {elem_id}: ({material_code}, {section_code})"
            elem_item: QTreeWidgetItem = QTreeWidgetItem(elements_item, [element_text])
            elem_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "element", "id": elem_id})

        # ---------------------------------------------
        # ADD CROSS SECTIONS TO GEOMETRY
        # ---------------------------------------------

        cross_sections_item: QTreeWidgetItem = QTreeWidgetItem(geometry_item, ["Cross Sections"])
        cross_sections_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "cross_sections"})

        for code, section_data in self.imported_data['cross_sections'].items():
            section_text: str = f"{code}"
            section_item: QTreeWidgetItem = QTreeWidgetItem(cross_sections_item, [section_text])
            section_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "cross_section", "code": code})

        # ---------------------------------------------
        # ADD NODAL DISPLACEMENTS TO BOUNDARY CONDITIONS
        # ---------------------------------------------

        disps_item: QTreeWidgetItem = QTreeWidgetItem(boundary_conditions_item, ["Nodal Displacements"])
        disps_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "nodal_displacements"})

        for node_id, disps in self.imported_data['nodal_displacements'].items():

            if node_id not in self.imported_data['nodes']:
                continue
            constraints: List[str] = [f"{self.imported_data['structure_info']['displacement_labels'][i]}: {disps[i]}"

                                    for i in range(len(disps)) if not np.isnan(disps[i])]

            if constraints:
                constraint: str = self.left_dock_window.get_constraint_type(self.imported_data['nodes'][node_id], self.imported_data['structure_info'])
                self.imported_data['nodes'][node_id]["constraint"] = constraint
                disp_item: QTreeWidgetItem = QTreeWidgetItem(disps_item, [f"Node {node_id}: {constraint}"])
                identifier: str = f"nodal_displacement{node_id}"
                disp_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "nodal_displacement", "id": node_id, "identifier": identifier})

        # ---------------------------------------------
        # ADD NODAL FORCES TO BOUNDARY CONDITIONS
        # ---------------------------------------------

        forces_item: QTreeWidgetItem = QTreeWidgetItem(boundary_conditions_item, ["Nodal Forces"])
        forces_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "concentrated_loads"})

        for node_id, forces in self.imported_data['concentrated_loads'].items():
            labels: List[str] = [
                f"{self.imported_data['structure_info']['force_labels'][i]}: {forces[i]}"

                for i in range(len(forces)) if not (np.isnan(forces[i]) or forces[i] == 0)
            ]

            if labels:
                force_item: QTreeWidgetItem = QTreeWidgetItem(forces_item, [f"Node {node_id}: {', '.join(labels)}"])
                force_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "concentrated_load", "id": node_id})

        # ---------------------------------------------
        # ADD DISTRIBUTED LOADS TO BOUNDARY CONDITIONS
        # ---------------------------------------------

        loads_item: QTreeWidgetItem = QTreeWidgetItem(boundary_conditions_item, ["Distributed Loads"])
        loads_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "distributed_loads"})

        for elem_id, load_data in self.imported_data['distributed_loads'].items():

            if elem_id:
                load_text: str = f"Element {elem_id}: {load_data['type']}"
                load_item: QTreeWidgetItem = QTreeWidgetItem(loads_item, [load_text])
                load_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "distributed_load", "id": elem_id})

        # ---------------------------------------------
        # ADD MATERIALS
        # ---------------------------------------------

        for code, _ in self.imported_data['materials'].items():
            section_text: str = f"{code}"
            section_item: QTreeWidgetItem = QTreeWidgetItem(materials_item, [section_text])
            section_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "material", "code": code})

        # ---------------------------------------------
        # ADD MESHING
        # ---------------------------------------------

        mesh_text: str = "Mesh Type: Automatic"
        mesh_item: QTreeWidgetItem = QTreeWidgetItem(meshing_item, [mesh_text])
        mesh_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "mesh"})

        # ---------------------------------------------
        # ADD SOLVER
        # ---------------------------------------------

        solver_text: str = "Solver Information"
        solver_item_sub: QTreeWidgetItem = QTreeWidgetItem(solver_item, [solver_text])
        solver_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "solvers"})
        solver_item_sub.setData(0, Qt.ItemDataRole.UserRole, {"type": "solver"})

        # ---------------------------------------------
        # ADD POST-PROCESSING
        # ---------------------------------------------

        post_processing_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "post_processings", "label": "post_processings"})
        post_processing_entries: List[Optional[Tuple[str, str]]] = [
            ("Geometry", "Geometry"),
            ("Stiffness Matrix", "Stiffness-Matrix"),
            ("Total Displacement", "Displacements"),
            ("von Mises Stress", "Stresses"),
            ("Internal Forces", "Forces") if self.imported_data['structure_info']['element_type'] not in ['2D_Truss', '3D_Truss'] else None,
            ("Solver Information", "Information")
        ]
        post_processing_entries = [entry for entry in post_processing_entries if entry is not None] 

        for display_label, internal_label in post_processing_entries: 
            post_item: QTreeWidgetItem = QTreeWidgetItem(post_processing_item, [f"{display_label}"])
            post_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "post_processing", "label": internal_label})

        # ---------------------------------------------
        # ADD TOP-LEVEL ITEMS
        # ---------------------------------------------

        self.addTopLevelItem(geometry_item)
        self.addTopLevelItem(boundary_conditions_item)
        self.addTopLevelItem(materials_item)
        self.addTopLevelItem(meshing_item)
        self.addTopLevelItem(solver_item)
        self.addTopLevelItem(post_processing_item)


    def show_context_menu(self, position: QPoint) -> None:
        """
        Displays a right-click context menu with options based on the clicked item.
        Args:
            position (QPoint): The global position of the mouse click.
        """
        item: Optional[QTreeWidgetItem] = self.itemAt(position)

        if not item:
            print('Not an item clicked.')
            return

        item_data: Optional[Dict[str, Any]] = item.data(0, Qt.ItemDataRole.UserRole)

        if not item_data:
            print('No item data found for the clicked item.')
            return

        menu: QMenu = QMenu(self)

        # ---------------------------------------------
        # CONTEXT MENU FOR MATERIALS
        # ---------------------------------------------

        if item_data["type"] == "materials":
            add_material_action: QAction = QAction("Add Material", self)
            delete_all_materials_action: QAction = QAction("Delete ALL Materials", self)
            menu.addAction(add_material_action)
            menu.addAction(delete_all_materials_action)
            add_material_action.triggered.connect(lambda: self.add_material()) 
            delete_all_materials_action.triggered.connect(lambda: self.delete_all_materials()) 
        elif item_data["type"] == "material":
            add_material_action: QAction = QAction("Add Material", self)
            delete_material_action: QAction = QAction("Delete Material", self)
            menu.addAction(add_material_action)
            menu.addAction(delete_material_action)
            add_material_action.triggered.connect(lambda: self.add_material(item_data["code"])) 
            delete_material_action.triggered.connect(lambda: self.delete_material(item_data["code"])) 

        # ---------------------------------------------
        # CONTEXT MENU FOR NODES
        # ---------------------------------------------

        elif item_data["type"] == "nodes":
            add_node_action: QAction = QAction("Add Node", self)
            delete_node_action: QAction = QAction("Delete ALL Nodes", self)
            menu.addAction(add_node_action)
            menu.addAction(delete_node_action)
            last_node_id: int = max(self.imported_data['nodes'].keys(), default=0)
            add_node_action.triggered.connect(lambda: self.add_node(last_node_id)) 
            delete_node_action.triggered.connect(lambda: self.delete_all_nodes()) 
        elif item_data["type"] == "node":
            add_node_action: QAction = QAction("Add Node", self)
            delete_node_action: QAction = QAction("Delete Node", self)
            menu.addAction(add_node_action)
            menu.addAction(delete_node_action)
            add_node_action.triggered.connect(lambda: self.add_node(item_data["id"])) 
            delete_node_action.triggered.connect(lambda: self.delete_node(item_data["id"])) 

        # ---------------------------------------------
        # CONTEXT MENU FOR ELEMENTS
        # ---------------------------------------------

        elif item_data["type"] == "elements":
            add_element_action: QAction = QAction("Add Element", self)
            delete_element_action: QAction = QAction("Delete ALL Elements", self)
            menu.addAction(add_element_action)
            menu.addAction(delete_element_action)
            last_element_id: int = max(self.imported_data['elements'].keys(), default=0)
            add_element_action.triggered.connect(lambda: self.add_element(last_element_id)) 
            delete_element_action.triggered.connect(lambda: self.delete_all_elements()) 
        elif item_data["type"] == "element":
            add_element_action: QAction = QAction("Add Element", self)
            delete_element_action: QAction = QAction("Delete Element", self)
            menu.addAction(add_element_action)
            menu.addAction(delete_element_action)
            add_element_action.triggered.connect(lambda: self.add_element(item_data["id"])) 
            delete_element_action.triggered.connect(lambda: self.delete_element(item_data["id"])) 
            assign_material_menu: QMenu = QMenu("Assign Material", self)

            for material_code, material_data in self.imported_data.get("materials", {}).items():
                action: QAction = QAction(f"{material_code} ({material_data['type']})", self)
                action.triggered.connect(lambda checked, code=material_code: self.assign_material_to_element(item_data, code)) 
                assign_material_menu.addAction(action)
            menu.addMenu(assign_material_menu)
            assign_cs_menu: QMenu = QMenu("Assign C/S", self)

            for cs_code, cs_data in self.imported_data.get("cross_sections", {}).items():
                action: QAction = QAction(f"{cs_code} ({cs_data['type']})", self)
                action.triggered.connect(lambda checked, code=cs_code: self.assign_cross_section_to_element(item_data, code)) 
                assign_cs_menu.addAction(action)
            menu.addMenu(assign_cs_menu)

        # ---------------------------------------------
        # CONTEXT MENU FOR CROSS SECTIONS
        # ---------------------------------------------

        elif item_data["type"] == "cross_sections":
            add_section_action: QAction = QAction("Add Cross Section", self)
            delete_all_sections_action: QAction = QAction("Delete ALL Cross Sections", self)
            menu.addAction(add_section_action)
            menu.addAction(delete_all_sections_action)
            add_section_action.triggered.connect(lambda: self.add_cross_section()) 
            delete_all_sections_action.triggered.connect(lambda: self.delete_all_cross_sections()) 
        elif item_data["type"] == "cross_section":
            add_section_action: QAction = QAction("Add Cross Section", self)
            delete_section_action: QAction = QAction("Delete Cross Section", self)
            menu.addAction(add_section_action)
            menu.addAction(delete_section_action)
            add_section_action.triggered.connect(lambda: self.add_cross_section(item_data["code"], add_section=True)) 
            delete_section_action.triggered.connect(lambda: self.delete_cross_section(item_data["code"])) 

        # ---------------------------------------------
        # CONTEXT MENU FOR NODAL DISPLACEMENTS
        # ---------------------------------------------

        elif item_data["type"] == "nodal_displacements":
            add_constraint_action: QAction = QAction("Add Constraint", self)
            delete_all_constraints_action: QAction = QAction("Delete ALL Constraints", self)
            menu.addAction(add_constraint_action)
            menu.addAction(delete_all_constraints_action)
            add_constraint_action.triggered.connect(lambda: self.add_node_displacement(show_node_id=True, new_constraint=True)) 
            delete_all_constraints_action.triggered.connect(lambda: self.delete_all_nodal_displacements()) 
        elif item_data["type"] == "nodal_displacement":
            add_constraint_action: QAction = QAction("Add Constraint", self)
            delete_constraint_action: QAction = QAction("Delete Constraint", self)
            menu.addAction(add_constraint_action)
            menu.addAction(delete_constraint_action)
            add_constraint_action.triggered.connect(lambda: self.add_node_displacement(show_node_id=True, new_constraint=True)) 
            delete_constraint_action.triggered.connect(lambda: self.delete_node_displacement(item_data["id"])) 

        # ---------------------------------------------
        # CONTEXT MENU FOR CONCENTRATED LOADS
        # ---------------------------------------------

        elif item_data["type"] == "concentrated_loads":
            add_load_action: QAction = QAction("Add Load", self)
            delete_all_loads_action: QAction = QAction("Delete All Loads", self)
            menu.addAction(add_load_action)
            menu.addAction(delete_all_loads_action)
            add_load_action.triggered.connect(lambda: self.add_concentrated_load(show_node_id=True, new_constraint=True)) 
            delete_all_loads_action.triggered.connect(lambda: self.delete_all_concentrated_loads()) 
        elif item_data["type"] == "concentrated_load":
            add_point_load_action: QAction = QAction("Add Point Load", self)
            delete_load_action: QAction = QAction("Delete Load", self)
            menu.addAction(add_point_load_action)
            menu.addAction(delete_load_action)
            add_point_load_action.triggered.connect(lambda: self.add_concentrated_load(show_node_id=True, new_constraint=True)) 
            delete_load_action.triggered.connect(lambda: self.delete_concentrated_load(item_data["id"])) 

        # ---------------------------------------------
        # CONTEXT MENU FOR DISTRIBUTED LOADS
        # ---------------------------------------------

        elif item_data["type"] == "distributed_loads":
            add_distributed_load_action: QAction = QAction("Add Distributed Load", self)
            delete_all_loads_action: QAction = QAction("Delete All Loads", self)
            menu.addAction(add_distributed_load_action)
            menu.addAction(delete_all_loads_action)
            add_distributed_load_action.triggered.connect(lambda: self.add_distributed_load()) 
            delete_all_loads_action.triggered.connect(lambda: self.delete_all_distributed_loads()) 
        elif item_data["type"] == "distributed_load":
            add_load_action: QAction = QAction("Add Load", self)
            delete_load_action: QAction = QAction("Delete Load", self)
            menu.addAction(add_load_action)
            menu.addAction(delete_load_action)
            add_load_action.triggered.connect(lambda: self.add_distributed_load()) 
            delete_load_action.triggered.connect(lambda: self.delete_distributed_load(item_data["id"])) 

        # ---------------------------------------------
        # CONTEXT MENU FOR SOLVERS
        # ---------------------------------------------

        elif item_data["type"] in ["solvers", "solver"]:
            solve_action: QAction = QAction("Solve", self)
            menu.addAction(solve_action)
            solve_action.triggered.connect(lambda: self._main_solver())

        # ---------------------------------------------
        # CONTEXT MENU FOR POST-PROCESSING
        # ---------------------------------------------

        elif item_data["type"] in ["post_processings", "post_processing"]:
            re_solve_action: QAction = QAction("Re-Solve", self)
            menu.addAction(re_solve_action)
            re_solve_action.triggered.connect(lambda: self._main_solver())
            clear_solution_action: QAction = QAction("Clear Solution", self)
            menu.addAction(clear_solution_action)
            clear_solution_action.triggered.connect(lambda: self._clear_solution())
        menu.exec(self.viewport().mapToGlobal(position))


    def _remain_solver(self) -> None:
        """
        Clears the current solution (without updating the tree immediately) and then
        re-solves the structure.
        """
        self._clear_solution(update_tree=False)
        self._main_solver()


    def _reset_structure(self, update_tree: bool = True) -> None:
        """
        Clears the current solution by reverting to the backup imported data.
        Args:
            update_tree (bool): If True, the tree widget will be updated after clearing.
        """
        self.imported_data: Dict[str, Any] = copy.deepcopy(self.left_dock_window.imported_data_bk) 

        if update_tree:
            self.update_tree(self.imported_data)


    def _clear_solution(self, update_tree: bool = True) -> None:
        """
        Clears the current solution by reverting to the backup imported data.
        Args:
            update_tree (bool): If True, the tree widget will be updated after clearing.
        """
        self.setup_docks.reset_post_process_tabs(tabs_to_reset=[ "Stiffness-Matrix", "Displacements", "Stresses", "Forces", "Information"])


    def _main_solver(self) -> None:
        """
        Initiates the structural analysis process. Converts data to standard units,
        creates a `MainSolver` instance, and plots the solution if successful.
        Displays a solution report in the CentralDockWindow.
        """
        self.main_solver: MainSolver = MainSolver(
            imported_data=copy.deepcopy(self.imported_data),
            solver_settings=self.left_dock_window.solver_settings, 
            mesh_settings=self.left_dock_window.mesh_settings 
        )
        solution: Optional[Any]
        solution_report: str
        solution, solution_report = self.main_solver.solve()

        if solution:
            self.clear_dictionary_except_keys(ResultsViewer.deformation_fig_cache, [-1]) 
            ResultsViewer.plot_solution(solution, solution_valid=True) 
            ResultsViewer.CentralDockWindow.info_canvas.setHtml(solution_report) 

    @staticmethod
    def clear_dictionary_except_keys(dictionary: Dict[Any, Any], keys_to_keep: List[Any]) -> None:
        """
        Clears a dictionary of all keys except those specified in `keys_to_keep`.
        Args:
            dictionary (Dict[Any, Any]): The dictionary to modify.
            keys_to_keep (List[Any]): A list of keys that should remain in the dictionary.
        """
        keys_to_delete: List[Any] = [k for k in dictionary if k not in keys_to_keep]

        for k in keys_to_delete:
            del dictionary[k]


    def get_expanded_items(self, tree_widget: 'TreeWidget') -> Set[str]:
        """
        Recursively traverses the tree widget to identify and return a set of
        texts of all currently expanded items.
        Args:
            tree_widget (TreeWidget): The QTreeWidget instance to traverse.
        Returns:
            Set[str]: A set containing the text of all expanded items.
        """
        expanded_items: Set[str] = set()


        def traverse(item: QTreeWidgetItem) -> None:
            """Helper function to recursively traverse items."""

            if item.isExpanded():
                expanded_items.add(item.text(0))

            for i in range(item.childCount()):
                traverse(item.child(i))

        for i in range(tree_widget.topLevelItemCount()):
            traverse(tree_widget.topLevelItem(i))
        return expanded_items

    def restore_expansion_state(self, tree_widget: 'TreeWidget', expanded_items: Set[str]) -> None:
        """
        Restores the expansion state of items in the tree widget based on a provided
        set of expanded item texts.
        Args:
            tree_widget (TreeWidget): The QTreeWidget instance to modify.
            expanded_items (Set[str]): A set containing the text of items that should be expanded.
        """


        def traverse(item: QTreeWidgetItem) -> None:
            """Helper function to recursively traverse items and set expansion state."""

            if item.text(0) in expanded_items:
                item.setExpanded(True)

            for i in range(item.childCount()):
                traverse(item.child(i))

        for i in range(tree_widget.topLevelItemCount()):
            traverse(tree_widget.topLevelItem(i))


    def print_tree_structure(self):

        for i in range(self.topLevelItemCount()):
            top = self.topLevelItem(i)
            print(f"Top: {top.text(0)}")

            for j in range(top.childCount()):
                mid = top.child(j)
                print(f"  └─ Mid: {mid.text(0)}")

                for k in range(mid.childCount()):
                    leaf = mid.child(k)
                    print(f"      └─ Leaf: {leaf.text(0)}")


    def set_current_tree_item(self, parent_text: str = "", parent2_text: str = "", item_initial_text: str = "") -> None:
        """
        Sets the current item in the QTreeWidget based on one or two levels of parent text
        and the initial part of the target item's text. This is useful for restoring
        selection after a tree refresh.
        Args:
            parent_text (str): The text of the top-level parent item.
            parent2_text (str): The text of the second-level parent item (child of parent_text).
                                If empty, the search is directly under the top-level parent.
            item_initial_text (str): The initial text of the item to select (e.g., "Node 1").
        """

        if not item_initial_text:
            return

        for i in range(self.topLevelItemCount()):
            top_level_item: Optional[QTreeWidgetItem] = self.topLevelItem(i)

            if top_level_item and top_level_item.text(0) == parent_text:

                if parent2_text:

                    for j in range(top_level_item.childCount()):
                        second_level_item: Optional[QTreeWidgetItem] = top_level_item.child(j)

                        if second_level_item and second_level_item.text(0) == parent2_text:

                            for k in range(second_level_item.childCount()):
                                target_item: Optional[QTreeWidgetItem] = second_level_item.child(k)

                                if target_item and target_item.text(0).startswith(item_initial_text):
                                    self.setCurrentItem(target_item)
                                    target_item.setSelected(True)
                                    return

                            return

                else:

                    for j in range(top_level_item.childCount()):
                        target_item: Optional[QTreeWidgetItem] = top_level_item.child(j)

                        if target_item and target_item.text(0).startswith(item_initial_text):
                            self.setCurrentItem(target_item)
                            target_item.setSelected(True)
                            return

                return

    def add_node(self, reference_node_id: Optional[int] = None, update: bool = True) -> None:
        """
        Adds a new node to the imported data and updates the tree widget.
        The new node's position is determined relative to a reference node or at (0,0,0) / (0,0).
        Args:
            reference_node_id (Optional[int]): The ID of an existing node to use as a reference

                                              for positioning the new node. If None or invalid,
                                              the node is added at the origin.
            update (bool): If True, the tree widget and plot will be updated after adding the node.
        """

        # ---------------------------------------------
        # DETERMINE NEW NODE ID AND PROPERTIES
        # ---------------------------------------------

        next_node_id: int = max(self.imported_data['nodes'].keys(), default=0) + 1
        dofs_per_node: int = self.imported_data['structure_info']['dofs_per_node']
        force_labels: List[str] = self.imported_data['structure_info']['force_labels']
        new_position: Dict[str, float]

        if not self.imported_data['nodes'] or reference_node_id not in self.imported_data['nodes']:
            new_position = {'X': 0.0, 'Y': 0.0}

            if 'Z' in force_labels:
                new_position['Z'] = 0.0
        else:
            ref_node: Dict[str, float] = self.imported_data['nodes'][reference_node_id]
            new_position = {
                'X': ref_node['X'] + 1.0,
                'Y': ref_node['Y']
            }

            if 'Z' in ref_node:
                new_position['Z'] = ref_node['Z']
        self.imported_data['nodes'][next_node_id] = {
            **new_position,
            'force': tuple(0.0 for _ in range(dofs_per_node)),
            'displacement': tuple(np.nan for _ in range(dofs_per_node))
        }
        print(f"Added Node {next_node_id} at {new_position}")
        node_text: str = f"({', '.join([str(self.imported_data['nodes'][next_node_id].get(coord, '')) for coord in ['X', 'Y', 'Z']])})"
        self.left_dock_window.add_tree_item(index=next_node_id, item_text=node_text, parent_text='Nodes', item_type='Node') 
        self.left_dock_window.clear_tree_widget_items() 
        self.left_dock_window.display_selected_item_info() 
        self.expand_section(section="Nodes")


    def update_tree(self, imported_data: Dict[str, Any], node_id: Optional[int] = None, update_plotdata: bool = True, reprocess_sructure: bool = True, item_text: str = None) -> None:
        """
        Updates the tree widget and preserves the selection and expansion state.
        It also updates the `imported_data` in `ResultsViewer` and `LeftDockWindow`.
        Args:
            imported_data (Dict[str, Any]): The updated structural data.
            node_id (Optional[int]): An optional old node ID, for specific handling.
            update_plotdata (bool): If True, updates the plot data in `ResultsViewer`.
            reprocess_sructure (bool): If True, reprocesses the imported structure data.
        """
        persistent_identifier: Optional[Tuple[Optional[str], Optional[str], str]] = None
        selected_item: Optional[QTreeWidgetItem] = self.currentItem()
        self.imported_data = imported_data
        self.left_dock_window.imported_data = imported_data 

        if update_plotdata:
            ResultsViewer.imported_data = imported_data 

        if selected_item:
            persistent_identifier = self.get_persistent_item_identifier(selected_item)
        expanded_items: Set[str] = self.get_expanded_items(self)

        if reprocess_sructure:
            ProcessImportedStructure(imported_data=imported_data) 
        self.left_dock_window.details_tree.clear() 
        self.populate_tree(self.imported_data)
        self.restore_expansion_state(self, expanded_items)
        ResultsViewer.plot_truss() 

        if persistent_identifier:
            parent1: Optional[str]
            parent2: Optional[str]
            item_initial_text: str
            parent1, parent2, item_initial_text = persistent_identifier 

            if parent1 is None: 
                parent1 = parent2
                parent2 = ""
            self.set_current_tree_item(parent1, parent2, item_text if item_text else item_initial_text)
            self.left_dock_window.display_selected_item_info() 


    def get_persistent_item_identifier(self, item: QTreeWidgetItem) -> Tuple[Optional[str], Optional[str], str]:
        """
        Generates a persistent identifier for a QTreeWidgetItem, useful for restoring
        selection across tree updates. The identifier includes the text of the
        grandparent, parent, and the significant part of the child's text (before a colon).
        Args:
            item (QTreeWidgetItem): The item for which to generate the identifier.
        Returns:
            Tuple[Optional[str], Optional[str], str]: A tuple containing (grandparent_text,
                                                    parent_text, child_significant_text).
        """
        parent1: Optional[str] = None
        parent2: Optional[str] = None
        parent_item: Optional[QTreeWidgetItem] = item.parent()

        if parent_item:
            parent1 = parent_item.text(0)
            grandparent_item: Optional[QTreeWidgetItem] = parent_item.parent()

            if grandparent_item:
                parent2 = grandparent_item.text(0)
        child_text: str = item.text(0)
        child: str

        if ":" in child_text:
            child = child_text.split(":")[0].strip()
        else:
            child = child_text
        return (parent2, parent1, child)

    def delete_node(self, node_ids: Union[int, List[int], Set[int]]) -> None:
        """
        Deletes one or more nodes and any elements, concentrated loads, or nodal displacements
        that are associated with the deleted nodes. Orphaned nodes (nodes no longer
        referenced by any element) are also removed.
        Args:
            node_ids (Union[int, List[int], Set[int]]): A single node ID (int) or a collection
                                                         of node IDs (list or set) to delete.
        """

        # ---------------------------------------------
        # NORMALIZE INPUT NODE_IDS
        # ---------------------------------------------

        node_id_0: int
        nodes_to_delete: Set[int]

        if isinstance(node_ids, int):
            node_id_0 = node_ids
            nodes_to_delete = {node_ids}
        else:
            nodes_to_delete = set(node_ids)
            node_id_0 = min(nodes_to_delete) if nodes_to_delete else 0

        # ---------------------------------------------
        # IDENTIFY AND DELETE ASSOCIATED ELEMENTS
        # ---------------------------------------------

        elements_to_delete: List[int] = []

        for elem_id, elem_data in list(self.imported_data['elements'].items()):

            if elem_data["node1"] in nodes_to_delete or elem_data["node2"] in nodes_to_delete:
                elements_to_delete.append(elem_id)

        for elem_id in elements_to_delete:
            self.delete_element(elem_id)

        # ---------------------------------------------
        # DELETE NODES
        # ---------------------------------------------

        for node_id in nodes_to_delete:

            if node_id in self.imported_data['nodes']:
                del self.imported_data['nodes'][node_id]
                print(f"Deleted Node {node_id}")
            else:
                print(f"Node {node_id} not found.")

        # ---------------------------------------------
        # REMOVE DELETED NODES FROM CONSTRAINTS AND FORCES
        # ---------------------------------------------

        for category in ["nodal_displacements", "concentrated_loads"]:

            for node_id in list(self.imported_data[category].keys()):

                if node_id in nodes_to_delete:
                    del self.imported_data[category][node_id]

        # ---------------------------------------------
        # UPDATE UI AND SELECTION
        # ---------------------------------------------

        self.update_tree(self.imported_data)
        self.left_dock_window.clear_tree_widget_items() 
        previous_node_id: int = node_id_0 - 1

        if previous_node_id in self.imported_data["nodes"]:
            self.set_current_tree_item("Geometry", "Nodes", f"Node {previous_node_id}:")
            self.left_dock_window.display_selected_item_info() 
        else:
            keys: List[int] = list(self.imported_data["nodes"].keys())

            if keys:
                first_key: int = keys[0]
                self.set_current_tree_item("Geometry", "Nodes", f"Node {first_key}:")
                self.left_dock_window.display_selected_item_info() 
            else:
                print("No nodes found in imported_data.")


    def delete_all_nodes(self) -> None:
        """
        Deletes all nodes, elements, concentrated loads, nodal displacements, and
        distributed loads from the imported data. Updates the tree widget afterward.
        """
        print("Deleting all nodes and affected elements...")

        # ---------------------------------------------
        # CLEAR ALL RELATED DICTIONARIES
        # ---------------------------------------------

        self.imported_data['nodes'].clear()
        self.imported_data['elements'].clear()
        self.imported_data['concentrated_loads'].clear()
        self.imported_data['nodal_displacements'].clear()
        self.imported_data['distributed_loads'].clear()

        # ---------------------------------------------
        # UPDATE UI
        # ---------------------------------------------

        self.update_tree(self.imported_data)
        print("All nodes and related data have been removed.")


    def assign_material_to_element(self, element_data: Dict[str, Any], material_code: str) -> None:
        """
        Assigns a specified material to the currently selected element in the tree.
        Args:
            element_data (Dict[str, Any]): The data of the element to which the material is assigned.
                                          (Note: This parameter might be redundant as element_id
                                          is retrieved from selectedItems).
            material_code (str): The code of the material to assign.
        """
        selected_items: List[QTreeWidgetItem] = self.selectedItems()

        if not selected_items:
            print("No element selected to assign material.")
            return

        item_data: Dict[str, Any] = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
        element_id: int = item_data["id"]
        self.imported_data["elements"][element_id]["material_code"] = material_code

        # ---------------------------------------------
        # UPDATE UI
        # ---------------------------------------------

        self.update_tree(self.imported_data)
        print(f"Assigned material '{material_code}' to element {element_id}")


    def assign_cross_section_to_element(self, element_data: Dict[str, Any], cs_code: str) -> None:
        """
        Assigns a specified cross-section to the currently selected element in the tree.
        Args:
            element_data (Dict[str, Any]): The data of the element to which the cross-section is assigned.
                                          (Note: This parameter might be redundant as element_id
                                          is retrieved from selectedItems).
            cs_code (str): The code of the cross-section to assign.
        """
        selected_items: List[QTreeWidgetItem] = self.selectedItems()

        if not selected_items:
            print("No element selected to assign cross-section.")
            return

        item_data: Dict[str, Any] = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
        element_id: int = item_data["id"]
        self.imported_data["elements"][element_id]["section_code"] = cs_code

        # ---------------------------------------------
        # UPDATE UI
        # ---------------------------------------------

        self.update_tree(self.imported_data)
        print(f"Assigned cross-section '{cs_code}' to element {element_id}")


    def add_element(self, clickedItem: int = 0) -> None:
        """
        Adds a new element to the imported data. This typically involves adding two new nodes
        and then creating an element connecting them with default properties.
        Args:
            clickedItem (int): A placeholder argument, not directly used in the logic

                                for determining element properties.
        """

        # ---------------------------------------------
        # ENSURE NODES EXIST FOR THE NEW ELEMENT
        # ---------------------------------------------

        existing_nodes: List[int] = sorted(self.imported_data['nodes'].keys())
        new_node1_id: int
        new_node2_id: int

        if not existing_nodes:
            new_node1_id = 1
            self.add_node(new_node1_id, update=False)
        else:
            new_node1_id = existing_nodes[-1]
        new_node2_id = max(self.imported_data['nodes'].keys(), default=0) + 1
        self.add_node(new_node2_id, update=False)

        # ---------------------------------------------
        # DEFINE DEFAULT ELEMENT PROPERTIES
        # ---------------------------------------------

        new_element_id: int = max(self.imported_data['elements'].keys(), default=0) + 1
        default_element: Dict[str, Any]

        if "DEFAULT" in self.imported_data['cross_sections'].keys() and 0 in self.imported_data["elements"].keys():
            default_element = self.imported_data["elements"][0].copy()
            default_element.update({
                "node1": new_node1_id,
                "node2": new_node2_id,
            })
        else:
            default_element = {
                "node1": new_node1_id,
                "node2": new_node2_id,
                "section_code": "DEFAULT",
                "material_code": "DEFAULT",
                "E": 200000000000.0,
                "v": 0.3,
                "G": 76923076923.08,
                "J": 1.0,
                "Iy": 1.0,
                "Iz": 1.0,
                "A": 1.0,
                "length": 1.0,
                "angle": 0.0
            }

        # ---------------------------------------------
        # ADD TO ELEMENTS DICTIONARY AND UPDATE UI
        # ---------------------------------------------

        self.imported_data['elements'][new_element_id] = default_element
        self.left_dock_window.display_element_info(new_element_id) 
        self.update_tree(self.imported_data)
        self.set_current_tree_item("Geometry", "Elements", item_initial_text=f"Element {new_element_id}:")
        self.left_dock_window.display_element_info(new_element_id) 


    def delete_element(self, elem_id: int) -> None:
        """
        Deletes a specified element from the imported data. It also removes any
        associated distributed loads and deletes any nodes that become "orphaned"
        (no longer connected to any remaining elements) after the element's removal.
        Args:
            elem_id (int): The ID of the element to delete.
        """

        if elem_id not in self.imported_data['elements']:
            print(f"Element {elem_id} not found.")
            return


        # ---------------------------------------------
        # GET NODES ASSOCIATED WITH THE ELEMENT
        # ---------------------------------------------

        element_data: Dict[str, Any] = self.imported_data['elements'][elem_id]
        node1: int = element_data["node1"]
        node2: int = element_data["node2"]

        # ---------------------------------------------
        # REMOVE THE ELEMENT
        # ---------------------------------------------

        del self.imported_data['elements'][elem_id]
        print(f"Deleted Element {elem_id}")

        # ---------------------------------------------
        # REMOVE ASSOCIATED DISTRIBUTED LOAD
        # ---------------------------------------------

        if elem_id in self.imported_data['distributed_loads']:
            del self.imported_data['distributed_loads'][elem_id]
            print(f"Deleted distributed load for Element {elem_id}")

        # ---------------------------------------------
        # CHECK FOR AND REMOVE ORPHANED NODES
        # ---------------------------------------------

        referenced_nodes: Set[int] = set()
        elements_values: List[Dict[str, Any]] = list(self.imported_data['elements'].values())

        for elem in elements_values:
            referenced_nodes.add(elem["node1"])
            referenced_nodes.add(elem["node2"])

        if len(elements_values) == 0:
            pass

        orphaned_nodes: List[int] = []

        if node1 not in referenced_nodes:
            orphaned_nodes.append(node1)

        if node2 not in referenced_nodes:
            orphaned_nodes.append(node2)

        for node_id in orphaned_nodes:
            self.delete_node(node_id)
            print(f"Deleted orphaned Node {node_id}")

        # ---------------------------------------------
        # UPDATE UI AND SELECTION
        # ---------------------------------------------

        self.update_tree(self.imported_data)
        self.left_dock_window.clear_tree_widget_items() 
        previous_elem_id: int = elem_id - 1

        if previous_elem_id in self.imported_data["elements"]:
            self.set_current_tree_item("Geometry", "Elements", f"Element {previous_elem_id}:")
            self.left_dock_window.display_selected_item_info() 
        else:
            keys: List[int] = list(self.imported_data["elements"].keys())

            if keys:
                first_key: int = keys[0]
                self.set_current_tree_item("Geometry", "Elements", f"Element {first_key}:")
                self.left_dock_window.display_selected_item_info() 
            else:
                print("No elements found in imported_data.")


    def delete_all_elements(self) -> None:
        """
        Deletes all elements and their associated distributed loads from the imported data.
        Nodes remain intact. Updates the tree widget afterward.
        """
        print("Deleting all elements and associated data...")
        self.delete_all_nodes()


    def get_last_section_number(self, parent_txt: Optional[str] = None) -> Optional[int]:
        """
        Retrieves the highest numerical ID (e.g., from "Section 5: ...") under a specified
        parent item in the tree.
        Args:
            parent_txt (Optional[str]): The text of the parent category item (e.g., "Cross Sections").
        Returns:
            Optional[int]: The highest section number found, or None if no sections are found
                           or the parent is invalid.
        """
        parent_item: Optional[QTreeWidgetItem] = self._find_parent_item(parent_txt)

        if parent_item:
            max_number: int = -1
            regex_pattern: str = rf"{re.escape(parent_txt[:-1])} (\d+):" if parent_txt and parent_txt.endswith('s') else rf"{re.escape(str(parent_txt))} (\d+):"

            for i in range(parent_item.childCount()):
                child_item: QTreeWidgetItem = parent_item.child(i)
                match: Optional[re.Match[str]] = re.match(regex_pattern, child_item.text(0))

                if match:
                    number: int = int(match.group(1))
                    max_number = max(max_number, number)
            return max_number if max_number != -1 else None

        else:
            return None

    def _find_child_item(self, parent_item: QTreeWidgetItem, child_text: str) -> Optional[QTreeWidgetItem]:
        """
        Recursively finds a child item with the specified text under a given parent item.
        Args:
            parent_item (QTreeWidgetItem): The parent item to start the search from.
            child_text (str): The exact text of the child item to find.
        Returns:
            Optional[QTreeWidgetItem]: The found QTreeWidgetItem, or None if not found.
        """

        for i in range(parent_item.childCount()):
            child_item: QTreeWidgetItem = parent_item.child(i)

            if child_item.text(0) == child_text:
                return child_item

            found_item: Optional[QTreeWidgetItem] = self._find_child_item(child_item, child_text)

            if found_item:
                return found_item

        return None

    def _find_parent_item(self, parent_text: Optional[str]) -> Optional[QTreeWidgetItem]:
        """
        Finds the QTreeWidgetItem corresponding to the given parent text, handling both
        top-level categories and sub-categories.
        Args:
            parent_text (Optional[str]): The text of the parent item to find.
        Returns:
            Optional[QTreeWidgetItem]: The found QTreeWidgetItem, or None if not found.
        """

        if not parent_text:
            return None

        top_level_parent_text: Optional[str] = None

        if parent_text in ["Nodes", "Elements", "Cross Sections"]:
            top_level_parent_text = "Geometry"
        elif parent_text in ["Nodal Forces", "Nodal Displacements", "Distributed Loads"]:
            top_level_parent_text = "Boundary Conditions"
        elif parent_text in ["Materials"]:
            top_level_parent_text = "Materials"
        elif parent_text in ["Meshing"]:
            top_level_parent_text = "Meshing"
        elif parent_text in ["Solver", "Solvers"]:
            top_level_parent_text = "Solver"
        elif parent_text in ["Post-Processing", "Post-Processings"]:
            top_level_parent_text = "Post-Processing"
        else:
            top_level_parent_text = parent_text

        if top_level_parent_text:

            for i in range(self.topLevelItemCount()):
                top_level_item: QTreeWidgetItem = self.topLevelItem(i)

                if top_level_item.text(0) == top_level_parent_text:

                    if top_level_parent_text == parent_text:
                        return top_level_item

                    else:
                        return self._find_child_item(top_level_item, parent_text)

        return None

    def add_cross_section(self, section_code: Optional[str] = None, add_section: bool = False) -> None:
        """
        Initiates the process of adding a new cross-section. This method primarily
        delegates to `LeftDockWindow` to display the cross-section information editor.
        After the section is potentially added, it updates the tree and sets the
        new section as the current item.
        Args:
            section_code (Optional[str]): The code for the cross-section to add/edit.
                                         If None, a new default section might be created.
            add_section (bool): A flag indicating if this is an explicit add operation.
        """

        if hasattr(self.left_dock_window, 'temp_cross_sections') and self.left_dock_window.temp_cross_sections: 
            return

        cross_sections: Dict[str, Any] = self.imported_data.get("cross_sections", {})

        if cross_sections:
            section_code = next(iter(cross_sections))
        base_code = "Solid_Circular"
        counter = 1

        while f"{base_code}_{counter}" in cross_sections:
            counter += 1
        section_code = f"{base_code}_{counter}"
        cross_sections[section_code] = {
            "type": "Solid_Circular",
            "dimensions": {"D": 0.1}
        }
        self.imported_data["cross_sections"] = cross_sections
        self.left_dock_window.display_cross_section_info(section_code) 

        if section_code:

            # ---------------------------------------------
            # EXPAND SECTION AND UPDATE TREE
            # ---------------------------------------------

            self.expand_section(section="Cross Sections")
            self.update_tree(self.imported_data)
            self.set_current_tree_item(parent_text="Geometry", parent2_text="Cross Sections", item_initial_text=section_code)
            self.left_dock_window.display_selected_item_info()


    def delete_cross_section(self, section_code: str) -> None:
        """
        Deletes a specified cross-section from the imported data. Any elements
        that were assigned this cross-section will have their cross-sectional
        properties (A, J, Iy, Iz) set to NaN.
        Args:
            section_code (str): The code of the cross-section to delete.
        """
        print(f"Deleting Section {section_code}")

        # ---------------------------------------------
        # REMOVE SECTION FROM imported_data
        # ---------------------------------------------

        if section_code in self.imported_data['cross_sections']:
            del self.imported_data['cross_sections'][section_code]

        # ---------------------------------------------
        # UPDATE AFFECTED ELEMENTS
        # ---------------------------------------------

        for elem_id, elem_data in self.imported_data['elements'].items():

            if elem_data['section_code'] == section_code:
                elem_data['A'] = np.nan
                elem_data['J'] = np.nan
                elem_data['Iy'] = np.nan
                elem_data['Iz'] = np.nan

        # ---------------------------------------------
        # UPDATE UI AND SELECTION
        # ---------------------------------------------

        self.update_tree(self.imported_data)
        self.left_dock_window.initialize_last_selected_cross_sections() 
        last_item_code: Optional[str] = next(reversed(self.left_dock_window.last_selected_cross_sections), None) 

        if last_item_code:
            self.set_current_tree_item(parent_text="Geometry", parent2_text="Cross Sections", item_initial_text=last_item_code)
            self.left_dock_window.display_cross_section_info(last_item_code) 
        else:
            self.left_dock_window.clear_tree_widget_items() 
        print(f"Section {section_code} removed. Affected elements updated with NaN values.")


    def delete_all_cross_sections(self) -> None:
        """
        Deletes all cross-sections from the imported data. All elements will have their
        cross-sectional properties (A, J, Iy, Iz) set to NaN, indicating no assigned section.
        Updates the tree widget afterward.
        """
        print(f"Deleting all {len(self.imported_data['cross_sections'])} sections...")

        # ---------------------------------------------
        # CLEAR ALL SECTIONS
        # ---------------------------------------------

        self.imported_data['cross_sections'].clear()

        # ---------------------------------------------
        # RESET CROSS-SECTIONAL PROPERTIES FOR ALL ELEMENTS
        # ---------------------------------------------

        for elem_id, elem_data in self.imported_data['elements'].items():
            elem_data['A'] = np.nan
            elem_data['J'] = np.nan
            elem_data['Iy'] = np.nan
            elem_data['Iz'] = np.nan

        # ---------------------------------------------
        # UPDATE UI
        # ---------------------------------------------

        self.update_tree(self.imported_data)
        self.left_dock_window.clear_tree_widget_items() 
        print("All sections removed. Affected elements updated with NaN values.")


    def expand_section(self, section: str) -> None:
        """
        Expands the specified top-level section in the tree widget.
        Args:
            section (str): The text of the top-level section to expand (e.g., "Geometry", "Nodes").
        """
        root: Optional[QTreeWidgetItem] = self.invisibleRootItem()

        if not root:
            return

        for i in range(root.childCount()):
            item: QTreeWidgetItem = root.child(i)

            if item.text(0) == section:
                item.setExpanded(True)
                return


    # ---------------------------------------------
    # MATERIAL MANAGEMENT
    # ---------------------------------------------


    def add_material(self, material_code: Optional[str] = None) -> None:
        """
        Adds a new material to the structural model and updates the UI.
        If `temp_materials` in `left_dock_window` is not empty, the function returns early.
        It then calls `display_material_info` to get material details and adds the material
        to the tree widget, making it the currently selected item.
        Args:
            material_code (Optional[str]): The code for the material to be added. If None,
                                           `display_material_info` will likely prompt for one.
        """

        if self.left_dock_window.temp_materials.keys():
            return

        material_code = self.left_dock_window.display_material_info(material_code, add_material=True)
        self.expand_section(section="Materials")
        last_section_number = self.get_last_section_number("Materials")
        new_section_number = last_section_number + 1 if last_section_number is not None else 0
        self.update_tree(self.imported_data)
        self.set_current_tree_item(parent_text="Materials", item_initial_text=material_code)
        self.left_dock_window.display_material_info(material_code, add_material=False)


    def _find_material_tree_item(self, parent1: str = "", parent2: str = "", item_text: str = "") -> Tuple[Optional[QTreeWidgetItem], Optional[QTreeWidgetItem], Optional[QTreeWidgetItem], int]:
        """
        Private method. Finds the QTreeWidgetItem for a given material based on its text (code/id).
        Args:
            parent1 (str): The text of the first-level parent item (e.g., "Materials").
            parent2 (str): The text of the second-level parent item (if applicable).
            item_text (str): The text (code or ID) of the item to find.
        Returns:
            Tuple[Optional[QTreeWidgetItem], Optional[QTreeWidgetItem], Optional[QTreeWidgetItem], int]:
            A tuple containing:
            - The first-level parent item (QTreeWidgetItem)
            - The second-level parent item (QTreeWidgetItem)
            - The found item itself (QTreeWidgetItem)
            - The index of the found item within its parent's children.
            Returns (None, None, None, -1) if the item is not found.
        """
        parent1_item = None

        for i in range(self.topLevelItemCount()):

            if self.topLevelItem(i).text(0) == parent1:
                parent1_item = self.topLevelItem(i)
                break

        if parent1_item:

            if parent2:
                parent2_item = None

                for i in range(parent1_item.childCount()):

                    if parent1_item.child(i).text(0) == parent2:
                        parent2_item = parent1_item.child(i)
                        break
            else:
                parent2_item = parent1_item

            if parent2_item:

                for i in range(parent2_item.childCount()):
                    item = parent2_item.child(i)
                    item_data = item.data(0, Qt.ItemDataRole.UserRole)

                    if isinstance(item_data, dict) and (item_data.get("id") == item_text or item_data.get("code") == item_text):
                        return parent1_item, parent2_item, item, i

        return None, None, None, -1

    def delete_material(self, material_code: str) -> None:
        """
        Deletes a material from the structural model and updates the UI.
        This method removes the material from `imported_data['materials']` and then
        updates the tree view. It attempts to select the next available material
        or clears the display if no materials remain.
        Args:
            material_code (str): The code of the material to be deleted.
        """
        print(f"Deleting Material {material_code}")
        materials_parent_item, _, deleted_item, deleted_index = self._find_material_tree_item(parent1="Materials", item_text=material_code)

        if material_code in self.imported_data['materials']:
            del self.imported_data['materials'][material_code]
        next_item_code = None

        if materials_parent_item:

            if deleted_index < materials_parent_item.childCount() - 1:
                next_item = materials_parent_item.child(deleted_index + 1)
            elif deleted_index > 0:
                next_item = materials_parent_item.child(deleted_index - 1)
            else:
                next_item = None

            if next_item:
                item_data = next_item.data(0, Qt.ItemDataRole.UserRole)

                if isinstance(item_data, dict) and item_data.get("type") == "material":
                    next_item_code = item_data.get("code")
        self.update_tree(self.imported_data)
        self.left_dock_window.initialize_last_selected_materials()

        if next_item_code:
            self.set_current_tree_item(parent_text="Materials", item_initial_text=next_item_code)
            self.left_dock_window.display_material_info(next_item_code)
        else:
            self.left_dock_window.clear_tree_widget_items()
        self.left_dock_window.temp_materials = {}
        print(f"Material {material_code} removed.")


    def delete_all_materials(self) -> None:
        """
        Deletes all materials from the structural model.
        This method clears the 'materials' dictionary in `imported_data`. It also
        iterates through all elements and sets their cross-sectional properties
        (A, J, Iy, Iz) to NaN, as these properties are typically linked to materials
        or cross-sections defined by materials. Finally, it updates the UI tree.
        """
        print(f"Deleting all {len(self.imported_data['materials'])} materials...")
        self.imported_data['materials'].clear()

        for elem_id, elem_data in self.imported_data['elements'].items():
            elem_data['A'] = np.nan
            elem_data['J'] = np.nan
            elem_data['Iy'] = np.nan
            elem_data['Iz'] = np.nan
        self.update_tree(self.imported_data)
        self.left_dock_window.clear_tree_widget_items()
        print("All materials removed. Affected elements updated with NaN values.")
   
   
    # ---------------------------------------------
    # NODAL DISPLACEMENT MANAGEMENT
    # ---------------------------------------------

    def add_node_displacement(self, show_node_id: bool = True, new_constraint: bool = True) -> None:
        """
        Adds a new nodal displacement constraint to the structural model.
        This method initializes a new nodal displacement for node 0, sets its
        displacement values based on a 'Fixed' constraint type, updates the
        `imported_data`, and then refreshes the UI tree. It also sets the
        newly added node as the current item in the tree and displays its info.
        Args:
            show_node_id (bool): If True, the node ID will be displayed.
            new_constraint (bool): If True, indicates that this is a new constraint being added.
        """
        self.left_dock_window.previous_node = 0
        node_data = self.left_dock_window.get_node_data(constraint_type="Fixed")
        self.imported_data['nodes'][0] = node_data.copy()
        self.imported_data['nodal_displacements'][0] = copy.deepcopy(node_data["displacement"])
        self.update_tree(self.imported_data)
        self.set_current_tree_item(parent_text="Boundary Conditions", parent2_text="Nodal Displacements", item_initial_text="Node 0:")
        self.left_dock_window.display_node_info(node_id=0, node_data=node_data, show_node_id=show_node_id, new_constraint=new_constraint)
        selected_item = self.currentItem()

        if selected_item:
            selected_item.setData(0, Qt.ItemDataRole.UserRole,
                                   {"type": "nodal_displacement", "id": 0,
                                    "identifier": f"nodal_displacement{0}"})
        self.left_dock_window.display_selected_item_info(item_data_dict=node_data, new_constraint=new_constraint)


    def delete_node_displacement(self, node_id: int) -> None:
        """
        Removes displacement constraints for a specific node and resets its DOFs.
        This method first removes the node from `imported_data['nodal_displacements']`.
        If the node exists in `imported_data['nodes']`, its `displacement` DOFs are
        set to NaN (indicating a free node). If no concentrated load exists for that
        node, its `force` DOFs are reset to zero. The UI tree is then updated.
        Args:
            node_id (int): The ID of the node whose displacement constraint is to be removed.
        """

        if 'nodal_displacements' in self.imported_data and node_id in self.imported_data['nodal_displacements']:
            print(f"Removing nodal displacement constraints for Node {node_id}...")
            del self.imported_data['nodal_displacements'][node_id]

        if 'nodes' in self.imported_data and node_id in self.imported_data['nodes'] and node_id == 0:
            del self.imported_data['nodes'][node_id]
        elif 'nodes' in self.imported_data and node_id in self.imported_data['nodes']:
            node_data = self.imported_data['nodes'][node_id]
            structure_info = self.imported_data.get('structure_info', {})
            dofs_per_node = structure_info.get('dofs_per_node', len(node_data['displacement']))
            node_data['displacement'] = tuple(np.nan for _ in range(dofs_per_node))

            if node_id not in self.imported_data.get('concentrated_loads', {}):
                node_data['force'] = tuple(0.0 for _ in range(dofs_per_node))
            print(f"Node {node_id} converted to a free node (displacements → NaN).")
        else:
            print(f"Warning: Node {node_id} not found in 'nodes'.")
        self.update_tree(self.imported_data)
        last_item_code = next(reversed(self.imported_data['nodal_displacements']), None)

        if last_item_code is not None:
            self.set_current_tree_item(parent_text="Boundary Conditions", parent2_text="Nodal Displacements", item_initial_text=f"Node {last_item_code}:")
            self.left_dock_window.clear_tree_widget_items()
            self.left_dock_window.display_node_info(node_id=last_item_code, show_node_id=True)
        else:
            self.left_dock_window.clear_tree_widget_items()
        print(f"Node {node_id} successfully reset to free conditions.")


    def delete_all_nodal_displacements(self) -> None:
        """
        Removes all nodal displacement constraints from the structural model.
        This method clears the `nodal_displacements` dictionary. For each affected
        node, it sets the `displacement` DOFs to NaN (free) and resets `force` DOFs
        to zero if no concentrated load is applied to that node. The UI tree is then updated.
        """

        if 'nodal_displacements' not in self.imported_data or not self.imported_data['nodal_displacements']:
            print("No nodal displacement constraints to remove.")
            return

        print("Removing nodal displacement constraints for affected nodes...")
        affected_nodes = list(self.imported_data['nodal_displacements'].keys())
        self.imported_data['nodal_displacements'].clear()

        if 'nodes' in self.imported_data:
            structure_info = self.imported_data.get('structure_info', {})
            dofs_per_node = structure_info.get('dofs_per_node', 6)

            for node_id in affected_nodes:

                if node_id in self.imported_data['nodes']:
                    self.imported_data['nodes'][node_id]['displacement'] = tuple(np.nan for _ in range(dofs_per_node))

                    if node_id not in self.imported_data.get('concentrated_loads', {}):
                        self.imported_data['nodes'][node_id]['force'] = tuple(0.0 for _ in range(dofs_per_node))
        print(f"Nodes {affected_nodes} converted to free nodes (displacements → NaN).")
        self.update_tree(self.imported_data)
        self.left_dock_window.clear_tree_widget_items()
        print("Selected nodal displacements successfully removed.")
  
    # ---------------------------------------------
    # DISTRIBUTED LOAD MANAGEMENT
    # ---------------------------------------------

    def add_distributed_load(self) -> None:
        """
        Adds a new distributed load to the structural model.
        This method attempts to add a distributed load to `element 0`. It checks if
        `element 1` exists (as a base for copying properties if element 0 doesn't exist).
        If `element 1` is missing, it displays an error. Otherwise, it initializes
        a uniform distributed load, adds it to `imported_data`, and updates the UI.
        """
        elem_id = 0
        self.left_dock_window.clear_tree_widget_items()
        node_data = {'type': 'Uniform', 'direction': 'Global_X', 'parameters': (elem_id,)}
        self.imported_data['distributed_loads'][elem_id] = node_data.copy()

        if 1 not in self.imported_data['elements']:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("Element 1 does not exist. Please create element 1 first.")
            msg.setWindowTitle("Element Error")
            msg.exec()
            return

        self.imported_data['elements'][elem_id] = copy.deepcopy(self.imported_data['elements'][1])
        self.left_dock_window.add_tree_item(index=elem_id, item_text='', parent_text='Distributed Loads', item_type='Element')
        self.left_dock_window.display_distributed_load_info(elem_id)


    def delete_distributed_load(self, element_id: int) -> None:
        """
        Deletes the distributed load associated with a specific element.
        This method removes the distributed load from `imported_data['distributed_loads']`.
        It also removes the corresponding item from the tree widget and refreshes the
        results viewer plot.
        Args:
            element_id (int): The ID of the element whose distributed load should be removed.
        """

        if "distributed_loads" in self.imported_data and element_id in self.imported_data["distributed_loads"]:
            del self.imported_data["distributed_loads"][element_id]
            self.left_dock_window.remove_tree_item(index=element_id, parent_text="Distributed Loads", item_type="Element")
            self.left_dock_window.display_selected_item_info()
            ResultsViewer.plot_truss()
            print(f"Distributed load: Element {element_id} was deleted successfully.")
        else:
            print(f"Distributed load for Element {element_id} not found.")


    def delete_all_distributed_loads(self) -> None:
        """
        Deletes all distributed loads from the imported data and removes them from the tree widget.
        This method clears the `distributed_loads` dictionary. It then iterates
        through the top-level items in the tree widget to find "Distributed Loads"
        and removes all its children. Finally, it clears the current display
        in the left dock and refreshes the results viewer plot.
        """

        if "distributed_loads" in self.imported_data:
            self.imported_data["distributed_loads"].clear()

            for i in range(self.topLevelItemCount()):
                parent_item = self.topLevelItem(i)

                if parent_item.text(0) == "Distributed Loads":
                    parent_item.takeChildren()
                    break
        self.left_dock_window.clear_tree_widget_items()
        self.update_tree(self.imported_data)
        ResultsViewer.plot_truss()
  
    # ---------------------------------------------
    # CONCENTRATED LOAD MANAGEMENT
    # ---------------------------------------------

    def add_concentrated_load(self, show_node_id: bool = True, new_constraint: bool = True) -> None:
        """
        Adds a new concentrated load to the structural model.
        This method adds a concentrated load to `node 0` if it doesn't already
        have a concentrated load or is not defined. It retrieves node data with
        a 'Free' constraint type, adds the node and its force to `imported_data`,
        and updates the UI tree and display.
        Args:
            show_node_id (bool): If True, the node ID will be displayed.
            new_constraint (bool): If True, indicates that this is a new constraint being added.
        """

        if 0 in self.imported_data['nodes'] or 0 in self.imported_data['concentrated_loads']:
            return

        self.left_dock_window.previous_node = 0
        self.left_dock_window.clear_tree_widget_items()
        node_data = self.left_dock_window.get_node_data(constraint_type="Free")
        self.imported_data['nodes'][0] = node_data.copy()
        self.imported_data['concentrated_loads'][0] = copy.deepcopy(node_data["force"])
        self.left_dock_window.add_tree_item(0, '', parent_text='Nodal Forces')
        self.set_current_tree_item(parent_text="Boundary Conditions", parent2_text="Nodal Forces", item_initial_text="Node 0:")   
        self.left_dock_window.display_node_info(node_id=0, node_data=node_data, show_node_id=show_node_id, new_constraint=new_constraint, enable_constraint_type=False)
        self.current_nodes_combo_index = 0
        selected_item = self.currentItem()
        selected_item.setData(0, Qt.ItemDataRole.UserRole, 
                                {"type": "concentrated_load", "id": 0})


    def delete_concentrated_load(self, node_id: int) -> None:
        """
        Removes the concentrated load from a node and resets its force based on its constraint type.
        This method removes the load from `imported_data['concentrated_loads']`.
        If the node is `node 0`, it's completely removed from `imported_data['nodes']`.
        For other nodes, the force DOFs are reset based on the node's constraint type.
        The corresponding item is removed from the tree widget, and the UI is updated.
        Args:
            node_id (int): The ID of the node whose concentrated load is to be removed.
        """

        if node_id in self.imported_data['concentrated_loads']:
            del self.imported_data['concentrated_loads'][node_id]
            print(f"Concentrated load for Node {node_id} removed.")

        if node_id in self.imported_data['nodes'] and node_id == 0:
            del self.imported_data['nodes'][node_id]
            print(f"Node {node_id} (special case) removed.")
        elif node_id in self.imported_data['nodes']:
            node = self.imported_data['nodes'][node_id]
            constraint_type = node.get('constraint', 'Free')
            structure_info = self.imported_data['structure_info']
            default_values = self.left_dock_window.get_constraint_values(constraint_type, structure_info)
            node['force'] = default_values['force']
            print(f"Node {node_id}'s force reset based on '{constraint_type}' constraint.")
        self.left_dock_window.remove_tree_item(index=node_id, parent_text="Nodal Forces")
        last_item_code = next(reversed(self.imported_data['concentrated_loads']), None)

        if last_item_code is not None:
            self.set_current_tree_item(parent_text="Boundary Conditions", parent2_text="Nodal Forces", item_initial_text=f"Node {last_item_code}:")
            self.left_dock_window.clear_tree_widget_items()
            self.left_dock_window.display_node_info(node_id=last_item_code, show_node_id=True)
        else:
            self.left_dock_window.clear_tree_widget_items()
        self.left_dock_window.update_imported_data()
        self.update_tree(self.imported_data)
        ResultsViewer.plot_truss()


    def delete_all_concentrated_loads(self) -> None:
        """
        Removes all concentrated loads and resets nodal forces based on their constraint types.
        This method clears the `concentrated_loads` dictionary. For each node in the
        `imported_data['nodes']`, it resets the force DOFs based on the node's
        `constraint_type`. All concentrated load items are then removed from the
        tree widget, the display is cleared, and the results viewer plot is refreshed.
        """
        self.imported_data['concentrated_loads'].clear()
        print("All concentrated loads cleared from imported data.")
        structure_info = self.imported_data['structure_info']

        for node_id, node in self.imported_data['nodes'].items():
            constraint_type = node.get('constraint_type', 'Free')
            default_values = self.left_dock_window.get_constraint_values(constraint_type, structure_info)
            node['force'] = default_values['force']
            print(f"Node {node_id}'s force reset based on '{constraint_type}' constraint.")
            self.left_dock_window.remove_tree_item(index=node_id, parent_text="Nodal Forces")
        self.left_dock_window.clear_tree_widget_items()
        ResultsViewer.plot_truss()