import numpy as np
from typing import Dict, Union, Tuple, Optional
from scipy.sparse.base import spmatrix
import math
from utils.error_handler import ErrorHandler 
from core.direct_solver import DirectSolver
from core.iterative_solver import IterativeSolver
from core.distributed_load_converter import DistributedLoadConverter
from core.isoparametric_elements.linear_bar_element import LinearBarElement
from core.isoparametric_elements.linear_beam_element import LinearBeamElement
from core.isoparametric_elements.linear_frame_element import LinearFrameElement
from gui.units_handling import UnitsHandling
from gui.report_viewer import ReportViewer

# =============================================
# FINITE ELEMENT ANALYSIS SOLVER ENTRY CLASS
# =============================================

class MainSolver:
    """
    Main controller class for finite element structural analysis.
    Orchestrates the complete analysis workflow:
    1. Mesh generation/processing based on mesh_settings (not yet implemented)
    2. Element stiffness matrix computation based on element type
    3. Global stiffness matrix assembly
    4. Solution using either direct or iterative solver
    5. Post-processing of results
    Attributes:
        imported_data (Dict): Structure definition including nodes, materials, sections, etc.
        solver_settings (Dict): Configuration for solution methods and parameters
        mesh_settings (Dict): Parameters controlling mesh generation and element types
        K_global (Union[np.ndarray, spmatrix]): Assembled global stiffness matrix.
        element_handlers (Dict): Mapping of element types to their handler classes.
        solver (Union[DirectSolver, IterativeSolver]): Instance of the active solver (direct or iterative).
    """


    def __init__(self, 
                 imported_data: Dict,
                 solver_settings: Dict = None,
                 mesh_settings: Dict = None) -> None:
        """
        Initializes the structural solver with problem data and settings.
        Args:
            imported_data (Dict): Dictionary containing structural definition including:
                - nodes: Node coordinates and boundary conditions.
                - elements: Element connectivity and properties.
                - materials: Material properties.
                - sections: Cross-section properties.
                - distributed_loads: Distributed loads applied to elements (optional).
            solver_settings (Dict, optional): Dictionary containing solution parameters:
                - solver_type (str): 'Direct' or 'Iterative'. Defaults to 'Direct'.
                - tolerance (float): Convergence tolerance for iterative solver. Defaults to 1e-6.
                - max_iterations (int): Maximum iterations for iterative solver. Defaults to 100.
                - sparse_matrix (bool): Whether to use sparse matrices for global stiffness. Defaults to True.
                - large_deflection (bool): Enable geometrically nonlinear analysis. Defaults to False.
            mesh_settings (Dict, optional): Dictionary containing mesh parameters:
                - element_type (str): 'Bar', 'Beam', 'Q4', 'Q8', etc.
                - element_order (str): 'Linear' or 'Quadratic'.
                - mesh_density (int): Elements per unit length/area (if applicable).
                - algorithm (str): Meshing algorithm (e.g., 'Automatic').
                - refinement_factor (float): Local refinement factor.
                - non_conforming (bool): Allow non-conforming meshes.
                - auto_remesh (bool): Enable automatic remeshing.
        """

        # ---------------------------------------------
        # DATA INITIALIZATION
        # ---------------------------------------------

        self.imported_data: Dict = imported_data.copy()
        UnitsHandling.convert_data_to_standard_units(data=self.imported_data) 
        self.imported_data['nodes'] = self.imported_data['converted_nodes'].copy()
        self.imported_data['elements'] = self.imported_data['converted_elements'].copy()
        self.imported_data['materials'] = self.imported_data['converted_materials'].copy()
        self.imported_data['cross_sections'] = self.imported_data['converted_cross_sections'].copy() 
        self.imported_data['distributed_loads'] = self.imported_data['converted_distributed_loads'].copy() 

        # ---------------------------------------------
        # DISTRIBUTED LOAD CONVERSION
        # ---------------------------------------------

        converter: DistributedLoadConverter = DistributedLoadConverter(self.imported_data)
        converter.convert_distributed_loads()
        self.imported_data = converter.data.copy()

        # ---------------------------------------------
        # SETTINGS MANAGEMENT
        # ---------------------------------------------

        self.solver_settings: Dict = self._ensure_solver_settings(solver_settings)
        self.mesh_settings: Dict = mesh_settings or self._default_mesh_settings()

        # ---------------------------------------------
        # ATTRIBUTE INITIALIZATION
        # ---------------------------------------------

        self.K_global: Union[np.ndarray, spmatrix] = None
        self.element_handlers: Dict = None
        self.solver: Union[DirectSolver, IterativeSolver] = None

        # ---------------------------------------------
        # INPUT VALIDATION
        # ---------------------------------------------

        self._validate_inputs()

    # ---------------------------------------------
    # DEFAULT SETTINGS
    # ---------------------------------------------

    def _ensure_solver_settings(self, solver_settings: Dict) -> Dict:
        """
        Ensures all necessary keys are present in `solver_settings` by adding missing defaults.
        Args:
            solver_settings (Dict): The user-provided solver settings.
        Returns:
            Dict: The `solver_settings` dictionary updated with default values for any missing keys.
        """
        default_settings: Dict = self._default_solver_settings()

        if solver_settings is None or solver_settings == {}:
            return default_settings

        for key, default_value in default_settings.items():

            if key not in solver_settings:
                solver_settings[key] = default_value
        return solver_settings

    def _default_solver_settings(self) -> Dict:
        """
        Returns a dictionary of default solver settings.
        Returns:
            Dict: Default solver settings.
        """
        return {

            'solver_type': 'Direct',
            'tolerance': 1e-6,
            'max_iterations': 100,
            'sparse_matrix': True,
            'large_deflection': False
        }


    def _default_mesh_settings(self) -> Dict:
        """
        Returns a dictionary of default mesh settings based on the structure type.
        Returns:
            Dict: Default mesh settings.
        """
        structure_type: str = self.imported_data['structure_info']['element_type']
        element_type: str = ''

        if structure_type.endswith('Truss'):
            element_type = 'Bar'
        elif structure_type.endswith('Beam'):
            element_type = 'Beam'
        elif structure_type.endswith('Frame'):
            element_type = 'Frame'                        
        return {

            'element_type': element_type,
            'element_order': 'Linear',
            'mesh_density': 10,
            'algorithm': 'Automatic',
            'refinement_factor': 1.0,
            'non_conforming': False,
            'auto_remesh': True
        }

    # ---------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------

    def _validate_inputs(self) -> None:
        """
        Validates the integrity and completeness of the imported data and settings dictionaries.
        Raises:
            ValueError: If any required structure data is missing, or if solver/element types are invalid.
        """
        required_structure_keys: set[str] = {'nodes', 'elements', 'materials', 'cross_sections'}

        if not required_structure_keys.issubset(self.imported_data.keys()):
            missing: set[str] = required_structure_keys - set(self.imported_data.keys())
            ErrorHandler.handle_validation_error(
                issue_type="missing_data",
                details=f"Missing required structure data: {missing}",
                parent=None
            )
        valid_solver_types: set[str] = {'Direct', 'Iterative'}

        if self.solver_settings['solver_type'] not in valid_solver_types:
            ErrorHandler.handle_solver_error(
                solver_type="linear",
                details=f"Invalid solver type '{self.solver_settings['solver_type']}'. Must be one of {valid_solver_types}",
                parent=None
            )
        valid_element_types: set[str] = {'Bar', 'Beam', 'Frame', 'Q4', 'Q8', 'T3', 'T6'}

        if self.mesh_settings['element_type'] not in valid_element_types:
            ErrorHandler.handle_mesh_error(
                error_type="compatibility",
                details=f"Invalid element type '{self.mesh_settings['element_type']}'. Must be one of {valid_element_types}",
                parent=None
            )

    # ---------------------------------------------
    # ELEMENT HANDLER MANAGEMENT
    # ---------------------------------------------

    def _initialize_element_handlers(self) -> None:
        """
        Initializes the appropriate element handlers based on mesh settings.
        Creates a mapping between element types and their corresponding handler classes.
        Raises:
            ValueError: If the requested element type or order is not supported.
        """
        self.element_handlers = {
            'Bar': {
                'Linear': LinearBarElement,
                'Quadratic': LinearBarElement
            },
            'Beam': {
                'Linear': LinearBeamElement,
                'Quadratic': LinearBeamElement
            },
            'Frame': {
                'Linear': LinearFrameElement,
                'Quadratic': LinearFrameElement  
            },
        }
        elem_type: str = self.mesh_settings['element_type']
        elem_order: str = self.mesh_settings['element_order']

        if elem_type not in self.element_handlers:
            ErrorHandler.handle_mesh_error(
                error_type="compatibility",
                details=f"Unsupported element type: {elem_type}",
                parent=None
            )

        if elem_order not in self.element_handlers[elem_type]:
            ErrorHandler.handle_mesh_error(
                error_type="compatibility",
                details=f"Unsupported element order {elem_order} for type {elem_type}",
                parent=None
            )

    # ---------------------------------------------
    # STIFFNESS MATRIX ASSEMBLY
    # ---------------------------------------------

    def assemble_global_stiffness(self, imported_data: Dict) -> Union[np.ndarray, spmatrix]:
        """
        Assembles the global stiffness matrix based on element handlers and mesh settings.
        Args:
            imported_data (Dict): The imported structural data, including nodes, elements, materials, etc.
        Returns:
            Union[np.ndarray, spmatrix]: The assembled global stiffness matrix, either as a dense NumPy array or a sparse matrix,
                                         depending on `solver_settings['sparse_matrix']`.
        """
        self._initialize_element_handlers()
        elem_type: str = self.mesh_settings['element_type']
        elem_order: str = self.mesh_settings['element_order']
        ElementHandler = self.element_handlers[elem_type][elem_order]
        handler = ElementHandler(imported_data)
        K_global: Union[np.ndarray, spmatrix] = handler.assemble_global_stiffness(self.solver_settings['sparse_matrix'])
        return K_global


    # ---------------------------------------------
    # SOLUTION PROCESS
    # ---------------------------------------------

    def solve(self) -> Tuple[Dict, Dict]:
        """
        Executes the complete finite element solution process.
        Returns:
            Tuple[Dict, Dict]: A tuple containing two dictionaries:
                - solution (Dict): Contains the solved data including:
                    - 'solution_data' (Dict): Updated node data with forces and displacements.
                    - 'elements_forces' (Dict): Element-level results (stresses, strains, internal forces).
                    - 'global_stiffness_matrix' (Union[np.ndarray, spmatrix]): The assembled global stiffness matrix.
                    - 'displacements' (Dict): Nodal displacement results.
                    - 'forces' (Dict): Nodal force results.
                    - 'solver_info' (Dict): Solver performance metrics.
                - solution_report (Dict): A dictionary containing the generated FE report.
        """

        # ---------------------------------------------
        # PRE-PROCESSING
        # ---------------------------------------------
        header = "#---------------------------------------------#"
        print(header)
        prefix = "# Solving system started..."
        total_width = len(header) - 1
        padding_needed = total_width - len(prefix) 
        print(f"{prefix}{' ' * padding_needed}#")
        print(header)
        print("Assembling global stiffness matrix...")
        self.K_global = self.assemble_global_stiffness(self.imported_data)
        # ---------------------------------------------
        # SOLUTION
        # ---------------------------------------------
        forces: Dict[int, Dict[str, float]] = {}
        displacements: Dict[int, Dict[str, float]] = {}
        iterations: int = 0
        residual: float = 0.0

        if self.solver_settings['solver_type'] == 'Direct':
            print("Selecting direct solver...")
            self.solver = DirectSolver(
                K_global=self.K_global,
                imported_data=self.imported_data,
                tolerance=self.solver_settings['tolerance'],
                large_deflection=self.solver_settings['large_deflection'],
                sparse_matrix=self.solver_settings['sparse_matrix']
            )
            iterations = 1
            forces, displacements, residual = self.solver.solve()
        else:
            print("Selecting iterative solver...")
            self.solver = IterativeSolver(
                K_global=self.K_global,
                imported_data=self.imported_data,
                tolerance=self.solver_settings['tolerance'],
                max_iterations=self.solver_settings['max_iterations'],
                large_deflection=self.solver_settings['large_deflection'],
                sparse_matrix=self.solver_settings['sparse_matrix']
            )
            forces, displacements, iterations, residual = self.solver.solve()
        # ---------------------------------------------
        # POST-PROCESSING
        # ---------------------------------------------
        solution_data: Dict = self.update_node_data(forces, displacements)
        print("Computing element results...")
        elements_forces: Dict = self._compute_element_results(displacements)
        is_valid = self.solver.verify_solution(forces, displacements)
        self.print_solution_results(forces, displacements, iterations, residual)
        solution: Dict = {
            'solution_data': solution_data,
            'elements_forces': elements_forces,
            'global_stiffness_matrix': self.K_global,
            'displacements': displacements,
            'forces': forces,
            'solver_info': self._get_solver_info(is_valid, iterations, residual)
        }
        self.print_element_result(elements_forces)
        report_viewer: ReportViewer = ReportViewer(solution, self.imported_data)
        solution_report: Dict = report_viewer.generate_fe_report()

        if is_valid:
            print("##### 🟢 Structure SOLVED successfully 🟢 #####\n")
        else:
            error_msg = "##### 🔴 Structure FAILED 🔴 #####\n"
            print(error_msg)
            ErrorHandler.handle_solver_error(
                solver_type=self.solver_settings['solver_type'].lower(),
                details=error_msg,
                parent=None,
            )
        return solution, solution_report

    def _compute_element_results(self, displacements: Dict[int, Dict[str, float]]) -> Dict:
        """
        Computes element-level results (stresses, strains, internal forces) from nodal displacements,
        accounting for distributed loads if present.
        Args:
            displacements (Dict[int, Dict[str, float]]): Dictionary of nodal displacements,
                                                       keyed by node ID, with a dictionary of component
                                                       displacements (e.g., 'u', 'v', 'w', 'ɸx', 'ɸy', 'ɸz').
        Returns:
            Dict: A dictionary of element results keyed by element ID. Each element's result
                  includes 'stresses', 'strains', 'nodal_forces', and 'internal_forces'.
        """
        element_results: Dict = {}
        elem_type: str = self.mesh_settings['element_type']
        elem_order: str = self.mesh_settings['element_order']
        ElementHandler = self.element_handlers[elem_type][elem_order]
        distributed_loads: Dict = self.imported_data.get('distributed_loads', {})

        for elem_id, elem_data in self.imported_data['elements'].items():

            if elem_id == 0:
                continue
            handler = ElementHandler(self.imported_data)
            node_ids: list[int] = [elem_data['node1'], elem_data['node2']]
            elem_disp: list[float] = []

            for node_id in node_ids:

                if node_id in displacements:
                    elem_disp.extend(displacements[node_id].values())
                else:
                    ErrorHandler.handle_warning(
                        code=1006,
                        details=f"Nodal displacement data not found for node ID {node_id} during element result computation for element {elem_id}.",
                        parent=None
                    )
                    elem_disp.extend([0.0] * len(self.imported_data['structure_info']['displacement_labels']))

            if not elem_disp:
                ErrorHandler.handle_warning(
                    code=1007,
                    details=f"No displacement data could be retrieved for element {elem_id}. Skipping element result computation.",
                    parent=None
                )
                continue
            elem_disp_array: np.ndarray = np.array(elem_disp)
            stresses: Dict = handler.compute_stresses(elem_id, elem_disp_array)
            strains: Dict = handler.compute_strains(elem_id, elem_disp_array)
            internal_forces_raw: np.ndarray = handler.compute_internal_forces(elem_id, elem_disp_array)
            element_forces: Dict = self._compute_element_forces(elem_type, internal_forces_raw)

            if elem_id in distributed_loads:

                if 'feq1' in elem_data and 'feq2' in elem_data:
                    equivalent_nodal_forces_1: np.ndarray = np.array(elem_data['feq1'])
                    equivalent_nodal_forces_2: np.ndarray = np.array(elem_data['feq2'])
                    equivalent_nodal_forces_total: np.ndarray = np.concatenate((equivalent_nodal_forces_1, equivalent_nodal_forces_2))
                    internal_forces_adjusted: np.ndarray = internal_forces_raw - equivalent_nodal_forces_total
                else:
                    ErrorHandler.handle_warning(
                        code=1008,
                        details=f"Distributed load equivalent nodal forces (feq1 or feq2) not found for element {elem_id}, adjustment skipped.",
                        parent=None
                    )
                    internal_forces_adjusted: np.ndarray = internal_forces_raw
            else:
                internal_forces_adjusted: np.ndarray = internal_forces_raw
            element_results[elem_id] = {
                'stresses': stresses,
                'strains': strains,
                'nodal_forces': internal_forces_adjusted.tolist(),
                'internal_forces': element_forces
            }
        return element_results

    def _compute_element_forces(self, element_type: str, internal_forces: np.ndarray) -> Dict:
        """
        Computes axial force, shear, and moment for an element based on its type from internal forces.
        Args:
            element_type (str): The type of the element (e.g., 'Bar', 'Beam', 'Frame').
            internal_forces (np.ndarray): The element's internal force vector (adjusted for distributed loads).
                                         This vector contains forces/moments at the element's nodes in its local coordinate system.
        Returns:
            Dict: A dictionary containing categorized element forces/moments.
                  - 'axial_force' (float): Axial force (positive for tension).
                  - 'shear_force' (float): Shear force (for 2D beams).
                  - 'shear_force_y' (float): Shear force in y-direction (for 3D frames).
                  - 'shear_force_z' (float): Shear force in z-direction (for 3D frames).
                  - 'torsional_moment' (float): Torsional moment (for 3D frames).
                  - 'bending_moment' (float): Bending moment (for 2D beams).
                  - 'bending_moment_y' (float): Bending moment about y-axis (for 3D frames).
                  - 'bending_moment_z' (float): Bending moment about z-axis (for 3D frames).
                  - 'internal_forces' (np.ndarray): For continuum elements, the raw internal forces vector.
        Raises:
            ValueError: If the element type is unsupported.
        """
        results: Dict[str, Union[float, np.ndarray]] = {}

        if element_type == 'Bar':

            if len(internal_forces) >= 1:
                results['axial_force'] = float(internal_forces[0])
            else:
                ErrorHandler.handle_warning(
                    code=1009,
                    details=f"Insufficient internal force data for Bar element. Expected at least 1 component, got {len(internal_forces)}.",
                    parent=None
                )
                results['axial_force'] = 0.0
        elif element_type == 'Beam':

            if len(internal_forces) >= 3:
                results['axial_force'] = float(internal_forces[0])
                results['shear_force'] = float(internal_forces[1])
                results['bending_moment'] = float(internal_forces[2])
            else:
                ErrorHandler.handle_warning(
                    code=1010,
                    details=f"Insufficient internal force data for Beam element. Expected at least 3 components, got {len(internal_forces)}.",
                    parent=None
                )
                results.update({'axial_force': 0.0, 'shear_force': 0.0, 'bending_moment': 0.0})
        elif element_type == 'Frame':

            if len(internal_forces) >= 6:
                results['axial_force'] = float(internal_forces[0])
                results['shear_force_y'] = float(internal_forces[1])
                results['shear_force_z'] = float(internal_forces[2])
                results['torsional_moment'] = float(internal_forces[3])
                results['bending_moment_y'] = float(internal_forces[4])
                results['bending_moment_z'] = float(internal_forces[5])
            else:
                ErrorHandler.handle_warning(
                    code=1011,
                    details=f"Insufficient internal force data for Frame element. Expected at least 6 components, got {len(internal_forces)}.",
                    parent=None
                )
                results.update({
                    'axial_force': 0.0, 'shear_force_y': 0.0, 'shear_force_z': 0.0,
                    'torsional_moment': 0.0, 'bending_moment_y': 0.0, 'bending_moment_z': 0.0
                })
        elif element_type in ['Plane', 'Solid']:
            results['internal_forces'] = internal_forces
        else:
            ErrorHandler.handle_error(
                code=110,
                details=f"Unsupported element type for force computation: {element_type}",
                fatal=True,
                exception_type=ValueError,
                parent=None
            )
        return results

    def _get_solver_info(self, is_valid: bool, iterations: int, residual: float) -> Dict:
        """
        Collects and returns solver performance metrics.
        Args:
            is_valid (bool): A boolean indicating if the solution is considered valid.
            iterations (int): The number of iterations taken by the solver. For direct solvers, this is typically 1.
            residual (float): The final residual value after solving. For direct solvers, this is typically 0.
        Returns:
            Dict: A dictionary containing solver information:
                - 'method' (str): 'Direct' or 'Iterative'.
                - 'iterations' (int): Number of iterations.
                - 'residual' (float): Final residual.
                - 'is_valid' (bool): Whether the solution is considered valid.
        """

        if isinstance(self.solver, DirectSolver):
            return {

                'method': 'Direct', 
                'iterations': 1, 
                'residual': 0.0,
                'is_valid': is_valid
            }
        else:
            return {

                'method': 'Iterative',
                'iterations': iterations,
                'residual': residual,
                'is_valid': is_valid
            }


    def update_node_data(self, forces: Dict[int, Dict[str, float]], displacements: Dict[int, Dict[str, float]]) -> Dict:
        """
        Updates the 'force' and 'displacement' tuples in `imported_data['nodes']`
        using information from the solved `forces` and `displacements` and
        `imported_data['structure_info']`.
        Args:
            forces (Dict[int, Dict[str, float]]): Dictionary of solved nodal forces,
                                                 keyed by node ID, with a dictionary of component
                                                 forces (e.g., 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz').
            displacements (Dict[int, Dict[str, float]]): Dictionary of solved nodal displacements,
                                                       keyed by node ID, with a dictionary of component
                                                       displacements (e.g., 'u', 'v', 'w', 'ɸx', 'ɸy', 'ɸz').
        Returns:
            Dict: The `imported_data` dictionary with updated nodal force and displacement information.
        """
        solution_data: Dict = self.imported_data
        force_labels: list[str] = solution_data['structure_info'].get('force_labels', [])
        displacement_labels: list[str] = solution_data['structure_info'].get('displacement_labels', [])

        for node_id, force_data in forces.items():

            if node_id in solution_data['nodes']:
                current_force: list[float] = list(solution_data['nodes'][node_id]['force'])

                for i, force_label in enumerate(force_labels):

                    if force_label in force_data:
                        current_force[i] = force_data[force_label] 
                    else:
                        ErrorHandler.handle_warning(
                            code=1012,
                            details=f"Force label '{force_label}' not found in solved force data for node {node_id}.",
                            parent=None
                        )
                solution_data['nodes'][node_id]['force'] = tuple(current_force)
            else:
                ErrorHandler.handle_warning(
                    code=1013,
                    details=f"Node {node_id} from solved forces not found in imported data.",
                    parent=None
                )

        for node_id, displacement_data in displacements.items():

            if node_id in solution_data['nodes']:
                current_displacement: list[float] = list(solution_data['nodes'][node_id]['displacement'])

                for i, displacement_label in enumerate(displacement_labels):
                    solved_value: Union[float, None] = displacement_data.get(displacement_label)

                    if solved_value is not None:
                        current_displacement[i] = solved_value
                    else:
                        alternative_label: str = ''

                        if displacement_label.startswith('ɸ'):
                            alternative_label = 'phi' + displacement_label[1:]
                        elif displacement_label.startswith('phi'):
                            alternative_label = 'ɸ' + displacement_label[3:]
                        solved_value_alt: Union[float, None] = displacement_data.get(alternative_label)

                        if solved_value_alt is not None:
                            current_displacement[i] = solved_value_alt
                        else:
                            ErrorHandler.handle_warning(
                                code=1014,
                                details=f"Displacement label '{displacement_label}' (and its alternative) not found in solved displacement data for node {node_id}.",
                                parent=None
                            )
                solution_data['nodes'][node_id]['displacement'] = tuple(current_displacement)
            else:
                ErrorHandler.handle_warning(
                    code=1013,
                    details=f"Node {node_id} from solved displacements not found in imported data.",
                    parent=None
                )
        return solution_data


    # ---------------------------------------------
    # PRINTING / REPORTING
    # ---------------------------------------------

    @staticmethod
    def _is_number(x: Union[str, float, int]) -> bool:
        """
        Static method: Checks if a value can be converted to a float.
        Args:
            x (Union[str, float, int]): The value to check.
        Returns:
            bool: True if the value can be converted to a float, False otherwise.
        """

        try:
            float(x)
            return True

        except (ValueError, TypeError):
            return False

    def print_element_result(self, element_results: Dict) -> None:
        """
        Prints formatted element-level results to the console, including stresses, strains,
        nodal forces, and internal forces, with units.
        Args:
            element_results (Dict): A dictionary containing element results keyed by element ID.
                                    Each element's result includes 'stresses', 'strains',
                                    'nodal_forces', and 'internal_forces'.
        """
        units_dict: Dict = self.imported_data['calc_units']


        def fmt_array(arr: Union[float, int, np.ndarray, list], precision: int = 4) -> str:
            """Formats a number or array of numbers to a string with specified precision."""

            if isinstance(arr, (float, int, np.float64, np.int64)):
                return f"{float(arr):.{precision}f}"

            elif MainSolver._is_number(arr):
                return f"{float(arr):.{precision}f}"

            elif hasattr(arr, '__iter__'):
                return ' '.join(f"{float(x):.{precision}f}" if MainSolver._is_number(x) else str(x) for x in arr)

            else:
                return str(arr)

        def section_text(label: str, values: Union[float, int, np.ndarray, list], unit: str, color: str) -> str:
            """Generates an HTML span for a section of results."""
            return (

                f"<span style='color:{color}; font-weight:bold;'>{label}:</span> "
                f"{fmt_array(values)} <span style='color:gray;'>[{unit}]</span><br>"
            )
        html: str = "<div style='font-family: Consolas, Monaco, monospace; font-size: 11px;'>"
        html += f"<br><span style='color:#1f4e79; font-size: 12px;'><b>Elements Forces Results</b></span><br>"

        for elem_id, result in element_results.items():
            html += f"<br><span style='color:#1f4e79; font-size: 11px;'><b>Element {elem_id}:</b></span><br>"
            stresses: Dict = result.get('stresses', {})
            directional: Union[np.ndarray, None] = stresses.get('directional', None)
            von_mises: Union[float, None] = stresses.get('von_mises', None)
            stress_unit: str = units_dict.get("Force/Length (F/L)", "N/m")

            if directional is not None and np.size(directional) > 0:
                directional_values: list[float] = directional.flatten().tolist()
                components_line: str = ', '.join(
                    [f"{float(val):.4f} <span style='color:gray;'>[{stress_unit}]</span>" for val in directional_values])
                html += f"<span><b style='color:#c00000;'>Stress Components:</b> {components_line}</span><br>"

            if von_mises is not None and self._is_number(von_mises):
                html += section_text("Von Mises Stress", von_mises, stress_unit, "#ff0000")
            strains: list = result.get('strains', [])
            html += section_text("Strains", strains, "-", "#548235")
            nodal_forces: list[float] = result.get('nodal_forces', [])
            per_node_labels: list[str] = self.imported_data['structure_info'].get('force_labels', [])
            num_dofs_per_node: int = len(per_node_labels)

            if isinstance(nodal_forces, (list, tuple, np.ndarray)) and num_dofs_per_node > 0:
                num_nodes: int = len(nodal_forces) // num_dofs_per_node
                html += f"<span style='color:#7030a0; font-weight:bold;'>Nodal Forces:</span><br>"

                for node_idx in range(num_nodes):

                    for dof_idx, label in enumerate(per_node_labels):
                        full_label: str = f"{label}{node_idx+1}"
                        value: float = nodal_forces[node_idx * num_dofs_per_node + dof_idx]
                        formatted_val: str = fmt_array(value)
                        unit: str = self.get_component_unit(label)
                        html += f"&nbsp;&nbsp;&nbsp;<span style='color:#7030a0;'>{full_label}:</span> {formatted_val} <span style='color:gray;'>[{unit}]</span><br>"
            else:
                default_unit: str = units_dict.get("Force (Fx,Fy,Fz)", "N")
                html += section_text("Nodal Forces", nodal_forces, default_unit, "#7030a0")
            internal_forces: Dict = result.get('internal_forces', {})
            force_unit: str = units_dict.get("Force (Fx,Fy,Fz)", "N")

            if isinstance(internal_forces, dict):
                html += f"<span style='color:#0070c0; font-weight:bold;'>Internal Forces:</span><br>"

                for label, value in internal_forces.items():
                    formatted_val: str = fmt_array(value)
                    display_label: str = label.replace('_', ' ').title()

                    if "moment" in label:
                        unit = units_dict.get("Moment (Mx,My,Mz)", "N·m")
                    elif "force" in label:
                        unit = units_dict.get("Force (Fx,Fy,Fz)", "N")
                    else:
                        unit = force_unit
                    html += f"&nbsp;&nbsp;&nbsp;<span style='color:#0070c0;'>{display_label}:</span> {formatted_val} <span style='color:gray;'>[{unit}]</span><br>"
            else:
                html += section_text("Internal Forces", internal_forces, force_unit, "#0070c0")
        html += "</div>"
        print(html)


    def print_solution_results(self, forces: Dict[int, Dict[str, float]], 
                               displacements: Dict[int, Dict[str, float]], 
                               iterations: int, 
                               residual: float) -> None:
        """
        Prints the overall solution results to the console, including solver information,
        solved nodal forces, and solved nodal displacements, with units.
        Args:
            forces (Dict[int, Dict[str, float]]): Dictionary of solved nodal forces.
            displacements (Dict[int, Dict[str, float]]): Dictionary of solved nodal displacements.
            iterations (int): The number of iterations taken by the solver.
            residual (float): The final residual value.
        """


        def format_components(components: Dict[str, float]) -> str:
            """Formats individual force/displacement components with their units."""
            parts: list[str] = []

            for comp, val in components.items():
                unit: str = self.get_component_unit(comp)
                parts.append(
                    f"<span style='font-weight:bold;'>{comp}:</span> {val:.6g} <span style='color:gray;'>[{unit}]</span>"
                )
            return ', '.join(parts)

        html: str = "<div style='font-family: Consolas, Monaco, monospace; font-size: 11px;'>"
        html += f"<br><span style='font-weight:bold; color:#1f4e79;'>Solution valid:</span> True<br>"
        html += f"<span style='font-weight:bold; color:#1f4e79;'>Converged in</span> {iterations} iterations, <span style='font-weight:bold; color:#1f4e79;'>error:</span> {residual:.6g}<br>"
        html += "<br><span style='color:#7030a0; font-weight:bold;'>Solved Forces:</span><br>"

        for node, comps in sorted(forces.items()):
            html += f"<span>Node {node}: {format_components(comps)}</span><br>"
        html += "<br><span style='color:#0070c0; font-weight:bold;'>Solved Displacements:</span><br>"

        for node, comps in sorted(displacements.items()):
            html += f"<span>Node {node}: {format_components(comps)}</span><br>"
        html += "</div>"
        print(html)


    def get_component_unit(self, comp: str) -> str:
        """
        Helper method to retrieve the appropriate unit for a given force or displacement component.
        Args:
            comp (str): The component label (e.g., 'Fx', 'u', 'Mx', 'ɸx').
        Returns:
            str: The unit string for the given component, or an empty string if not found.
        """
        units_dict: Dict = self.imported_data['calc_units']
        force_unit: str = units_dict.get("Force (Fx,Fy,Fz)", "N")
        disp_unit: str = units_dict.get("Displacement (Dx,Dy,Dz)", "m")
        length_unit: str = units_dict.get("Length (L)", "m")
        rotation_unit: str = units_dict.get("Rotation (ɸx,ɸy,ɸz)", "rad")
        moment_unit: str = f"{force_unit}·{length_unit}" if force_unit and length_unit else ''

        if comp in ('Fx', 'Fy', 'Fz'):
            return force_unit

        elif comp in ('Mx', 'My', 'Mz'):
            return moment_unit

        elif comp in ('u', 'v', 'w'):
            return disp_unit

        elif comp in ('ɸx', 'ɸy', 'ɸz', 'phix', 'phiy', 'phiz'):
            return rotation_unit

        else:

            for key, val in units_dict.items():

                if f'({comp})' in key or comp in key.lower():
                    return val

            return ''