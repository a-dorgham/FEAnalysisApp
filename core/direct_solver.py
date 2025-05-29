# =============================================
# DIRECT SOLVER CLASS FOR STRUCTURAL ANALYSIS
# =============================================
import numpy as np
from typing import Dict, Tuple
from utils.error_handler import ErrorHandler


class DirectSolver:
    """
    A class to solve for unknown forces and displacements in a structural system
    given the global stiffness matrix and boundary conditions.
    The solver handles both prescribed displacements (Dirichlet boundary conditions)
    and applied forces (Neumann boundary conditions), with NaN values indicating
    unknown quantities to be solved for.
    Attributes:
        K_global (np.ndarray): Global stiffness matrix of the structure
        imported_data (dict): Dictionary containing structural information including:
            - structure_info: Metadata about DOFs, dimensions, and labels
            - nodes: Nodal data with coordinates, forces, and displacements
        large_deflection (bool): Flag for geometrically nonlinear analysis
        sparse_matrix (bool): Flag for sparse matrices usage
        dofs_per_node (int): Number of degrees of freedom per node
        total_dofs (int): Total degrees of freedom in the system
        force_labels (list): Labels for force components
        displacement_labels (list): Labels for displacement components
    """


    def __init__(self, K_global: np.ndarray, imported_data: Dict, tolerance: float = 1e-6,
                 large_deflection: bool = False, sparse_matrix: bool = False) -> None:
        """
        Initialize the direct solver with system stiffness matrix and input data.
        Args:
            K_global: Global stiffness matrix (n x n) where n is total DOFs
            imported_data: Dictionary containing structural information and boundary conditions
            tolerance (float): Convergence tolerance for iterative solution
            large_deflection: Whether to account for geometric nonlinearity
            sparse_matrix: Whether to use sparse matrices
        """
        self.K_global = K_global
        self.imported_data = imported_data
        self.tolerance = tolerance
        self.large_deflection = large_deflection
        self.sparse_matrix = sparse_matrix
        self.dofs_per_node = imported_data['structure_info']['dofs_per_node']
        self.force_labels = imported_data['structure_info']['force_labels']
        self.displacement_labels = imported_data['structure_info']['displacement_labels']
        self.total_dofs = len(imported_data['nodes']) * self.dofs_per_node

        if K_global.shape != (self.total_dofs, self.total_dofs):
            error_msg = f"Stiffness matrix dimensions {K_global.shape} don't match expected {self.total_dofs}x{self.total_dofs}"
            ErrorHandler.handle_error(code=106, details=error_msg)
            
    # ---------------------------------------------
    # EXTRACT BOUNDARY CONDITIONS
    # ---------------------------------------------
    
    def get_boundary_conditions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Enhanced version that also identifies which reaction forces are unknown.
        Returns:
            Tuple containing:
                - F_known: Vector of known forces (NaN for unknown)
                - D_known: Vector of known displacements (NaN for unknown)
                - prescribed_dofs: Indices of DOFs with prescribed displacements
                - free_dofs: Indices of DOFs with unknown displacements
                - unknown_reaction_dofs: Indices where displacement is known but force is unknown
        """
        nodes = self.imported_data['nodes']
        num_nodes = len(nodes)
        F_known = np.full(self.total_dofs, np.nan)
        D_known = np.full(self.total_dofs, np.nan)

        for node_id, node_data in nodes.items():
            start_dof = (node_id - 1) * self.dofs_per_node

            if 'force' in node_data:

                for i in range(self.dofs_per_node):
                    F_known[start_dof + i] = node_data['force'][i]

            if 'displacement' in node_data:

                for i in range(self.dofs_per_node):
                    D_known[start_dof + i] = node_data['displacement'][i]
        prescribed_dofs = np.where(~np.isnan(D_known))[0]
        free_dofs = np.where(np.isnan(D_known))[0]
        unknown_reaction_dofs = np.intersect1d(
            prescribed_dofs,
            np.where(np.isnan(F_known))[0]
        )
        return F_known, D_known, prescribed_dofs, free_dofs, unknown_reaction_dofs


    # ---------------------------------------------
    # SOLVE SYSTEM
    # ---------------------------------------------

    def solve(self) -> Tuple[Dict, Dict]:
        """
        Enhanced solver that can handle unknown reaction forces at prescribed DOFs,
        with support for both dense and sparse matrices.
        Returns:
            Tuple containing:
                - solved_forces: Dictionary of solved forces by node
                - solved_displacements: Dictionary of solved displacements by node
        """
        F_known, D_known, prescribed_dofs, free_dofs, unknown_reaction_dofs = self.get_boundary_conditions()

        if self.sparse_matrix:
            from scipy.sparse.linalg import spsolve
            from scipy.sparse import issparse

        if self.sparse_matrix:
            K_ff = self.K_global[free_dofs, :][:, free_dofs]
            K_fp = self.K_global[free_dofs, :][:, prescribed_dofs]
            K_pf = self.K_global[prescribed_dofs, :][:, free_dofs]
            K_pp = self.K_global[prescribed_dofs, :][:, prescribed_dofs]
        else:
            K_ff = self.K_global[np.ix_(free_dofs, free_dofs)]
            K_fp = self.K_global[np.ix_(free_dofs, prescribed_dofs)]
            K_pf = self.K_global[np.ix_(prescribed_dofs, free_dofs)]
            K_pp = self.K_global[np.ix_(prescribed_dofs, prescribed_dofs)]
        D_p = D_known[prescribed_dofs]
        F_f = F_known[free_dofs]

        if np.any(np.isnan(F_f)):
            error_msg = "Cannot solve system with unknown applied forces at free DOFs"
            ErrorHandler.handle_error(code=105, details=error_msg)        

        # ---------------------------------------------
        # SOLVE FOR DISPLACEMENTS
        # ---------------------------------------------

        if len(free_dofs) > 0:

            try:

                if self.sparse_matrix:
                    D_f = spsolve(K_ff, F_f - K_fp.dot(D_p))
                else:
                    D_f = np.linalg.solve(K_ff, F_f - K_fp @ D_p)

            except (np.linalg.LinAlgError, RuntimeError) as e:
                ErrorHandler.handle_error(code=108, details=str(e))
        else:
            D_f = np.array([])
        D_full = np.zeros(self.total_dofs)
        D_full[free_dofs] = D_f
        D_full[prescribed_dofs] = D_p

        # ---------------------------------------------
        # SOLVE FOR FORCES
        # ---------------------------------------------

        if self.sparse_matrix:
            F_full = self.K_global.dot(D_full)
        else:
            F_full = self.K_global @ D_full

        if len(unknown_reaction_dofs) > 0:
            F_known[unknown_reaction_dofs] = F_full[unknown_reaction_dofs]
        solved_forces = self._vector_to_node_dict(F_full, 'force')
        solved_displacements = self._vector_to_node_dict(D_full, 'displacement')

        if self.sparse_matrix:
            residual = F_full - self.K_global.dot(D_full)
        else:
            residual = F_full - self.K_global @ D_full
        residual_norm = np.linalg.norm(residual)

        if residual_norm > self.tolerance:
            error_msg = f"Solution did not converge within the predefined residual of {residual_norm}."
            ErrorHandler.handle_error(code=111, details=error_msg)
        return solved_forces, solved_displacements, residual_norm


    # ---------------------------------------------
    # UTILITY FUNCTIONS
    # ---------------------------------------------

    def _vector_to_node_dict(self, vector: np.ndarray, quantity: str) -> Dict[int, Dict[str, float]]:
        """
        Convert a DOF vector to a node-based dictionary with proper labels.
        Args:
            vector: Full DOF vector (forces or displacements)
            quantity: Either 'force' or 'displacement' to determine labels
        Returns:
            Dictionary with node IDs as keys and component dictionaries
        """
        nodes = self.imported_data['nodes']
        result_dict = {}

        if quantity == 'force':
            labels = self.force_labels
        elif quantity == 'displacement':
            labels = self.displacement_labels
        else:
            ErrorHandler.handle_error(code=112)

        for node_id in nodes.keys():
            start_dof = (node_id - 1) * self.dofs_per_node
            end_dof = start_dof + self.dofs_per_node
            values = vector[start_dof:end_dof]
            component_dict = {label: float(value) for label, value in zip(labels, values)}
            result_dict[node_id] = component_dict
        return result_dict


    # ---------------------------------------------
    # RESULTS VERIFICATION
    # ---------------------------------------------

    def verify_solution(self, solved_forces: Dict, solved_displacements: Dict) -> bool:
        """
        Verify that the solution satisfies equilibrium and boundary conditions.
        Args:
            solved_forces: Dictionary of solved forces
            solved_displacements: Dictionary of solved displacements
        Returns:
            True if solution is valid, False otherwise
        """
        F_full = np.zeros(self.total_dofs)
        D_full = np.zeros(self.total_dofs)

        for node_id in self.imported_data['nodes'].keys():
            start_dof = (node_id - 1) * self.dofs_per_node

            if node_id in solved_forces:

                for i, label in enumerate(self.force_labels):
                    F_full[start_dof + i] = solved_forces[node_id][label]

            if node_id in solved_displacements:

                for i, label in enumerate(self.displacement_labels):
                    D_full[start_dof + i] = solved_displacements[node_id][label]
        residual = self.K_global @ D_full - F_full
        max_error = np.max(np.abs(residual))
        _, D_known, prescribed_dofs, _, _ = self.get_boundary_conditions()
        bc_error = np.max(np.abs(D_full[prescribed_dofs] - D_known[prescribed_dofs]))
        tolerance = 1e-8 * self.total_dofs

        if max_error > tolerance or bc_error > tolerance:
            error_msg = f"Solution verification failed. Max residual: {max_error:.2e}, BC error: {bc_error:.2e}"
            ErrorHandler.handle_error(code=111, details=error_msg)
            return False

        return True



# =============================================
# EXAMPLE USAGE
# =============================================

if __name__ == "__main__":
    imported_data = {
        'structure_info': {
            'dofs_per_node': 3, 
            'dimension': '2D', 
            'force_labels': ['Fx', 'Fy', 'M'], 
            'displacement_labels': ['u', 'v', 'ɸ'], 
            'element_type': '2D_Beam'
            }, 
        'nodes': {
            1: {'X': 0.0, 'Y': 0.0, 'force': (np.nan, np.nan, 0.0), 'displacement': (0.0, 0.0, np.nan)}, 
            2: {'X': 1.0, 'Y': 0.0, 'force': (0.0, 0.0, 0.0), 'displacement': (np.nan, np.nan, np.nan)}, 
            3: {'X': 4.0, 'Y': 0.0, 'force': (0.0, 0.0, 0.0), 'displacement': (np.nan, np.nan, np.nan)}, 
            4: {'X': 5.0, 'Y': 0.0, 'force': (0.0, -36000.0, 20000.0), 'displacement': (np.nan, np.nan, np.nan)}, 
            5: {'X': 6.0, 'Y': 0.0, 'force': (0.0, np.nan, 0.0), 'displacement': (np.nan, 0.0, np.nan)}
            }
    }
    K_global = np.array([
        [6.59734457e+05, 0.00000000e+00, 0.00000000e+00, -6.59734457e+05,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 1.97920337e+08, 9.89601686e+07, 0.00000000e+00,
        -1.97920337e+08, 9.89601686e+07, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 9.89601686e+07, 6.59734457e+07, 0.00000000e+00,
        -9.89601686e+07, 3.29867229e+07, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [-6.59734457e+05, 0.00000000e+00, 0.00000000e+00, 8.79645943e+05,
        0.00000000e+00, 0.00000000e+00, -2.19911486e+05, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, -1.97920337e+08, -9.89601686e+07, 0.00000000e+00,
        2.05250720e+08, -8.79645943e+07, 0.00000000e+00, -7.33038286e+06,
        1.09955743e+07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 9.89601686e+07, 3.29867229e+07, 0.00000000e+00,
        -8.79645943e+07, 8.79645943e+07, 0.00000000e+00, -1.09955743e+07,
        1.09955743e+07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.19911486e+05,
        0.00000000e+00, 0.00000000e+00, 8.79645943e+05, 0.00000000e+00,
        0.00000000e+00, -6.59734457e+05, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        -7.33038286e+06, -1.09955743e+07, 0.00000000e+00, 2.05250720e+08,
        8.79645943e+07, 0.00000000e+00, -1.97920337e+08, 9.89601686e+07,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.09955743e+07, 1.09955743e+07, 0.00000000e+00, 8.79645943e+07,
        8.79645943e+07, 0.00000000e+00, -9.89601686e+07, 3.29867229e+07,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, -6.59734457e+05, 0.00000000e+00,
        0.00000000e+00, 1.31946891e+06, 0.00000000e+00, 0.00000000e+00,
        -6.59734457e+05, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.97920337e+08,
        -9.89601686e+07, 0.00000000e+00, 3.95840674e+08, 0.00000000e+00,
        0.00000000e+00, -1.97920337e+08, 9.89601686e+07],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.89601686e+07,
        3.29867229e+07, 0.00000000e+00, 0.00000000e+00, 1.31946891e+08,
        0.00000000e+00, -9.89601686e+07, 3.29867229e+07],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, -6.59734457e+05, 0.00000000e+00, 0.00000000e+00,
        6.59734457e+05, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, -1.97920337e+08, -9.89601686e+07,
        0.00000000e+00, 1.97920337e+08, -9.89601686e+07],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 9.89601686e+07, 3.29867229e+07,
        0.00000000e+00, -9.89601686e+07, 6.59734457e+07]
    ])
    solver = DirectSolver(K_global, imported_data)
    forces, displacements = solver.solve()
    is_valid = solver.verify_solution(forces, displacements)
    print(f"Solution is valid: {is_valid}")
    print("\nSolved Forces:")

    for node, components in forces.items():
        print(f"Node {node}: {components}")
    print("\nSolved Displacements:")

    for node, components in displacements.items():
        print(f"Node {node}: {components}")