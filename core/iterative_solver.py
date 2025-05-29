# ==============================================
# ITERATIVE SOLVER CLASS FOR STRUCTURAL ANALYSIS
# ==============================================
import numpy as np
from typing import Dict, Tuple, List, Union
from utils.error_handler import ErrorHandler


class IterativeSolver:
    """
    A class to solve for unknown forces and displacements using iterative methods.
    Handles both linear and geometrically nonlinear problems (when large_deflection=True).
    Attributes:
        K_global (np.ndarray): Global stiffness matrix (linear or initial stiffness)
        imported_data (dict): Dictionary containing structural information
        tolerance (float): Convergence tolerance for iterative solution
        max_iterations (int): Maximum allowed iterations
        large_deflection (bool): Flag for geometrically nonlinear analysis
        sparse_matrix (bool): Flag for sparse matrices usage
        dofs_per_node (int): Number of degrees of freedom per node
        force_labels (list): Labels for force components
        displacement_labels (list): Labels for displacement components
        total_dofs (int): Total degrees of freedom in the system
    """


    def __init__(self, K_global: np.ndarray, imported_data: Dict,
        tolerance: float = 1e-6, max_iterations: int = 100,
        large_deflection: bool = False, sparse_matrix: bool = False) -> None:
        """
        Initialize the iterative solver.
        Args:
            K_global: Global stiffness matrix (n x n)
            imported_data: Dictionary containing structural information
            tolerance: Convergence tolerance for displacements
            max_iterations: Maximum number of iterations
            large_deflection: Whether to account for geometric nonlinearity
            sparse_matrix: Whether to use sparse matrices
        """
        self.K_global = K_global
        self.imported_data = imported_data
        self.tolerance = tolerance
        self.max_iterations = max_iterations
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
    # BOUNDARY CONDITION PROCESSING
    # ---------------------------------------------

    def get_boundary_conditions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract boundary conditions from imported data.
        Returns:
            Tuple containing:
                - F_known: Vector of known forces (NaN for unknown)
                - D_known: Vector of known displacements (NaN for unknown)
                - prescribed_dofs: Indices of DOFs with prescribed displacements
                - free_dofs: Indices of DOFs with unknown displacements
                - unknown_reaction_dofs: Indices where displacement is known but force is unknown
        """
        nodes = self.imported_data['nodes']
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
        unknown_reaction_dofs = np.intersect1d(prescribed_dofs, np.where(np.isnan(F_known))[0])
        return F_known, D_known, prescribed_dofs, free_dofs, unknown_reaction_dofs


    # ---------------------------------------------
    # ITERATIVE SOLUTION METHODS
    # ---------------------------------------------

    def solve(self) -> Tuple[Dict, Dict, int]:
        """
        Main iterative solver method.
        Returns:
            Tuple containing:
                - solved_forces: Dictionary of solved forces by node
                - solved_displacements: Dictionary of solved displacements by node
                - iterations: Number of iterations performed
                - residual: Finial residual error of the solution
        """
        F_known, D_known, prescribed_dofs, free_dofs, unknown_reaction_dofs = self.get_boundary_conditions()
        D_full = np.zeros(self.total_dofs)
        D_full[prescribed_dofs] = D_known[prescribed_dofs]
        residual_norm = float('inf')
        iterations = 0

        if self.sparse_matrix:
            from scipy.sparse.linalg import splu
        else:
            from scipy.linalg import lu_factor, lu_solve

        # ---------------------------------------------
        # PRE-COMPUTE FACTORIZATION (OUTSIDE LOOP)
        # ---------------------------------------------

        if self.large_deflection:
            ErrorHandler.handle_error(code=138, details="Geometrically nonlinear analysis not yet implemented")
        else:
            K_current = self.K_global

        if self.sparse_matrix:
            K_ff = K_current[free_dofs, :][:, free_dofs].tocsc()
            K_fp = K_current[free_dofs, :][:, prescribed_dofs]
            lu_factorization = splu(K_ff)
        else:
            K_ff = K_current[np.ix_(free_dofs, free_dofs)]
            K_fp = K_current[np.ix_(free_dofs, prescribed_dofs)]
            lu_piv = lu_factor(K_ff)
        F_f = F_known[free_dofs]

        if np.any(np.isnan(F_f)):
            error_msg = "Cannot solve system with unknown applied forces at free DOFs"
            ErrorHandler.handle_error(code=105, details=error_msg)     
        rhs_constant = F_f - K_fp @ D_known[prescribed_dofs]

        # ---------------------------------------------
        # ITERATION LOOP
        # ---------------------------------------------

        while residual_norm > self.tolerance and iterations < self.max_iterations:
            D_prev = D_full.copy()

            if self.sparse_matrix:
                D_f = lu_factorization.solve(rhs_constant)
            else:
                D_f = lu_solve(lu_piv, rhs_constant)
            D_full[free_dofs] = D_f
            delta_D = D_full - D_prev
            residual_norm = np.linalg.norm(delta_D[free_dofs])
            iterations += 1
            print(f"Iteration {iterations}: Residual norm = {residual_norm:.4g}")

        # ---------------------------------------------
        # POST-PROCESSING
        # ---------------------------------------------

        if self.sparse_matrix:
            F_full = K_current.dot(D_full)
        else:
            F_full = K_current @ D_full

        if len(unknown_reaction_dofs) > 0:
            F_known[unknown_reaction_dofs] = F_full[unknown_reaction_dofs]
        solved_forces = self._vector_to_node_dict(F_full, 'force')
        solved_displacements = self._vector_to_node_dict(D_full, 'displacement')

        if residual_norm > self.tolerance:
            error_msg = f"Warning: Solution did not converge within {self.max_iterations} iterations. Final residual = {residual_norm}."
            ErrorHandler.handle_error(code=111, details=error_msg)       
        return solved_forces, solved_displacements, iterations, residual_norm


    # ---------------------------------------------
    # UTILITY FUNCTIONS
    # ---------------------------------------------

    def _vector_to_node_dict(self, vector: np.ndarray, quantity: str) -> Dict[int, Dict[str, float]]:
        """
        Convert DOF vector to node-based dictionary with component labels.
        Args:
            vector: Full DOF vector (forces or displacements)
            quantity: 'force' or 'displacement' to determine labels
        Returns:
            Dictionary mapping node IDs to component dictionaries
        """
        nodes = self.imported_data['nodes']
        labels = self.force_labels if quantity == 'force' else self.displacement_labels
        result_dict = {}

        for node_id in nodes.keys():
            start_dof = (node_id - 1) * self.dofs_per_node
            values = vector[start_dof:start_dof + self.dofs_per_node]
            result_dict[node_id] = {label: float(value) for label, value in zip(labels, values)}
        return result_dict


    # ---------------------------------------------
    # SOLUTION VERIFICATION
    # ---------------------------------------------

    def verify_solution(self, solved_forces: Dict, solved_displacements: Dict) -> bool:
        """
        Verify solution satisfies equilibrium and boundary conditions.
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
            error_msg = f"Verification failed. Max residual: {max_error:.2e}, BC error: {bc_error:.2e}"
            ErrorHandler.handle_error(code=103, details=error_msg)     
            return False

        return True


    # ---------------------------------------------
    # NONLINEAR ANALYSIS METHODS
    # ---------------------------------------------

    def update_stiffness(self, displacements: np.ndarray) -> np.ndarray:
        """
        Compute the updated tangent stiffness matrix accounting for geometric nonlinearity.
        Uses the corotational formulation for large displacements while assuming small strains.
        The tangent stiffness consists of:
        - Material stiffness (constant)
        - Geometric stiffness (displacement-dependent)
        Args:
            displacements: Current displacement vector (nodal displacements for entire structure)
        Returns:
            Updated tangent stiffness matrix (Kₜ = Kₘ + K₉) where:
            - Kₘ = Material stiffness (constant)
            - K₉ = Geometric stiffness (function of current displacements)
        """
        K_tangent = np.zeros_like(self.K_global)
        elements = self._get_element_properties()

        for elem_id, elem_data in elements.items():
            node_i, node_j = elem_data['nodes']
            E = elem_data['E']
            A = elem_data['A']
            I = elem_data['I']
            X_i, Y_i = self.imported_data['nodes'][node_i]['X'], self.imported_data['nodes'][node_i]['Y']
            X_j, Y_j = self.imported_data['nodes'][node_j]['X'], self.imported_data['nodes'][node_j]['Y']
            dofs_i = (node_i - 1) * self.dofs_per_node
            dofs_j = (node_j - 1) * self.dofs_per_node
            u_i, v_i, θ_i = displacements[dofs_i:dofs_i+3]
            u_j, v_j, θ_j = displacements[dofs_j:dofs_j+3]
            L0 = np.sqrt((X_j - X_i)**2 + (Y_j - Y_i)**2)
            L = np.sqrt((X_j + u_j - X_i - u_i)**2 + (Y_j + v_j - Y_i - v_i)**2)
            c = (X_j + u_j - X_i - u_i)/L
            s = (Y_j + v_j - Y_i - v_i)/L
            k_material_local = np.array([
                [E*A/L, 0, 0, -E*A/L, 0, 0],
                [0, 12*E*I/L**3, 6*E*I/L**2, 0, -12*E*I/L**3, 6*E*I/L**2],
                [0, 6*E*I/L**2, 4*E*I/L, 0, -6*E*I/L**2, 2*E*I/L],
                [-E*A/L, 0, 0, E*A/L, 0, 0],
                [0, -12*E*I/L**3, -6*E*I/L**2, 0, 12*E*I/L**3, -6*E*I/L**2],
                [0, 6*E*I/L**2, 2*E*I/L, 0, -6*E*I/L**2, 4*E*I/L]
            ])
            axial_force = E*A*(L - L0)/L0
            k_geo_local = (axial_force/L) * np.array([
                [0, 0, 0, 0, 0, 0],
                [0, 6/5, L/10, 0, -6/5, L/10],
                [0, L/10, 2*L**2/15, 0, -L/10, -L**2/30],
                [0, 0, 0, 0, 0, 0],
                [0, -6/5, -L/10, 0, 6/5, -L/10],
                [0, L/10, -L**2/30, 0, -L/10, 2*L**2/15]
            ])
            T = np.array([
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            k_material = T.T @ k_material_local @ T
            k_geo = T.T @ k_geo_local @ T
            k_element = k_material + k_geo
            elem_dofs = np.concatenate([
                range((node_i-1)*self.dofs_per_node, node_i*self.dofs_per_node),
                range((node_j-1)*self.dofs_per_node, node_j*self.dofs_per_node)
            ])

            for i, dof_i in enumerate(elem_dofs):

                for j, dof_j in enumerate(elem_dofs):
                    K_tangent[dof_i, dof_j] += k_element[i, j]
        return K_tangent

    def _get_element_properties(self) -> Dict:
        """
        Helper method to extract element properties from imported_data.
        Modify according to your actual data structure.
        """
        elements = {}

        for elem_id, elem_data in self.imported_data.get('elements', {}).items():
            elements[elem_id] = {
                'nodes': elem_data['nodes'],
                'E': elem_data['material']['E'],
                'A': elem_data['section']['A'],
                'I': elem_data['section']['I']
            }
        return elements 

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
    solver = IterativeSolver(
        K_global=K_global,
        imported_data=imported_data,
        tolerance=1e-6,
        max_iterations=50,
        large_deflection=False
    )
    forces, displacements, iterations = solver.solve()
    is_valid = solver.verify_solution(forces, displacements)
    print(f"\nSolution valid: {is_valid}")
    print(f"Converged in {iterations} iterations")
    print("\nSolved Forces:")

    for node, components in forces.items():
        print(f"Node {node}: {components}")
    print("\nSolved Displacements:")

    for node, components in displacements.items():
        print(f"Node {node}: {components}")