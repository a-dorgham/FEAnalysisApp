import numpy as np
from typing import Tuple, Dict
from scipy.sparse import lil_matrix


class LinearBarElement:
    """
    A class for computing the stiffness matrix of 1D bar/truss elements and assembling the global stiffness matrix.
    Attributes:
        imported_data (Dict): A dictionary containing the structural information, including nodes, elements,
                                 cross-sections, material properties, and boundary conditions.
    """

    def __init__(self, imported_data: Dict):
        """
        Initializes the LinearBarElement class with the structural data.
        Args:
            imported_data (Dict): A dictionary holding all the structural information.
        """
        self.imported_data = imported_data

    # ---------------------------------------------
    # SHAPE FUNCTIONS & DERIVATIVES
    # ---------------------------------------------

    def shape_functions_1D_bar(self, xi: float) -> np.ndarray:
        """
        Returns the shape functions for a 1D linear bar/truss element in natural coordinates.
        Args:
            xi (float): Natural coordinate (-1 to 1).
        Returns:
            np.ndarray: Shape functions evaluated at xi, shape (2,).
        """
        N1 = 0.5 * (1 - xi)
        N2 = 0.5 * (1 + xi)
        return np.array([N1, N2])

    def dshape_functions_1D_bar(self) -> np.ndarray:
        """
        Returns the derivatives of shape functions with respect to natural coordinate xi for a 1D bar/truss.
        Returns:
            np.ndarray: Derivatives of shape functions w.r.t xi, shape (2,).
        """
        return np.array([-0.5, 0.5])


    # ---------------------------------------------
    # JACOBIAN CALCULATION
    # ---------------------------------------------

    def jacobian_bar(self, dN_dxsi: np.ndarray, node_coords: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Computes the Jacobian for a 1D bar/truss element.
        Args:
            dN_dxsi (np.ndarray): Derivatives of shape functions w.r.t xi, shape (2,).
            node_coords (np.ndarray): Node coordinates, shape (2, n_dim).
        Returns:
            Tuple[np.ndarray, float, np.ndarray]:
                - J (np.ndarray): Jacobian vector, shape (dim,).
                - J_norm (float): Element length (norm of Jacobian vector).
                - J_inv (np.ndarray): Inverse of Jacobian vector (for projecting strain), shape (dim,).
        Raises:
            ValueError: If the Jacobian vector norm is close to zero (degenerate element).
        """
        J = dN_dxsi @ node_coords
        J_norm = np.linalg.norm(J)

        if J_norm < 1e-12:
            raise ValueError("Degenerate element: Jacobian vector is near zero.")
        J_inv = J / (J_norm ** 2)
        return J, J_norm, J_inv


    # ---------------------------------------------
    # STRAIN-DISPLACEMENT MATRIX (B)
    # ---------------------------------------------

    def compute_B_matrix(self, dN_dxi: np.ndarray, J_inv: np.ndarray) -> np.ndarray:
        """
        Computes the B matrix (strain-displacement matrix) for a 1D truss element.
        Args:
            dN_dxi (np.ndarray): Derivatives of shape functions w.r.t xi, shape (2,).
            J_inv (np.ndarray): Inverse of the Jacobian vector, shape (dim,).
        Returns:
            np.ndarray: Strain-displacement matrix, shape (1, 2*dim).
        """
        dim = len(J_inv)
        B = np.zeros((1, 2 * dim))

        for i in range(dim):
            B[0, i] = dN_dxi[0] * J_inv[i]
            B[0, dim + i] = dN_dxi[1] * J_inv[i]
        return B


    # ---------------------------------------------
    # MATERIAL MATRIX (C)
    # ---------------------------------------------

    def material_matrix_1D(self, E: float) -> np.ndarray:
        """
        Returns the constitutive matrix (material matrix) for a 1D truss element (only axial).
        Args:
            E (float): Young's modulus.
        Returns:
            np.ndarray: Material matrix, shape (1, 1).
        """
        return np.array([[E]])


    # ---------------------------------------------
    # CHECK & REORDER NODES
    # ---------------------------------------------

    def check_and_reorder_nodes_bar(self, node_coords: np.ndarray) -> np.ndarray:
        """
        Ensures nodes of a 1D bar element are ordered from left to right (or increasing x-coordinate).
        Reorders the node coordinates if necessary.
        Args:
            node_coords (np.ndarray): Node coordinates, shape (2, n_dim).
        Returns:
            np.ndarray: Possibly reordered node coordinates, shape (2, n_dim).
        """

        if node_coords[1][0] < node_coords[0][0]:
            return node_coords[::-1]

        return node_coords


    # ---------------------------------------------
    # ELEMENT STIFFNESS MATRIX
    # ---------------------------------------------

    def stiffness_matrix_bar(self, node_coords: np.ndarray, A: float, E: float, n_dim: int) -> np.ndarray:
        """
        Computes the stiffness matrix of a 1D bar/truss element using isoparametric formulation.
        Args:
            node_coords (np.ndarray): Node coordinates, shape (2, n_dim).
            A (float): Cross-sectional area.
            E (float): Young's modulus.
            n_dim (int): Problem dimensionality.
        Returns:
            np.ndarray: Element stiffness matrix, shape (2*n_dim, 2*n_dim).
        """
        node_coords = self.check_and_reorder_nodes_bar(node_coords)
        xi = 0.0
        N = self.shape_functions_1D_bar(xi)
        dN_dxi = self.dshape_functions_1D_bar()
        J_vec, J_norm, J_inv = self.jacobian_bar(dN_dxi, node_coords)
        B = self.compute_B_matrix(dN_dxi, J_inv)
        C = self.material_matrix_1D(E)
        ke = 2 * A * J_norm * B.T @ C @ B
        return C, B, ke


    # ---------------------------------------------
    # GLOBAL ELEMENT STIFFNESS MATRIX
    # ---------------------------------------------

    def assemble_global_stiffness(self, sparse=False) -> np.ndarray:
        """
        Assembles the global stiffness matrix for the entire bar/truss system.
        Returns:
            np.ndarray: The global stiffness matrix, shape (ndof, ndof), where ndof is the total number of degrees of freedom.
        """
        nodes = self.imported_data['nodes']
        elements = self.imported_data['elements']
        n_nodes = len(nodes)

        if not nodes:
            return np.array([[]])

        dofs_per_node = self.imported_data['structure_info']['dofs_per_node']
        ndof = dofs_per_node * n_nodes

        if sparse:
            K_global = lil_matrix((ndof, ndof))
        else:
            K_global = np.zeros((ndof, ndof))

        for element_id, element in elements.items():

            if element_id == 0:
                continue
            node1_id = element['node1']
            node2_id = element['node2']
            node1 = nodes[node1_id]
            node2 = nodes[node2_id]

            if dofs_per_node == 3:
                coords = np.array([
                    [node1['X'], node1['Y'], node1['Z']],
                    [node2['X'], node2['Y'], node2['Z']]
                ], dtype=float)
            elif dofs_per_node == 2:
                coords = np.array([
                    [node1['X'], node1['Y']],
                    [node2['X'], node2['Y']]
                ], dtype=float)
            else:
                raise ValueError(f"Unsupported element type with dofs per node = {dofs_per_node}")
            A = element['A']
            E = element['E']
            C, B, K_e = self.stiffness_matrix_bar(coords, A, E, dofs_per_node)
            elements[element_id]['C'] = C
            elements[element_id]['B'] = B
            elements[element_id]['k'] = K_e
            dof_map = []

            for node_id in [node1_id, node2_id]:
                dof_map.extend([(node_id - 1) * dofs_per_node + i for i in range(dofs_per_node)])

            for i in range(len(dof_map)):

                for j in range(len(dof_map)):
                    K_global[dof_map[i], dof_map[j]] += K_e[i, j]

        if sparse:
            K_global = K_global.tocsr()
        return K_global

    def compute_strains(self, elem_id, elem_disp: np.ndarray) -> np.ndarray:
        """
        Compute strains at integration points using the strain-displacement matrix B.
        Args:
            elem_disp: Element displacement vector
        Returns:
            Strains at integration points (shape depends on element type)
        """
        B = self.imported_data['elements'][elem_id]['B']
        strains = np.dot(B, elem_disp)
        return strains

    def compute_stresses(self, elem_id, elem_disp: np.ndarray) -> np.ndarray:
        """
        Compute stresses at integration points including von Mises equivalent stress.
        Args:
            elem_id: Element ID
            elem_disp: Element displacement vector
        Returns:
            Array containing both principal stresses and von Mises stress in the format:
            [σ_xx, σ_yy, τ_xy, σ_vonMises] for 2D elements or
            [σ_xx, σ_vonMises] for 1D elements
        """
        strains = self.compute_strains(elem_id, elem_disp)
        D = self.imported_data['elements'][elem_id]['C']
        stresses = np.dot(D, strains)
        element_type = self.imported_data['structure_info']['element_type']

        if element_type.endswith('Truss'):
            sigma_xx = stresses[0]
            von_mises = np.abs(sigma_xx)
        elif element_type.endswith(('Beam', 'Frame')):
            sigma_xx = stresses[0]
            sigma_bending = stresses[1]
            von_mises = np.sqrt(sigma_xx**2 + 3*sigma_bending**2)
        elif element_type.endswith(('Plane_Stress', 'Plane_Strain')):
            sigma_xx = stresses[0]
            sigma_yy = stresses[1]
            tau_xy = stresses[2]
            von_mises = np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*tau_xy**2)
        elif element_type.endswith('Solid'):
            sigma_xx = stresses[0]
            sigma_yy = stresses[1]
            sigma_zz = stresses[2]
            tau_xy = stresses[3]
            tau_yz = stresses[4]
            tau_xz = stresses[5]
            von_mises = np.sqrt(0.5*((sigma_xx-sigma_yy)**2 + 
                                    (sigma_yy-sigma_zz)**2 + 
                                    (sigma_zz-sigma_xx)**2 + 
                                    6*(tau_xy**2 + tau_yz**2 + tau_xz**2)))
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        return {

                'directional': stresses,
                'von_mises': von_mises
            }


    def compute_internal_forces(self, elem_id,  elem_disp: np.ndarray) -> np.ndarray:
        """
        Compute internal forces for the element using the local stiffness matrix.
        Args:
            elem_disp: Element displacement vector
        Returns:
            Internal forces vector
        """
        k = self.imported_data['elements'][elem_id]['k']
        internal_forces = np.dot(k, elem_disp)
        return internal_forces


    # ---------------------------------------------
    # Example usage
    # ---------------------------------------------
    
if __name__ == "__main__":
    imported_data = {
        'nodes': {
            1: {'X': 0.0, 'Y': 0.0, 'Z': 0.0},
            2: {'X': 1.0, 'Y': 0.0, 'Z': 0.0},
            3: {'X': 1.0, 'Y': 1.0, 'Z': 0.0}
        },
        'elements': {
            1: {'node1': 1, 'node2': 2, 'E': 200e9, 'A': np.pi * 0.05**2},
            2: {'node1': 2, 'node2': 3, 'E': 200e9, 'A': np.pi * 0.05**2}
        },
    }
    solver = LinearBarElement(imported_data)
    global_stiffness_matrix = solver.assemble_global_stiffness()
    print("Global Stiffness Matrix:")
    print(global_stiffness_matrix)