import numpy as np
import math
from typing import Tuple, Dict
from scipy.special import roots_legendre
from scipy.sparse import lil_matrix


class LinearBeamElement:
    """
    A linear 1D beam element with 2 nodes and 3 DOFs per node (u, v, θ).
    Uses isoparametric formulation with shape functions for both geometry and field variables.
    Attributes:
        imported_data (Dict): A dictionary containing the full structure information,
                             including nodes and elements data. Expected keys:
                             'nodes': Dictionary of node IDs to their coordinates (X, Y[, Z]).
                             'elements': Dictionary of element IDs to their connectivity
                                         and properties (node1, node2, E, A, Iy).
        nodes (np.ndarray): Array of node coordinates (x, y) for the current element.
        youngs_modulus (float): Young's modulus of the material for the current element.
        area (float): Cross-sectional area of the current element.
        moment_inertia (float): Area moment of inertia of the current element (Iy for bending in xy-plane).
        length (float): Length of the current element (computed from node coordinates).
        node_order_checked (bool): Flag indicating if node order has been checked for the current element.
    """


    def __init__(self, imported_data: Dict):
        """
        Initialize the LinearBeamElement class with the imported structure data.
        Args:
            imported_data (Dict): A dictionary containing the full structure information.
        """
        self.imported_data = imported_data
        self.nodes = None
        self.elements = None
        self.youngs_modulus = None
        self.area = None
        self.moment_inertia = None
        self.length = None
        self.node_order_checked = False


    def reinit_element(self, element_id: int) -> None:
        """
        Re-initializes the element properties based on the provided element ID
        from the imported data.
        Args:
            element_id (int): The ID of the element to re-initialize.
        Returns:
            None
        """
        element_data = self.imported_data['elements'].get(element_id)

        if not element_data:
            raise ValueError(f"Element with ID {element_id} not found in imported data.")
        node1_id = element_data['node1']
        node2_id = element_data['node2']
        node_coords_dict = self.imported_data['nodes']

        if node1_id not in node_coords_dict or node2_id not in node_coords_dict:
            raise ValueError(f"Nodes {node1_id} or {node2_id} not found in imported data.")
        self.nodes = np.array([
            [node_coords_dict[node1_id]['X'], node_coords_dict[node1_id]['Y']],
            [node_coords_dict[node2_id]['X'], node_coords_dict[node2_id]['Y']]
        ])
        self.youngs_modulus = element_data['E']
        self.area = element_data['A']
        self.moment_inertia = element_data['Iz']  
        self.length = element_data.get('length', self._compute_length(self.nodes))
        self.node_order_checked = False

    # ---------------------------------------------
    # NODE ORDERING AND LENGTH COMPUTATION
    # ---------------------------------------------

    def _check_and_order_nodes(self, node_coords: np.ndarray) -> np.ndarray:
        """
        Check node coordinates and ensure they're in proper order (increasing x).
        If nodes are in reverse order, swap them.
        Args:
            node_coords (np.ndarray): Input node coordinates (2x2 array).
        Returns:
            np.ndarray: Ordered node coordinates (2x2 array).
        Raises:
            ValueError: If the input `node_coords` does not have the shape (2, 2).
        """

        if node_coords.shape != (2, 2):
            raise ValueError("Node coordinates must be a 2x2 array")

        if node_coords[0, 0] > node_coords[1, 0]:
            return np.array([node_coords[1], node_coords[0]])

        return node_coords

    def _compute_length(self, node_coords: np.ndarray) -> float:
        """
        Compute the length of the beam element from node coordinates.
        Args:
            node_coords (np.ndarray): Array of shape (2, 2) containing (x,y) coordinates for each node.
        Returns:
            float: Length of the element.
        """
        dx = node_coords[1, 0] - node_coords[0, 0]
        dy = node_coords[1, 1] - node_coords[0, 1]
        return np.sqrt(dx**2 + dy**2)

    # ---------------------------------------------
    # SHAPE FUNCTIONS & DERIVATIVES
    # ---------------------------------------------

    def get_shape_functions(self, xi: float) -> np.ndarray:
        """
        Compute shape functions for the beam element at natural coordinate xi.
        For a 2-node beam with 3 DOFs per node, we use cubic Hermitian shape functions

        for transverse displacement and rotation, and linear shape functions for axial displacement.
        Args:
            xi (float): Natural coordinate in the range [-1, 1].
        Returns:
            np.ndarray: Shape function matrix (6x1) for [u1, v1, θ1, u2, v2, θ2].
        """
        N1_u = 0.5 * (1 - xi)
        N2_u = 0.5 * (1 + xi)
        N1_v = 0.25 * (1 - xi)**2 * (2 + xi)
        N2_v = 0.25 * (1 + xi)**2 * (2 - xi)
        N1_θ = 0.125 * (1 - xi)**2 * (1 + xi) * self.length
        N2_θ = 0.125 * (1 + xi)**2 * (xi - 1) * self.length
        N = np.array([
            [N1_u, 0,    0,    N2_u, 0,    0   ],
            [0,    N1_v, N1_θ, 0,    N2_v, N2_θ],
            [0,    0,    0,    0,    0,    0   ]
        ])
        return N

    def get_shape_function_derivatives(self, xi: float, J: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives of shape functions with respect to the natural coordinate xi and the global coordinate x.
        Args:
            xi (float): Natural coordinate in the range [-1, 1].
            J (float): Jacobian determinant (scalar for 1D).
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - dN_dxi: Derivatives of shape functions with respect to xi (2x6).
                - dN_dx: Derivatives of shape functions with respect to x (2x6).
        """
        dN1_u_dxi = -0.5
        dN2_u_dxi = 0.5
        dN1_v_dxi = 0.25 * (-3 + 3 * xi**2)
        dN2_v_dxi = 0.25 * (3 - 3 * xi**2)
        dN1_θ_dxi = 0.125 * (-1 - 2 * xi + 3 * xi**2) * self.length
        dN2_θ_dxi = 0.125 * (-1 + 2 * xi + 3 * xi**2) * self.length
        dN_dxi = np.array([
            [dN1_u_dxi, 0,         0,         dN2_u_dxi, 0,         0        ],
            [0,         dN1_v_dxi, dN1_θ_dxi, 0,         dN2_v_dxi, dN2_θ_dxi],
            [0,         0,         0,         0,         0,         0        ]
        ])
        dN_dx = dN_dxi / J
        return dN_dxi, dN_dx

    # ---------------------------------------------
    # JACOBIAN COMPUTATION
    # ---------------------------------------------

    def _compute_jacobian(self, node_coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the Jacobian for coordinate transformation from natural to global coordinates.
        Args:
            node_coords (np.ndarray): Node coordinates of the element (2x2 array).
        Returns:
            Tuple[float, np.ndarray]: A tuple containing:
                - J (float): The Jacobian determinant (scalar for 1D).
                - dxdxi (np.ndarray): Derivative of physical coordinates with respect to xi (1x2 array).
        """
        dxdxi = 0.5 * (node_coords[1] - node_coords[0])
        J = np.linalg.norm(dxdxi)
        return J, dxdxi

    # ---------------------------------------------
    # STRAIN-DISPLACEMENT MATRIX
    # ---------------------------------------------

    def get_strain_displacement_matrix(self, xi: float, J: float, dxdxi: np.ndarray) -> np.ndarray:
        """
        Compute the strain-displacement matrix B that relates nodal displacements to strains
        (axial strain and curvature).
        Args:
            xi (float): Natural coordinate in the range [-1, 1].
            J (float): Jacobian (scalar).
            dxdxi (np.ndarray): Array of shape (1, 2), derivative of physical coordinates wrt xi.
        Returns:
            np.ndarray: B matrix (2x6) for axial strain (ε) and curvature (κ).
        """
        c = dxdxi[0] / J
        s = dxdxi[1] / J
        B = np.zeros((2, 6))
        dN_dxi, dN_dx = self.get_shape_function_derivatives(xi, J)
        f = np.sqrt(3)
        d2N_dx2 = dN_dxi / J**2 * f
        B[0, 0] = c * dN_dx[0, 0]
        B[0, 1] = s * dN_dx[0, 0]
        B[0, 3] = c * dN_dx[0, 3]
        B[0, 4] = s * dN_dx[0, 3]
        N = self.get_shape_functions(xi) / J * f
        B[1, 0] = -s * d2N_dx2[0, 0]
        B[1, 1] =  c * d2N_dx2[0, 0]
        B[1, 2] = -1 * N[0, 0]
        B[1, 3] = -s * d2N_dx2[0, 3]
        B[1, 4] =  c * d2N_dx2[0, 3]
        B[1, 5] = -1 * N[0, 3]
        return B

    # ---------------------------------------------
    # CONSTITUTIVE MATRIX
    # ---------------------------------------------

    def get_constitutive_matrix(self) -> np.ndarray:
        """
        Compute the constitutive matrix C that relates strains to stresses.
        For a 2D beam element, we consider axial stress and bending moment.
        Returns:
            np.ndarray: C matrix (2x2) for axial stiffness (EA) and bending stiffness (EI).
        """
        C = np.array([
            [self.youngs_modulus * self.area, 0],
            [0, self.youngs_modulus * self.moment_inertia]
        ])
        return C

    # ---------------------------------------------
    # LOCAL STIFFNESS MATRIX
    # ---------------------------------------------

    def get_local_stiffness_matrix(self, node_coords: np.ndarray) -> np.ndarray:
        """
        Compute the local stiffness matrix for the beam element using Gauss quadrature.
        Args:
            node_coords (np.ndarray): Node coordinates of the element (2x2 array).
        Returns:
            np.ndarray: Local stiffness matrix (6x6).
        """
        gauss_points, gauss_weights = roots_legendre(3)
        node_coords = self._check_and_order_nodes(node_coords)
        self.length = self._compute_length(node_coords)
        K_local = np.zeros((6, 6))
        C = self.get_constitutive_matrix()

        for xi, weight in zip(gauss_points, gauss_weights):
            J, dxdxi = self._compute_jacobian(node_coords)
            B = self.get_strain_displacement_matrix(xi, J, dxdxi)
            K_local += B.T @ C @ B * J * weight
        return C, B, K_local

    # ---------------------------------------------
    # GLOBAL STIFFNESS ASSEMBLY
    # ---------------------------------------------

    def assemble_global_stiffness(self, sparse=False) -> np.ndarray:
        """
        Assemble the global stiffness matrix from all elements defined in the imported data.
        Returns:
            np.ndarray: Global stiffness matrix (ndof x ndof), where ndof is the total
                        number of degrees of freedom in the structure.
        """
        num_nodes = len(self.imported_data['nodes'])
        dof_per_node = 3
        ndof = num_nodes * dof_per_node

        if sparse:
            K_global = lil_matrix((ndof, ndof))
        else:
            K_global = np.zeros((ndof, ndof))
        elements_data = self.imported_data['elements']
        nodes_data = self.imported_data['nodes']

        for element_id, element in elements_data.items():

            if element_id == 0:
                continue
            n1_id = element['node1']
            n2_id = element['node2']

            if n1_id not in nodes_data or n2_id not in nodes_data:
                print(f"Warning: Nodes {n1_id} or {n2_id} not found for element {element_id}")
                continue
            node_coords = np.array([
                [nodes_data[n1_id]['X'], nodes_data[n1_id]['Y']],
                [nodes_data[n2_id]['X'], nodes_data[n2_id]['Y']]
            ])
            self.youngs_modulus = element['E']
            self.area = element['A']
            self.moment_inertia = element['Iz']
            self.length = element.get('length', self._compute_length(node_coords))
            C, B, ke = self.get_local_stiffness_matrix(node_coords)
            self.imported_data['elements'][element_id]['C'] = C
            self.imported_data['elements'][element_id]['B'] = B
            self.imported_data['elements'][element_id]['k'] = ke
            dof_map = [
                3 * (n1_id - 1), 3 * (n1_id - 1) + 1, 3 * (n1_id - 1) + 2,
                3 * (n2_id - 1), 3 * (n2_id - 1) + 1, 3 * (n2_id - 1) + 2
            ]

            for i in range(6):

                for j in range(6):
                    K_global[dof_map[i], dof_map[j]] += ke[i, j]

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
    np.set_printoptions(precision=3, suppress=True)
    """Test case for a simple horizontal beam (p242)."""
    imported_data = {
        'structure_info': {'dofs_per_node': 2, 'dimension': 2, 'element_type': 'Truss'},
        'nodes': {
            1: {'X': 0.0, 'Y': 0.0},
            2: {'X': 0.707, 'Y': 0.707},
        },
        'elements': {
            1: {'node1': 1, 'node2': 2, 'E': 1.0, 'A': 1.0, 'Iy': 1.0},
        },
    }
    solver = LinearBeamElement(imported_data)
    K_global = solver.assemble_global_stiffness()
    print("Global Stiffness Matrix:")
    print(K_global)
    """Test case for a 2D frame (Ex 5.1, p243)."""
    imported_data = {
        'structure_info': {'dofs_per_node': 2, 'dimension': 2, 'element_type': 'Truss'},
        'nodes': {
            1: {'X': 0.0, 'Y': 0.0},
            2: {'X': 0.0, 'Y': 3.0},
            3: {'X': 3.0, 'Y': 3.0},
            4: {'X': 3.0, 'Y': 0.0},
        },
        'elements': {
            1: {'node1': 1, 'node2': 2, 'E': 200.0E9, 'A': 6.5E-3, 'Iy': 80E-6},
            2: {'node1': 2, 'node2': 3, 'E': 200.0E9, 'A': 6.5E-3, 'Iy': 40E-6},
            3: {'node1': 3, 'node2': 4, 'E': 200.0E9, 'A': 6.5E-3, 'Iy': 80E-6},
        },
    }
    solver = LinearBeamElement(imported_data)
    K_global = solver.assemble_global_stiffness()
    print("Global Stiffness Matrix:")
    print(K_global/66.67e6)