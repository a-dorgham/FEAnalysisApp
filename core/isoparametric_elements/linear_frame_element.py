import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from scipy.special import roots_legendre
from scipy.sparse import lil_matrix


class LinearFrameElement:
    """
    A class for 3D frame elements (2 nodes, 9 DOF per node) with full isoparametric formulation.
    Attributes:
        imported_data (Dict): Dictionary containing:
            - nodes: {node_id: {'X': float, 'Y': float, 'Z': float}}
            - elements: {elem_id: {'node1': int, 'node2': int, ...}}
            - materials: Material properties
            - cross_sections: Section properties
        dim (int): Problem dimension (3 for 3D)
        dof_per_node (int): Degrees of freedom per node (9 for 3D frame)
    """


    def __init__(self, imported_data: Dict):
        """
        Initializes the 3D frame element class.
        Args:
            imported_data (Dict): Dictionary containing structural data with:
                - nodes: Node coordinates
                - elements: Element connectivity and properties
                - materials: Material properties
                - cross_sections: Section properties
        """
        self.imported_data = imported_data
        self.dim = 3
        self.dof_per_node = 6
        self.length = 1
        self._validate_input_data()

    # ---------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------

    def _validate_input_data(self) -> None:
        """Validates the input data structure."""
        required_keys = ['nodes', 'elements']

        for key in required_keys:

            if key not in self.imported_data:
                raise ValueError(f"Missing required key in input data: {key}")

    # ---------------------------------------------
    # SHAPE FUNCTIONS & DERIVATIVES
    # ---------------------------------------------

    def linear_shape_functions(self, xi: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes shape functions and their derivatives for a 3D frame element.
        Each node has 6 DOFs: [u, v, w, rx, ry, rz] corresponding to:
            - u: axial displacement
            - v, w: transverse displacements (in y and z)
            - rx, ry, rz: rotations about x, y, z axes
        Axial displacement and torsion use linear shape functions.
        Bending in Y-Z planes use cubic Hermitian shape functions.
        Args:
            xi (float): Natural coordinate (-1 to 1)
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - N: Shape function matrix (12,) for the 12 DOFs
                - dN_dxi: Derivatives of shape functions (12,)
        """
        L = self.length
        N1_lin = (1 - xi)/2
        N2_lin = (1 + xi)/2
        dN1_lin = -0.5
        dN2_lin = 0.5
        N = np.array([
            [N1_lin], [N2_lin]
        ])
        dN_dxi = np.array([
            dN1_lin, dN2_lin
        ])
        return N, dN_dxi

    # ---------------------------------------------
    # JACOBIAN CALCULATION
    # ---------------------------------------------

    def jacobian(self, node_coords: np.ndarray, dN_dxi: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Jacobian and returns orientation information for 3D frame element.
        Args:
            node_coords (np.ndarray): (2, 3) array of node coordinates
            dN_dxi (np.ndarray): Shape function derivatives (2,)
        Returns:
            Tuple containing:
                - J: Jacobian determinant (element length/2)
                - J_inv: Inverse of Jacobian (2/element length)
                - dx_dxi: Derivative of physical position w.r.t. xi (3,)
                - lmn_x: Direction cosines for local x-axis (3,)
                - lmn_y: Direction cosines for local y-axis (3,)
                - lmn_z: Direction cosines for local z-axis (3,)
        """
        dx_dxi = np.dot(dN_dxi, node_coords)
        J = np.linalg.norm(dx_dxi)

        if J < 1e-12:
            raise ValueError("Degenerate element: Jacobian is near zero")
        J_inv = 1.0/J
        element_length = 2 * J
        lmn_x = dx_dxi / J
        ref_vector = np.array([0.0, 1.0, 0.0])

        if np.abs(np.dot(lmn_x, ref_vector)) > 0.99:
            ref_vector = np.array([0.0, 0.0, 1.0])
        lmn_z = np.cross(lmn_x, ref_vector)
        lmn_z /= np.linalg.norm(lmn_z)
        lmn_y = np.cross(lmn_z, lmn_x)
        lmn_y /= np.linalg.norm(lmn_y)
        return J, J_inv, dx_dxi, lmn_x, lmn_y, lmn_z

    # ---------------------------------------------
    # STRAIN-DISPLACEMENT MATRIX (B)
    # ---------------------------------------------

    @staticmethod
    def hermite_shape_functions(xi):
        """Returns Hermite shape functions and their derivatives up to second order."""
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = xi - 2*xi**2 + xi**3
        N3 = 3*xi**2 - 2*xi**3
        N4 = -xi**2 + xi**3
        N_bending = np.array([N1, N2, N3, N4])
        dN1_dxi = -6*xi + 6*xi**2
        dN2_dxi = 1 - 4*xi + 3*xi**2
        dN3_dxi = 6*xi - 6*xi**2
        dN4_dxi = -2*xi + 3*xi**2
        dN_bending_dxi = np.array([dN1_dxi,
                                    dN2_dxi,
                                    dN3_dxi,
                                    dN4_dxi])
        d2N1_dxi2 = -0 + 12*xi
        d2N2_dxi2 = -2 + 6*xi
        d2N3_dxi2 = 0 - 12*xi
        d2N4_dxi2 = +2 + 6*xi
        d2N_bending_dxi2 = np.array([d2N1_dxi2,
                                    d2N2_dxi2,
                                    d2N3_dxi2,
                                    d2N4_dxi2])
        return N_bending, dN_bending_dxi, d2N_bending_dxi2

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

    def get_shape_function_derivatives(self, xi: float, J: float):
        """
        Compute derivatives of shape functions with respect to physical coordinate x.
        Args:
            xi (float): Local coordinate in [-1, 1]
            J (float): Jacobian = L / 2 (L = element length)
        Returns:
            dN_dx (np.ndarray): Derivatives of linear shape functions w.r.t x (2,)
            d2N_dx2 (np.ndarray): Second derivatives of Hermite shape functions for bending (4,)
        """
        dN_dxi = np.array([-0.5, 0.5])
        dN_dx = dN_dxi / J
        d2H_dxi2 = np.array([
            3*xi / J**2 /2,
            (3*xi - 1) / J/2,
        -3*xi / J**2/2,
            (3*xi + 1) / J/2
        ])
        d2N_dx2 = d2H_dxi2 
        return dN_dx, d2N_dx2

    def B_matrix(self, xi: float, J: float, lmn_x: np.ndarray, lmn_y: np.ndarray, lmn_z: np.ndarray) -> np.ndarray:
        """
        Constructs the 6x12 strain-displacement matrix B for a 3D frame element.
        Args:
            - xi (float): Local coordinate in [-1, 1]
            - J (float): Jacobian (element length / 2)
            - lmn_x (np.ndarray): Direction cosines for local x-axis (3,)
            - lmn_y (np.ndarray): Direction cosines for local y-axis (3,)
            - lmn_z (np.ndarray): Direction cosines for local z-axis (3,)
        Returns:
            np.ndarray: 6x12 B-matrix [ε, κy, κz, γx, γy, γz]
        """
        dN_dx, d2N_dx2 = self.get_shape_function_derivatives(xi, J)
        B = np.zeros((6, 12))

        # -----------------------
        # 1. Axial strain: ε = du/dx
        # -----------------------

        B[0, 0:3] = dN_dx[0] * lmn_x
        B[0, 6:9] =  dN_dx[1] * lmn_x 

        # -----------------------
        # 2. Torsion: γx = dθx/dx
        # -----------------------

        B[1, 3:6] = dN_dx[0] * lmn_x
        B[1, 9:12] = dN_dx[1] * lmn_x

        # -----------------------
        # 3. Bending about local z (v, θy): κz = d²v/dx²
        # -----------------------

        B[2, 0:3] = d2N_dx2[0] * lmn_y
        B[2, 3:6] = d2N_dx2[1] * lmn_z
        B[2, 6:9] = d2N_dx2[2] * lmn_y 
        B[2, 9:12] = d2N_dx2[3] * lmn_z

        # -----------------------
        # 4. Bending about local y (w, θz): κy = -d²w/dx²
        # -----------------------

        B[3, 0:3] = d2N_dx2[2] * lmn_z 
        B[3, 3:6] = d2N_dx2[1] * lmn_y 
        B[3, 6:9] = d2N_dx2[0]  * lmn_z 
        B[3, 9:12] = d2N_dx2[3] * lmn_y 
        return B


    # ---------------------------------------------
    # MATERIAL MATRIX (C)
    # ---------------------------------------------

    def material_matrix(self, E: float, G: float, A: float, 
                       J: float, Iy: float, Iz: float) -> np.ndarray:
        """
        Constructs material matrix for 3D frame.
        Args:
            E (float): Young's modulus
            G (float): Shear modulus
            A (float): Cross-sectional area
            J (float): Torsional constant
            Iy (float): Moment of inertia about y-axis
            Iz (float): Moment of inertia about z-axis
        Returns:
            np.ndarray: (6, 6) material matrix
        """
        C = np.diag([E*A, G*J, E*Iy, E*Iz, G*A, G*A])
        return C


    # ---------------------------------------------
    # ELEMENT STIFFNESS MATRIX
    # ---------------------------------------------

    def element_stiffness_matrix(self, element_id: int) -> np.ndarray:
        """
        Computes stiffness matrix for a 3D frame element.
        Args:
            element_id (int): Element identifier
        Returns:
            np.ndarray: (12, 12) element stiffness matrix
        """
        element = self.imported_data['elements'][element_id]
        nodes = self.imported_data['nodes']
        node1 = nodes[element['node1']]
        node2 = nodes[element['node2']]
        coords = np.array([
            [node1['X'], node1['Y'], node1['Z']],
            [node2['X'], node2['Y'], node2['Z']]
        ], dtype=float)
        E = element['E']
        G = element['G']
        A = element['A']
        J = element['J']
        Iy = element['Iy']
        Iz = element['Iz']
        K_e = np.zeros((12, 12))
        gauss_points, gauss_weights = roots_legendre(3)

        for xi, w in zip(gauss_points, gauss_weights):
            N, dN_dxi = self.linear_shape_functions(xi)
            J_det, _, _, lmn_x, lmn_y, lmn_z = self.jacobian(coords, dN_dxi)
            B = self.B_matrix(xi, J_det, lmn_x, lmn_y, lmn_z)
            C = self.material_matrix(E, G, A, J, Iy, Iz)
            K_e += w * J_det * B.T @ C @ B 
        K_e_global = K_e
        return C, B, K_e_global


    # ---------------------------------------------
    # GLOBAL STIFFNESS ASSEMBLY
    # ---------------------------------------------

    def assemble_global_stiffness(self, sparse=False) -> np.ndarray:
        """
        Assembles global stiffness matrix for the structure.
        Returns:
            np.ndarray: Global stiffness matrix
        """
        nodes = self.imported_data['nodes']
        elements = self.imported_data['elements']
        n_nodes = len(nodes)
        ndof = n_nodes * self.dof_per_node

        if sparse:
            K_global = lil_matrix((ndof, ndof))
        else:
            K_global = np.zeros((ndof, ndof))

        for elem_id, elem_data in elements.items():

            if elem_id == 0:
                continue
            C, B, K_e = self.element_stiffness_matrix(elem_id)
            elements[elem_id]['C'] = C
            elements[elem_id]['B'] = B
            elements[elem_id]['k'] = K_e
            node1 = elem_data['node1']
            node2 = elem_data['node2']

            if node1 not in nodes or node2 not in nodes:
                print(f"Warning: Nodes {node1} or {node2} not found for element {elem_id}")
                continue
            dofs = []

            for node in [node1, node2]:
                base_dof = (node-1)*self.dof_per_node
                dofs.extend(range(base_dof, base_dof+self.dof_per_node))

            for i, dof_i in enumerate(dofs):

                for j, dof_j in enumerate(dofs):
                    K_global[dof_i, dof_j] += K_e[i, j]

        if sparse:
            K_global = K_global.tocsr()
        return K_global


    # ---------------------------------------------
    # POST-PROCESSING
    # ---------------------------------------------

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


def cprint(matrix):

    for row in matrix:
        colored_row = ""

        for element in row:

            if element > 0:
                colored_row += f"\033[92m{element:8.3f}\033[0m "
            elif element < 0:
                colored_row += f"\033[91m{element:8.3f}\033[0m "
            else:
                colored_row += f"{element:8.3f}"
        print(colored_row)


    # ---------------------------------------------
    # Example usage
    # ---------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    """Test case for a 3D frame (Ex 5.8, p287)."""
    imported_data = {
        'nodes': {
            1: {'X': 2.5, 'Y': 0.0, 'Z': 0.0},
            2: {'X': 0.0, 'Y': 0.0, 'Z': 0.0},
            3: {'X': 2.5, 'Y': 0.0, 'Z': -2.5},
            4: {'X': 2.5, 'Y': -2.5, 'Z': 0.0},
        },
        'elements': {
            1: {
                'node1': 2, 'node2': 1,
                'E': 200e9, 'G': 60e9,
                'A': 6.25E-3, 'J': 20e-6,
                'Iy': 40e-6, 'Iz': 40e-6
            },
            2: {
                'node1': 3, 'node2': 1,
                'E': 200e9, 'G': 60e9,
                'A': 6.25E-3, 'J': 20e-6,
                'Iy': 40e-6, 'Iz': 40e-6
            },
            3: {
                'node1': 4, 'node2': 1,
                'E': 200e9, 'G': 60e9,
                'A': 6.25E-3, 'J': 20e-6,
                'Iy': 40e-6, 'Iz': 40e-6
            },
        }
    }
    frame = LinearFrameElement(imported_data)
    K_global = frame.assemble_global_stiffness()
    print('\n\n')
    cprint(K_global/1e6)