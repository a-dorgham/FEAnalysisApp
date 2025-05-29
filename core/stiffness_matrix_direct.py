import numpy as np
from typing import Dict, Any, Tuple


class StiffnessMatrixDirect:
    """
    A class to assemble global stiffness matrices for various structural analysis types
    using the direct stiffness method.
    This class provides static methods to compute local stiffness matrices,
    transformation matrices, and to assemble these into a global stiffness matrix

    for different structural elements like springs, trusses, and beams/frames in 2D and 3D.
    """
    imported_data: Dict[str, Any] = {}


    def __init__(self, imported_data: Dict[str, Any]) -> None:
        """
        Initializes the StiffnessMatrixDirect class with imported structural data.
        Args:
            imported_data (Dict[str, Any]): A dictionary containing all necessary
                                             structural data, including node information,
                                             element information, and overall
                                             structure properties.
        """
        super().__init__()
        self.imported_data = imported_data

    # ---------------------------------------------
    # GLOBAL STIFFNESS MATRIX ASSEMBLY
    # ---------------------------------------------

    @staticmethod
    def assemble_stiffness_matrix(imported_data: Dict[str, Any] = None) -> np.ndarray:
        """
        Assembles the global stiffness matrix for the entire structure.
        This method iterates through each element, computes its local stiffness matrix,
        transforms it to the global coordinate system, and then assembles it
        into the global stiffness matrix.
        Args:
            imported_data (Dict[str, Any], optional): A dictionary containing all necessary
                                                       structural data. If provided, it updates
                                                       the class's `imported_data`. Defaults to None.
        Returns:
            np.ndarray: The assembled global stiffness matrix (K_global).
        """

        if imported_data:
            StiffnessMatrixDirect.imported_data = imported_data
        converted_node_data: Dict[int, Dict[str, Any]] = StiffnessMatrixDirect.imported_data["converted_nodes"]
        converted_element_data: Dict[int, Dict[str, Any]] = StiffnessMatrixDirect.imported_data["converted_elements"]
        structure_properties: Dict[str, Any] = StiffnessMatrixDirect.imported_data['structure_info']
        dofs_per_node: int = structure_properties["dofs_per_node"]
        structure_type: str = structure_properties["element_type"]
        num_nodes: int = len(converted_node_data)
        K_global: np.ndarray = np.zeros((dofs_per_node * num_nodes, dofs_per_node * num_nodes))

        for element_id, element in converted_element_data.items():

            if element_id == 0:
                continue
            K_local_element: np.ndarray = StiffnessMatrixDirect.compute_local_stiffness_matrix(
                element, structure_type, dofs_per_node
            )
            T: np.ndarray = StiffnessMatrixDirect.compute_local_transformation_matrix(
                element, structure_type, dofs_per_node
            )
            K_local_transformed: np.ndarray = T.T @ K_local_element @ T
            converted_element_data[element_id]['kT'] = K_local_element @ T
            node1: int = element['node1'] - 1
            node2: int = element['node2'] - 1
            node1_dofs: list[int] = [dofs_per_node * node1 + i for i in range(dofs_per_node)]
            node2_dofs: list[int] = [dofs_per_node * node2 + i for i in range(dofs_per_node)]
            element_dofs: list[int] = node1_dofs + node2_dofs

            for i in range(K_local_transformed.shape[0]):

                for j in range(K_local_transformed.shape[1]):
                    K_global[element_dofs[i], element_dofs[j]] += K_local_transformed[i, j]
        return K_global


    # ---------------------------------------------
    # LOCAL STIFFNESS MATRIX COMPUTATION
    # ---------------------------------------------


    @staticmethod
    def compute_local_stiffness_matrix(
        element: Dict[str, Any], structure_type: str, dofs_per_node: int) -> np.ndarray:
        """
        Computes the local stiffness matrix for a given element based on its type.
        Args:
            element (Dict[str, Any]): A dictionary containing element properties
                                       such as Young's modulus (E), shear modulus (G),
                                       cross-sectional area (A), moments of inertia (Iy, Iz),
                                       torsional constant (J), and length (L).
            structure_type (str): The type of structure (e.g., "Spring", "2D_Truss",
                                  "3D_Truss", "2D_Beam", "3D_Frame").
            dofs_per_node (int): Degrees of freedom per node for the current structure type.
        Returns:
            np.ndarray: The local stiffness matrix (K_local) for the element.
        """
        E: float = element.get('E', 1.0)
        G: float = element.get('G', 1.0)
        A: float = element.get('A', 1.0)
        Iy: float = element.get('Iy', 1.0)
        Iz: float = element.get('Iz', 1.0)
        J: float = element.get('J', 1.0)
        L: float = element.get('length', 1.0)
        a: float = E * A / L
        by: float = 12 * E * Iz / L**3
        cy: float = 6 * E * Iz / L**2
        dy: float = 4 * E * Iz / L
        ey: float = 2 * E * Iz / L
        bz: float = 12 * E * Iy / L**3
        cz: float = 6 * E * Iy / L**2
        dz: float = 4 * E * Iy / L
        ez: float = 2 * E * Iy / L
        t: float = G * J / L
        ndofs: int = dofs_per_node * 2
        K_local: np.ndarray = np.zeros((ndofs, ndofs))

        if structure_type == "Spring":
            K_local = np.array([[a, -a], 
                                [-a, a]])
        elif structure_type == "2D_Truss":
            K_local = np.array([[a, 0, -a, 0],
                                [0, 0, 0, 0],
                                [-a, 0, a, 0],
                                [0, 0, 0, 0]])
        elif structure_type == "3D_Truss":
            K_local = np.array([
                [a, 0, 0, -a, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-a, 0, 0, a, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])
        elif structure_type == "2D_Beam":
            K_local = np.array([
                [ a,  0,  0, -a,  0,  0],
                [ 0, by, cy,  0, -by, cy],
                [ 0, cy, dy,  0, -cy, ey],
                [-a,  0,  0,  a,  0,  0],
                [ 0, -by, -cy,  0, by, -cy],
                [ 0, cy, ey,  0, -cy, dy]
            ])
        elif structure_type == "3D_Frame":
            K_local = np.array([
                [ a,  0,   0,   0,   0,   0,  -a,   0,   0,   0,   0,   0],
                [ 0, by,   0,   0,   0,  cy,   0, -by,   0,   0,   0,  cy],
                [ 0,  0, bz,   0, -cz,   0,   0,   0, -bz,   0, -cz,   0],
                [ 0,  0,   0,  t,   0,   0,   0,   0,   0, -t,   0,   0],
                [ 0,  0, -cz,  0, dz,   0,   0,   0,  cz,  0, ez,   0],
                [ 0, cy,   0,   0,   0, dy,   0, -cy,   0,   0,   0, ey],
                [-a,  0,   0,   0,   0,   0,  a,   0,   0,   0,   0,   0],
                [ 0, -by,  0,   0,   0, -cy,  0,  by,   0,   0,   0, -cy],
                [ 0,  0, -bz,  0,  cz,   0,   0,   0,  bz,   0,  cz,   0],
                [ 0,  0,   0, -t,   0,   0,   0,   0,   0,  t,   0,   0],
                [ 0,  0, -cz,  0, ez,   0,   0,   0,  cz,  0, dz,   0],
                [ 0, cy,   0,   0,   0, ey,   0, -cy,   0,   0,   0, dy]
            ])
        return K_local


    # ---------------------------------------------
    # TRANSFORMATION MATRIX COMPUTATION
    # ---------------------------------------------

    @staticmethod
    def compute_local_transformation_matrix(
        element: Dict[str, Any], structure_type: str, dofs_per_node: int) -> np.ndarray:
        """
        Computes the transformation matrix (T) for a given element to convert
        between local and global coordinate systems.
        Args:
            element (Dict[str, Any]): A dictionary containing element properties,
                                       specifically directional cosines 'lmn', 'lmn_y', and 'lmn_z'.
            structure_type (str): The type of structure (e.g., "2D_Truss", "3D_Truss",
                                  "2D_Beam", "3D_Frame").
            dofs_per_node (int): Degrees of freedom per node for the current structure type.
        Returns:
            np.ndarray: The transformation matrix (T).
        """
        l: float = element['lmn'][0]
        m: float = element['lmn'][1]
        n: float = element['lmn'][2]
        ndofs: int = dofs_per_node * 2
        T: np.ndarray = np.zeros((ndofs, ndofs)) 

        if structure_type == "2D_Truss":
            T_base: np.ndarray = np.array([[l, m], [-m, l]])
            T[:2, :2] = T_base
            T[2:, 2:] = T_base
        elif structure_type == "3D_Truss":
            lx = np.array([l, m, n])

            if np.isclose(np.linalg.norm(np.cross(lx, np.array([0, 0, 1]))), 0):
                ref_vec = np.array([0, 1, 0])
            else:
                ref_vec = np.array([0, 0, 1])
            lz = np.cross(lx, ref_vec)
            lz = lz / np.linalg.norm(lz)
            ly = np.cross(lz, lx)
            ly = ly / np.linalg.norm(ly)
            T_base: np.ndarray = np.array([
                lx,
                ly,
                lz
            ])
            T[:3, :3] = T_base
            T[3:, 3:] = T_base
        elif structure_type == "2D_Beam":
            T_base: np.ndarray = np.array([[l, m, 0], [-m, l, 0], [0, 0, 1]])
            T[:3, :3] = T_base
            T[3:, 3:] = T_base
        elif structure_type == "3D_Frame":
            l_y: float = element['lmn_y'][0]
            m_y: float = element['lmn_y'][1]
            n_y: float = element['lmn_y'][2]
            l_z: float = element['lmn_z'][0]
            m_z: float = element['lmn_z'][1]
            n_z: float = element['lmn_z'][2]
            T_base: np.ndarray = np.array([
                [l,    m,    n],
                [l_y,  m_y,  n_y],
                [l_z,  m_z,  n_z]
            ])
            T[0:3, 0:3] = T_base
            T[3:6, 3:6] = T_base
            T[6:9, 6:9] = T_base
            T[9:12, 9:12] = T_base
        return T