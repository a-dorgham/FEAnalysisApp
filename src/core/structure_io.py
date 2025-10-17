import math
import numpy as np
from typing import Dict, Any, Tuple, Union, List, Union, Callable

class ProcessImportedStructure:
    """
    A class to process and enrich imported structural data for analysis.
    This class handles the calculation of element lengths, angles, cross-section
    properties, and the conversion of distributed loads to equivalent nodal forces.
    It also determines the constraint types for nodes based on their displacement
    boundary conditions.
    - cross_section_data (Dict[str, Any]): Stores processed cross-section properties.
                                            Initialized as an empty dictionary.
    - boundary_conditions (Dict[str, Any]): Stores processed boundary condition data.
                                            Initialized as an empty dictionary.
    - nodes_with_distributed_load (Dict[int, Tuple[float, ...]]): A dictionary to store the
                                        equivalent nodal forces resulting from distributed loads. Initialized as an empty dictionary.
    - structure_type (str): The type of the structure (e.g., '2D_Beam', '3D_Frame').
                            Initialized as an empty string.
    - imported_data (Dict[str, Any]): The raw imported structural data.
                                        Initialized as an empty dictionary.
    """
    cross_section_data: Dict[str, Any] = {}
    boundary_conditions: Dict[str, Any] = {}
    nodes_with_distributed_load: Dict[int, Tuple[float, ...]] = {}
    structure_type: str = ""
    imported_data: Dict[str, Any] = {}


    def __init__(self, structure_type: Union[str, None] = None, imported_data: Dict[str, Any] = {}) -> None:
        """
        Initializes the ProcessImportedStructure with structural data.
        Args:
            structure_type (Union[str, None]): The type of the structure (e.g., '2D_Beam', '3D_Frame').
                                                If None, it's inferred from `imported_data`.
            imported_data (Dict[str, Any]): A dictionary containing all imported structural data,
                                            including nodes, elements, cross_sections, and materials.
        """
        super().__init__()
        ProcessImportedStructure.imported_data = imported_data

        if structure_type is None:
            structure_type = imported_data['structure_info']['element_type']
        ProcessImportedStructure.structure_type = structure_type

        if imported_data:

            # ---------------------------------------------
            # GEOMETRY & MATERIAL PROPERTIES CALCULATION
            # ---------------------------------------------

            ProcessImportedStructure.compute_section_properties(
                element_data=imported_data["elements"],
                cross_section_data=imported_data["cross_sections"],
                materials_data=imported_data["materials"]
            )
            ProcessImportedStructure.calculate_element_lengths(
                node_data=imported_data["nodes"],
                element_data=imported_data["elements"]
            )
            ProcessImportedStructure.calculate_element_angles(
                node_data=imported_data["nodes"],
                element_data=imported_data["elements"]
            )

            # ---------------------------------------------
            # BOUNDARY CONDITIONS & FORCES PROCESSING
            # ---------------------------------------------

            ProcessImportedStructure.get_constraint_type(
                node_data=imported_data["nodes"],
                structure_info=imported_data['structure_info']
            )
            ProcessImportedStructure.process_forces_based_on_displacements(
                node_data=imported_data["nodes"],
                nodal_displacements=imported_data.get("nodal_displacements", {})
            )

            # ---------------------------------------------
            # 3D FRAME SPECIFIC CALCULATIONS
            # ---------------------------------------------

            if structure_type == '3D_Frame':
                ProcessImportedStructure.calculate_element_angles3D(
                    node_data=imported_data["nodes"],
                    element_data=imported_data["elements"]
                )

    # ---------------------------------------------
    # GEOMETRY CALCULATIONS
    # ---------------------------------------------

    @staticmethod
    def calculate_element_lengths(node_data: Dict[int, Dict[str, float]], element_data: Dict[int, Dict[str, Any]]) -> None:
        """
        Calculates and updates the length of each element in the `element_data` dictionary.
        The calculation considers whether the structure is 2D or 3D based on node coordinates.
        Args:
            node_data (Dict[int, Dict[str, float]]): A dictionary where keys are node IDs
                                                       and values are dictionaries containing
                                                       'X', 'Y', and optionally 'Z' coordinates.
            element_data (Dict[int, Dict[str, Any]]): A dictionary where keys are element IDs
                                                        and values are dictionaries containing
                                                        'node1' and 'node2' (the connecting node IDs).
                                                        The 'length' key will be added/updated.
        Raises:
            ValueError: If a node specified in `element_data` is missing from `node_data`.
        """

        for elem_num, elem in element_data.items():
            node1: int = elem['node1']
            node2: int = elem['node2']

            if node1 not in node_data or node2 not in node_data:

                if node1 == 0 and node2 == 0:
                    element_data[elem_num]['length'] = 0.0
                    continue                
                else:
                    raise ValueError(f"Node {node1} or {node2} is missing from node_data for element {elem_num}")
            coord1: Dict[str, float] = node_data[node1]
            coord2: Dict[str, float] = node_data[node2]
            is_3D: bool = 'Z' in coord1 and 'Z' in coord2

            if is_3D:
                length: float = np.sqrt(
                    (coord2['X'] - coord1['X'])**2 +
                    (coord2['Y'] - coord1['Y'])**2 +
                    (coord2['Z'] - coord1['Z'])**2
                )
            else:
                length: float = np.sqrt(
                    (coord2['X'] - coord1['X'])**2 +
                    (coord2['Y'] - coord1['Y'])**2
                )
            element_data[elem_num]['length'] = length

    @staticmethod
    def calculate_element_angles(node_data: Dict[int, Dict[str, float]], element_data: Dict[int, Dict[str, Any]]) -> None:
        """
        Calculates and updates element angles (for 2D) or direction cosines (for 3D)

        for each element in the `element_data` dictionary.
        Args:
            node_data (Dict[int, Dict[str, float]]): A dictionary where keys are node IDs
                                                       and values are dictionaries containing
                                                       'X', 'Y', and optionally 'Z' coordinates.
            element_data (Dict[int, Dict[str, Any]]): A dictionary where keys are element IDs
                                                        and values are dictionaries containing
                                                        'node1', 'node2', and 'length'.
                                                        'angle' and 'lmn' keys will be added/updated.
        """

        for element_num, element in element_data.items():

            if element_num == 0:
                continue
            node1: int = element['node1']
            node2: int = element['node2']
            L: float = element['length']

            if node1 not in node_data or node2 not in node_data:
                print(f"Warning: Nodes {node1} or {node2} not found for element {element_num}")
                continue
            x1: float = node_data[node1]['X']
            y1: float = node_data[node1]['Y']
            x2: float = node_data[node2]['X']
            y2: float = node_data[node2]['Y']

            if 'Z' in node_data[node1] and 'Z' in node_data[node2]:
                z1: float = node_data[node1]['Z']
                z2: float = node_data[node2]['Z']
                theta_xy: float = math.atan2(y2 - y1, x2 - x1)  
                theta_z: float = math.atan2(z2 - z1, math.sqrt((x2 - x1)**2 + (y2 - y1)**2))  
                angle: Tuple[float, float] = (theta_xy, theta_z)
                l: float = (x2 - x1) / L
                m: float = (y2 - y1) / L
                n: float = (z2 - z1) / L  
            else:
                angle = math.atan2(y2 - y1, x2 - x1)
                l, m, n = (x2 - x1) / L, (y2 - y1) / L, 0.0  
            element_data[element_num]['angle'] = angle
            element_data[element_num]['lmn'] = (l, m, n) 

    @staticmethod
    def calculate_element_angles3D(node_data: Dict[int, Dict[str, float]], element_data: Dict[int, Dict[str, Any]]) -> None:
        """
        Updates the imported structural data for 3D elements with the length,
        directional cosines for the local x-axis (lmn), and the direction
        cosines for the local y-axis (lmn_y) and z-axis (lmn_z).
        This is specifically for 3D Frame analysis where a full local coordinate
        system is needed.
        Args:
            node_data (Dict[int, Dict[str, float]]): Dictionary containing node coordinates.
            element_data (Dict[int, Dict[str, Any]]): Dictionary containing element connectivity
                                                      and to be updated with directional cosines.
        """
        ref_vector: np.ndarray = np.array([0, 1, 0])  

        for element_id, element_data_i in element_data.items():

            if element_id == 0:
                continue
            node1_id: int = element_data_i.get('node1', 0)
            node2_id: int = element_data_i.get('node2', 0)

            if node1_id in node_data and node2_id in node_data:
                p1: np.ndarray = np.array([node_data[node1_id]['X'], node_data[node1_id]['Y'], node_data[node1_id]['Z']])
                p2: np.ndarray = np.array([node_data[node2_id]['X'], node_data[node2_id]['Y'], node_data[node2_id]['Z']])
                vec_x: np.ndarray = p2 - p1
                length: float = np.linalg.norm(vec_x)

                if length > 1e-9:
                    l, m, n = vec_x / length

                    if np.allclose(vec_x / length, ref_vector) or np.allclose(vec_x / length, -ref_vector):
                        temp_ref_vector: np.ndarray = np.array([0, 0, 1])
                    else:
                        temp_ref_vector = ref_vector
                    vec_z_unnormalized: np.ndarray = np.cross(vec_x, temp_ref_vector)
                    vec_z: np.ndarray = vec_z_unnormalized / np.linalg.norm(vec_z_unnormalized)
                    vec_y_unnormalized: np.ndarray = np.cross(vec_z, vec_x)
                    vec_y: np.ndarray = vec_y_unnormalized / np.linalg.norm(vec_y_unnormalized)
                    l2, m2, n2 = vec_y
                    l3, m3, n3 = vec_z
                    element_data_i.update({
                        'length': length,
                        'lmn': (l, m, n),
                        'lmn_y': (l2, m2, n2),
                        'lmn_z': (l3, m3, n3)
                    })
                else:
                    print(f"Warning: Zero-length element {element_id} detected. Setting all direction cosines to zero.")
                    element_data_i.update({
                        'length': 0.0,
                        'lmn': (0.0, 0.0, 0.0),
                        'lmn_y': (0.0, 0.0, 0.0),
                        'lmn_z': (0.0, 0.0, 0.0)
                    })
            else:
                print(f"Warning: Missing node(s) for element {element_id}. Cannot calculate angles/direction cosines.")

    # ---------------------------------------------
    # CROSS-SECTION PROPERTY CALCULATIONS
    # ---------------------------------------------

    @staticmethod
    def compute_section_properties(element_data: Dict[int, Dict[str, Any]], 
                                   cross_section_data: Dict[str, Any], 
                                   materials_data: Dict[str, Any]) -> None:
        """
        Computes and updates the shear modulus (G), torsional constant (J),
        moments of inertia (Iy, Iz), and cross-sectional area (A) for each element
        based on its material and cross-section properties.
        Args:
            element_data (Dict[int, Dict[str, Any]]): Dictionary containing element data,
                                                      including 'material_code' and 'section_code'.
                                                      This dictionary will be updated with computed properties.
            cross_section_data (Dict[str, Any]): Dictionary containing cross-section definitions,
                                                  including 'type' and 'dimensions'.
            materials_data (Dict[str, Any]): Dictionary containing material properties,
                                              including 'E' (Elastic Modulus) and 'v' (Poisson's Ratio).
        """

        for elem_id, elem in element_data.items():
            material_code: str = elem.get('material_code', '')

            if material_code and material_code in materials_data:
                E: Union[float, None] = materials_data[material_code]['properties'].get('E')
                v: Union[float, None] = materials_data[material_code]['properties'].get('v')
                section_code: str = elem.get('section_code', '')

                if section_code and section_code in cross_section_data:
                    section_type: str = cross_section_data[section_code]['type']
                    section_dimensions: Dict[str, float] = cross_section_data[section_code]['dimensions']
                    properties: Dict[str, Union[float, None]] = \
                        ProcessImportedStructure.get_section_properties(
                            section_type=section_type,
                            dimensions=section_dimensions,
                            elastic_modulus=E,
                            poisson_ratio=v
                        )
                    element_data[elem_id].update({
                        'E': E, 
                        'v': v,
                        'G': properties.get("G"), 
                        'J': properties.get("J"), 
                        'Iy': properties.get("Iy"),
                        'Iz': properties.get("Iz"), 
                        'A': properties.get("A")
                    })
                else:
                    print(f"Warning: Section code '{section_code}' for element {elem_id} not found in cross_section_data.")
            else:
                print(f"Warning: Material code '{material_code}' for element {elem_id} not found in materials_data.")

    @staticmethod
    def get_section_properties(section_type: str, dimensions: Dict[str, float], 
                               elastic_modulus: Union[float, None], poisson_ratio: Union[float, None]) -> Dict[str, Union[float, None]]:
        """
        Computes the cross-sectional properties (Area, Moments of Inertia, Torsional Constant, Shear Modulus)

        for various standard section types.
        Args:
            section_type (str): The type of the cross-section (e.g., "Solid_Circular", "Hollow_Rectangular").
            dimensions (Dict[str, float]): A dictionary containing the dimensions required for the section type.
                                            Keys vary based on `section_type` (e.g., 'D', 'B', 'H', 'tw', 'tf', 'angle').
            elastic_modulus (Union[float, None]): The Young's Modulus (E) of the material.
            poisson_ratio (Union[float, None]): The Poisson's Ratio (v) of the material.
        Returns:
            Dict[str, Union[float, None]]: A dictionary containing the computed properties:
                                            "A" (Area), "J" (Torsional Constant),
                                            "Iy" (Moment of Inertia about local y-axis),
                                            "Iz" (Moment of Inertia about local z-axis),
                                            and "G" (Shear Modulus).
        Raises:
            ValueError: If a required dimension is missing for a specific section type.
        """
        properties: Dict[str, Union[float, None]] = {}

        # ---------------------------------------------
        # SHEAR MODULUS CALCULATION
        # ---------------------------------------------

        if elastic_modulus is not None and poisson_ratio is not None:
            properties["G"] = elastic_modulus / (2 * (1 + poisson_ratio))
        else:
            properties["G"] = None

        # ---------------------------------------------
        # HELPER FUNCTIONS FOR DIMENSIONS AND ROTATION
        # ---------------------------------------------

        def get_dim(key: str) -> float:
            """Helper function to safely retrieve dimension with error handling."""

            try:
                return dimensions[key]

            except KeyError:

                if key == 'angle':
                    return 0.0

                else:
                    raise ValueError(f"Missing required dimension '{key}' for {section_type} section.")


        def rotate_inertia(Iy: float, Iz: float, angle_deg: float) -> Tuple[float, float]:
            """Rotates moments of inertia given an angle in degrees."""
            theta: float = math.radians(angle_deg)
            cos2: float = math.cos(theta) ** 2
            sin2: float = math.sin(theta) ** 2
            Iy_rot: float = Iy * cos2 + Iz * sin2
            Iz_rot: float = Iy * sin2 + Iz * cos2
            return Iy_rot, Iz_rot


        # ---------------------------------------------
        # SECTION PROPERTY CALCULATIONS
        # ---------------------------------------------

        if section_type == "Solid_Circular":
            d: float = get_dim('D')
            properties["A"] = math.pi * (d**2) / 4
            properties["J"] = math.pi * (d**4) / 32
            properties["Iy"] = properties["Iz"] = math.pi * (d**4) / 64
        elif section_type == "Hollow_Circular":
            D: float = get_dim('D')
            d_inner: float = get_dim('d')
            properties["A"] = math.pi * (D**2 - d_inner**2) / 4
            properties["J"] = math.pi * (D**4 - d_inner**4) / 32
            properties["Iy"] = properties["Iz"] = math.pi * (D**4 - d_inner**4) / 64
        elif section_type == "Solid_Rectangular":
            b: float = get_dim('B')
            h: float = get_dim('H')
            angle: float = get_dim('angle')
            A: float = b * h
            Iz_unrotated: float = (b * h**3) / 12
            Iy_unrotated: float = (h * b**3) / 12
            Iy_rotated, Iz_rotated = rotate_inertia(Iy_unrotated, Iz_unrotated, angle)
            J: float = Iy_rotated + Iz_rotated
            properties.update({"A": A, "Iy": Iy_rotated, "Iz": Iz_rotated, "J": J})
        elif section_type == "Hollow_Rectangular":
            B: float = get_dim('B')
            H: float = get_dim('H')
            b_inner: float = get_dim('b')
            h_inner: float = get_dim('h')
            angle: float = get_dim('angle')
            A: float = (B * H) - (b_inner * h_inner)
            Iz_unrotated: float = (B * H**3 - b_inner * h_inner**3) / 12
            Iy_unrotated: float = (H * B**3 - h_inner * b_inner**3) / 12
            Iy_rotated, Iz_rotated = rotate_inertia(Iy_unrotated, Iz_unrotated, angle)
            J: float = Iy_rotated + Iz_rotated
            properties.update({"A": A, "Iy": Iy_rotated, "Iz": Iz_rotated, "J": J})
        elif section_type == "I_Beam":
            B: float = get_dim('B')
            H: float = get_dim('H')
            tw: float = get_dim('tw')
            tf: float = get_dim('tf')
            angle: float = get_dim('angle')
            h_web: float = H - 2 * tf
            b_web_effective: float = B - tw
            A: float = B * H - b_web_effective * h_web
            Iz_unrotated: float = (B * H**3 - b_web_effective * h_web**3) / 12
            Iy_unrotated: float = (H * B**3 - h_web * b_web_effective**3) / 12 
            Iy_rotated, Iz_rotated = rotate_inertia(Iy_unrotated, Iz_unrotated, angle)
            J: float = Iy_rotated + Iz_rotated
            properties.update({"A": A, "Iy": Iy_rotated, "Iz": Iz_rotated, "J": J})
        elif section_type == "C_Beam":
            B: float = get_dim('B')
            H: float = get_dim('H')
            tw: float = get_dim('tw')
            tf: float = get_dim('tf')
            angle: float = get_dim('angle')
            A: float = (B * tf * 2) + (H - 2 * tf) * tw
            x_c: float = ((tw * H * (tw / 2)) + (2 * B * tf * (tw + B / 2))) / A
            Ixx_flange: float = (B * tf**3) / 12 + B * tf * (H/2 - tf/2)**2
            Iyy_flange: float = (tf * B**3) / 12 + tf * B * (B/2 - x_c)**2
            Iz_unrotated: float = (B * H**3 - (B - tw) * (H - 2 * tf)**3) / 12
            Iy_unrotated: float = (2 * tf * B**3 + (H - 2 * tf) * tw**3) / 12
            Iy_rotated, Iz_rotated = rotate_inertia(Iy_unrotated, Iz_unrotated, angle)
            J: float = Iy_rotated + Iz_rotated
            properties.update({"A": A, "Iy": Iy_rotated, "Iz": Iz_rotated, "J": J})
        elif section_type == "L_Beam":
            B: float = get_dim('B')
            H: float = get_dim('H')
            tw: float = get_dim('tw')
            tf: float = get_dim('tf')
            angle: float = get_dim('angle')
            A: float = (B * tf) + (H * tw) - (tf * tw)
            x_c: float = ((B * tf) * (B / 2)) + ((H - tf) * tw * (tw / 2)) / A
            y_c: float = ((B * tf) * (tf / 2)) + ((H - tf) * tw * (tf + (H - tf) / 2)) / A
            Ixx_B: float = (B * tf**3) / 12 + (B * tf) * (y_c - tf / 2)**2
            Iyy_B: float = (tf * B**3) / 12 + (B * tf) * (x_c - B / 2)**2
            Ixx_H: float = (tw * (H - tf)**3) / 12 + (tw * (H - tf)) * (y_c - (tf + (H - tf) / 2))**2
            Iyy_H: float = ((H - tf) * tw**3) / 12 + (tw * (H - tf)) * (x_c - tw / 2)**2
            Iz_unrotated: float = Ixx_B + Ixx_H
            Iy_unrotated: float = Iyy_B + Iyy_H
            Iy_rotated, Iz_rotated = rotate_inertia(Iy_unrotated, Iz_unrotated, angle)
            J: float = (B * tf**3 + H * tw**3) / 3  
            properties.update({"A": A, "Iy": Iy_rotated, "Iz": Iz_rotated, "J": J})
        elif section_type == "Custom":
            properties["A"] = get_dim('A')
            properties["Iy"] = get_dim('Iy')
            properties["Iz"] = get_dim('Iz')
            properties["J"] = get_dim('J')
        else:
            raise ValueError(f"Unsupported section type: {section_type}")
        return properties


    # ---------------------------------------------
    # DISTRIBUTED LOAD CONVERSION
    # ---------------------------------------------

    @staticmethod
    def convert_distributed_loads_to_nodal(node_data: Dict[int, Dict[str, Any]], 
                                           element_data: Dict[int, Dict[str, Any]], 
                                           distributed_loads: Dict[int, Dict[str, Any]]) -> None:
        """
        Converts distributed loads acting on elements into equivalent nodal forces and moments.
        These equivalent forces are stored in `ProcessImportedStructure.nodes_with_distributed_load`.
        Args:
            node_data (Dict[int, Dict[str, Any]]): Dictionary containing node data, specifically
                                                   used to determine the size of the force tuples.
            element_data (Dict[int, Dict[str, Any]]): Dictionary containing element data, including
                                                      'node1', 'node2', and 'length'.
            distributed_loads (Dict[int, Dict[str, Any]]): Dictionary containing distributed load data.
                                                            Keys are element numbers, values are dictionaries
                                                            with 'type' ('uniform'), 'parameters' (magnitude),
                                                            and 'direction' (e.g., 'Global_Y', '-Global_X').
        """
        nodes_with_distributed_load = set()
        ProcessImportedStructure.nodes_with_distributed_load = {}

        for element_num, load in distributed_loads.items():
            load_type: str = load["type"]
            params: float = load["parameters"]
            direction: str = load["direction"]
            node1: int = element_data[element_num]["node1"]
            node2: int = element_data[element_num]["node2"]
            L: float = element_data[element_num]['length']
            len_force: int = len(node_data[node1]['force']) if node_data[node1].get('force') else 6
            F1_combined: np.ndarray = np.zeros(len_force)
            F2_combined: np.ndarray = np.zeros(len_force)
            sign: float = 1.0 if direction.startswith('-') else -1.0

            if load_type.lower() == 'uniform':
                q: float = params
                P: float = abs(q) * L

                # ---------------------------------------------
                # 2D BEAM / 3D FRAME LOADING
                # ---------------------------------------------

                if ProcessImportedStructure.structure_type in ['2D_Beam', '3D_Frame']:

                    if 'Global_Y' in direction:
                        F1_combined[1] = sign * (-P / 2)
                        F1_combined[2] = sign * (-P * L / 12)
                        F2_combined[1] = sign * (-P / 2)
                        F2_combined[2] = sign * (P * L / 12)
                    elif 'Global_Z' in direction and ProcessImportedStructure.structure_type == '3D_Frame':
                        F1_combined[2] = sign * (-P / 2)
                        F1_combined[4] = sign * (-P * L / 12)
                        F2_combined[2] = sign * (-P / 2)
                        F2_combined[4] = sign * (P * L / 12)
                    elif 'Global_X' in direction:
                        F1_combined[0] = sign * (P / 2)
                        F2_combined[0] = sign * (P / 2)

                # ---------------------------------------------
                # 2D TRUSS / 3D TRUSS LOADING
                # ---------------------------------------------

                elif ProcessImportedStructure.structure_type in ['2D_Truss', '3D_Truss']:

                    if 'Global_X' in direction:
                        F1_combined[0] = sign * (P / 2)
                        F2_combined[0] = sign * (P / 2)
                    elif 'Global_Y' in direction:
                        F1_combined[1] = sign * (P / 2)
                        F2_combined[1] = sign * (P / 2)
                    elif ProcessImportedStructure.structure_type == '3D_Truss' and 'Global_Z' in direction:
                        F1_combined[2] = sign * (P / 2)
                        F2_combined[2] = sign * (P / 2)
            nodes_with_distributed_load.add(node1)
            nodes_with_distributed_load.add(node2)
            ProcessImportedStructure.nodes_with_distributed_load[node1] = tuple(
                ProcessImportedStructure.nodes_with_distributed_load.get(node1, (0,) * len(F1_combined)) + F1_combined
            )
            ProcessImportedStructure.nodes_with_distributed_load[node2] = tuple(
                ProcessImportedStructure.nodes_with_distributed_load.get(node2, (0,) * len(F2_combined)) + F2_combined
            )

            for element_num, element in element_data.items():
                node1 = element["node1"]
                node2 = element["node2"]

                if node1 not in ProcessImportedStructure.nodes_with_distributed_load:
                    ProcessImportedStructure.nodes_with_distributed_load[node1] = tuple(np.zeros(len_force))

                if node2 not in ProcessImportedStructure.nodes_with_distributed_load:
                    ProcessImportedStructure.nodes_with_distributed_load[node2] = tuple(np.zeros(len_force))
        ProcessImportedStructure.nodes_with_distributed_load = dict(sorted(ProcessImportedStructure.nodes_with_distributed_load.items()))

    @staticmethod
    def add_distributed_load(node_data: Dict[int, Dict[str, Any]]) -> None:
        """
        Adds the pre-calculated equivalent nodal forces from distributed loads
        (stored in `nodes_with_distributed_load`) to the existing forces
        of the respective nodes in `node_data`. This function should be called
        after `convert_distributed_loads_to_nodal`.
        Args:
            node_data (Dict[int, Dict[str, Any]]): The dictionary containing node data.
                                                   The 'force' tuple for each node will be updated.
        """

        for node_num, node_info in node_data.items():
            F_combined_dist: Tuple[float, ...] = ProcessImportedStructure.nodes_with_distributed_load.get(node_num, ())
            force_values: Tuple[float, ...] = node_info.get("force", ())
            force_array: np.ndarray = np.array(force_values, dtype=float) 
            F_combined_dist_array: np.ndarray = np.array(F_combined_dist, dtype=float)

            if force_array.size == 0 and F_combined_dist_array.size > 0:
                force_array = np.zeros_like(F_combined_dist_array)
            elif F_combined_dist_array.size == 0 and force_array.size > 0:
                F_combined_dist_array = np.zeros_like(force_array)
            elif force_array.size != F_combined_dist_array.size and force_array.size > 0 and F_combined_dist_array.size > 0:
                raise ValueError(f"Force array size mismatch for node {node_num}: "
                                 f"Existing force size {force_array.size}, Distributed load size {F_combined_dist_array.size}")
            force_without_nan: np.ndarray = np.nan_to_num(force_array, nan=0.0)
            updated_force: np.ndarray = np.where(
                F_combined_dist_array != 0, 
                force_without_nan + F_combined_dist_array, 
                force_array
            )
            node_data[node_num]["force"] = tuple(updated_force)

    @staticmethod
    def subtract_distributed_load(node_data: Dict[int, Dict[str, Any]]) -> None:
        """
        Subtracts the pre-calculated equivalent nodal forces from distributed loads
        (stored in `nodes_with_distributed_load`) from the existing forces
        of the respective nodes in `node_data`. This function is useful for undoing
        the effect of `add_distributed_load`.
        Args:
            node_data (Dict[int, Dict[str, Any]]): The dictionary containing node data.
                                                   The 'force' tuple for each node will be updated.
        """

        for node_num, node_info in node_data.items():
            F_combined_dist: Tuple[float, ...] = ProcessImportedStructure.nodes_with_distributed_load.get(node_num, ())
            force_values: Tuple[float, ...] = node_info.get("force", ())
            force_array: np.ndarray = np.array(force_values, dtype=float)  
            F_combined_dist_array: np.ndarray = np.array(F_combined_dist, dtype=float)

            if force_array.size == 0 and F_combined_dist_array.size > 0:
                force_array = np.zeros_like(F_combined_dist_array)
            elif F_combined_dist_array.size == 0 and force_array.size > 0:
                F_combined_dist_array = np.zeros_like(force_array)
            elif force_array.size != F_combined_dist_array.size and force_array.size > 0 and F_combined_dist_array.size > 0:
                raise ValueError(f"Force array size mismatch for node {node_num}: "
                                 f"Existing force size {force_array.size}, Distributed load size {F_combined_dist_array.size}")
            force_without_nan: np.ndarray = np.nan_to_num(force_array, nan=0.0)
            updated_force: np.ndarray = np.where(
                F_combined_dist_array != 0, 
                force_without_nan - F_combined_dist_array, 
                force_array
            )
            node_data[node_num]["force"] = tuple(updated_force)

    # ---------------------------------------------
    # CONSTRAINT & DISPLACEMENT PROCESSING
    # ---------------------------------------------

    @staticmethod
    def get_constraint_type(node_data: Dict[int, Dict[str, Any]], structure_info: Dict[str, Any]) -> None:
        """
        Determines the type of constraint for each node based on its displacement boundary conditions
        and the overall structure type (e.g., 2D Beam, 3D Truss). The determined constraint type
        is added to each node's data.
        Args:
            node_data (Dict[int, Dict[str, Any]]): A dictionary where keys are node IDs and values
                                                   are dictionaries containing 'displacement' tuples.
                                                   The 'constraint_type' key will be added/updated.
            structure_info (Dict[str, Any]): A dictionary containing information about the structure,
                                             including 'element_type' and 'dimension' ('2D' or '3D').
        """

        for node_id, node_data_i in node_data.items():
            displacement: Tuple[float, ...] = node_data_i.get("displacement", (np.nan,) * 6)
            element_type: str = structure_info.get("element_type", "General")
            is_3d: bool = structure_info.get("dimension") == "3D"
            constraint_type: Union[str, None] = None
            disp_array: np.ndarray = np.array(displacement)

            if np.all(np.isnan(disp_array)):
                constraint_type = "Free"
            elif np.all(~np.isnan(disp_array)):
                constraint_type = "Fixed"
            else:

                if not is_3d:
                    disp_2d: np.ndarray = disp_array[:3] if len(disp_array) >= 3 else np.pad(disp_array, (0, 3 - len(disp_array)), constant_values=np.nan)

                    if element_type == "2D_Beam":

                        if disp_2d[0] == 0 and disp_2d[1] == 0 and np.isnan(disp_2d[2]):
                            constraint_type = "Pinned"
                        elif disp_2d[0] == 0 and np.isnan(disp_2d[1]) and np.isnan(disp_2d[2]):
                            constraint_type = "Roller(x)"
                        elif np.isnan(disp_2d[0]) and disp_2d[1] == 0 and np.isnan(disp_2d[2]):
                            constraint_type = "Roller(y)"
                        elif disp_2d[0] == 0 and disp_2d[1] == 0 and disp_2d[2] == 0:
                            constraint_type = "Fixed"
                    elif element_type == "2D_Truss":

                        if disp_2d[0] == 0 and disp_2d[1] == 0:
                            constraint_type = "Pinned"
                        elif disp_2d[0] == 0 and np.isnan(disp_2d[1]):
                            constraint_type = "Roller(x)"
                        elif np.isnan(disp_2d[0]) and disp_2d[1] == 0:
                            constraint_type = "Roller(y)"
                    elif element_type == "2D_Plane":

                        if disp_array[0] == 0 and disp_array[1] == 0 and disp_array[2] == 0 and \
                           np.isnan(disp_array[3]) and np.isnan(disp_array[4]) and np.isnan(disp_array[5]):
                            constraint_type = "Fixed"
                        elif disp_array[0] == 0 and np.isnan(disp_array[1]) and np.isnan(disp_array[2]) and \
                             np.all(np.isnan(disp_array[3:])):
                            constraint_type = "Roller(x)"
                        elif np.isnan(disp_array[0]) and disp_array[1] == 0 and np.isnan(disp_array[2]) and \
                             np.all(np.isnan(disp_array[3:])):
                            constraint_type = "Roller(y)"
                        elif np.isnan(disp_array[0]) and np.isnan(disp_array[1]) and disp_array[2] == 0 and \
                             np.all(np.isnan(disp_array[3:])):
                            constraint_type = "Roller(z)"
                elif is_3d:

                    if element_type == "3D_Frame":

                        if np.all(disp_array == 0):
                            constraint_type = "Fixed"
                        elif disp_array[0] == 0 and disp_array[1] == 0 and disp_array[2] == 0 and \
                             np.all(np.isnan(disp_array[3:])):
                            constraint_type = "Pinned"
                        elif disp_array[0] == 0 and np.all(np.isnan(np.delete(disp_array, 0))):
                            constraint_type = "Roller(x)"
                        elif disp_array[1] == 0 and np.all(np.isnan(np.delete(disp_array, 1))):
                            constraint_type = "Roller(y)"
                        elif disp_array[2] == 0 and np.all(np.isnan(np.delete(disp_array, 2))):
                            constraint_type = "Roller(z)"
                    elif element_type == "3D_Truss":

                        if np.all(disp_array[:3] == 0):
                            constraint_type = "Fixed"
                        elif disp_array[0] == 0 and disp_array[1] == 0 and np.isnan(disp_array[2]):
                            constraint_type = "Pinned"
                        elif disp_array[0] == 0 and np.isnan(disp_array[1]) and np.isnan(disp_array[2]):
                            constraint_type = "Roller(x)"
                        elif np.isnan(disp_array[0]) and disp_array[1] == 0 and np.isnan(disp_array[2]):
                            constraint_type = "Roller(y)"
                        elif np.isnan(disp_array[0]) and np.isnan(disp_array[1]) and disp_array[2] == 0:
                            constraint_type = "Roller(z)"
                    elif element_type == "3D_Solid":

                        if np.all(disp_array[:3] == 0):
                            constraint_type = "Fixed"
                        elif disp_array[0] == 0 and np.all(np.isnan(np.delete(disp_array[:3], 0))):
                            constraint_type = "Roller(x)"
                        elif disp_array[1] == 0 and np.all(np.isnan(np.delete(disp_array[:3], 1))):
                            constraint_type = "Roller(y)"
                        elif disp_array[2] == 0 and np.all(np.isnan(np.delete(disp_array[:3], 2))):
                            constraint_type = "Roller(z)"

            if constraint_type is None:
                constraint_type = "Custom"
            node_data_i['constraint_type'] = constraint_type

    @staticmethod
    def process_forces_based_on_displacements(node_data: Dict[int, Dict[str, Any]], nodal_displacements: Dict[int, Any]) -> None:
        """
        Sets NaN force components to 0.0 for nodes that are NOT explicitly
        defined in the `nodal_displacements` dictionary. This implies that

        for such nodes, any NaN force values should be treated as zero,
        as their displacements are not being directly input by the user.
        Args:
            node_data (Dict[int, Dict[str, Any]]): A dictionary containing node data.
                                                   The 'force' tuple for each node will be updated.
            nodal_displacements (Dict[int, Any]): A dictionary containing explicitly defined
                                                   nodal displacements (e.g., from user input).
                                                   Nodes not in this dictionary will have their
                                                   NaN forces converted to 0.0.
        """

        for node_id, node_data_i in node_data.items():

            if node_id not in nodal_displacements:
                force: Union[Tuple[float, ...], None] = node_data_i.get('force')

                if force:
                    updated_force: Tuple[float, ...] = tuple(0.0 if np.isnan(f) else f for f in force)
                    node_data[node_id]['force'] = updated_force




class DistributedLoadConverter:
    """
    A class to convert distributed loads applied to structural elements into
    equivalent concentrated nodal forces and moments. This is crucial for
    finite element analysis where only nodal forces are directly considered
    in the global stiffness matrix.
    The class handles various types of distributed loads (uniform, triangular,
    trapezoidal, and equation-defined) and distributes them correctly based
    on the element type (truss or beam/frame) and structural dimension.
    Attributes:
        data (Dict[str, Any]): The imported structural data, which will be
                               modified to include converted concentrated loads.
        structure_info (Dict[str, Any]): Dictionary containing general
                                         structure information like DOFs per node,
                                         dimension, and element type.
        dofs_per_node (int): Number of degrees of freedom per node.
        dimension (str): The dimension of the structure ('2D' or '3D').
        element_type (str): The type of element being analyzed (e.g., '2DTruss',
                            '2DBeam', '3DFrame').
        element_nodes (Dict[str, Tuple[str, str]]): A mapping from element ID
                                                     to its connected node IDs.
        force_labels (List[str]): List of force labels (e.g., 'Fx', 'Fy', 'Mz')
                                  defining the DOFs at each node.
        direction_to_dofs (Dict[str, List[int]]): Maps global directions ('x', 'y', 'z')
                                                   to their corresponding force DOF indices.
        direction_to_moments (Dict[str, List[int]]): Maps global directions ('x', 'y', 'z')
                                                      to their corresponding moment DOF indices
                                                      (only for beam/frame elements).
    """


    def __init__(self, imported_data: Dict[str, Any]):
        """
        Initializes the DistributedLoadConverter with imported structural data.
        Args:
            imported_data (Dict[str, Any]): A dictionary containing all
                                            structural information, including
                                            'structure_info', 'elements', and
                                            optionally 'distributed_loads'.
        """
        self.data: Dict[str, Any] = imported_data
        self.structure_info: Dict[str, Any] = imported_data['structure_info']
        self.dofs_per_node: int = self.structure_info['dofs_per_node']
        self.dimension: str = self.structure_info['dimension']
        self.element_type: str = self.structure_info['element_type']

        if 'concentrated_loads' not in self.data:
            self.data['concentrated_loads'] = {}
        self.element_nodes: Dict[str, Tuple[str, str]] = {}

        for elem_id, elem_data in self.data['elements'].items():
            self.element_nodes[elem_id] = (elem_data['node1'], elem_data['node2'])
        self._setup_dof_mapping()


    # ---------------------------------------------
    # DOF MAPPING
    # ---------------------------------------------

    def _setup_dof_mapping(self) -> None:
        """
        Sets up DOF mappings based on the element type and structural dimension.
        This method populates `direction_to_dofs` for forces and
        `direction_to_moments` for moments, crucial for correctly
        distributing loads to the right DOFs.
        """
        self.force_labels: List[str] = self.structure_info['force_labels']
        self.direction_to_dofs: Dict[str, List[int]] = {
            'x': [i for i, label in enumerate(self.force_labels) if 'F' in label and 'x' in label.lower()],
            'y': [i for i, label in enumerate(self.force_labels) if 'F' in label and 'y' in label.lower()],
            'z': [i for i, label in enumerate(self.force_labels) if 'F' in label and 'z' in label.lower()]
        }

        if not self.element_type.endswith('Truss'):
            self.direction_to_moments: Dict[str, List[int]] = {
                'x': [i for i, label in enumerate(self.force_labels) if 'M' in label and 'x' in label.lower()],
                'y': [i for i, label in enumerate(self.force_labels) if 'M' in label and 'y' in label.lower()],
                'z': [i for i, label in enumerate(self.force_labels) if 'M' in label and 'z' in label.lower()]
            }

            if self.dimension == '2D' and self.element_type.endswith('Beam'):
                self.direction_to_moments['y'] = [i for i, label in enumerate(self.force_labels) 

                                                if 'M' in label and 'z' in label.lower()]
   
    # ---------------------------------------------
    # DISTRIBUTED LOAD CONVERSION
    # ---------------------------------------------

    def convert_distributed_loads(self) -> None:
        """
        Processes all distributed loads defined in the `self.data` and converts
        them into equivalent concentrated nodal forces and moments.
        This method iterates through each element that has distributed loads,
        determines the load type, calculates the equivalent nodal forces,
        and then distributes these forces to the connected nodes.
        """
        distributed_loads: Dict[str, Any] = self.data.get('distributed_loads', {})

        for elem_id, load_data in distributed_loads.items():

            if elem_id not in self.data['elements']:
                continue
            elem_info: Dict[str, Any] = self.data['elements'][elem_id]
            node1: str = elem_info['node1']
            node2: str = elem_info['node2']
            length: float = elem_info['length']
            load_type: str = load_data['type']
            direction: str = load_data['direction']
            params: Union[float, List[float], str] = load_data['parameters']
            feq: Tuple[float, ...]

            if load_type == 'Uniform':
                q: float

                if isinstance(params, (int, float)):
                    q = float(params)
                else:      
                    q = float(params[0])
                feq = self._calculate_uniform_load(q, length)
            elif load_type == 'Triangular':
                q1 = float(params[0]) 
                q2 = float(params[1]) 
                feq = self._calculate_triangular_load(q1, q2, length)
            elif load_type == 'Trapezoidal':
                q1 = float(params[0]) 
                q2 = float(params[1]) 
                feq = self._calculate_trapezoidal_load(q1, q2, length)
            elif load_type == 'Equation':
                equation: str = str(params) 
                feq = self._calculate_equation_load(equation, length)
            else:
                raise ValueError(f"Unknown load type: {load_type}")
            self._distribute_forces_to_nodes(elem_id, node1, node2, feq, direction)
 
    # ---------------------------------------------
    # EQUIVALENT NODAL FORCE CALCULATIONS
    # ---------------------------------------------

    def _calculate_uniform_load(self, q: float, length: float) -> Tuple[float, ...]:
        """
        Calculates equivalent nodal forces and moments for a uniform distributed load.
        Args:
            q (float): The magnitude of the uniform distributed load.
            length (float): The length of the element.
        Returns:
            Tuple[float, ...]: A tuple containing the equivalent nodal forces and moments.
                               For truss: (F1, F2)
                               For beam/frame: (F1, F2, M1, M2)
        """

        if self.element_type.endswith('Truss'):
            f1: float = q * length / 2
            f2: float = q * length / 2
            return (f1, f2)

        else:
            f1: float = q * length / 2
            f2: float = q * length / 2
            m1: float = -q * length**2 / 12
            m2: float = q * length**2 / 12
            return (f1, f2, m1, m2)

    def _calculate_triangular_load(self, q1: float, q2: float, length: float) -> Tuple[float, ...]:
        """
        Calculates equivalent nodal forces and moments for a triangular distributed load.
        Note: This method can also handle a simple triangular load by setting one of the q values to 0.
        Args:
            q1 (float): The load intensity at the first node.
            q2 (float): The load intensity at the second node.
            length (float): The length of the element.
        Returns:
            Tuple[float, ...]: A tuple containing the equivalent nodal forces and moments.
                               For truss: (F1, F2)
                               For beam/frame: (F1, F2, M1, M2)
        """

        if self.element_type.endswith('Truss'):
            f1: float = (2 * q1 + q2) * length / 6
            f2: float = (q1 + 2 * q2) * length / 6
            return (f1, f2)

        else:
            f1: float = (7 * q1 + 3 * q2) * length / 20
            f2: float = (3 * q1 + 7 * q2) * length / 20
            m1: float = -(q1 / 2 + q2 / 6) * length**2 / 10
            m2: float = (q1 / 6 + q2 / 2) * length**2 / 10
            return (f1, f2, m1, m2)

    def _calculate_trapezoidal_load(self, q1: float, q2: float, length: float) -> Tuple[float, ...]:
        """
        Calculates equivalent nodal forces and moments for a trapezoidal distributed load.
        This is done by superimposing a uniform load and a triangular load.
        Args:
            q1 (float): The load intensity at the first node.
            q2 (float): The load intensity at the second node.
            length (float): The length of the element.
        Returns:
            Tuple[float, ...]: A tuple containing the equivalent nodal forces and moments.
                               For truss: (F1, F2)
                               For beam/frame: (F1, F2, M1, M2)
        """
        uniform_q: float = min(q1, q2)
        triangular_q1: float
        triangular_q2: float = 0.0 

        if q1 > q2:
            triangular_q1 = q1 - uniform_q
        else:
            triangular_q1 = q2 - uniform_q
        uniform_fe: Tuple[float, ...] = self._calculate_uniform_load(uniform_q, length)

        if q2 > q1:
            triangular_fe: Tuple[float, ...] = self._calculate_triangular_load(triangular_q1, triangular_q2, length)

            if self.element_type.endswith('Truss'):
                triangular_fe = (triangular_fe[1], triangular_fe[0])
            else:
                triangular_fe = (triangular_fe[1], triangular_fe[0], -triangular_fe[3], -triangular_fe[2])
        else:
            triangular_fe = self._calculate_triangular_load(triangular_q1, triangular_q2, length)

        if self.element_type.endswith('Truss'):
            return (uniform_fe[0] + triangular_fe[0], uniform_fe[1] + triangular_fe[1])

        else:
            return (

                uniform_fe[0] + triangular_fe[0],
                uniform_fe[1] + triangular_fe[1],
                uniform_fe[2] + triangular_fe[2],
                uniform_fe[3] + triangular_fe[3]
            )


    def _calculate_equation_load(self, equation_str: str, length: float, n_points: int = 100) -> Tuple[float, ...]:
        """
        Numerically calculates equivalent nodal forces and moments for a distributed load
        defined by a mathematical equation using numerical integration (trapezoidal rule).
        Args:
            equation_str (str): The load magnitude as a function of x (e.g., "6000*x**2 + 3000*x + 2000").
                                'x' represents the local coordinate along the element, from 0 to length.
            length (float): The length of the element.
            n_points (int, optional): The number of integration points to use for numerical integration.
                                      Defaults to 100.
        Returns:
            Tuple[float, ...]: A tuple containing the equivalent nodal forces and moments.
                               For truss: (F1, F2)
                               For beam/frame: (F1, F2, M1, M2)
        Raises:
            ValueError: If the provided equation string is invalid.
        """

        try:
            q_func: Callable[[float], float] = eval(f"lambda x: {equation_str}", {"__builtins__": {}}, {})

        except Exception as e:
            raise ValueError(f"Invalid load equation: {equation_str}") from e
        x_vals: np.ndarray = np.linspace(0, length, n_points)
        q_vals: np.ndarray = np.array([q_func(x) for x in x_vals])

        if self.element_type.endswith('Truss'):
            total_force: float = np.trapz(q_vals, x_vals)
            f1: float = total_force / 2
            f2: float = total_force / 2
            return (f1, f2)

        else:
            shape_f1: np.ndarray = 1 - x_vals / length
            shape_f2: np.ndarray = x_vals / length
            f1: float = np.trapz(q_vals * shape_f1, x_vals)
            f2: float = np.trapz(q_vals * shape_f2, x_vals)
            m1_val: float = np.trapz(q_vals * x_vals * shape_f1, x_vals)
            m2_val: float = -np.trapz(q_vals * x_vals * shape_f2, x_vals)
            return (f1, f2, m1_val, m2_val)


    # ---------------------------------------------
    # FORCE DISTRIBUTION TO NODES
    # ---------------------------------------------

    def _distribute_forces_to_nodes(self, elem_id: str, node1: str, node2: str, feq: Tuple[float, ...], direction: str) -> None:
        """
        Distributes the calculated equivalent nodal forces and moments (`feq`)
        to the global degrees of freedom of the connected nodes.
        Args:
            elem_id (str): The ID of the element currently being processed.
            node1 (str): The ID of the first node of the element.
            node2 (str): The ID of the second node of the element.
            feq (Tuple[float, ...]): A tuple containing the equivalent nodal forces and moments
                                      (F1, F2, M1, M2 or F1, F2 for truss).
            direction (str): The global direction of the distributed load (e.g., 'Global_X', '-Global_Y').
        """
        sign: float = -1.0 if direction.startswith('-') else 1.0
        direction_axis: str = direction.split('_')[-1].lower()
        feq_signed: List[float] = [component * sign for component in feq]
        self.data['elements'][elem_id]['feq'] = feq_signed
        ndof: int = self.structure_info['dofs_per_node']
        node1_load: List[float] = [0.0] * ndof
        node2_load: List[float] = [0.0] * ndof
        force_dofs: List[int] = self.direction_to_dofs[direction_axis]

        if force_dofs:
            force_dof_idx = force_dofs[0]
            node1_load[force_dof_idx] += feq_signed[0]
            node2_load[force_dof_idx] += feq_signed[1]

        if not self.element_type.endswith('Truss') and hasattr(self, 'direction_to_moments'):
            moment_dofs: List[int] = self.direction_to_moments.get(direction_axis, [])

            if moment_dofs and len(feq_signed) > 2:
                moment_dof_idx = moment_dofs[0]
                node1_load[moment_dof_idx] += feq_signed[2]
                node2_load[moment_dof_idx] += feq_signed[3]
        self.data['elements'][elem_id]['feq1'] = node1_load
        self.data['elements'][elem_id]['feq2'] = node2_load
        self._add_nodal_force(node1, node1_load)
        self._add_nodal_force(node2, node2_load)
        
        
    # ---------------------------------------------
    # NODAL FORCE ACCUMULATION
    # ---------------------------------------------

    def _add_nodal_force(self, node_id: str, forces_to_add: List[float]) -> None:
        """
        Accumulates force values to the specified node's concentrated loads
        and its overall force vector.
        Args:
            node_id (str): The ID of the node to which forces are being added.
            forces_to_add (List[float]): A list of force values corresponding
                                         to each DOF of the node (e.g., [Fx, Fy, Mz]).
        """

        if 'force_dist' not in self.data['nodes'][node_id] or self.data['nodes'][node_id]['force_dist'] is None:
            self.data['nodes'][node_id]['force_dist'] = [0.0] * self.dofs_per_node
        node_force_dist: List[float] = list(self.data['nodes'][node_id]['force_dist'])

        if node_id not in self.data['concentrated_loads']:
            initial_load_tuple: List[Union[float, Any]] = [np.nan] * self.dofs_per_node

            for i, val in enumerate(forces_to_add):

                if val != 0.0:
                    initial_load_tuple[i] = val
            self.data['concentrated_loads'][node_id] = tuple(initial_load_tuple)
        else:
            current_concentrated_load: List[Union[float, Any]] = list(self.data['concentrated_loads'][node_id])

            for i, val in enumerate(forces_to_add):

                if val != 0.0:

                    if np.isnan(current_concentrated_load[i]):
                        current_concentrated_load[i] = val
                    else:
                        current_concentrated_load[i] += val
            self.data['concentrated_loads'][node_id] = tuple(current_concentrated_load)

        if node_id in self.data['nodes']:
            node_data: Dict[str, Any] = self.data['nodes'][node_id]
            current_node_force: List[Union[float, Any]] = list(node_data['force'])

            for i, val in enumerate(forces_to_add):

                if val != 0.0:

                    if np.isnan(current_node_force[i]):
                        current_node_force[i] = val
                    else:
                        current_node_force[i] += val
                node_force_dist[i] += val
            node_data['force'] = tuple(current_node_force)
            node_data['force_dist'] = tuple(node_force_dist)
            node_data['from_dist'] = True
                   