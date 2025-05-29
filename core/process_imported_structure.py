import math
import numpy as np
from typing import Dict, Any, Tuple, Union


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