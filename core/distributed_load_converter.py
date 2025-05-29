import numpy as np
from typing import Dict, Any, Tuple, List, Union, Callable


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



if __name__ == "__main__":
    imported_data = {
        'structure_info': {
            'dofs_per_node': 6, 'dimension': '3D',
            'force_labels': ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'],
            'displacement_labels': ['u', 'v', 'w', 'ɸx', 'ɸy', 'ɸz'],
            'element_type': '3D_Frame'
        },
        'cross_sections': {
            'RECT1': {'type': 'Solid_Rectangular', 'dimensions': (0.2, 0.3)}
        },
        'elements': {
            1: {'node1': 1, 'node2': 2, 'section_code': 'RECT1', 'E': 210.0, 'v': 0.3, 'G': 80.76923076923076, 'J': 0.0017999999999999997, 'Iy': 0.00044999999999999993, 'Iz': 0.00020000000000000006, 'A': 0.06, 'length': 5.0, 'angle': (0.0, 0.0), 'lmn': (1.0, 0.0, 0.0), 'angle_x': 0.7853981633974483, 'angle_y': 0.0, 'angle_z': 0.0},
            2: {'node1': 2, 'node2': 3, 'section_code': 'RECT1', 'E': 210.0, 'v': 0.3, 'G': 80.76923076923076, 'J': 0.0017999999999999997, 'Iy': 0.00044999999999999993, 'Iz': 0.00020000000000000006, 'A': 0.06, 'length': 5.0, 'angle': (0.0, 0.0), 'lmn': (1.0, 0.0, 0.0), 'angle_x': 0.7853981633974483, 'angle_y': 0.0, 'angle_z': 0.0},
            3: {'node1': 3, 'node2': 4, 'section_code': 'RECT1', 'E': 210.0, 'v': 0.3, 'G': 80.76923076923076, 'J': 0.0017999999999999997, 'Iy': 0.00044999999999999993, 'Iz': 0.00020000000000000006, 'A': 0.06, 'length': 5.0, 'angle': (0.0, 0.0), 'lmn': (1.0, 0.0, 0.0), 'angle_x': 0.7853981633974483, 'angle_y': 0.0, 'angle_z': 0.0},
            4: {'node1': 4, 'node2': 5, 'section_code': 'RECT1', 'E': 210.0, 'v': 0.3, 'G': 80.76923076923076, 'J': 0.0017999999999999997, 'Iy': 0.00044999999999999993, 'Iz': 0.00020000000000000006, 'A': 0.06, 'length': 5.0, 'angle': (0.0, 0.0), 'lmn': (1.0, 0.0, 0.0), 'angle_x': 0.7853981633974483, 'angle_y': 0.0, 'angle_z': 0.0}
        },
        'nodes': {
            1: {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'force': (20.0, np.nan, np.nan, np.nan, np.nan, np.nan), 'displacement': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            2: {'X': 5.0, 'Y': 0.0, 'Z': 0.0, 'force': (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), 'displacement': (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)},
            3: {'X': 5.0, 'Y': 3.0, 'Z': 0.0, 'force': (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), 'displacement': (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)},
            4: {'X': 5.0, 'Y': 3.0, 'Z': 4.0, 'force': (0.0, -36.0, 0.0, 0.0, 0.0, 0.0), 'displacement': (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)},
            5: {'X': 0.0, 'Y': 4.0, 'Z': 5.0, 'force': (np.nan, np.nan, np.nan, 0, 0, 0), 'displacement': (0, 0.0, 0, np.nan, np.nan, np.nan)}
        },
        'nodal_displacements': {1: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 5: (0, 0.0, 0, np.nan, np.nan, np.nan)},
        'concentrated_loads': {1: (20.0, np.nan, np.nan, np.nan, np.nan, np.nan), 4: (0.0, -36.0, 0.0, 0.0, 0.0, 0.0)},
        'distributed_loads': {
            1: {'type': 'Trapezoidal', 'direction': '-Global_Y', 'parameters': (9, 12)},
            2: {'type': 'Triangular', 'direction': '-Global_X', 'parameters': (16, 0.5)},
            3: {'type': 'Uniform', 'direction': 'Global_X', 'parameters': (6,)},
            4: {'type': 'Equation', 'direction': '-Global_Z', 'parameters': '6*x**2 + 3*x + 2'}
        },
        'units': {
            'Modulus (E,G)': 'GPa', 'Moment of Inertia (Iy,Iz,J)': 'm⁴',
            'Cross-Sectional Area (A)': 'cm²', 'Force (Fx,Fy,Fz)': 'kN',
            'Displacement (Dx,Dy,Dz)': 'mm', 'Position (X,Y,Z)': 'm'
        }
    }
    converter = DistributedLoadConverter(imported_data)
    converter.convert_distributed_loads()
    print(imported_data['elements'][3]['feq1'])