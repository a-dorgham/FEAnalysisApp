import sys
import os
from PyQt6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QVBoxLayout, QWidget
)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from utils.error_handler import ErrorHandler

class UnitsHandling:
    """
    The `UnitsHandling` class provides functionalities for managing and converting units
    within the structural analysis application. It stores conversion factors for various
    physical quantities and offers methods to convert imported data to a consistent
    unit system (either metric or imperial) for calculations, and to allow users
    to select their preferred display units.
    - `imported_data` (dict): A dictionary holding all the imported structural data,
      including nodes, elements, materials, cross-sections, and loads, along with
      their original and converted unit information.
    - `left_dock_window` (QWidget): A reference to the left dock QWidget window, used
      primarily for updating the UI after unit conversions.
    """
    imported_data: dict = {}
    left_dock_window: QWidget = {} 

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self, imported_data: dict, left_dock_window: QWidget) -> None:
        """
        Initializes the `UnitsHandling` class.
        - `imported_data` (dict): The dictionary containing all imported structural data.
        - `left_dock_window` (QWidget): A reference to the left dock window, used for UI updates.
        """
        super().__init__()
        UnitsHandling.imported_data = imported_data
        UnitsHandling.left_dock_window = left_dock_window

    # ---------------------------------------------
    # UNIT CONVERSION LOGIC
    # ---------------------------------------------

    @staticmethod
    def convert_units(value: float, current_unit: str, target_unit: str) -> float:
        """
        Converts a numerical value from a `current_unit` to a `target_unit`.
        This method supports conversions for various quantities including length,
        area, inertia, force, displacement, modulus, volume, mass, force/length, and density.
        It also handles compound units like force/length.
        - `value` (float): The numerical value to be converted.
        - `current_unit` (str): The unit of the `value` (e.g., "m", "kN", "MPa").
        - `target_unit` (str): The desired unit for the `value` (e.g., "mm", "lbf", "ksi").
        - `float`: The converted value in the `target_unit`.
        """
        conversions = {
            "length": {
                "m": {"mm": 1e3, "cm": 1e2, "in": 39.3701, "ft": 3.28084},
                "mm": {"m": 1e-3, "cm": 0.1, "in": 0.0393701, "ft": 0.00328084},
                "cm": {"m": 1e-2, "mm": 10, "in": 0.393701, "ft": 0.0328084},
                "in": {"m": 0.0254, "mm": 25.4, "cm": 2.54, "ft": 1/12},
                "ft": {"m": 0.3048, "mm": 304.8, "cm": 30.48, "in": 12},
            },
            "area": {
                "m²": {"mm²": 1e6, "cm²": 1e4, "in²": 1550.003, "ft²": 10.7639},
                "mm²": {"m²": 1e-6, "cm²": 0.01, "in²": 0.00155, "ft²": 1.07639e-5},
                "cm²": {"m²": 1e-4, "mm²": 100, "in²": 0.155, "ft²": 0.00107639},
                "in²": {"m²": 0.00064516, "mm²": 645.16, "cm²": 6.4516, "ft²": 0.00694444},
                "ft²": {"m²": 0.092903, "mm²": 92903, "cm²": 929.03, "in²": 144},
            },
            "inertia": {
                "m⁴": {"mm⁴": 1e12, "cm⁴": 1e8, "in⁴": 2.4025e7, "ft⁴": 115.862},
                "mm⁴": {"m⁴": 1e-12, "cm⁴": 1e-4, "in⁴": 2.4025e-5, "ft⁴": 1.15862e-5},
                "cm⁴": {"m⁴": 1e-8, "mm⁴": 1e4, "in⁴": 0.00064516, "ft⁴": 0.00000115862},
                "in⁴": {"m⁴": 4.16231e-7, "mm⁴": 416231, "cm⁴": 1550.003, "ft⁴": 0.000578704},
                "ft⁴": {"m⁴": 0.00863097, "mm⁴": 8.63097e9, "cm⁴": 8.63097e7, "in⁴": 1728},
            },
            "force": {
                "N": {"kN": 1e-3, "lbf": 0.224809, "kip": 0.000224809},
                "kN": {"N": 1e3, "lbf": 224.809, "kip": 0.224809},
                "lbf": {"N": 4.44822, "kN": 0.00444822, "kip": 0.001},
                "kip": {"N": 4448.22, "kN": 4.44822, "lbf": 1000},
            },
            "displacement": {
                "m": {"mm": 1e3, "cm": 1e2, "in": 39.3701, "ft": 3.28084},
                "mm": {"m": 1e-3, "cm": 0.1, "in": 0.0393701, "ft": 0.00328084},
                "cm": {"m": 1e-2, "mm": 10, "in": 0.393701, "ft": 0.0328084},
                "in": {"m": 0.0254, "mm": 25.4, "cm": 2.54, "ft": 1/12},
                "ft": {"m": 0.3048, "mm": 304.8, "cm": 30.48, "in": 12},
            },
            "modulus": {
                "Pa": {"GPa": 1e-9, "MPa": 1e-6, "ksi": 1.45038e-4, "psi": 0.000145038},
                "GPa": {"Pa": 1e9, "MPa": 1e3, "ksi": 145.038, "psi": 145038},
                "MPa": {"Pa": 1e6, "GPa": 1e-3, "ksi": 0.145038, "psi": 145.038},
                "ksi": {"Pa": 6.89476e6, "GPa": 0.00689476, "MPa": 6.89476, "psi": 1000},
                "psi": {"Pa": 6894.76, "GPa": 6.89476e-6, "MPa": 0.00689476, "ksi": 0.001},
            },
            "volume": {
                "m³": {"cm³": 1e6, "mm³": 1e9, "in³": 61023.7, "ft³": 35.3147},
                "cm³": {"m³": 1e-6, "mm³": 1e3, "in³": 0.0610237, "ft³": 3.53147e-5},
                "mm³": {"m³": 1e-9, "cm³": 1e-3, "in³": 6.10237e-5, "ft³": 3.53147e-8},
                "in³": {"m³": 1.63871e-5, "cm³": 16.3871, "mm³": 16387.1, "ft³": 5.78704e-4},
                "ft³": {"m³": 0.0283168, "cm³": 28316.8, "mm³": 2.83168e7, "in³": 1728},
            },
            "mass": {
                "kg": {"g": 1e3, "lb": 2.20462, "slug": 0.0685218},
                "g": {"kg": 1e-3, "lb": 0.00220462, "slug": 6.85218e-5},
                "lb": {"kg": 0.453592, "g": 453.592, "slug": 0.031081},
                "slug": {"kg": 14.5939, "g": 14593.9, "lb": 32.174},
            },
            "force/length": {
                "N/m": {
                    "N/mm": 1e-3,
                    "kN/m": 1e-3,
                    "lbf/in": 0.00571015,
                    "lbf/ft": 0.0685218,
                    "kip/in": 5.71015e-6,
                    "kip/ft": 6.85218e-5
                },
                "N/mm": {
                    "N/m": 1e3,
                    "kN/m": 1,
                    "lbf/in": 5.71015,
                    "lbf/ft": 68.5218,
                    "kip/in": 0.00571015,
                    "kip/ft": 0.0685218
                },
                "kN/m": {
                    "N/m": 1e3,
                    "N/mm": 1,
                    "lbf/in": 5.71015,
                    "lbf/ft": 68.5218,
                    "kip/in": 0.00571015,
                    "kip/ft": 0.0685218
                },
                "lbf/in": {
                    "N/m": 175.127,
                    "N/mm": 0.175127,
                    "kN/m": 0.175127,
                    "lbf/ft": 12,
                    "kip/in": 0.001,
                    "kip/ft": 0.012
                },
                "lbf/ft": {
                    "N/m": 14.5939,
                    "N/mm": 0.0145939,
                    "kN/m": 0.0145939,
                    "lbf/in": 0.0833333,
                    "kip/in": 8.33333e-5,
                    "kip/ft": 0.001
                },
                "kip/in": {
                    "N/m": 175126.8,
                    "N/mm": 175.1268,
                    "kN/m": 175.1268,
                    "lbf/in": 1000,
                    "lbf/ft": 12000,
                    "kip/ft": 12
                },
                "kip/ft": {
                    "N/m": 14593.9,
                    "N/mm": 14.5939,
                    "kN/m": 14.5939,
                    "lbf/in": 83.3333,
                    "lbf/ft": 1000,
                    "kip/in": 0.0833333
                }
            },
            "density": {
                "kg/m³": {
                    "g/cm³": 0.001,
                    "lb/in³": 3.61273e-5,
                    "lb/ft³": 0.062428,
                    "slug/ft³": 0.00194032
                },
                "g/cm³": {
                    "kg/m³": 1000,
                    "lb/in³": 0.0361273,
                    "lb/ft³": 62.428,
                    "slug/ft³": 1.94032
                },
                "lb/in³": {
                    "kg/m³": 27679.9,
                    "g/cm³": 27.6799,
                    "lb/ft³": 1728,
                    "slug/ft³": 53.7079
                },
                "lb/ft³": {
                    "kg/m³": 16.0185,
                    "g/cm³": 0.0160185,
                    "lb/in³": 0.000578704,
                    "slug/ft³": 0.031081
                },
                "slug/ft³": {
                    "kg/m³": 515.379,
                    "g/cm³": 0.515379,
                    "lb/in³": 0.0186169,
                    "lb/ft³": 32.174
                }
            }
        }

        for category, unit_map in conversions.items():

            if current_unit in unit_map and target_unit in unit_map[current_unit]:
                return value * unit_map[current_unit][target_unit]

        if "/" in current_unit and "/" in target_unit:
            current_num, current_den = current_unit.split("/")
            target_num, target_den = target_unit.split("/")

            for category, unit_map in conversions.items():

                if current_num in unit_map and target_num in unit_map[current_num]:
                    num_factor = unit_map[current_num][target_num]
                    break
            else:
                num_factor = 1

            for category, unit_map in conversions.items():

                if current_den in unit_map and target_den in unit_map[current_den]:
                    den_factor = unit_map[current_den][target_den]
                    break
            else:
                den_factor = 1

            if num_factor != 1 or den_factor != 1:
                return value * (num_factor / den_factor)

        return value

    @classmethod
    def convert_data_to_standard_units(cls, data: dict = None, target_units: dict = None, convert_all: bool = True,
                                       convert_nodes: bool = False, convert_elements: bool = False,
                                       convert_materials: bool = False, convert_cross_sections: bool = False,
                                       convert_distributed_loads: bool = False) -> None:
        """
        Converts relevant data within the `imported_data` dictionary to a standard
        unit system (metric or imperial). This method can be called to convert
        all data or specific subsets (nodes, elements, materials, etc.).
        If `target_units` are not provided, it infers the target system (metric/imperial)
        from the saved units of the "Position (X,Y,Z)" and sets up a default set of
        target units for calculation.
        - `data` (dict, optional): The data dictionary to be converted. If `None`,
          `UnitsHandling.imported_data` is used. Defaults to `None`.
        - `target_units` (dict, optional): A dictionary specifying the target units

          for each physical quantity. If `None`, standard units are inferred.
          Defaults to `None`.
        - `convert_all` (bool): If `True`, all categories of data will be converted.
          Individual `convert_` flags will be ignored. Defaults to `True`.
        - `convert_nodes` (bool): If `True`, node data (positions, forces, displacements)
          will be converted. Only applicable if `convert_all` is `False`. Defaults to `False`.
        - `convert_elements` (bool): If `True`, element data (material properties,
          geometric properties, length) will be converted. Only applicable if `convert_all`
          is `False`. Defaults to `False`.
        - `convert_materials` (bool): If `True`, material properties (modulus, density)
          will be converted. Only applicable if `convert_all` is `False`. Defaults to `False`.
        - `convert_cross_sections` (bool): If `True`, cross-section dimensions will be
          converted. Only applicable if `convert_all` is `False`. Defaults to `False`.
        - `convert_distributed_loads` (bool): If `True`, distributed load magnitudes
          and equation parameters will be converted. Only applicable if `convert_all`
          is `False`. Defaults to `False`.
        """
        calculation_conversion = False

        if data is not None:
            cls.imported_data = data

        if target_units is None:
            calculation_conversion = True
            is_metric = cls.imported_data['saved_units']["Position (X,Y,Z)"] in {"m", "cm", "mm"}
            target_units = {
                "Position (X,Y,Z)": "m" if is_metric else "in",
                "Cross-Sectional Area (A)": "m²" if is_metric else "in²",
                "Moment of Inertia (Iy,Iz,J)": "m⁴" if is_metric else "in⁴",
                "Modulus (E,G)": "Pa" if is_metric else "psi",
                "Force (Fx,Fy,Fz)": "N" if is_metric else "lbf",
                "Force/Length (F/L)": "N/m" if is_metric else "lbf/in",
                "Displacement (Dx,Dy,Dz)": "m" if is_metric else "in",
                "Density (ρ)": "kg/m³" if is_metric else "lb/in³"
            }
        cls.imported_data['calc_units'] = target_units
        structure_info = UnitsHandling.imported_data['structure_info']
        force_labels = structure_info["force_labels"]
        displacement_labels = structure_info["displacement_labels"]
        is_3D = "Fz" in force_labels
        node_data = {}
        element_data = {}
        material_data = {}
        cross_section_data = {}
        distributed_load_data = {}

        if convert_all or convert_nodes:

            for node, data in cls.imported_data['nodes'].items():
                x = cls.convert_units(data["X"], cls.imported_data['saved_units']["Position (X,Y,Z)"], target_units["Position (X,Y,Z)"])
                y = cls.convert_units(data["Y"], cls.imported_data['saved_units']["Position (X,Y,Z)"], target_units["Position (X,Y,Z)"])
                z = cls.convert_units(data["Z"], cls.imported_data['saved_units']["Position (X,Y,Z)"], target_units["Position (X,Y,Z)"]) if is_3D and "Z" in data else None
                force = tuple(cls.convert_units(f, cls.imported_data['saved_units']["Force (Fx,Fy,Fz)"], target_units["Force (Fx,Fy,Fz)"]) for f in data["force"])
                displacement = tuple(cls.convert_units(d, cls.imported_data['saved_units']["Displacement (Dx,Dy,Dz)"], target_units["Displacement (Dx,Dy,Dz)"]) for d in data["displacement"])
                node_data[node] = {
                    "X": x,
                    "Y": y,
                    "force": force,
                    "displacement": displacement
                }

                if is_3D:
                    node_data[node]["Z"] = z
        else:
            node_data = cls.imported_data.get('nodes', {})

        if convert_all or convert_elements:

            for element, data in cls.imported_data['elements'].items():
                E = cls.convert_units(data["E"], cls.imported_data['saved_units']["Modulus (E,G)"], target_units["Modulus (E,G)"])
                G = cls.convert_units(data["G"], cls.imported_data['saved_units']["Modulus (E,G)"], target_units["Modulus (E,G)"])
                A = cls.convert_units(data["A"], cls.imported_data['saved_units']["Cross-Sectional Area (A)"], target_units["Cross-Sectional Area (A)"])
                Iy = cls.convert_units(data["Iy"], cls.imported_data['saved_units']["Moment of Inertia (Iy,Iz,J)"], target_units["Moment of Inertia (Iy,Iz,J)"])
                Iz = cls.convert_units(data["Iz"], cls.imported_data['saved_units']["Moment of Inertia (Iy,Iz,J)"], target_units["Moment of Inertia (Iy,Iz,J)"])
                J = cls.convert_units(data["J"], cls.imported_data['saved_units']["Moment of Inertia (Iy,Iz,J)"], target_units["Moment of Inertia (Iy,Iz,J)"])
                length = cls.convert_units(data["length"], cls.imported_data['saved_units']["Position (X,Y,Z)"], target_units["Position (X,Y,Z)"])
                element_data[element] = data
                element_data[element].update({
                    "node1": data["node1"],
                    "node2": data["node2"],
                    "E": E,
                    "G": G,
                    "A": A,
                    "Iy": Iy,
                    "Iz": Iz,
                    "J": J,
                    "length": length,
                    "angle": data.get("angle", 0)
                })
        else:
            element_data = cls.imported_data.get('elements', {})

        if convert_all or convert_materials:

            for code, data in cls.imported_data['materials'].items():
                converted_properties = {}

                for prop, value in data['properties'].items():

                    if prop == 'E' or prop == 'G':
                        converted_properties[prop] = cls.convert_units(value, cls.imported_data['saved_units']["Modulus (E,G)"], target_units["Modulus (E,G)"])
                    elif prop == 'density':
                        converted_properties[prop] = cls.convert_units(value, cls.imported_data['saved_units']["Density (ρ)"], target_units["Density (ρ)"])
                    else:
                        converted_properties[prop] = value
                material_data[code] = data.copy()
                material_data[code]['properties'] = converted_properties
        else:
            material_data = cls.imported_data.get('materials', {})

        if convert_all or convert_cross_sections:

            for code, data in cls.imported_data['cross_sections'].items():
                section_type = data['type']
                dim_dict = data['dimensions']
                converted_dims = {
                    param: cls.convert_units(float(value),
                                        cls.imported_data['saved_units']["Position (X,Y,Z)"],
                                        target_units["Position (X,Y,Z)"])

                    for param, value in dim_dict.items()
                }
                cross_section_data[code] = {
                    **data,
                    'dimensions': converted_dims
                }
        else:
            cross_section_data = cls.imported_data.get('cross_sections', {})

        if convert_all or convert_distributed_loads:
            param_labels = {
                "Uniform": ["Magnitude"],
                "Rectangular": ["Start Magnitude", "End Magnitude"],
                "Triangular": ["Start Mag.", "End Mag."],
                "Trapezoidal": ["Start Mag.", "End Mag."],
                "Equation": ["Equation"]
            }
            forcePerLength_unit = cls.imported_data['saved_units']["Force/Length (F/L)"]
            position_unit = cls.imported_data['saved_units']["Position (X,Y,Z)"]

            for element, load_data in cls.imported_data.get('distributed_loads', {}).items():
                load_type = load_data['type']
                parameters = load_data['parameters']
                converted_params = []

                if load_type == "Equation":
                    forcePerLength_conv = cls.convert_units(1, forcePerLength_unit, target_units["Force/Length (F/L)"])
                    equation = parameters

                    if forcePerLength_conv != 1:
                        equation = cls.convert_equation_units(equation, forcePerLength_conv)
                    converted_params = equation
                else:

                    for i, param in enumerate(parameters if isinstance(parameters, (tuple, list)) else [parameters]):
                        label = param_labels[load_type][i]

                        if "Magnitude" in label or "Mag." in label:
                            param_value = cls.convert_units(param, forcePerLength_unit,  target_units["Force/Length (F/L)"])
                        else:
                            param_value = cls.convert_units(param, position_unit, target_units["Position (X,Y,Z)"])
                        converted_params.append(param_value)
                distributed_load_data[element] = {
                    'type': load_type,
                    'direction': load_data['direction'],
                    'parameters': tuple(converted_params) if isinstance(converted_params, list) else converted_params
                }
        else:
            distributed_load_data = cls.imported_data.get('distributed_loads', {})

        if calculation_conversion:
            cls.imported_data['converted_nodes'] = node_data
            cls.imported_data['converted_elements'] = element_data
            cls.imported_data['converted_materials'] = material_data
            cls.imported_data['converted_cross_sections'] = cross_section_data
            cls.imported_data['converted_distributed_loads'] = distributed_load_data
        else:
            cls.imported_data['nodes'] = node_data
            cls.imported_data['elements'] = element_data
            cls.imported_data['materials'] = material_data
            cls.imported_data['cross_sections'] = cross_section_data
            cls.imported_data['distributed_loads'] = distributed_load_data

    @classmethod
    def convert_equation_units(cls, equation: str, conversion_factor: float) -> str:
        """
        Converts units within an equation string by applying a `conversion_factor`
        to each numerical term.
        This method is designed to handle mathematical expressions like "6*x**2 + 3*x + 2".
        - `equation` (str): The equation string to be converted.
        - `conversion_factor` (float): The factor by which numerical coefficients
          in the equation should be multiplied for unit conversion.
        - `str`: The equation string with converted units.
        """

        if conversion_factor == 1:
            return equation

        terms = equation.split('+')
        converted_terms = []

        for term in terms:
            term = term.strip()

            if not term:
                continue

            if '-' in term and term.index('-') > 0:
                sub_terms = term.split('-')

                if sub_terms[0]:
                    converted_sub_term = cls._convert_single_term(sub_terms[0], conversion_factor)
                    converted_terms.append(converted_sub_term)

                for sub_term in sub_terms[1:]:

                    if sub_term:
                        converted_sub_term = cls._convert_single_term(sub_term, conversion_factor)
                        converted_terms.append(f"-{converted_sub_term}")
            else:
                converted_term = cls._convert_single_term(term, conversion_factor)
                converted_terms.append(converted_term)
        converted_eq = ' + '.join(converted_terms)
        converted_eq = converted_eq.replace('+ -', '- ')
        return converted_eq

    @classmethod
    def convert_default_values(cls, imported_data: dict, current_units: dict = {}, target_units: dict = {}) -> dict:
        """
        Converts default material and cross-section properties within the
        `imported_data` to `target_units` from `current_units`.
        This is specifically for default values, ensuring consistency.
        - `imported_data` (dict): The dictionary containing imported structural data.
        - `current_units` (dict): A dictionary mapping property names to their current units.
        - `target_units` (dict): A dictionary mapping property names to their target units.
        - `dict`: The `imported_data` dictionary with converted default values.
        """

        for code, material_data in imported_data['materials'].items():

            if code == "DEFAULT":
                converted_properties = {}

                for prop, value in material_data['properties'].items():

                    if prop == 'E' or prop == 'G':
                        converted_properties[prop] = cls.convert_units(value, current_units["Modulus (E,G)"], target_units["Modulus (E,G)"])
                    elif prop == 'density':
                        converted_properties[prop] = cls.convert_units(value, current_units["Density (ρ)"], target_units["Density (ρ)"])
                    else:
                        converted_properties[prop] = value
                imported_data['materials'][code]['properties'] = converted_properties

        for code, cross_section_data in imported_data['cross_sections'].items():

            if code == "DEFAULT":
                section_type = cross_section_data['type']
                dim_dict = cross_section_data['dimensions']
                converted_dims = {
                    param: cls.convert_units(float(value),
                                        current_units["Position (X,Y,Z)"],
                                        target_units["Position (X,Y,Z)"])

                    for param, value in dim_dict.items()
                }
                imported_data['cross_sections'][code]['dimensions'] = converted_dims
        return imported_data

    @staticmethod
    def _convert_single_term(term: str, conversion_factor: float) -> str:
        """
        Converts a single algebraic term by multiplying its numerical coefficient
        by the `conversion_factor`.
        This is a helper method used by `convert_equation_units`. It attempts to
        identify and modify the numerical part of a term while preserving variables.
        - `term` (str): The single term from an equation (e.g., "6*x**2", "3*x", "2", "x").
        - `conversion_factor` (float): The factor by which the coefficient should be multiplied.
        - `str`: The converted term string.
        """
        term = term.strip()

        if not term:
            return term

        if term.replace('.', '').replace('-', '').isdigit():
            return str(float(term) * conversion_factor)

        parts = term.split('*')

        if len(parts) > 1:

            try:
                coeff = float(parts[0])
                new_coeff = coeff * conversion_factor
                return f"{new_coeff}*{'*'.join(parts[1:])}"

            except ValueError:
                return term

        else:
            try:
                for i, c in enumerate(term):

                    if not c.isdigit() and c not in {'.', '-'}:
                        break

                if i > 0:
                    coeff_part = term[:i]
                    var_part = term[i:]
                    coeff = float(coeff_part)
                    new_coeff = coeff * conversion_factor
                    return f"{new_coeff}{var_part}"

                else:
                    return f"{conversion_factor}*{term}"

            except ValueError:
                return term

    @classmethod
    def convert_material_units(cls, materials_data: dict, target_units: dict) -> dict:
        """
        Converts the units of properties within a dictionary of material data
        to the specified `target_units`.
        This method is useful when importing materials from a library where
        units might differ from the application's current or target unit system.
        - `materials_data` (dict): A dictionary containing material names as keys
          and their properties (including a "Unit" dictionary) as values.
        - `target_units` (dict): A dictionary specifying the target units for
          "Modulus (E,G)" and "Density (ρ)".
        - `dict`: A new dictionary with material properties converted to the
          `target_units`.
        """
        converted_materials = {}

        for material_name, material_data in materials_data.items():
            converted_material = material_data.copy()
            units = material_data.get("Unit", {})

            for prop in ['E', 'density']:

                if prop in material_data and prop in units:

                    try:
                        value = float(material_data[prop])
                        current_unit = units[prop]

                        if prop == 'E':
                            target_unit = target_units["Modulus (E,G)"]
                        elif prop == 'density':
                            target_unit = target_units["Density (ρ)"]
                        converted_value = cls.convert_units(value, current_unit, target_unit)
                        converted_material[prop] = converted_value

                    except (ValueError, KeyError) as e:
                        print(f"Warning: Could not convert {prop} for {material_name}: {e}")
                        converted_material[prop] = value if 'value' in locals() else material_data[prop]
            converted_materials[material_name] = converted_material
        return converted_materials

    # ---------------------------------------------
    # UI FOR UNIT SELECTION
    # ---------------------------------------------

    @staticmethod
    def unit_selection_dialog(parent: QWidget, selected_units: dict = None) -> dict | None:
        """
        Opens a QDialog for selecting units. This is a deprecated method,
        `unit_selection_window` should be used instead.
        - `parent` (`QWidget`): The parent widget for the dialog.
        - `selected_units` (dict, optional): A dictionary of pre-selected units
          to initialize the combo boxes. Defaults to `None`.
        - `dict | None`: A dictionary of selected units if the dialog is accepted,
          otherwise `None`.
        """
        dialog = QDialog(parent)
        dialog.setWindowTitle("Select Units")
        dialog.setGeometry(300, 300, 400, 250)
        unit_options = {
            "Modulus (E,G)": ["Pa", "GPa", "MPa", "ksi"],
            "Moment of Inertia (Iy,Iz,J)": ["m⁴", "mm⁴", "cm⁴", "in⁴"],
            "Cross-Sectional Area (A)": ["m²", "mm²", "cm²", "in²"],
            "Force (Fx,Fy,Fz)": ["kN", "N", "lbf", "kip"],
            "Displacement (Dx,Dy,Dz)": ["m", "mm", "cm", "in"],
            "Position (X,Y,Z)": ["m", "mm", "cm", "in"],
        }
        selected_units = selected_units.copy() if selected_units else {}
        unit_selectors = {}
        layout = QVBoxLayout()

        for key, options in unit_options.items():
            layout.addWidget(QLabel(key))
            combo = QComboBox()
            combo.addItems(options)

            if key in selected_units:
                combo.setCurrentText(selected_units[key])
            unit_selectors[key] = combo
            layout.addWidget(combo)
        button_box = QWidget()
        button_layout = QVBoxLayout()
        save_button = QPushButton("Save")
        button_layout.addWidget(save_button)
        button_box.setLayout(button_layout)
        layout.addWidget(button_box)
        dialog.setLayout(layout)
        def save_units_dialog() -> None:
            """Saves the selected units from the dialog."""

            for key, combo in unit_selectors.items():
                selected_units[key] = combo.currentText()
            dialog.accept()
        save_button.clicked.connect(save_units_dialog)

        if dialog.exec() == QDialog.Accepted:
            return selected_units

        else:
            return None

    @staticmethod
    def unit_selection_window(parent: QWidget, selected_units: dict = None) -> None:
        """
        Displays a modal dialog allowing the user to select preferred units

        for various physical quantities. The selected units are then applied
        to the imported data and saved.
        - `parent` (`QWidget`): The parent widget for the dialog.
        - `selected_units` (dict, optional): A dictionary of pre-selected units
          to initialize the combo boxes. Defaults to `None`.
        """
        dialog = QDialog(parent)
        dialog.setWindowTitle("Select Units")
        dialog.setGeometry(700, 300, 400, 250)
        unit_options = {
            "Modulus (E,G)": ["Pa", "GPa", "MPa", "ksi"],
            "Moment of Inertia (Iy,Iz,J)": ["m⁴", "mm⁴", "cm⁴", "in⁴"],
            "Cross-Sectional Area (A)": ["m²", "mm²", "cm²", "in²"],
            "Force (Fx,Fy,Fz)": ["kN", "N", "lbf", "kip"],
            "Force/Length (F/L)": ["N/m","N/mm", "kN/m","lbf/in","lbf/ft","kip/in","kip/ft"],
            "Displacement (Dx,Dy,Dz)": ["m", "mm", "cm", "in"],
            "Position (X,Y,Z)": ["m", "mm", "cm", "in"],
            "Density (ρ)": ["kg/m³", "g/cm³","lb/in³","lb/ft³", "slug/ft³"]
        }
        selected_units = selected_units.copy() if selected_units else {}
        unit_selectors = {}
        layout = QGridLayout()

        try:
            row = 0

            for param, options in unit_options.items():
                label = QLabel(param)
                layout.addWidget(label, row, 0)
                combo_box = QComboBox()
                combo_box.addItems(options)
                combo_box.setCurrentText(selected_units.get(param, options[0]))
                layout.addWidget(combo_box, row, 1)
                unit_selectors[param] = combo_box
                row += 1
            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 1)
            button_layout = QHBoxLayout()
            apply_button = QPushButton("Apply")
            apply_button.clicked.connect(lambda: UnitsHandling.save_units(parent=dialog, file_path=None, unit_selectors=unit_selectors))
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(dialog.close)
            button_layout.addWidget(apply_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout, row, 0, 1, 2)
            dialog.setLayout(layout)
            dialog.exec()

        except Exception as e:
            ErrorHandler.handle_error(code=201, details=str(e))

    @staticmethod
    def save_units(parent: QDialog, file_path: str = None, unit_selectors: dict = None) -> None:
        """
        Saves the units selected in the unit selection dialog, applies the conversions
        to the `imported_data`, and updates the GUI.
        - `parent` (`QDialog`): The parent dialog from which the units were selected.
        - `file_path` (str, optional): The path to a JSON file where the selected units
          could be saved. If `None`, it defaults to "../data/selected_units.json".
          (Note: The actual file writing functionality is commented out in the provided code).
        - `unit_selectors` (dict, optional): A dictionary of unit selector widgets (QComboBox)
          from which the selected units are retrieved. If `None`, this method will not
          function correctly as it relies on these widgets. Defaults to `None`.
        """
        selected_units = {}

        if file_path is None:
            file_path = "../data/selected_units.json"

        try:

            for param, combo_box in unit_selectors.items():
                selected_units[param] = combo_box.currentText()
            UnitsHandling.convert_data_to_standard_units(target_units=selected_units.copy())
            UnitsHandling.imported_data['selected_units'] = selected_units.copy()
            UnitsHandling.imported_data['saved_units'] = selected_units.copy()
            UnitsHandling.left_dock_window.tree_widget.update_tree(imported_data=UnitsHandling.imported_data)
            parent.close()

        except Exception as e:
            ErrorHandler.handle_error(code=201, details=str(e))