import os
import re
import json
import pandas as pd
import numpy as np
import ast
import difflib
from typing import Optional, Dict, Any, Union, List, Tuple
from PyQt6.QtWidgets import (QFileDialog, QMessageBox, QVBoxLayout, QWidget, QDialog)
from src.utils.errors import ErrorHandler
from src.config import StructureConfig
from src.utils.classes import ScrollableMessageBox
from PyQt6.QtWebEngineWidgets import QWebEngineView
from src.gui.viewers.file import FileViewer
from src.utils.units import UnitsHandling
from src.constants import theme
import math

class FileIO:
    """
    The `FileIO` class handles all file input and output operations for the FEAnalysisApp.
    This includes loading geometry files, reading structured input files for structural data
    (nodes, elements, loads, boundary conditions, materials, and cross-sections),
    and exporting data to various formats like Excel. It also provides utility
    functions for managing cross-section definitions and unit conversions.
    - `_cross_section_definitions` (dict or None): A cached dictionary of cross-section
      definitions loaded from `cross_section_library.json`. It's initialized to `None`
      and loaded only once when needed.
    """
    _cross_section_definitions: dict | None = None
    valid_units: Dict[str, List[str]] = None

    # ---------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------

    def __init__(self) -> None:
        """
        Initializes the `FileIO` class.
        Sets up a `QWebEngineView` for displaying file content,
        and initializes an `imported_data` dictionary to store parsed structural information.
        """
        super().__init__()
        self.setWindowTitle("FEAnalysisApp v1.0")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget() 
        self.main_layout = QVBoxLayout(self.central_widget)
        self.text_edit = QWebEngineView()  
        self.main_layout.addWidget(self.text_edit)
        self.imported_data: dict = {  
            "structure_info": {"element_type": None},
            "nodes": {},
            "elements": {},
            "nodal_displacements": {},
            "nodal_loads": {},
            "units": {},
            "materials": {},
            "cross_sections": {},
            "concentrated_loads": {},
            "distributed_loads": {},
            "saved_units": {},
            "selected_units": {},
            "converted_nodes": {},
            "converted_elements": {},
            "converted_materials": {},
            "converted_cross_sections": {},
            "converted_distributed_loads": {},
        }

    # ---------------------------------------------
    # CROSS-SECTION HANDLING
    # ---------------------------------------------

    @staticmethod
    def load_cross_section_definitions() -> dict:
        """
        Loads cross-section definitions from the `cross_section_library.json` file.
        The definitions are cached after the first load to prevent redundant file I/O.
        - `dict`: A dictionary containing the loaded cross-section definitions.
        """

        if FileIO._cross_section_definitions is None:

            try:
                with open("../data/cross_section_library.json", "r", encoding="utf-8") as file:
                    FileIO._cross_section_definitions = json.load(file)

            except FileNotFoundError as e:
                ErrorHandler.handle_error(code=301, details=f"Cross-section library file not found. {e}")
                FileIO._cross_section_definitions = {}

            except json.JSONDecodeError as e:
                ErrorHandler.handle_error(code=300, details=f"Invalid JSON format in cross-section library. {e}")
                FileIO._cross_section_definitions = {}
        return FileIO._cross_section_definitions

# ---------------------------------------------
# EXTENDED INPUT FILE VALIDATION FUNCTION
# ---------------------------------------------

    @staticmethod
    def validate_input_file(lines: List[str]) -> Tuple[bool, List[str]]:
        """
        Validates a list of lines representing a FEA input file.
        Performs checks for:
        - Required section presence
        - Structure type validity with suggestions
        - Syntax and expected input count for nodes, elements, materials, cross-sections, displacements, loads
        - Reports format issues using ErrorHandler
        Returns:
            Tuple[bool, List[str]]: (True, []) if valid, else (False, list of errors)
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        structure_types_data = FileIO.read_json(os.path.join(script_dir, '..', '..', "data", "defaults", "structures.json"))
        structure_types = structure_types_data if structure_types_data else {
            "2D_Beam": {"dofs_per_node": 3},
            "3D_Frame": {"dofs_per_node": 6},
            "2D_Truss": {"dofs_per_node": 2},
            "3D_Truss": {"dofs_per_node": 3},
            "2D_Plane": {"dofs_per_node": 1},
            "3D_Solid": {"dofs_per_node": 6}
        }
        cross_section_types: Dict[str, int] = {
            "Solid_Circular": 1,
            "Hollow_Circular": 2,
            "Solid_Rectangular": 3,
            "Hollow_Rectangular": 5,
            "I_Beam": 5,
            "C_Beam": 5,
            "L_Beam": 5
        }
        load_types: Dict[str, int] = {
            "Uniform": 1,
            "Trapezoidal": 2,
            "Triangular": 2,
            "Equation": 1
        }
        required_sections: List[str] = [
            "STRUCTURE_TYPE",
            "CROSS_SECTION_START", "CROSS_SECTION_END",
            "MATERIAL_START", "MATERIAL_END",
            "ELEMENT_DATA_START", "ELEMENT_DATA_END",
            "NODE_DATA_START", "NODE_DATA_END",
            "UNITS_DATA_START", "UNITS_DATA_END"
        ]
        section_data: Dict[str, List[str]] = {}
        current_section: Optional[str] = None
        structure_type: Optional[str] = None
        errors: List[str] = []

        for i, raw_line in enumerate(lines):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("STRUCTURE_TYPE"):
                section_data["STRUCTURE_TYPE"] = []

                try:
                    _, stype = line.split(":")
                    structure_type = stype.strip()

                    if structure_type not in structure_types:
                        closest = difflib.get_close_matches(structure_type, list(structure_types.keys()), n=1)
                        suggestion = f" Did you mean '{closest[0]}'?" if closest else ""
                        msg = f"Invalid STRUCTURE_TYPE '{structure_type}'.{suggestion}"
                        ErrorHandler.handle_error(code=201, details=msg, fatal=False)
                        errors.append(msg)

                except ValueError:
                    msg = f"Malformed STRUCTURE_TYPE line: '{line}'"
                    ErrorHandler.handle_error(code=201, details=msg, fatal=False)
                    errors.append(msg)
                continue

            if line in required_sections:
                current_section = line
                section_data[current_section] = []
                continue

            if current_section:
                section_data.setdefault(current_section, []).append(line)

        for section in required_sections:

            if section not in section_data:
                msg = f"Missing required section: {section}"
                ErrorHandler.handle_error(code=203, details=msg, fatal=False)
                errors.append(msg)
        def validate_lines(pattern: str, section: str, code: int):

            for line in section_data.get(section, []):
                stripped_line = line.split('#', 1)[0].strip()

                if not re.match(pattern, stripped_line):
                    msg = f"Malformed line in {section}: '{line}'"
                    ErrorHandler.handle_error(code=code, details=msg, fatal=False)
                    errors.append(msg)
        validate_lines(r"^\d+\s*\(.*\)$", "NODE_DATA_START", 110)
        validate_lines(r"^\d+\s*\(\d+,\s*\d+,\s*\w+,\s*\w+\)$", "ELEMENT_DATA_START", 111)
        dofs = structure_types.get(structure_type, {}).get("dofs_per_node", 0)
        def check_dof_count(section: str, code: int):

            for line in section_data.get(section, []):
                stripped_line = line.split('#', 1)[0].strip()
                match = re.match(r'^(\d+)\s*\((.*)\)$', stripped_line)

                if not match:
                    msg = f"Malformed line in {section}: '{line}'"
                    ErrorHandler.handle_error(code=code, details=msg, fatal=False)
                    errors.append(msg)
                    continue
                values = [v.strip() for v in match.group(2).split(',')]

                if len(values) != dofs:
                    msg = f"Incorrect number of DOF values for node {match.group(1)} in {section}. Expected {dofs}, got {len(values)}. Line: '{line}'"
                    ErrorHandler.handle_error(code=code, details=msg, fatal=False)
                    errors.append(msg)
        check_dof_count("NODES_DISPLACEMENT_START", 220)
        check_dof_count("NODES_CONCENTRATED_LOADS_START", 230)
        validate_lines(r"^\w+\s+\w+\s*\(\s*\d+(\.\d+)?,\s*\d+(\.\d+)?,\s*\d+(\.\d+)?\)$", "MATERIAL_START", 240)

        for line in section_data.get("CROSS_SECTION_START", []):
            stripped_line = line.split('#', 1)[0].strip()
            match = re.match(r"^(\w+)\s+(\w+)\s*\(([^)]+)\)$", stripped_line)

            if not match:
                msg = f"Malformed cross-section line: '{line}'"
                ErrorHandler.handle_error(code=200, details=msg, fatal=False)
                errors.append(msg)
                continue
            cs_code, cs_type, params = match.groups()
            param_count = len([p.strip() for p in params.split(",")])
            expected = cross_section_types.get(cs_type)

            if expected is None:
                msg = f"Unknown cross-section type '{cs_type}' in line: '{line}'"
                ErrorHandler.handle_error(code=200, details=msg, fatal=False)
                errors.append(msg)
            elif param_count != expected:
                msg = (f"Invalid number of parameters for cross-section '{cs_code}' of type '{cs_type}'. "
                       f"Expected {expected}, got {param_count}. Line: '{line}'")
                ErrorHandler.handle_error(code=200, details=msg, fatal=False)
                errors.append(msg)

        for line in section_data.get("ELEMENTS_DISTRIBUTED_LOADS_START", []):
            match = re.match(r"^(\d+)\s+(\w+)\s+(-?Global_[XYZ])\s*\(([^)]*)\)$", line)

            if not match:
                msg = f"Malformed distributed load line: '{line}'"
                ErrorHandler.handle_error(code=220, details=msg, fatal=False)
                errors.append(msg)
                continue
            elem_id, load_type, direction, param_str = match.groups()
            param_list = [p.strip() for p in param_str.split(",") if p.strip()]
            expected_count = load_types.get(load_type)

            if expected_count is None:
                msg = f"Unsupported load type '{load_type}' on element {elem_id}. Line: '{line}'"
                ErrorHandler.handle_error(code=220, details=msg, fatal=False)
                errors.append(msg)
            elif len(param_list) != expected_count:
                msg = (f"Incorrect number of parameters for load type '{load_type}' on element {elem_id}. "
                       f"Expected {expected_count}, got {len(param_list)}. Line: '{line}'")
                ErrorHandler.handle_error(code=220, details=msg, fatal=False)
                errors.append(msg)
        return len(errors) == 0, errors

    @staticmethod
    def parse_cross_section(section_code: str, section_type: str, dimensions: list) -> dict | None:
        """
        Validates and formats cross-section data based on predefined definitions.
        This method checks if the `section_type` is valid and if the number of
        provided `dimensions` matches the expected parameters for that section type.
        - `section_code` (str): A unique identifier for the cross-section.
        - `section_type` (str): The type of cross-section (e.g., "Solid_Circular", "I_Beam").
        - `dimensions` (list): A list of numerical values representing the dimensions
          of the cross-section.
        - `dict | None`: A dictionary containing the parsed cross-section data if valid,
          otherwise `None`. The dictionary includes 'code', 'type', and a 'dimensions'
          sub-dictionary mapping parameter names to their values.
        """
        cross_section_data = FileIO.load_cross_section_definitions()

        if section_type not in cross_section_data:
            ErrorHandler.handle_error(code=300, details=f"Invalid cross-section type: {section_type}")
            return None

        expected_params = cross_section_data[section_type]["parameters"]

        if len(dimensions) != len(expected_params):
            ErrorHandler.handle_error(code=300, details=f"Incorrect dimensions for {section_type}: Expected {expected_params}, Got {dimensions}")
            return None

        return {

            "code": section_code,
            "type": section_type,
            "dimensions": dict(zip(expected_params, dimensions))
        }
    # ---------------------------------------------
    # FILE LOADING & READING
    # ---------------------------------------------

    @staticmethod
    def load_geometry_file(parent: QWidget) -> None:
        """
        Opens a file dialog for the user to select a geometry file.
        If a file is selected and successfully read, its content is displayed
        in a scrollable message box. Error messages are shown for loading failures.
        - `parent` (`QWidget`): The parent widget for the file dialog and message boxes.
        """
        examples_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
        file_dialog = QFileDialog(parent, directory=examples_folder)
        file_path, _ = file_dialog.getOpenFileName(
            parent,
            "Load Geometry File",
            "",
            "Geometry Files (*.txt *.geo *.inp);;All Files (*)",
        )

        if file_path:

            if os.path.exists(file_path):

                try:
                    with open(file_path, "r") as file:
                        geometry_data = file.read()
                    msg_box = ScrollableMessageBox(parent, QMessageBox.Icon.Information)
                    msg_box.setWindowTitle("File Loaded")
                    msg_box.setText(f"Loaded file: {file_path}")
                    msg_box.setScrollableText(geometry_data)
                    max_height = int(parent.height() * 0.7)
                    msg_box.setMaximumScrollAreaHeight(max_height)
                    msg_box.exec()
                    msg_box.alignCenter()

                except PermissionError:
                    ErrorHandler.handle_error(
                        code=302,
                        details=f"Permission denied when trying to read file: {file_path}",
                        parent=parent,
                        fatal=False
                    )

                except UnicodeDecodeError:
                    ErrorHandler.handle_error(
                        code=303,
                        details=f"File encoding issue detected in: {file_path}",
                        parent=parent,
                        fatal=False
                    )

                except Exception as e:
                    ErrorHandler.handle_error(
                        code=300,
                        details=f"Unexpected error while reading file: {file_path}\n{str(e)}",
                        parent=parent,
                        fatal=False
                    )
            else:
                ErrorHandler.handle_warning(
                    code=1000,
                    details=f"File does not exist: {file_path}",
                    parent=parent
                )

    @staticmethod
    def read_input_file(file_path: str) -> dict | None:
        """
        Reads and parses a structured input file (`.fea` or `.txt`) containing
        FEA model data. It extracts information about structure type, nodes,
        elements, cross-sections, materials, boundary conditions (displacements),
        concentrated loads, distributed loads, and units.
        - `file_path` (str): The full path to the input file.
        - `dict | None`: A dictionary containing the parsed structural data if
          successful, otherwise `None`.
        - `FileNotFoundError`: If the specified `file_path` does not exist.
        - `ValueError`: If an unsupported structure type is encountered.
        - `Exception`: For various parsing errors within the file.
        """
        data = FileIO.initialize_default_data()
        default_units = data['units'].copy()

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines: List[str] = file.readlines()

        except Exception as e:
            ErrorHandler.handle_error(code=101, details=str(e), fatal=True)
            return None

        is_valid, validation_errors = FileIO.validate_input_file(lines)

        if not is_valid:
            print("Input file validation failed. Details:")

            for err in validation_errors:
                print(" -", err)
            return None

        section = None

        try:
            with open(file_path, 'r', encoding='utf-8') as file:

                for line in file:
                    line = line.strip()

                    if line.startswith("STRUCTURE_TYPE:"):
                        structure_type = line.split(":")[1].strip()
                        structure_info = StructureConfig.load_config("structure")[structure_type]

                        if structure_info:
                            dofs_per_node = structure_info["dofs_per_node"]
                            structure_dim = structure_info["dimension"]
                            data['structure_info'] = structure_info
                        else:
                            ErrorHandler.handle_error(code=310, details=f"Unsupported structure type: {structure_type}")
                    elif line.endswith("_START"):
                        section = line.replace("_START", "").lower()
                    elif line.endswith("_END"):
                        section = None
                    elif not line or line.startswith("#"):
                        continue
                    else:

                        try:

                            if section == "cross_section":
                                section_params = {
                                    "Solid_Circular": ["D"],
                                    "Hollow_Circular": ["D", "d"],
                                    "Solid_Rectangular": ["B", "H", "angle"],
                                    "Hollow_Rectangular": ["B", "H", "b", "h", "angle"],
                                    "I_Beam": ["B", "H", "tf", "tw", "angle"],
                                    "C_Beam": ["B", "H", "tf", "tw", "angle"],
                                    "L_Beam": ["B", "H", "tf", "tw", "angle"]
                                }
                                cross_section_parts = re.findall(r'\([^()]*\)|\S+', line)
                                section_code = cross_section_parts[0]
                                section_type = cross_section_parts[1]
                                raw_dimensions = ast.literal_eval(cross_section_parts[2])
                                param_list = section_params.get(section_type, [])

                                if isinstance(raw_dimensions, tuple):
                                    dimension_values = list(map(float, raw_dimensions))
                                    num_missing = len(param_list) - len(dimension_values)
                                    dimension_values.extend([0.0] * num_missing)
                                    dimensions = dict(zip(param_list, dimension_values))
                                elif isinstance(raw_dimensions, (int, float)) and param_list:
                                    dimensions = {param_list[0]: float(raw_dimensions)}
                                data['cross_sections'][section_code] = {
                                    'type': section_type,
                                    'dimensions': dimensions
                                }

                            if section == "material":
                                cross_section_parts = re.findall(r'\([^()]*\)|\S+', line)
                                section_code = cross_section_parts[0]
                                section_type = cross_section_parts[1]
                                properties_names = ["E", "v", "density"]
                                properties_values = ast.literal_eval(cross_section_parts[2])
                                data['materials'][section_code] = {
                                    'type': section_type,
                                    'properties': {
                                        "E": float(properties_values[0]),
                                        "v": float(properties_values[1]),
                                        "density": float(properties_values[2]),
                                    }
                                }
                            elif section == "element_data":
                                parts = re.findall(r'\([^()]*\)|\S+', line)
                                element_num = int(parts[0])
                                parts1 = parts[1].strip("()").split(',')
                                material_code = parts1[3].strip()
                                data["elements"][element_num] = {
                                    "node1": int(parts1[0]),
                                    "node2": int(parts1[1]),
                                    "section_code": parts1[2].strip(),
                                    "material_code": parts1[3].strip()
                                }
                            elif section == "node_data":
                                parts = re.findall(r'\([^()]*\)|\S+', line)
                                node_num = int(parts[0])
                                x, y, *z = ast.literal_eval(parts[1])
                                data["nodes"][node_num] = {
                                    "X": float(x),
                                    "Y": float(y)
                                }

                                if structure_dim == "3D":
                                    data["nodes"][node_num]["Z"] = float(z[0])
                                data["nodes"][node_num]["force"] = tuple(np.nan for _ in range(dofs_per_node))
                                data["nodes"][node_num]["displacement"] = tuple(np.nan for _ in range(dofs_per_node))
                            elif section == "nodes_displacement":
                                disp_parts = re.findall(r'\([^()]*\)|\S+', line)
                                node_num = int(disp_parts[0])
                                values = disp_parts[1].strip("()").split(",")
                                displacement_values = tuple(float(v) if v.strip().lower() != "free" else np.nan for v in values)
                                data["nodal_displacements"][node_num] = displacement_values
                                data["nodes"][node_num]['displacement'] = displacement_values
                            elif section == "nodes_concentrated_loads":
                                load_parts = re.findall(r'\([^()]*\)|\S+', line)
                                node_num = int(load_parts[0])
                                values = load_parts[1].strip("()").split(",")
                                force_values = tuple(float(v) if v.strip().lower() != "free" else np.nan for v in values)
                                data["concentrated_loads"][node_num] = force_values
                                data["nodes"][node_num]['force'] = force_values
                            elif section == "elements_distributed_loads":
                                parts = re.findall(r'\([^()]*\)|\S+', line)
                                element_number = int(parts[0])
                                load_type = parts[1]
                                load_direction = parts[2]
                                load_parameters = ast.literal_eval(" ".join(parts[3:]))

                                if isinstance(load_parameters, list):
                                    load_parameters_floats = list(map(float, load_parameters))

                                if isinstance(load_parameters, tuple):
                                    load_parameters_floats = tuple(map(float, load_parameters))
                                elif isinstance(load_parameters, (int, float)):
                                    load_parameters_floats = [float(load_parameters)]
                                elif isinstance(load_parameters, (str)):
                                    load_parameters_floats = load_parameters
                                else:
                                    load_parameters_floats = []
                                data["distributed_loads"][element_number] = {
                                    "type": load_type,
                                    "direction": load_direction,
                                    "parameters": load_parameters_floats
                                }
                            elif section == "units_data":
                                key, value = line.split(":")
                                key = key.strip()
                                value = value.strip()
                                valid_units = FileIO.valid_units

                                if key not in valid_units:
                                    ErrorHandler.handle_error(
                                        code=310,
                                        details=f"Unrecognized quantity in units section: '{key}",
                                        fatal=False
                                    )

                                if value not in valid_units[key]:
                                    allowed = ', '.join(valid_units[key])
                                    ErrorHandler.handle_error(
                                        code=310,
                                        details=f"Invalid unit '{value}' for '{key}'. Allowed units: {allowed}",
                                        fatal=False
                                    )
                                data["units"][key] = value
                                file_path_units = "../data/original_units.json"

                        except Exception as e:
                            ErrorHandler.handle_error(code=300, details=f"Error in importing line: {line} - {e}")
            data = UnitsHandling.convert_default_values(imported_data=data, current_units=default_units, target_units=data['units'])
            return data

        except FileNotFoundError as e:
            ErrorHandler.handle_error(code=301, details=f"{e}")
            return None

    @staticmethod
    def toolbar_open(parent: QWidget, file_path: str = None, element_type: str = None) -> tuple[dict | None, str | None]:
        """
        Opens a file dialog (if `file_path` is not provided), loads and validates
        the selected FEA input file, and returns the parsed data and its HTML representation.
        It also checks for structure type consistency.
        - `parent` (`QWidget`): The parent widget for the file dialog and message boxes.
        - `file_path` (str, optional): The explicit path to the file to open. If `None`,
        a file dialog is displayed. Defaults to `None`.
        - `element_type` (str, optional): The expected structure type (e.g., "2D_Beam").
        If the loaded file's structure type doesn't match, an error is shown. Defaults to `None`.
        - `tuple[dict | None, str | None]`: A tuple containing:
            - `dict | None`: The loaded and parsed data if successful, otherwise `None`.
            - `str | None`: The HTML representation of the loaded file content if successful,
            otherwise `None`.
        """
        loaded_data = {}
        file_content_html = {}
        file_viewer = FileViewer()

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                parent, "Open FEA Input File", "", "FEA Input Files (*.txt *.dat *.mak);;All Files (*)"
            )
        
        if not file_path:
            return None, None

        if os.path.exists(file_path):

            try:
                loaded_data = FileIO.read_input_file(file_path)

                if (
                    element_type is not None
                    and loaded_data["structure_info"]["element_type"] != element_type
                ):
                    error_message = (
                        "Structure type in the loaded file does not match the current structure type.\n"
                        f"Expected: {element_type}\n"
                        f"Found: {loaded_data['structure_info']['element_type']}"
                    )
                    ErrorHandler.handle_error(
                        code=200, 
                        details=error_message,
                        parent=parent,
                        fatal=False
                    )
                    print(error_message)
                    return None, None

                else:
                    file_content_html = file_viewer.generate_input_file_report(loaded_data, theme)
                    print(f"file: {file_path} was loaded successfully.")

            except PermissionError:
                ErrorHandler.handle_error(
                    code=302,
                    details=f"Permission denied when accessing file: {file_path}",
                    parent=parent,
                    fatal=False
                )
                return None, None

            except Exception as e:
                ErrorHandler.handle_error(
                    code=300,
                    details=f"Error loading or validating file: {str(e)}",
                    parent=parent,
                    fatal=False
                )
                print(f"Error loading or validating file: {e}")
                return None, None

        else:
            ErrorHandler.handle_error(
                code=301,
                details=f"File does not exist: {file_path}",
                parent=parent,
                fatal=False
            )
            return None, None

        return loaded_data, file_content_html

    # ---------------------------------------------
    # DATA VALIDATION
    # ---------------------------------------------

    def parse_dimensions(self, dimensions_str: str, line_number: int) -> tuple[float, ...]:
        """
        Parses a string of dimensions into a tuple of floats.
        This method is used during file parsing to convert string representations
        of dimensions (e.g., "(1.0, 2.5, 0.0)") into numerical tuples.
        - `dimensions_str` (str): The string containing dimension values, typically
          enclosed in parentheses and comma-separated.
        - `line_number` (int): The line number in the input file where the dimension
          string was found, used for error reporting.
        - `tuple[float, ...]`: A tuple of floating-point numbers representing the parsed dimensions.
        - `Exception`: If any dimension value cannot be converted to a float.
        """
        dimensions_str = dimensions_str.strip().strip("()")

        if not dimensions_str:
            return ()

        dimensions = []

        for d in dimensions_str.split(","):

            try:
                dimensions.append(float(d.strip()))

            except ValueError:
                ErrorHandler.handle_error(
                    code=310, details=f"Line {line_number}: Invalid dimension value: {d.strip()}", fatal=False
                )
        return tuple(dimensions)

    def validate_data(self, data: dict) -> None:
        """
        Validates the parsed structural data for consistency and completeness.
        This function performs several checks to ensure the integrity of the
        FEA model data, including:
        - Verification that all nodes referenced by elements, displacements, and loads exist.
        - Detection of duplicate node or element IDs.
        - Confirmation that the number of displacement and load components matches
        the expected degrees of freedom for the defined structure type.
        - Validation of cross-section and material existence for elements.
        - `data` (dict): The dictionary containing the parsed structural data.
        - `Exception`: If any inconsistencies or errors are found in the data.
        """
        nodes = data["nodes"]
        elements = data["elements"]
        nodal_displacements = data.get("nodal_displacements", {})
        nodal_loads = data.get("nodal_loads", {})
        element_type = data["structure_info"]["element_type"]
        node_ids = set(nodes.keys())
        element_ids = set(elements.keys())

        if len(node_ids) != len(nodes):
            ErrorHandler.handle_error(
                code=112,
                details="Duplicate node IDs found in the model data",
                fatal=True,
                exception_type=ValueError
            )

        if len(element_ids) != len(elements):
            ErrorHandler.handle_error(
                code=113,
                details="Duplicate element IDs found in the model data",
                fatal=True,
                exception_type=ValueError
            )

        for element_id, element_data in elements.items():

            if element_data["node1"] not in nodes:
                ErrorHandler.handle_error(
                    code=111,
                    details=f"Element {element_id} references non-existent node {element_data['node1']}",
                    fatal=True,
                    exception_type=ValueError
                )

            if element_data["node2"] not in nodes:
                ErrorHandler.handle_error(
                    code=111,
                    details=f"Element {element_id} references non-existent node {element_data['node2']}",
                    fatal=True,
                    exception_type=ValueError
                )

            if element_data["section_code"] not in data["cross_sections"] and element_data["section_code"] != "DEFAULT":
                ErrorHandler.handle_error(
                    code=115,
                    details=f"Element {element_id} references undefined cross-section {element_data['section_code']}",
                    fatal=True,
                    exception_type=ValueError
                )

            if element_data["material_code"] not in data["materials"] and element_data["material_code"] != "DEFAULT":
                ErrorHandler.handle_error(
                    code=114,
                    details=f"Element {element_id} references undefined material {element_data['material_code']}",
                    fatal=True,
                    exception_type=ValueError
                )

        for node_id in nodal_displacements:

            if node_id not in nodes:
                ErrorHandler.handle_error(
                    code=220,
                    details=f"Nodal displacement condition references non-existent node {node_id}",
                    fatal=True,
                    exception_type=ValueError
                )

        for node_id in nodal_loads:

            if node_id not in nodes:
                ErrorHandler.handle_error(
                    code=230,
                    details=f"Nodal load condition references non-existent node {node_id}",
                    fatal=True,
                    exception_type=ValueError
                )
        expected_dof = data["structure_info"]["dofs_per_node"] 

        for node_id, displacements in nodal_displacements.items():

            if len(displacements) != expected_dof:
                ErrorHandler.handle_error(
                    code=220,
                    details=f"Incorrect number of displacement components for node {node_id}. Expected {expected_dof}, got {len(displacements)}",
                    fatal=True,
                    exception_type=ValueError
                )

        for node_id, loads in nodal_loads.items():

            if len(loads) != expected_dof:
                ErrorHandler.handle_error(
                    code=230,
                    details=f"Incorrect number of load components for node {node_id}. Expected {expected_dof}, got {len(loads)}",
                    fatal=True,
                    exception_type=ValueError
                )

        if element_type is None:
            ErrorHandler.handle_error(
                code=200,
                details="Structure type is not defined in the input file",
                fatal=True,
                exception_type=ValueError
            )

    # ---------------------------------------------
    # DATA INITIALIZATION & CONVERSION
    # ---------------------------------------------

    @staticmethod
    def initialize_default_data() -> dict:
        """
        Initializes a dictionary with default structural data, including
        default units, element properties, cross-sections, and materials.
        It attempts to load these defaults from JSON files if they exist,
        otherwise it falls back to hardcoded default values.
        - `dict`: A dictionary populated with default structural model data.
        """
        def load_json_with_default(filepath: str, default_value: dict) -> dict:
            """
            Loads a JSON file or returns a default value if the file is not found
            or an error occurs during reading.
            - `filepath` (str): The path to the JSON file.
            - `default_value` (dict): The default dictionary to return if loading fails.
            - `dict`: The loaded JSON data or the `default_value`.
            """

            if os.path.exists(filepath):

                try:
                    with open(filepath, 'r') as f:
                        return json.load(f)

                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    ErrorHandler.handle_error(
                        code=300, 
                        details=e,
                        fatal=True,
                        exception_type=ValueError
                    )
            return default_value

        FileIO.valid_units = {
            "Modulus (E,G)": ["GPa", "MPa", "ksi"],
            "Moment of Inertia (Iy,Iz,J)": ["m⁴", "cm⁴", "mm⁴", "in⁴"],
            "Cross-Sectional Area (A)": ["m²", "cm²", "mm²", "in²"],
            "Volume (V)": ["m³", "cm³", "mm³", "in³"],
            "Force (Fx,Fy,Fz)": ["kN", "N", "lbf", "kip"],
            "Displacement (Dx,Dy,Dz)": ["m", "cm", "mm", "in"],
            "Position (X,Y,Z)": ["m", "cm", "mm", "in"],
            "Mass": ["kg", "g", "lb"],
            "Density (ρ)": ["kg/m³", "g/cm³", "lb/in³"],
            "Force/Length (F/L)": ["kN/m", "N/m", "lbf/in"]
        }   
        default_units = {
            "Modulus (E,G)": "GPa",
            "Moment of Inertia (Iy,Iz,J)": "m⁴",
            "Cross-Sectional Area (A)": "m²",
            "Force (Fx,Fy,Fz)": "N",
            "Force/Length (F/L)": "N/m",
            "Displacement (Dx,Dy,Dz)": "m",
            "Position (X,Y,Z)": "m",
            "Density (ρ)": "kg/m³"
        }
        default_element = {
            "node1": 0,
            "node2": 0,
            "section_code": "DEFAULT",
            "material_code": "DEFAULT",
            "E": 210.0,
            "v": 0.3,
            "G": 76.92,
            "J": 1.0,
            "Iy": 1.0,
            "Iz": 1.0,
            "A": 1.0,
            "length": 1.0,
            "angle": 0.0
        }
        default_cross_section = {
            "DEFAULT": {
                "type": "Solid_Rectangular",
                "dimensions": {
                    "B": 0.1,
                    "H": 0.1,
                    "angle": 0.0
                }
            }
        }
        default_material = {
            "type": "Steel",
            "properties": {
                "E": 210.0,
                "v": 0.3,
                "density": 7200.0
            }
        }
        script_dir = os.path.dirname(__file__)
        unit_file = os.path.join(script_dir, "default_units.json")
        element_file = os.path.join(script_dir, "default_element.json")
        cs_file = os.path.join(script_dir, "default_cross_section.json")
        mat_file = os.path.join(script_dir, "default_material.json")
        units = load_json_with_default(unit_file, default_units)
        element = load_json_with_default(element_file, default_element)
        cross_sections = load_json_with_default(cs_file, default_cross_section)
        material = load_json_with_default(mat_file, default_material)
        data: dict = {
            "structure_info": {},
            "cross_sections": cross_sections,
            "materials": {"DEFAULT": material},
            "elements": {0: {
                'node1': element["node1"],
                'node2': element["node2"],
                'section_code': element["section_code"],
                'material_code': element["material_code"]
            }},
            "nodes": {},
            "nodal_displacements": {},
            "concentrated_loads": {},
            "distributed_loads": {},
            "units": units,
            "saved_units": units
        }
        return data

    def parse_tuple_input(dofs_per_node: int, text_value: str, expected_length: int = None) -> tuple[float, ...]:
        """
        Parses a string input representing a tuple of force or displacement
        components (e.g., "(1.0, 2.5, nan)") into a tuple of floats or `np.nan`.
        It ensures the resulting tuple has the correct number of degrees of freedom (DOFs).
        - `dofs_per_node` (int): The number of degrees of freedom per node for the current structure.
        - `text_value` (str): The string to be parsed (e.g., "(100.0, nan, 0.0)" or "nan").
        - `expected_length` (int, optional): The expected length of the tuple. If `None`,
          `dofs_per_node` is used. Defaults to `None`.
        - `tuple[float, ...]`: A tuple of floats or `np.nan` values, with the length
          adjusted to `expected_length`.
        """

        if expected_length is None:
            expected_length = dofs_per_node

        try:

            if text_value.lower() == 'nan':
                return tuple(float('nan') for _ in range(expected_length))

            tuple_parts = text_value.strip('()').split(',')
            parsed_tuple = [float(val.strip()) if val.strip().lower() != 'nan' else float('nan') for val in tuple_parts]

            if len(parsed_tuple) < expected_length:
                parsed_tuple.extend([float('nan')] * (expected_length - len(parsed_tuple)))
            elif len(parsed_tuple) > expected_length:
                parsed_tuple = parsed_tuple[:expected_length]
            return tuple(parsed_tuple)

        except (ValueError, SyntaxError) as e:
            ErrorHandler.handle_error(
                code=201,
                details=f"Failed to parse tuple input '{text_value}': {e}",
                fatal=False
            )
            return tuple(float('nan') for _ in range(expected_length))

    # ---------------------------------------------
    # EXPORT FUNCTIONS
    # ---------------------------------------------

    @staticmethod
    def export_to_excel(data: dict, output_path: str) -> bool:
        """
        Exports various structural data (nodes, elements, displacements,
        concentrated loads, distributed loads) to different sheets within
        a single Excel file.
        - `data` (dict): The dictionary containing the structural data.
        - `output_path` (str): The full path including filename for the Excel output.
        - `bool`: `True` if the export was successful, `False` otherwise.
        """

        try:
            with pd.ExcelWriter(output_path) as writer:

                if "nodes" in data and data["nodes"]:
                    pd.DataFrame(data["nodes"]).T.to_excel(writer, sheet_name="Nodes", index=True, index_label="Node ID")

                if "elements" in data and data["elements"]:
                    pd.DataFrame(data["elements"]).T.to_excel(writer, sheet_name="Elements", index=True, index_label="Element ID")

                if "nodal_displacements" in data and data["nodal_displacements"]:
                    pd.DataFrame.from_dict(data["nodal_displacements"], orient='index').to_excel(writer, sheet_name="Nodal_Displacements", index=True, header=False, index_label="Node ID")

                if "concentrated_loads" in data and data["concentrated_loads"]:
                    pd.DataFrame.from_dict(data["concentrated_loads"], orient='index').to_excel(writer, sheet_name="Concentrated_Loads", index=True, header=False, index_label="Node ID")

                if "distributed_loads" in data and data["distributed_loads"]:
                    pd.DataFrame(data["distributed_loads"]).T.to_excel(writer, sheet_name="Distributed_Loads", index=True, index_label="Element ID")
            return True

        except Exception as e:
            ErrorHandler.handle_error(
                code=302,
                details=f"Error exporting to Excel: {output_path} - {e}",
                fatal=False
            )
            return False

    @staticmethod
    def generate_input_file(node_data: dict, element_data: dict) -> str:
        """
        Generates a basic input file string based on provided node and element data.
        This function is intended for internal representation or simple exports.
        - `node_data` (dict): A dictionary of node information.
        - `element_data` (dict): A dictionary of element information.
        - `str`: A formatted string representing the input file content.
        """
        input_file = "NODE_DATA_START\n"

        for node_id, data in node_data.items():
            force = data.get('force', [np.nan, np.nan, np.nan])
            displacement = data.get('displacement', [np.nan, np.nan, np.nan])
            force_str_parts = []

            for f in force:

                if isinstance(f, (int, float)) and math.isnan(f):
                    force_str_parts.append("nan")
                else:
                    force_str_parts.append(str(f))
            force_str = f"({','.join(force_str_parts)})"
            displacement_str_parts = []

            for d in displacement:

                if isinstance(d, (int, float)) and math.isnan(d):
                    displacement_str_parts.append("nan")
                else:
                    displacement_str_parts.append(str(d))
            displacement_str = f"({','.join(displacement_str_parts)})"
            z_coord = data.get('Z')
            node_coords = f"({data['X']},{data['Y']},{z_coord})" if z_coord is not None else f"({data['X']},{data['Y']})"
            input_file += f"{node_id} {node_coords} {force_str} {displacement_str}\n"
        input_file += "NODE_DATA_END\n\nELEMENT_DATA_START\n"

        for element_id, data in element_data.items():
            E_val = float(data.get('E', 0.0))
            A_val = float(data.get('A', 0.0))
            I_val = float(data.get('I', 0.0))
            input_file += f"{element_id} ({data['node1']},{data['node2']},{data.get('section_code', 'DEFAULT')},{data.get('material_code', 'DEFAULT')}) {E_val} {A_val} {I_val}\n"
        input_file += "ELEMENT_DATA_END"
        return input_file

    # ---------------------------------------------
    # JSON FILE OPERATIONS
    # ---------------------------------------------

    @staticmethod
    def read_json(file_path: str) -> dict | None:
        """
        Reads a JSON file from the specified path and returns its content as a dictionary.
        - `file_path` (str): The full path to the JSON file.
        - `dict | None`: The content of the JSON file as a dictionary if successful,
          otherwise `None` if the file is not found or has invalid JSON format.
        """

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)

        except FileNotFoundError:
            ErrorHandler.handle_error(
                code=301,
                details=f"File not found: {file_path}",
                fatal=False
            )
            return None

        except json.JSONDecodeError:
            ErrorHandler.handle_error(
                code=303,
                details=f"Invalid JSON in file: {file_path}",
                fatal=False
            )
            return None

    @staticmethod
    def write_json(file_path: str, data: dict) -> None:
        """
        Writes a dictionary to a JSON file at the specified path.
        - `file_path` (str): The full path to the JSON file to be created or overwritten.
        - `data` (dict): The dictionary content to be written to the file.
        """

        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)

        except Exception as e:
            ErrorHandler.handle_error(
                code=302,
                details=f"Error writing JSON file: {file_path} - {e}",
                fatal=False
            )

    @staticmethod
    def read_from_json(filename: str) -> dict:
        """
        Reads a JSON file and converts any `None` values within the loaded data
        to `np.nan`.
        - `filename` (str): The path to the JSON file.
        - `dict`: The loaded JSON data with `None` values replaced by `np.nan`.
        """

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return FileIO.convert_none_to_nan(data)

        except FileNotFoundError:
            ErrorHandler.handle_error(
                code=301,
                details=f"File not found: {filename}",
                fatal=False
            )
            return {}

        except json.JSONDecodeError:
            ErrorHandler.handle_error(
                code=303,
                details=f"Invalid JSON in file: {filename}",
                fatal=False
            )
            return {}

        except Exception as e:
            ErrorHandler.handle_error(
                code=300,
                details=f"Unexpected error reading JSON file: {filename} - {e}",
                fatal=False
            )
            return {}

    @staticmethod
    def convert_none_to_nan(obj: any) -> any:
        """
        Recursively converts `None` values in a given object (dictionary, list, or scalar)
        to `np.nan`.
        - `obj` (any): The object (dict, list, or scalar) to process.
        - `any`: The object with all `None` values replaced by `np.nan`.
        """

        if obj is None:
            return np.nan

        elif isinstance(obj, list):
            return [FileIO.convert_none_to_nan(item) for item in obj]

        elif isinstance(obj, dict):
            return {key: FileIO.convert_none_to_nan(value) for key, value in obj.items()}

        else:
            return obj

    # ---------------------------------------------
    # FILE UTILITIES
    # ---------------------------------------------

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        Checks if a file exists at the specified path.
        - `file_path` (str): The full path to the file.
        - `bool`: `True` if the file exists, `False` otherwise.
        """
        return os.path.exists(file_path)

    # ---------------------------------------------
    # UI MESSAGE BOXES
    # ---------------------------------------------

    @staticmethod
    def warning_message(parent: QWidget = None, title: str = "Warning", text: str = "Are you sure?", informative_text: str = "", detailed_text: str = "") -> bool:
        """
        Displays a warning message box with customizable text and OK/Cancel buttons.
        - `parent` (`QWidget`, optional): The parent widget for the message box. Defaults to `None`.
        - `title` (str, optional): The title of the message box window. Defaults to "Warning".
        - `text` (str, optional): The primary message displayed. Defaults to "Are you sure?".
        - `informative_text` (str, optional): Additional, more detailed information displayed
          below the primary text. Defaults to "".
        - `detailed_text` (str, optional): Extensive details available by clicking a "Show Details..."
          button. Defaults to "".
        - `bool`: Bool representing the button clicked by the user (`QMessageBox.Ok`=True or `QMessageBox.Cancel`=False).
        """
        message_box = QMessageBox(parent)
        message_box.setIcon(QMessageBox.Icon.Warning)
        message_box.setWindowTitle(title)
        message_box.setText(text)
        message_box.setInformativeText(informative_text)

        if detailed_text:
            message_box.setDetailedText(detailed_text)
            dialog = message_box.findChild(QDialog)

            if dialog:
                dialog.setDetailedTextVisible(False)
        message_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        message_box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        response = message_box.exec()

        if response == QMessageBox.StandardButton.Ok:
            result = True
        else:
            result = False
        return result

    @staticmethod
    def save_structure_to_file(imported_data: dict, filename: str = "input_file.txt") -> None:
        """
        Saves the complete structural data from the `imported_data` dictionary
        into a structured text file (e.g., `.txt` or `.fea` format).
        This function writes sections for structure type, cross-sections, materials,
        element data, node coordinates, nodal displacements, concentrated loads,
        distributed loads, and units. It formats the data appropriately,
        including handling `NaN` values and specific dimension requirements for
        cross-sections.
        - `imported_data` (dict): A dictionary containing all the structural data
          (nodes, elements, cross-sections, materials, loads, boundary conditions, units).
        - `filename` (str, optional): The name of the file to save the data to.
          Defaults to "input_file.txt".
        """

        # ---------------------------------------------
        # HELPER FUNCTIONS FOR FORMATTING
        # ---------------------------------------------

        def format_tuple(tup: tuple) -> str:
            """
            Formats a tuple of values into a comma-separated string,
            converting `np.nan` values to the string "nan".
            - `tup` (tuple): The input tuple to format.
            - `str`: The formatted string representation of the tuple.
            """
            return ", ".join("nan" if isinstance(v, float) and math.isnan(v) else str(v) for v in tup)

        def get_required_dims(section_type: str, dims: dict) -> tuple:
            """
            Extracts the required dimensions for a given cross-section type
            from a dictionary of all dimensions.
            - `section_type` (str): The type of the cross-section (e.g., "Solid_Circular").
            - `dims` (dict): A dictionary containing all dimensions for the cross-section,
              with dimension names as keys.
            - `tuple`: A tuple containing the dimension values in the order
              required for the specified `section_type`.
            """
            keys_by_type = {
                "Solid_Circular": ["D"],
                "Hollow_Circular": ["D", "d"],
                "Solid_Rectangular": ["B", "H", "angle"],
                "Hollow_Rectangular": ["B", "H", "b", "h", "angle"],
                "I_Beam": ["B", "H", "tw", "tf", "angle"],
                "C_Beam": ["B", "H", "tw", "tf", "angle"],
                "L_Beam": ["B", "H", "tw", "tf", "angle"],
            }
            return tuple(dims.get(k, 0.0) for k in keys_by_type.get(section_type, []))

        # ---------------------------------------------
        # WRITING TO FILE
        # ---------------------------------------------

        try:
            with open(filename, "w", encoding='utf-8') as f:

                # ---------------------------------------------
                # STRUCTURE TYPE SECTION
                # ---------------------------------------------

                element_type = imported_data['structure_info']['element_type']
                f.write(f"STRUCTURE_TYPE: {element_type}\n")
                f.write("# Accepted type 2D_Beam, 3D_Frame, 2D_Truss, 3D_Truss, 2D_Plane, 3D_Solid\n\n")

                # ---------------------------------------------
                # CROSS SECTIONS SECTION
                # ---------------------------------------------

                f.write("CROSS_SECTION_START\n")
                f.write("# Code, Type, Required Dimensions\n")
                f.write("# Supported types: Solid_Circular, Hollow_Circular, Solid_Rectangular, Hollow_Rectangular, I_Beam, C_Beam, L_Beam\n")
                f.write("# Dimensions depend on type:\n")
                f.write("# - Solid_Circular: (D)\n")
                f.write("# - Hollow_Circular: (D, d)\n")
                f.write("# - Solid_Rectangular: (B, H, angle)\n")
                f.write("# - Hollow_Rectangular: (Bo, Ho, Bi, Hi, angle)\n")
                f.write("# - I_Beam: (B, H, tw, tf, angle)\n")
                f.write("# - C_Beam: (B, H, tw, tf, angle)\n")
                f.write("# - L_Beam: (B, H, tw, tf, angle)\n")

                for code, section in imported_data['cross_sections'].items():

                    if code == "DEFAULT":
                        continue
                    section_type = section["type"]
                    dims = get_required_dims(section_type, section["dimensions"])
                    f.write(f"{code} {section_type} ({', '.join(map(str, dims))})\n")
                f.write("CROSS_SECTION_END\n\n")

                # ---------------------------------------------
                # MATERIALS SECTION
                # ---------------------------------------------

                f.write("MATERIAL_START\n")
                f.write("# Code Name Required Properties (E, nu, density)\n")

                for code, mat in imported_data['materials'].items():

                    if code == "DEFAULT":
                        continue
                    props = mat["properties"]
                    f.write(f"{code} {mat['type']} ({props['E']}, {props['v']}, {props['density']})\n")
                f.write("MATERIAL_END\n\n")

                # ---------------------------------------------
                # ELEMENT DATA SECTION
                # ---------------------------------------------

                f.write("ELEMENT_DATA_START\n")
                f.write("# Element_number, Start_node, End_node, Cross-section_code, Material_code\n")

                for eid, e in imported_data['elements'].items():

                    if eid == 0:
                        continue
                    f.write(f"{eid} ({e['node1']}, {e['node2']}, {e['section_code']}, {e['material_code']})\n")
                f.write("ELEMENT_DATA_END\n\n")

                # ---------------------------------------------
                # NODE COORDINATES SECTION
                # ---------------------------------------------

                f.write("NODE_DATA_START\n")
                f.write("# Node_number, X, Y, Z\n" if imported_data['structure_info']['dimension'] == '3D' else "# Node_number, X, Y\n")

                for nid, n in imported_data['nodes'].items():
                    coords = [n["X"], n["Y"]]

                    if imported_data['structure_info']['dimension'] == '3D':
                        coords.append(n["Z"])
                    f.write(f"{nid} ({', '.join(map(str, coords))})\n")
                f.write("NODE_DATA_END\n\n")

                # ---------------------------------------------
                # NODAL DISPLACEMENTS SECTION
                # ---------------------------------------------

                if imported_data.get("nodal_displacements"):
                    f.write("NODES_DISPLACEMENT_START\n")
                    f.write("# Node_number, Displacement DOFs\n")

                    for nid, disp in imported_data["nodal_displacements"].items():
                        f.write(f"{nid} ({format_tuple(disp)})\n")
                    f.write("NODES_DISPLACEMENT_END\n\n")

                # ---------------------------------------------
                # NODAL CONCENTRATED LOADS SECTION
                # ---------------------------------------------

                if imported_data.get("concentrated_loads"):
                    f.write("NODES_CONCENTRATED_LOADS_START\n")
                    f.write("# Node_number, Force DOFs\n")

                    for nid, force in imported_data["concentrated_loads"].items():
                        f.write(f"{nid} ({format_tuple(force)})\n")
                    f.write("NODES_CONCENTRATED_LOADS_END\n\n")

                # ---------------------------------------------
                # ELEMENTS DISTRIBUTED LOADS SECTION
                # ---------------------------------------------

                if imported_data.get("distributed_loads"):
                    f.write("ELEMENTS_DISTRIBUTED_LOADS_START\n")
                    f.write("# Element_number, Load Type, Direction, Parameters\n")

                    for eid, load in imported_data["distributed_loads"].items():
                        params = load['parameters']

                        if load['type'] == 'Equation' and isinstance(params, str):
                            param_str = f'"{params}"'
                        elif isinstance(params, (list, tuple)):
                            param_str = ', '.join(map(str, params))
                        else:
                            param_str = str(params)
                        f.write(f"{eid} {load['type']} {load['direction']} ({param_str})\n")
                    f.write("ELEMENTS_DISTRIBUTED_LOADS_END\n\n")

                # ---------------------------------------------
                # UNITS DATA SECTION
                # ---------------------------------------------

                f.write("UNITS_DATA_START\n")
                f.write("# Quantity, Unit\n")

                for key, val in imported_data.get("saved_units", {}).items():
                    f.write(f"{key}: {val}\n")
                f.write("UNITS_DATA_END\n")

        except Exception as e:
            ErrorHandler.handle_error(
                code=302,
                details=f"Error saving structure to file: {filename} - {e}",
                fatal=True
            )