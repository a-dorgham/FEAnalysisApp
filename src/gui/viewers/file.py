import json
from datetime import datetime
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from io import BytesIO
import base64



class FileViewer:
    """
    The `FileViewer` class is responsible for generating comprehensive HTML reports
    from imported FEA model data. These reports are designed for display in a
    PyQt6 WebEngineView and provide a structured and visually enhanced overview
    of the model's components, including structure information, material properties,
    cross-section details with visualizations, node coordinates, element connectivity,
    and applied loads.
    """


    def __init__(self):
        """
        Initializes the FileViewer.
        """
        super().__init__()

    # ---------------------------------------------
    # REPORT GENERATION
    # ---------------------------------------------


    def generate_input_file_report(self, imported_data: Dict[str, Any], theme: str = "light") -> str:
        """
        Generates a professional HTML report of the `imported_data` for display
        in a PyQt6 WebEngineView. This report covers various aspects of the FEA model.
        Args:
            imported_data (Dict[str, Any]): A dictionary containing FEA model data,
                                            including keys like 'structure_info', 'materials',
                                            'cross_sections', 'nodes', 'elements', 'concentrated_loads',
                                            'distributed_loads', and 'units'.
            theme (str): The theme for the report - "light" or "dark". Defaults to "light".
        Returns:
            str: An HTML string representing the complete report, ready for display.
        """
        html_content: str = ""
        structure_info: Dict[str, Any] = imported_data.get('structure_info', {})

        if structure_info:
            structure_content: str = self.create_table_from_dict(structure_info, "Structure Information")
            html_content += self.create_section("1. Structure Information", structure_content)
        materials: Dict[str, Any] = imported_data.get('materials', {})

        if materials:
            materials_table: str = self.create_materials_table(materials, imported_data.get('units', {}))
            html_content += self.create_section("2. Material Properties", materials_table)
        cross_sections: Dict[str, Any] = imported_data.get('cross_sections', {})

        if cross_sections:
            sections_content: str = self.create_cross_sections_table(cross_sections, imported_data.get('units', {}))
            html_content += self.create_section("3. Cross-Section Properties", sections_content)
        nodes: Dict[str, Any] = imported_data.get('nodes', {})
        nodes_table: str = self.create_nodes_table(nodes, imported_data)
        html_content += self.create_section("4. Nodes Information", nodes_table)
        elements: Dict[str, Any] = imported_data.get('elements', {})
        elements_table: str = self.create_elements_table(elements, imported_data)
        html_content += self.create_section("5. Elements Information", elements_table)
        loads_content: str = ""
        concentrated_loads: Dict[str, Any] = imported_data.get('concentrated_loads', {})

        if concentrated_loads:
            force_labels: List[str] = imported_data['structure_info'].get('force_labels', [])
            loads_content += f"""
            <h3 class="subsection-title">Concentrated Loads</h3>
            {self.create_concentrated_loads_table(concentrated_loads, force_labels, imported_data.get('units', {}))}
            """
        distributed_loads: Dict[str, Any] = imported_data.get('distributed_loads', {})

        if distributed_loads:
            loads_content += f"""
            <h3 class="subsection-title">Distributed Loads</h3>
            {self.create_distributed_loads_table(distributed_loads, imported_data.get('units', {}))}
            """

        if loads_content:
            html_content += self.create_section("6. Loads Information", loads_content)
        units: Dict[str, Any] = imported_data.get('units', {})

        if units:
            html_content += self.create_section("7. Units", self.create_table_from_dict(units))
        html: str = self.generate_final_html(structure_info, html_content, theme)
        return html

    # ---------------------------------------------
    # HELPER FUNCTIONS - HTML STRUCTURE & FORMATTING
    # ---------------------------------------------

    @staticmethod
    def format_value(val: Any) -> str:
        """
        Formats a given value for display in the HTML report.
        Handles `None` and `np.nan` values, and formats numerical values

        for better readability.
        Args:
            val (Any): The value to format.
        Returns:
            str: The formatted string representation of the value.
        """

        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"

        elif isinstance(val, (int, float)):
            return f"{val:.4g}" if abs(val) < 1e-3 or abs(val) > 1e6 else f"{val:.4g}"

        return str(val)

    def create_key_value(self, key: str, value: Any) -> str:
        """
        Creates an HTML string for a key-value pair, formatted

        for consistent display within the report.
        Args:
            key (str): The label for the key.
            value (Any): The value associated with the key.
        Returns:
            str: An HTML string representing the key-value pair.
        """
        return f"""

        <div class="key-value">
            <div class="key">{key}:</div>
            <div class="value">{self.format_value(value)}</div>
        </div>
        """

    @staticmethod
    def create_section(title: str, content: str) -> str:
        """
        Creates an HTML section block with a title and its content.
        Args:
            title (str): The title of the section.
            content (str): The HTML content to be placed within the section.
        Returns:
            str: An HTML string representing the section.
        """
        return f"""

        <div class="section">
            <h2 class="section-title">{title}</h2>
            {content}
        </div>
        """


    def create_table_from_dict(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """
        Generates an HTML table from a dictionary, handling nested dictionaries
        by creating nested table rows.
        Args:
            data (Dict[str, Any]): The dictionary from which to create the table.
            title (Optional[str]): An optional title for the table. If `None`, "Data" is used.
        Returns:
            str: An HTML string representing the table. Returns an empty string if `data` is empty.
        """

        if not data:
            return ""

        table: str = f"""
        <table>
            <thead>
                <tr>
                    <th colspan="2">{title if title else 'Data'}</th>
                </tr>
            </thead>
            <tbody>
        """

        for key, value in data.items():

            if isinstance(value, dict):
                table += f"""
                <tr>
                    <td>{key}</td>
                    <td>
                        {self.create_table_from_dict(value)}
                    </td>
                </tr>
                """
            else:
                table += f"""
                <tr>
                    <td>{key}</td>
                    <td>{self.format_value(value)}</td>
                </tr>
                """
        table += """
            </tbody>
        </table>
        """
        return table


    @staticmethod
    def get_css_style(font_scale: float = 0.8, theme: str = "light") -> str:
        """
        Returns a comprehensive CSS style string for the HTML report,
        with configurable font scaling and theme.
        Args:
            font_scale (float): A multiplier for all font sizes in the CSS. Defaults to 0.8.
            theme (str): The theme for the report - "light" or "dark". Defaults to "light".
        Returns:
            str: The CSS style definitions embedded within `<style>` tags.
        """
        if theme == "dark":
            css_vars = {
                'bg-primary': '#1a1a1a',
                'bg-secondary': '#2d2d2d',
                'bg-accent': '#3a3a3a',
                'text-primary': '#e0e0e0',
                'text-secondary': '#b0b0b0',
                'text-accent': '#ffffff',
                'border-color': '#444444',
                'primary-color': '#4a9cff',
                'secondary-color': '#6bb5ff',
                'accent-color': '#ff6b6b',
                'light-color': '#2d2d2d',
                'dark-color': '#e0e0e0',
                'table-header-bg': '#3a3a3a',
                'table-even-bg': '#2a2a2a',
                'table-hover-bg': '#3d3d3d',
                'notes-bg': '#2a2a2a',
                'json-bg': '#252525'
            }
        else:  # light theme
            css_vars = {
                'bg-primary': '#ffffff',
                'bg-secondary': '#f8f9fa',
                'bg-accent': '#e9ecef',
                'text-primary': '#333333',
                'text-secondary': '#555555',
                'text-accent': '#000000',
                'border-color': '#dee2e6',
                'primary-color': '#2c3e50',
                'secondary-color': '#3498db',
                'accent-color': '#e74c3c',
                'light-color': '#ecf0f1',
                'dark-color': '#34495e',
                'table-header-bg': '#2c3e50',
                'table-even-bg': '#f8f9fa',
                'table-hover-bg': '#f1f1f1',
                'notes-bg': '#f8f9fa',
                'json-bg': '#f5f5f5'
            }

        css: str = f"""
        <style>
            :root {{
                --bg-primary: {css_vars['bg-primary']};
                --bg-secondary: {css_vars['bg-secondary']};
                --bg-accent: {css_vars['bg-accent']};
                --text-primary: {css_vars['text-primary']};
                --text-secondary: {css_vars['text-secondary']};
                --text-accent: {css_vars['text-accent']};
                --border-color: {css_vars['border-color']};
                --primary-color: {css_vars['primary-color']};
                --secondary-color: {css_vars['secondary-color']};
                --accent-color: {css_vars['accent-color']};
                --light-color: {css_vars['light-color']};
                --dark-color: {css_vars['dark-color']};
                --table-header-bg: {css_vars['table-header-bg']};
                --table-even-bg: {css_vars['table-even-bg']};
                --table-hover-bg: {css_vars['table-hover-bg']};
                --notes-bg: {css_vars['notes-bg']};
                --json-bg: {css_vars['json-bg']};
                
                /* Font scaling variables */
                --font-scale: {font_scale};
                --font-body: calc(1rem * var(--font-scale));
                --font-small: calc(0.875rem * var(--font-scale));
                --font-medium: calc(1.125rem * var(--font-scale));
                --font-large: calc(1.25rem * var(--font-scale));
                --font-xlarge: calc(1.5rem * var(--font-scale));
                --font-xxlarge: calc(2rem * var(--font-scale));
                --font-title: calc(2.5rem * var(--font-scale));
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: var(--font-body);
                line-height: 1.6;
                color: var(--text-primary);
                background-color: var(--bg-primary);
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: var(--primary-color);
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                position: relative;
            }}
            .logo {{
                font-size: var(--font-xlarge);
                font-weight: bold;
            }}
            .report-title {{
                font-size: var(--font-title);
                text-align: center;
                margin: 0;
                color: var(--text-primary);
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 20px;
            }}
            .section-title {{
                font-size: var(--font-large);
                color: var(--secondary-color);
                border-bottom: 2px solid var(--secondary-color);
                padding-bottom: 5px;
                margin-bottom: 15px;
            }}
            .subsection-title {{
                font-size: var(--font-medium);
                color: var(--dark-color);
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                box-shadow: 0 2px 3px rgba(0,0,0,0.1);
                font-size: var(--font-small);
                background-color: var(--bg-primary);
            }}
            th {{
                background-color: var(--table-header-bg);
                color: white;
                padding: 10px;
                text-align: left;
                font-size: var(--font-body);
            }}
            td {{
                padding: 8px 10px;
                border-bottom: 1px solid var(--border-color);
                color: var(--text-primary);
            }}
            tr:nth-child(even) {{
                background-color: var(--table-even-bg);
            }}
            tr:hover {{
                background-color: var(--table-hover-bg);
            }}
            .plot-container {{
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                margin: 20px 0;
            }}
            .key-value {{
                display: grid;
                grid-template-columns: 200px 1fr;
                gap: 10px;
                margin: 10px 0;
            }}
            .key {{
                font-weight: bold;
                color: var(--dark-color);
                font-size: var(--font-body);
            }}
            .value {{
                color: var(--text-secondary);
                font-size: var(--font-body);
            }}
            .notes {{
                background-color: var(--notes-bg);
                padding: 15px;
                border-left: 4px solid var(--secondary-color);
                margin: 20px 0;
                font-size: var(--font-body);
                color: var(--text-primary);
            }}
            .json-container {{
                background-color: var(--json-bg);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 15px;
                margin: 10px 0;
                font-family: monospace;
                white-space: pre-wrap;
                max-height: 300px;
                overflow-y: auto;
                font-size: var(--font-small);
                color: var(--text-primary);
            }}
            .plot {{
                margin: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-radius: 5px;
                overflow: hidden;
                max-width: 100%;
                background-color: var(--bg-secondary);
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: var(--text-secondary);
                font-size: var(--font-small);
                border-top: 1px solid var(--border-color);
            }}
            .export-btn {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: var(--secondary-color);
                color: white;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                font-size: var(--font-medium);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                z-index: 100;
            }}
            .export-btn:hover {{
                background: var(--accent-color);
                transform: scale(1.1);
            }}
            .pdf-loading-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                font-size: var(--font-large);
            }}
            .pdf-loading-content {{
                background: var(--bg-primary);
                padding: 30px;
                border-radius: 8px;
                text-align: center;
                max-width: 300px;
                color: var(--text-primary);
            }}
        </style>
        """
        return css

    def generate_final_html(self, structure_info: Dict[str, Any], body_content: str, theme: str = "light") -> str:
        report_date = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        css = self.get_css_style(theme=theme)
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FEA Model Data Report</title>
            {css}
        </head>
        <body>
            <div class="header">
                <div class="logo">FEAnalysisApp</div>
                <div>Finite Element Model Data Report</div>
                <div>{report_date}</div>
            </div>
            <div class="section">
                <h1 class="report-title">Finite Element Model Data</h1>
                <div style="text-align: center; margin-bottom: 30px;">
                    <span class="badge">
                        Element Type: {structure_info.get('element_type', 'Unknown')}
                    </span>
                </div>
            </div>
            {body_content}
            <div class="footer">
                <p>This report was automatically generated by FEAnalysisApp on {report_date}</p>
                <p>© {datetime.now().year} Finite Element Analysis App.</p>
            </div>
        </body>
        </html>
        """
        return html      

    # ---------------------------------------------
    # TABLE GENERATION
    # ---------------------------------------------


    def create_nodes_table(self, nodes: Dict[str, Any], imported_data: Dict[str, Any]) -> str:
        """
        Generates an HTML table displaying information about all nodes
        in the model, including coordinates, applied forces, and prescribed displacements.
        The columns adapt based on the model's dimensionality (2D vs. 3D).
        Args:
            nodes (Dict[str, Any]): A dictionary where keys are node IDs and values are
                                    dictionaries containing node properties (e.g., 'X', 'Y', 'Z',
                                    'force', 'displacement').
            imported_data (Dict[str, Any]): The full imported data dictionary, used to
                                           retrieve 'structure_info' and 'units'.
        Returns:
            str: An HTML string representing the nodes table. Returns an empty string if `nodes` is empty.
        """

        if not nodes:
            return ""

        disp_labels: List[str] = imported_data['structure_info'].get('displacement_labels', [])
        force_labels: List[str] = imported_data['structure_info'].get('force_labels', [])
        is_3d: bool = imported_data['structure_info'].get('dimension') == '3D'
        nodes_table: str = """
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>X</th>
                    <th>Y"""

        if is_3d:
            nodes_table += "<th>Z</th>"
        nodes_table += f"""
                    <th colspan="{len(force_labels)}">Forces</th>
                    <th colspan="{len(disp_labels)}">Displacements</th>
                </tr>
                <tr>
                    <th></th>
                    <th>{imported_data['units'].get('Position (X,Y,Z)', 'm')}</th>
                    <th>{imported_data['units'].get('Position (X,Y,Z)', 'm')}</th>"""

        if is_3d:
            nodes_table += f"<th>{imported_data['units'].get('Position (X,Y,Z)', 'm')}</th>"

        for label in force_labels:
            nodes_table += f"<th>{label} ({imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN')})</th>"

        for label in disp_labels:
            nodes_table += f"<th>{label} ({imported_data['units'].get('Displacement (Dx,Dy,Dz)', 'mm')})</th>"
        nodes_table += """
                </tr>
            </thead>
            <tbody>
        """

        for node_id, node_data in nodes.items():

            if node_id == 0:
                continue
            nodes_table += f"""
            <tr>
                <td>{node_id}</td>
                <td>{self.format_value(node_data.get('X'))}</td>
                <td>{self.format_value(node_data.get('Y'))}</td>"""

            if is_3d:
                nodes_table += f"<td>{self.format_value(node_data.get('Z'))}</td>"
            forces: Tuple[float, ...] = node_data.get('force', (np.nan,) * len(force_labels))

            for i in range(len(force_labels)):
                force_value: float = forces[i] if i < len(forces) else np.nan
                nodes_table += f"<td>{self.format_value(force_value)}</td>"
            displacements: Tuple[float, ...] = node_data.get('displacement', (np.nan,) * len(disp_labels))

            for i in range(len(disp_labels)):
                disp_value: float = displacements[i] if i < len(displacements) else np.nan
                nodes_table += f"<td>{self.format_value(disp_value)}</td>"
            nodes_table += "</tr>"
        nodes_table += """
            </tbody>
        </table>
        """
        return nodes_table

    def create_elements_table(self, elements: Dict[str, Any], imported_data: Dict[str, Any]) -> str:
        """
        Generates an HTML table displaying information about all elements
        in the model, including their connectivity, assigned section and material,
        length, and relevant properties (e.g., E, A, Iy, Iz, J). Columns adapt
        based on the element type (e.g., 2D_Beam, 3D_Frame).
        Args:
            elements (Dict[str, Any]): A dictionary where keys are element IDs and values are
                                       dictionaries containing element properties.
            imported_data (Dict[str, Any]): The full imported data dictionary, used to
                                           retrieve 'structure_info' and 'units'.
        Returns:
            str: An HTML string representing the elements table. Returns an empty string if `elements` is empty.
        """

        if not elements:
            return ""

        element_type: str = imported_data['structure_info'].get('element_type', '')
        is_beam_or_frame: bool = element_type in ['2D_Beam', '3D_Frame']
        elements_table: str = """
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Nodes</th>
                    <th>Section</th>
                    <th>Material</th>
                    <th>Length</th>
                    <th>E</th>
                    <th>Area</th>"""

        if is_beam_or_frame:
            elements_table += """
                        <th>Iy</th>
                        <th>Iz</th>
                        <th>J</th>"""
        elements_table += """
                </tr>
            </thead>
            <tbody>
        """

        for elem_id, elem_data in elements.items():

            if elem_id == 0:
                continue
            elements_table += f"""
            <tr>
                <td>{elem_id}</td>
                <td>{elem_data.get('node1', 'N/A')} → {elem_data.get('node2', 'N/A')}</td>
                <td>{elem_data.get('section_code', 'N/A')}</td>
                <td>{elem_data.get('material_code', 'N/A')}</td>
                <td>{self.format_value(elem_data.get('length'))} {imported_data['units'].get('Position (X,Y,Z)', 'm')}</td>
                <td>{self.format_value(elem_data.get('E'))} {imported_data['units'].get('Modulus (E,G)', 'GPa')}</td>
                <td>{self.format_value(elem_data.get('A'))} {imported_data['units'].get('Cross-Sectional Area (A)', 'cm²')}</td>"""

            if is_beam_or_frame:
                elements_table += f"""
                <td>{self.format_value(elem_data.get('Iy'))} {imported_data['units'].get('Moment of Inertia (Iy,Iz,J)', 'm⁴')}</td>
                <td>{self.format_value(elem_data.get('Iz'))} {imported_data['units'].get('Moment of Inertia (Iy,Iz,J)', 'm⁴')}</td>
                <td>{self.format_value(elem_data.get('J'))} {imported_data['units'].get('Moment of Inertia (Iy,Iz,J)', 'm⁴')}</td>"""
            elements_table += "</tr>"
        elements_table += """
            </tbody>
        </table>
        """
        return elements_table

    def create_materials_table(self, materials: Dict[str, Any], units: Dict[str, str]) -> str:
        """
        Creates a single comprehensive HTML table summarizing properties
        of all defined materials.
        Args:
            materials (Dict[str, Any]): A dictionary where keys are material names and
                                        values are dictionaries containing material properties.
            units (Dict[str, str]): A dictionary mapping property names to their respective units.
        Returns:
            str: An HTML string representing the materials table. Returns an empty string if `materials` is empty.
        """

        if not materials:
            return ""

        materials_table: str = """
        <table>
            <thead>
                <tr>
                    <th>Material</th>
                    <th>Type</th>
                    <th>E ({})</th>
                    <th>ν</th>
                    <th>G ({})</th>
                    <th>Density ({})</th>
                </tr>
            </thead>
            <tbody>
        """.format(
            units.get('Modulus (E,G)', 'GPa'),
            units.get('Modulus (E,G)', 'GPa'),
            units.get('Density (ρ)', 'kg/m³')
        )

        for mat_name, mat_data in materials.items():
            props: Dict[str, Any] = mat_data.get('properties', {})
            materials_table += f"""
            <tr>
                <td>{mat_name}</td>
                <td>{mat_data.get('type', 'N/A')}</td>
                <td>{self.format_value(props.get('E'))}</td>
                <td>{self.format_value(props.get('v'))}</td>
                <td>{self.format_value(props.get('G'))}</td>
                <td>{self.format_value(props.get('density'))}</td>
            </tr>
            """
        materials_table += """
            </tbody>
        </table>
        """
        return materials_table

    def create_cross_sections_table(self, cross_sections: Dict[str, Any], units: Dict[str, str]) -> str:
        """
        Creates a single comprehensive HTML table summarizing properties
        of all defined cross-sections. Each row includes basic properties and attempts
        to embed an SVG visualization of the cross-section.
        Args:
            cross_sections (Dict[str, Any]): A dictionary where keys are section names and
                                            values are dictionaries containing section properties.
            units (Dict[str, str]): A dictionary mapping property names to their respective units.
        Returns:
            str: An HTML string representing the cross-sections table, potentially with embedded SVGs.
                 Returns an empty string if `cross_sections` is empty.
        """

        if not cross_sections:
            return ""

        sections_table: str = """
        <table>
            <thead>
                <tr>
                    <th>Section</th>
                    <th>Type</th>
                    <th>Dimensions</th>
                    <th>Area ({})</th>
                    <th>Iy ({})</th>
                    <th>Iz ({})</th>
                    <th>J ({})</th>
                    <th>Visualization</th>
                </tr>
            </thead>
            <tbody>
        """.format(
            units.get('Cross-Sectional Area (A)', 'cm²'),
            units.get('Moment of Inertia (Iy,Iz,J)', 'm⁴'),
            units.get('Moment of Inertia (Iy,Iz,J)', 'm⁴'),
            units.get('Moment of Inertia (Iy,Iz,J)', 'm⁴')
        )

        for sec_name, sec_data in cross_sections.items():
            dims: Dict[str, Any] = sec_data.get('dimensions', {})
            dims_str: str = ", ".join(f"{k}={self.format_value(v)}" for k, v in dims.items())
            visualization_html: str = self.create_cross_section_visualization(sec_data.get('type', ''), dims)
            sections_table += f"""
            <tr>
                <td>{sec_name}</td>
                <td>{sec_data.get('type', 'N/A')}</td>
                <td>{dims_str}</td>
                <td>{self.format_value(sec_data.get('A'))}</td>
                <td>{self.format_value(sec_data.get('Iy'))}</td>
                <td>{self.format_value(sec_data.get('Iz'))}</td>
                <td>{self.format_value(sec_data.get('J'))}</td>
                <td>{visualization_html}</td>
            </tr>
            """
        sections_table += """
            </tbody>
        </table>
        """
        return sections_table

    def create_concentrated_loads_table(self, concentrated_loads: Dict[str, List[float]], force_labels: List[str], units: Dict[str, str]) -> str:
        """
        Creates an HTML table displaying concentrated loads applied
        to nodes, with columns for each force component based on `force_labels`.
        Args:
            concentrated_loads (Dict[str, List[float]]): A dictionary where keys are node IDs
                                                          and values are lists of force magnitudes.
            force_labels (List[str]): A list of strings representing the labels for each force component
                                      (e.g., ["Fx", "Fy", "Fz"]).
            units (Dict[str, str]): A dictionary mapping unit types to their symbols.
        Returns:
            str: An HTML string representing the concentrated loads table. Returns an empty string if `concentrated_loads` is empty.
        """

        if not concentrated_loads:
            return ""

        table: str = f"""
        <table>
            <thead>
                <tr>
                    <th>Node</th>
                    {''.join(f'<th>{label} ({units.get("Force (Fx,Fy,Fz)", "kN")})</th>' for label in force_labels)}
                </tr>
            </thead>
            <tbody>
        """

        for node_id, loads in concentrated_loads.items():
            table += f"<tr><td>{node_id}</td>"

            for i in range(len(force_labels)):
                load_value: float = loads[i] if i < len(loads) else np.nan
                table += f"<td>{self.format_value(load_value)}</td>"
            table += "</tr>"
        table += """
            </tbody>
        </table>
        """
        return table

    def create_distributed_loads_table(self, distributed_loads: Dict[str, Any], units: Dict[str, str]) -> str:
        """
        Creates an HTML table displaying distributed loads applied
        to elements, including load type, direction, parameters, and magnitude.
        Args:
            distributed_loads (Dict[str, Any]): A dictionary where keys are element IDs
                                                 and values are dictionaries containing
                                                 distributed load properties.
            units (Dict[str, str]): A dictionary mapping unit types to their symbols.
        Returns:
            str: An HTML string representing the distributed loads table. Returns an empty string if `distributed_loads` is empty.
        """

        if not distributed_loads:
            return ""

        table: str = """
        <table>
            <thead>
                <tr>
                    <th>Element</th>
                    <th>Type</th>
                    <th>Direction</th>
                    <th>Parameters</th>
                    <th>Magnitude ({})</th>
                </tr>
            </thead>
            <tbody>
        """.format(units.get("Force/Length (F/L)", "kN/m"))

        for elem_id, load_data in distributed_loads.items():
            params: Any = load_data.get('parameters', [])
            magnitude: str = ""
            load_type: str = load_data.get('type', 'N/A')

            if load_type == 'Uniform':
                magnitude = self.format_value(params[0] if isinstance(params, list) and len(params) > 0 else np.nan)
            elif load_type in ['Triangular', 'Trapezoidal']:

                if isinstance(params, list) and len(params) >= 2:
                    magnitude = f"{self.format_value(params[0])} → {self.format_value(params[1])}"
                else:
                    magnitude = "N/A"
            elif load_type == 'Equation':
                magnitude = params if isinstance(params, str) else str(params)
            params_display: str = json.dumps(params) if not isinstance(params, str) else params
            table += f"""
            <tr>
                <td>{elem_id}</td>
                <td>{load_type}</td>
                <td>{load_data.get('direction', 'N/A')}</td>
                <td>{params_display}</td>
                <td>{magnitude}</td>
            </tr>
            """
        table += """
            </tbody>
        </table>
        """
        return table

    # ---------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------

    @staticmethod
    def create_cross_section_visualization(section_type: str, dimensions: Dict[str, Any]) -> str:
        """
        Generates an SVG visualization of a cross-section based on its
        type and dimensions. This function requires `matplotlib` to be installed.
        If matplotlib is not available or an error occurs during generation, it
        returns a fallback HTML block.
        Args:
            section_type (str): The type of cross-section (e.g., 'Solid_Circular', 'I_Beam').
            dimensions (Dict[str, Any]): A dictionary containing the dimensions of the cross-section.
        Returns:
            str: An HTML string containing the SVG visualization embedded within a `<div>`,
                 or a fallback HTML string if visualization fails.
        """

        if plt is None:
            return f"""

            <div class="notes">
                <h4 class="subsection-title">{section_type} Section</h4>
                <div class="json-container">
                    Visualization not available (matplotlib not installed).<br>
                    {json.dumps(dimensions, indent=4)}
                </div>
            </div>
            """

        try:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.set_aspect('equal')
            ax.axis('off')
            angle: float = dimensions.get('angle', 0.0)

            if section_type == 'Solid_Circular':
                D: float = dimensions['D']
                ax.add_patch(Circle((0, 0), D / 2, fill=False, linewidth=1))
                ax.set_xlim(-D / 2 * 1.1, D / 2 * 1.1)
                ax.set_ylim(-D / 2 * 1.1, D / 2 * 1.1)
            elif section_type == 'Hollow_Circular':
                D_outer: float = dimensions['D']
                d_inner: float = dimensions['d']
                ax.add_patch(Circle((0, 0), D_outer / 2, fill=False, linewidth=1))
                ax.add_patch(Circle((0, 0), d_inner / 2, fill=False, linewidth=1))
                ax.set_xlim(-D_outer / 2 * 1.1, D_outer / 2 * 1.1)
                ax.set_ylim(-D_outer / 2 * 1.1, D_outer / 2 * 1.1)
            elif section_type == 'Solid_Rectangular':
                B_rect: float = dimensions['B']
                H_rect: float = dimensions['H']
                rect = Rectangle((-B_rect / 2, -H_rect / 2), B_rect, H_rect, angle=angle,
                                 fill=False, linewidth=1)
                ax.add_patch(rect)
                ax.set_xlim(-B_rect * 0.6, B_rect * 0.6)
                ax.set_ylim(-H_rect * 0.6, H_rect * 0.6)
            elif section_type == 'Hollow_Rectangular':
                Bo_outer: float = dimensions['B']
                Ho_outer: float = dimensions['H']
                Bi_inner: float = dimensions.get('b', 0.0)
                Hi_inner: float = dimensions.get('h', 0.0)
                outer = Rectangle((-Bo_outer / 2, -Ho_outer / 2), Bo_outer, Ho_outer, angle=angle,
                                  fill=False, linewidth=1)
                inner = Rectangle((-Bi_inner / 2, -Hi_inner / 2), Bi_inner, Hi_inner, angle=angle,
                                  fill=False, linewidth=1)
                ax.add_patch(outer)
                ax.add_patch(inner)
                ax.set_xlim(-Bo_outer * 0.6, Bo_outer * 0.6)
                ax.set_ylim(-Ho_outer * 0.6, Ho_outer * 0.6)
            elif section_type in ['I_Beam', 'C_Beam', 'L_Beam']:
                B: float = dimensions['B']
                H: float = dimensions['H']
                tw: float = dimensions['tw']
                tf: float = dimensions['tf']
                angle_rad = np.radians(angle)


                def rotate(coords):

                    if angle == 0:
                        return coords

                    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                        [np.sin(angle_rad), np.cos(angle_rad)]])
                    return [tuple(rot_matrix @ np.array(p)) for p in coords]

                patches = []

                if section_type == 'I_Beam':
                    tf_top = [(-B/2, H/2), (B/2, H/2), (B/2, H/2 - tf), (-B/2, H/2 - tf)]
                    patches.append(tf_top)
                    web = [(-tw/2, H/2 - tf), (tw/2, H/2 - tf), (tw/2, -H/2 + tf), (-tw/2, -H/2 + tf)]
                    patches.append(web)
                    tf_bot = [(-B/2, -H/2 + tf), (B/2, -H/2 + tf), (B/2, -H/2), (-B/2, -H/2)]
                    patches.append(tf_bot)
                elif section_type == 'C_Beam':
                    web = [(-tw/2, H/2), (tw/2, H/2), (tw/2, -H/2), (-tw/2, -H/2)]
                    patches.append(web)
                    top = [(tw/2, H/2), (B, H/2), (B, H/2 - tf), (tw/2, H/2 - tf)]
                    patches.append(top)
                    bot = [(tw/2, -H/2), (B, -H/2), (B, -H/2 + tf), (tw/2, -H/2 + tf)]
                    patches.append(bot)
                elif section_type == 'L_Beam':
                    l_path = [
                        (-tw/2, H/2), (tw/2, H/2), (tw/2, -H/2 + tf),
                        (B, -H/2 + tf), (B, -H/2),
                        (-tw/2, -H/2), (-tw/2, H/2)
                    ]
                    patches = [l_path]

                for coords in patches:
                    rotated = rotate(coords)
                    ax.add_patch(Polygon(rotated, closed=True, fill=False, linewidth=1))
                all_points = np.vstack([rotate(patch) for patch in patches])
                x_vals, y_vals = all_points[:, 0], all_points[:, 1]
                x_center, y_center = np.mean(x_vals), np.mean(y_vals)
                x_span = max(x_vals) - min(x_vals)
                y_span = max(y_vals) - min(y_vals)
                margin = 0.2
                ax.set_xlim(x_center - x_span * (0.5 + margin), x_center + x_span * (0.5 + margin))
                ax.set_ylim(y_center - y_span * (0.5 + margin), y_center + y_span * (0.5 + margin))
            buffer = BytesIO()
            plt.savefig(buffer, format='svg', bbox_inches='tight', pad_inches=0.5)
            plt.close(fig)
            svg: str = buffer.getvalue().decode('utf-8')
            svg_content: str = svg[svg.find('<svg'):]
            return f"""

            <div class="plot-container">
                <div class="plot" style="width: 100px; height: 100px; display: flex; align-items: center; justify-content: center;">
                    {svg_content}
                </div>
            </div>
            """

        except Exception as e:
            print(f"Error generating cross-section visualization for {section_type}: {e}")
            return f"""

            <div class="notes">
                <div class="json-container">
                    <p><i>Visualization Failed</i></p>
                </div>
            </div>
            """