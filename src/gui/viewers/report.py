import base64
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from typing import Dict, Any, List, Union, Tuple, Optional


class ReportViewer(QMainWindow):
    """
    The `ReportViewer` class generates and displays a comprehensive HTML report
    of Finite Element Analysis (FEA) solution results. This report includes
    summary information, nodal displacements and forces, element internal forces,
    and detailed model properties (materials and cross-sections).
    It leverages Matplotlib for generating plots and renders the report
    within a PyQt6 `QWebEngineView`.
    """


    def __init__(self, solution_data: Dict[str, Any], imported_data: Dict[str, Any], theme: str = "light"):
        """
        Initializes the ReportViewer.
        Args:
            solution_data (Dict[str, Any]): A dictionary containing the results from the FEA solver,
                                            including displacements, forces, element forces, and solver info.
            imported_data (Dict[str, Any]): A dictionary containing the original imported FEA model data,
                                            such as nodes, elements, materials, cross-sections, and structure info.
            theme (str): The theme for the report. Can be "light" or "dark". Defaults to "light".
        """
        super().__init__()
        self.solution_data: Dict[str, Any] = solution_data
        self.imported_data: Dict[str, Any] = imported_data.copy()
        self.theme: str = theme.lower()

        if 0 in self.imported_data.get('nodes', {}):
            del self.imported_data['nodes'][0]

        if 0 in self.imported_data.get('elements', {}):
            del self.imported_data['elements'][0]
        self.structure_info: Dict[str, Any] = imported_data.get('structure_info', {})

    # ---------------------------------------------
    # REPORT GENERATION
    # ---------------------------------------------


    def generate_fe_report(self) -> str:
        """
        Generates the complete HTML content for the FEA solution report.
        This method orchestrates the creation of plots, data tables, and
        integrates them into a single, styled HTML document.
        Returns:
            str: A string containing the full HTML report.
        """
        displacement_plot: str = self.generate_displacement_plot()
        force_plot: str = self.generate_force_plot()
        element_plot: str = self.generate_element_plot()
        nodes_table: str = self.format_nodes_table()
        elements_table: str = self.format_elements_table()
        summary_table: str = self.format_summary_table()
        report_date: str = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        css: str = self.get_css_styles()
        html: str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FEA Solution Report</title>
            {css}
        </head>
        <body>
            <div class="header">
                <div class="logo">FEAnalysisApp</div>
                <div>Finite Element Analysis Report</div>
                <div>{report_date}</div>
            </div>
            <div class="section">
                <h1 class="report-title">Finite Element Analysis Solution Report</h1>
                <div style="text-align: center; margin-bottom: 30px;">
                    <span class="badge {self.get_solution_status_badge()}">
                        Solution Status: {'Valid' if self.solution_data['solver_info']['is_valid'] else 'Invalid'}
                    </span>
                </div>
            </div>
            <div class="section">
                <h2 class="section-title">1. Analysis Summary</h2>
                <div class="plot-container">
                    <div class="plot">
                        <img src="data:image/png;base64,{displacement_plot}" alt="Displacements Plot">
                    </div>
                    <div class="plot">
                        <img src="data:image/png;base64,{force_plot}" alt="Forces Plot">
                    </div>
                </div>
                {summary_table}
                <div class="notes">
                    <h3 class="subsection-title">Analysis Notes</h3>
                    <p>The finite element analysis was performed using a <strong>{self.structure_info.get('element_type', 'N/A')}</strong> element type
                    in <strong>{self.structure_info.get('dimension', 'N/A')}</strong> space. The model contains <strong>{len(self.imported_data.get('nodes', {}))}</strong> nodes
                    and <strong>{len(self.imported_data.get('elements', {}))}</strong> elements.</p>
                    <p>The solver converged in <span class="highlight"><strong>{self.solution_data['solver_info']['iterations']} iterations</strong></span>
                    with a residual of <span class="highlight"><strong>{self.solution_data['solver_info']['residual']:.4e}</strong></span>.</p>
                </div>
            </div>
            <div class="section">
                <h2 class="section-title">2. Nodal Results</h2>
                <div class="plot-container">
                    <div class="plot">
                        <img src="data:image/png;base64,{element_plot}" alt="Element Results Plot">
                    </div>
                </div>
                <h3 class="subsection-title">Nodal Displacements and Forces</h3>
                {nodes_table}
            </div>
            <div class="section">
                <h2 class="section-title">3. Element Results</h2>
                <h3 class="subsection-title">Element Forces and Stresses</h3>
                {elements_table}
            </div>
            <div class="section">
                <h2 class="section-title">4. Model Information</h2>
                {self.format_material_properties()}
                {self.format_cross_section_properties()}
            </div>
            <div class="footer">
                <p>This report was automatically generated by FEAnalysisApp on {report_date}</p>
                <p>© {datetime.now().year} Finite Element Analysis App.</p>
            </div>
        </body>
        </html>
        """
        return html

    # ---------------------------------------------
    # CSS STYLES AND HELPERS
    # ---------------------------------------------

    def get_css_styles(self, font_scale: float = 0.8) -> str:
        """
        Returns the CSS styles for the HTML report.
        Allows for scaling of font sizes and supports light/dark themes.
        Args:
            font_scale (float): A multiplier to scale all font sizes. Defaults to 0.8.
        Returns:
            str: A string containing the CSS rules within `<style>` tags.
        """
        if self.theme == "dark":
            return self._get_dark_theme_css(font_scale)
        else:
            return self._get_light_theme_css(font_scale)

    def _get_light_theme_css(self, font_scale: float) -> str:
        """Returns CSS for light theme."""
        return f"""
        <style>
            :root {{
                --primary-color: #2c3e50;
                --secondary-color: #3498db;
                --accent-color: #e74c3c;
                --light-color: #ecf0f1;
                --dark-color: #34495e;
                --bg-color: #ffffff;
                --text-color: #333333;
                --border-color: #dddddd;
                --table-header-bg: var(--primary-color);
                --table-header-color: white;
                --table-row-even: #f8f9fa;
                --table-row-hover: #f1f1f1;
                --notes-bg: #f8f9fa;
                --highlight-bg: #fffde7;
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
                color: var(--text-color);
                background-color: var(--bg-color);
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
            }}
            .logo {{
                font-size: var(--font-xlarge);
                font-weight: bold;
            }}
            .report-title {{
                font-size: var(--font-title);
                text-align: center;
                margin: 0;
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
            }}
            th {{
                background-color: var(--table-header-bg);
                color: var(--table-header-color);
                padding: 10px;
                text-align: left;
                font-size: var(--font-body);
            }}
            td {{
                padding: 8px 10px;
                border-bottom: 1px solid var(--border-color);
            }}
            tr:nth-child(even) {{
                background-color: var(--table-row-even);
            }}
            tr:hover {{
                background-color: var(--table-row-hover);
            }}
            .plot-container {{
                display: flex;
                flex-direction: row;
                justify-content: space-around;
                align-items: center;
                width: 100%;
            }}
            .plot {{
                margin: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-radius: 5px;
                overflow: hidden;
                max-width: 100%;
            }}
            .plot img {{
                max-width: 100%;
                height: auto;
                display: block;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #777;
                font-size: var(--font-small);
                border-top: 1px solid var(--border-color);
            }}
            .highlight {{
                background-color: var(--highlight-bg);
                padding: 2px 5px;
                border-radius: 3px;
            }}
            .badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
            }}
            .badge-success {{
                background-color: #2ecc71;
                color: white;
            }}
            .badge-warning {{
                background-color: #f39c12;
                color: white;
            }}
            .badge-danger {{
                background-color: #e74c3c;
                color: white;
            }}
            .key-value {{
                display: grid;
                grid-template-columns: 220px 1fr;
                gap: 10px;
                margin: 10px 0;
            }}
            .key {{
                font-weight: bold;
                color: var(--dark-color);
                font-size: var(--font-body);
            }}
            .value {{
                color: #555;
                font-size: var(--font-body);
            }}
            .notes {{
                background-color: var(--notes-bg);
                padding: 15px;
                border-left: 4px solid var(--secondary-color);
                margin: 20px 0;
                font-size: var(--font-body);
            }}
        </style>
        """

    def _get_dark_theme_css(self, font_scale: float) -> str:
        """Returns CSS for dark theme."""
        return f"""
        <style>
            :root {{
                --primary-color: #4a9cff;
                --secondary-color: #6bb5ff;
                --accent-color: #ff6b6b;
                --light-color: #2d2d2d;
                --dark-color: #e0e0e0;
                --bg-color: #1a1a1a;
                --text-color: #e0e0e0;
                --border-color: #444444;
                --table-header-bg: #3a3a3a;
                --table-header-color: white;
                --table-row-even: #2a2a2a;
                --table-row-hover: #3d3d3d;
                --notes-bg: #2a2a2a;
                --highlight-bg: #2d2d00;
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
                color: var(--text-color);
                background-color: var(--bg-color);
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
            }}
            .logo {{
                font-size: var(--font-xlarge);
                font-weight: bold;
            }}
            .report-title {{
                font-size: var(--font-title);
                text-align: center;
                margin: 0;
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
                box-shadow: 0 2px 3px rgba(0,0,0,0.3);
                font-size: var(--font-small);
            }}
            th {{
                background-color: var(--table-header-bg);
                color: var(--table-header-color);
                padding: 10px;
                text-align: left;
                font-size: var(--font-body);
            }}
            td {{
                padding: 8px 10px;
                border-bottom: 1px solid var(--border-color);
            }}
            tr:nth-child(even) {{
                background-color: var(--table-row-even);
            }}
            tr:hover {{
                background-color: var(--table-row-hover);
            }}
            .plot-container {{
                display: flex;
                flex-direction: row;
                justify-content: space-around;
                align-items: center;
                width: 100%;
            }}
            .plot {{
                margin: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                border-radius: 5px;
                overflow: hidden;
                max-width: 100%;
                background-color: var(--light-color);
            }}
            .plot img {{
                max-width: 100%;
                height: auto;
                display: block;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #999;
                font-size: var(--font-small);
                border-top: 1px solid var(--border-color);
            }}
            .highlight {{
                background-color: var(--highlight-bg);
                padding: 2px 5px;
                border-radius: 3px;
            }}
            .badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
            }}
            .badge-success {{
                background-color: #27ae60;
                color: white;
            }}
            .badge-warning {{
                background-color: #f39c12;
                color: white;
            }}
            .badge-danger {{
                background-color: #c0392b;
                color: white;
            }}
            .key-value {{
                display: grid;
                grid-template-columns: 220px 1fr;
                gap: 10px;
                margin: 10px 0;
            }}
            .key {{
                font-weight: bold;
                color: var(--dark-color);
                font-size: var(--font-body);
            }}
            .value {{
                color: #bbb;
                font-size: var(--font-body);
            }}
            .notes {{
                background-color: var(--notes-bg);
                padding: 15px;
                border-left: 4px solid var(--secondary-color);
                margin: 20px 0;
                font-size: var(--font-body);
            }}
        </style>
        """

    def get_solution_status_badge(self) -> str:
        """
        Determines the CSS class for the solution status badge
        based on the solver's validity.
        Returns:
            str: "badge-success" if the solution is valid, otherwise "badge-danger".
        """
        return "badge-success" if self.solution_data['solver_info']['is_valid'] else "badge-danger"

    def format_value(self, val: Any) -> str:
        """
        Formats a numeric value for display in the HTML tables.
        Handles `None` and `np.nan` values, and formats numerical values

        for better readability (scientific notation for very small/large numbers,
        fixed precision for others).
        Args:
            val (Any): The value to format.
        Returns:
            str: The formatted string representation of the value.
        """

        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"

        elif isinstance(val, (int, float)):
            return f"{val:.4e}" if abs(val) < 1e-3 or abs(val) > 1e6 else f"{val:.4f}"

        return str(val)

    # ---------------------------------------------
    # PLOT GENERATION
    # ---------------------------------------------

    def generate_displacement_plot(self) -> str:
        """
        Generates a Matplotlib plot of nodal displacements.
        The plot is saved as a PNG image in a BytesIO buffer and then
        base64 encoded for embedding directly into the HTML report.
        Returns:
            str: A base64 encoded string of the displacement plot PNG image.
        """
        plt.style.use('dark_background' if self.theme == "dark" else 'default')
        
        plt.figure(figsize=(8, 5))
        displacements: Dict[int, Dict[str, float]] = self.solution_data['displacements']
        nodes: List[int] = sorted(displacements.keys())
        labels: List[str] = self.structure_info['displacement_labels']
        data: Dict[str, List[float]] = {label: [] for label in labels}

        for node in nodes:

            for label_idx, label in enumerate(labels):
                val: float = displacements[node].get(label, np.nan)
                data[label].append(val if not np.isnan(val) else 0.0)

        for label, values in data.items():
            plt.plot(nodes, values, marker='o', label=label)
        plt.title('Nodal Displacements')
        plt.xlabel('Node Number')
        plt.ylabel(f"Displacement ({self.imported_data['units']['Displacement (Dx,Dy,Dz)']})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate_force_plot(self) -> str:
        """
        Generates a Matplotlib plot of nodal forces.
        The plot is saved as a PNG image in a BytesIO buffer and then
        base64 encoded for embedding directly into the HTML report.
        Returns:
            str: A base64 encoded string of the force plot PNG image.
        """
        plt.style.use('dark_background' if self.theme == "dark" else 'default')
        
        plt.figure(figsize=(8, 5))
        forces: Dict[int, Dict[str, float]] = self.solution_data['forces']
        nodes: List[int] = sorted(forces.keys())
        labels: List[str] = self.structure_info['force_labels']
        data: Dict[str, List[float]] = {label: [] for label in labels}

        for node in nodes:

            for label_idx, label in enumerate(labels):
                val: float = forces[node].get(label, np.nan)
                data[label].append(val if not np.isnan(val) else 0.0)

        for label, values in data.items():
            plt.plot(nodes, values, marker='s', label=label)
        plt.title('Nodal Forces')
        plt.xlabel('Node Number')
        plt.ylabel(f"Force ({self.imported_data['units']['Force (Fx,Fy,Fz)']})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate_element_plot(self) -> str:
        """
        Generates a Matplotlib plot of element internal forces and moments.
        The type and number of plots generated adapt based on the element type
        (Truss, 2D Beam, 3D Frame). The plot(s) are saved as a PNG image in a
        BytesIO buffer and then base64 encoded for embedding.
        Returns:
            str: A base64 encoded string of the element results plot PNG image.
        """
        plt.style.use('dark_background' if self.theme == "dark" else 'default')
        
        element_results: Dict[int, Dict[str, Dict[str, float]]] = self.solution_data['elements_forces']
        elements: List[int] = sorted(element_results.keys())
        metrics: List[str]
        titles: List[str]
        units: List[str]
        figsize: Tuple[int, int]
        nrows: int
        ncols: int
        element_type: str = self.structure_info.get('element_type', '')

        if element_type in ['2D_Truss', '3D_Truss']:
            metrics = ['axial_force']
            titles = ['Axial Force']
            units = [self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN')]
            figsize = (6, 4)
            nrows, ncols = 1, 1
        elif element_type == '2D_Beam':
            metrics = ['axial_force', 'shear_force', 'bending_moment']
            titles = ['Axial Force', 'Shear Force', 'Bending Moment']
            units = [
                self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN'),
                self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN'),
                f"{self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN')}·m"
            ]
            figsize = (12, 4)
            nrows, ncols = 1, 3
        else:
            metrics = ['axial_force', 'shear_force_y', 'shear_force_z',
                       'torsional_moment', 'bending_moment_y', 'bending_moment_z']
            titles = ['Axial Force', 'Shear Force Y', 'Shear Force Z',
                      'Torsional Moment', 'Bending Moment Y', 'Bending Moment Z']
            units = [
                self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN'),
                self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN'),
                self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN'),
                f"{self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN')}·m",
                f"{self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN')}·m",
                f"{self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN')}·m"
            ]
            figsize = (12, 8)
            nrows, ncols = 2, 3
        data: Dict[str, List[float]] = {metric: [] for metric in metrics}

        for element in elements:

            for metric in metrics:
                val: float = element_results[element]['internal_forces'].get(metric, 0.0)
                data[metric].append(val)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle('Element Internal Forces and Moments')

        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flat

        for i, (ax, metric, title, unit) in enumerate(zip(axes, metrics, titles, units)):
            ax.plot(elements, data[metric], marker='o', color=f'C{i}')
            ax.set_title(title)
            ax.set_xlabel('Element Number')
            ax.set_ylabel(unit)
            ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    # ---------------------------------------------
    # TABLE FORMATTING
    # ---------------------------------------------

    def format_nodes_table(self) -> str:
        """
        Formats the nodal displacements and forces into an HTML table.
        The table dynamically adjusts columns based on the labels defined in `structure_info`.
        Returns:
            str: An HTML string representing the table of nodal results.
        """
        displacements: Dict[int, Dict[str, float]] = self.solution_data['displacements']
        forces: Dict[int, Dict[str, float]] = self.solution_data['forces']
        nodes: List[int] = sorted(displacements.keys())
        disp_labels: List[str] = self.structure_info['displacement_labels']
        force_labels: List[str] = self.structure_info['force_labels']
        table: str = f"""
        <table>
            <thead>
                <tr>
                    <th>Node</th>
                    <th colspan="{len(disp_labels)}">Displacements ({self.imported_data['units'].get('Displacement (Dx,Dy,Dz)', 'mm')})</th>
                    <th colspan="{len(force_labels)}">Forces ({self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN')})</th>
                </tr>
                <tr>
                    <th></th>
        """

        for label in disp_labels:
            table += f'<th>{label}</th>'

        for label in force_labels:
            table += f'<th>{label}</th>'
        table += """
                </tr>
            </thead>
            <tbody>
        """

        for node in nodes:
            table += f'<tr><td>{node}</td>'

            for label in disp_labels:
                val: float = displacements[node].get(label, np.nan)
                table += f'<td>{self.format_value(val)}</td>'

            for label in force_labels:
                val: float = forces[node].get(label, np.nan)
                table += f'<td>{self.format_value(val)}</td>'
            table += '</tr>'
        table += """
            </tbody>
        </table>
        """
        return table

    def format_elements_table(self) -> str:
        """
        Formats the element internal forces and moments into an HTML table.
        The columns of the table dynamically adjust based on the element type
        (Truss, 2D Beam, 3D Frame).
        Returns:
            str: An HTML string representing the table of element results.
        """
        element_results: Dict[int, Dict[str, Dict[str, float]]] = self.solution_data['elements_forces']
        elements: List[int] = sorted(element_results.keys())
        columns: List[str]
        element_type: str = self.structure_info.get('element_type', '')
        units_force: str = self.imported_data['units'].get('Force (Fx,Fy,Fz)', 'kN')
        units_moment: str = f"{units_force}·m"
        units_length: str = self.imported_data['units'].get('Position (X,Y,Z)', 'm')

        if element_type in ['2D_Truss', '3D_Truss']:
            columns = ['Element', 'Nodes', f'Axial Force ({units_force})', f'Length ({units_length})']
        elif element_type == '2D_Beam':
            columns = ['Element', 'Nodes', f'Axial Force ({units_force})', f'Shear ({units_force})',
                       f'Moment ({units_moment})', f'Length ({units_length})']
        elif element_type == '3D_Frame':
            columns = ['Element', 'Nodes', f'Axial ({units_force})', f'Shear Y ({units_force})',
                       f'Shear Z ({units_force})', f'Torsion ({units_moment})',
                       f'Moment Y ({units_moment})', f'Moment Z ({units_moment})', f'Length ({units_length})']
        else:
            columns = ['Element', 'Nodes', f'Length ({units_length})']
        table: str = f"""
        <table>
            <thead>
                <tr>
                    {"".join(f'<th>{col}</th>' for col in columns)}
                </tr>
            </thead>
            <tbody>
        """

        for element in elements:
            res: Dict[str, float] = element_results[element]['internal_forces']
            elem_data: Dict[str, Any] = self.imported_data['elements'][element]
            table += f"""
            <tr>
                <td>{element}</td>
                <td>{elem_data.get('node1', 'N/A')} → {elem_data.get('node2', 'N/A')}</td>
            """

            if element_type in ['2D_Truss', '3D_Truss']:
                table += f"""
                <td>{self.format_value(res.get('axial_force', np.nan))}</td>
                <td>{elem_data.get('length', np.nan):.2f}</td>
                """
            elif element_type == '2D_Beam':
                table += f"""
                <td>{self.format_value(res.get('axial_force', np.nan))}</td>
                <td>{self.format_value(res.get('shear_force', np.nan))}</td>
                <td>{self.format_value(res.get('bending_moment', np.nan))}</td>
                <td>{elem_data.get('length', np.nan):.2f}</td>
                """
            elif element_type == '3D_Frame':
                table += f"""
                <td>{self.format_value(res.get('axial_force', np.nan))}</td>
                <td>{self.format_value(res.get('shear_force_y', np.nan))}</td>
                <td>{self.format_value(res.get('shear_force_z', np.nan))}</td>
                <td>{self.format_value(res.get('torsional_moment', np.nan))}</td>
                <td>{self.format_value(res.get('bending_moment_y', np.nan))}</td>
                <td>{self.format_value(res.get('bending_moment_z', np.nan))}</td>
                <td>{elem_data.get('length', np.nan):.2f}</td>
                """
            else:
                table += f"""
                <td>{elem_data.get('length', np.nan):.2f}</td>
                """
            table += "</tr>"
        table += """
            </tbody>
        </table>
        """
        return table

    def format_summary_table(self) -> str:
        """
        Formats the analysis summary information into an HTML table.
        This includes details about the model size, solver type, iterations, and residual.
        Returns:
            str: An HTML string representing the analysis summary table.
        """
        solver_info: Dict[str, Any] = self.solution_data['solver_info']
        table: str = f"""
        <table>
            <thead>
                <tr>
                    <th colspan="2">Analysis Summary</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Model Type</td>
                    <td>{self.structure_info.get('element_type', 'N/A')} {self.structure_info.get('dimension', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Nodes</td>
                    <td>{len(self.imported_data.get('nodes', {}))}</td>
                </tr>
                <tr>
                    <td>Elements</td>
                    <td>{len(self.imported_data.get('elements', {}))}</td>
                </tr>
                <tr>
                    <td>Solver Type</td>
                    <td>{'Direct' if solver_info['iterations'] == 1 else 'Iterative'}</td>
                </tr>
                <tr>
                    <td>Iterations</td>
                    <td>{solver_info.get('iterations', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Residual</td>
                    <td>{solver_info.get('residual', np.nan):.4e}</td>
                </tr>
                <tr>
                    <td>Solution Status</td>
                    <td>{'Valid' if solver_info['is_valid'] else 'Invalid'}</td>
                </tr>
            </tbody>
        </table>
        """
        return table

    def format_material_properties(self) -> str:
        """
        Formats the properties of the first detected material
        into an HTML key-value pair display. Assumes a consistent material

        for simplicity in the report.
        Returns:
            str: An HTML string representing the material properties section.
                 Returns an empty string if no materials are found.
        """
        materials: Dict[str, Any] = self.imported_data.get('materials', {})

        if not materials:
            return ""

        material_code: str = next(iter(materials.keys()))
        material: Dict[str, Any] = materials[material_code]
        material_props: Dict[str, Any] = material.get('properties', {})
        return f"""

        <h3 class="subsection-title">Material Properties</h3>
        <div class="key-value">
            <div class="key">Material Name:</div>
            <div class="value">{material_code}</div>
            <div class="key">Material Type:</div>
            <div class="value">{material.get('type', 'N/A')}</div>
            <div class="key">Modulus of Elasticity (E):</div>
            <div class="value">{self.format_value(material_props.get('E'))} {self.imported_data['units'].get('Modulus (E,G)', 'GPa')}</div>
            <div class="key">Poisson's Ratio (ν):</div>
            <div class="value">{self.format_value(material_props.get('v'))}</div>
            <div class="key">Density (ρ):</div>
            <div class="value">{self.format_value(material_props.get('density'))} {self.imported_data['units'].get('Density (ρ)', 'kg/m³')}</div>
        </div>
        """

    def format_cross_section_properties(self) -> str:
        """
        Formats the properties of the first detected cross-section
        into an HTML key-value pair display. Assumes a consistent cross-section

        for simplicity in the report and includes relevant geometric properties.
        Returns:
            str: An HTML string representing the cross-section properties section.
                 Returns an empty string if no cross-sections are found.
        """
        cross_sections: Dict[str, Any] = self.imported_data.get('cross_sections', {})

        if not cross_sections:
            return ""

        section_code: str = next(iter(cross_sections.keys()))
        section: Dict[str, Any] = cross_sections[section_code]
        section_dimensions: Dict[str, Any] = section.get('dimensions', {})
        elem1_props: Dict[str, Any] = self.imported_data['elements'].get(1, {})
        dimensions_html: str = ""

        for dim, value in section_dimensions.items():
            dimensions_html += f"""
            <div class="key">{dim}:</div>
            <div class="value">{self.format_value(value)}</div>
            """
        return f"""

        <h3 class="subsection-title">Cross-Section Properties</h3>
        <div class="key-value">
            <div class="key">Section Name:</div>
            <div class="value">{section_code}</div>
            <div class="key">Section Type:</div>
            <div class="value">{section.get('type', 'N/A')}</div>
            {dimensions_html}
            <div class="key">Area (A):</div>
            <div class="value">{self.format_value(elem1_props.get('A'))} {self.imported_data['units'].get('Cross-Sectional Area (A)', 'cm²')}</div>
            <div class="key">Moment of Inertia (Iy):</div>
            <div class="value">{self.format_value(elem1_props.get('Iy'))} {self.imported_data['units'].get('Moment of Inertia (Iy,Iz,J)', 'm⁴')}</div>
            <div class="key">Moment of Inertia (Iz):</div>
            <div class="value">{self.format_value(elem1_props.get('Iz'))} {self.imported_data['units'].get('Moment of Inertia (Iy,Iz,J)', 'm⁴')}</div>
            <div class="key">Torsional Constant (J):</div>
            <div class="value">{self.format_value(elem1_props.get('J'))} {self.imported_data['units'].get('Moment of Inertia (Iy,Iz,J)', 'm⁴')}</div>
        </div>
        """
    
    # ---------------------------------------------
    # REPORT DISPLAY
    # ---------------------------------------------

    def show_report(self) -> None:
        """
        Displays the generated HTML report in a PyQt6 `QWebEngineView` window.
        This method creates a simple application window to host the web view.
        """
        app = QApplication([])
        window = QMainWindow()
        window.setWindowTitle("FEA Solution Report")
        window.resize(1200, 800)
        central_widget = QWidget()
        layout = QVBoxLayout()
        web_view = QWebEngineView()
        html_content: str = self.generate_fe_report()
        web_view.setHtml(html_content)
        layout.addWidget(web_view)
        central_widget.setLayout(layout)
        window.setCentralWidget(central_widget)
        window.show()
        app.exec()