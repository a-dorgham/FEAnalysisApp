
# FEAnalysisApp

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**FEAnalysisApp** is a cross-platform, Python-based graphical user interface (GUI) for performing Finite Element Analysis (FEA). Designed using PyQt6, it provides an intuitive and interactive environment for structural engineers, educators, and researchers to model, analyze, and visualize various types of structures, including beams, trusses, and frames in both 2D and 3D.

---

## Features

- Graphical User Interface built with PyQt6.
- Supports 2D and 3D structures: Trusses, Beams, and Frames.
- Customizable input using structured `.txt` files or GUI forms.
- Modular architecture with extensible solvers and element libraries.
- Supports both direct and iterative solvers.
- Real-time visualization of analysis results.
- Integrated input/output file viewer and result report generator.
- Built-in library for materials, cross-sections, and units.
- Exports results to human-readable reports and visual plots.

---

## Project Structure

```
FEAnalysisApp/
├── LICENSE                    # The project's license file.
├── README.md                  # This file.
├── main.py                    # Application entry point; starts the GUI.
├── requirements.txt           # Python dependencies for the project.
├── setup.cfg                  # Configuration file for packaging/tools (e.g., setuptools, pytest).
|
├── assets/                    # Static resources (icons, styling, web libraries).
│   ├── icons/                 # All icon files, structured by their context.
│   │   ├── analysis_window/   # Icons used in the analysis/viewer window (e.g., solve, view).
│   │   └── main_window/       # Icons and images for the main application window (e.g., structure types).
│   ├── plotly/                # External JavaScript libraries (e.g., for data visualization).
│   └── styles/                # CSS/QSS files for application styling (e.g., dark/light mode).
|
├── data/                      # Default, user-defined, and library data files (JSON format).
│   ├── defaults/              # Default settings for various analysis components (nodes, elements, units, etc.).
│   └── libraries/             # User-editable libraries for materials and cross-sections.
|
├── examples/                  # Sample input text files for running analyses.
│   └── *.txt
|
├── src/                       # Source code for the application's logic and modules.
│   ├── config.py              # Application-wide configuration settings.
│   ├── constants.py           # Definition of numerical or physical constants.
│   ├── core/                  # Core FEA logic, including structure handling and solvers.
│   │   ├── structure_io.py    # Logic for reading and writing structure data.
│   │   ├── elements/          # Implementation of different element types (bar, beam, frame).
│   │   └── solvers/           # Numerical solvers (direct, iterative) for the stiffness matrix.
│   ├── gui/                   # GUI components using the Qt framework (PyQt/PySide).
│   │   ├── setup_docks.py     # Configuration for the docked windows.
│   │   ├── viewers/           # Modules for displaying file, report, and analysis results.
│   │   ├── widgets/           # Custom reusable GUI components (spinner, tree widget).
│   │   └── windows/           # Definition of main application windows and dialogs.
│   └── utils/                 # Utility functions for common tasks.
│       ├── classes.py         # Custom helper classes.
│       ├── errors.py          # Custom exception and error handling.
│       ├── http.py            # HTTP client/server utilities (if used for data/API).
│       ├── io.py              # General Input/Output and file handling utilities.
│       └── units.py           # Unit conversion and management logic.
|
└── tests/                     # Test suite using pytest.
    ├── conftest.py            # Fixtures and configuration for pytest.
    ├── test_*.py              # Test files for individual elements.
    ├── integration/           # High-level integration tests.
    └── unit/                  # Low-level unit tests for specific functions/classes.
```

---

## Getting Started

### Prerequisites

- Python 3.10.5
- Recommended: Virtual environment

Install the dependencies using:

```bash
pip install -r requirements.txt
```

### Running the App

```bash
python main.py
```

The GUI will launch, providing access to structure modeling, analysis configuration, result viewing, and report generation.

---

## Usage

1. Load a Model: Use sample input files from `/examples/` folder or import your own.
2. Set Parameters: Define material properties, cross-sections, boundary conditions, and loads.
3. Solve: Choose between direct or iterative solvers.
4. Visualize: View deformation, internal forces, and support reactions graphically.
5. Export: Generate formatted reports and save the analysis results.

---

## Supported Structure Types

- 2D Truss
- 2D Beam
- 3D Truss
- 3D Frame

More element types and general 3D solids will be supported in future versions.

---

## Example Inputs

The `examples/` directory contains ready-to-use structural models. Example:

```text
2D_Truss.txt
2D_Beam.txt
3D_Frame.txt
```

Input format is simple, human-readable, and structured for easy editing.

---

## Developer Notes

### Modular Solvers

- `core/direct_solver.py`: Gauss elimination and banded matrix solver.
- `core/iterative_solver.py`: Iterative solving for large systems.
- `core/main_solver.py`: Solver interface and result integration.

### Extendable Element Library

- Linear bar, beam, and frame elements under `core/isoparametric_elements/`.

---

## Testing

Run tests using:

```bash
pytest tests/
```

Includes unit tests for core finite element components and shape functions.

---

## Screenshots

> _Main window_
<img width="712" alt="FEAnalysisApp_1" src="https://github.com/user-attachments/assets/c8095ec6-9637-4a1d-b81d-e84c86dbaa5f" />

> _Analysis window_
<img width="1312" alt="FEAnalysisApp_2" src="https://github.com/user-attachments/assets/ae688535-9133-4cf3-a8ee-c645b19e468d" />

> _File Viewer_
<img width="1312" alt="FEAnalysisApp_4" src="https://github.com/user-attachments/assets/407c713d-efb3-414a-89fe-253d8110fd40" />

> _Stiffness Matrix Viewer_
<img width="1312" alt="FEAnalysisApp_5" src="https://github.com/user-attachments/assets/93d62821-773c-4758-8c70-33b62a290568" />

> _Solved Displacements Viewer_
<img width="1312" alt="FEAnalysisApp_6" src="https://github.com/user-attachments/assets/7f4aff85-2c4f-42cd-befa-a64fe5bde3ca" />

> _Solved Stresses Viewer_
<img width="1312" alt="FEAnalysisApp_7" src="https://github.com/user-attachments/assets/b91a010f-2ca7-42a8-9918-b7edd474384e" />

> _Beam Shear and Bending Moments_
<img width="1312" alt="FEAnalysisApp_8" src="https://github.com/user-attachments/assets/893c7230-a901-4ea4-a8f7-762a627deed7" />

> _Analysis Report_
<img width="1312" alt="FEAnalysisApp_9" src="https://github.com/user-attachments/assets/02756b55-a6c5-4ac8-9ff1-0309cda7dcb1" />

> _Logging Console_
<img width="878" alt="FEAnalysisApp_10" src="https://github.com/user-attachments/assets/7ab71914-b181-4367-a522-1483dc929f21" />

---

## Roadmap

- Add support for shell and solid elements.
- Web-based remote analysis interface.
- Advanced post-processing with Plotly.
- Improved mesh generation and visualization.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

```bash
git clone https://github.com/a-dorgham/FEAnalysisApp.git
cd FEAnalysisApp
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For bug reports, feature requests, or collaboration:

- GitHub Issues: [FEAnalysisApp Issues](https://github.com/a-dorgham/FEAnalysisApp/issues)
- Email: a.k.y.dorgham@gmail.com
- Connect: [LinkedIn](https://www.linkedin.com/in/abdeldorgham) | [GoogleScholar](https://scholar.google.com/citations?user=EOwjslcAAAAJ&hl=en)  | [ResearchGate](https://www.researchgate.net/profile/Abdel-Dorgham-2) | [ORCiD](https://orcid.org/0000-0001-9119-5111)
  

