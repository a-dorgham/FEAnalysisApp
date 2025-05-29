from typing import Optional, Dict, Any, Union
from PyQt6.QtWidgets import QMessageBox, QWidget
from PyQt6.QtCore import Qt

class ErrorHandler:
    """
    A centralized error handling class to manage warnings and error messages and codes.
    Supports for both fatal errors and non-fatal warnings. Provides Detailed error logging capabilities.
        100-199: Core FEA errors
        200-299: Model definition errors
        300-399: File I/O errors
        400-499: GUI interaction errors
        500-599: System/performance errors
        600-699: Multiphysics/special analysis
        1000-1999: Warnings
    """

    # ---------------------------------------------
    # ERROR CODE DEFINITIONS
    # ---------------------------------------------

    ERROR_CODES: Dict[int, str] = {

        # =============================================
        # CORE FEA ERRORS (100-199)
        # =============================================
        100: "FEA System Error.",
        101: "Numerical instability detected in solution.",
        102: "Convergence not achieved in nonlinear analysis.",
        103: "Maximum iterations reached without convergence.",
        110: "Invalid node definition.",
        111: "Invalid element connectivity.",
        112: "Duplicate node ID detected.",
        113: "Duplicate element ID detected.",
        114: "Invalid material assignment.",
        115: "Invalid cross-section assignment.",
        116: "Invalid thickness definition for shell elements.",
        117: "Invalid orientation definition.",
        130: "Stiffness matrix computation failed.",
        131: "Singular stiffness matrix detected - structure may be unconstrained or have mechanisms.",
        132: "Zero pivot encountered during factorization.",
        133: "Ill-conditioned matrix detected.",
        134: "Eigenvalue solution failed.",
        135: "Linear solver failed.",
        136: "Nonlinear solver failed.",
        137: "Time step too small in transient analysis.",
        138: "Missing method.",
        150: "Result processing failed.",
        151: "Stress recovery failed.",
        152: "Strain recovery failed.",
        153: "Invalid result request.",
        160: "Meshing failed.",
        161: "Invalid mesh parameters.",
        162: "Poor element quality detected.",
        163: "Mesh generation timeout.",
        164: "Incompatible mesh types.",
        # =============================================
        # MODEL DEFINITION ERRORS (200-299)
        # =============================================
        200: "Invalid geometry definition.",
        201: "Self-intersecting geometry detected.",
        202: "Degenerate geometry element.",
        203: "Unknow error.",
        210: "Invalid material definition.",
        211: "Missing material properties.",
        212: "Unsupported material model.",
        213: "Material nonlinearity parameters invalid.",
        220: "Invalid boundary condition.",
        221: "Conflicting boundary conditions.",
        222: "Over-constrained condition detected.",
        223: "Under-constrained condition detected.",
        230: "Invalid load definition.",
        231: "Load application point not found.",
        232: "Distributed load parameters invalid.",
        233: "Pressure load parameters invalid.",
        234: "Thermal load parameters invalid.",
        243: "Unknow error.",
        # =============================================
        # FILE I/O ERRORS (300-399)
        # =============================================
        300: "File operation failed.", 
        301: "File not found.",
        302: "Permission denied for file operation.",
        303: "File format not recognized.",
        304: "File version incompatible.",
        310: "Import failed.",
        311: "Unsupported import format.",
        312: "Import file corrupted.",
        313: "Import geometry repair needed.",
        320: "Export failed.",
        321: "Unsupported export format.",
        322: "Export template missing.",
        323: "Export data incomplete.",
        330: "Project file corrupted.",
        331: "Project version mismatch.",
        332: "Project component missing.",
        # =============================================
        # GUI & USER INTERACTION ERRORS (400-499)
        # =============================================
        400: "Visualization error.",
        401: "View configuration invalid.",
        402: "Result scaling failed.",
        403: "Deformation visualization failed.",
        410: "Invalid selection.",
        411: "No selection made.",
        412: "Multiple selection not allowed.",
        413: "Selection type mismatch.",
        420: "Input validation failed.",
        421: "Required field empty.",
        422: "Numerical value out of range.",
        423: "Invalid input format.",
        424: "Conflicting inputs detected.",
        # =============================================
        # SYSTEM & PERFORMANCE ERRORS (500-599)
        # =============================================
        500: "Memory allocation failed.",
        501: "Insufficient memory for operation.",
        502: "Memory access violation.",
        510: "Operation timeout.",
        511: "Computation too intensive for current settings.",
        512: "Hardware acceleration failed.",
        520: "License validation failed.",
        521: "Feature not licensed.",
        522: "System configuration invalid.",
        # =============================================
        # MULTIPHYSICS & SPECIAL ANALYSIS (600-699)
        # =============================================
        600: "Coupled analysis failed.",
        601: "Thermal-mechanical interaction failed.",
        602: "Fluid-structure interaction failed.",
        603: "Contact detection failed.",
    }

    # =============================================
    # WARNINGS (1000-1999)
    # =============================================

    WARNING_CODES: Dict[int, str] = {
        1000: "Warning.",
        1001: "Numerical approximation warning.",
        1002: "Small value detected - possible numerical instability.",
        1003: "Large deformation detected - linear analysis may be inappropriate.",
        1004: "Thin structure detected - consider shell elements.",
        1005: "Potential over-constraint warning.",
        1006: "Model check recommended.",
        1007: "Units inconsistency detected.",
        1008: "Approximate boundary condition applied.",
        1009: "Coarse mesh warning - results may be inaccurate.",
        1010: "Singularity detected in results.",
        1011: "Insufficient internal force data.",
        1012: "Missing force label.",
        1013: "Node not found.",
        1014: "missing displacement label.",
        1015: "Non-numeric value.",
    }

    # ---------------------------------------------
    # PUBLIC INTERFACE
    # ---------------------------------------------

    @staticmethod
    def handle_error(
        code: int,
        details: str = "",
        parent: Optional[QWidget] = None,
        fatal: bool = False,
        exception_type: type[Exception] = RuntimeError
    ) -> None:
        """
        Handle an error with GUI notification and optional exception raising.
        Args:
            code: The error code from ERROR_CODES
            details: Additional error context (default: "")
            parent: Parent widget for dialogs (default: None)
            fatal: Whether to raise an exception (default: True)
            exception_type: Type of exception to raise (default: RuntimeError)
        Raises:
            Exception: When fatal=True, raises the specified exception type
        """
        message = ErrorHandler._get_full_message(code, details, is_warning=False)
        ErrorHandler._show_message_box(
            message=message,
            title=f"Error {code}",
            icon=QMessageBox.Icon.Critical,
            parent=parent
        )

        if fatal:
            raise exception_type(message)

    @staticmethod
    def handle_warning(
        code: int,
        details: str = "",
        parent: Optional[QWidget] = None
        ) -> None:
        """
        Display a non-fatal warning message.
        Args:
            code: The warning code from WARNING_CODES
            details: Additional warning context (default: "")
            parent: Parent widget for dialogs (default: None)
        """
        message = ErrorHandler._get_full_message(code, details, is_warning=True)
        ErrorHandler._show_message_box(
            message=message,
            title=f"Warning {code}",
            icon=QMessageBox.Icon.Warning,
            parent=parent
        )
    # ---------------------------------------------
    # PRIVATE UTILITIES
    # ---------------------------------------------

    @staticmethod
    def _get_full_message(
        code: int,
        details: str,
        is_warning: bool
    ) -> str:
        """
        Construct the complete error/warning message.
        Args:
            code: The error/warning code
            details: Additional context information
            is_warning: Whether this is a warning (vs error)
        Returns:
            Formatted complete message string
        Raises:
            ValueError: If code is not found in the appropriate dictionary
        """
        code_dict = ErrorHandler.WARNING_CODES if is_warning else ErrorHandler.ERROR_CODES

        if code not in code_dict:
            raise ValueError(f"Unknown {'warning' if is_warning else 'error'} code: {code}")
        base_msg = code_dict[code]
        return f"Error {code}\n{base_msg}\n\n {details}" if details else base_msg

    @staticmethod
    def _show_message_box(
        message: str,
        title: str,
        icon: QMessageBox.Icon,
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Display a standardized message box.
        Args:
            message: The main message content
            title: Dialog window title
            icon: QMessageBox icon to display
            parent: Parent widget (default: None)
        """
        msg_box = QMessageBox(parent)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)
        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        msg_box.exec()
        
    # ---------------------------------------------
    # CONVENIENCE METHODS
    # ---------------------------------------------

    @classmethod
    def get_error_message(cls, code: int) -> str:
        """Get the base message for an error code."""
        return cls.ERROR_CODES.get(code, f"Unknown error code: {code}")

    @classmethod
    def get_warning_message(cls, code: int) -> str:
        """Get the base message for a warning code."""
        return cls.WARNING_CODES.get(code, f"Unknown warning code: {code}")

    @staticmethod
    def critical(
        message: str,
        details: str = "",
        parent: Optional[QWidget] = None,
        exception_type: type[Exception] = RuntimeError
    ) -> None:
        """
        Display a critical error and raise exception (convenience method).
        Args:
            message: Primary error message
            details: Additional context (default: "")
            parent: Parent widget (default: None)
            exception_type: Exception class to raise (default: RuntimeError)
        """
        full_msg = f"{message}: {details}" if details else message
        ErrorHandler._show_message_box(
            message=full_msg,
            title="Critical Error",
            icon=QMessageBox.Icon.Critical,
            parent=parent
        )
        raise exception_type(full_msg)

    @staticmethod
    def warning(
        message: str,
        details: str = "",
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Display a warning message (convenience method).
        Args:
            message: Primary warning message
            details: Additional context (default: "")
            parent: Parent widget (default: None)
        """
        full_msg = f"{message}: {details}" if details else message
        ErrorHandler._show_message_box(
            message=full_msg,
            title="Warning",
            icon=QMessageBox.Icon.Warning,
            parent=parent
        )
        
    # ---------------------------------------------
    # SPECIALIZED FEA ERROR HANDLERS
    # ---------------------------------------------

    @staticmethod
    def handle_solver_error(
        solver_type: str,
        details: str = "",
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Handle solver-specific errors with appropriate messaging.
        Args:
            solver_type: Type of solver ('linear', 'nonlinear', 'eigen')
            details: Additional error context
            parent: Parent widget for dialogs
        """
        base_code = {
            'linear': 135,
            'nonlinear': 136,
            'eigen': 134
        }.get(solver_type.lower(), 130)
        ErrorHandler.handle_error(
            code=base_code,
            details=details,
            parent=parent,
            fatal=True,
            exception_type=RuntimeError
        )

    @staticmethod
    def handle_mesh_error(
        error_type: str,
        details: str = "",
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Handle mesh generation/quality errors.
        Args:
            error_type: Type of mesh error ('generation', 'quality', 'compatibility')
            details: Additional context
            parent: Parent widget
        """
        base_code = {
            'generation': 160,
            'quality': 162,
            'compatibility': 164
        }.get(error_type.lower(), 160)
        ErrorHandler.handle_error(
            code=base_code,
            details=details,
            parent=parent,
            fatal=True
        )