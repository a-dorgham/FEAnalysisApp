from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                             QLabel, QMessageBox, QScrollArea, 
                             QStyledItemDelegate, QStyleOptionViewItem, QLineEdit)
from PyQt6.QtCore import Qt, QPoint, QRect, QTimer, QModelIndex, QAbstractItemModel
from PyQt6.QtGui import (QTextCursor, QDoubleValidator, QIntValidator, QValidator)
import sys
from PyQt6.QtWebEngineCore import QWebEnginePage
from typing import Tuple, Optional, Dict, Any

# ---------------------------------------------
# UTILITY CLASSES
# ---------------------------------------------

class OutputStream:
    """
    A custom output stream class that redirects standard output to a QTextEdit widget.
    This class is useful for displaying print statements and other console output directly
    within a PyQt application's GUI, allowing for custom formatting and easy viewing
    of application logs or messages.
    """

    def __init__(self, text_edit_widget: Any) -> None:
        """
        Initializes the OutputStream with a QTextEdit widget.
        Args:
            text_edit_widget (Any): The QTextEdit widget where the output will be displayed.
                                   It's typed as Any because QTextEdit is a PyQt widget
                                   and its specific type might not be directly available
                                   without importing the entire QtWidgets module,
                                   which is already done.
        """
        self.text_edit_widget: Any = text_edit_widget

    def write(self, text: str) -> None:
        """
        Writes the given text to the associated QTextEdit widget.
        The text is formatted with a specific font, size, weight, and color, and
        'white-space: pre;' is applied to preserve formatting (e.g., newlines, spaces).
        Args:
            text (str): The text string to be written to the QTextEdit widget.
        """

        if self.text_edit_widget is None or not text.strip():
            return

        try:
            cursor: QTextCursor = self.text_edit_widget.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            html: str = (
                f"<span style='"
                f"font-family: Consolas, Monaco, monospace;"
                f"font-size: 9pt;"
                f"font-weight: 400;"
                f"color: #83a78c;"
                f"white-space: pre;"
                f"'>{text}</span><br>"
            )
            cursor.insertHtml(html)
            self.text_edit_widget.ensureCursorVisible()

        except Exception as e:
            sys.__stdout__.write(f"Exception occurred in OutputStream: {e}\n")

    def flush(self) -> None:
        """
        This method is required for file-like objects but does nothing in this implementation.
        It's included to satisfy the interface for `sys.stdout` redirection.
        """
        pass

# ---------------------------------------------
# MESSAGE BOXES
# ---------------------------------------------

class ScrollableMessageBox(QMessageBox):
    """
    A custom QMessageBox that includes a scrollable area for displaying lengthy text content.
    This class extends the standard QMessageBox to provide a more flexible way to present
    messages that might exceed the typical fixed size of a message box, preventing
    text truncation. It also includes methods for precise positioning.
    """

    def __init__(self, parent: Optional[QWidget] = None, icon: QMessageBox.Icon = QMessageBox.Icon.NoIcon) -> None:
        """
        Initializes the ScrollableMessageBox.
        Args:
            parent (Optional[QWidget]): The parent widget of the message box. Defaults to None.
            icon (QMessageBox.Icon): The icon to display in the message box. Defaults to NoIcon.
        """
        super().__init__(parent)
        self.setIcon(icon)
        self.scroll_area: QScrollArea = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.content_label: QLabel = QLabel()
        self.content_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.content_label.setWordWrap(True)
        self.scroll_area.setWidget(self.content_label)
        layout: QVBoxLayout = self.layout()
        layout.addWidget(self.scroll_area, layout.rowCount(), 0, 1, layout.columnCount())

    def setScrollableText(self, text: str) -> None:
        """
        Sets the text content to be displayed within the scrollable area of the message box.
        Args:
            text (str): The text string to display.
        """
        self.content_label.setText(text)

    def setMaximumScrollAreaHeight(self, height: int) -> None:
        """
        Sets the maximum height of the scrollable area within the message box.
        This helps control the overall size of the message box when displaying large amounts of text.
        Args:
            height (int): The maximum height in pixels for the scroll area.
        """
        self.scroll_area.setMaximumHeight(height)

    def alignCenter(self) -> None:
        """
        Centers the message box on the screen after it has been displayed.
        This method uses a QTimer.singleShot to delay the centering operation,
        ensuring that the message box has already been rendered and its geometry
        is finalized before attempting to center it.
        """

        def center_message_box() -> None:
            """Helper function to perform the actual centering."""
            screen: Any = QApplication.screens()[0]
            screen_geometry: QRect = screen.geometry()
            msg_box_rect: QRect = self.frameGeometry()
            center_point: QPoint = screen_geometry.center()
            top_left_x: int = center_point.x() - msg_box_rect.width() // 2
            top_left_y: int = center_point.y() - msg_box_rect.height() // 2
            top_left_point: QPoint = QPoint(top_left_x, top_left_y)
            msg_box_rect.moveTopLeft(top_left_point)
            self.setGeometry(msg_box_rect)
        QTimer.singleShot(0, center_message_box)   

    def alignCenterParent(self, parent: QWidget) -> None:
        """
        Centers the message box horizontally relative to its parent widget and aligns its
        top edge with the parent's top edge.
        This ensures the message box appears centered within the parent's bounds
        horizontally, starting from the parent's top.
        Args:
            parent (QWidget): The parent widget to align the message box against.
        """
        parent_rect: QRect = parent.geometry()
        parent_global_pos: QPoint = parent.mapToGlobal(parent_rect.topLeft())
        parent_global_rect: QRect = QRect(parent_global_pos, parent_rect.size())
        QApplication.processEvents()
        msg_box_rect: QRect = self.frameGeometry()
        msg_box_rect.moveCenter(parent_global_rect.center())
        msg_box_rect.moveTop(parent_global_rect.top())
        self.setGeometry(msg_box_rect)
# ---------------------------------------------
# DELEGATES
# ---------------------------------------------

class TypedColumnDelegate(QStyledItemDelegate):
    """
    A custom QStyledItemDelegate that provides specific editors and validators for
    different column types in a QAbstractItemView (e.g., QTreeWidget, QTableView).
    This delegate allows for type-specific input validation (e.g., float, int)
    and ensures that only designated columns are editable.
    """

    def __init__(self, editable_columns: Optional[set[int]] = None, 
                 column_types: Optional[Dict[int, str]] = None, 
                 parent: Optional[QWidget] = None) -> None:
        """
        Initializes the TypedColumnDelegate.
        Args:
            editable_columns (Optional[set[int]]): A set of column indices that are editable.
                                                    If None, no columns are editable by default.
            column_types (Optional[Dict[int, str]]): A dictionary mapping column indices to their
                                                      data types (e.g., {0: 'float', 1: 'int'}).
                                                      Supported types are 'float' and 'int'.
                                                      If None, no type-specific validation is applied.
            parent (Optional[QWidget]): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.editable_columns: set[int] = editable_columns or set()
        self.column_types: Dict[int, str] = column_types or {}

    def createEditor(self, parent: QWidget, option: 'QStyleOptionViewItem', index: 'QModelIndex') -> Optional[QWidget]:
        """
        Creates and returns a suitable editor widget for the given index.
        If the column is not in `editable_columns`, returns None, preventing editing.
        If the column has a specified type ('float' or 'int'), a QLineEdit with the
        corresponding validator (QDoubleValidator or QIntValidator) is created.
        Args:
            parent (QWidget): The parent widget for the editor.
            option (QStyleOptionViewItem): Provides style options for the item.
            index (QModelIndex): The model index of the item to be edited.
        Returns:
            Optional[QWidget]: The editor widget (QLineEdit) or None if the column is not editable.
        """
        col: int = index.column()
        dtype: Optional[str] = self.column_types.get(col)

        if col not in self.editable_columns:
            return None

        editor: QLineEdit = QLineEdit(parent)

        if dtype == 'float':
            editor.setValidator(QDoubleValidator())
        elif dtype == 'int':
            editor.setValidator(QIntValidator())
        return editor

    def setEditorData(self, editor: QWidget, index: 'QModelIndex') -> None:
        """
        Sets the data from the model to the editor widget.
        For QLineEdit editors, the model data is converted to a string and set as the editor's text.
        Args:
            editor (QWidget): The editor widget to set the data to.
            index (QModelIndex): The model index from which to retrieve data.
        """
        value: Any = index.model().data(index, Qt.ItemDataRole.EditRole)

        if isinstance(editor, QLineEdit):
            editor.setText(str(value))

    def setModelData(self, editor: QWidget, model: 'QAbstractItemModel', index: 'QModelIndex') -> None:
        """
        Gets the data from the editor widget and sets it back to the model.
        The input from the editor is validated and converted to the specified data type
        ('float' or 'int') before being set in the model. Invalid inputs are ignored.
        Args:
            editor (QWidget): The editor widget to retrieve data from.
            model (QAbstractItemModel): The model to which the data will be set.
            index (QModelIndex): The model index where the data will be stored.
        """

        if not isinstance(editor, QLineEdit):
            return

        text: str = editor.text()
        dtype: Optional[str] = self.column_types.get(index.column())

        try:

            if dtype == 'float':
                value: float = float(text)
            elif dtype == 'int':
                value: int = int(text)
            else:
                value: str = text
            model.setData(index, value, Qt.ItemDataRole.EditRole)

        except ValueError:
            pass

# ---------------------------------------------
# VALIDATORS
# ---------------------------------------------

class NaNValidator(QValidator):
    """
    A custom QValidator that extends QDoubleValidator to additionally allow "nan" (case-insensitive)
    and empty strings as valid inputs.
    This is useful for input fields where numerical values, NaN, or empty values are permissible.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initializes the NaNValidator.
        Args:
            parent (Optional[QWidget]): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.double_validator: QDoubleValidator = QDoubleValidator(parent)

    def validate(self, text: str, pos: int) -> Tuple[QValidator.State, str, int]:
        """
        Validates the given input text.
        Allows "nan" (case-insensitive) or an empty/space-only string as `Acceptable`.
        Otherwise, delegates to the `QDoubleValidator` for standard float validation.
        Args:
            text (str): The input text to validate.
            pos (int): The current cursor position within the text.
        Returns:
            Tuple[QValidator.State, str, int]: A tuple containing the validation state,
                                               the potentially fixed text, and the cursor position.
        """

        if text.lower() == "nan" or text.strip() == "":
            return QValidator.State.Acceptable, text, pos

        return self.double_validator.validate(text, pos)

    def fixup(self, text: str) -> str:
        """
        Attempts to fix up the input text if it's invalid.
        If the text is "nan" (case-insensitive) or empty/space-only, it returns "nan".
        Otherwise, it delegates to the `QDoubleValidator` for standard float fixup.
        Args:
            text (str): The input text to fix.
        Returns:
            str: The fixed-up text.
        """

        if text.lower() == "nan" or text.strip() == "":
            return "nan"

        return self.double_validator.fixup(text)

# ---------------------------------------------
# WEB ENGINE PAGES
# ---------------------------------------------


from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import Qt
import logging

logger = logging.getLogger("simulationCore")


# ---------------------------------------------
# CUSTOM SILENT WEBENGINEPAGE
# ---------------------------------------------
class SilentWebEnginePage(QWebEnginePage):
    """
    A custom QWebEnginePage that suppresses specific JavaScript console messages
    and allows setting the background color upon initialization.

    This class overrides the `javaScriptConsoleMessage` method to filter out
    messages containing "willReadFrequently", which can often be noisy in the
    QtWebEngineView console.
    """

    def __init__(
        self, parent=None, qt_global_color: Qt.GlobalColor = Qt.GlobalColor.white
    ):
        """
        Initializes the SilentWebEnginePage and sets the background color.

        Args:
            parent (QObject, optional): The parent object. Defaults to None.
            qt_global_color (Qt.GlobalColor, optional): The initial background color.
                Defaults to Qt.GlobalColor.white.
        """
        super().__init__(parent)
        self.setBackgroundColor(qt_global_color)

    def javaScriptConsoleMessage(
        self,
        level: "QWebEnginePage.JavaScriptConsoleMessageLevel",
        message: str,
        lineNumber: int,
        sourceID: str,
    ) -> None:
        """
        Overrides the base method to filter JavaScript console messages.

        If the message contains "willReadFrequently", it is ignored to reduce noise.
        Otherwise, the message is passed to the base class implementation for default handling.

        Args:
            level (QWebEnginePage.JavaScriptConsoleMessageLevel): The severity level of the message.
            message (str): The content of the JavaScript console message.
            lineNumber (int): The line number in the source file where the message originated.
            sourceID (str): The URL of the source file where the message originated.
        """
        # logger.info(f"[JS Message] Line-{lineNumber}: {message}")
        if "willReadFrequently" in message:
            return

        return
        # super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)

    def setDefaultBackground(self, bg_color: str = "white"):
        """
        Sets the background color of the web page to either white or black.

        Args:
            bg_color (str, optional): The desired background color. Must be "white" or "black".
                Defaults to "white".
        """
        qt_global_color = (
            Qt.GlobalColor.white if bg_color == "white" else Qt.GlobalColor.black
        )
        self.setBackgroundColor(qt_global_color)
