# -*- coding: utf-8 -*-
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTextEdit,
)

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter


DIALOG_CSS = """     
QDialog {
    border-color: red;
}
"""

TEXT_EDIT_CSS = """
QWidget {{
    background: {0};
    border-radius: 2px;
}}
"""

class Fractal_code_editor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Preferred)

        self.setStyleSheet(TEXT_EDIT_CSS.format("#1e1e27"))
        self.ce = Fractal_code_widget(self)
        _layout = QVBoxLayout()
        _layout.addWidget(self.ce, stretch=1)
        self.setLayout(_layout)

    def set_text(self, text):
        self.ce.setText(text)
    
    def sizeHint(self):
        return QtCore.QSize(650, 650)


class Fractal_code_widget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(TEXT_EDIT_CSS.format("#1e1e27"))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)

        self.lexer = get_lexer_by_name(
            "python3",
            stripall=False, # Strip all leading and trailing whitespaces
            stripnl=False # Strip leading and trailing newlines from the input
        )
        # To get all stypes availables :
        # from pygments.styles import get_all_styles
        # list(get_all_styles())
        self.formatter = HtmlFormatter(linenos=False, style='inkpot')
        self.formatter.noclasses = True

        # Event binding
        self.textChanged.connect(self.highlighter)

    def highlighter(self):
        text = self.toPlainText()

        trailing_spaces = 0
        n = len(text)
        for i in range(n):
            if text[n - i - 1] == " ":
                trailing_spaces += 1
            else:
                break
        if trailing_spaces > 2:
            text = text[:-1]
            trailing_spaces -= 1
        
        if not text.endswith("\n" + " " * trailing_spaces):
            text = text + "\n "
        result = highlight(text, self.lexer, self.formatter)

        with QtCore.QSignalBlocker(self):
            pos = self.textCursor().position()
            self.setText(result)
            cursor = self.textCursor()
            cursor.setPosition(min(pos, len(self.toPlainText())))
            self.setTextCursor(cursor)
