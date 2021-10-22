#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QDialog,
    QApplication,
    QVBoxLayout,
    QTextEdit,
)

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

#def highlighter():
#        text = area.toPlainText()
#        result = highlight(text, lexer, formatter)
#        area.setText(result)

#1e1e27
# QPlainTextEdit

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
        
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred)
#        self.setMinimumWidth(650)
#        self.setMinimumHeight(650)

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
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

        self.lexer = get_lexer_by_name(
            "python3",
            stripall=False, # Strip all leading and trailing whitespaces
            stripnl=False # Strip leading and trailing newlines from the input
        )
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



if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication 
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    
    win = Fractal_code_editor()
    win.set_text('print ("Hello World")\n# Test Program')
    win.exec()
