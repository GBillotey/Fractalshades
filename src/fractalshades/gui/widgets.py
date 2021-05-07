# -*- coding: utf-8 -*-
import inspect
import fractalshades.utils as fsutils
import sys
import os
import copy
import math
#
#import inspect


#from PyQt5.QtCore import QCoreApplication
#from PyQt5.QtWidgets import  QInputDialog, QLineEdit, QApplication
#from PyQt5 import QtGui
#from PyQt5 import QtCore, QtGui#, QtWidgets
#from PyQt5 import QtWidgets #as QtWidgets#import QApplication, qApp, QWidget, QMainWindow, QGridLayout, QMenuBar, QAction, QToolBar, QStatusBar

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
#from PyQt5.QtGui import QIcon

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (QApplication, qApp, QWidget, QMainWindow, 
                             QMenuBar, QAction, QToolBar, QLabel, QLineEdit,
                             QStatusBar, QMenu, QHBoxLayout, QVBoxLayout,
                             QGridLayout, QSpacerItem, QSizePolicy,
                             QGraphicsScene, QGraphicsView,
                             QGraphicsPixmapItem, QGraphicsItemGroup,
                             QGraphicsRectItem, QFrame
                             )



#class MyToolBar(QToolBar):
#    def __init__(self, title, parent):
#        super().__init__(title, parent)
#        self.addAction("Mon action")
#        self.addAction("Mon action2")
#        self.addAction("Mon action3")
#
#class MyMenuBar(QMenuBar):
#    def __init__(self, parent):
#        super().__init__(parent)
#        self.addMenu(QMenu("Mon menu", self))
#        self.addAction("About")

class QDict_viewer(QWidget):
    def __init__(self, parent, qdict):
        """
        A Widget to view an ordered dict.
        The ordered dict is not user-editable but can be programmatically
        updated.
        """
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self.setLayout(self._layout)
        self.widgets_reset(qdict)

    def widgets_reset(self, qdict):
        """
        Clears and reset all child widgets
        """
        self._del_ranges()
        self._qdict = qdict
        self._key_row = dict()
        row = 0
        for k, v in qdict.items(): #kwargs_dic.items():
            self._layout.addWidget(QLabel(k), row, 0, 1, 1)
            self._layout.addWidget(QLabel(str(v)), row, 1, 1, 1)
            self._key_row[k] = row
            row += 1
        spacer = QSpacerItem(1, 1,
                             QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._layout.addItem(spacer, row, 1, 1, 1)

    def values_update(self, update_dic):
        """
        Updates in-place with update_dic values
        """
        for k, v in update_dic.items():
            row = self._key_row[k]
            widget = self._layout.itemAtPosition(row, 1).widget()
            if widget is not None:
                self._qdict[k] = v
                widget.setText(str(v))

    def _del_ranges(self):
        """ Delete every item in self._layout """
        for i in reversed(range(self._layout.count())): 
            w = self._layout.itemAt(i).widget()
            if w is not None:
                w.setParent(None)
                # w.deleteLater()


class Param_Widget(QWidget):
    def __new__(cls, parent, ):
        pass


if __name__ == "__main__":
    # https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
    # import inspect
    import typing
    
    def a(*, i: float):
        return i
#    sgn = inspect.signature(a)
#    print("sgn", sgn)
#    ba = sgn.bind_partial(i=None)
#    print(a.__annotations__)
    var_type = a.__annotations__["i"]
    print("var_type", var_type, var_type is float)
    
    t = typing.Union[str, int]
    def a(*, i: t):
        return i
    var_type = a.__annotations__["i"]
    print("var_type", var_type, var_type is t)
    
    Point = typing.Tuple[float, float]
    def a(*, i: Point):
            return i
    var_type = a.__annotations__["i"]
    print("var_type", var_type, var_type is typing.Tuple[float, float])
    
    
#    Point = typing.TypedDict[float, float]
#    def a(*, i: Point):
#            return i
#    var_type = a.__annotations__["i"]
#    print("var_type", var_type, var_type is typing.Tuple[float, float])
    
    from dataclasses import dataclass, fields, MISSING
    @dataclass
    class Employee_dc:
        name: str
        id: int = 3
    
    print(Employee_dc.__annotations__)
    print(Employee_dc.__init__.__annotations__)
    print(fields(Employee_dc))
    
    for field in fields(Employee_dc):
        print(field.type, field.default, field.default is MISSING)
        
    # defualt factory
    # dataclasses.field(*, default=MISSING, default_factory=MISSING, repr=True, hash=None, init=True, compare=True, metadata=None)