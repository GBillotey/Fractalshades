# -*- coding: utf-8 -*-
import inspect
import fractalshades.utils as fsutils
import sys
import os
import copy
import math

import inspect


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

