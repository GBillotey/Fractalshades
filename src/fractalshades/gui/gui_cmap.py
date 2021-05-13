# -*- coding: utf-8 -*-
import inspect
import typing
import dataclasses


from PyQt5 import QtCore
from PyQt5.QtCore import Qt
#from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (QWidget, QAction,
                              QMenu, QHBoxLayout, QVBoxLayout, QCheckBox,
                              QLabel, QMenuBar, QToolBar, QComboBox,
                             QGridLayout, QSpacerItem, QSizePolicy,
                             QGraphicsScene, QGraphicsView,
                             QGraphicsPixmapItem, QGraphicsItemGroup,
                             QGraphicsRectItem, QFrame
                             )
from PyQt5.QtWidgets import (QMainWindow, QApplication)

from inspector import Func_inspector
import functools

import fractalshades.colors as fscolors
import numpy as np
import tempfile
import os


class Qcmap_icone(QWidget):
    def __init__(self, parent, cmap):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(20)
        self.setMaximumHeight(20)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                           QtWidgets.QSizePolicy.Expanding)
        self._cmap = cmap
        
        
    def paintEvent(self, evt):
        size = self.size()
        nx, ny = size.width(), size.height()
        image = self._cmap.output_ImageQt(nx, ny)
        QtGui.QPainter(self).drawImage(0, 0, image)
        



def getapp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_cmap_widget():
    
    gold = np.array([255, 210, 66]) / 255.
    black = np.array([0, 0, 0]) / 255.

    colors1 = np.vstack((gold[np.newaxis, :],
                         black[np.newaxis, :]))
    colors2 = np.vstack((black[np.newaxis, :],
                         gold[np.newaxis, :]))
    kinds = ["Lch", "Lch"]
    n = 100
    funcs = [lambda x: x**6, lambda x: 1.- (1. - x)**6]

    colormap = fscolors.Fractal_colormap(
            kinds, colors1, colors2, n, funcs, extent="clip")
    
    
    class Mywindow(QMainWindow):
        def __init__(self):
            super().__init__(parent=None)

            self.setWindowTitle('Testing inspector')
            tb = QToolBar(self)
            self.addToolBar(tb)
#            print_dict = QAction("print dict")
            tb.addAction("print_dict")
            
            tb.actionTriggered[QAction].connect(self.on_tb_action)
            #self.setWindowState(Qt.WindowMaximized)
            # And don't forget to call setCentralWidget to your main layout widget.
            icone =  Qcmap_icone(self, colormap) # fsgui.
            self.setCentralWidget(icone)
            self._icone = icone

        def on_tb_action(self, qa):
            print("ON ACTION qa", qa)
            


    app = getapp()
    win = Mywindow()
    win.show()
    app.exec()
    
    
    

if __name__ == "__main__":
    test_cmap_widget()