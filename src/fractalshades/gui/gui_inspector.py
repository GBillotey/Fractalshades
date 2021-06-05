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





class Inspector(QWidget):
    func_arg_changed = pyqtSignal()
    
    def __init__(self, parent, func):
        super().__init__(parent)
        self._func = func
        self._fi = Func_inspector(func)
        self._layout = QGridLayout(self)
        self.layout()
        
    
    def layout(self):
        fd = self._fi.func_dict
        n_params = fd["n_params"]
        for i_param in range(n_params):
            self.layout_param(i_param)
            
    def layout_param(self, i_param):
        fd = self._fi.func_dict
        
        name = fd[(i_param, "name")]
        name_label = QLabel(name)
        myFont=QtGui.QFont()#QtGui.QFont()
        myFont.setWeight(QtGui.QFont.ExtraBold)
        name_label.setFont(myFont)
        self._layout.addWidget(name_label, i_param, 0, 1, 1)

        # Adds a check-box for default
        is_default = QCheckBox("(default)", self)
        is_default.setChecked(fd[(i_param, "is_def")])
        is_default.stateChanged.connect(functools.partial(self.slot, 
                                        (i_param, "is_def")))
        self._layout.addWidget(is_default, i_param, 1, 1, 1)

        # Handles Union types
        n_uargs = fd[(i_param, "n_types")]
        if n_uargs == 0:
            utype = fd[(i_param, "types")][0]
            utype_label = QLabel(utype.__name__)
            utype_label.setFont(QtGui.QFont("Times", italic=True))
            self._layout.addWidget(utype_label, i_param, 2, 1, 1)
        else:
            utypes = fd[(i_param, "types")]
            utypes_combo = QComboBox()
            utypes_combo.addItems(t.__name__ for t in utypes)
            utypes_combo.setFont(QtGui.QFont("Times", italic=True))
            self._layout.addWidget(utypes_combo, i_param, 2, 1, 1)
        
        # adds a spacer at bottom
#        spacer = QSpacerItem(0, 0,
#                             hPolicy=QtWidgets.QSizePolicy.Minimum,
#                             vPolicy=QtWidgets.QSizePolicy.Expanding)
        self._layout.setRowStretch(i_param + 1, 1.)
            
        pass
        # 
    
    def layout_uarg(self, i_param, i_union):
        fd = self._fi.func_dict
        pass
    
    def layout_field(self, i_param, i_union, ifield):
        fd = self._fi.func_dict
        pass
    
    def reset_layout(self):
        """ Delete every item in self._layout """
        for i in reversed(range(self._layout.count())): 
            w = self._layout.itemAt(i).widget()
            if w is not None:
                w.setParent(None)
                # Alternative deletion instruction :
                # w.deleteLater() 

    def slot(self, val, key):
        print("local slot", key, val, type(key),  type(val))


def getapp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    return app



def test_Inspector_widget():
    
    def f_atomic(x: int=1, y: str="aa", z:float=1.0):
        pass 

    def f_union(x: int=1, y: str="aa", z: typing.Union[float, str]=1.0):
        pass 
    
    func = f_union
    
    
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
            ins =  Inspector(self, func) # fsgui.
            self.setCentralWidget(ins)
            self._ins = ins

        def on_tb_action(self, qa):
            print("qa", qa)
            d = self._ins._fi.func_dict
            for k, v in d.items():
                print(k, " --> ", v)


    app = getapp()
    win = Mywindow()
    win.show()
    app.exec()
    
    
    

if __name__ == "__main__":
    test_Inspector_widget()