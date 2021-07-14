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
                             QGraphicsRectItem, QFrame, QSpinBox, QGroupBox,
                             QTableWidget, QColorDialog, QStyledItemDelegate,
                             QTableWidgetItem, QStyleOptionViewItem
                             )
from PyQt5.QtWidgets import (QMainWindow, QApplication)

#from inspector import Func_inspector
#import functools

import fractalshades.colors as fscolors
import numpy as np
import functools
import tempfile
import os


class ColorDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        dialog = QColorDialog(None) #
        dialog.setOption(QColorDialog.DontUseNativeDialog)
        dialog.setCurrentColor(index.data(Qt.BackgroundRole))
        dialog.setCustomColor(0, index.data(Qt.BackgroundRole))
        # The returned editor widget should have Qt::StrongFocus
        dialog.setFocusPolicy(Qt.StrongFocus)
        dialog.setFocusProxy(parent)
        return dialog

    def setEditorData(self, editor, index):
        color = index.data(Qt.BackgroundRole)
        editor.setCurrentColor(color)

    def setModelData(self, editor, model, index):
        """ If modal QColorDialog result code is Accepted, save color"""
        if editor.result():
            color = editor.currentColor()
            model.setData(index, color, Qt.BackgroundRole)

    def paint(self, painter, option, index):
        """ Fill with BackgroundRole color + red rectangle for selection."""
        # After painting, you should ensure that the painter is returned to the
        # state it was supplied in when this function was called
        painter.save()
        selected = bool(option.state & QtWidgets.QStyle.State_Selected)
        painter.fillRect(option.rect, index.data(Qt.BackgroundRole))
        if selected:
            rect = option.rect
            rect.adjust(1, 1, -1, -1)
            pen = QtGui.QPen(Qt.red)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
        painter.restore()


class Qcmap_editor(QWidget):
    """
    Widget of a cmap image with expanding width, fixed height
    """
    def __init__(self, parent, cmap, minwidth=200, height=20):
        super().__init__(parent)
        self._cmap = cmap
        
        layout = QVBoxLayout()
        layout.addWidget(self.add_param_box())
        layout.addWidget(self.add_table_box())
        # layout.addWidget(self.add_preview_box())
        layout.addStretch(1)
        self.setLayout(layout)
        
        self._wget_n.valueChanged.connect(functools.partial(
                self.event_filter, "size"))
        
    def add_param_box(self):
        
        param_box = QGroupBox("Cmap parameters")
        # Choice of number of lines
        self._wget_n = QSpinBox(self)
        self._wget_n.setRange(2, 256)
        self._wget_n.setValue(len(self._cmap.colors))
        # Choice of cmap "extent"
        extent_choices = ["mirror", "repeat", "clip"]
        self._wget_extent = QComboBox(self)
        self._wget_extent.addItems(extent_choices)
        # preview
        self._preview = Qcmap_image(self, self._cmap)
        param_layout = QHBoxLayout()
        param_layout.addWidget(self._wget_n)#self._param_widget)
        param_layout.addWidget(self._wget_extent)#self._param_widget)
        param_layout.addWidget(self._preview, stretch=1)
        param_box.setLayout(param_layout)
        return param_box

    def add_table_box(self):
        table_box = QGroupBox("Cmap data")
        table_layout = QHBoxLayout()

        self._table = QTableWidget()
        # COLUMNS : colors, kinds, n, funcs=None
        self._table.setColumnCount(4)
        self._table.setStyleSheet('''
                QTableView {
                selection-background-color: white;
                }
                QTableView::item::selected {
                  border: 2px solid red;
                }
            ''')

        self._table.setItemDelegateForColumn(0, ColorDelegate(self._table))
        self._table.setHorizontalHeaderLabels((
                "color",
                "kind",
                "grad_pts",
                "grad_func"))
        self._table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.Stretch)
        self.populate_table()
        
        table_layout = QHBoxLayout()
        table_box.setLayout(table_layout)
        table_layout.addWidget(self._table)#self._param_widget)
        return table_box
    
    def populate_table(self):
        n_rows = len(self._cmap.colors)
        self._table.setRowCount(n_rows)

        for irow in range(n_rows):
            val = self._cmap.colors[irow, :]
            color_item = self._table.item(irow, 0)
            if color_item is None:
                color_item = QTableWidgetItem("")
            color_item.setData(Qt.BackgroundRole, QtGui.QColor(
                    *list(int(255 * f) for f in val)))
            self._table.setItem(irow, 0, color_item)

    def event_filter(self, source, val):
        print("event", source, val)


class Qcmap_image(QWidget):
    """
    Widget of a cmap image with expanding width, fixed height
    """
    cmap_modified = pyqtSignal()

    def __init__(self, parent, cmap, minwidth=200, height=20):
        super().__init__(parent)
        self._cmap = cmap
        self.setMinimumWidth(minwidth)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                           QtWidgets.QSizePolicy.Expanding)

    def paintEvent(self, evt):
        size = self.size()
        nx, ny = size.width(), size.height()
        QtGui.QPainter(self).drawImage(0, 0, self._cmap.output_ImageQt(nx, ny))





def getapp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    return app



def test_cmap_widget():
    
    gold = np.array([255, 210, 66]) / 255.
    black = np.array([0, 0, 0]) / 255.

    colors = np.vstack((gold[np.newaxis, :],
                         black[np.newaxis, :],
                         gold[np.newaxis, :]))
#    colors2 = np.vstack((black[np.newaxis, :],
#                         gold[np.newaxis, :]))
    kinds = ["Lch", "Lch"]
    n = 100
    funcs = [lambda x: x**6, lambda x: 1.- (1. - x)**6]

    colormap = fscolors.Fractal_colormap(
            colors, kinds, n, funcs, extent="clip")
    print("probes", colormap._probes)
    
    
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
            icone =  Qcmap_editor(self, colormap) # fsgui.
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