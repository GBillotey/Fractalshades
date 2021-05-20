# -*- coding: utf-8 -*-
#import inspect
#import fractalshades.utils as fsutils
#import sys
#import os
#import copy
import math
from collections import OrderedDict
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
from PyQt5.QtWidgets import (QWidget, QAction, QLabel,
                              QMenu, QHBoxLayout, QVBoxLayout,
                             QGridLayout, QSpacerItem, QSizePolicy,
                             QGraphicsScene, QGraphicsView,
                             QGraphicsPixmapItem, QGraphicsItemGroup,
                             QGraphicsRectItem, QFrame
                             )
import PIL
#import fractalshades.gui as fsgui
#from fractalshades.gui import QDict_viewer



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


class Fractalimage_widget(QWidget):
    def __init__(self, parent, fsim_smodel): # im=None):#, xy_ratio=None):
        super().__init__(parent)
        # self.setWindowFlags(Qt.BypassGraphicsProxyWidget)
        self._model = fsim_smodel._model
        self._func_keys = fsim_smodel._keys
        self._submodel = fsim_smodel# model[func_keys]
            
#        if xy_ratio is None:
#            self._im = parent._im
#        else:
#            self._im = im
            
        
        # sets graphics scene and view
        self._scene = QGraphicsScene()
        self._group = QGraphicsItemGroup()
        self._view = QGraphicsView()
        self._scene.addItem(self._group)
        self._view.setScene(self._scene)
        self._view.setFrameStyle(QFrame.Box)
        

        
#        # always scrollbars
#        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
#        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # special sursor
        self._view.setCursor(QtGui.QCursor(Qt.CrossCursor))
        
        # sets property widget
        self._labels = QDict_viewer(self, OrderedDict([
                ("px", None), ("py", None), ("zoom", 1.)]))
        

    
    
        # sets layout
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        self._layout.addWidget(self._view, stretch=1)
        #self._layout.addStretch(1)
        self._layout.addWidget(self._labels, stretch=0)
        
        # Sets Image
        self._qim = None
        self.reset_im()
        self._view.setBackgroundBrush(
                QtGui.QBrush(QtGui.QColor(200, 200, 200),
                Qt.SolidPattern))

        
        # Zoom rectangle disabled
        self._rect = None
        self._drawing_rect = False
        self._dragging_rect = False

        # zooms anchors for wheel events - note this is only active 
        # when the 
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setAlignment(Qt.AlignCenter)

        # events filters
        self._view.viewport().installEventFilter(self)
        self._scene.installEventFilter(self)
        
        

#        self._view.setContextMenuPolicy(Qt.ActionsContextMenu)
#        self._scene.customContextMenuRequested.connect(self.useless)
#        useless_action = QAction("DoNothing", self)
#        self._scene.addAction(useless_action)
#        useless_action.triggered.connect(self.useless)

    def on_context_menu(self, event):
        menu = QMenu(self)
        NoAction = QAction("Does nothing", self)
        menu.addAction(NoAction)
        NoAction.triggered.connect(self.doesnothing)
        menu.popup(self._view.viewport().mapToGlobal(event.pos()))
        return True

    def doesnothing(self, event):
        print("voili voilou")

    @property
    def zoom(self):
        view = self._view
        pc = 100. * math.sqrt(view.transform().determinant())
        return "{0:.2f} %".format(pc)

    @property
    def xy_ratio(self):
        return 1
#        return self.parent().xy_ratio


    def reset_im(self):
        
        image_file = os.path.join(rep, "dev.png")
        with PIL.Image.open(image_file) as im:
            # im.load()
            info = im.info
            # This class is a subclass of QtGui.QImage, 
            imqt = PIL.ImageQt.ImageQt(im)
        
        # Storing the "initial" zoom info
        self.x_init = info["x"]
        self.y_init = info["y"]
        self.dx_init = info["dx"]
        self.xy_ratio_init = info["xy_ratio"]
        self.check_zoom_init()

        if self._qim is not None:
            self._group.removeFromGroup(self._qim)
        self._qim = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(imqt)) # QtGui.QImage(self._im)))
        self._qim.setAcceptHoverEvents(True)
        self._group.addToGroup(self._qim)
        self.fit_image()
        
    def check_zoom_init(self):
        print(self.x_init, self.y_init, self.dx_init, self.xy_ratio_init)
        
    def fit_image(self):
        if self._qim is None:
            return
        rect = QtCore.QRectF(self._qim.pixmap().rect())
        if not rect.isNull():
            view = self._view
            view.setSceneRect(rect)
            unity = view.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            view.scale(1 / unity.width(), 1 / unity.height())
            viewrect = view.viewport().rect()
            scenerect = view.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            view.scale(factor, factor)
            self._labels.values_update({"zoom": self.zoom})

    def eventFilter(self, source, event):
        # ref: https://doc.qt.io/qt-5/qevent.html
        if source is self._scene:
            if type(event) is QtWidgets.QGraphicsSceneMouseEvent:
                return self.on_viewport_mouse(event)
            elif type(event) is QtGui.QEnterEvent:
                return self.on_enter(event)
            elif (event.type() == QtCore.QEvent.Leave):
                return self.on_leave(event)

        elif source is self._view.viewport():
            # Catch context menu
            if type(event) == QtGui.QContextMenuEvent:
                return self.on_context_menu(event)
            elif event.type() == QtCore.QEvent.Wheel:
                return self.on_wheel(event)
            elif event.type() == QtCore.QEvent.ToolTip:
                return True

        return False

    def on_enter(self, event):
        print("enter")
        return False

    def on_leave(self, event):
        print("leave")
        return False

    def on_wheel(self, event):
        if self._qim is not None:
            view = self._view
            if event.angleDelta().y() > 0:
                factor = 1.25
            else:
                factor = 0.8
            view.scale(factor, factor)
            self._labels.values_update({"zoom": self.zoom})
        return True


    def on_viewport_mouse(self, event):

        if event.type() == QtCore.QEvent.GraphicsSceneMouseMove:
            print("viewport_mouse")
            self.on_mouse_move(event)
            return True

        elif (event.type() == QtCore.QEvent.GraphicsSceneMousePress
              and event.button() == Qt.LeftButton):
            self.on_mouse_left_press(event)
            return True

        elif (event.type() == QtCore.QEvent.GraphicsSceneMouseRelease
              and event.button() == Qt.LeftButton):
            self.on_mouse_left_release(event)
            return True

        elif (event.type() == QtCore.QEvent.GraphicsSceneMouseDoubleClick
              and event.button() == Qt.LeftButton):
            self.on_mouse_double_left_click(event)
            return True

        else:
            print("Uncatched mouse event", event.type())
            return False

    def on_mouse_left_press(self, event):
        self._drawing_rect = True
        self._dragging_rect = False
        self._rect_pos0 = event.scenePos()

    def on_mouse_left_release(self, event):
        if self._drawing_rect:
            self._rect_pos1 = event.scenePos()
            if (self._rect_pos0 == self._rect_pos1):
                self._group.removeFromGroup(self._rect)
                self._rect = None
            self._drawing_rect = False

    def on_mouse_double_left_click(self, event):
        self.fit_image()

    def on_mouse_move(self, event):
        scene_pos = event.scenePos()
        self._labels.values_update({"px": scene_pos.x(),
                                    "py": scene_pos.y()})
        if self._drawing_rect:
            self._dragging_rect = True
            self._rect_pos1 = event.scenePos()
            self.draw_rect(self._rect_pos0, self._rect_pos1)
            

    def draw_rect(self, pos0, pos1):
        # Enforce the correct ratio
        diffx = pos1.x() - pos0.x()
        diffy = pos1.y() - pos0.y()
        radius_sq = diffx ** 2 + diffy ** 2
        diffx0 = math.sqrt(radius_sq / (1. + self.xy_ratio ** 2))
        diffy0 = diffx0 * self.xy_ratio
        diffx0 = math.copysign(diffx0, diffx)
        diffy0 = math.copysign(diffy0, diffy)
        pos1 = QtCore.QPointF(pos0.x() + diffx0, pos0.y() + diffy0)

        topleft = QtCore.QPointF(min(pos0.x(), pos1.x()),
                                 min(pos0.y(), pos1.y()))
        bottomRight = QtCore.QPointF(max(pos0.x(), pos1.x()),
                                     max(pos0.y(), pos1.y()))
        rectF = QtCore.QRectF(topleft, bottomRight)
        if self._rect is not None:
            self._rect.setRect(rectF)
        else:
            self._rect = QGraphicsRectItem(rectF)
            self._rect.setPen(QtGui.QPen(QtGui.QColor("red"), 0, Qt.DashLine))
            self._group.addToGroup(self._rect)


if __name__ == "__main__":
    pass