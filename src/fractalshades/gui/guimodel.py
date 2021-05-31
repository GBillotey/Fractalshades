# -*- coding: utf-8 -*-
import inspect
import typing
import dataclasses
import math
import os
import textwrap

import PIL
import functools
import copy
from operator import getitem, setitem
import mpmath

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
#from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (QWidget, QAction, QDockWidget, QPushButton,
                              QMenu, QHBoxLayout, QVBoxLayout, QCheckBox,
                              QLabel, QMenuBar, QToolBar, QComboBox,
                              QLineEdit, QStackedWidget, QGroupBox,
                             QGridLayout, QSpacerItem, QSizePolicy,
                             QGraphicsScene, QGraphicsView,
                             QGraphicsPixmapItem, QGraphicsItemGroup,
                             QGraphicsRectItem, QFrame, QScrollArea, 
                             QPlainTextEdit
                             )
from PyQt5.QtWidgets import (QMainWindow, QApplication)
#

import fractalshades as fs
from model import (Model, Fractal_submodel, View_submodel, Func_submodel,
                   type_name, Image_presenter)

from QCodeEditor import QCodeEditor



#from viewer import Image_widget

def getapp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    return app

class MinimizedStackedWidget(QStackedWidget):
    def sizeHint(self):
        return self.currentWidget().sizeHint()
    def minimumSizeHint(self):
        return self.currentWidget().sizeHint()
#{
#  QSize sizeHint() const override
#  {
#    return currentWidget()->sizeHint();
#  }
#
#  QSize minimumSizeHint() const override
#  {
#    return currentWidget()->minimumSizeHint();
#  }
#};
class Action_func_widget(QFrame):#Widget):#QWidget):
    """
    A Func_widget with parameters & actions group
    """
    func_performed = pyqtSignal()

#    def __init__(self, atom_type, model):
#        super().__init__()
#        self._model = model
#        self._type = atom_type
#        if atom_type is mpmath.ctx_mp_python.mpf:
#            self.mp_dps_used.connect(functools.partial(
#                    model.setting_modified, "dps"))
    
    
    def __init__(self, parent, func_smodel, action_setting=None):#model, func_keys):
        super().__init__(parent)
        self._submodel = func_smodel
        # Parameters and action boxes
        param_box = self.add_param_box(func_smodel)
        action_box = self.add_action_box()

        # general layout
        layout = QVBoxLayout()
        layout.addWidget(param_box)
        layout.addWidget(action_box)
        layout.addStretch(1)
        self.setLayout(layout)
            
        # Connect events
        self._source.clicked.connect(self.show_func_source)
        self._params.clicked.connect(self.show_func_params)
        self._run.clicked.connect(self.run_func)
        
        # adds a binding to the image modified
        if action_setting is not None:
            (setting, keys) = action_setting
            model = func_smodel._model
            model.declare_setting(setting, keys)
            self.func_performed.connect(functools.partial(
                model.setting_touched, setting))

    def add_param_box(self, func_smodel):
        self._param_widget = Func_widget(self, func_smodel)
        param_box = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        param_scrollarea = QScrollArea(self)
        param_scrollarea.setWidget(self._param_widget)
        param_scrollarea.setWidgetResizable(True)
        
        param_layout.addWidget(param_scrollarea)#self._param_widget)
        param_box.setLayout(param_layout)
        self.set_border_style(param_box)
        return param_box

    def add_action_box(self):
        self._source = QPushButton("Show source")
        self._params = QPushButton("Show params")
        self._run = QPushButton("Run")
        action_box = QGroupBox("Actions")
        action_layout = QHBoxLayout()
        action_layout.addWidget(self._source)
        action_layout.addWidget(self._params)
        action_layout.addWidget(self._run)
        action_box.setLayout(action_layout)
        self.set_border_style(action_box)
        return action_box
    
    def set_border_style(self, gb):
        """ adds borders to an action box"""
        gb.setStyleSheet(
            "QGroupBox{border:1px solid #646464;"
                + "border-radius:5px;margin-top: 1ex;}"
            + "QGroupBox::title{subcontrol-origin: margin;"
                + "subcontrol-position:top left;" #padding:-6 3px;"
                + "left: 15px;}")# ;

    def run_func(self):
        sm = self._submodel
        print(sm.getkwargs())
        sm._func(**sm.getkwargs())
        self.func_performed.emit()

    def show_func_params(self):
        sm = self._submodel
        print(sm.getkwargs())

    def show_func_source(self):
        sm = self._submodel
        print(sm.getsource())
        ce = QCodeEditor(DISPLAY_LINE_NUMBERS=True,
            HIGHLIGHT_CURRENT_LINE=True, SyntaxHighlighter=None)
        ce.setPlainText(sm.getsource())
        ce.show()
        

class Func_widget(QFrame):#Widget):#QWidget):
    # Signal to inform the model that a parameter has been modified by the 
    # user.
    # item_mod_evt = pyqtSignal(tuple, object)
    func_user_modified = pyqtSignal(object, object)

    def __init__(self, parent, func_smodel):#model, func_keys):
        super().__init__(parent)
        
#        self.setFrameStyle(QFrame.Box )#| QFrame.Raised);
#        self.setLineWidth(1);
        
        self._model = func_smodel._model
        self._func_keys = func_smodel._keys
        self._submodel = func_smodel# model[func_keys]
        self._widgets = dict() # Will store references to the widgets that can
                               # be programmatically updated 
        
        # Compoenents and layout
        self._layout = QGridLayout(self)
        self.layout()

        # Publish / subscribe signals with the submodel
        self.func_user_modified.connect(self._submodel.func_user_modified_slot)
        self._model.model_event.connect(self.model_event_slot)

    def layout(self):
        fd = self._submodel._dict
        print("fd", fd)
        n_params = fd["n_params"]
        for i_param in range(n_params):
            self.layout_param(i_param)

    def layout_param(self, i_param):
        fd = self._submodel._dict
        
        name = fd[(i_param, "name")]
        name_label = QLabel(name)
        myFont = QtGui.QFont()#QtGui.QFont()
        myFont.setWeight(QtGui.QFont.ExtraBold)
        name_label.setFont(myFont)
        self._layout.addWidget(name_label, i_param, 0, 1, 1)

        # Adds a check-box for default
        if fd[(i_param, "has_def")]:
            is_default = self._widgets[(i_param, "is_def")] = QCheckBox()#"(default)", self)
    #        is_default.setFont(QtGui.QFont("Times", italic=True))
            is_default.setChecked(fd[(i_param, "is_def")])
            is_default.stateChanged.connect(functools.partial(
                self.on_user_mod, (i_param, "is_def"), is_default.isChecked))
            self._layout.addWidget(is_default, i_param, 1, 1, 1)

        # Handles Union types
        qs = QStackedWidget()# MinimizedStackedWidget()
        qs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        n_uargs = fd[(i_param, "n_types")]
        if n_uargs == 0:
            utype = fd[(i_param, 0, "type")]
            print("utype", utype)
            utype_label = QLabel(type_name(utype))
            self._layout.addWidget(utype_label, i_param, 2, 1, 1)
            self.layout_uarg(qs, i_param, 0)
        else:
            utypes = [fd[(i_param, utype, "type")] for utype in range(n_uargs)]
            utypes_combo = self._widgets[(i_param, "type_sel")] = QComboBox()
            self._widgets[(i_param, 'qs_type_sel')] = utypes_combo
            utypes_combo.addItems(type_name(t) for t in utypes)
            utypes_combo.setCurrentIndex(fd[(i_param, "type_sel")])
            utypes_combo.activated.connect(functools.partial(
                self.on_user_mod, (i_param, "type_sel"),
                utypes_combo.currentIndex))
            # Connect to the QS
            utypes_combo.currentIndexChanged[int].connect(qs.setCurrentIndex)
            # utypes_combo.activated.connect(qs.setCurrentIndex)
            
            self._layout.addWidget(utypes_combo, i_param, 2, 1, 1)
            for utype in range(n_uargs):
                self.layout_uarg(qs, i_param, utype)
        # The displayed item of the union is denoted by "type_sel" :
        # self.layout_uarg(qs, i_param, fd[(i_param, "type_sel")])
        qs.setCurrentIndex(fd[(i_param, "type_sel")])
        self._layout.addWidget(qs, i_param, 3, 1, 1)
        self._layout.setRowStretch(i_param, 0)
#        label = QGroupBox()#self)
#        label.add
#        label.setTitle("Hello World")
#        label.setAttribute(Qt.WA_TranslucentBackground)
#        self._layout.addWidget(label, 2*i_param, 0, 1, 3)

        # adds a spacer at bottom
        self._layout.setRowStretch(i_param + 1, 1)
            
        pass
        # 
    
    def layout_uarg(self, qs, i_param, i_union):

        
        fd = self._submodel._dict
        # n_uargs = fd[(i_param, "n_types")]
        utype = fd[(i_param, i_union, "type")]
        if dataclasses.is_dataclass(utype):
            for ifield, field in enumerate(dataclasses.fields(utype)):
                self.layout_field(qs, i_param, i_union, ifield)
        else:
            uval = fd[(i_param, i_union, "val")]
            print("UVAL", uval)
            atom_wget = atom_wget_factory(utype)(utype, uval, self._model)
            self._widgets[(i_param, i_union, "val")] = atom_wget
            print("atom_wget", atom_wget, type(atom_wget))
            atom_wget.user_modified.connect(functools.partial(
                    self.on_user_mod, (i_param, i_union, "val"),
                    atom_wget.value))
            qs.addWidget(atom_wget)#, i_param, 3, 1, 1)
            
    
#    def layout_field(self, qs, i_param, i_union, ifield):
#        fd = self._submodel.func_dict
#        pass
    
    def reset_layout(self):
        """ Delete every item in self._layout """
        for i in reversed(range(self._layout.count())): 
            w = self._layout.itemAt(i).widget()
            if w is not None:
                w.setParent(None)
                # Alternative deletion instruction :
                # w.deleteLater() 

    def on_user_mod(self, key, val_callback, *args):
        print("*args", args)
        val = val_callback()
        print("item evt",  key, val, type(val))
        self.func_user_modified.emit(key, val)

    def model_event_slot(self, keys, val):
        # Does the event impact one of my subwidgets ? otherwise, retiurn
        if keys[:-1] != self._func_keys:
            return
        key = keys[-1]
        print("IN MY Wideget, I KNOW has been modified", key, val)
        wget = self._widgets[key]
        print("with associated Widget", wget)

        # check first Atom_Mixin
        if isinstance(wget, Atom_Edit_mixin):
            wget.on_model_event(val)
        elif isinstance(wget, QCheckBox):
            wget.setChecked(val)
        elif isinstance(wget, QComboBox):
            wget.setCurrentIndex(val)
        else:
            raise NotImplementedError("Func_widget.model_event_slot {}".format(
                                      wget))


def atom_wget_factory(atom_type):
    if typing.get_origin(atom_type) is typing.Literal:
        return Atom_QComboBox
    elif issubclass(atom_type, fs.Fractal):
        return Atom_fractal_button
    else:
        wget_dic = {int: Atom_QLineEdit,
                    float: Atom_QLineEdit,
                    str: Atom_QLineEdit,
                    mpmath.mpf: Atom_QPlainTextEdit, #Atom_QLineEdit,
                    type(None): Atom_QLineEdit}
        return wget_dic[atom_type]
    
class Atom_Edit_mixin:
    pass

class Atom_QLineEdit(QLineEdit, Atom_Edit_mixin): 
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(str(val), parent)
        self._type = atom_type
        self.textChanged[str].connect(self.validate)
        self.editingFinished.connect(self.on_user_event)
        self.setValidator(Atom_Text_Validator(atom_type, model))
        if atom_type is type(None):
            self.setReadOnly(True)

    def value(self):
        if self._type is type(None):
            return None
        # should we do this ??? or rather in the model
        return self._type(self.text()) 

    def on_user_event(self):
        print("ATOMIC UPDATED from user")
        self.user_modified.emit()
    
    def on_model_event(self, val):
        print("ATOMIC UPDATED from model", val, type(val))
        self.setText(str(val))
        self.validate(self.text(), acceptable_color="#ffffff")

    def validate(self, text, acceptable_color="#c8c8c8"):
        validator = self.validator()
        if validator is not None:
            ret, _, _ = validator.validate(text, self.pos())
            if ret == QtGui.QValidator.Acceptable:
                self.setStyleSheet("background-color: {}".format(
                       acceptable_color))
            else:
                self.setStyleSheet("background-color: #dc4646")


class Atom_QPlainTextEdit(QPlainTextEdit, Atom_Edit_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(str(val), parent)
        self._type = atom_type
        self.setStyleSheet("border: 1px solid  lightgrey")
        # self.setMaximumBlockCount(1)
        # Wrapping parameters
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth) 
        self.setWordWrapMode(QtGui.QTextOption.WrapAnywhere)
        # signals / slots
        self._validator = Atom_Text_Validator(atom_type, model)
        self.textChanged.connect(self.validate)

    def value(self):
        return self.toPlainText()

#    @staticmethod
#    def val_to_str(val):
#        return str(val)

    def on_model_event(self, val):
        """ 
        """
        print("ATOMIC UPDATED from model", val, type(val))
        # Signals shall be blocked to avoid an infinite event loop.
        str_val = val # self.val_to_str(val)
        if str_val != self.toPlainText():
            blocker = QtCore.QSignalBlocker(self)
            self.setPlainText(str_val)
            blocker.unblock()
        self.validate(from_user=False)

    def validate(self, from_user=True):
        """ Sets background color according to the text validation
        """
        text = self.toPlainText()
        validator = self._validator
        if validator is not None:
            ret, _, _ = validator.validate(text, self.pos())
            if ret == QtGui.QValidator.Acceptable:
                self.setStyleSheet("background-color: #ffffff;"
                    + "border: 1px solid  lightgrey")
                if from_user:
                    self.user_modified.emit()
            else:
                self.setStyleSheet("background-color: #dc4646;"
                    + "border: 1px solid  lightgrey")
            cursor = QtGui.QTextCursor(self.document())
            cursor.movePosition(QtGui.QTextCursor.End)

    def paintEvent(self, event):
        """ Adjust widget size to its text content
        ref: https://doc.qt.io/qt-5/qplaintextdocumentlayout.html
        """
        doc = self.document()
        nrows = doc.lineCount()
        row_height = QtGui.QFontMetricsF(self.font()).lineSpacing()
        margins = (self.contentsMargins().top()
                   + self.contentsMargins().bottom()
                   + 2 * doc.rootFrame().frameFormat().margin()
                   + 2)
        doc_height = int(row_height * nrows + margins)
        if self.height() != doc_height:
            self.adjust_size(doc_height)
        else:
            super().paintEvent(event)

    def adjust_size(self, doc_height):
        """ Auto-adjust the text edit to its wrapped content
        """
        self.setMaximumHeight(doc_height)
        self.setMinimumHeight(doc_height)
        self.updateGeometry()

    def sizeHint(self):
        return QtCore.QSize(self.width(), self.height())


class Atom_QComboBox(QComboBox, Atom_Edit_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent)
        self._type = atom_type
        self._choices = typing.get_args(atom_type)
        self.currentTextChanged.connect(self.on_user_event)
        self.addItems(str(c) for c in self._choices)
        self.setCurrentIndex(val) #self.findText(val))

    def value(self):
        return self.currentIndex()

    def on_user_event(self):
        print("ATOMIC UPDATED from user")#, val, type(val))
        self.user_modified.emit()
    
    def on_model_event(self, val):
        print("ATOMIC UPDATED from model", val, type(val))
        self.setCurrentIndex(val) #self.findText(val))

class Atom_fractal_button(QPushButton, Atom_Edit_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent)
        self._fractal = val

    def value(self):
        return self._fractal


class Atom_Text_Validator(QtGui.QValidator):
    mp_dps_used = pyqtSignal(int)

    def __init__(self, atom_type, model):
        super().__init__()
        self._model = model
        self._type = atom_type
        if atom_type is mpmath.ctx_mp_python.mpf:
            self.mp_dps_used.connect(functools.partial(
                    model.setting_modified, "dps"))

    def validate(self, val, pos):
        print("validate", val, pos, type(val), self._type)
        valid = {True: QtGui.QValidator.Acceptable,
                 False: QtGui.QValidator.Intermediate}

        if self._type is type(None):
            return (valid[val == "None"], val, pos)

        try:
            casted = self._type(val)
        except ValueError:
            return (valid[False], val, pos)

        if self._type is mpmath.ctx_mp_python.mpf:
            needed_dps = len(val)
            # Trailing carriage return are invalid
            if val[-1] == "\n":
                return (valid[False], val, pos)
            # Automatically correct the dps 'in the model' to hold at least
            # this value
            if self._model.setting("dps") < needed_dps:
                self.mp_dps_used.emit(needed_dps)

        return (valid[isinstance(casted, self._type)], val, pos)


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


class Image_widget(QWidget):
    def __init__(self, parent, view_presenter): # im=None):#, xy_ratio=None):
        super().__init__(parent)
        # self.setWindowFlags(Qt.BypassGraphicsProxyWidget)
        self._model = view_presenter._model
        self._mapping = view_presenter._mapping
        self._presenter = view_presenter# model[func_keys]
            
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
        self._labels = QDict_viewer(self,
                                    {"px": None, "py": None, "zoom": 1.})

        # sets layout
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        self._layout.addWidget(self._view, stretch=1)
        #self._layout.addStretch(1)
        self._layout.addWidget(self._labels, stretch=0)

        # Sets Image
        self._qim = None
        self.reset_im()

        
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
        return self._presenter["xy_ratio"]

#        return self.parent().xy_ratio


    def reset_im(self):
        image_file = os.path.join((self._presenter["fractal"]).directory, 
                                  self._presenter["image"] + ".png")
        try:
            with PIL.Image.open(image_file) as im:
                # im.load()
                info = im.info
        except FileNotFoundError:
            info = {"x": None, "y": None, "dx": None, "xy_ratio": None}
            # This class is a subclass of QtGui.QImage, 
            #imqt = PIL.ImageQt.ImageQt(im)

        # Storing the "initial" zoom info
        self.x_init = info["x"]
        self.y_init = info["y"]
        self.dx_init = info["dx"]
        self.xy_ratio_init = info["xy_ratio"]
        self.validate()

        if self._qim is not None:
            self._group.removeFromGroup(self._qim)
        self._qim = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(
                QtGui.QImage(image_file)))#QtGui.QImage()))#imqt)) # QtGui.QImage(self._im)))
        self._qim.setAcceptHoverEvents(True)
        self._group.addToGroup(self._qim)
        self.fit_image()
        
    def check_zoom_init(self):
        for (value, expected) in zip(
                (self.x_init, self.y_init, self.dx_init, self.xy_ratio_init),
                list(self._presenter[key] for key in [
                        "x", "y", "dx", "xy_ratio"])):
            if value != expected:
                return False
            print("check", value, expected)
        return True

    def validate(self):
        """ Sets bg color according to bool"""
        self.validated = self.check_zoom_init()
        color = {True: (200, 200, 200),
                 False: (220, 70, 70)}
        self._view.setBackgroundBrush(
                QtGui.QBrush(QtGui.QColor(*color[self.validated]),
                Qt.SolidPattern))
#        print(self.x_init, self.y_init, self.dx_init, self.xy_ratio_init)
        
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
            # print("viewport_mouse")
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
            # print("Uncatched mouse event", event.type())
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






#def getapp():
#    app = QtCore.QCoreApplication.instance()
#    if app is None:
#        app = QApplication([])
#    return app
class Fractal_MainWindow(QMainWindow):
    # copy paste elsewhere...
    # mp_dps_used = pyqtSignal(int)

    
    def __init__(self, gui):
        super().__init__(parent=None)
        self.build_model(gui)
        self.layout()
        # self.mp_dps_used.connect(model.dps_used_slot)
    
    def build_model(self, gui):
        model = self._model = Model()
        
        # Adds the submodels
#        Fractal_submodel(model, tuple(["fractal"]), gui._fractal)
#        View_submodel(model, tuple(["view"]), gui._fractal)

        Func_submodel(model, ("func",), gui._func, dps_var=gui._dps)

        # Adds the presenters
        mapping = {"fractal": ("func", gui._fractal),
                   "image": ("func", gui._image),
                   "x": ("func", gui._x),
                   "y": ("func", gui._y),
                   "dx": ("func", gui._dx),
                   "xy_ratio": ("func", gui._xy_ratio),
                   "dps": ("func", gui._dps)}
        Image_presenter(model, mapping, register_key="image")

    def layout(self):
        self.add_func_wget()
        self.add_image_wget()
        
    def add_func_wget(self):
        func_wget = Action_func_widget(self, self.from_register(("func",)),
            action_setting=("image_updated",
                            self.from_register("image")["image"]))
        dock_widget = QDockWidget(None, Qt.SubWindow)
        dock_widget.setWidget(func_wget)
        dock_widget.setWindowTitle("Parameters")
        dock_widget.setStyleSheet(
                "QDockWidget {color: white; font: bold 14px;"
                    + "border: 2px solid  #646464;}"
                + "QDockWidget::title {text-align: left; background: #646464;"
                    + "padding-left: 5px;}")
        self.addDockWidget(Qt.RightDockWidgetArea, dock_widget)
        self._func_wget = func_wget

    def add_image_wget(self):
        mw = Image_widget(self, self.from_register("image"))
        self.setCentralWidget(mw)

    def from_register(self, register_key):
        return self._model._register[register_key]
    
    def on_image_event(self):
        print("image updated")

#        self.setWindowTitle('Fractalshades')
#        tb = QToolBar(self)
#        self.addToolBar(tb)
##            print_dict = QAction("print dict")
#        tb.addAction("print_dict")
#        
#        # tb.actionTriggered[QAction].connect(self.on_tb_action)
#        tb.actionTriggered.connect(self.on_tb_action)
#        #self.setWindowState(Qt.WindowMaximized)
#        # And don't forget to call setCentralWidget to your main layout widget.
#         # fsgui.
#
#        wget = Action_func_widget(self, func_smodel)
#        self._wget = wget
#        
##            im = os.path.join("/home/geoffroy/Pictures/math/github_fractal_rep/Fractal-shades/tests/images_REF",
##                      "test_M2_antialias_E0_2.png")
##            mw =  Image_widget(self, im)
#        
##            main_frame = QFrame(self)
##            main_frame.setFixedSize(800, 800)
#        
#        dock_widget = QDockWidget(None, Qt.SubWindow)
#        dock_widget.setWidget(wget)
#        dock_widget.setWindowTitle(func.__name__)
#        dock_widget.setStyleSheet(
#            "QDockWidget {color: white; font: bold 14px;"
#                + "border: 2px solid  #646464;}"
#            + "QDockWidget::title {text-align: left; background: #646464;"
#                + "padding-left: 5px;}");
#        
#        # self.setCentralWidget(mw)
#        
#        self.addDockWidget(Qt.RightDockWidgetArea, dock_widget)
#        self._wget = wget
#        # self.setFixedSize(800, 800)
#
#    def on_tb_action(self, qa):
#        print("qa", qa)
#        d = self._wget._submodel._dict
#        for k, v in d.items():
#            print(k, " --> ", v)
    
    
#
#if __name__ == "__main__":
#    test_Inspector_widget()
class Fractal_GUI:
    def __init__(self, func):
        """
        *func* callable with signature (fractal, **kwargs). It shall
               provide 'type hints' that are allowed by Func-widget. It will be
               displayed interactively 
        """
#        self._fractal = fractal
        self._func = func
        param_names = inspect.signature(func).parameters.keys()
        param0 = next(iter(param_names))
        self._fractal = param0
#        self._fractal = inspect.signature(func).parameters.values().next()
        print("_fractal", self._fractal)
#        self._view = view

    def connect_image(self, image_param="file_prefix"):
        self._image = image_param

    def connect_mouse(self, x="x", y="y", dx="dx", xy_ratio="xy_ratio",
                      dps="dps"):
        """
        Connect specific parameters from self._view to the self._func kwargs
        outputs is a list of 3 parameters names from func
        inputs is a list of 1 parameters names from func
        """
        self._x, self._y, self._dx = x, y, dx
        self._xy_ratio, self._dps = xy_ratio, dps

    def show(self):
        app = getapp()
        win = Fractal_MainWindow(self)
#        win = Mywindow()
        win.show()
        app.exec()
        