# -*- coding: utf-8 -*-
import inspect
import typing
import dataclasses
import math
import os
# import textwrap
#import pprint

import PIL
import functools
#import copy
#from operator import getitem, setitem
import mpmath
import threading
import ast

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
#from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal #, pyqtSlot

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QWidget,
    QAction,
    QDockWidget,
    QPushButton,
    QMenu,
    QHBoxLayout,
    QVBoxLayout,
    QCheckBox,
    QLabel,
#    QMenuBar,
#    QToolBar,
    QComboBox,
    QLineEdit,
    QStackedWidget,
    QGroupBox,
    QGridLayout,
    QSpacerItem,
    QSizePolicy,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QFrame,
    QScrollArea, 
    QPlainTextEdit,
    QColorDialog,
    QGraphicsOpacityEffect,
    QSpinBox,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem
)

from PyQt5.QtWidgets import (QMainWindow, QApplication)
#

import fractalshades as fs
import fractalshades.colors as fscolors
from fractalshades.gui.model import (
    Model,
    Func_submodel,
    Colormap_presenter,
    Presenter,
    type_name,
)

from fractalshades.gui.QCodeEditor import QCodeEditor
import fractalshades.numpy_utils.expr_parser as fs_parser

# QMainWindow
MAIN_WINDOW_CSS = """
QWidget {background-color: #646464;
        color: white;}
QMainWindow::separator {
    background: #7e7e7e;
    width: 4px; /* when vertical */
    height: 4px; /* when horizontal */
}
QMainWindow::separator:hover {
    background: #df4848;
}
"""

DOCK_WIDGET_CSS = """
QDockWidget {
    font-size: 8pt;
    font: bold;
    color: #A0A5AA;
}
QDockWidget::title {
    padding-left: 12px;
    padding-top: 3px;
    padding-bottom: 3px;
    text-align: center left;
    border-left: 1px solid #32363F;
    border-top: 1px solid #32363F;
    border-right: 1px solid #32363F;
    background: #25272C;
}
"""

# QLineEdit
PARAM_LINE_EDIT_CSS = """
QLineEdit {{
    color: white;
    background: {0};
}}
"""

# QPlainTextEdit
PLAIN_TEXT_EDIT_CSS = """
QWidget {{
    color: white;
    background: {0};
    border-radius: 2px;
}}
"""

COMBO_BOX_CSS = """
QComboBox {
    color: white;
    background: #25272C;
}
QComboBox:item {
    color: white;
    background: #25272C;
}
QComboBox:item:selected {
    color: #df4848;
    background: #25272C;
}
"""

CHECK_BOX_CSS = """
QCheckBox:indicator {
    border: 2px solid #25272C;
    background: #646464;
}
QCheckBox:indicator:pressed {
    background: #df4848;
}
QCheckBox:indicator:checked {
    background: #25272C;
}
QCheckBox:indicator:checked:pressed {
    background: #df4848;
}
"""

TABLE_WIDGET_CSS = """
QTableView {
selection-background-color: #646464;
}
QTableView::item::selected {
  border: 2px solid red;
}
"""
#
#TABLE_WIDGET_CSS_INVALID = """
#QTableView::item::selected {
#  border: 2px solid red;
#}
#"""

def getapp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def getmainwindow(win):
    """ 
    win : QWidget
    Return the QMainWindow that is in `win` ancestors list, if found.
    """
    parent = win
    while parent is not None:
        if parent.inherits('QMainWindow'):
            return parent
        parent = parent.parent()
    raise RuntimeError('Count not find QMainWindow instance.', win)


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
    
    def __init__(self, parent, func_smodel, action_setting=None):
        super().__init__(parent)
        self._submodel = func_smodel
        # Parameters and action boxes
        param_box = self.add_param_box(func_smodel)
        action_box = self.add_action_box()

        # general layout
        layout = QVBoxLayout()
        layout.addWidget(param_box, stretch=1)
        layout.addWidget(action_box)
#        layout.addStretch(1)
        self.setLayout(layout)
            
        # Connect events
        self._source.clicked.connect(self.show_func_source)
        self._params.clicked.connect(self.show_func_params)
        self._run.clicked.connect(self.run_func)
        
        # adds a binding to the image modified
        if action_setting is not None:
            (setting, keys) = action_setting
            print("*********************action_setting", action_setting)
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
        
        param_layout.addWidget(param_scrollarea)
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
        sm._func(**sm.getkwargs())
        # Run in separate thread TODO 
        # PS do NOT use QThread 
        # https://doc.qt.io/archives/qt-4.8/qthread.html#details
        # https://codereview.stackexchange.com/questions/208766/capturing-stdout-in-a-qthread-and-update-gui
#        print("#######THREAD LAUNCHING")
#        th = threading.Thread(target=sm._func, kwargs=sm.getkwargs())
#        th.start()
#        print("#######THREAD LAUNCHED")
#        th.join()
        self.func_performed.emit()

    def show_func_params(self):
        sm = self._submodel
        ce = QCodeEditor(DISPLAY_LINE_NUMBERS=True,
            HIGHLIGHT_CURRENT_LINE=True, SyntaxHighlighter=None)
        str_args = "\n".join([(k + " = " + repr(v)) for (k, v)
                              in sm.getkwargs().items()])
        ce.setPlainText(str_args)
        ce.show()

    def show_func_source(self):
        sm = self._submodel
        ce = QCodeEditor(DISPLAY_LINE_NUMBERS=True,
            HIGHLIGHT_CURRENT_LINE=True, SyntaxHighlighter=None)
        ce.setPlainText(sm.getsource())
        ce.show()
        

class Func_widget(QFrame):
    # Signal to inform the model that a parameter has been modified by the 
    # user.
    func_user_modified = pyqtSignal(object, object)

    def __init__(self, parent, func_smodel):
        super().__init__(parent)
        self._model = func_smodel._model
        self._func_keys = func_smodel._keys
        self._submodel = func_smodel# model[func_keys]
        self._widgets = dict() # Will store references to the widgets that can
                               # be programmatically updated 

        # Components and layout
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
        myFont = QtGui.QFont()
        myFont.setWeight(QtGui.QFont.ExtraBold)
        name_label.setFont(myFont)
        self._layout.addWidget(name_label, i_param, 0, 1, 1)

        # Adds a check-box for default value
        if fd[(i_param, "has_def")]:
            is_default = self._widgets[(i_param, "is_def")] = QCheckBox()
            is_default.setChecked(fd[(i_param, "is_def")])
            is_default.stateChanged.connect(functools.partial(
                self.on_user_mod, (i_param, "is_def"), is_default.isChecked))
            self._layout.addWidget(is_default, i_param, 1, 1, 1)

        # Handles Union types
        qs = QStackedWidget()
        qs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        n_uargs = fd[(i_param, "n_types")]
        if n_uargs == 0:
            utype = fd[(i_param, 0, "type")]
            print("utype", utype)
            utype_label = QLabel(type_name(utype))
            self._layout.addWidget(utype_label, i_param, 3, 1, 1)
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
            
            self._layout.addWidget(utypes_combo, i_param, 3, 1, 1)
            for utype in range(n_uargs):
                self.layout_uarg(qs, i_param, utype)
        # The displayed item of the union is denoted by "type_sel" :
        # self.layout_uarg(qs, i_param, fd[(i_param, "type_sel")])
        qs.setCurrentIndex(fd[(i_param, "type_sel")])
        self._layout.addWidget(qs, i_param, 2, 1, 1)
        self._layout.setRowStretch(i_param, 0)

        # adds a spacer at bottom
        self._layout.setRowStretch(i_param + 1, 1)

    
    def layout_uarg(self, qs, i_param, i_union):

        
        fd = self._submodel._dict
        # n_uargs = fd[(i_param, "n_types")]
        utype = fd[(i_param, i_union, "type")]
        if dataclasses.is_dataclass(utype):
            for ifield, field in enumerate(dataclasses.fields(utype)):
                self.layout_field(qs, i_param, i_union, ifield)
        else:
            uval = fd[(i_param, i_union, "val")]
#            print("UVAL", uval)
            atom_wget = atom_wget_factory(utype)(utype, uval, self._model)
            self._widgets[(i_param, i_union, "val")] = atom_wget
#            print("atom_wget", atom_wget, type(atom_wget))
            atom_wget.user_modified.connect(functools.partial(
                    self.on_user_mod, (i_param, i_union, "val"),
                    atom_wget.value))
            qs.addWidget(atom_wget)#, i_param, 3, 1, 1)
            
            if isinstance(atom_wget, Atom_Presenter_mixin):
                atom_wget.request_presenter.connect(functools.partial(
                    self.on_presenter, (i_param, i_union, "val")))
            
    
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
        """ Notify the model of modification by the user of a widget"""
        val = val_callback()
        self.func_user_modified.emit(key, val)

    def model_event_slot(self, keys, val):
        """ Handles modification of widget triggered from model """
        # Does the event impact one of my child widgets ? otherwise, return
        if keys[:-1] != self._func_keys:
            return
        key = keys[-1]
        try:
            wget = self._widgets[key]
        except KeyError:
            # Not a widget, could be a parameter notification
            return

        # Check first Atom_Mixin
        if isinstance(wget, Atom_Edit_mixin):
            wget.on_model_event(val)
        elif isinstance(wget, QCheckBox):
            wget.setChecked(val)
        elif isinstance(wget, QComboBox):
            wget.setCurrentIndex(val)
        else:
            raise NotImplementedError("Func_widget.model_event_slot {}".format(
                                      wget))

    def on_presenter(self, keys, presenter_class, wget_class):
        """ Handles creation of a parameter presenter or visibility toggling
        when clicked
        """
        if not hasattr(self, "presenters"):
            self.presenters = dict()

        varname = self._submodel._dict[(keys[0], "name")]
        register_key = "{}({})".format(presenter_class.__name__, varname)

        if register_key not in self.presenters.keys():
            # We create the Docwidget and presenter
            # parameter presenter, the only mapping key should be the class
            # name 
            mapping = {presenter_class.__name__:  self._func_keys + (keys,)}
            presenter_class(self._model, mapping, register_key)
            wget = wget_class(None, self._model._register[register_key])
            main_window = getmainwindow(self)
            dock_widget = QDockWidget(register_key, None, Qt.Window)
            dock_widget.setWidget(wget)
            dock_widget.setStyleSheet(DOCK_WIDGET_CSS)
            main_window.addDockWidget(Qt.RightDockWidgetArea, dock_widget)
            self.presenters[register_key] = dock_widget
        else:
            # Docwidget of presenter exists, we only toggles visibility
            dock_widget = self.presenters[register_key]
            toggle = not(dock_widget.isVisible())
            dock_widget.setVisible(toggle)


def atom_wget_factory(atom_type):
    if typing.get_origin(atom_type) is typing.Literal:
        return Atom_QComboBox
    elif issubclass(atom_type, fs.Fractal):
        return Atom_fractal_button
    else:
        wget_dic = {int: Atom_QLineEdit,
                    float: Atom_QLineEdit,
                    str: Atom_QLineEdit,
                    bool: Atom_QCheckBox,
                    mpmath.mpf: Atom_QPlainTextEdit, #Atom_QLineEdit,
                    QtGui.QColor: Atom_QColor,
                    fscolors.Fractal_colormap: Atom_cmap_button,
                    type(None): Atom_QLineEdit}
        return wget_dic[atom_type]
    
class Atom_Edit_mixin:
    def value(self):
        raise NotImplementedError("Subclasses should implement")

class Atom_Presenter_mixin:
    def value(self):
        raise NotImplementedError("Subclasses should implement")

class Atom_QCheckBox(QCheckBox, Atom_Edit_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__("", parent)
        self.setChecked(val)
        self._type = atom_type
        self.stateChanged.connect(self.on_user_event)
        self.setStyleSheet(CHECK_BOX_CSS)

    def value(self):
        return self.isChecked()

    def on_user_event(self):
        self.user_modified.emit()

    def on_model_event(self, val):
        self.setChecked(val)


class Atom_QColor(QPushButton, Atom_Edit_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        """
        val is given in rgb float, or rgba float (in the range [0., 1.])
        Internally we use QtGui.QColor rgb or rgba i.e. uint8 format
        """
        super().__init__("", parent)
        self._type = atom_type
        self._kind = {3: "rgb", 4: "rgba"}[len(val)]
        self._color = None
        self.update_color(QtGui.QColor(*list(
                int(channel * 255) for channel in val)))
        self.clicked.connect(self.on_user_event)

    def update_color(self, color):
        """ color: QtGui.QColor """
        if color != self._color:
            self._color = color
            self.setStyleSheet("background-color: {0};"
                               "border-color: {1};"
                               "border-style: solid;"
                               "border-width: 1px;"
                               "border-radius: 4px;".format(
                           self._color.name(), "grey"))

            if self._kind == "rgba":
                # Paint a gradient from the color with transparency to the
                # color with no transparency (rgba "a" value set to 255)
                gradient = QtGui.QLinearGradient(0, 0, 1, 0)
                gradient.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
                gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0, color.alpha()))
                gradient.setColorAt(0.1, QtGui.QColor(0, 0, 0, color.alpha()))
                gradient.setColorAt(0.9, Qt.black)
                gradient.setColorAt(1.0, Qt.black)
                effect = QGraphicsOpacityEffect(self)
                effect.setOpacity(1.)
                effect.setOpacityMask(gradient)
                self.setGraphicsEffect(effect)

            self.repaint()
            self.user_modified.emit()

    def value(self):
        c = self._color
        if self._kind == "rgb":
            ret = (c.redF(), c.greenF(), c.blueF())
        elif self._kind == "rgba":
            ret = (c.redF(), c.greenF(), c.blueF(), c.alphaF())
        return ret

    def on_user_event(self):
        colord = QColorDialog()
        colord.setOption(QColorDialog.DontUseNativeDialog)
        if self._kind == "rgba":
            colord.setOption(QColorDialog.ShowAlphaChannel)
        colord.setCurrentColor(self._color)
        colord.setCustomColor(0, self._color)
        colord.currentColorChanged.connect(self.update_color)
        old_col = self._color
        if colord.exec():
            self.update_color(colord.currentColor())
        else:
            self.update_color(old_col)

    def on_model_event(self, val):
        self.update_color(QtGui.QColor(*list(
                int(channel * 255) for channel in val)))


class Atom_QLineEdit(QLineEdit, Atom_Edit_mixin): 
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(str(val), parent)
        self._type = atom_type
        self.textChanged[str].connect(self.validate)
        self.editingFinished.connect(self.on_user_event)
        self.setValidator(Atom_Text_Validator(atom_type, model))
        self.setStyleSheet(PARAM_LINE_EDIT_CSS.format(
                       "#25272C"))
        if atom_type is type(None):
            self.setReadOnly(True)

    def value(self):
        if self._type is type(None):
            return None
        return self._type(self.text()) 

    def on_user_event(self):
        self.user_modified.emit()
    
    def on_model_event(self, val):
        self.setText(str(val))
        self.validate(self.text(), acceptable_color="#25272C")

    def validate(self, text, acceptable_color="#c8c8c8"):
        validator = self.validator()
        if validator is not None:
            ret, _, _ = validator.validate(text, self.pos())
            if ret == QtGui.QValidator.Acceptable:
                self.setStyleSheet(PARAM_LINE_EDIT_CSS.format(
                       acceptable_color))
            else:
                self.setStyleSheet(PARAM_LINE_EDIT_CSS.format(
                       "#dc4646"))


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
        self.setStyleSheet(PLAIN_TEXT_EDIT_CSS.format(
                       "#25272C"))
        
        # signals / slots
        self._validator = Atom_Text_Validator(atom_type, model)
        self.textChanged.connect(self.validate)

    def value(self):
        return self.toPlainText()

    def on_model_event(self, val):
        str_val = val
        if str_val != self.toPlainText():
            # Signals shall be blocked to avoid an infinite event loop.
            with QtCore.QSignalBlocker(self):
                self.setPlainText(str_val)
        self.validate(from_user=False)

    def validate(self, from_user=True):
        """ Sets background color according to the text validation
        """
        text = self.toPlainText()
        validator = self._validator
        if validator is not None:
            ret, _, _ = validator.validate(text, self.pos())
            if ret == QtGui.QValidator.Acceptable:
                self.setStyleSheet(PLAIN_TEXT_EDIT_CSS.format(
                       "#25272C"))
                if from_user:
                    self.user_modified.emit()
            else:
                self.setStyleSheet(PLAIN_TEXT_EDIT_CSS.format(
                       "#dc4646"))
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
        self.setCurrentIndex(val)
        
        self.setStyleSheet(COMBO_BOX_CSS)#"background:#000000")

    def value(self):
        return self.currentIndex()

    def on_user_event(self):
        self.user_modified.emit()
    
    def on_model_event(self, val):
        self.setCurrentIndex(val)


class Atom_fractal_button(QPushButton, Atom_Edit_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent)
        self._fractal = val

    def value(self):
        return self._fractal

    def on_model_event(self, val):
        pass


class Qcmap_image(QWidget):
    """ Widget of a cmap image, expanding width """
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


class Atom_cmap_button(Qcmap_image, Atom_Edit_mixin, Atom_Presenter_mixin):
    user_modified = pyqtSignal()
    request_presenter = pyqtSignal(object, object)

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent, val)
        self._cmap = None
        self.update_cmap(val)

    def update_cmap(self, cmap):
        """ cmap: fscolor.Fractal_colormap """
        if cmap != self._cmap:
            self._cmap = cmap
            self.repaint()
            # Note : we do not emit self.user_modified, this shall be done at
            # Qcmap_editor widget level
            print("CMAP MODIFIED")

    def value(self):
        return self._cmap

    def on_model_event(self, val):
        print("CMAP img model event")
        self.update_cmap(val)

    def mouseReleaseEvent(self, event):
        print("clicked")
        self.request_presenter.emit(Colormap_presenter, Qcmap_editor)


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
            # Starting or trailing carriage return are invalid
            if (val[-1] == "\n") or (val[0] == "\n"):
                return (valid[False], val, pos)
        return (valid[isinstance(casted, self._type)], val, pos)


class ColorDelegate(QStyledItemDelegate):
    def __init__(self, parent):
        """ Custom cell delegate to display / edit a colors
        parent : the QTableWidget
        """
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        dialog = QColorDialog(None) #
        dialog.setOption(QColorDialog.DontUseNativeDialog)
        dialog.setCurrentColor(index.data(Qt.BackgroundRole))
        dialog.setCustomColor(0, index.data(Qt.BackgroundRole))
        # QT doc: The returned editor widget should have Qt::StrongFocus
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
        # QT doc: After painting, you should ensure that the painter is
        # returned to the state it was supplied in when this function was
        # called.
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

#    def validate(self, index):
#        return True

class IntDelegate(QStyledItemDelegate):
    def __init__(self, parent, options):
        """ Custom cell delegate to display / edit an int
        parent : the QTableWidget
        """
        super().__init__(parent)
        self.min_val = options["min"]
        self.max_val = options["max"]

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        """
        index: PyQt5.QtCore.QModelIndex
        """
        val = index.data(Qt.DisplayRole)
        editor.setText(val)

    def setModelData(self, editor, model, index):
        """ save int val to the model"""
        val = editor.text()
        model.setData(index, val, Qt.DisplayRole)
        if self.validate(index):
            color = QtGui.QColor("#646464")
        else:
            color = QtGui.QColor("red")
        model.setData(index, color, Qt.BackgroundRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def validate(self, index):
        val = index.data(Qt.DisplayRole)
        try:
            val = int(val)
        except (TypeError, ValueError):
            return False
        return (val >= self.min_val and val <= self.max_val)


class ComboDelegate(QStyledItemDelegate):
    def __init__(self, parent, options):
        """ Custom cell delegate to display / edit a combo box
        parent : the QTableWidget
        """
        print("init combo delegate", options)
        super().__init__(parent)
        self.choices = options["choices"]

    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.addItems(self.choices)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        val = index.data(Qt.DisplayRole)
        editor.setCurrentText(val)

    def setModelData(self, editor, model, index):
        val = editor.currentText()
        model.setData(index, val, Qt.DisplayRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def validate(self, index):
        val = index.data(Qt.DisplayRole)
        return val in self.choices


class ExprDelegate(QStyledItemDelegate):
    def __init__(self, parent, options):
        """ Custom cell delegate to display / edit an expr
        parent : the QTableWidget
        """
        print("init combo delegate", options)
        super().__init__(parent)
        self.modifier = options["modifier"]

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        val = index.data(Qt.DisplayRole)
        editor.setText(val)

    def setModelData(self, editor, model, index):
        val = editor.text()
        model.setData(index, val, Qt.DisplayRole)
        if self.validate(index):
            color = QtGui.QColor("#646464")
        else:
            color = QtGui.QColor("red")
        model.setData(index, color, Qt.BackgroundRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def validate(self, index):
        val = self.modifier(index.data(Qt.DisplayRole))
        try:
            return fs_parser.acceptable_expr(ast.parse(val, mode="eval"))
        except (SyntaxError):
            return False


class Qcmap_editor(QWidget):
    """
    Widget of a cmap editor : parameters & data table
    """
    cmap_user_modified = pyqtSignal(object, object)
    
    std_flags = (Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
    unvalidated_flags = (Qt.ItemIsEnabled  | Qt.ItemIsEditable)
    locked_flags = (Qt.ItemIsEnabled)

    def __init__(self, parent, cmap_presenter):
        super().__init__(parent)        
        self._model = cmap_presenter._model
        self._mapping = cmap_presenter._mapping
        self._presenter = cmap_presenter
        self.extent_choices = self._presenter.extent_choices

        layout = QVBoxLayout()
        layout.addWidget(self.add_param_box())
        layout.addWidget(self.add_table_box())
        layout.addStretch(1)
        self.setLayout(layout)

        self._wget_n.valueChanged.connect(functools.partial(
                self.event_filter, "size"))
        self._wget_extent.currentTextChanged.connect(functools.partial(
                self.event_filter, "extent"))
        self._table.itemChanged.connect(functools.partial(
                self.event_filter, "table"))

        self.cmap_user_modified.connect(cmap_presenter.cmap_user_modified_slot)
        self._model.model_event.connect(self.model_event_slot)

    @property
    def _cmap(self):
        return self._presenter.cmap

    def add_param_box(self):
        
        param_box = QGroupBox("Cmap parameters")
        # Choice of number of lines
        self._wget_n = QSpinBox(self)
        self._wget_n.setRange(2, 256)
        # Choice of cmap "extent"
        self._wget_extent = QComboBox(self)
        self._wget_extent.addItems(self.extent_choices)
        # preview as an image
        self._preview = Qcmap_image(self, self._cmap)
        param_layout = QHBoxLayout()
        param_layout.addWidget(self._wget_n)
        param_layout.addWidget(self._wget_extent)
        param_layout.addWidget(self._preview, stretch=1)
        param_box.setLayout(param_layout)

        self.populate_param_box()
        return param_box

    def populate_param_box(self):
        self._wget_n.setValue(len(self._cmap.colors))
        val_extent = self.extent_choices.index(self._cmap.extent)
        self._wget_extent.setCurrentIndex(val_extent)
        self._preview._cmap = self._cmap
        self._preview.repaint()

    def add_table_box(self):
        table_box = QGroupBox("Cmap data")
        table_layout = QHBoxLayout()

        self._table = QTableWidget()
        # COLUMNS : colors, kinds, n, funcs=None
        self._table.setColumnCount(4)
        self.populate_table()

        self._table.setStyleSheet(TABLE_WIDGET_CSS)

        # Setup the delegates
        self._table.setItemDelegateForColumn(0, ColorDelegate(self._table))
        self._table.setItemDelegateForColumn(1, ComboDelegate(self._table,
                {"choices": ("Lch", "Lab")}))
        self._table.setItemDelegateForColumn(2, IntDelegate(self._table,
                {"min": 1, "max":256}))
        self._table.setItemDelegateForColumn(3, ExprDelegate(self._table,
                {"modifier": lambda expr: ("lambda x: " + expr)}))

        self._table.setHorizontalHeaderLabels((
                "color",
                "kind",
                "grad_pts",
                "grad_func"))
        self._table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.Stretch)

        # QAbstractItemView::SelectionMode QAbstractItemView::SingleSelection
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        
        table_layout = QHBoxLayout()
        table_box.setLayout(table_layout)
        table_layout.addWidget(self._table)
        return table_box
    
    def populate_table(self):
        # Signals shall be temporarly blocked to avoid infinite event loop.
        with QtCore.QSignalBlocker(self._table):
            n_rows = len(self._cmap.colors)
            self._table.setRowCount(n_rows)
        
            # Color editor items - Note : flags not needed as last line item is
            # never frozen.
            self.populate_column(
                col=0,
                row_range=range(n_rows),
                role=Qt.BackgroundRole,
                tab=self._cmap.colors,
                val_func=lambda v: QtGui.QColor(*list(int(255 * f) for f in v))
            )
            # kinds editor items
            self.populate_column(
                col=1,
                row_range=range(n_rows - 1),
                role=Qt.DisplayRole,
                tab=self._cmap.kinds,
                val_func=lambda v: v, # <class 'numpy.str_'> to str
                flags=self.std_flags
            )
            # grad_npts editor items
            self.populate_column(
                col=2,
                row_range=range(n_rows - 1),
                role=Qt.DisplayRole,
                tab=self._cmap.grad_npts,
                val_func=lambda v: str(v),
                flags=self.std_flags
            )
            # grad_npts editor items
            self.populate_column(
                col=3,
                row_range=range(n_rows - 1),
                role=Qt.DisplayRole,
                tab=self._cmap.grad_funcs,
                val_func=lambda v: v, # <class 'numpy.str_'> to str
                flags=self.std_flags
            )

            self.freeze_row(n_rows - 1, range(1, 4))

    def populate_column(self, col, row_range, role, tab, val_func, flags=None):
        for irow in row_range:
            val = tab[irow]
            val = val_func(val)
            item = self._table.item(irow, col)
            if item is None:
                item = QTableWidgetItem()
                self._table.setItem(irow, col, item)
            if flags is not None:
                item.setFlags(flags)
            if col == 3:
                print("populate role", role, val, type(val))
            item.setData(role, val)

    def freeze_row(self, row, col_range):
        for icol in col_range:
            # https://forum.qt.io/topic/3489/solved-how-enable-editing-to-qtablewidgetitem/2
            freezed_item = QTableWidgetItem()
            freezed_item.setFlags(self.locked_flags)
            self._table.setItem(row, icol, freezed_item)


    def event_filter(self, source, val):
        print("event", source, val)
        if source in ["size", "extent"]:
            self.cmap_user_modified.emit(source, val)
        elif source == "table":
            # val : PyQt5.QtWidgets.QTableWidgetItem
            row, col = val.row(), val.column()

            if col == 0:
                self.cmap_user_modified.emit(source, val)
                return
            delegate = self._table.itemDelegateForColumn(col)

            if delegate is None:
                raise RuntimeError()

            model = self._table.model()
            index = model.index(row, col) 
            validated = delegate.validate(index)

            if validated:
                # make sure that the normal flags are activateed (selection
                # allowed)
                with QtCore.QSignalBlocker(self._table):
                    val.setFlags(self.std_flags)
                self.cmap_user_modified.emit(source, val)
            else:
                # Invalid value, the event is not emited
                # Prevent the cell from being selected, to display the "red" bg
                # color (through flags)
                with QtCore.QSignalBlocker(self._table):
                    val.setFlags(self.unvalidated_flags)
        else:
            raise ValueError(source)

    def model_event_slot(self, keys, val):
        print("In Qcmap_editor model event filter", keys, val,
              self._presenter._mapping["Colormap_presenter"])
        if keys == self._presenter._mapping["Colormap_presenter"]:
            # Sets the value of the sub-widgets according to the smodel
            print("populate & update !")
            self.populate_param_box()
            self.populate_table()


























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
    param_user_modified = pyqtSignal(object) # (px1, py1, px2, px2)
    # zoom_params = ["x", "y", "dx", "xy_ratio"]

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
        #self._view.setFrameStyle(QFrame.Box)
        

        
#        # always scrollbars
#        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
#        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # special sursor
        self._view.setCursor(QtGui.QCursor(Qt.CrossCursor))
        
        # sets property widget
        self._labels = QDict_viewer(self,
            {"Image metadata": None, "px": None, "py": None, "zoom": None})

        # sets layout
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        self._layout.addWidget(self._view, stretch=1)
        #self._layout.addStretch(1)
        
        dock_widget = QDockWidget(None, Qt.Window)
        dock_widget.setWidget(self._labels)
        # Not closable :
        dock_widget.setFeatures(QDockWidget.DockWidgetFloatable | 
                                QDockWidget.DockWidgetMovable)
        dock_widget.setWindowTitle("Image")
        dock_widget.setStyleSheet(DOCK_WIDGET_CSS)
        parent.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)
        # self._layout.addWidget(self._labels, stretch=0)

        # Zoom rectangle disabled
        self._rect = None
        self._drawing_rect = False
        self._dragging_rect = False

        # Sets Image
        self._qim = None
        self.reset_im()

        # zooms anchors for wheel events - note this is only active 
        # when the 
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setAlignment(Qt.AlignCenter)

        # events filters
        self._view.viewport().installEventFilter(self)
        self._scene.installEventFilter(self)
        
        # Publish / subscribe signals with the submodel
        # self.zoom_user_modified.connect(self._model.)
        self._model.model_event.connect(self.model_event_slot)

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
        valid_image = True
        try:
            with PIL.Image.open(image_file) as im:
                info = im.info
                nx, ny = im.size
                # print("info debug", info["debug"])
        except FileNotFoundError:
            valid_image = False
            info = {"x": None, "y": None, "dx": None, "xy_ratio": None}
            nx = None
            ny = None

        # Storing the "initial" zoom info
        self._fractal_zoom_init = {k: info[k] for k in 
                                   ["x", "y", "dx", "xy_ratio"]}
        self._fractal_zoom_init["nx"] = nx
        self._fractal_zoom_init["ny"] = ny
        self.validate()

#        if self._qim is not None:
#            self._group.removeFromGroup(self._qim)
        for item in [self._qim, self._rect]:
            if item is not None:
                self._group.removeFromGroup(item)

        if valid_image:
            self._qim = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(
                    QtGui.QImage(image_file)))#QtGui.QImage()))#imqt)) # QtGui.QImage(self._im)))
            self._qim.setAcceptHoverEvents(True)
            self._group.addToGroup(self._qim)
            self.fit_image()
        else:
            self._qim = None
        
        self._rect = None
        self._drawing_rect = False

    @staticmethod
    def cast(val, example):
        """ Casts value to the same type as example """
        return type(example)(val)

    def check_zoom_init(self):
        """ Checks if the image 'zoom init' matches the parameters ;
        otherwise, updates """
        ret = 0
        for key in ["x", "y", "dx", "xy_ratio"]:#, "dps"]: # TODO : or precision ??
            expected = self._presenter[key]
            value = self._fractal_zoom_init[key]
            if value is None:
                ret = 2
            else:
                casted = self.cast(value, expected)
                # Send a model modification request
                self._presenter[key] = casted
                if casted != str(expected) and (ret != 2):
                    ret = 1
                    self._fractal_zoom_init[key] = casted
        return ret

    def validate(self):
        """ Sets Image metadata message """
        self.validated = self.check_zoom_init()
        message = {0: "OK, matching",
                   1: "OK, zoom params updated",
                   2: "No image found"}
        self._labels.values_update({"Image metadata": 
            message[self.validated]})

    def fit_image(self):
        if self._qim is None:
            return
        rect = QtCore.QRectF(self._qim.pixmap().rect())
        if not rect.isNull():
            #        # always scrollbars off
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            
            view = self._view
            view.setSceneRect(rect)
            unity = view.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            view.scale(1 / unity.width(), 1 / unity.height())
            viewrect = view.viewport().rect()
            scenerect = view.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            view.scale(factor, factor)
            
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
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
#        print("enter")
        return False

    def on_leave(self, event):
#        print("leave")
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
                print("cancel drawing RECT")
                self.cancel_drawing_rect()
            else:
                print("finish drawing RECT", self._rect_pos0, self._rect_pos1)
                self.publish_drawing_rect()
            self._drawing_rect = False
            
    def cancel_drawing_rect(self, dclick=False):
        if self._qim is None:
            return
        keys = ["x", "y", "dx"]
        if dclick:
            keys = ["x", "y", "dx", "xy_ratio"]
        # resets everything except the zoom ratio 
        for key in keys: #, "xy_ratio"]:
            value = self._fractal_zoom_init[key]
            if value is not None:
                # Send a model modification request
                # TODO: avoid update cancel xy_ratio 1.0 <class 'str'>
                print("update cancel", key, value, type(value))
                self._presenter[key] = value

    def publish_drawing_rect(self):
        print("------*----- publish zoom")
        if self._qim is None:
            return
#        print("publish", self._rect_pos0, self._rect_pos1)
#        print("fractal", self._presenter["fractal"])
#        print("fractal", self._presenter["image"])
        nx = self._fractal_zoom_init["nx"]
        ny = self._fractal_zoom_init["ny"]
        # new center offet in pixel
        topleft, bottomRight = self.selection_corners(self._rect_pos0,
                                                      self._rect_pos1)
        center_off_px = 0.5 * (topleft.x() + bottomRight.x() - nx)
        center_off_py = 0.5 * (ny - topleft.y() - bottomRight.y())
        dx_pix = abs(topleft.x() - bottomRight.x())
#        print("center px", center_off_px)
#        print("center py", center_off_py)
        ref_zoom = self._fractal_zoom_init.copy()
        # str -> mpf as needed
        to_mpf = {k: isinstance(self._fractal_zoom_init[k], str) for k in
                  ["x", "y", "dx"]}
        # We may need to increase the dps to hold sufficent digits
        if to_mpf["dx"]:
            ref_zoom["dx"] = mpmath.mpf(ref_zoom["dx"])
        pix = ref_zoom["dx"] / float(ref_zoom["nx"])
        with mpmath.workdps(6):
            # Sets the working dps to 10e-8 x pixel size
            ref_zoom["dps"] = int(-mpmath.log10(pix * dx_pix / nx) + 8)
        print("------*----- NEW dps from zoom", ref_zoom["dps"])

#        if ref_zoom["dps"] > mpmath.dps:
#        zoom_dps = max(ref_zoom["dps"], mpmath.mp.dps)
        with mpmath.workdps(ref_zoom["dps"]):
            for k in ["x", "y"]:
                if to_mpf[k]:
                    ref_zoom[k] = mpmath.mpf(ref_zoom[k])
    
    #        print("is_mpf", to_mpf, ref_zoom)
    
            ref_zoom["x"] += center_off_px * pix
            ref_zoom["y"] += center_off_py * pix
            ref_zoom["dx"] = dx_pix * pix
    
            
            
            #  mpf -> str (back)
            for (k, v) in to_mpf.items():
                if v:
                    if k == "dx":
                        ref_zoom[k] = mpmath.nstr(ref_zoom[k], 16)
                    else:
                        ref_zoom[k] = str(ref_zoom[k])

        for key in ["x", "y", "dx", "dps"]:
            self._presenter[key] = ref_zoom[key]
        

        
        
#        keys = ["x", "y", "dx"]
#        if dclick:
#            keys = ["x", "y", "dx", "xy_ratio"]
#        # resets everything except the zoom ratio 
#        for key in keys: #, "xy_ratio"]:
#            value = self._fractal_zoom_init[key]
#            if value is not None:
#                # Send a model modification request
#                # TODO: avoid update cancel xy_ratio 1.0 <class 'str'>
#                print("update cancel", key, value, type(value))
#                self._presenter[key] = value


    def on_mouse_double_left_click(self, event):
        self.fit_image()
        self.cancel_drawing_rect(dclick=True)

    def on_mouse_move(self, event):
        scene_pos = event.scenePos()
        self._labels.values_update({"px": scene_pos.x(),
                                    "py": scene_pos.y()})
        if self._drawing_rect:
            self._dragging_rect = True
            self._rect_pos1 = event.scenePos()
            self.draw_rect(self._rect_pos0, self._rect_pos1)
            

    def draw_rect(self, pos0, pos1):
        """ Draws the selection rectangle """
        # Enforce the correct ratio
        topleft, bottomRight = self.selection_corners(pos0, pos1)
        rectF = QtCore.QRectF(topleft, bottomRight)
        if self._rect is not None:
            self._rect.setRect(rectF)
        else:
            self._rect = QGraphicsRectItem(rectF)
            self._rect.setPen(QtGui.QPen(QtGui.QColor("red"), 0, Qt.DashLine))
            self._group.addToGroup(self._rect)

    def selection_corners(self, pos0, pos1):
        # Enforce the correct ratio
        diffx = abs(pos1.x() - pos0.x())
        diffy = abs(pos1.y() - pos0.y())
        # Enforce the correct ratio
        radius_sq = diffx ** 2 + diffy ** 2
        diffx0 = math.sqrt(radius_sq / (1. + self.xy_ratio ** 2))
        diffy0 = diffx0 * self.xy_ratio
        topleft = QtCore.QPointF(pos0.x() - diffx0, pos0.y() - diffy0)
        bottomRight = QtCore.QPointF(pos0.x() + diffx0, pos0.y() + diffy0)
        return topleft, bottomRight

    def model_event_slot(self, keys, val):
        """ A model item has been modified - will it impact the widget ? """
        # Find the mathching "mapping" - None if no match
        mapped = next((k for k, v in self._mapping.items() if v == keys), None)
        if mapped in ["image", "fractal"]:
            self.reset_im()
        elif mapped in ["x", "y", "dx", "xy_ratio", "dps"]:
            pass
        else:
            if mapped is not None:
                raise NotImplementedError("Mapping event not implemented: " 
                                          + "{}".format(mapped))






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
        self.setStyleSheet(MAIN_WINDOW_CSS)
#                "QDockWidget {background: #dadada; font: bold 14px;" #dadada;
#                              "border: 2px solid  #646464;}"
#                "QDockWidget::title {text-align: left; background: #646464;"
#                                     "padding-left: 5px;}")
        self.build_model(gui)
        self.layout()
        # self.mp_dps_used.connect(model.dps_used_slot)
    
    def build_model(self, gui):
        model = self._model = Model()
        
        # Adds the submodels
        Func_submodel(model, ("func",), gui._func, dps_var=gui._dps)

        # Adds the presenters
        mapping = {"fractal": ("func", gui._fractal),
                   "image": ("func", gui._image),
                   "x": ("func", gui._x),
                   "y": ("func", gui._y),
                   "dx": ("func", gui._dx),
                   "xy_ratio": ("func", gui._xy_ratio),
                   "dps": ("func", gui._dps)}
        Presenter(model, mapping, register_key="image")

    def layout(self):
        self.add_func_wget()
        self.add_image_wget()

    def add_func_wget(self):
        func_wget = Action_func_widget(self, self.from_register(("func",)),
            action_setting=("image_updated", 
                self.from_register("image")._mapping["image"]))
        dock_widget = QDockWidget(None, Qt.Window)
        dock_widget.setWidget(func_wget)
        # Not closable :
        dock_widget.setFeatures(QDockWidget.DockWidgetFloatable | 
                                QDockWidget.DockWidgetMovable)
        dock_widget.setWindowTitle("Parameters")
        dock_widget.setStyleSheet(DOCK_WIDGET_CSS)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)
        self._func_wget = func_wget

    def add_image_wget(self):
        mw = Image_widget(self, self.from_register("image"))
        self.setCentralWidget(mw)

    def from_register(self, register_key):
        return self._model._register[register_key]

#    def on_image_event(self):
#        print("image updated")

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
