# -*- coding: utf-8 -*-
import inspect
import typing
import math
import os
import sys
import traceback
import copy
import datetime
import time
import logging
# import textwrap
#import pprint
import pickle

import PIL
import functools
#import copy
#from operator import getitem, setitem
import mpmath
import threading
import ast

if sys.version_info < (3, 9):
# See :
# https://discuss.python.org/t/deprecating-importlib-resources-legacy-api/11386/24
    import importlib_resources
else:
    import importlib.resources as importlib_resources 

import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import Qt
#from PyQt5.QtGui import QIcon
from PyQt6.QtCore import (
     pyqtSignal,
     pyqtSlot,
     QTimer,
     QPropertyAnimation
)

from PyQt6 import QtWidgets, QtGui
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QDialog,
    QInputDialog,
    QDockWidget,
    QPushButton,
    QMenu,
    QHBoxLayout,
    QVBoxLayout,
    QCheckBox,
    QLabel,
    QStatusBar,
#    QMenuBar,
#    QToolBar,
    QToolButton,
    QComboBox,
    QLineEdit,
    QStackedWidget,
    QGroupBox,
    QTextEdit,
    QMessageBox,
    QFileDialog,
    QGridLayout,
#    QSpacerItem,
    QSizePolicy,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsLineItem,
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


import fractalshades as fs
import fractalshades.colors
import fractalshades.settings
from fractalshades.gui import (
    separator,
    collapsible_separator
)
from fractalshades.gui.model import (
    Model,
    Func_submodel,
    Colormap_presenter,
    Lighting_presenter,
    Fractal_presenter,
    Presenter,
    type_name,
    typing_litteral_choices,
)
from fractalshades.gui.QCodeEditor import Fractal_code_editor
import fractalshades.numpy_utils.expr_parser as fs_parser


logger = logging.getLogger(__name__)


# Setting allocation limit for QImageReader - to allow displaying larger image
# in the GUI 
# setAllocationLimit() is  currently not wrapped in pyQt6 implementation
# as of 2022.08.07 ; see: 
# https://doc.qt.io/qt-6/qimagereader.html#setAllocationLimit
# https://stackoverflow.com/questions/71458968/pyqt6-how-to-set-allocation-limit-in-qimagereader
def QImageReader_setAllocationLimit(mblimit: int):
    # Note: The memory requirements are calculated for a minimum of 32 bits per
    # pixel, since Qt will typically convert an image to that depth when it is
    # used in GUI. This means that the effective allocation limit is
    # significantly smaller than mbLimit when reading 1 bpp and 8 bpp images.
    os.environ['QT_IMAGEIO_MAXALLOC'] = str(mblimit)
QImageReader_setAllocationLimit(fs.settings.GUI_image_Mblimit)


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

GROUP_BOX_CSS = """
QGroupBox{{
    border:1px solid {0};
    border-radius:5px;margin-top: 1ex;
}}
QGroupBox::title{{
    subcontrol-origin: margin;
    subcontrol-position:top left;
    left: 15px;
}}
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

## QSpinBox
SPINBOX_CSS = """
QSpinBox{{
    background-color : {0};
}}
"""


CHECK_BOX_CSS = """
QCheckBox::indicator:unchecked {
    border: 1px solid #25272c;
}
"""

# https://stackoverflow.com/questions/28255641/how-to-style-qtableview-css
TABLE_WIDGET_CSS = """
QTableView {
    selection-background-color: #25272c;
}
QTableView::item 
{   
    background: #25272c;        
}
QTableView::item::selected {
    border: 2px solid red;
    border-radius: 2px;
}
QHeaderView::section { background-color: #646464 }
QTableCornerButton::section { background-color: #646464 }
"""

STATUS_BAR_CSS = """
QStatusBar {
background: #7e7e7e;
}
QStatusBar::item {
background: #646464;
}
QStatusBar QLabel {
margin: 2px;
border: 0;
background: #646464;
}
"""

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

class _Pixmap_figure:
    def __init__(self, img):
        """
        This class is a wrapper that can be used to redirect a Fractal_plotter
        output, for instance when generating the documentation.
        """
        self.img = img

    def save_png(self, im_path):
        self.img.save(im_path, format="PNG")

class Calc_status_bar(QStatusBar):
    
    def __init__(self, func_model):
        super().__init__()
        self.setStyleSheet(STATUS_BAR_CSS)
        self._func_model = func_model
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_time_incr)
        self.reset_status()
        self.layout()

    def reset_status(self):
        """ Reset the properties of the status according to the fractal
        object """
        status = {
            "elapsed": {
                "val": 0.,
                "str_val": str(datetime.timedelta()),
            },
        }
        fractal = self._func_model.param0
        status.update(fractal.new_status(self))
        self._status = status

    def layout(self):
        self._layout = dict()
        for k, v in self._status.items():
            str_label = self.label(k)
            self._layout[k] = QLabel(str_label)
            self.addWidget(self._layout[k])

    def start_timer(self):
        self.reset_status()
        for key in self._layout.keys():
            self.update_status(key)
        # Update timer display every second
        self.timer.start(1000 * 1)

    def stop_timer(self):
        self.timer.stop()

    def on_time_incr(self):
        self._status["elapsed"]["val"] += 1.
        curr_time = datetime.timedelta(seconds=self._status["elapsed"]["val"])
        self.update_status("elapsed", str(curr_time))

    def update_status(self, key, str_val=None):
        """ Update the text status and refresh the display
        if str_val is None, only refresh the display
        """
        if str_val is not None:
            self._status[key]["str_val"] = str_val
        wget = self._layout[key]
        wget.setText(self.label(key))

    def label(self, key):
        return key + ": " + self._status[key]["str_val"]


class Action_func_widget(QFrame):
    """
    A Func_widget with parameters & actions group
    """
    func_started = pyqtSignal()
    func_performed = pyqtSignal()
    lock_navigation = pyqtSignal(bool)
    error_in_thread = QtCore.pyqtSignal(Exception)
    
    def __init__(self, parent, func_smodel, refresh_alias=None,
                 callback=False, may_interrupt=False,
                 locks_navigation=False):
        super().__init__(parent)
        self._submodel = func_smodel
        self.may_interrupt = may_interrupt
        self.locks_navigation = locks_navigation

        # Parameters and action boxes
        param_box = self.add_param_box(func_smodel)
        action_box = self.add_action_box()

        # general layout
        layout = QVBoxLayout()
        layout.addWidget(param_box, stretch=1)
        layout.addWidget(action_box)
        self.setLayout(layout)
            
        # Connect events
        self._script.clicked.connect(self.show_script)
        self._params.clicked.connect(self.show_func_params)
        if may_interrupt:
            self._interrupt.clicked.connect(self.raise_interruption)
        self._run.clicked.connect(self.run_func)
        
        # adds a binding to the image modified of other setting
        if refresh_alias is not None:
            (alias, keys) = refresh_alias
            model = func_smodel._model
            model.set_alias(alias, keys)
            self.func_performed.connect(functools.partial(
                model.item_refresh, alias)
            )
        
        # adds a binding to the parent slot
        if callback:
            self.func_performed.connect(functools.partial(
                parent.func_callback, self)
            )
        
        # add a binding to the navigation window
        if locks_navigation: 
            nav_win = getmainwindow(self).centralWidget() 
            self.lock_navigation.connect(nav_win.lock)
        
        # Adds Exception handling
        self.error_in_thread.connect(parent.on_error_in_thread)

        # Starts / stops status bar timer
        self.func_started.connect(parent.status_bar.start_timer)#)
        self.func_performed.connect(parent.status_bar.stop_timer)#)

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
        action_layout = QHBoxLayout()
        self._script = QPushButton("Show script")
        action_layout.addWidget(self._script)
        self._params = QPushButton("Show params")
        action_layout.addWidget(self._params)
        if self.may_interrupt:
            self._interrupt= QPushButton("Interrupt")
            action_layout.addWidget(self._interrupt)
        self._run = QPushButton("Run")
        action_layout.addWidget(self._run)
        
        action_box = QGroupBox("Actions")
        action_box.setLayout(action_layout)
        self.set_border_style(action_box)
        return action_box

    def set_border_style(self, gb):
        """ adds borders to an action box"""
        gb.setStyleSheet(
            "QGroupBox{border:1px solid #646464;"
                + "border-radius:5px;margin-top: 1ex;}"
            + "QGroupBox::title{subcontrol-origin: margin;"
                + "subcontrol-position:top left;"
                + "left: 15px;}")

    def raise_interruption(self):
        self._submodel.param0.raise_interruption()

    def lower_interruption(self):
        self._submodel.param0.lower_interruption()

    def load_calling_kwargs(self):
        """ Reload parameters stored from last call """
        with open(self.kwargs_path(), 'rb') as param_file:
            return pickle.load(param_file)

    def run_func(self):
        # Reset the interruption setting
        self.lower_interruption()
        # Save the func kwargs
        func_kwargs = self._submodel.getkwargs()
        self._submodel.save_func_dict()

        def thread_job():
            self.func_started.emit()
            if self.locks_navigation:
                self.lock_navigation.emit(True)
            self._run.setStyleSheet("background-color: red")
            try:
                self._submodel._func(**func_kwargs)
            except Exception as e:
                # send exception to the mainloop
                self.error_in_thread.emit(e)
            finally:
                # Does some clean-up to not lock everything on Error
                self._run.setStyleSheet("background-color: #646464")
                if self.locks_navigation:
                    self.lock_navigation.emit(False)
                self.func_performed.emit()

        # Now run the function in a dedicated thread
        threading.Thread(target=thread_job).start()


    def show_func_params(self):
        sm = self._submodel
        ce = Fractal_code_editor(self)
        str_assign = fs.gui.guitemplates.script_assignments(sm.getkwargs())
        
        
        ce.set_text(str_assign)
        ce.setWindowTitle("Parameters")
        ce.show()


    def show_script(self):
        """ Display the script in GUI """
        sm = self._submodel
        script = sm.getscript()
        ce = Fractal_code_editor()
        ce.set_text(script)
        ce.setWindowTitle("Script")
        ce.exec()


class Layout_col_synchronizer(QtCore.QObject):
    def __init__(self, cols):
        """
        Ensure several Param_Box share the same column width
        """
        super().__init__()
        self.cols = cols
        self._grid_pool = []

    def add_parambox(self, box):
        if not isinstance(box, Param_Box):
            raise ValueError("expecting a Param_Box")
        self._grid_pool += [box.gridLayout]
        box.synchro_size_evt.connect(self.synchro_slot)

    @pyqtSlot()
    def synchro_slot(self):
        """ On synchro_slot event, align all col width"""
        for icol in self.cols:
            self.synchro_width(icol)

    def synchro_width(self, icol):
        col_w_hint = 0
        for b in self._grid_pool:
            for irow in range(b.rowCount()):
                 l_item = b.itemAtPosition(irow, icol)
                 if l_item is not None:
                     col_w_hint = max(col_w_hint, l_item.sizeHint().width())
        # Now setting the common Width Hint
        for b in self._grid_pool:
            b.setColumnMinimumWidth(icol, col_w_hint)


class Param_Box(QWidget):
    
    synchro_size_evt = pyqtSignal()

    def __init__(self, title="", parent=None):
        """
        A QGridLayout with a nice title block
        """
        super().__init__(parent)
        # 
        box_layout = QVBoxLayout(self)
        box_layout.setSpacing(0)
        box_layout.setContentsMargins(0, 0, 0, 0)
        
        if title != "":
            self.make_label(title)
            box_layout.addWidget(self.label)

        self.make_content_area()
        box_layout.addWidget(self.content_area)
        

    def make_label(self, title):
        """ Create a label child item """
        # Adds the label at 0 - 0 position
        self.label = label = QLabel(title)
        sep_Font = QtGui.QFont()
        sep_Font.setStyle(QtGui.QFont.Style.StyleItalic)
        label.setFont(sep_Font)
        label.setStyleSheet(
            "border-bottom-width: 1px; "
            "border-bottom-style: solid; "
            "border-bottom-color: #b8b8b8; "
            "border-radius: 0px; "
        )

    def make_content_area(self):
        """ Create the content area, with its gridLayout """
        content_area = self.content_area = QWidget()
        self.gridLayout = QGridLayout(content_area)
    
    def resizeEvent(self, evt):
        super().resizeEvent(evt)
        self.synchro_size_evt.emit()




class Clickable_QLabel(QLabel):
    """ Just a QLabel with clicked evt"""
    clicked = pyqtSignal()
    def mousePressEvent(self, ev):
        self.clicked.emit()


class Collapsible_Param_Box(Param_Box):
    
    def __init__(self, title="", parent=None):
        """
        A QGridLayout with a nice title block ; content is collapsible
        """
        super().__init__(title, parent)
        self.anim = QPropertyAnimation(self.content_area, b"maximumHeight")


    def make_label(self, title):
        """ Create a label child item """
        # Adds the label at 0 - 0 position
        self.label = QWidget()
        box_layout = QHBoxLayout(self.label)
        
        label_txt = Clickable_QLabel(title)
        sep_Font = QtGui.QFont()
        sep_Font.setStyle(QtGui.QFont.Style.StyleItalic)
        label_txt.setFont(sep_Font)
        label_txt.setStyleSheet(
            "border-bottom-width: 1px; "
            "border-bottom-style: solid; "
            "border-bottom-color: #b8b8b8; "
            "border-radius: 0px; "
        )
        label_txt.clicked.connect(self.on_label_clicked) 

        toggle_button = self.toggle_button = QToolButton(
            text="", checkable=True, checked=False
        )
        toggle_button.setStyleSheet("QToolButton { border: none; }")
        toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        toggle_button.toggled.connect(self.on_toggle)
        
        box_layout.addWidget(toggle_button)
        box_layout.addWidget(label_txt)
        

    def make_content_area(self):
        """ Create the content area, with its gridLayout """
        content_area = self.content_area = QWidget()
        self.gridLayout = QGridLayout(content_area)
        content_area.setMaximumHeight(0)
        content_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

    def on_label_clicked(self):
        self.toggle_button.toggle()

    def on_toggle(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

        if checked:
            start = 0 # self.content_area.maximumHeight()
            end = self.gridLayout.sizeHint().height()
        else:
            start = self.gridLayout.sizeHint().height()
            end = 0
        # self.layout()

        anim = self.anim
        anim.setDuration(250)
        anim.setStartValue(start)
        anim.setEndValue(end)

        if checked:
            def no_limit():
                # No limit on height
                self.content_area.setMaximumHeight(16777215)
                anim.finished.disconnect()
            anim.finished.connect(no_limit)

        anim.start()


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
        
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )

    def layout(self):
        fd = self._submodel._dict

        synchro = self.synchro = Layout_col_synchronizer(cols=(0, 2))

        box = Param_Box("", self)
        synchro.add_parambox(box)
        self._layout.addWidget(box, 0, 0, 1, 4)
        current_layout = box.gridLayout # self._layout
        current_layout_row = 0
        main_layout_row = 1

        for i_param in range(fd["n_params"]):
            uni_typed = (fd[(i_param, "n_types")] == 0)
            if uni_typed and (fd[(i_param, 0, "type")] is separator):
                # This is a new 'normal, non-collapsible' block
                box = self.layout_separator(
                    i_param, self._layout, main_layout_row
                )
                synchro.add_parambox(box)
                current_layout = box.gridLayout
                current_layout_row = 1
                main_layout_row += 1

            elif (uni_typed
                  and (fd[(i_param, 0, "type")] is collapsible_separator)):
                # This is a new 'collapsible' block
                box = self.layout_collapsible_separator(
                    i_param, self._layout, main_layout_row
                )
                synchro.add_parambox(box)
                current_layout = box.gridLayout
                current_layout_row = 1
                main_layout_row += 1

            else:
                # adding a parameter editor to the current block layout,
                # or directly the main layout box
                self.layout_param(i_param, current_layout, current_layout_row)
                current_layout_row += 1
                if (current_layout is self._layout):
                    raise ValueError()
                    # main_layout_row += 1

        # adds a spacer at bottom
        self._layout.setRowStretch(main_layout_row, 1)
        
        # Middle column is allocated  all the strech space
        self._layout.setColumnStretch(0, 0)
        self._layout.setColumnStretch(1, 1)
        self._layout.setColumnStretch(2, 0)


    def layout_separator(self, i_param, layout, layout_row):
        """ Adds a separator to the main layout - Returns a handle to the
        separator area sublayout """
        fd = self._submodel._dict
        sep_name = fd[(i_param, 0, "val")]
        box = Param_Box(sep_name, self)
        layout.addWidget(box, layout_row, 0, 1, 4)
        layout.setRowStretch(i_param, 0)
        return box


    def  layout_collapsible_separator(self, i_param, layout, layout_row):
        """ Adds a collapsible separator to the main layout
        Returns a handle to the separator area sublayout """
        fd = self._submodel._dict
        sep_name = fd[(i_param, 0, "val")]
        box = Collapsible_Param_Box(sep_name, self)
        layout.addWidget(box, layout_row, 0, 1, 4)
        layout.setRowStretch(i_param, 0)
        return box
        

    def layout_param(self, i_param, layout, layout_row): # Added : layout
        fd = self._submodel._dict
        
        name = fd[(i_param, "name")]
        name_label = QLabel(name)
        myFont = QtGui.QFont()
        myFont.setWeight(QtGui.QFont.Weight.ExtraBold)
        name_label.setFont(myFont)
        layout.addWidget(name_label, layout_row, 0, 1, 1)

        # Handles Union types
        qs = QStackedWidget()
        qs.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Minimum
        )
        n_uargs = fd[(i_param, "n_types")]
        if n_uargs == 0:
            utype = fd[(i_param, 0, "type")]
            utype_label = QLabel(type_name(utype))
            layout.addWidget(utype_label, layout_row, 2, 1, 1)
            self.layout_uarg(qs, i_param, 0)
        else:
            utypes = [fd[(i_param, utype, "type")] for utype in range(n_uargs)]
            utypes_combo = self._widgets[(i_param, "type_sel")] = QComboBox()
            self._widgets[(i_param, 'qs_type_sel')] = utypes_combo
            utypes_combo.addItems(type_name(t) for t in utypes)
            utypes_combo.setCurrentIndex(fd[(i_param, "type_sel")])
            utypes_combo.activated.connect(functools.partial(
                self.on_user_mod, (i_param, "type_sel"),
                utypes_combo.currentIndex
            ))
            # Connect to the QS
            utypes_combo.currentIndexChanged[int].connect(qs.setCurrentIndex)

            self._layout.addWidget(utypes_combo, layout_row, 2, 1, 1)
            for utype in range(n_uargs):
                self.layout_uarg(qs, i_param, utype)

        # The displayed item of the union is denoted by "type_sel" :
        # self.layout_uarg(qs, i_param, fd[(i_param, "type_sel")])
        qs.setCurrentIndex(fd[(i_param, "type_sel")])
        layout.addWidget(qs, layout_row, 1, 1, 1)
        layout.setRowStretch(layout_row, 0)


    
    def layout_uarg(self, qs, i_param, i_union):

        fd = self._submodel._dict
        # n_uargs = fd[(i_param, "n_types")]
        utype = fd[(i_param, i_union, "type")]
        uval = fd[(i_param, i_union, "val")]
        atom_wget = atom_wget_factory(utype)(utype, uval, self._model)
        self._widgets[(i_param, i_union, "val")] = atom_wget

        atom_wget.user_modified.connect(functools.partial(
                self.on_user_mod, (i_param, i_union, "val"),
                atom_wget.value)
        )
        qs.addWidget(atom_wget)
        
        if isinstance(atom_wget, Atom_Presenter_mixin):
            atom_wget.request_presenter.connect(functools.partial(
                self.on_presenter, (i_param, i_union, "val")))


    def reset_layout(self):
        """ Delete every item in self._layout """
        raise RuntimeError("Shall never be called ?")
        for i in reversed(range(self._layout.count())): 
            w = self._layout.itemAt(i).widget()
            if w is not None:
                w.setParent(None)
                # Alternative deletion instruction :
                # w.deleteLater() 
# https://stackoverflow.com/questions/41053306/removing-a-widget-from-its-wxpython-parent

    def on_user_mod(self, key, val_callback, *args):
        """ Notify the model of modification by the user of a widget"""
        val = val_callback()
        self.func_user_modified.emit(key, val)

    def model_event_slot(self, keys, val):
        """ Handles modification of widget triggered from model """
        # Does the event impact one of my child widgets ? otherwise, return
        if keys[:-1] != self._func_keys:
            return # This is not for this Func_widget
        key = keys[-1]
        try:
            wget = self._widgets[key]
        except KeyError:
            # Not a widget, probably a parameter default signal
            return

        # Check first Atom_Mixin
        if isinstance(wget, Atom_Edit_mixin):
            wget.on_model_event(val)
        else:
            raise NotImplementedError(
                f"Func_widget.model_event_slot {wget}"
            )

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
            self._model.register(
                presenter_class(self._model, mapping),
                register_key
            )
            wget = wget_class(None, self._model._register[register_key])
            main_window = getmainwindow(self)
            dock_widget = QDockWidget(register_key, None, Qt.WindowType.Window)
            dock_widget.setWidget(wget)
            dock_widget.setStyleSheet(DOCK_WIDGET_CSS)
            main_window.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea,
                dock_widget
            )
            self.presenters[register_key] = dock_widget
        else:
            # Docwidget of presenter exists, we only toggles visibility
            dock_widget = self.presenters[register_key]
            toggle = not(dock_widget.isVisible())
            dock_widget.setVisible(toggle)
            if toggle:
                dock_widget.raise_()

def atom_wget_factory(atom_type):
    if typing.get_origin(atom_type) is typing.Literal:
        return Atom_QComboBox
    elif issubclass(atom_type, fs.Fractal): 
        return Atom_fractal_button
    elif issubclass(atom_type, fs.numpy_utils.Numpy_expr):
        return Atom_QLineEdit
    else:
        wget_dic = {
            int: Atom_QLineEdit,
            float: Atom_QLineEdit,
            str: Atom_QLineEdit,
            bool: Atom_QCheckBox, # Atom_QBoolComboBox
            mpmath.mpf: Atom_QPlainTextEdit,
            fs.colors.Color: Atom_Color,
            fs.colors.Fractal_colormap: Atom_cmap_button,
            fs.colors.Blinn_lighting: Atom_lighting_button,
            type(None): Atom_QLineEdit
        }
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
        with QtCore.QSignalBlocker(self):
            self.setChecked(val)


class NoWheel_mixin:
    def eventFilter(self, widget, evt):
        """ Prevent annoying wheel action for Combobox derived classes """
        if evt.type() == QtCore.QEvent.Type.Wheel:
            evt.ignore()
            return True
        return super(QComboBox, self).eventFilter(widget, evt)


class Atom_QBoolComboBox(QComboBox, Atom_Edit_mixin, NoWheel_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent)
        self.installEventFilter(self)
        self.setStyleSheet(COMBO_BOX_CSS)

        self._type = atom_type
        self._choices = ["True", "False"]
        self._values = [True, False]
        self.currentTextChanged.connect(self.on_user_event)
        self.addItems(str(c) for c in self._choices)
        self.setCurrentIndex(self._values.index(val))

    def value(self):
        return self._values[self.currentIndex()]

    def on_user_event(self):
        self.user_modified.emit()

    def on_model_event(self, val):    
        with QtCore.QSignalBlocker(self):
            self.setCurrentIndex(self._values.index(val))


class Atom_Color(QPushButton, Atom_Edit_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        """
        val is given in rgb float, or rgba float (in the range [0., 1.])
        Internally we use QtGui.QColor rgb or rgba i.e. uint8 format
        """
        super().__init__("", parent)
        self._type = atom_type
        self._kind = {3: "rgb", 4: "rgba"}[len(val)]
        self._qcolor = None
        self.update_color(val)
        self.clicked.connect(self.on_user_event)

    def update_color(self, color):
        """ color: QtGui.QColor or fs.colors.Color """
        if isinstance(color, QtGui.QColor):
            qcolor = color
        else:
            qcolor = QtGui.QColor(
                *list(int(channel * 255) for channel in color)
            )
        if qcolor != self._qcolor:
            self._qcolor = qcolor
            self.setStyleSheet("background-color: {0};"
                               "border-color: {1};"
                               "border-style: solid;"
                               "border-width: 1px;"
                               "border-radius: 4px;".format(
                           self._qcolor.name(), "grey"))

            if self._kind == "rgba":
                # Paint a gradient from the color with transparency to the
                # color with no transparency (rgba "a" value set to 255)
                gradient = QtGui.QLinearGradient(0, 0, 1, 0)
                gradient.setCoordinateMode(
                    QtGui.QGradient.CoordinateMode.ObjectBoundingMode
                )
                gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0, qcolor.alpha()))
                gradient.setColorAt(0.1, QtGui.QColor(0, 0, 0, qcolor.alpha()))
                gradient.setColorAt(0.9, Qt.GlobalColor.black)
                gradient.setColorAt(1.0, Qt.GlobalColor.black)
                effect = QGraphicsOpacityEffect(self)
                effect.setOpacity(1.)
                effect.setOpacityMask(gradient)
                self.setGraphicsEffect(effect)

            self.repaint()
            self.user_modified.emit()

    def value(self):
        c = self._qcolor
        if self._kind == "rgb":
            ret = (c.redF(), c.greenF(), c.blueF())
        elif self._kind == "rgba":
            ret = (c.redF(), c.greenF(), c.blueF(), c.alphaF())
        return ret

    def on_user_event(self):
        colord = QColorDialog()
        colord.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if self._kind == "rgba":
            colord.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel)
        colord.setCurrentColor(self._qcolor)
        colord.setCustomColor(0, self._qcolor)
        colord.currentColorChanged.connect(self.update_color)
        old_col = self._qcolor
        if colord.exec():
            self.update_color(colord.currentColor())
        else:
            self.update_color(old_col)

    def on_model_event(self, val):
        with QtCore.QSignalBlocker(self):
            self.update_color(val)


class Atom_QLineEdit(QLineEdit, Atom_Edit_mixin): 
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(str(val), parent)
        self._type = atom_type
        self.textChanged[str].connect(self.validate)
        self.editingFinished.connect(self.on_user_event)
        self.setValidator(Atom_Text_Validator(atom_type, val))
        self.setStyleSheet(PARAM_LINE_EDIT_CSS.format("#25272C"))
        if atom_type is type(None):
            self.setReadOnly(True)
        self.validate(self.text(), acceptable_color="#25272C")

    def value(self):
        if self._type is type(None):
            return None

        elif issubclass(self._type, fs.numpy_utils.Numpy_expr):
            # We return a Nump_expr, the variables are those of the default
            # value, as stored in the validator
            variables = self.validator()._variables
            text = self.text()
            return self._type(variables, text)

        return self._type(self.text())

    def on_user_event(self):
        self.user_modified.emit()

    def on_model_event(self, val): 
        with QtCore.QSignalBlocker(self):
            self.setText(str(val))
            self.validate(self.text(), acceptable_color="#25272C")

    def validate(self, text, acceptable_color="#c8c8c8"):
        validator = self.validator()
        if validator is not None:
            ret, _, _ = validator.validate(text, self.pos())
            if ret == QtGui.QValidator.State.Acceptable:
                self.setStyleSheet(PARAM_LINE_EDIT_CSS.format(
                       acceptable_color))
            else:
                self.setStyleSheet(PARAM_LINE_EDIT_CSS.format(
                       "#dc4646"))


class Atom_QPlainTextEdit(QPlainTextEdit, Atom_Edit_mixin):
    user_modified = pyqtSignal()
    # TODO could use this to update mode dps in line with x, y text 
    mp_dps_used = pyqtSignal(int)  

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(str(val), parent)
        self._type = atom_type
        self.setStyleSheet("border: 1px solid  lightgrey")
        # Wrapping parameters
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth) 
        self.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)
        self.setStyleSheet(PLAIN_TEXT_EDIT_CSS.format("#25272C"))

        self._validator = Atom_Text_Validator(atom_type, val)

        # signals / slots
        self.textChanged.connect(self.validate)

    def value(self):
        return self.toPlainText()

    def on_model_event(self, val):
        with QtCore.QSignalBlocker(self):
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
            if ret == QtGui.QValidator.State.Acceptable:
                self.setStyleSheet(PLAIN_TEXT_EDIT_CSS.format(
                       "#25272C"))
                if from_user:
                    self.user_modified.emit()
            else:
                self.setStyleSheet(PLAIN_TEXT_EDIT_CSS.format(
                       "#dc4646"))
            cursor = QtGui.QTextCursor(self.document())
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)

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


class Atom_QComboBox(QComboBox, Atom_Edit_mixin, NoWheel_mixin):
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent)
        self.setStyleSheet(COMBO_BOX_CSS)
        self.installEventFilter(self)

        self._type = atom_type
        self._choices = typing_litteral_choices(atom_type)
        self.currentTextChanged.connect(self.on_user_event)
        self.addItems(str(c) for c in self._choices)
        self.setCurrentIndex(val)

    def value(self):
        return self.currentIndex()

    def on_user_event(self):
        self.user_modified.emit()

    def on_model_event(self, val):
        with QtCore.QSignalBlocker(self):
            self.setCurrentIndex(val)


class Atom_fractal_button(QPushButton, Atom_Edit_mixin, Atom_Presenter_mixin):
    user_modified = pyqtSignal()
    request_presenter = pyqtSignal(object, object)

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent)
        self._fractal = val
        self.setText(self.value().__class__.__name__)

    def value(self):
        return self._fractal

    def on_model_event(self, val):
        pass

    def mouseReleaseEvent(self, event):
        self.request_presenter.emit(Fractal_presenter, Fractal_editor)


class Qobject_image(QWidget):
    """ Widget of an object implementing "output_ImageQt" image,
    fixed height and  expanding width """
    def __init__(self, parent, img_object, minwidth=200, height=20):
        super().__init__(parent)
        self._object = img_object
        self.setMinimumWidth(minwidth)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.setSizePolicy(
                QSizePolicy.Policy.Minimum,
                QSizePolicy.Policy.Expanding
        )

    def paintEvent(self, evt):
        size = self.size()
        nx, ny = size.width(), size.height()
        QtGui.QPainter(self).drawImage(0, 0, self._object.output_ImageQt(nx, ny))


class Atom_cmap_button(Qobject_image, Atom_Edit_mixin, Atom_Presenter_mixin):
    user_modified = pyqtSignal()
    request_presenter = pyqtSignal(object, object)

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent, val)
        self._object = None
        self.update_cmap(val)

    def update_cmap(self, cmap):
        """ cmap: fs.color.Fractal_colormap """
        self._object = cmap
        self.repaint()
        # Note : we do not emit self.user_modified, this shall be done at
        # Qcmap_editor widget level

    def value(self):
        return self._object

    def on_model_event(self, val):
        with QtCore.QSignalBlocker(self):
            self.update_cmap(val)

    def mouseReleaseEvent(self, event):
        self.request_presenter.emit(Colormap_presenter, Qcmap_editor)


class Atom_lighting_button(Qobject_image, Atom_Edit_mixin, Atom_Presenter_mixin):
    user_modified = pyqtSignal()
    request_presenter = pyqtSignal(object, object)

    def __init__(self, atom_type, val, model, parent=None):
        super().__init__(parent, val)
        self._object = None
        self.update_lighting(val)

    def update_lighting(self, lighting):
        """ cmap: fs.color.Fractal_colormap """
        if lighting != self._object:
            self._object = lighting
            self.repaint()
            # Note : we do not emit self.user_modified, this shall be done at
            # Qcmap_editor widget level

    def value(self):
        return self._object

    def on_model_event(self, val):
        with QtCore.QSignalBlocker(self):
            self.update_lighting(val)

    def mouseReleaseEvent(self, event):
        self.request_presenter.emit(Lighting_presenter, Qlighting_editor)


class Atom_Text_Validator(QtGui.QValidator):

    def __init__(self, atom_type, val):
        super().__init__()
        self._type = atom_type

        if issubclass(atom_type, fs.numpy_utils.Numpy_expr):
            # The only way to keep track of the variables : keep those of
            # the provided default (the type will not hold this info)
            self._variables = copy.deepcopy(val.variables)

    def validate(self, val, pos):
        valid = {True: QtGui.QValidator.State.Acceptable,
                 False: QtGui.QValidator.State.Intermediate}
        if self._type is type(None):
            return (valid[val == "None"], val, pos)
        if issubclass(self._type, fs.numpy_utils.Numpy_expr):
            return (
                valid[self._type.validates_expr(self._variables, val)],
                val,
                pos
            )
        try:
            casted = self._type(val)
        except ValueError:
            return (valid[False], val, pos)

        if self._type is mpmath.ctx_mp_python.mpf:
            # Starting or trailing carriage return are invalid
            if (val[-1] == "\n") or (val[0] == "\n"):
                return (valid[False], val, pos)
        return (valid[isinstance(casted, self._type)], val, pos)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Delegates implementation for array cells

class ColorDelegate(QStyledItemDelegate):
    def __init__(self, parent, options=None):
        """ Custom cell delegate to display / edit a colors
        parent : the QTableWidget
        """
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        dialog = QColorDialog(None) #
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        dialog.setCurrentColor(index.data(Qt.ItemDataRole.BackgroundRole))
        dialog.setCustomColor(0, index.data(Qt.ItemDataRole.BackgroundRole))
        # QT doc: The returned editor widget should have Qt::StrongFocus
        dialog.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        dialog.setFocusProxy(parent)
        return dialog

    def setEditorData(self, editor, index):
        color = index.data(Qt.ItemDataRole.BackgroundRole)
        editor.setCurrentColor(color)

    def setModelData(self, editor, model, index):
        """ If modal QColorDialog result code is Accepted, save color"""
        if editor.result():
            color = editor.currentColor()
            model.setData(index, color, Qt.ItemDataRole.BackgroundRole)

    def paint(self, painter, option, index):
        """ Fill with BackgroundRole color + red rectangle for selection."""
        # QT doc: After painting, you should ensure that the painter is
        # returned to the state it was supplied in when this function was
        # called.
        painter.save()
        selected = bool(option.state 
                        & QtWidgets.QStyle.StateFlag.State_Selected)
        painter.fillRect(option.rect, index.data(Qt.ItemDataRole.BackgroundRole))
        if selected:
            rect = option.rect
            rect.adjust(1, 1, -1, -1)
            pen = QtGui.QPen(Qt.GlobalColor.red)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
        painter.restore()


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
        val = index.data(Qt.ItemDataRole.DisplayRole)
        editor.setText(val)

    def setModelData(self, editor, model, index):
        """ save int val to the model"""
        val = editor.text()
        model.setData(index, val, Qt.ItemDataRole.DisplayRole)
        if self.validate(index):
            color = QtGui.QColor("#646464")
        else:
            color = QtGui.QColor("red")
        model.setData(index, color, Qt.ItemDataRole.BackgroundRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def validate(self, index):
        val = index.data(Qt.ItemDataRole.DisplayRole)
        try:
            val = int(val)
        except (TypeError, ValueError):
            return False
        return (val >= self.min_val and val <= self.max_val)


class FloatDelegate(QStyledItemDelegate):
    def __init__(self, parent, options):
        """ Custom cell delegate to display / edit a float
        parent : the QTableWidget
        """
        super().__init__(parent)
#        self.min_val = options["min"]
#        self.max_val = options["max"]

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        """
        index: PyQt5.QtCore.QModelIndex
        """
        val = index.data(Qt.ItemDataRole.DisplayRole)
        editor.setText(val)

    def setModelData(self, editor, model, index):
        """ save int val to the model"""
        val = editor.text()
        model.setData(index, val, Qt.ItemDataRole.DisplayRole)
        if self.validate(index):
            color = QtGui.QColor("#646464")
        else:
            color = QtGui.QColor("red")
        model.setData(index, color, Qt.ItemDataRole.BackgroundRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def validate(self, index):
        val = index.data(Qt.ItemDataRole.DisplayRole)
        try:
            val = float(val)
        except (TypeError, ValueError):
            return False
        return True # (val >= self.min_val and val <= self.max_val)


class ComboDelegate(QStyledItemDelegate):
    def __init__(self, parent, options):
        """ Custom cell delegate to display / edit a combo box
        parent : the QTableWidget
        """
        super().__init__(parent)
        self.choices = options["choices"]

    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.addItems(self.choices)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        val = index.data(Qt.ItemDataRole.DisplayRole)
        editor.setCurrentText(val)

    def setModelData(self, editor, model, index):
        val = editor.currentText()
        model.setData(index, val, Qt.ItemDataRole.DisplayRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def validate(self, index):
        val = index.data(Qt.ItemDataRole.DisplayRole)
        return val in self.choices


class ExprDelegate(QStyledItemDelegate):
    def __init__(self, parent, options):
        """ Custom cell delegate to display / edit an expr
        parent : the QTableWidget
        """
        super().__init__(parent)
        self.modifier = options["modifier"]

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        val = index.data(Qt.ItemDataRole.DisplayRole)
        editor.setText(val)

    def setModelData(self, editor, model, index):
        val = editor.text()
        model.setData(index, val, Qt.ItemDataRole.DisplayRole)
        if self.validate(index):
            color = QtGui.QColor("#646464")
        else:
            color = QtGui.QColor("red")
        model.setData(index, color, Qt.ItemDataRole.BackgroundRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def validate(self, index):
        data = index.data(Qt.ItemDataRole.DisplayRole)
        if data is None:
            data = "x"
        val = self.modifier(data)
        try:
            return fs_parser.acceptable_expr(
                ast.parse(val, mode="eval"), safe_vars=["x"]
            )
        except (SyntaxError):
            return False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Array Editors
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Table_QSpinBox(QSpinBox, Atom_Edit_mixin):

    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, model, parent=None):
        """ SpinBox wich request a user explicit validation is when text is
        modified (event editingFinished) before sending the new val.
        Dedicated to the nrow chooser"""
        super().__init__(parent)
        assert atom_type == int
        self._type = int
        self.editingFinished.connect(self.on_edit_finished)
        self.setStyleSheet(SPINBOX_CSS.format("#25272C"))

    def keyPressEvent(self, evt):
        self.setStyleSheet(SPINBOX_CSS.format("#c8c8c8"))
        super().keyPressEvent(evt)

    def stepBy(self, int_steps):
        super().stepBy(int_steps)
        self.user_modified.emit()

    def on_edit_finished(self):
        self.setStyleSheet(SPINBOX_CSS.format("#25272C"))
        self.user_modified.emit()

    def on_model_event(self, val):
        self.setText(str(val))


class Base_array_editor(QWidget):
    """
    Base widget for editors which feature a data table and a parameter panel
    """
    data_user_modified = pyqtSignal(object, object)
    
    std_flags = (
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsEditable
    )
    unvalidated_flags = (
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable
    )
    locked_flags = (Qt.ItemFlag.ItemIsEnabled)

    min_row = 1
    max_row = 256

    def __init__(self, parent, data_presenter):
        super().__init__(parent)    
        # General data from presenter
        # Load titles
        self.param_title = data_presenter.param_title
        self.table_title = data_presenter.table_title

        # Load columns data
        self.col_arr_items = data_presenter.col_arr_items
        self.col_arr_dtypes = data_presenter.col_arr_dtypes
        self.build_delegates(data_presenter.special_hooks)

        # Load extra parameters list
        # Detailed implementation is delegated to child classes
        self.extra_parameters = data_presenter.extra_parameters

        # Model pointers
        self._model = data_presenter._model
        self._mapping = data_presenter._mapping
        self._presenter = data_presenter

        # Bypass machinery for efficient table update
        self.data_update_lock = True
        
        layout = QVBoxLayout()
        layout.addWidget(self.add_param_box())
        layout.addWidget(self.add_table_box(), stretch=1)
        self.setLayout(layout)

        self._wget_n.user_modified.connect(
            functools.partial(self.on_ncol_mod, self._wget_n.value)
        )
        self._table.itemChanged.connect(
            functools.partial(self.event_filter, "table")
        )

        self.data_user_modified.connect(
            data_presenter.data_user_modified_slot
        )
        self._model.model_event.connect(self.model_event_slot)

        self.data_update_lock = False
        
    def on_ncol_mod(self, val_callback):
        """ Notify of modification by the user of col number """
        val = val_callback()
        self.event_filter("size", val)


    def build_delegates(self, special_hooks):
        """ Builds the GUI implementation for each column based on data type
        """
        delegates = list()
        delegates_options = list()
        roles = list()
        val_funcs = list()
        row_ranges_func = list()
        
        bgr = Qt.ItemDataRole.BackgroundRole
        dpr = Qt.ItemDataRole.DisplayRole

        for icol, item_type in enumerate(self.col_arr_dtypes):
            is_class = inspect.isclass(item_type)

            if item_type is int:
                delegates.append(IntDelegate)
                delegates_options.append({"min": 1, "max":256})
                roles.append(dpr)
                val_funcs.append(lambda v: str(v))
                row_ranges_func.append(lambda l: l)

            elif item_type is float:
                delegates.append(FloatDelegate)
                delegates_options.append(None)
                roles.append(dpr)
                val_funcs.append(lambda v: str(v))
                row_ranges_func.append(lambda l: l)

            elif is_class and issubclass(item_type, fs.colors.Color):
                delegates.append(ColorDelegate)
                delegates_options.append(None)
                roles.append(bgr)
                val_funcs.append(
                        lambda v: QtGui.QColor(*list(int(255 * f) for f in v))
                )
                row_ranges_func.append(lambda l: l)

            elif is_class and issubclass(item_type, fs.numpy_utils.Numpy_expr):
                delegates.append(ExprDelegate)
                delegates_options.append(
                        {"modifier": lambda expr: ("lambda x: " + expr)}
                )
                roles.append(dpr)
                val_funcs.append(lambda v: v)
                row_ranges_func.append(lambda l: l)

            elif typing.get_origin(item_type) is typing.Literal:
                # it is an enumeration...
                delegates.append(ComboDelegate)
                delegates_options.append(
                        {"choices": typing_litteral_choices(item_type)}
                )
                roles.append(dpr)
                val_funcs.append(lambda v: v)
                row_ranges_func.append(lambda l: l)

            else:
                raise ValueError(item_type)

        # Now we apply the special hooks from the presenter, if any
        for (icol, item), value in special_hooks.items():
            if item == "row_ranges_func":
                row_ranges_func[icol] = value
            elif item == "delegates_options":
                delegates_options[icol] = value
            else:
                raise ValueError(item)
        
        self.col_delegates = delegates
        self.col_delegates_options = delegates_options
        self.col_roles = roles
        self.col_val_funcs = val_funcs
        self.row_ranges_func = row_ranges_func


    @property
    def n_cols(self):
        return len(self.col_arr_items)

    @property
    def n_rows(self):
        return self._presenter.n_rows

    @property
    def presenter_classname(self):
        # e.g., "Colormap_presenter"
        return self._presenter.__class__.__name__ 

    def add_param_box(self):
        # Note: derived classes should implement the detailed layout
        param_box = QGroupBox(self.param_title)
        # Widget for the number of lines
        self._wget_n = Table_QSpinBox(
            atom_type=int, val=self._presenter.n_rows,
            model=None, parent=self
        ) # 
        self._wget_n.setRange(self.min_row, self.max_row)
        return param_box

    def populate_param_box(self):
        self._wget_n.setRange(self.min_row, self.max_row)
        self._wget_n.setValue(self._presenter.n_rows)

    def add_table_box(self):
        table_box = QGroupBox(self.table_title)
        table_layout = QVBoxLayout()

        self._table = QTableWidget()
        
        # COLUMNS : colors, kinds, n, funcs=None
        n_cols = self.n_cols
        self._table.setColumnCount(n_cols)
        self.populate_table()
        self._table.setStyleSheet(TABLE_WIDGET_CSS)


        # Set up the delegates
        for icol in range(n_cols):
            self._table.setItemDelegateForColumn(
                icol,
                self.col_delegates[icol](
                        self._table, self.col_delegates_options[icol]
                )
            )
        self._table.setHorizontalHeaderLabels(tuple(self.col_arr_items))

        h_header = self._table.horizontalHeader()
        h_header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        h_header.setStretchLastSection(False)
        h_header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)

        self._table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )

        table_layout = QHBoxLayout()
        table_box.setLayout(table_layout)
        table_layout.addWidget(self._table, stretch=0)




        return table_box

    def populate_table(self):
        # Signals shall be temporarly blocked to avoid infinite event loop.
        with QtCore.QSignalBlocker(self._table):
            n_rows =  self._presenter.n_rows
            self._table.setRowCount(n_rows)

            n_cols = self.n_cols
            for icol in range(n_cols):
                col_item = self.col_arr_items[icol]
                self.populate_column(
                    col=icol,
                    row_range=range(self.row_ranges_func[icol](n_rows)),
                    role=self.col_roles[icol],
                    tab=self._presenter.col_data(col_item),
                    old_tab=self._presenter.old_col_data(col_item),
                    val_func=self.col_val_funcs[icol],
                    flags=self.std_flags
                )

    def populate_column(self, col, row_range, role, tab, old_tab,
                        val_func, flags=None):
        for irow in row_range:
            val = tab[irow]
            # to speed up we have to explicitely track the modifications
            if self.match_old_val(val, irow, old_tab):
                continue

            val = val_func(val)
            item = self._table.item(irow, col)
            if item is None:
                item = QTableWidgetItem()
                self._table.setItem(irow, col, item)
            if flags is not None:
                item.setFlags(flags)
            item.setData(role, val)

    def match_old_val(self, val, irow, old_tab):
        """ Return True if the value has not been modifed """
        if self.data_update_lock:
            return False
        if irow >= (len(old_tab) - 1):
            # always update the last row
            return False
        try:
            old_val = old_tab[irow]
        except IndexError:
            # Update if we have a new row
            return False
        if isinstance(val, (np.ndarray)):
            # special case of RGB cells
            return np.all(val == old_val)
        return val == old_val


    def event_filter(self, source, val):
        # event handling on _table.itemChanged.connect
        if source in ["size",] + self.extra_parameters:
            self.data_user_modified.emit(source, val)

        elif source == "table":
            # val : PyQt5.QtWidgets.QTableWidgetItem. We need to process it for
            # the Presenter
            row, col = val.row(), val.column()
            role = self.col_roles[col]
            item_data = val.data(role)
            item_type = self.col_arr_dtypes[col]
            is_class = inspect.isclass(item_type)

            if is_class and issubclass(item_type, fs.colors.Color):
                # No validation needed, as values are selected programmatically
                validated = True
                # QColor to arrray concersion
                item_data = [
                    item_data.redF(), item_data.greenF(), item_data.blueF()
                ]
            else:
                # Need delegate validation
                delegate = self._table.itemDelegateForColumn(col)
                model = self._table.model()
                index = model.index(row, col) 
                validated = delegate.validate(index)
                if (item_type is int) and (item_data is not None):
                    item_data = int(item_data)

            if validated:
                # make sure that the normal flags are activated (selection
                # allowed)
                with QtCore.QSignalBlocker(self._table):
                    val.setFlags(self.std_flags)

                self.data_user_modified.emit(source, (row, col, item_data))

            else:
                # Invalid value, the event is not emited
                # Prevent the cell from being selected, to display the "red" bg
                # color (through flags)
                with QtCore.QSignalBlocker(self._table):
                    val.setFlags(self.unvalidated_flags)

        else:
            raise ValueError(source)

    def model_event_slot(self, keys, val):
        if keys == self._presenter._mapping[self.presenter_classname]:
            # Sets the value of the sub-widgets according to the smodel
            self.populate_param_box()
            with QtCore.QSignalBlocker(self._table):
                reset = (self.count_changes() > 10)
                if reset: # Massive change, faster to start from new
                    self._table.setRowCount(0)
                    self.data_update_lock = True
                self.populate_table()
                self._presenter.reset_old_data()
                if reset:
                    self.data_update_lock = False

    def count_changes(self):
        """ Number of items that have changed"""
        counter = 0
        n_cols = self.n_cols

        for icol in range(n_cols):
            col_item = self.col_arr_items[icol]
            tab=self._presenter.col_data(col_item)
            old_tab=self._presenter.old_col_data(col_item)
            
            for irow in range(min(len(tab), len(old_tab))):
                val = tab[irow]
                if not(self.match_old_val(val, irow, old_tab)):
                    counter += 1
        return counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Base_array_editor implementations

class Qcmap_editor(Base_array_editor):
    """
    Widget of a cmap editor : parameters & data table
    """
    data_user_modified = pyqtSignal(object, object)
    min_row = 2

    def __init__(self, parent, cmap_presenter):
        # Additional data from the presenter
        self.extent_choices = cmap_presenter.extent_choices

        super().__init__(parent, cmap_presenter)
        self._wget_extent.currentTextChanged.connect(
            functools.partial(self.event_filter, "extent")
        )

    @property
    def _cmap(self):
        return self._presenter.data

    def add_param_box(self):
        """ Customize base class layout """
        param_box = super().add_param_box() #param_box_base_items()
        param_layout = QHBoxLayout()
        param_box.setLayout(param_layout)

        # Other add-hoc items :
        self._wget_extent = QComboBox(self)
        self._wget_extent.addItems(self.extent_choices)
        self._preview = Qobject_image(self, self._cmap)

        param_layout.addWidget(self._wget_n)
        param_layout.addWidget(self._wget_extent)
        param_layout.addWidget(self._preview, stretch=1)

        self.populate_param_box()
        return param_box

    def populate_param_box(self):
        super().populate_param_box()
        val_extent = self.extent_choices.index(self._cmap.extent)
        self._wget_extent.setCurrentIndex(val_extent)
        self._preview._object = self._cmap
        self._preview.repaint()

    def populate_table(self):
        """ Customizing the table with a frozen last row... """
        super().populate_table()
        with QtCore.QSignalBlocker(self._table):
            self.freeze_row(self.n_rows - 1, range(1, 4))

    def freeze_row(self, row, col_range):
        for icol in col_range:
            # https://forum.qt.io/topic/3489/solved-how-enable-editing-to-qtablewidgetitem/2
            freezed_item = QTableWidgetItem()
            freezed_item.setFlags(self.locked_flags)
            self._table.setItem(row, icol, freezed_item)


class Qlighting_editor(Base_array_editor):
    """
    Widget of a cmap editor : parameters & data table
    """
    data_user_modified = pyqtSignal(object, object)
    min_row = 1

    def __init__(self, parent, lighting_presenter):

        super().__init__(parent, lighting_presenter)

        for wg, source in {
                self._wget_k_ambient: "k_ambient",
                self._wget_color_ambient:"color_ambient"
        }.items():
            wg.user_modified.connect(
                functools.partial(self.on_user_mod, source, wg.value)
            )

    @property
    def _lighting(self):
        return self._presenter.data

    def add_param_box(self):
        """ Customize base class layout """
        param_box = super().add_param_box() #param_box_base_items()
        
        param_layout = QGridLayout(self)
        param_box.setLayout(param_layout)

        myFont = QtGui.QFont()
        myFont.setWeight(QtGui.QFont.Weight.ExtraBold)

        k_ambient_label = QLabel("Ambiant strength:")
        color_ambient_label = QLabel("Ambiant color:")
        n_label = QLabel("Lights count:")
        for lbl in (k_ambient_label, color_ambient_label, n_label):
            lbl.setFont(myFont)

        # Other add-hoc items :
        self._preview = Qobject_image(self, self._lighting, height=80)
        self._wget_k_ambient = Atom_QLineEdit(
            self._presenter.ambiant_intensity_type, self._lighting.k_ambient,
            model=None , parent=self
        )
        self._wget_color_ambient = Atom_Color(
            self._presenter.ambiant_color_type, self._lighting.color_ambient,
            model=None , parent=self
        )
        # labels
        param_layout.addWidget(k_ambient_label, 0, 0, 1, 1)
        param_layout.addWidget(color_ambient_label, 1, 0, 1, 1)
        param_layout.addWidget(n_label, 2, 0, 1, 1)
        # wgets
        param_layout.addWidget(self._preview, 3, 0, 1, 2)
        param_layout.addWidget(self._wget_k_ambient, 0, 1, 1, 1)
        param_layout.addWidget(self._wget_color_ambient, 1, 1, 1, 1)
        param_layout.addWidget(self._wget_n, 2, 1, 1, 1)

        self.populate_param_box()
        return param_box

    def populate_param_box(self):
        super().populate_param_box()
        self._wget_k_ambient.setText(str(self._lighting.k_ambient))
        self._wget_color_ambient.update_color(self._lighting.color_ambient)
        self._preview._object = self._lighting
        self._preview.repaint()


    def on_user_mod(self, source, val_callback):
        """ Notify of modification by the user of a Atom widget"""
        val = val_callback()
        self.event_filter(source, val)
        if source == "k_ambient":
            self._wget_k_ambient.on_model_event(val)

#==============================================================================

class Fractal_editor(QWidget):
    """
    Widget for a Fractal parameter
    """
    data_user_modified = pyqtSignal(object, object)

    def __init__(self, parent, data_presenter):
        super().__init__(parent)

        # Model pointers
        self._model = data_presenter._model
        self._mapping = data_presenter._mapping
        self._presenter = data_presenter

        layout = QVBoxLayout()
        f_name = self.f_name = QLineEdit()
        self.populate_fname()
        param_box = self.param_box = QGroupBox("Fractal parameters")
        param_box.setStyleSheet(GROUP_BOX_CSS.format("#32363F"))

        self.populate_param_box()

        layout.addWidget(f_name)
        layout.addWidget(param_box)
        layout.addStretch()
        self.setLayout(layout)
        
        self.data_user_modified.connect(
            data_presenter.data_user_modified_slot
        )
        self._model.model_event.connect(self.model_event_slot)


    def populate_fname(self):
        """ Just a reminder of the fractal class """
        self.f_name.setText(self._presenter.data_class.__name__)

    def populate_param_box(self):
        """ Editor for each parameter """
        wgets = self._wgets = {}
        
        param_layout = QGridLayout()
        data_init_kwargs = self._presenter.data_init_kwargs
        sgn = self._presenter.data_init_signature

        for i_param, (pname, param) in enumerate(sgn.parameters.items()):
            # Adds the paramter name
            name_label = QLabel(pname)
            myFont = QtGui.QFont()
            myFont.setWeight(QtGui.QFont.Weight.ExtraBold)
            name_label.setFont(myFont)
            param_layout.addWidget(name_label, i_param, 0, 1, 1)

            # Adds the paramter value
            val = data_init_kwargs[pname]
            ptype = param.annotation
            if typing.get_origin(ptype) is typing.Union:
                raise NotImplementedError(
                    "Union type not supported in GUI for Fractal __init__ "
                    f"parameter: {pname}"
                )
            wget = wgets[pname] = self.get_wget(pname, ptype, val)
            param_layout.addWidget(wget, i_param, 1, 1, 1)

            # Adds the paramter type
            type_label = QLabel(type_name(ptype))
            param_layout.addWidget(type_label, i_param, 2, 1, 1)
            param_layout.setRowStretch(i_param, 0) # extend at bottom
        
        self.param_box.setLayout(param_layout)

        # Middle column is allocated  all the strech space
        param_layout.setColumnStretch(0, 0)
        param_layout.setColumnStretch(1, 1)
        param_layout.setColumnStretch(2, 0)
        # param_layout.setRowStretch(-1, 1) # extend at bottom
    
    def update_param_box(self, new_fractal):
        """ Updates the individual wget editors according to the new fractal"""
        for pname, wget in self._wgets.items():
            if pname == "directory":
                continue
            val = getattr(new_fractal, pname)
            wget_val = self._presenter.get_wget_val(pname, val)

            
            wget.on_model_event(wget_val)
        
    
    def get_wget(self, pname, ptype, val):
        if pname == "directory":
            # Directory is read-only
            wget = QLabel(val)
            wget.setWordWrap(True)
            wget.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
        else:
            origin = typing.get_origin(ptype)
            if origin is typing.Literal:
                choices = typing_litteral_choices(ptype, p_name=None)
                val = choices.index(val)
            # manage the droplist choice case
            wget = atom_wget_factory(ptype)(ptype, val, None)
            wget.user_modified.connect(functools.partial(
                self.on_user_mod, pname, wget.value
            ))
        return wget

    def on_user_mod(self, pname, val_callback):
        """ Notify the model of modification by the user of a widget"""
        val = wget_val = val_callback()
        wget_val = self._presenter.get_wget_val(pname, val)
        self._wgets[pname].on_model_event(wget_val)
        self.data_user_modified.emit(pname, wget_val)


    def model_event_slot(self, keys, val):
        if keys == self._presenter._mapping["Fractal_presenter"]:
            self.update_param_box(val)
            # Asks user to rester zoom parameters 
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Icon.Question)
            msgBox.setText(
                    "Fractal object parameters have been updated, "
                    "reset zoom to default values ?"
            )
            msgBox.setWindowTitle("Zoom reset Dialog")
            msgBox.setStandardButtons(
                QMessageBox.StandardButton.Ok
                | QMessageBox.StandardButton.Cancel
            )
            user_ret = msgBox.exec()
            if user_ret == QMessageBox.StandardButton.Ok:
                self.reset_zoom_parameters()


    def reset_zoom_parameters(self):
        """ Reset the zoom parameters to default """
        gui = getmainwindow(self)._gui

        full_zoom_keys = Image_widget.full_zoom_keys
        other_parameters = tuple(gui.other_parameters)
        reset_listing = (full_zoom_keys + other_parameters)

        self._presenter.reset_zoom_parameters(gui, reset_listing)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QDict_viewer(QWidget):
    def __init__(self, parent, qdict):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setSpacing(0)
        self.setLayout(self._layout)
        self.widgets_reset(qdict)
        
        self.setSizePolicy(
            QSizePolicy.Policy.Minimum, 
            QSizePolicy.Policy.Fixed
        )

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

#==============================================================================
# Graphics Scene classes
#==============================================================================                
class Zoomable_Drawer_mixin:
    """
    Commom methods that allow to wheel-zoom, and draw objects
    Pass mouse position (self._object_pos, self._object_drag) to the subclasses
    """
    def __init__(self):
        """ Initiate a GaphicsScene """
        # sets graphics scene and view
        self._scene = QGraphicsScene()
        self._group = QGraphicsItemGroup()
        self._view = QGraphicsView()
        self._scene.addItem(self._group)
        self._view.setScene(self._scene)
        self._view.setCursor(QtGui.QCursor(Qt.CursorShape.CrossCursor))
        
        # Initialize the object drawn
        self._object_pos = tuple() # No coords
        self._object_drag = None
        self._drawing = False
        
        # zooms anchors for wheel events - note this is only active 
        # when the image fully occupies the widget
        self._view.setTransformationAnchor(
                QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(
                QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # events filters
        self._view.viewport().installEventFilter(self)
        self._scene.installEventFilter(self)

        # Locker
        self._lock = 0

    # Activates / desactivates locking ========================================
    @pyqtSlot(bool)
    def lock(self, val):
        if val:
            self._lock += 1
        else:
            self._lock -= 1

    @property
    def is_locked(self):
        return self._lock > 0

    # Mouse interaction =======================================================
    def eventFilter(self, source, event):
        # ref: https://doc.qt.io/qt-5/qevent.html
        if source is self._scene:
            if type(event) is QtWidgets.QGraphicsSceneMouseEvent:
                return self.on_viewport_mouse(event)
            elif type(event) is QtGui.QEnterEvent:
                return self.on_enter(event)
            elif (event.type() == QtCore.QEvent.Type.Leave):
                return self.on_leave(event)

        elif source is self._view.viewport():
            # Catch context menu
            if type(event) == QtGui.QContextMenuEvent:
                # return self.on_context_menu(event)
                return True
            elif event.type() == QtCore.QEvent.Type.Wheel:
                return self.on_wheel(event)
            elif event.type() == QtCore.QEvent.Type.ToolTip:
                return True

        return False

    def on_enter(self, event):
        return False

    def on_leave(self, event):
        return False

    def on_wheel(self, event):
        """
        - Updates the zoom
        - Send the current zoom value to `pos_tracker` if exists 
        """
        if self._qim is not None:
            view = self._view
            if event.angleDelta().y() > 0:
                factor = 1.25
            else:
                factor = 0.8
            view.scale(factor, factor)
            if hasattr(self, "pos_tracker"):
                self.pos_tracker(kind="zoom", val=self.zoom)
        return True

    @property
    def zoom(self):
        view = self._view
        pc = 100. * math.sqrt(view.transform().determinant())
        return "{0:.2f} %".format(pc)
    
    def fit_image(self):
        """ Reset the zoom so that the image fits in the widget """
        if self._qim is None:
            return
        rect = QtCore.QRectF(self._qim.pixmap().rect())
        if not rect.isNull():
            # always scrollbars off
            self._view.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._view.setHorizontalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            
            view = self._view
            view.setSceneRect(rect)
            unity = view.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            view.scale(1. / unity.width(), 1. / unity.height())
            viewrect = view.viewport().rect()
            scenerect = view.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            view.scale(factor, factor)
            
            self._view.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self._view.setHorizontalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )

            if hasattr(self, "pos_tracker"):
                self.pos_tracker(kind="zoom", val=self.zoom)


    def on_viewport_mouse(self, event):

        if event.type() == QtCore.QEvent.Type.GraphicsSceneMouseMove:
            self.on_mouse_move(event)
            return True

        elif (event.type() == QtCore.QEvent.Type.GraphicsSceneMousePress
              and event.button() == Qt.MouseButton.LeftButton):
            self.on_mouse_left_press(event)
            return True
        
        elif (event.type() == QtCore.QEvent.Type.GraphicsSceneMousePress
              and event.button() == Qt.MouseButton.RightButton):
            self.on_mouse_right_press(event)
            return True

        elif (event.type() == QtCore.QEvent.Type.GraphicsSceneMouseDoubleClick
              and event.button() == Qt.MouseButton.LeftButton):
            self.on_mouse_double_left_click(event)
            return True

        else:
            return False

    def on_mouse_left_press(self, event):
        """ Left-clicking adds the point and might finish editing if max
        number of points reached """
        if self.is_locked:
            return
        self._drawing = True
        self._object_pos += (event.scenePos(),)
        if len(self._object_pos) == self.object_max_pts:
            self._drawing = False
            self.publish_object()
            self._object_pos = tuple()

    def on_mouse_double_left_click(self, event):
        """ Double-left-clicking finishes editing """
        if self.is_locked:
            return
        self._drawing = False
        self.publish_object()
        self._object_pos = tuple()

    def on_mouse_right_press(self, event):
        pass


    def on_mouse_move(self, event):
        """ 
        - Publish the position to a pos tracker if exists 
        - If object is being drawn, send a draw_object
        """
        if hasattr(self, "pos_tracker"):
            self.pos_tracker(kind="pos", val=event.scenePos())
        if self._drawing:
            self._object_drag = event.scenePos()
            self.draw_object()

    def publish_object(self):
        """ Object has been drawn """
        raise NotImplementedError("Derived classes should implement")

    def draw_object(self):
        """ Object is being drawn """
        raise NotImplementedError("Derived classes should implement")


#==============================================================================
# Base widget for displaying the fractal
#==============================================================================
class Image_widget(QWidget, Zoomable_Drawer_mixin):
    
    param_user_modified = pyqtSignal(object)
    on_fractal_result = pyqtSignal(object, object)

    zoom_keys = ("x", "y", "dx", "xy_ratio", "theta_deg")
    full_zoom_keys = ("x", "y", "nx", "dx", "xy_ratio", "theta_deg", "dps")
    editable_keys = ("x", "y", "dx")

    def __init__(self, parent, view_presenter):
        super().__init__(parent)
        self._parent = parent

        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, 
            QSizePolicy.Policy.Expanding
        )

        self._model = view_presenter._model
        self._mapping = view_presenter._mapping
        self._presenter = view_presenter
        
        # Need dps ?
        self.has_dps = (parent._gui._dps is not None)

        # Sets layout, with only one Widget, the image itself
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        self._layout.addWidget(self._view, stretch=1)
        
        # Sets property widget "image_doc_widget"
        self._labels = QDict_viewer(self,
            {"Image metadata": None, "px": None, "py": None, "zoom": None})
        dock_widget = QDockWidget(None, Qt.WindowType.Window)
        dock_widget.setWidget(self._labels)
        # Not closable :
        dock_widget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable 
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        dock_widget.setWindowTitle("Image")
        dock_widget.setStyleSheet(DOCK_WIDGET_CSS)
        self.image_doc_widget = dock_widget

        # Sets the objects being drawn
        self._rect= None
        self._rect_under = None
        self.object_max_pts = 2

        # Sets Image
        self._qim = None
        self.set_zoom_init(try_reload=True)
        self.set_im()
        
        # Publish / subscribe signals with the submodel
        self._model.model_event.connect(self.model_event_slot)
        self.on_fractal_result.connect(self.fractal_result_slot)


    def on_mouse_right_press(self, event):
        """ Interactive menu with all the defined "interactive_options"
        """
        f = self._presenter["fractal"]
        methods = fs.utils.interactive_options.methods(f.__class__)
        pos = event.scenePos()

        menu = QMenu(self)
        for m in methods:
            action = QAction(m, self)
            menu.addAction(action)
            action.triggered.connect(functools.partial(
                    self.on_fractal_method, f, m, pos))
        menu.popup(event.screenPos())

        return True

    def on_fractal_method(self, f, m_name, pos, evt):
        """ Runs the selected method of the fractal, passing the event coords
        """
        px = pos.x()
        py = pos.y()
        ref_zoom = self._fractal_zoom_init.copy()

        nx = ref_zoom["nx"]
        ny = ref_zoom["ny"]
        dx = mpmath.mpf(ref_zoom["dx"])
        pix = dx / float(ref_zoom["nx"])
        theta = ref_zoom["theta_deg"] * (np.pi / 180.)
        
        # The skew matrice
        skew_params = self.skew_from_ref_zoom(ref_zoom)

        with mpmath.workdps(6):
            # Sets the working dps to 10e-8 x pixel size
            dps = int(-mpmath.log10(pix / nx) + 8)

        with mpmath.workdps(dps):
            dx = mpmath.mpf(ref_zoom["dx"])
            x_center = mpmath.mpf(ref_zoom["x"])
            y_center = mpmath.mpf(ref_zoom["y"])
            
            pix = dx / float(ref_zoom["nx"])
            center_off_px = px - 0.5 * nx
            center_off_py = 0.5 * ny - py
            
            off_x, off_y = self.get_coord_offset(
                center_off_px, center_off_py, pix, theta, *skew_params
            )
            x = x_center + off_x
            y = y_center + off_y

            def thread_job(**kwargs):
                res = getattr(f, m_name)(x, y, pix, dps, **kwargs)
                # Send a signal rather than direct call, to avoid a crash
                # QBasicTimer::start: Timers cannot be started from another
                # thread Segmentation fault (core dumped)
                self.on_fractal_result.emit(m_name, res)

            self._view.setCursor(QtGui.QCursor(Qt.CursorShape.WaitCursor))

            suppl_kwargs = self.method_kwargs(f, m_name)
            if suppl_kwargs is None:
                self._view.setCursor(QtGui.QCursor(Qt.CursorShape.CrossCursor))
            else:
                threading.Thread(target=thread_job,
                                 kwargs=suppl_kwargs).start()

    @staticmethod
    def skew_from_ref_zoom(ref_zoom):
        """ Return the skew params or default value if no skew stored """
        try:
            has_skew = ref_zoom["has_skew"] 
            skew_00 = ref_zoom["skew_00"] 
            skew_01 = ref_zoom["skew_01"] 
            skew_10 = ref_zoom["skew_10"] 
            skew_11 = ref_zoom["skew_11"] 
        except KeyError:
            has_skew = False
            skew_00 = skew_11 = 1.
            skew_01 = skew_10 = 0.
        return has_skew, skew_00, skew_01, skew_10, skew_11

    @staticmethod
    def get_coord_offset(
        center_off_px, center_off_py, pix, theta,
        has_skew, skew_00, skew_01, skew_10, skew_11
    ):
        """ Return the coords offset from the screen offset """
        c = np.cos(theta)
        s = np.sin(theta)
        dx = c * center_off_px - s * center_off_py
        dy = s * center_off_px + c * center_off_py

        # The skew part
        if has_skew:
            tmpx = dx
            tmpy = dy
            dx = skew_00 * tmpx + skew_01 * tmpy
            dy = skew_10 * tmpx + skew_11 * tmpy

        return dx * pix, dy * pix


    def method_kwargs(self, f, m_name):
        """ Collect the additionnal parameters that might be needed"""
        f = getattr(f, m_name)
        # Note: use inspect for a fractal GUI-method 
        sign = inspect.signature(f)

        add_params = dict()
        for i_param, (name, param) in enumerate(sign.parameters.items()):
            if name in ("x", "y", "pix", "dps"):
                # These we already know them
                continue
            ptype = param.annotation
            default = param.default

            if ptype is int:
                if default is inspect.Parameter.empty:
                    default = 0
                int_param, ok = QInputDialog.getInt(
                    None,
                    "Integer input for ball method",
                    f"Enter {name}",
                    value=default,
                )
                if ok:
                    add_params[name] = int_param
                else:
                    return None # Cancel semaphore 
            else:
                raise ValueError(f"Unsupported type {ptype}")

        return add_params


    def fractal_result_slot(self, m_name, res):
        self._view.setCursor(QtGui.QCursor(Qt.CursorShape.CrossCursor))
        res_display = Fractal_code_editor(self)
        res_display.set_text(res)
        res_display.setWindowTitle(f"{m_name} results")
        res_display.show()


    def pos_tracker(self, kind, val):
        """ Updated the displayed info """
        if kind == "pos": # val is event.scenePos()
           self._labels.values_update({"px": val.x(), "py": val.y()})
        elif kind == "zoom":
            self._labels.values_update({"zoom": val})

    @property
    def xy_ratio(self):
        return self._presenter["xy_ratio"]

    @property
    def other_parameters(self):
        return tuple(self._parent._gui.other_parameters)

    def set_zoom_init(self, try_reload=False):
        """ Resets the zoom init, 
         - from the saved pickled files
         - of, if not found, from the script parameters
        """
        # parent is Fractal_MainWindow - func_wget is not initialized at this
        # point, so we use the model itself
        func_sm = self._parent.from_register(("func",))
        if try_reload:
            try:
                func_sm.load_func_dict()
            except FileNotFoundError:
                pass

        # We now load the parameters current value from the func model
        fm_params = func_sm.getkwargs()

        # Setting _fractal_zoom_init from script_params
        gui = self._parent._gui
        self._fractal_zoom_init = dict()
        for key in (self.full_zoom_keys + self.other_parameters):
            # Mapping with func param as defined through `connect_mouse` method
            # of the gui object
            if key == "dps":
                if gui._dps is None:
                    continue
            self._fractal_zoom_init[key] = fm_params[getattr(gui, "_" + key)]

        self._fractal_zoom_init["ny"] = int(
            self._fractal_zoom_init["nx"] / self._fractal_zoom_init["xy_ratio"]
            + 0.5
        )


    def set_im(self):
        """
        This reloads the image and checks that the 
        metadata is matching the expected values from 'last_call'
        """
        image_file = os.path.join((self._presenter["fractal"]).directory, 
                                   self._presenter["image"] + ".png")
        valid_image = True
        try:
            with PIL.Image.open(image_file) as im:
                info = im.info
                nx, ny = im.size

        except FileNotFoundError:
            valid_image = False
            info = dict(zip(self.zoom_keys, (None,) * len(self.zoom_keys)))
            nx = None
            ny = None

        # Storing the "image" zoom info 
        self._image_zoom_init = {
            k: info[k] 
            for k in self.zoom_keys
        }
        self._image_zoom_init["nx"] = nx
        self._image_zoom_init["ny"] = ny
        if self.has_dps:
            self._image_zoom_init["dps"] = info.get(
                "precision", mpmath.mp.dps
            )
        self.validate()

        for item in [self._qim, self._rect, self._rect_under]:
            if item is not None:
                self._group.removeFromGroup(item)

        if valid_image:
            self._qim = QGraphicsPixmapItem(
                QtGui.QPixmap.fromImage(QtGui.QImage(image_file))
            )
            # Antialiasing activated :
            self._qim.setTransformationMode(
                Qt.TransformationMode.SmoothTransformation
            )
            self._qim.setAcceptHoverEvents(True)
            self._group.addToGroup(self._qim)
            self.fit_image()
        else:
            self._qim = None

        self._rect = None
        self._rect_under = None
        self._drawing_rect = False

    @staticmethod
    def cast(val, example):
        """ Casts value to the same type as example """
        return type(example)(val)

    def check_zoom_init(self):
        """ Checks if the image 'zoom init' matches the parameters ;
        otherwise, warns in the image text display
        """
        # We store 2 values of the parameters :
        # the _fractal_zoom_init is a copy of the parameters values "when the 
        # image has been produced"
        # the _presenter value is a convenience to access current parameter
        # value
        # Here, the check performed is to ensure the image metadata matches
        # _fractal_zoom_init
        ret = 0
        keys = self.zoom_keys
        if self.has_dps:
            keys += ("dps",)

        for key in keys:
            expected = self._image_zoom_init[key]
            value = self._fractal_zoom_init[key]
            if expected is None: # No image data
                ret = 2
                break
            else:
                casted = self.cast(value, expected)
                if casted != expected:
                    logger.warning(
                        f"GUI Unmatching image parameter for {key}:\n"
                        f"  {casted} -> {expected}"
                    )
                    ret = 1
        return ret

    def validate(self):
        """ Sets Image metadata message """
        self.validated = self.check_zoom_init()
        message = {0: "OK, matching",
                   1: "/!\ image metadata mismatch",
                   2: "Image data missing"}
        self._labels.values_update({"Image metadata": 
            message[self.validated]})


    def cancel_drawing_rect(self, dclick=False):
        """ Cancellation cases : double-click or empty rectangle
        if dclick, we also reset xy_ratio and theta_deg
        """
        if self._qim is None:
            return
        keys = self.editable_keys
        if dclick:
            keys = self.zoom_keys
        # resets everything - the zoom ratio only if dclick
        for key in keys:
            value = self._fractal_zoom_init[key]
            if value is not None:
                # Send a model modification request
                self._presenter[key] = value
        # Removes the objects
        if self._rect is not None:
            self._group.removeFromGroup(self._rect)
            self._rect = None
        if self._rect_under is not None:
            self._group.removeFromGroup(self._rect_under)
            self._rect_under = None

    def publish_object(self):
        """
        Export the zoom area at model level
        """
        if self._qim is None:
            return

        cancel = False
        dclick = False
        if len(self._object_pos) != 2:
            # We can only be there if double-click
            cancel = True
            dclick = True
        else:
            rect_pos0, rect_pos1 = self._object_pos
            if (rect_pos0 == rect_pos1):
                cancel = True # zoom is invalid

        if cancel:
            self.cancel_drawing_rect(dclick=dclick)
            if dclick:
                self.fit_image()
            return

        nx = self._fractal_zoom_init["nx"]
        ny = self._fractal_zoom_init["ny"]
        theta = self._fractal_zoom_init["theta_deg"] * (np.pi / 180.)

        # The skew matrice
        skew_params = self.skew_from_ref_zoom(self._fractal_zoom_init)

        # new center offet in pixel - independent of angle
        topleft, bottomRight = self.selection_corners(rect_pos0, rect_pos1)
        center_off_px = 0.5 * (topleft.x() + bottomRight.x() - nx)
        center_off_py = 0.5 * (ny - topleft.y() - bottomRight.y())
        dx_pix = abs(topleft.x() - bottomRight.x())

        ref_zoom = self._fractal_zoom_init.copy()

        # str -> mpf as needed
        if self.has_dps:
            to_mpf = {k: isinstance(self._fractal_zoom_init[k], str) for k in
                      ["x", "y", "dx"]}
            # We may need to increase the dps to hold sufficent digits
            if to_mpf["dx"]:
                ref_zoom["dx"] = mpmath.mpf(ref_zoom["dx"])
            pix = ref_zoom["dx"] / float(ref_zoom["nx"])
            with mpmath.workdps(6):
                # Sets the working dps to 10e-8 x pixel size
                ref_zoom["dps"] = int(-mpmath.log10(pix * dx_pix / nx) + 8)
    
            with mpmath.workdps(ref_zoom["dps"]):
                # x, y position is measured on the image -> ref angle is theta
                for k in ["x", "y"]:
                    if to_mpf[k]:
                        ref_zoom[k] = mpmath.mpf(ref_zoom[k])
                        
                off_x, off_y = self.get_coord_offset(
                    center_off_px, center_off_py, pix, theta, *skew_params
                )
                ref_zoom["x"] += off_x
                ref_zoom["y"] += off_y

                ref_zoom["dx"] = dx_pix * pix

                #  mpf -> str (back)
                for (k, v) in to_mpf.items():
                    if v:
                        if k == "dx":
                            ref_zoom[k] = mpmath.nstr(ref_zoom[k], 16)
                        else:
                            ref_zoom[k] = str(ref_zoom[k])
        else:
            ref_zoom["x"] = float(ref_zoom["x"])
            ref_zoom["y"] = float(ref_zoom["y"])
            ref_zoom["dx"] = float(ref_zoom["dx"])

            pix = ref_zoom["dx"] / float(ref_zoom["nx"])
            ref_zoom["x"] += pix * (
                np.cos(theta) * center_off_px
                - np.sin(theta) * center_off_py
            )
            ref_zoom["y"] += pix * (
                np.sin(theta) * center_off_px
                + np.cos(theta) * center_off_py
            )
            ref_zoom["dx"] = dx_pix * pix

        keys = self.editable_keys # Note that theta is not editable by mouse
        if self.has_dps:
            keys += ("dps",)
        for key in keys:
            self._presenter[key] = ref_zoom[key]
            

    def draw_object(self):
        """ Draws the selection rectangle """
        if len(self._object_pos) != 1:
            raise RuntimeError(f"Invalid drawing {self._object_pos}")

        (pos0,) = self._object_pos
        pos1 = self._object_drag

        # Enforce the correct rotation angle
        topleft, bottomRight = self.selection_corners(pos0, pos1)
        rotation = self.selection_rotation()

        rectF = QtCore.QRectF(topleft, bottomRight)
        if self._rect is not None:
            self._rect.setRect(rectF)
            self._rect_under.setRect(rectF)
        else:
            self._rect_under = QGraphicsRectItem(rectF)
            self._rect_under.setPen(QtGui.QPen(QtGui.QColor("red"), 0, Qt.PenStyle.SolidLine))
            self._group.addToGroup(self._rect_under)

            self._rect = QGraphicsRectItem(rectF)
            self._rect.setPen(QtGui.QPen(QtGui.QColor("black"), 0, Qt.PenStyle.DotLine))
            self._group.addToGroup(self._rect)

        # Now apply the rotation
        for r in (self._rect_under, self._rect):
            r.setTransformOriginPoint(r.boundingRect().center())
            r.setRotation(rotation)

    def selection_corners(self, pos0, pos1):
        """ These are the selection rectangle corners """
        # Enforce the correct ratio
        diffx = abs(pos1.x() - pos0.x())
        diffy = abs(pos1.y() - pos0.y())
        # Enforce the correct ratio
        radius_sq = diffx ** 2 + diffy ** 2
        diffx0 = math.sqrt(radius_sq / (1. + 1. / self.xy_ratio ** 2))
        diffy0 = diffx0 / self.xy_ratio
        topleft = QtCore.QPointF(pos0.x() - diffx0, pos0.y() - diffy0)
        bottomRight = QtCore.QPointF(pos0.x() + diffx0, pos0.y() + diffy0)
        return topleft, bottomRight

    def selection_rotation(self):
        """ The rotation angle """
        theta_diff_deg = (
            self._fractal_zoom_init["theta_deg"]
            - self._presenter["theta_deg"]
        )
        return theta_diff_deg

    def model_event_slot(self, keys, val):
        """ A model item has been modified - will it impact the widget ? """
        # Find the matching "mapping" - None if no match
        mapped = next((k for k, v in self._mapping.items() if v == keys), None)
        if mapped in ["image", "fractal"]:
            self.set_zoom_init()
            self.set_im()
        elif mapped in ["x", "y", "dx", "xy_ratio", "theta_deg", "dps"]:
            pass
        else:
            if mapped is not None:
                raise NotImplementedError(
                    "Mapping event not implemented: {}".format(mapped)
                )


#==============================================================================
# Cmap picker from image
#==============================================================================

class Cmap_Image_widget(QDialog, Zoomable_Drawer_mixin):
    model_changerequest = pyqtSignal(object, object)
    param_user_drawn= pyqtSignal(object) # (px1, py1, px2, px2)
    # zoom_params = ["x", "y", "dx", "xy_ratio"]

    def __init__(self, parent, file_path): # im=None):#, xy_ratio=None):
        super().__init__(parent)
        self.setWindowTitle("Cmap creator: from image")
        self.file_path = file_path

        # Sets Image
        self.set_im()
        self._cmap = fs.colors.Fractal_colormap(
            colors=[[0., 0., 0.],
                    [1., 1., 1.]],
            kinds=['Lch'],
            grad_npts=[20],
            grad_funcs=['x'],
            extent='mirror'
        )

        # Sets the objects being drawn
        self._line = None
        self._line_under = None
        self.object_max_pts = 2

        # Sets layout, with only one Widget, the image itself
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        self._layout.addWidget(self._view, stretch=1)
        self._layout.addWidget(self.add_param_box(), stretch=0)
        self._layout.addWidget(self.add_cmap_box(), stretch=0)
        self._layout.addWidget(self.add_action_box(), stretch=0)

        # Event binding
        self._n_grad.valueChanged.connect(self.update_cmap)
        self._n_pts.valueChanged.connect(self.update_cmap)
        self._kind.currentTextChanged.connect(self.update_cmap)
        self._go.clicked.connect(self.display_cmap_code)
        self._push.clicked.connect(self.push_to_param)
        self._save.clicked.connect(self.save)
        
        # Signal / slot
        self.model_changerequest.connect(
                parent._model.model_changerequest_slot)


    def set_im(self):
        self._qim = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(
                QtGui.QImage(self.file_path)))
        self._qim.setAcceptHoverEvents(True)
        self._group.addToGroup(self._qim)
        self.fit_image()
        # interpolator object
        self.im_interpolator = fs.colors.Image_interpolator(
                PIL.Image.open(self.file_path))

    def add_param_box(self):
        param_layout = QHBoxLayout()

        ngrad_layout = QVBoxLayout()
        lbl_n_grad = QLabel("Number of gradients:")
        self._n_grad = QSpinBox()
        self._n_grad.setRange(1, 255)
        self._n_grad.setValue(64)
        ngrad_layout.addWidget(lbl_n_grad)
        ngrad_layout.addWidget(self._n_grad)
        param_layout.addLayout(ngrad_layout)

        npts_layout = QVBoxLayout()
        lbl_n_pts = QLabel("Points per gradient:")
        self._n_pts = QSpinBox()
        self._n_pts.setRange(1, 255)
        self._n_pts.setValue(2)
        npts_layout.addWidget(lbl_n_pts)
        npts_layout.addWidget(self._n_pts)
        param_layout.addLayout(npts_layout)

        kind_layout = QVBoxLayout()
        kind = QLabel("Kind of gradient:")
        self._kind = QComboBox()
        self._kind.addItems(("Lch", "Lab"))
        kind_layout.addWidget(kind)
        kind_layout.addWidget(self._kind)
        param_layout.addLayout(kind_layout)


        action_box = QGroupBox("Parameters")
        action_box.setLayout(param_layout)
        self.set_border_style(action_box)

        return action_box

    def add_action_box(self):
        action_layout = QHBoxLayout()

        go_layout = QVBoxLayout()
        go = QLabel("Gradient code:")
        self._go = QPushButton("Source code")
        go_layout.addWidget(go)
        go_layout.addWidget(self._go)
        action_layout.addLayout(go_layout)

        push_layout = QVBoxLayout()
        push = QLabel("Gradient to parameter:")
        self._push = QPushButton("Push")
        push_layout.addWidget(push)
        push_layout.addWidget(self._push)
        action_layout.addLayout(push_layout)

        save_layout = QVBoxLayout()
        save = QLabel("Save to file:")
        self._save = QPushButton("Save")
        save_layout.addWidget(save)
        save_layout.addWidget(self._save)
        action_layout.addLayout(save_layout)

        action_box = QGroupBox("Actions")
        action_box.setLayout(action_layout)
        self.set_border_style(action_box)

        return action_box

    def display_cmap_code(self):
        """ displays source code for Fractal object """
        ce = Fractal_code_editor(self)
        str_args = self._cmap.script_repr()
        ce.set_text(str_args)
        ce.setWindowTitle("Fractal code")
        ce.show()

    def push_to_param(self):
        """ Try to push to a colormap param if there is one """
        # sign = inspect.signature(self.parent()._gui._func)
        sign = fs.gui.guitemplates.signature(
                self.parent()._gui._func
        )
        
        cmap_params_index = dict()
        for i_param, (name, param) in enumerate(sign.parameters.items()):
            if param.annotation is fs.colors.Fractal_colormap:
                cmap_params_index[name] = i_param

        if len(cmap_params_index) == 0:
            raise RuntimeError("No fs.colors.Fractal_colormap parameter")

        elif len(cmap_params_index) == 1:
            i_param = next(iter(cmap_params_index.values()))
        else:
            params = list(cmap_params_index.keys())
            param, ok = QInputDialog.getItem(self, "Select parameter", 
                "available parameters", params, 0, False)
            if ok and param:
                 i_param = cmap_params_index[param]
            else:
                return

        self.model_changerequest.emit(
                ("func", (i_param, 0, "val")), self._cmap
        )

    def save(self):
        """ Save the camp as a pickle .cmap file"""
        cmap = self._cmap
        mainw = self.parent()

        if cmap is not None:
            file_path = mainw.gui_file_path(
                    _filter="Colormap (*.cmap)", mode="save"
            )
            if file_path == "":
                return 

            root, ext = os.path.splitext(file_path)
            if ext != "cmap":
                file_path = root + "." + "cmap"
                logger.info(f"Changed cmap save file to: {file_path}")

            cmap.save_as_pickle(file_path)

    def add_cmap_box(self):
        cmap_layout = QHBoxLayout()
        
        self._preview = Qobject_image(self, self._cmap)
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(self._preview, stretch=1)

        cmap_box = QGroupBox("Colormap")
        cmap_box.setLayout(cmap_layout)
        self.set_border_style(cmap_box)
        return cmap_box

    def set_border_style(self, gb):
        """ adds borders to an action box"""
        gb.setStyleSheet(GROUP_BOX_CSS.format("#32363F"))

    def cancel_drawing_line(self):
        # Removes the objects
        if self._line is not None:
            self._group.removeFromGroup(self._line)
            self._line = None
        if self._line_under is not None:
            self._group.removeFromGroup(self._line_under)
            self._line_under = None

    def publish_object(self):
        """
        Create a colormap from the line
        """
        cancel = False
        dclick = False
        if len(self._object_pos) != 2:
            # We can only be there if double-click
            cancel = True
            dclick = True
        else:
            line_pos0, line_pos1 = self._object_pos
            if (line_pos0 == line_pos1):
                cancel = True # zoom is invalid

        if cancel:
            self.cancel_drawing_line()
            if dclick:
                self.fit_image()
            return

        self.validated_object_pos = self._object_pos
        self.update_cmap()
        

    def update_cmap(self):
        if not(hasattr(self, "validated_object_pos")):
            # Nothing to update
            return

        (pos0, pos1) = self.validated_object_pos
        x0, y0 = pos0.x(), pos0.y()
        x1, y1 = pos1.x(), pos1.y()
        n_grad, npts = self._n_grad.value(), self._n_pts.value()
        kinds = str(self._kind.currentText())

        x = np.linspace(x0, x1, n_grad + 1)
        y = np.linspace(y0, y1, n_grad + 1)
        colors = self.interpolate(x, y) / 255.

        # Creates the new cmap widget, replace the old one (in-place)
        self._cmap = fs.colors.Fractal_colormap(
            colors=colors,
            kinds=kinds,
            grad_npts=npts,
            grad_funcs='x',
            extent='mirror'
        )
        new_cmap = Qobject_image(self, self._cmap)
        containing_layout = self._preview.parent().layout()
        containing_layout.replaceWidget(self._preview, new_cmap)
        self._preview = new_cmap

    def interpolate(self, x, y):
        return self.im_interpolator.interpolate(x, y)

    def draw_object(self):
        """ Draws the selection rectangle """
        
        if len(self._object_pos) != 1:
            raise RuntimeError(f"Invalid drawing {self._object_pos}")
        
        (pos0,) = self._object_pos
        pos1 = self._object_drag 
        qlineF = QtCore.QLineF(pos0, pos1)
        if self._line is not None:
            self._line.setLine(qlineF)
            self._line_under.setLine(qlineF)
        else:
            self._line_under = QGraphicsLineItem(qlineF)
            self._line_under.setPen(QtGui.QPen(QtGui.QColor("red"), 0, Qt.PenStyle.SolidLine))
            self._group.addToGroup(self._line_under)

            self._line = QGraphicsLineItem(qlineF)
            self._line.setPen(QtGui.QPen(QtGui.QColor("black"), 0, Qt.PenStyle.DotLine))
            self._group.addToGroup(self._line)


#==============================================================================

class Fractal_MessageBox(QMessageBox):
    def __init__(self, *args, **kwargs):            
        super().__init__(*args, **kwargs)
        self.setStyleSheet("QLabel{min-width: 700px;}")

    def resizeEvent(self, event):
        result = super().resizeEvent(event)
        details_box = self.findChild(QTextEdit)
        if details_box is not None:
            details_box.setFixedHeight(details_box.sizeHint().height())
        return result

class Fractal_cmap_choser(QDialog):
    
    model_changerequest = pyqtSignal(object, object)

    def __init__(self, parent):            
        super().__init__(parent)
        self.setWindowTitle("Chose a colormap ...")

        self.setStyleSheet("QLabel{min-width: 700px;}")
        self.cmap_list = list(fs.colors.cmap_register.keys())
        
        cmap_combo = QComboBox(self)
        cmap_combo.addItems(self.cmap_list)
        cmap_combo.setCurrentIndex(0)
        cmap_combo.setStyleSheet(COMBO_BOX_CSS)
        cmap_combo.currentTextChanged.connect(self.on_combo_event)
        
        self.cmap_name = cmap_name = self.cmap_list[0]
        cmap0 = fs.colors.cmap_register[cmap_name]
        self.cmap = cmap = Qobject_image(self, cmap0, minwidth=400, height=20)

        push = QPushButton("Push to parameter")
        push.clicked.connect(self.push_to_param)

        self.layout = layout = QVBoxLayout()
        layout.addWidget(cmap_combo)
        layout.addWidget(cmap)
        layout.addWidget(push)
        self.setLayout(layout)
        
        # Signal / slot
        self.model_changerequest.connect(
                parent._model.model_changerequest_slot
        )

    @property
    def cmap_parameter(self):
        return fs.colors.cmap_register[self.cmap_name]

    def push_to_param(self):
        """ Try to push to a colormap param if there is one """
        sign = fs.gui.guitemplates.signature(
                self.parent()._gui._func
        )
        
        cmap_params_index = dict()
        for i_param, (name, param) in enumerate(sign.parameters.items()):
            if param.annotation is fs.colors.Fractal_colormap:
                cmap_params_index[name] = i_param
        
        if len(cmap_params_index) == 0:
            raise RuntimeError("No fs.colors.Fractal_colormap parameter")

        elif len(cmap_params_index) == 1:
            i_param = next(iter(cmap_params_index.values()))
        else:
            params = list(cmap_params_index.keys())
            param, ok = QInputDialog.getItem(
                self, "Select parameter", 
                "available parameters", params, 0, False
            )
            if ok and param:
                 i_param = cmap_params_index[param]
            else:
                return

        self.model_changerequest.emit(
            ("func", (i_param, 0, "val")),
            copy.deepcopy(self.cmap_parameter) # We copy to make it editable
        )
        self.close()

    def on_combo_event(self, event):
        self.cmap_name = event
        new_cmap = Qobject_image(self, fs.colors.cmap_register[event],
                                minwidth=400, height=20)
        self.layout.replaceWidget(self.cmap, new_cmap)
        self.cmap = new_cmap
        


class Fractal_MainWindow(QMainWindow):
    
    model_changerequest = pyqtSignal(object, object)
    
    def __init__(self, gui):
        super().__init__(parent=None)
        self.setStyleSheet(MAIN_WINDOW_CSS)
        self.build_model(gui)
        self.layout()
        self.set_menubar()
        self.setWindowTitle(f"Fractashades {fs.__version__}")
        if fs.settings.output_context["doc"] or True:
            # Needed for github build where QT_QPA_PLATFORM=offscreen
            self.setMinimumSize(1200, 744 - 24)
        
        # Signal / slot
        self.model_changerequest.connect(
                self._model.model_changerequest_slot
        )

    def set_menubar(self) :
      bar = self.menuBar()

      tools = bar.addMenu("Tools")
      layers_data = QAction('View Layers data', tools)
      clear_cache = QAction('Clear calculation cache', tools)
      png_info = QAction("Show png info", tools)
      tools.addActions((layers_data, clear_cache, png_info))
      tools.triggered[QAction].connect(self.actiontrig)
      
      cmap = bar.addMenu("Colormaps")
      png_cbar = QAction('Colormap from png image', cmap)
      template_cbar = QAction('Colormap from templates', cmap)
      save_cmap_cbar = QAction('Save cmap (.cmap)', cmap)
      load_cmap_cbar = QAction('Load cmap (.cmap)', cmap)
      cmap.addActions((png_cbar, template_cbar, save_cmap_cbar,
                       load_cmap_cbar))
      cmap.triggered[QAction].connect(self.actiontrig)

      about = bar.addMenu("About")
      license_txt = QAction('License', about)
      about.addAction(license_txt)
      about.triggered[QAction].connect(self.actiontrig)

    def actiontrig(self, action):
        """  Dispatch the action to the matching method
        """
        txt = action.text()
        if txt == "View Layers data":
            self.layers_data()
        elif txt == "Clear calculation cache":
            self.clear_cache()
        elif txt == "Show png info":
            self.show_png_info()
        elif txt == "Colormap from png image":
            self.cmap_from_png()
        elif txt == "Colormap from templates":
            self.cmap_from_template()
        elif txt == "Save cmap (.cmap)":
            self.save_cmap()
        elif txt == "Load cmap (.cmap)":
            self.load_cmap()
        elif txt == "License":
            self.show_license()
        else:
            print("Unknow actiontrig")

    def show_license(self):
        """
        Displays the program license
        """
        fs_resources = importlib_resources.files("fractalshades")
        with importlib_resources.as_file(
            fs_resources / "data" / "LICENSE.txt"
        ) as license_file:
            with open(str(license_file.resolve())) as f:
                license_str = f.read()

        msg = Fractal_MessageBox()
        msg.setWindowTitle("Fractalshades " + fs.__version__)
        msg.setText(license_str.splitlines()[0])
        msg.setInformativeText(license_str.splitlines()[2])
        msg.setDetailedText(license_str)
        msg.exec()
    
    def gui_file_path(self, _filter=None, mode="open"):
        """
        Load a file, browsing from the __main__ directory 
        """
        try:
            import __main__
            script_dir = os.path.abspath(os.path.dirname(__main__.__file__))
        except NameError:
            script_dir = None
        if mode == "open":
            file_path = QFileDialog.getOpenFileName(
                    self,
                    directory=script_dir,
                    caption="Select File",
                    filter=_filter
            )
        elif mode == "save":
            file_path = QFileDialog.getSaveFileName(
                    self,
                    directory=script_dir,
                    caption="Save File",
                    filter=_filter
            )
            
        if isinstance(file_path, tuple):
            file_path = file_path[0]
        return file_path


    def show_png_info(self):
        """
        Loads an image file and displays the associated tag info
        """
        file_path = self.gui_file_path(_filter="Images (*.png)")
        if file_path == "":
            return
        with PIL.Image.open(file_path) as im:
            png_info = im.info
        data_len = len(png_info)
        info = ""
        for key, val in png_info.items():
            info += f"{key} = {val}\n"
        msg = Fractal_MessageBox()
        msg.setWindowTitle("Image metadata")
        msg.setText(file_path)
        msg.setInformativeText(f"Number of fields found: {data_len}")
        msg.setDetailedText(info)
        msg.exec()
        
    def cmap_from_png(self):
        file_path = self.gui_file_path(_filter="Images (*.png)")
        if file_path == "":
            return
        image_display = Cmap_Image_widget(self, file_path)
        image_display.exec()

    def cmap_from_template(self):
        """ Dialog to chose a cmap """
        choser = Fractal_cmap_choser(self)
        choser.exec()

    def clear_cache(self):
        func_submodel = self.from_register(("func",))
        fractal = next(iter(func_submodel.getkwargs().values()))
        fractal.clean_up()
        msg = Fractal_MessageBox()
        msg.setWindowTitle("Cache cleared")
        msg.setText("Directory :")
        msg.setInformativeText(f"{fractal.directory}")
        msg.exec()
    
    def layers_data(self):
        """ display in GUI the file with layers info"""
        func_submodel = self.from_register(("func",))
        fractal = next(iter(func_submodel.getkwargs().values()))
        txt_report_path = fractal.txt_report_path
        txt_report_file = os.path.basename(txt_report_path)

        with open(txt_report_path, "r") as f:
            report = f.read()

        ce = Fractal_code_editor(self)
        ce.set_text(report)
        ce.setWindowTitle(f"Layers info [./{txt_report_file}]")
        ce.show()


    def save_cmap(self):
        """ Select one of the parameters of type Colormap, and save its as
        a .cmap file"""
        key, cmap = self.chose_cmap_param()
        if cmap is not None:
            file_path = self.gui_file_path(
                    _filter="Colormap (*.cmap)", mode="save"
            )
            if file_path == "":
                return 

            root, ext = os.path.splitext(file_path)
            if ext != "cmap":
                file_path = root + "." + "cmap"
                logger.info(f"Changed cmap save file to: {file_path}")

            cmap.save_as_pickle(file_path)

    def load_cmap(self):
        """ Load a Colormap from a .cmap file, and push it to one of the
        parameters """
        file_path = self.gui_file_path(_filter="Colormap (*.cmap)")
        if file_path == "":
            return 

        key = self.chose_cmap_param(with_val=False)
        if key is not None:
            cmap = fs.colors.Fractal_colormap.load_as_pickle(file_path)
            self.model_changerequest.emit(key, cmap)


    def chose_cmap_param(self, with_val=True):
        """ Return the model-evel key associated with one of the 
        cmap parameters (user can chose if there are multiple)
        + its current value
        """
        sign = fs.gui.guitemplates.signature(self._gui._func)

        cmap_params_index = dict()
        for i_param, (name, param) in enumerate(sign.parameters.items()):
            if param.annotation is fs.colors.Fractal_colormap:
                cmap_params_index[name] = i_param

        if len(cmap_params_index) == 0:
            raise RuntimeError("No fs.colors.Fractal_colormap parameter")

        elif len(cmap_params_index) == 1:
            i_param = next(iter(cmap_params_index.values()))
        else:
            params = list(cmap_params_index.keys())
            param, ok = QInputDialog.getItem(
                self, "Select parameter", 
                "available parameters", params, 0, False
            )
            if ok and param:
                 i_param = cmap_params_index[param]
            else:
                if with_val:
                    return None, None
                return None

        key = ("func", (i_param, 0, "val"))
        if not(with_val):
            return key

        val = self._model[key]
        return key, val


    def build_model(self, gui):
        self._gui = gui
        model = self._model = Model()
        # Adds the submodels
        model.register(
            Func_submodel(model, ("func",), gui._func, dps_var=gui._dps),
            ("func",)
        )
        # Adds the presenters - Note
        mapping = {
            "fractal": ("func", gui._fractal),
            "image": ("func", gui._image),
            "x": ("func", gui._x),
            "y": ("func", gui._y),
            "dx": ("func", gui._dx),
            "xy_ratio": ("func", gui._xy_ratio),
            "theta_deg": ("func", gui._theta_deg),
            "dps": ("func", gui._dps)
        }
        # mapping other parameters (skew, ...):
        for param in gui.other_parameters:
            attr = "_" + param
            mapping[param] = ("func", getattr(gui, attr))

        model.register(
            Presenter(model, mapping), register_key="image"
        )

    def layout(self):
        self.add_status_bar()
        self.add_image_wget()
        self.add_func_wget()
        self.add_image_status()
    
    def sizeHint(self):
        return QtCore.QSize(1200, 800) 
    
    def add_image_status(self):
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self.centralWidget().image_doc_widget)

    def add_func_wget(self):
        action_setting = (
            "image_updated", 
            self.from_register("image")._mapping["image"]
        )
        func_wget = Action_func_widget(
            self,
            self.from_register(("func",)),
            action_setting,
            callback=True,
            may_interrupt=True,
            locks_navigation=True
        )
        dock_widget = QDockWidget(None, Qt.WindowType.Window)
        dock_widget.setWidget(func_wget)
        # Not closable :
        dock_widget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        dock_widget.setWindowTitle("Parameters")
        dock_widget.setStyleSheet(DOCK_WIDGET_CSS)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_widget)

        self._func_wget = func_wget
        
        if fs.settings.output_context["doc"]:
            # We are building the doc we need an image
            func_wget.run_func()

    def add_status_bar(self):
        # the status bar need access to the fractal object, which will be
        # provided through the func model
        self.status_bar = Calc_status_bar(self.from_register(("func",)))
        self.setStatusBar(self.status_bar)

    @pyqtSlot(object)
    def func_callback(self, func_widget):
        """ A simple callback when the main computation is finished """
        # output gui-image (for documentation)
        if fs.settings.output_context["doc"]: 
            time.sleep(1)
            img = self.grab()
            fs.settings.add_figure(_Pixmap_figure(img))
            QApplication.quit()

    @pyqtSlot(Exception)
    def on_error_in_thread(self, exc):
        """ A simple callback when error occured in computation computation
        """
        raise exc

    def add_image_wget(self):
        mw = Image_widget(self, self.from_register("image"))
        self.setCentralWidget(mw)

    def from_register(self, register_key):
        return self._model._register[register_key]


def excepthook(exc_type, exc_value, exc_traceback):
    """ Handling GUI Exceptions"""
    exc_str = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )
    QMessageBox.critical(None, 'GUI Error', exc_str)


class Fractal_GUI:
    def __init__(self, func):
        """
Parameters
----------
func : callable with signature (`fs.Fractal`, ``**kwargs`` ). 
    The function definition shall provide 'type hints' that will be used by the
    the GUI to select / customize the appropriate editor.
    Each parameter will be displayed interactively and will be editable.
    The editor might be a simple text box, or for more complex objects
    a full pop-up or a dockable window.

    The first parameter shall be a `fractalshades.Fractal` object, and its name
    shall be "fractal".
    It is also partially editable: the parameters from
    the __init__ method will be user-tunable (it is not possible to change
    the fractal type or the base directory during a GUI interactive session)

Notes
-----     

Theses notes give further details regarding the definition of the `func`
parameter

.. note::

    Regarding ``Type hints`` in Python, for a full specification, please see
    PEP_484_. For a simplified introduction to type hints, see PEP_483_.
    Fractalshades only support a subset of these, details below.

    .. _PEP_484: https://www.python.org/dev/peps/pep-0484/
    .. _PEP_483: https://www.python.org/dev/peps/pep-0483/

.. note::
    
    Currently the following parameters types are supported :
        - `float`
        - `int`
        - `bool`
        - `mpmath.mpf` (arbitrary precision floating point)
        - `fs.colors.Color=(0., 0., 1.)` (RGB color)
        - `fs.colors.Color=(0., 0., 1., 0)` (RGBA color)
        - `fs.numpy_utils.Numpy_expr` (defines a numpy function from string)
        - `fs.Fractal` subclasses
        - `fs.colors.Fractal_colormap`
        - `fs.colors.Blinn_lighting`
        - `fs.gui.separator` (used to group a set of parameters under a title)
        - `fs.gui.collapsible_separator` (same as above, collapsible)

    A parameter that the user will chose among a list of discrete values can
    be represented by a `typing.Literal` :

    ::

        listed: typing.Literal["a", "b", "c", 1, None]="c"
    
    Also, `typing.Union` or `typing.Optional` derived of supported  base types
    are supported (in this case a combo box to chose one type among
    those authorized be available):
    
    ::

        typing.Union[int, float, str]
        typing.Optional[float]

.. note::
    
    An example of a valid ``func`` parameter signature is show below:
    
    ::
        
        import typing
        import fractalshades.models as fsm

        def func(
            fractal: fsm.Perturbation_mandelbrot=fractal,
            calc_name: str=calc_name,
            _1: fsgui.separator="Zoom parameters",
            x: mpmath.mpf=x,
            y: mpmath.mpf=y,
            dx: mpmath.mpf=dx,
            xy_ratio: float=xy_ratio,
            dps: int= dps,
            _2: fsgui.collapsible_separator="Calculation parameters",
            max_iter: int=max_iter,
            optional_float: typing.Optional[float]=3.14159,
            choices_str: typing.Literal["a", "b", "c"]="c",
            nx: int=nx,
            interior_detect: bool=interior_detect,
            interior_color: QtGui.QColor=(0., 0., 1.),
            transparent_color: QtGui.QColor=(0., 0., 1., 0.),
            probes_zmax: float=probes_zmax,
            epsilon_stationnary: float=epsilon_stationnary,
            colormap: fs.colors.Fractal_colormap=colormap
            func: fs.numpy_utils.Numpy_expr = (
                fs.numpy_utils.Numpy_expr("x", "np.log(x)")
            ),
        ):

"""
        self._func = func
        sign = fs.gui.guitemplates.signature(func)
        param_names = sign.parameters.keys()
        param0 = next(iter(param_names))
        self._fractal = param0

        if param0 != "fractal":
            raise ValueError(
                f"Expected a first parameters named fractal, found: {param0}"
            )

        if isinstance(func, fs.gui.guitemplates.GUItemplate):
            self.connect_image(**func.connect_image_params)
            self.connect_mouse(**func.connect_mouse_params)

    def connect_image(self, image_param="calc_name"):
        """
Associate a image file with the GUI main diplay

Parameters
---------- 
image_param: str
    Name of the image file to display. This image file shall be created by the
    function ``func`` in the same directory as the main script.
        """
        self._image = image_param

    def connect_mouse(
        self, x="x", y="y", dx="dx", nx="nx", xy_ratio="xy_ratio",
        theta_deg="theta_deg", dps="dps", **kwargs
    ):
        """
Binds some parameters of the ``func`` passed to the
`fractalshades.gui.Fractal_GUI` constructor with GUI mouse events.

Parameters
---------- 
x: str
    Name of the parameter for the x-axis center of the image
y: str
    Name of the parameter for the y-axis center of the image
dx: str
    Name of the parameter for the x-axis width of the image
nx: str
    Name of the parameter for the x-axis width of the image
xy_ratio: str
    Name of the parameter for the ratio width / height of the image
dps: str | None
    Name of the parameter for the precision in base-10 digits (mpmath arbitrary
    precision). If not using arbitrary precision, it is NEEDED to pass None.
theta_deg: str
    Name of the parameter for the image rotation angle in degree.
other_parameters: dict
    Pairs of (key, value) for additionnal parameters (skew, ...)
"""
        self._x, self._y, self._dx, self._nx = x, y, dx, nx
        self._xy_ratio, self._dps = xy_ratio, dps
        self._theta_deg = theta_deg

        # Other parameters: skew, ...
        for key, val in kwargs.items():
            attr = "_" + key
            setattr(self, attr, key)
        self.other_parameters = kwargs.keys()


    def show(self):
        """
        Launches the GUI mainloop.
        """
        app = getapp()
        self.mainwin = Fractal_MainWindow(self)
        self.mainwin.show()
        fs.settings.output_context["gui_iter"] = 1
        sys.excepthook = excepthook
        try:
            app.exec()
        finally:
            fs.settings.output_context["gui_iter"] = 0
            sys.excepthook = sys.__excepthook__
