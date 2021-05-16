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
                              QLineEdit, QStackedWidget, QGroupBox,
                             QGridLayout, QSpacerItem, QSizePolicy,
                             QGraphicsScene, QGraphicsView,
                             QGraphicsPixmapItem, QGraphicsItemGroup,
                             QGraphicsRectItem, QFrame
                             )
from PyQt5.QtWidgets import (QMainWindow, QApplication)
#
from inspector import Func_submodel, type_name
import functools
import copy
from operator import getitem, setitem
import mpmath


class Model(QtCore.QObject):
    """
    Model composed of nested dict. An item of this dict is identified by a 
    list of nested keys.
    """
    model_event = pyqtSignal(object, object) # key, newval

    def __init__(self, source=None):
        super().__init__()
        self._model = source
        if self._model is None:
            self._model = dict()
            
#    def add_submodel(self, submodel):
#        self._model[submodel.keys] = submodel

    def __setitem__(self, keys, val):
        # Sets an item value. keys iterable of nested keys
        setitem(functools.reduce(getitem, keys[:-1], self._model),
                keys[-1], val)
        # self.model_item_updated.emit(keys)

    def __getitem__(self, keys):
        # Gets an item value. keys is an iterable of nested keys
        return functools.reduce(getitem, keys, self._model)
    
    @pyqtSlot(object, object, object)
    def model_notified_slot(self, keys, oldval, val):
        print("Model widget_modified", keys, oldval, val)
        # Here we can implement UNDO / REDO stack
        self.model_event.emit(keys, val)


class Submodel(QtCore.QObject):
    """
    A submodel holds the data for a model key.
    Pubsub model
       On user modification the submodel
           - Notifies the model of the modifications that need to be
             commited
    """
    @property
    def keys(self): return self._keys
    def __setitem__(self, keys, val): raise NotImplementedError()
    def __getitem__(self, keys): raise NotImplementedError()
    


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

class Func_widget(QWidget):
    # Signal to inform the model that a parameter has been modified by the 
    # user.
    # item_mod_evt = pyqtSignal(tuple, object)
    func_user_modified = pyqtSignal(object, object)

    def __init__(self, parent, func_smodel):#model, func_keys):
        super().__init__(parent)
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
        fd = self._submodel.func_dict
        print("fd", fd)
        n_params = fd["n_params"]
        for i_param in range(n_params):
            self.layout_param(i_param)

    def layout_param(self, i_param):
        fd = self._submodel.func_dict
        
        name = fd[(i_param, "name")]
        name_label = QLabel(name)
        myFont = QtGui.QFont()#QtGui.QFont()
        myFont.setWeight(QtGui.QFont.ExtraBold)
        name_label.setFont(myFont)
        self._layout.addWidget(name_label, i_param, 0, 1, 1)

        # Adds a check-box for default
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

        
        fd = self._submodel.func_dict
        # n_uargs = fd[(i_param, "n_types")]
        utype = fd[(i_param, i_union, "type")]
        if dataclasses.is_dataclass(utype):
            for ifield, field in enumerate(dataclasses.fields(utype)):
                self.layout_field(qs, i_param, i_union, ifield)
        else:
            uval = fd[(i_param, i_union, "val")]
            print("UVAL", uval)
            atom_wget = atom_wget_factory(utype)(utype, uval)
            self._widgets[(i_param, i_union, "val")] = atom_wget
            print("atom_wget", atom_wget, type(atom_wget))
            atom_wget.user_modified.connect(functools.partial(
                    self.on_user_mod, (i_param, i_union, "val"),
                    atom_wget.value))
            qs.addWidget(atom_wget)#, i_param, 3, 1, 1)
            
    
    def layout_field(self, qs, i_param, i_union, ifield):
        fd = self._submodel.func_dict
        pass
    
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
        print("item evt",  key, val)
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
        if isinstance(wget, Atom_Mixin):
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
    else:
        wget_dic = {int: Atom_QLineEdit,
                    float: Atom_QLineEdit,
                    str: Atom_QLineEdit,
                    mpmath.mpf: Atom_QLineEdit,
                    type(None): Atom_QLineEdit}
        return wget_dic[atom_type]
    
class Atom_Mixin:
    pass

class Atom_QLineEdit(QLineEdit, Atom_Mixin): 
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, parent=None):
        super().__init__(str(val), parent)
        self._type = atom_type

        self.textChanged[str].connect(self.validate)
        self.editingFinished.connect(self.on_user_event)
        self.setValidator(Atom_QLineEdit_Validator(atom_type))
        if atom_type is type(None):
            self.setReadOnly(True)

    def value(self):
        if self._type is type(None):
            return None
        return self._type(self.text())
        

    def on_user_event(self):
        print("ATOMIC UPDATED from user")
        self.user_modified.emit()
    
    def on_model_event(self, val):
        print("ATOMIC UPDATED from model", val, type(val))
        self.setText(str(val))
        # self.reset_bgcolor()
        self.validate(self.text(), QtGui.QColor.fromRgb(255, 255, 255))
        

    def validate(self, text,
                 acceptable_color=QtGui.QColor.fromRgb(200, 200, 200)):
        validator = self.validator()
        if validator is not None:
            ret, _, _ = validator.validate(text, self.pos())
            p = self.palette()
            if ret == QtGui.QValidator.Acceptable:
                p.setColor(self.backgroundRole(), acceptable_color)
                           # QtGui.QColor.fromRgb(200, 200, 200))
            else:
                p.setColor(self.backgroundRole(),
                           QtGui.QColor.fromRgb(220, 70, 70))
            self.setPalette(p)



class Atom_QComboBox(QComboBox, Atom_Mixin): #, QComboBox): # QLabel, 
    user_modified = pyqtSignal()

    def __init__(self, atom_type, val, parent=None):
        super().__init__(parent)
        self._type = atom_type
        self._choices = typing.get_args(atom_type)
        self.currentTextChanged.connect(self.on_user_event)
        self.addItems(self._choices)
        self.setCurrentIndex(self.findText(val))

    def value(self):
        return self.currentText()

    def on_user_event(self):
        print("ATOMIC UPDATED from user")#, val, type(val))
        self.user_modified.emit()
    
    def on_model_event(self, val):
        print("ATOMIC UPDATED from model", val, type(val))
        self.setCurrentIndex(self.findText(val))



class Atom_QLineEdit_Validator(QtGui.QValidator):
    def __init__(self, atom_type):
        super().__init__()
        self._type = atom_type

    def validate(self, val, pos):
        valid = {True: QtGui.QValidator.Acceptable,
                 False: QtGui.QValidator.Intermediate}
        if self._type is type(None):
            return (valid[val == "None"], val, pos)
        try:
            casted = self._type(val)
        except ValueError:
            return (QtGui.QValidator.Intermediate, val, pos)
        return (valid[isinstance(casted, self._type)], val, pos)



def getapp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    return app



def test_Inspector_widget():
    
    def f_atomic(x: int=1, yyyy: float=10., y: str="aa", z:float=1.0):
        pass 

    def f_union(x: int=1, yyyy: float=10., y: str="aa", z: typing.Union[int, float, str]="abc"):
        pass 

    def f_optional(x: int=1, yyyy: float=10., y: str="aa", z: typing.Union[int, float, str]="abc",
                option: typing.Optional[float]=12.354, option2: typing.Optional[float]=None):
        pass 
    
    def f_listed(x: int=1, yyyy: float=10., y: str="aa", z: typing.Union[int, float, str]="abc",
                option: typing.Optional[float]=12.354, option2: typing.Optional[float]=None,
                listed: typing.Literal["a", "b", "c"]="a"):
        pass 
    
    func = f_listed
    
    
    class Mywindow(QMainWindow):
        def __init__(self):
            super().__init__(parent=None)

            self.setWindowTitle('Testing inspector')
            tb = QToolBar(self)
            self.addToolBar(tb)
#            print_dict = QAction("print dict")
            tb.addAction("print_dict")
            
            # tb.actionTriggered[QAction].connect(self.on_tb_action)
            tb.actionTriggered.connect(self.on_tb_action)
            #self.setWindowState(Qt.WindowMaximized)
            # And don't forget to call setCentralWidget to your main layout widget.
             # fsgui.
            model = Model()
            func_smodel = Func_submodel(model, tuple(["func"]), func)
            print("#################", model)
            wdget = Func_widget(self, func_smodel)

            self.setCentralWidget(wdget)
            self._wdget = wdget
            # self.setFixedSize(800, 800)

        def on_tb_action(self, qa):
            print("qa", qa)
            d = self._wdget._submodel.func_dict
            for k, v in d.items():
                print(k, " --> ", v)


    app = getapp()
    win = Mywindow()
    win.show()
    app.exec()
    
    
    

if __name__ == "__main__":
    test_Inspector_widget()