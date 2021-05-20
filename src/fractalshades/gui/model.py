# -*- coding: utf-8 -*-
import inspect
import typing
import functools
import dataclasses
from operator import getitem, setitem

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot



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
        """ A change has been done, need to notify the widgets """
        print("Model widget_modified", keys, oldval, val)
        # Here we can implement UNDO / REDO stack
        self.model_event.emit(keys, val)

    @pyqtSlot()
    def model_changerequest_slot(self, keys, oldval, val):
        """ A change has been requested by a widget,
        connected to a presenter i.e not holding the data"""
        print("Model widget_change request", keys, oldval, val)
        submodel_keys = keys[:-1]
        if len(submodel_keys) == 0:
            # This change can be handled at model level
            self.model[keys] = val
            self.model_notified.emit(keys, oldval, val)
        else:
            # This change needs to be handled at sub-model level
            raise NotImplementedError()

    



    

class Submodel(QtCore.QObject):
    """
    A submodel holds the data for a widget, stored in self.dict
    Pubsub model
    """
    def __init__(self, model, submodel_keys):
        super().__init__()  # Mandatory for the signal / slot machinery !
        self._model = model
        self._keys = submodel_keys

    def __getitem__(self, key):
        return getitem(self._dict, key)

    def __setitem__(self, key, val):
        oldval = self[key]
        setitem(self._dict, key, val)
        self.model_notification.emit(self._keys + tuple([key]), oldval, val)  


class Presenter(QtCore.QObject):
    """
    A presenter holds a mapping to the real model data
    """
    # key, oldval, newval signal
    model_notification = pyqtSignal(object, object, object)

    def __init__(self, model, mapping):
        super().__init__()  # Mandatory for the signal / slot machinery !
        self._model = model
        self._mapping = mapping

    def __getitem__(self, key):
        return getitem(self._model, self._mapping[key])

    def __setitem__(self, key, val):
        oldval = self[key]
        self.model_changerequest.emit(self._mapping[key], oldval, val)
    


class Image_presenter(Presenter):
    # key, oldval, newval signal
    model_changerequest = pyqtSignal(object, object, object)

    def __init__(self, model, mapping):
        """
        Submodel
        Wraps and inspect an parameter and a key of the main model / submodel

        mapping should map 
            {"folder": folder_key
             "image": image_key
             "xy_ratio": xy_ratio_key
             "x": x_key
             "y": y_key
             "dx": dx_key}
        """
        super().__init__(model, mapping)  # signal / slot machinery !
        self.model_changerequest.connect(self._model.model_changerequest_slot)

#    @property
#    def image(self):
#        return self["image"]
#    @property
#    def xy_ratio(self):
#        return self["xy_ratio"]
#    @property
#    def x(self):
#        return self["x"]
#    @property
#    def y(self):
#        return self["y"]
#    @property
#    def dx(self):
#        return self["dx"]
    
    

    @pyqtSlot(object, object)
    def func_user_modified_slot(self, key_list, val_list):
        """
        An event consist in a list of keys fired simultaneously
        """
        if key_list == ("px", "py"):
            px = val_list[0]
            py = val_list[1]
            print("mouse tracked in view presenter")
            print(px, py)

        elif key_list == ("px", "py", "pxx", "pyy"):
            px = val_list[0]
            py = val_list[1]
            pxx = val_list[2]
            pyy = val_list[3]
            print("zoom proposed in view presenter")
            print(px, py, pxx, pyy)

    @pyqtSlot()
    def zoom_reset_slot(self):
        pass
    
    @pyqtSlot(object)
    def image_modifed_slot(self, im):
        pass




def default_val(utype):
    """ Returns a generic default value for a given type"""
    if utype is int:
        return 0
    elif utype is float:
        return 0.
    elif utype is str:
        return ""
    elif utype is type(None):
        return None
    elif typing.get_origin(utype) is typing.Literal:
        return typing.get_args(utype)[0] # first element of the Literal
    elif typing.get_origin(utype) is typing.Union:
        return default_val(typing.get_args(utype)[0]) # first def of the Union
    else:
        raise NotImplementedError("No default for this subtype {}".format(
                                   utype))

def can_cast(val, cast_type):
    """ Returns True if val is a valid default value for a given type"""
    if typing.get_origin(cast_type) is typing.Literal:
        return (val in typing.get_args(cast_type))
    try:
        casted = cast_type(val)
        return (casted == val)
    except (ValueError, TypeError):
        return False

#def cast(val, cast_type):
#    if typing.get_origin(cast_type) is typing.Literal:
#        raise ValueError("Cannot cast a Litteral")
#    else:
#        return cast_type(val)

def matching_instance(val, cast_type):
    if typing.get_origin(cast_type) is typing.Literal:
        return False
    else:
        return isinstance(val, cast_type)

def best_match(val, types):
    # Best match : casting litterals
    is_litteral = tuple(typing.get_origin(t) is typing.Literal for t in types)
    casted = tuple(can_cast(val, t) for t in types)
    match = is_litteral and casted
    if any(match):
        index = match.index(max(match))
        return index, types[index]
    # then instance
    match = tuple(matching_instance(val, t) for t in types)
    if any(match):
        index = match.index(max(match))
        return index, types[index]
    # then can_cast
    match = casted
    if any(match):
        index = match.index(max(match))
        return index, types[index]
    raise ValueError("No match for val {} and types {}".format(val, types))

def type_name(naming_type):
    if typing.get_origin(naming_type) is typing.Literal:
        return("choice")
    else:
        return naming_type.__name__



class Func_submodel(Submodel):
    # key, oldval, newval signal
    model_notification = pyqtSignal(object, object, object)

    def __init__(self, model, submodel_keys, func):
        """
        Submodel
        Wraps and inspect a function parameter and code
        """
        super().__init__(model, submodel_keys)
        self._func = func
        self.build_dict()
        # Publish-subscribe model
        self.model_notification.connect(self._model.model_notified_slot)

    def build_dict(self):
        """
        Keys - those user settables denoted by (*)
        "n_params" number of parameter
        for each iparam
            (iparam, "has_def") has a default val
            (iparam, "is_def") default val is selected (*)
            (iparam, "val_def") default value or ""
            (iparam, "name") name of the parameter
            (i_param, "n_types") number of allowed types in typing.Union
            (i_param, "n_types") number of allowed types in typing.Litteral
            (i_param, "type_def") default type
            (i_param, "type_sel") current type selection if Union (*)
        for each iparam, i_union
            (iparam, i_union, "type") the type
            (iparam, i_union, 'val') the value (*)
        """
        fd = self._dict = dict()
        sign = inspect.signature(self._func)
        fd["n_params"] = len(sign.parameters.items())
        i_param = 0
        for name, param in sign.parameters.items():
            self.insert_param(i_param, name, param)
            i_param += 1

    def insert_param(self, i_param, name, param):
        """
        for each "iparam":
        "has_def", "val_def", "name", "n_union"
        """
        fd = self._dict
        fd[(i_param, "name")] = name
        default = param.default
        ptype = param.annotation
        uargs = typing.get_args(ptype)
        origin = typing.get_origin(ptype)

        fd[(i_param, "has_def")] = True
        if default is inspect.Parameter.empty:
            fd[(i_param, "has_def")] = False
            default = default_val(ptype)
#            raise ValueError("No default value provided for " + 
#                             "parameter {}".format(name))
        fd[(i_param, "is_def")] = fd[(i_param, "has_def")] # at initialization
        fd[(i_param, "val_def")] = default
        # fd[(i_param, "type_def")] = None # to be completed later

        # inspect the datatype - is it a union
        
        if origin is typing.Union:
            fd[(i_param, "n_types")] = len(uargs)
            fd[(i_param, "n_choices")] = 0
            type_index, type_match = best_match(fd[(i_param, "val_def")],
                                                   uargs)
            fd[(i_param, "type_sel")] = type_index
            fd[(i_param, "type_def")] = type_index #type_match

        elif origin is typing.Literal:
            fd[(i_param, "n_types")] = 0
            fd[(i_param, "n_choices")] = len(uargs)
            fd[(i_param, "type_sel")] = 0
            fd[(i_param, "type_def")] = 0
            fd[(i_param, "choices")] = uargs
            fd[(i_param, "val_def")] = uargs.index(default)

        elif origin is None:
            fd[(i_param, "n_types")] = 0
            fd[(i_param, "n_choices")] = 0
            fd[(i_param, "type_sel")] = 0
            fd[(i_param, "type_def")] = 0

        else:
            raise ValueError("Unsupported type origin: {}".format(origin))
            
#        fd[(i_param, "types")] = []
        if fd[(i_param, "n_types")] > 0:
            for i_union, utype in enumerate(uargs):
                self.insert_uarg(i_param, i_union, utype)
        else:
            self.insert_uarg(i_param, 0, ptype)
        


    def insert_uarg(self, i_param, i_union, utype):
        """
        for each "iparam", "i_union":
        """
        fd = self._dict
        fd[(i_param, i_union, "type")] = utype
        fd[(i_param, i_union, "type")] = utype #.__name__

        if ((fd[(i_param, "n_types")] > 0)
                and (i_union != fd[(i_param, "type_def")])):
            # Default val NOT applicable to this *union* type
            # we use a generic default based on type
            fd[(i_param, i_union, "val")] = default_val(utype)
        else:
            # We can use the user default val
            fd[(i_param, i_union, "val")] = fd[(i_param, "val_def")]

        


    @pyqtSlot(object, object)
    def func_user_modified_slot(self, key, val):
        print("in submodel, widget_modified", key, val)
        # self[key] = val
        
        keylen = len(key)
        if keylen == 1:
            raise ValueError("Untracked key{}".format(key))

        elif keylen == 2:
            iparam, identifier = key
            if identifier == "type_sel":
                self[key] = val
                if self[(iparam, "has_def")]:
                    self[(iparam, "is_def")] = False
            elif identifier == "is_def":
                # If type is Union, sets selector is 'default' as ticked
                self[key] = val
                if val:
                    type_def = self[(iparam, 'type_def')]
                    val_def = self[(iparam, 'val_def')]
                    self[(iparam, type_def, "val")] = val_def
                    if self[(iparam, "n_types")] > 0:
                        self[(iparam, "type_sel")] = type_def
            else:
                raise ValueError("Untracked key {}".format(key))

        elif keylen == 3:
            iparam, i_union, identifier = key
            if identifier == "val":
                self[key] = val
                # untick default if val doesn't match
                if (self[(iparam, "is_def")]
                    and (val != self[(iparam, 'val_def')])):
                    self[(iparam, "is_def")] = False
            else:
                raise ValueError("Untracked key {}".format(key))

        else:
            raise ValueError("Untracked key {}".format(key))
                
            
            
    
        # Do we need more locally
        
        # Now we notify the model upstream
        
#    @pyqtSlot(tuple, str)
#    def on_func_arg_changed(self, key, val):
#        print("called slot", key, val)

