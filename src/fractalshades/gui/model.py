# -*- coding: utf-8 -*-
import inspect
import typing
import functools
import os
import pickle

#import dataclasses
import mpmath
from operator import getitem, setitem

import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal, pyqtSlot

import fractalshades as fs

"""
Implementation of a GUI following a Model-View-Presenter interface
architectural pattern.

- The model data is stored in a nested dict wrapped in Model or Submodel
instances The class in charge of the model is always the more deeply nested
Submodel

- the GUI-compenents "Views" are the hosted in guimodel.py
- Presenter make the link between the model and the view. However in the case
  where a view stricly implements a (sub)model, then no presenter is needed, the
  view direclty send events to its submodel.

  Event Publish / subscribe chain is as follows :
  
  model_changerequest_event :  Model -> Submodel (request to update)
  model_event : Model -> View (notify has been updated)
  
  model_notification : Submodel -> Model (Notify has been updated)
  
  model_changerequest : Presenter -> Model (request to update)
  
  xxx_user_modified : - Viewer -> Submodel (User action)
                      - Viewer -> Presenter (User action)

Model-level 'settings' provide a convenient direct access to a subset of keys
at model level. They are instanciated by a Model *declare_setting* method.

Note on the datatype:
    float -> user input casted as float
    int -> user input casted as int
    mpmath.mpf -> user input kept as str to avoid losing precision
"""


class Model(QtCore.QObject):
    """
    Model composed of nested dict. An item of this dict is identified by a 
    list of nested keys.
    
    Beside the this dict, a Model keeps also internally references to :
       - all Submodel class intances
       - a list of keys that are directly exposed with an alias at model level
         a.k.a "settings"
    """
    # Notify the widget (view)
    model_event = pyqtSignal(object, object) # key, newval
    # Notify the model
    model_changerequest_event = pyqtSignal(object, object) # key, newval

    def __init__(self, source=None):
        super().__init__()
        self._model = source
        if self._model is None:
            self._model = dict()
        # Register all Submodel and Presenter classes
        self._register = dict()
        # Register for model-level settings (dps, ...)
        self._settings = dict()
            

    def to_register(self, register_key, data):
        """
        Keep references to Submodel or Presenter objects, allowing to interpret
        self._model data.
        """
        self._register[register_key] = data
    
    def from_register(self, register_key):
        """
        Retrives Submodel or Presenter objects
        """
        return self._register[register_key]

    def __setitem__(self, keys, val):
        # Sets an item value. keys iterable of nested keys
#        setitem(functools.reduce(getitem, keys[:-1], self._model),
#                keys[-1], val)
#        print("setitem", keys, val)
#        print("self[keys[:-1]]", self[keys[:-1]])
        setitem(self[keys[:-1]], keys[-1], val)
        # self.model_item_updated.emit(keys)
        # self.model_item_updated.emit(keys)

    def __getitem__(self, keys):
        # Gets an item value. keys is an iterable of nested keys
        try:
           return functools.reduce(getitem, keys, self._model)
        except KeyError:
            # let a chance to handle at submodel level
            return self.from_register(keys[:-1])[keys[-1]]

    
    def declare_setting(self, setting_name, setting_keys):
        """ Declares a setting, eg: dps """
        self._settings[setting_name] = setting_keys

    def setting(self, setting_name):
        """ Declares a global setting, eg: dps """
        return self[self._settings[setting_name]]

    @pyqtSlot(object)
    def setting_touched(self, setting_name):
        """ Reload setting with same value (side effect)"""
        # print("SETTING SIDE EFFECT", setting_name)
#        print("settings", self._settings)
        setting_val = self.setting(setting_name)
#        print("val, key", self._settings[setting_name], setting_val)
        self.model_changerequest_event.emit(
                self._settings[setting_name], setting_val)

    @pyqtSlot(object, object)
    def setting_modified(self, setting_name, setting_val):
        # print("SETTING MODIFIED", setting_name)
        self.model_changerequest_event.emit(
                self._settings[setting_name], setting_val)

    @pyqtSlot(object, object, object)
    def model_notified_slot(self, keys, oldval, val):
        """ A change has been done in a model / submodel,
        need to notify the widgets (viewers) """
        # print("MODEL model_notified_slot") # , keys, oldval, val)
        # Here we could implement UNDO / REDO stack
        self.model_event.emit(keys, val)


    @pyqtSlot(object, object)
    def model_changerequest_slot(self, keys, val):
        """ A change has been requested by a widget (viewer),
        connected to a presenter i.e not holding the data """
        if len(keys) == 0:
            # This change can be handled at model level
            self._model[keys] = val
            self.model_event.emit(keys, val)
        else:
            # This change need to be be handled at submodel level
            self.model_changerequest_event.emit(keys, val)


class Submodel(QtCore.QObject):
    """
    A submodel holds the data for a widget, stored in self.dict
    Pubsub model
    """
    def __init__(self, model, submodel_keys):
        super().__init__()  # Mandatory for the signal / slot machinery !
        self._model = model
        self._keys = submodel_keys
        self._model[submodel_keys] = self._dict = dict()
        self._model.to_register(submodel_keys, self)
        # connect to change request from model
        self._model.model_changerequest_event.connect(self.model_event_slot)

    def __getitem__(self, key):
        return getitem(self._dict, key)

    def __setitem__(self, key, val):
        if key not in self._dict.keys():
            raise KeyError(key)
        oldval = self[key]
        setitem(self._dict, key, val)
        self.model_notification.emit(self._keys + tuple([key]), oldval, val)  

separator = typing.TypeVar('gui_separator')

class Func_submodel(Submodel):
    # key, oldval, newval signal
    model_notification = pyqtSignal(object, object, object)

    def __init__(self, model, submodel_keys, func, dps_var=None):
        """
        Submodel
        Wraps and inspect a function parameter and code
        """
        super().__init__(model, submodel_keys)
        self._func = func
        self.build_dict()
        # Publish-subscribe model
        self.model_notification.connect(self._model.model_notified_slot)
        # Bindings for dps
        if dps_var is not None:
            print("dps_var", dps_var)
            self.connect_dps(dps_var)
    
    def connect_dps(self, dps_var):
        """
        Register this var as the dps event listener at model level
        """
        sign = inspect.signature(self._func)
        for i_param, (name, param) in enumerate(sign.parameters.items()):
#            print(name, param)
            if name == dps_var:
                if typing.get_origin(param.annotation) is not None:
                    raise ValueError("Unexpected type for math.dps: {}".format(
                            typing.get_origin(param.annotation)))
                key = (i_param, 0, "val")
#                print("setting dps listener",key )
                self._model.declare_setting("dps", self._keys + tuple([key]))
                return
        raise ValueError("Parameter not found", dps_var)

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
        fd = self._dict #= dict()
        sign = inspect.signature(self._func)
        fd["n_params"] = len(sign.parameters.items())
        for i_param, (name, param) in enumerate(sign.parameters.items()):
            self.insert_param(i_param, name, param)

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

        # inspect the datatype - is it a Union a Litteral or standard
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

    def getkwargs(self):
        """ Returns the current value of the kwargs """
        fd = self._dict
        n_params = fd["n_params"]
        kwargs = dict()
        for iparam in range(n_params):
            iparam_name = fd[(iparam, "name")]
            iunion = fd[(iparam, "type_sel")]
            ichoices = fd[(iparam, "n_choices")]
            # If typing.Literal we infer the val from the choice index
            if ichoices == 0:
                iparam_val = fd[(iparam, iunion, "val")]
            else:
                choices = fd[(iparam, "choices")]
                iparam_val = choices[fd[(iparam, iunion, "val")]]
                
            kwargs[iparam_name] = iparam_val
        return kwargs
    
    def setkwarg(self, kwarg, val):
        """ sets the current value of the kwarg specified by its name """
        fd = self._dict
        n_params = fd["n_params"]
        for iparam in range(n_params):
            if fd[(iparam, "name")] != kwarg:
                continue
            iunion = fd[(iparam, "type_sel")] # currently selected TODO  n_types / n_choices
            ichoices = fd[(iparam, "n_choices")]
            # If typing.Literal we infer the val from the choice index
            if ichoices == 0 and iunion == 0:
                param_key = (iparam, iunion, "val")
                self.func_user_modified_slot(param_key, val)
            else:
                raise NotImplementedError()
#            print("Setting param", param_key, kwarg, val)
            

    @property
    def param0(self):
        """ Return the value of the first parameter - which should be, the
        Fractal object
        """
        return next(iter(self.getkwargs().values()))

    def save_func_path(self):
        """ Return the file where the calling parameters are saved (pickled)
        """
        return os.path.join(self.param0.directory, "gui", "params.pickle")

    def save_func_dict(self):
        """ Save the calling parameters - except unpickable parameters
        - main Fractal object
        - gui-separators
        """
        print("*** saving kwargs")
        fd = self._dict.copy()

        unpickables = []
        for key, val in fd.items():
            if isinstance(key, tuple) and len(key) == 3:
                # We filter parameters : if any of the possible type is 
                # unpickable, we remove
                if key[2] == "type":
                    if inspect.isclass(val) and issubclass(val, fs.Fractal):
                        # This is a Fractal instance, usually only first param
                        unpickables += [key[0]]
                    if (
                        isinstance(val, typing.TypeVar)
                        and (val.__name__ == "gui_separator")
                    ):
                        # This is a gui separator, not something the user may
                        # modify
                        unpickables += [key[0]]

        for key in list(fd.keys()): # copy through list as we modify on the fly
            if isinstance(key, tuple):
                iparam = key[0]
                if iparam in unpickables:
                    fd.pop(key)

        save_path = self.save_func_path()
        fs.utils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as param_file:
            pickle.dump(fd, param_file, pickle.HIGHEST_PROTOCOL)

    def load_func_dict(self):
        """ Reload parameters stored from last call """
        print("*** load kwargs")
        with open(self.save_func_path(), 'rb') as param_file:
            fd = pickle.load(param_file)
        
        if fd["n_params"] != self._dict["n_params"]:
            raise RuntimeError("Incompatible saved parameters")

        for key in fd.keys():
            self._dict[key] = fd[key]


    def __getitem__(self, key):
        """ Adapted to also return the "current value" of a specific kwarg """
        try:
            return super().__getitem__(key)
            # getitem(self._dict, key)
        except KeyError:
            return self.getkwargs()[key]

    def __setitem__(self, key, val):
        """ Adapted to also set the "current value" of a specific kwarg """
        try:
            super().__setitem__(key, val)
        except KeyError:
            oldval = self.getkwargs()[key]
            self.setkwarg(key, val)
            # Emit a model modification for the presenter that might be
            # tracking
            self.model_notification.emit(self._keys + tuple([key]),
                                         oldval, val)

    def getsource(self):
        """ Returns the function source code """
        return inspect.getsource(self._func)

    def model_event_slot(self, keys, val):
        if keys[:-1] != self._keys:
            return
        self.func_user_modified_slot(keys[-1], val)

    @pyqtSlot(object, object)
    def func_user_modified_slot(self, key, val):
        #  print("in submodel, widget_modified", key, val)
        if isinstance(key, str):
            # Accessing directly a kwarg by its name
            # TODO: should be only nparams
            self[key] = val
            return
        elif not(isinstance(key, tuple)):
            raise ValueError("Untracked key{}".format(key))
        # From here we are dealing with a 'tuple' key ie directly pointing
        # at the data structure
        keylen = len(key)
        if keylen == 1:
            raise ValueError("Untracked key{}".format(key))

        elif keylen == 2:
            # Changing default tickbox or data type
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
            # changing parameter value
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
        


class Presenter(QtCore.QObject):
    """
    A Presenter holds a mapping key -> model_key to the model data.
    When a mapping item is modified, it emits a model_changerequest signal
    transmitted to the model.

    Example: for an interactive fractal image explorer the mapping should be
    of the form :
        {"folder": folder_key
         "image": image_key
         "xy_ratio": xy_ratio_key
         "x": x_key
         "y": y_key
         "dx": dx_key}
    """
    model_changerequest = pyqtSignal(object, object)

    def __init__(self, model, mapping, register_key):
        super().__init__()
        self._model = model
        self._mapping = mapping
        self._model.to_register(register_key, self)
        self.model_changerequest.connect(self._model.model_changerequest_slot)

    def __getitem__(self, key):
        return self._model[self._mapping[key]]

    def __setitem__(self, key, val):
        # oldval = self[key]
        self.model_changerequest.emit(self._mapping[key], val)



class Colormap_presenter(Presenter):
    # model_notification = pyqtSignal(object, object, object)
    
    model_changerequest = pyqtSignal(object, object)
    
    
    cmap_arr_attr = ["colors", "kinds", "grad_npts", "grad_funcs"]
    cmap_arr_roles = [Qt.ItemDataRole.BackgroundRole,
                      Qt.ItemDataRole.DisplayRole,
                      Qt.ItemDataRole.DisplayRole,
                      Qt.ItemDataRole.DisplayRole] # TODO

    cmap_attr =  cmap_arr_attr + ["extent"]
    extent_choices = ["mirror", "repeat", "clip"]
    
    
   # kwargs = ["colors", "kinds", "grad_npts", "grad_funcs"]

    def __init__(self, model, mapping, register_key):
        """
        Presenter for a `fractalshades.colors.Fractal_colormap` parameter
        Mapping expected : mapping = {"cmap": cmap_key}
        """
        super().__init__(model, mapping, register_key)

    @property
    def cmap(self):
        # To use as a parameter presenter, the only key should be the class
        # name (see `on_presenter` from `Func_widget`)
        return self._model[self._mapping["Colormap_presenter"]]

    @property
    def cmap_dic(self):
        cmap = self.cmap
        return {attr: getattr(cmap, attr) for attr in self.cmap_attr}

    @staticmethod
    def default_cmap_attr(attr):
        if attr == "colors":
            return [0.5, 0.5, 0.5]
        elif attr == "kinds":
            return "Lch"
        elif attr == "grad_npts":
            return 32
        elif attr == "grad_funcs":
            return "x"
        else:
            raise ValueError(attr)

#    def build_dict(self):
#        cd = self._dict
#        print("build dict", self._cmap, type(self._cmap))
#        for attr in self.cmap_attr:
#            cd[attr] = getattr(self._cmap, attr)

#    def model_event_slot(self, keys, val):
#        if keys[:-1] != self._keys:
#            return
#        self.cmap_user_modified_slot(keys[-1], val)

    @pyqtSlot(object, object)
    def cmap_user_modified_slot(self, key, val):
#        print("cmap model event", key, val)

        if key == "size":
            cmap = self.adjust_size(val)
        elif key == "extent":
            cmap = self.cmap
            cmap.extent = val
        elif key == "table":
            cmap = self.update_table(val)
        else:
            raise ValueError(key)

        self.model_changerequest.emit(self._mapping["Colormap_presenter"],
                                      cmap)
    
    def adjust_size(self, val):
        npts = self.cmap.n_probes
        cmap_dic = self.cmap_dic
        if val < npts:
            for attr in self.cmap_arr_attr:
                # Should work even for multidomentionnal ndarray (e.g., colors)
                cmap_dic[attr] = cmap_dic[attr][:val] #, ...]
        elif  val > npts:
            for attr in self.cmap_arr_attr:
                old_arr = cmap_dic[attr]
                default = self.default_cmap_attr(attr)
                if isinstance(old_arr, np.ndarray):
                    old_arr = cmap_dic[attr]
                    sh = (val,) + old_arr.shape[1:]
                    new_arr = np.empty(sh, old_arr.dtype)
                    new_arr[:npts, ...] = old_arr
                    new_arr[npts:, ...] = default
                else:
                    new_arr = old_arr + [default] * (val - npts)
                cmap_dic[attr] = new_arr

        return fs.colors.Fractal_colormap(**cmap_dic)
        self.build_dict()
#        print("cmap model_notification", self._keys)
    
    def update_table(self, item):
        """ item : modified QTableWidgetItem """
        row, col = item.row(), item.column()
        role = self.cmap_arr_roles[col]
        kwarg_key = self.cmap_arr_attr[col]

        cmap_dic = self.cmap_dic


        modified_kwarg = cmap_dic[kwarg_key]
        data = item.data(role)
        if col == 0:
            modified_kwarg[row] = [data.redF(), data.greenF(), data.blueF()]
        else:
            modified_kwarg[row] = data

        cmap_dic[kwarg_key] = modified_kwarg
        return fs.colors.Fractal_colormap(**cmap_dic)



def default_val(utype):
    """ Returns a generic default value for a given type"""
    if utype is int:
        return 0
    elif utype is float:
        return 0.
    elif utype is mpmath.mpf:
        return mpmath.mpf("0.0")
    elif utype is str:
        return ""
    elif utype is bool:
        return False
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



