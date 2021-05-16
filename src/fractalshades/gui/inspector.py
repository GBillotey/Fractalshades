# -*- coding: utf-8 -*-
import inspect
import typing
import dataclasses
from operator import getitem, setitem

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot

def default_val(utype):
    if utype is int:
        return 0
    elif utype is float:
        return 0.
    elif utype is str:
        return ""
    elif utype is type(None):
        return None
    elif typing.get_origin(utype) is typing.Literal:
        # Returns the first item if the list
        return typing.get_args(utype)[0]
    else:
        raise NotImplementedError("No default for this subtype {}".format(
                                   utype))

def can_cast(val, cast_type):
#    print("can_cast", val, cast_type)
    if typing.get_origin(cast_type) is typing.Literal:
        return (val in typing.get_args(cast_type))
    try:
        casted = cast_type(val)
        return (casted == val)
    except (ValueError, TypeError):
        return False

def cast(val, cast_type):
    if typing.get_origin(cast_type) is typing.Literal:
        if (val in typing.get_args(cast_type)):
            return val
        else:
            raise ValueError("Could not cast {} to {}".format(cast_type, val))
    else:
        return cast_type(val)

def matching_instance(val, cast_type):
    if typing.get_origin(cast_type) is typing.Literal:
        return False
    else:
        return isinstance(val, cast_type)

def best_match(val, types):
    # Best match : matching litterals
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
    

class Submodel(QtCore.QObject):
    """
    A submodel holds the data for a widget.
    Pubsub model
    """
    #model_item_modified = pyqtSignal(object, object)
    @property
    def keys(self): return self._keys
    def __setitem__(self, keys, val): raise NotImplementedError()
    def __getitem__(self, keys): raise NotImplementedError()



class Func_submodel(Submodel):
    # key, oldval, newval signal
    model_notification = pyqtSignal(object, object, object)

    def __init__(self, model, keys, func):
        """
        Submodel
        Wraps and inspect a function parameter and code
        """
        super().__init__()  # Mandatory for the signal / slot machinery !
        
        self._model = model
        self._keys = keys
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
            (iparam, "val_def") default value or ""
            (iparam, "name") name of the parameter
            (i_param, "n_types") number of allowed types in union
            (i_param, "type_def") default type
            (i_param, "type_sel") current type selection (*)
        for each iparam, i_union
            (iparam, i_union, "type") the type
            (iparam, i_union, "has_fields") the type is a datatype with 
                                            several fields
            (iparam, i_union, 'val') the value (*)
        for each iparam, i_union, i_field
            (iparam, i_union, i_field, "type") the type
            (iparam, i_union, i_field, "val") the value (*)
            
        dict[(i, "has_def")] param i has a default val
        dict[(i, "val_def")] default value or ""
        dict[(i, "name")] name of i-th param
        dict[(i, "n_union")] len(typing.get_args(ptype))
                             0 or number of union type accepted
        dict[(i, i_union, "datatype")] datatype for this union member
        dict[(i, i_union, n_fields)] 0 or number of subfields if is a 
                                     dataclass
        dict[(i_param, i_union, "has_fields") True if is a dataclass
                    (several fields) False either.
        dict[(i, i_union, "val")] Placeholder for val if not dataclass
        dict[(i, i_union, i_field, "valids")] List of acceptable values if 
                                     if dict[(i, "datatype")] a listed val
                                     typing.Literal["a", "b", "c"]
                                     casted to QVector<QString> stringVector
        dict[(i, i_union, i_field, "has_atom_def")] has atom def if dataclass
        dict[(i, i_union, i_field, "has_atom_def")] def atom val if dataclass
        dict[(i, i_union, i_field, "val")] Placeholder for field val if dataclass
        
        #
        """
        fd = self.func_dict = dict()
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
        fd = self.func_dict
        fd[(i_param, "name")] = name
        has_def = param.default is not inspect.Parameter.empty
        if not has_def:
            raise ValueError("No default value provided for " + 
                             "param {}".format(name))
        fd[(i_param, "is_def")] = has_def # at initialization
        fd[(i_param, "val_def")] = param.default
        # fd[(i_param, "type_def")] = None # to be completed later

        # inspect the datatype - is it a union
        ptype = param.annotation
        uargs = typing.get_args(ptype)
        origin = typing.get_origin(ptype)
        if not(origin in [typing.Union, None, typing.Literal]):
            raise ValueError("Unsupported type origin: {}".format(origin))
        
        if origin is typing.Union:
            fd[(i_param, "n_types")] = len(uargs)
        else:
            fd[(i_param, "n_types")] = 0
            
#        fd[(i_param, "types")] = []
        if fd[(i_param, "n_types")] != 0:
            print("uargs", uargs)
            for i_union, utype in enumerate(uargs):
                self.insert_uarg(i_param, i_union, utype)
        else:
            self.insert_uarg(i_param, 0, ptype)
        
#        param_types = [fd[(i_param, "type") for u_param in uargs ]
        param_types =  uargs if (origin is typing.Union) else [ptype]
        type_index, type_match = best_match(
                fd[(i_param, "val_def")], param_types)
        fd[(i_param, "type_def")] = type_index #type_match
        fd[(i_param, "type_sel")] = type_index
        print("##################### type sel", type_index, type_match)

    def insert_uarg(self, i_param, i_union, utype):
        """
        for each "iparam", "i_union":
        """
        fd = self.func_dict
#        print("ZZZ", fd[(i_param, "val_def")])
#        fd[(i_param, i_union, "val")] = fd[(i_param, "val_def")]
        fd[(i_param, i_union, "type")] = utype
        if dataclasses.is_dataclass(utype):
            # loop over the ifields
            fd[(i_param, i_union, "has_fields")] = True
            fd[(i_param, i_union, "type")] = utype #.__name__
            for ifield, field in enumerate(dataclasses.fields(utype)):
                self.insert_field(i_param, i_union, ifield, field)
        else:
            fd[(i_param, i_union, "has_fields")] = False
            fd[(i_param, i_union, "type")] = utype #.__name__
            # Default val applicable to this type ?
            possible_default = fd[(i_param, "val_def")]
            # 
            # if 
            if can_cast(possible_default, utype):
                default = cast(possible_default, utype)
            else:
                default = default_val(utype)
            fd[(i_param, i_union, "val")] = default #None #fd[(i_param, "val_def")]
#                self.insert_field(i_param, i_union, uarg)
        # add the current type to the list of pickable types
#            if i_union == 0:
#                md[(i_param, i_union, "valid")] = []
#            md[(i_param, i_union, "valid")]
            
    def insert_field(self, i_param, i_union, ifield, field):
        """
        for each "iparam", "i_union", "i_field":
        """
        fd = self.func_dict
#            md[(i_param, i_union, "val")] = ""
        print("field", field, type(field), field)
        ftype = field.type
        fname = field.name
        # No nested datatypes
        if dataclasses.is_dataclass(ftype):
            raise ValueError("Nested datatypes args unsupported")
#            md[(i_param, i_union, ifield, "type")] = field.__name__
        # Sets default
        if field.default is not(dataclasses.MISSING):
            default = field.default
        elif field.default_factory is not(dataclasses.MISSING):
            default = field.default_factory()
        else:
            default = dataclasses.MISSING
        has_field_default = default is not dataclasses.MISSING
        # fill in md dict
        fd[(i_param, i_union, ifield, "has_default")
            ] = has_field_default
        fd[(i_param, i_union, ifield, "name")] = fname # placeholder
        fd[(i_param, i_union, ifield, "ftype")] = ftype # placeholder
        if has_field_default:
            fd[(i_param, i_union, ifield, "default")] = default
        fd[(i_param, i_union, ifield, "val")] = "" # placeholder


#    def flatten(self):
#        """
#        build a flattened representation of the dict
#        self.arr + a mapping self.mapping so that
#        self.arr[self.mapping(key)] = self.dict[key]
#        then arr can be stored as the successive user_roles of our 
#        QAbstractItemModel
#        """
#        self.arr = tuple(self.func_dict.keys())
#        self.mapping = dict(zip(range(len(self.func_dict)),
#                                tuple(self.func_dict.keys())))
        
    def __getitem__(self, key):
        return getitem(self.func_dict, key)

    def __setitem__(self, key, val):
        oldval = self[key]
        setitem(self.func_dict, key, val)
        self.model_notification.emit(self._keys + tuple([key]),
                                     oldval, val)

    @pyqtSlot(object, object)
    def func_user_modified_slot(self, key, val):
        print("in submodel, widget_modified", key, val)
        # self[key] = val
        
        keylen = len(key)
        if keylen == 1:
            raise ValueError("Untracked key{}".format(key))
        elif keylen == 2:
            iparam, ident = key
            if ident == "type_sel":
                self[key] = val
                self[(iparam, "is_def")] = False
            elif ident == "is_def":
                # If type is Union, sets selector is 'default' is ticked
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
            iparam, i_union, ident = key
            if ident == "val":
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

