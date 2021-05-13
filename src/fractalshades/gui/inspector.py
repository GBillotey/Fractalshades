# -*- coding: utf-8 -*-
import inspect
import typing
import dataclasses

from PyQt5.QtCore import pyqtSignal, pyqtSlot

def can_cast(val, cast_type):
    try:
        casted = cast_type(val)
        return (casted == val)
    except (ValueError, TypeError):
        return False

def best_match(val, types):
    match = tuple(isinstance(val, t) for t in types)
    if any(match):
        return types[match.index(max(match))]
    match = tuple(can_cast(val, t) for t in types)
    if any(match):
        return types[match.index(max(match))]
    raise ValueError("No match for val {} and types {}".format(val, types))

class Func_inspector:
    def __init__(self, func):
        """
        Provides inspection for the fractalshades code
        i_class, i_method: class and method.__name__ we want to inspect
        """
        self._func = func
        self.build_dict()
        self.flatten()

    def build_dict(self):
        """
        we want 
        dict["n_params"] number of parameter
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
        https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
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
        fd[(i_param, "is_def")] = True # at initialization
        fd[(i_param, "val_def")] = param.default
        fd[(i_param, "type_def")] = None # to be completed later

        # inspect the datatype - is it a union
        ptype = param.annotation
        uargs = typing.get_args(ptype)
        fd[(i_param, "n_types")] = len(uargs)
        fd[(i_param, "types")] = []
        if len(uargs) != 0:
            if typing.get_origin(ptype) is not typing.Union:
                raise ValueError("Unsupported type origin: {}".format(
                        typing.get_origin(ptype)))
            print("uargs", uargs)
            for i_union, uarg in enumerate(uargs):
                self.insert_uarg(i_param, i_union, uarg)
        else:
            self.insert_uarg(i_param, 0, ptype)
        fd[(i_param, "type_def")] = best_match(param.default,
                                               fd[(i_param, "types")])

    def insert_uarg(self, i_param, i_union, uarg):
        """
        for each "iparam", "i_union":
        """
        fd = self.func_dict
        fd[(i_param, i_union, "val")] = ""
        utype = uarg# .annotation
        fd[(i_param, "types")] += [utype]
        if dataclasses.is_dataclass(utype):
            # loop over the ifields
            fd[(i_param, i_union, "has_fields")] = True
            fd[(i_param, i_union, "type")] = utype #.__name__
            for ifield, field in enumerate(dataclasses.fields(utype)):
                self.insert_field(i_param, i_union, ifield, field)
        else:
            fd[(i_param, i_union, "has_fields")] = False
            fd[(i_param, i_union, "type")] = utype #.__name__
            fd[(i_param, i_union, "val")] = None #fd[(i_param, "val_def")]
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


    def flatten(self):
        """
        build a flattened representation of the dict
        self.arr + a mapping self.mapping so that
        self.arr[self.mapping(key)] = self.dict[key]
        then arr can be stored as the successive user_roles of our 
        QAbstractItemModel
        """
        self.arr = tuple(self.func_dict.keys())
        self.mapping = dict(zip(range(len(self.func_dict)),
                                tuple(self.func_dict.keys())))
        
    @pyqtSlot(tuple, str)
    def on_func_arg_changed(self, key, val):
        print("called slot", key, val)

