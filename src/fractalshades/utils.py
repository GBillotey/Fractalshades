# -*- coding: utf-8 -*-
import os
import sys
import errno
import functools
import copy
import inspect
import collections.abc
import numbers

import mpmath

def mkdir_p(path):
    """ Creates directory ; if exists does nothing """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc


def dic_flatten(nested_dic, key_prefix="", sep="@"):
    """
    Flatten a dic of nested dics. %sep% will be inserted between nested keys
    so that dic[key1]...[keyn] becomes flatten_dic[key1%sep%...%sep%keyn]
    """
    res = dict()
    if isinstance(nested_dic, dict):
        for key, val in nested_dic.items():
            pre_key = "" if key_prefix == "" else key_prefix + sep
            res.update(dic_flatten(val, pre_key + key, sep))
    else:
        # Simple key-value
        res[key_prefix] = nested_dic
    return res

class Protected_mapping(collections.abc.MutableMapping):
    """
    A read-only dictionnary.
    """
    def __init__(self, dic):
        self._dict = dic
    
    def __getitem__(self, key):
        return copy.deepcopy(self._dict[key])

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        raise RuntimeError("Attempt to modify a Protected_mapping")

    def __delitem__(self, key):
        raise RuntimeError("Attempt to modify a Protected_mapping")


class Rawcode:
    def __init__(self, raw_code_str):
        """ Utility class for a user-specified code fragment """
        self._str = raw_code_str

    def script_repr(self, indent):
        return " " * (4 * indent) + self._str


class Code_writer:
    """
    A set of static methods allowing to write Python source code
    """

    @staticmethod
    def var_tocode(var, indent=0):
        """
        Returns a string of python source code to serialize the variable var.

        Parameters
        ----------
        var: object
            The parameter to reconstruct. Supported types:
                None
                Numbers (int, float, bool)
                Dict
                list
                Class
        indent: int
            The current indentation level
        """
        shift = " " * (4 * indent)

        if var is None:
            return "None"
        if isinstance(var, numbers.Number):
            return repr(var)
        if isinstance(var, str):
            return f'"{var}"'
        if isinstance(var, mpmath.mpf):
            return repr(var)
        if isinstance(var, dict):
            shift_inc = shift + " " * 4
            ret = (shift_inc).join([
                f"{Code_writer.var_tocode(k, indent+1)}: "
                f"{Code_writer.var_tocode(v, indent+1)},\n"
                for (k, v) in var.items()
            ])
            ret = f"{{\n{shift_inc}{ret}{shift}}}" # {{ for \{ in f-string
            return ret
        if isinstance(var, list):
            shift_inc = shift + " " * 4
            ret = (shift_inc).join([
                f"{Code_writer.var_tocode(v, indent+1)},\n"
                for v in var
            ])
            ret = f"[\n{shift_inc}{ret}{shift}]" # {{ for \{ in f-string
            return ret
        if isinstance(var, tuple):
            shift_inc = shift + " " * 4
            ret = (shift_inc).join([
                f"{Code_writer.var_tocode(v, indent+1)},\n"
                for v in var
            ])
            ret = f"(\n{shift_inc}{ret}{shift})" # {{ for \{ in f-string
            return ret

        if inspect.isclass(var):
            ret = f"{Code_writer.fullname(var)}"
            return ret

        if hasattr(var, "script_repr"):
            # Complex object: check first if has a dedicated script_repr
            # implementation
            return var.script_repr(indent)

        if hasattr(var, "init_kwargs"):
            # Complex object: defauts implementation serialize by calling the 
            # __init__ method
            return Code_writer.instance_tocode(var, indent)

        else:
            raise NotImplementedError(var)


    @staticmethod
    def fullname(class_):
        """ returns the fullname of a class """
        module = class_.__module__
        if module == 'builtins':
            return class_.__qualname__ # avoid outputs like 'builtins.str'
        return module + '.' + class_.__qualname__


    @staticmethod
    def instance_tocode(obj, indent=0):
        """ Unserialize by calling init method
        """
        shift = " " * (4 * indent)
        fullname = Code_writer.fullname(obj.__class__)
        kwargs = obj.init_kwargs
        kwargs_code = Code_writer.func_args(kwargs, indent + 1)
        str_call_init = f"{shift}{fullname}(\n{kwargs_code}{shift})"
        return str_call_init


    @staticmethod
    def write_assignment(varname, value, indent=0):
        """
        %varname = %value
        """
        shift = " " * (4 * indent)

        try:
            var_str = Code_writer.var_tocode(value, indent)
        except NotImplementedError: # rethrow with hopefully better descr
            raise NotImplementedError(varname, value)
        str_assignment = f"{shift}{varname} = {var_str}\n"
        return str_assignment


    @staticmethod
    def func_args(kwargs, indent=0):
        """
           key1=value1,
           key2=value2,
        """
        shift = " " * (4 * indent)
        try:
            ret = shift.join([
                f"{k}={Code_writer.var_tocode(v)},\n"
                for (k, v) in kwargs.items()
            ])
        except NotImplementedError:
            etype, evalue, etraceback = sys.exc_info()
            raise  NotImplementedError(f"{evalue}  raised from {kwargs}")

        return shift + ret



def exec_no_output(func, *args, **kwargs):
    """ Building the doc figures without stdout, stderr"""
    import sys
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _store_kwargs(indentifier):
    """ Decorator for an instance method, 
    - stores the individual kwargs as instance attributes 
    - stores a (deep) copy of the kwargs in a dictionary self.<indentifier>_kwargs
    Note that:
        - Default values are taken into account
        - this decorator will  raise a ValueError if given positional
          arguments.
    """
    def wraps(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # bind arguments, ommiting self
            if len(args) > 0:
                raise TypeError(
                    ("{0:} should only accept keyword-arguments ; "
                    + "given positionnal: {1:}").format(method.__name__, args)
                )
            ba = inspect.signature(method).bind_partial(**kwargs)
            ba.apply_defaults()
            kwargs_dic = dict(ba.arguments)
            setattr(self, indentifier + "_kwargs", kwargs_dic)
            for key, val in kwargs_dic.items():
                # If attributes are modified, we do not want to track
                # So, passing a copy.
                setattr(self, key, copy.deepcopy(val))
            return method(self, *args, **kwargs)
        return wrapper
    return wraps


class _store_func_name__add_hook:
    def __new__(cls, indentifier):
        """ 
    Decorator for an instance method.

    Parameters
    ----------
    indentifier, str

    At initialisation:
    - keep track of the set of decorated method by tagging them with attribute
      "_@" + indentifier, so these can be retrieved by calling 
      _store_func_name__add_hook.methods(Instance_CLASS)

    At function call the wrapper:

    - calls the wrapped function
    - then tries to forwards the calling kwargs to instance-method
      'post-hook' (if it exists) with signature: (method.__name__,
      kwargs_dic, return_dic) where:

          - kwargs_dic is the calling kwargs
          - return_dic is the returned result

    Note that:
        - Default values are taken into account if argument not provided
        - this decorator will raise a ValueError if method is called with any
         positional arguments.
    """
        cls.indentifier = indentifier
        return object.__new__(cls)

    def __call__(self, method):
        indentifier = self.indentifier

        @functools.wraps(method)
        def wrapper(instance, *args, **kwargs):
            # bind arguments, ommiting self
            if len(args) > 0:
                raise TypeError(("{0:} should only accept keyword-arguments ; "
                    + "given positionnal: {1:}").format(
                            method.__name__, args))
            ba = inspect.signature(method).bind_partial(**kwargs)
            ba.apply_defaults()
            kwargs_dic = dict(ba.arguments)
            return_dic = method(instance, *args, **kwargs)

            # post call hook
            try:
                getattr(instance, self.indentifier + "_hook")(
                    method.__name__,
                    kwargs_dic,
                    return_dic
                )
            except AttributeError:
                pass

        setattr(wrapper, "_@" + indentifier, True)
        return wrapper

    @classmethod
    def methods(cls, subject):
        indentifier = cls.indentifier
        def m_iter():
            for name, method in vars(subject).items():
                if callable(method) and hasattr(method, "_@" + indentifier):
                    yield name, method
        return {name: method for name, method in m_iter()}


class _store_func_name:
    def __new__(cls, dic_name):
        """ 
    Decorator for an instance method.

    Parameters
    ----------
    dicname, str

    At initialisation:
    - keep track of the set of decorated method by tagging them with attribute
      "_@" + dicname
    - these can be retrieved by calling 
      _store_func_name.methods(Instance_class)

    At function call
    - stores the individual kwargs as instance attributes
    - stores a (deep) copy of the kwargs as an instance-attribute
      dictionary: instance.dic_name
    - stores the name of the last decorated methos called as instance-attribute
      string (dic_name + "_lastcall")

    Note that:
        - Default values are taken into account if argument not provided
        - this decorator will raise a ValueError if method is called with any
         positional arguments.
    """
        cls.dic_name = dic_name
        return object.__new__(cls)

    def __call__(self, method):
        dic_name = self.dic_name
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        setattr(wrapper, "_@" + dic_name, True)
        return wrapper

    @classmethod
    def methods(cls, subject):
        dic_name = cls.dic_name
        def m_iter():
            for name, method in vars(subject).items():
                if callable(method) and hasattr(method, "_@" + dic_name):
                    yield name, method
        return {name: method for name, method in m_iter()}


zoom_options = _store_kwargs("zoom")
zoom_options.__doc__ = """ Decorates the method used to define the zooming
(only one per `fractalshades.Fractal` class).
The last kwargs passed to the zoom method can be retreived as
:code:`fractal.zoom_kwargs`
"""

calc_options = _store_func_name__add_hook("calc")
calc_options.__doc__ = """ Decorates the calculation methods
(there can be several such methods per `fractalshades.Fractal` class)
The list of such calculation methods can be retrieved as
:code:`fs.utils.calc_options.methods(f.__class__)`
After each call, the calling kwargs and results are forwarded to the Fractal
instance method `calc_hook` for further processing
"""

interactive_options = _store_func_name("interactive_options")
interactive_options.__doc__ = """ Decorates the methods that can be called
interactively from the GUI 
(There can be several such methods for a `fractalshades.Fractal` class)
The list of these methods can be retreived with:
:code:`fs.utils.interactive_options.methods(f.__class__)`
"""
