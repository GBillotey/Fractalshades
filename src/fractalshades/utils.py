# -*- coding: utf-8 -*-
import os
import errno
import functools
import copy
import inspect


def mkdir_p(path):
    """ Creates directory ; if exists does nothing """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc

def _store_kwargs(dic_name):
    """ Decorator for an instance method, 
    - stores the individual kwargs as instance attributes 
    - stores a (deep) copy of the kwargs in a dictionary self.dic_name
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
                raise TypeError(("{0:} should only accept keyword-arguments ; "
                    + "given positionnal: {1:}").format(
                            method.__name__, args))
            ba = inspect.signature(method).bind_partial(**kwargs)
            ba.apply_defaults()
            kwargs_dic = dict(ba.arguments)
            setattr(self, dic_name, kwargs_dic)
            for key, val in kwargs_dic.items():
                # If attributes are modified, we do not want to track
                # So, passing a copy.
                setattr(self, key, copy.deepcopy(val))
            return method(self, *args, **kwargs)
        return wrapper
    return wraps


class _store_kwargs_and_func_name:
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
      store_kwargs_and_func_name.methods(Instance_class)

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
        def wrapper(instance, *args, **kwargs):
            ret = _store_kwargs(dic_name)(method)(instance, *args, **kwargs)
            setattr(instance, dic_name + "_callable", method.__name__)
            return ret
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


zoom_options = _store_kwargs("zoom_options")
zoom_options.__doc__ = """
Decorates the method used to define the zooming (only one per 
`fractalshades.Fractal` class)
"""

calc_options = _store_kwargs_and_func_name("calc_options")
calc_options.__doc__ = """
Decorates the calculation methods (can be several per 
`fractalshades.Fractal` class)
"""

interactive_options = _store_func_name("interactive_options")
interactive_options.__doc__ = """
Decorates the methods that can be called interactively from the GUI 
(can be several per `fractalshades.Fractal` class)
"""
