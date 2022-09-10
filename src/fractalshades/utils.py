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
    - stores a (deep) copy of the kwargs in a dictionary self.<indentifier>__kwargs
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
      "_@" + indentifier
    - these can be retrieved by calling 
      store_kwargs_and_func_name.methods(Instance_class)

    After function call
    - calls an instance-method 'post-hook' with parameter:
        (method.__name__, kwargs_dic, return_dic)
#    - stores the individual kwargs as instance attributes
#    - stores a (deep) copy of the kwargs as an instance-attribute
#      dictionary: instance.<indentifier>_kwargs
#    - stores the name of the last decorated method called as instance-attribute
#      string: instance.<indentifier>_callable

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
            getattr(instance, self.indentifier + "_hook")(
                method.__name__,
                kwargs_dic,
                return_dic
            )

#            kwargs_attr = indentifier + "_kwargs"
#            callable_attr = indentifier + "_callable"
#
#            ret = _store_kwargs(kwargs_attr)(method)(instance, *args, **kwargs)
#            setattr(instance, callable_attr, method.__name__)
#            return ret
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
zoom_options.__doc__ = """
Decorates the method used to define the zooming (only one per 
`fractalshades.Fractal` class).
The last kwargs passed to the zoom method can be retrived as
fractal.zoom_kwargs
"""

calc_options = _store_func_name__add_hook("calc")
calc_options.__doc__ = """
Decorates the calculation methods (can be several per 
`fractalshades.Fractal` class)
The last kwargs passed to any calculation method can be retrived as
fractal.calc_kwargs. The name of the method called can be retrieved as
fractal.calc_callable
"""

interactive_options = _store_func_name("interactive_options")
interactive_options.__doc__ = """
Decorates the methods that can be called interactively from the GUI 
(can be several per `fractalshades.Fractal` class)
The list of such methods can be retrived as:
    fs.utils.interactive_options.methods(f.__class__)
"""
