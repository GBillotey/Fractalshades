# -*- coding: utf-8 -*-
import os
import errno
import functools
import copy


def mkdir_p(path):
    """ Creates directory ; if exists does nothing """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc

def store_kwargs(dic_name):
    """ Decorator for an instance method, 
    - stores the individual kwargs as instance attributes 
    - stores a copy of the kwargs in a dictionary self.dic_name
    """
    def wraps(method):
        @functools.wraps(method)
        def wrapper(self, **kwargs):
            setattr(self, dic_name, copy.deepcopy(kwargs))
            for key, val in kwargs.items():
                setattr(self, key, val)
            return method(self, **kwargs)
        return wrapper
    return wraps


if __name__ == "__main__":
    def test_store_kwargs():
        class A:
            def __init__(self):
                pass
            
            @store_kwargs("my_options")
            def foo(self, *, x, y, z):
                pass

            @store_kwargs("my_options")
            def foo2(self, *, a, b, z):
                pass

        a = A()

        a.foo(x=[1], y=2, z=3)
        print(a.__dict__)
        a.x[0] = -100
        # check is a deepcopy
        print(a.__dict__)
        a.foo2(a=[1], b=2, z=300)
        print(a.__dict__)
        
    test_store_kwargs()
