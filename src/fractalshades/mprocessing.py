# -*- coding: utf-8 -*-
#import pickle
#import cloudpickle
#pickle.Pickler = cloudpickle.Pickler

import multiprocessing
import os
import sys
import functools

import concurrent.futures


import fractalshades.utils as fsutils
import fractalshades.settings as fssettings
import uuid
import contextlib

from inspect import signature, getsource
#import dill
#import marshal
#from types import FunctionType


# https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f
@contextlib.contextmanager
def globalized(func):
    namespace = sys.modules[func.__module__]
    name, qualname = func.__name__, func.__qualname__
    func.__name__ = func.__qualname__ = f'_{name}_{uuid.uuid4().hex}'
    setattr(namespace, func.__name__, func)
    try:
        yield
    finally:
        delattr(namespace, func.__name__)
        func.__name__, func.__qualname__ = name, qualname

# https://newbedev.com/python-multiprocessing-picklingerror-can-t-pickle-type-function
#def run_dill_encoded(payload):
#    import numpy as np
#    fun, args = dill.loads(payload)
#    return fun(*args)


def redirect_output(redirect_path):
    """
    Save *job* as a global for the child-process ; redirects stdout and stderr.
    -> 'On Unix a child process can make use of a shared resource created in a
    parent process using a global resource. However, it is better to pass the
    object as an argument to the constructor for the child process.'
    """
    if redirect_path is not None:
        fsutils.mkdir_p(redirect_path)
        out_file = str(os.getpid())
        sys.stdout = open(os.path.join(redirect_path, out_file + ".out"), "a")
        sys.stderr = open(os.path.join(redirect_path, out_file + ".err"), "a")

def spawn_process_job(instance, wrapped_name, args, kwargs, iter_kwarg, key):
    """ We change the value of the iterated kwarg, than call the wrapped
    function (/!\ do not call the decorated function of you will try to launch
    an infinite loop of subprocesses"""
    wrapped_method = getattr(instance, wrapped_name)
    kwargs[iter_kwarg] = key
    return wrapped_method(*args, **kwargs) # Bound method



#def _applicable(*args, **kwargs):
#    name = kwargs['__pw_name']
#    code = marshal.loads(kwargs['__pw_code'])
#    gbls = globals() #gbls = marshal.loads(kwargs['__pw_gbls'])
#    defs = marshal.loads(kwargs['__pw_defs'])
#    clsr = marshal.loads(kwargs['__pw_clsr'])
#    fdct = marshal.loads(kwargs['__pw_fdct'])
#    func = FunctionType(code, gbls, name, defs, clsr)
#    func.fdct = fdct
#    del kwargs['__pw_name']
#    del kwargs['__pw_code']
#    #del kwargs['__pw_gbls']
#    del kwargs['__pw_defs']
#    del kwargs['__pw_clsr']
#    del kwargs['__pw_fdct']
#    return func(*args, **kwargs)
#
#def make_applicable(f, *args, **kwargs):
#    if not isinstance(f, FunctionType): raise ValueError('argument must be a function')
#    kwargs['__pw_name'] = f.__name__
#    kwargs['__pw_code'] = marshal.dumps(f.__code__)
#    #kwargs['__pw_gbls'] = marshal.dumps(f.func_globals)
#    kwargs['__pw_defs'] = marshal.dumps(f.__defaults__)
#    kwargs['__pw_clsr'] = marshal.dumps(f.__closure__)
#    kwargs['__pw_fdct'] = marshal.dumps(f.__dict__)
#    return _applicable, args, kwargs



#def job_proxy(key):
#    """ returns result of global job variable from the child-process """
#    global process_job
#    return process_job(key)

class Multiprocess_filler:
    def __init__(self, iterable_attr, res_attr=None, redirect_path_attr=None,
                 iter_kwargs="key", veto_multiprocess=False,
                 finalize_attr=None):
        """
        Decorator class for an instance-method *method*

        *iterable_attr* : string, getattr(instance, iterable_attr) is a
            Generator function (i.e. `yields` the successive values).
#        *res_attr* : string or None, if not None (instance, res_attr) is a
#            dict-like.
        *redirect_path_attr* : string or None. If veto_multiprocess is False,
            getattr(instance, redirect_path_attr) is the directory where
            processes redirects sys.stdout and sys.stderr.
        veto_multiprocess : if True, defaults to normal iteration without
            multiprocessing

        Usage :
        @Multiprocess_filler(iterable_attr, res_attr, redirect_path_attr,
                             iter_kwarg)
        def method(self, *args, iter_kwarg=None, **otherkwargs):
            (... CPU-intensive calculations ...)
            return val

        - iter_kwargs will be filled with successive values yielded
            and successive outputs will calculated by child-processes,
        - these outputs will be pushed in place to res in parent process:
            res[key] = val
        - the child-processes stdout and stderr are redirected to os.getpid()
            files in subdir *redirect_path* (extensions .out, in) - if provided
        """
        self.iterable = iterable_attr
        self.res = res_attr
        self.redirect_path_attr = redirect_path_attr
        self.iter_kwarg = iter_kwargs
        self.veto_multiprocess = veto_multiprocess
        self.finalize_attr = finalize_attr

    def __call__(self, method):

        @functools.wraps(method)
        def wrapper(instance, *args, **kwargs):
            
            
            if (fssettings.enable_multiprocessing and 
                    not(self.veto_multiprocess)):
                # https://docs.python.org/3/library/os.html#os.cpu_count
                if os.name == "posix":
                    cpu_count = len(os.sched_getaffinity(0))
                else:
                    cpu_count = os.cpu_count()

                redirect_path = None
                if self.redirect_path_attr is not None:
                    redirect_path = getattr(instance, self.redirect_path_attr)

                # Allow to test "spawn" on a UNIX machine (debuggin purpose)
                context = fssettings.multiprocessing_start_method
                instance.multiprocessing_start_method = context
                
                print("call multiproc with context: ", context)
            
                if context == "fork":
                    # Default for *nix
                    self.call_mp_fork(cpu_count, redirect_path,
                                      instance, method, args, kwargs)
                elif context == "spawn":
                    # Windows, mac
                    self.call_mp_spawn(cpu_count, redirect_path,
                                       instance, method, args, kwargs)
                else:
                    raise RuntimeError(context)
            else:
                self.call_std(instance, method, *args, **kwargs)

        return wrapper
                
                
    def call_mp_fork(self, cpu_count, redirect_path,
                     instance, method, args, kwargs):
        
#        def process_job(key):
#            kwargs[self.iter_kwarg] = key
#            method(instance, *args, **kwargs)

        def process_job(key):
            kwargs[self.iter_kwarg] = key
            method(instance, *args, **kwargs)

        with  globalized(process_job
                ), multiprocessing.get_context("fork").Pool(
                        initializer=redirect_output,
                        initargs=(redirect_path,),
                        processes=cpu_count
                ) as pool:
            for key in getattr(instance, self.iterable)():
                pool.apply_async(process_job, (key,))
            pool.close()
            pool.join()

    def call_mp_spawn(self, cpu_count, redirect_path,
                      instance, method, args, kwargs):


        # Do not call the decorated function of you will try to launch
        # an infinite loop of subprocesses
#        wrapped_name = "__wrapped__" + method.__name__
#        setattr(instance, wrapped_name, method)
#        print("__wrapped__:", wrapped_name, getattr(instance, wrapped_name))
        


        with multiprocessing.get_context("spawn").Pool(
                        initializer=redirect_output,
                        initargs=(redirect_path,),
                        processes=cpu_count
                ) as pool:
            
            wrapped_name = "_wrapped_" + method.__name__
#            setattr(instance, wrapped_name, method)
            finalize = None
            if self.finalize_attr is not None:
                finalize = getattr(instance, self.finalize_attr)

            for key in getattr(instance, self.iterable)():
#                print("spawn_process_job", spawn_process_job)
                spawn_process_args = (
                    instance,
                    wrapped_name,
                    args,
                    kwargs,
                    self.iter_kwarg,
                    key
                )
                # payload = dill.dumps((spawn_process_job, spawn_process_args))
                
                if finalize is None:
                    # pool.apply_async(run_dill_encoded, (payload,))
                    pool.apply_async(spawn_process_job, spawn_process_args)
                else:
                    # finalize(pool.apply_async(run_dill_encoded,
                    #                          (payload,)))
                    finalize(pool.apply_async(spawn_process_job,
                                              spawn_process_args))
            pool.close()
            pool.join()


    def call_std(self, instance, method, *args, **kwargs):
        
        for key in getattr(instance, self.iterable)():
            kwargs[self.iter_kwarg] = key
            # Still some multithreading
#            with concurrent.futures.ThreadPoolExecutor(
#                    max_workers=1) as threadpool:
#                full_args = (instance,) + args
#                threadpool.submit(method, full_args, kwargs).result()
            method(instance, *args, **kwargs)
  
