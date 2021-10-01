# -*- coding: utf-8 -*-
import multiprocessing
import os
import sys
import functools

import fractalshades as fs
import fractalshades.utils as fsutils
import fractalshades.settings as fssettings
import uuid
import contextlib

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

#def job_proxy(key):
#    """ returns result of global job variable from the child-process """
#    global process_job
#    return process_job(key)

class Multiprocess_filler():
    def __init__(self, iterable_attr, res_attr=None, redirect_path_attr=None,
                 iter_kwargs="key", veto_multiprocess=False):
        """
        Decorator class for an instance-method *method*

        *iterable_attr* : string, getattr(instance, iterable_attr) is a
            Generator function (i.e. `yields` the successive values).
        *res_attr* : string or None, if not None (instance, res_attr) is a
            dict-like.
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

    def __call__(self, method):
        @functools.wraps(method)
        def wrapper(instance, *args, **kwargs):
            if (fssettings.enable_multiprocessing and 
                    not(self.veto_multiprocess)
                    ):
                self.call_mp(instance, method, *args, **kwargs)
            else:
                self.call_std(instance, method, *args, **kwargs)

        return wrapper
                
                
    def call_mp(self, instance, method, *args, **kwargs):
        
        def process_job(key):
            kwargs[self.iter_kwarg] = key
            method(instance, *args, **kwargs)
        
        # https://docs.python.org/3/library/os.html#os.cpu_count
        cpu_count = len(os.sched_getaffinity(0))

        # Allow to test "spawn" on a UNIX machine (debuggin purpose)
        context = "fork"
        if (os.name != "posix") or  fssettings.multiprocessing_context_spawn:
            context = "spawn"

        redirect_path = None
        if self.redirect_path_attr is not None:
            redirect_path = getattr(instance, self.redirect_path_attr)
        

        with  globalized(process_job
                ), multiprocessing.get_context(context).Pool(
                        initializer=redirect_output,
                        initargs=(redirect_path,),
                        processes=cpu_count
                ) as pool:
            for key in getattr(instance, self.iterable)():
                pool.apply_async(process_job, (key,))
            pool.close()
            pool.join()
        
    def call_std(self, instance, method, *args, **kwargs):
        
        for key in getattr(instance, self.iterable)():
            kwargs[self.iter_kwarg] = key
            method(instance, *args, **kwargs)

                
                    


        
