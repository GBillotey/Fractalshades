# -*- coding: utf-8 -*-
#import multiprocessing
import os
#import sys
import functools
#import uuid
#import contextlib
import concurrent


import fractalshades as fs
#import fractalshades.utils as fsutils



## https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f
#@contextlib.contextmanager
#def globalized(func):
#    namespace = sys.modules[func.__module__]
#    name, qualname = func.__name__, func.__qualname__
#    func.__name__ = func.__qualname__ = f'_{name}_{uuid.uuid4().hex}'
#    setattr(namespace, func.__name__, func)
#    try:
#        yield
#    finally:
#        delattr(namespace, func.__name__)
#        func.__name__, func.__qualname__ = name, qualname
#
#def redirect_output(redirect_path):
#    """
#    Redirects stdout and stderr.
#    """
#    if redirect_path is None:
#        devnull = open(os.devnull, 'w')
#        sys.stdout = devnull
#        sys.stderr = devnull
#    else:
#        fsutils.mkdir_p(redirect_path)
#        out_file = str(os.getpid())
#        sys.stdout = open(os.path.join(redirect_path, out_file + ".out"), "a")
#        sys.stderr = open(os.path.join(redirect_path, out_file + ".err"), "a")


class Multithreading_iterator():
    def __init__(self, iterable_attr, redirect_path_attr=None,
                 iter_kwargs="key", veto_parallel=False):
        """
        Decorator class for an instance-method *method*

        *iterable_attr* : string, getattr(instance, iterable_attr) is a
            Generator function (i.e. `yields` the successive values).
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
        self.redirect_path_attr = redirect_path_attr
        self.iter_kwargs = iter_kwargs
        self.veto_parallel = veto_parallel

    def __call__(self, method):

        @functools.wraps(method)
        def wrapper(instance, *args, **kwargs):
            parallel = (
                fs.settings.enable_multithreading and not self.veto_parallel
            )
            print("in wrapper, parallel:", parallel)
            if parallel:
                self.call_multi_thread(instance, method, *args, **kwargs)
            else:
                self.call_std(instance, method, *args, **kwargs)
#            
#            
#            if (fssettings.enable_multiprocessing and
#                    not(self.veto_multiprocess)):
#
#                # https://docs.python.org/3/library/os.html#os.cpu_count
#                cpu_count = len(os.sched_getaffinity(0))
#
#                redirect_path = None
#                if self.redirect_path_attr is not None:
#                    redirect_path = getattr(instance, self.redirect_path_attr)
#
#                self.call_mp_fork(cpu_count, redirect_path,
#                                      instance, method, args, kwargs)
#            else:
#                self.call_std(instance, method, *args, **kwargs)

        return wrapper


#    def call_mp_fork(self, cpu_count, redirect_path, instance, method,
#                     args, kwargs):
#        print("Launch Multiprocess_filler of:", method.__name__)
#        print("cpu count:", multiprocessing.cpu_count())
#        
#        def process_job(key):
#            kwargs[self.iter_kwargs] = key
#            method(instance, *args, **kwargs)
#    
#        with  globalized(process_job
#                ), multiprocessing.Pool(
#                        initializer=redirect_output,
#                        initargs=(redirect_path,),
#                        processes=cpu_count
#                ) as pool:
#            for key in getattr(instance, self.iterable)():
#                pool.apply_async(process_job, (key,))
#            pool.close()
#            pool.join()

    def call_multi_thread(self, instance, method, *args, **kwargs):
        """ """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as threadpool:
#            for key in getattr(instance, self.iterable)():
#                kwargs[self.iter_kwargs] = key
            full_args = (instance,) + args
            def get_kwargs(key):
                kwargs[self.iter_kwargs] = key
                return kwargs

            futures = (
                threadpool.submit(
                    method,
                    *full_args,
                    **get_kwargs(key))
                for key in getattr(instance, self.iterable)()
            )
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
    
    # old
#        with concurrent.futures.ThreadPoolExecutor(
#            max_workers=os.cpu_count()
#        ) as threadpool:
#            for key in getattr(instance, self.iterable)():
#                kwargs[self.iter_kwargs] = key
#                full_args = (instance,) + args
#                threadpool.submit(method, *full_args, **kwargs).result()   
        # from core.py
#        if fs.settings.enable_multithreading:
#            print(">>> Launching multithreading parallel calculation loop")
#            with concurrent.futures.ThreadPoolExecutor(
#                max_workers=os.cpu_count()
#            ) as threadpool:
#                futures = (
#                    threadpool.submit(
#                        self.cycles,
#                        chunk_slice,
#                    )
#                    for chunk_slice in self.chunk_slices()
#                )
#                for fut in concurrent.futures.as_completed(futures):
#                    fut.result()
#        else:
#            print(">>> Launching standard calculation loop")
#            for chunk_slice in self.chunk_slices():
#                self.cycles(chunk_slice)
                
                

    def call_std(self, instance, method, *args, **kwargs):
        """ """
        for key in getattr(instance, self.iterable)():
            kwargs[self.iter_kwargs] = key
            full_args = (instance,) + args
            method(*full_args, **kwargs)
#        with concurrent.futures.ThreadPoolExecutor(max_workers=1
#                ) as threadpool:
#            for key in getattr(instance, self.iterable)():
#                kwargs[self.iter_kwargs] = key
#                full_args = (instance,) + args
#                threadpool.submit(method, *full_args, **kwargs).result()
# ========================
#        if fs.settings.enable_multithreading:
#            print(">>> Launching multithreading parallel calculation loop")
#            with concurrent.futures.ThreadPoolExecutor(
#                max_workers=os.cpu_count()
#            ) as threadpool:
#                futures = (
#                    threadpool.submit(
#                        self.cycles,
#                        chunk_slice,
#                    )
#                    for chunk_slice in self.chunk_slices()
#                )
#                for fut in concurrent.futures.as_completed(futures):
#                    fut.result()
#        else:
#            print(">>> Launching standard calculation loop")
#            for chunk_slice in self.chunk_slices():
#                self.cycles(chunk_slice)