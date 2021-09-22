# -*- coding: utf-8 -*-
import multiprocessing
import os
import sys
import functools

import fractalshades.utils as fsutils
import fractalshades.settings as fssettings


def process_init(job, redirect_path):
    """
    Save *job* as a global for the child-process ; redirects stdout and stderr.
    -> 'On Unix a child process can make use of a shared resource created in a
    parent process using a global resource. However, it is better to pass the
    object as an argument to the constructor for the child process.'
    """
    global process_job
    process_job = job
    if redirect_path is not None:
        fsutils.mkdir_p(redirect_path)
        out_file = str(os.getpid())
        sys.stdout = open(os.path.join(redirect_path, out_file + ".out"), "a")
        sys.stderr = open(os.path.join(redirect_path, out_file + ".err"), "a")

def job_proxy(key):
    """ returns result of global job variable from the child-process """
    global process_job
    return process_job(key)

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
        self.iter_kwargs = iter_kwargs
        self.veto_multiprocess = veto_multiprocess

    def __call__(self, method):
        @functools.wraps(method)
        def wrapper(instance, *args, **kwargs):
            iterable = getattr(instance, self.iterable)
            res = None
            if self.res is not None:
                res = getattr(instance, self.res)
            def job(key):
                kwargs[self.iter_kwargs] = key
                return method(instance, *args, **kwargs)

            if (fssettings.enable_multiprocessing
                and not(self.veto_multiprocess)):
                print("Launch Multiprocess_filler of ", method.__name__)
                print("cpu count:", multiprocessing.cpu_count())

                redirect_path=None
                if self.redirect_path_attr is not None:
                    redirect_path = getattr(instance, self.redirect_path_attr)

                with multiprocessing.Pool(
                        initializer=process_init,
                        initargs=(job, redirect_path),
                        processes=multiprocessing.cpu_count()) as pool:
                    worker_res = {key: pool.apply_async(job_proxy, (key,))
                                  for key in iterable()}
                    for key, val in worker_res.items():
                        if res is not None:
                            res[key] = val.get()
                        else:
                            val.get()
                    pool.close()
                    pool.join()
            else:
                for key in iterable():
                    if res is not None:
                        res[key] = job(key)
                    else:
                        job(key)
        return wrapper
