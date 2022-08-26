# -*- coding: utf-8 -*-
import os
import functools
import concurrent.futures
import fractalshades as fs


class Multithreading_iterator():
    def __init__(self, iterable_attr, iter_kwargs="key", veto_parallel=False):
        """
Decorator class for multithreading looping of an instance-method.

Parameters:
-----------
iterable_attr : string 
    getattr(instance, iterable_attr) is a Generator function (i.e.
    `yields` the successive values).
iter_kwargs : string
    name of the wrapped method keyword-argument which will be filled with
    successive values yielded by the generator function pointed to by
    iterable_attr
veto_parallel : bool
    if True, defaults to normal iteration without multi-threading

Usage:
------
@Multiprocess_filler(iterable_attr, iter_kwarg)
def method(self, *args, iter_kwarg=None, **otherkwargs):
    (... CPU-intensive calculations ...)
    return None
"""
        self.iterable_attr = iterable_attr
        self.iter_kwargs = iter_kwargs
        self.veto_parallel = veto_parallel

    def __call__(self, method):
        @functools.wraps(method)
        def wrapper(instance, *args, **kwargs):
            parallel = (
                fs.settings.enable_multithreading and not self.veto_parallel
            )
            if parallel:
                self.call_multi_thread(instance, method, *args, **kwargs)
            else:
                self.call_std(instance, method, *args, **kwargs)
        return wrapper

    def call_multi_thread(self, instance, method, *args, **kwargs):
        """ Parallel (multi-threading) loop """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as threadpool:
            full_args = (instance,) + args
            def get_kwargs(key):
                kwargs[self.iter_kwargs] = key
                return kwargs

            futures = (
                threadpool.submit(
                    method,
                    *full_args,
                    **get_kwargs(key)
                )
                for key in getattr(instance, self.iterable_attr)()
            )
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    def call_std(self, instance, method, *args, **kwargs):
        """ Standard loop """
        for key in getattr(instance, self.iterable_attr)():
            kwargs[self.iter_kwargs] = key
            full_args = (instance,) + args
            method(*full_args, **kwargs)
