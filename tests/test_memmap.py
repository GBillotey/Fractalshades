# -*- coding: utf-8 -*-
import os
import unittest
import concurrent.futures
import contextlib
import sys
import uuid

import numpy as np
from numpy.lib.format import open_memmap


import fractalshades.utils as fsutils
import fractalshades.settings as fssettings
import test_config


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


class Test_mmap(unittest.TestCase):
    
    def setUp(self):
        _mmap_dir = os.path.join(test_config.temporary_data_dir, "_mmap_dir")
        fsutils.mkdir_p(_mmap_dir)
        self._mmap_dir = _mmap_dir
        self.process_file = "test_mmap_process"
        self.thread_file = "test_mmap_thread"
        self.nx = 100
        self.ny = 100

    def run_mmap(self, file):
        print("Executing our Task on Process: {}".format(os.getpid()))
        arr = np.ones([self.nx, self.ny], dtype=np.float32)
        
        mmap = open_memmap(
            filename=os.path.join(self._mmap_dir, file), 
            mode='w+',
            dtype=np.float32,
            shape=(self.nx, self.ny),
            fortran_order=False,
            version=None)
        mmap[:] = arr
        print("Finished our Task on Process: {}".format(os.getpid()))
        

    def test_in_thread(self):
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=1) as threadpool:
            threadpool.submit(self.run_mmap, self.process_file).result()
        print("in main loop")
            
        # check correct execution
        mmap = open_memmap(
            filename=os.path.join(self._mmap_dir, self.process_file), 
            mode='r+',
            dtype=np.float32,
            shape=(self.nx, self.ny),
            fortran_order=False,
            version=None)
        res = mmap[:]
        print("arr", res)
        expected = np.ones([self.nx, self.ny], dtype=np.float32)
        np.testing.assert_array_equal(res, expected)

   




if __name__ == "__main__":

    fssettings.multiprocessing_start_method = "fork"
    full_test = False
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_mmap]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_mmap("test_in_thread"))
        runner.run(suite)

