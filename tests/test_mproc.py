# -*- coding: utf-8 -*-
import os
import unittest

import numpy as np
from numpy.lib.format import open_memmap
import sys

import fractalshades.utils as fsutils
import fractalshades.settings as fssettings
from fractalshades.mprocessing import Multiprocess_filler #, globalize
import test_config

class Runner:
    def __init__(self, count, cols, directory, mmap_file):
        # OK with C order
        self.count = count
        self.cols = cols
        self.directory = directory
        self.filepath = os.path.join(directory, mmap_file)
    
    @property
    def output(self):
        return "output"

    def crange(self):
        for item in range(self.count):
            yield item

    @Multiprocess_filler(iterable_attr="crange", iter_kwargs="item",
                         redirect_path_attr="output",
                         finalize_attr="_finalize_run")
    def run(self, item=None):
        return self._wrapped_run(item)

    def _wrapped_run(self, item=None):
        vec = item * 10 * np.ones([self.cols - 1])
        val = item * 100
        context = self.multiprocessing_start_method
        if context == "fork":
            self._finalize_run((item, vec, val))
        else:
          return item, vec, val

    def _finalize_run(self, args):
        try:
            item, vec, val = args
        except TypeError:
            # If called from a spawned process
            item, vec, val = args.get()

        mmap = open_memmap(
            filename=os.path.join(self.directory, mmap_file), 
            mode='r+')

        nx, ny = mmap.shape
        mmap[item, 1:ny] = vec #item * 10
        mmap[item, -1] = val #item * 100
        # DO NOT mmap.flush() here - performance hit and no benefit
#        mmap.flush()

        del mmap
        sys.stdout.flush()
        

    
    @Multiprocess_filler(iterable_attr="crange", iter_kwargs="item")
    def run_fancy(self, item=None):
        return self.wrapped_run_fancy(self, item)

    def wrapped_run_fancy(self, item=None):
        # OK with C order
        mmap = open_memmap(
            filename=os.path.join(self.directory, mmap_file), 
            mode='r+')
        nx, ny = mmap.shape
        fancy = np.empty((ny, ), bool)
        fancy[:] = 1 + (-1)**np.arange(ny)
        mmap[item, fancy] = 100
        mmap[item, ~fancy] = -200
        
        # mmap.flush()
        # del mmap

class Test_mproc(unittest.TestCase):
    
    def setUp(self):
        mproc_dir = os.path.join(test_config.temporary_data_dir, "_mproc_dir")
        fsutils.mkdir_p(mproc_dir)
        self.mproc_dir = mproc_dir
    
    def test_memmap(self):
        """
        Testing that a np.memmap array can be filled by multiprocessing batches
        /!\ Note that we do not use mmap.flush() in each subprocess
        """
        # Create a 200 Mb file
        count = 1000
        cols = 50000 # > 3

        global mmap_file
        mmap_file = "runners"
        mmap = open_memmap(
            filename=os.path.join(self.mproc_dir, mmap_file), 
            mode='w+',
            dtype=np.int32,
            shape=(count, cols),
            fortran_order=False,
            version=None)
        mmap.flush()
        del mmap

        runner = Runner(count, cols, self.mproc_dir, mmap_file)
        runner.run()

        mmap = open_memmap(
            filename=os.path.join(self.mproc_dir, mmap_file), 
            mode='r')

        res = mmap[:, 1]
        expected = np.arange(count, dtype=np.int32) * 10
        np.testing.assert_array_equal(res, expected)

        res = mmap[:, cols - 1]
        expected = np.arange(count, dtype=np.int32) * 100
        np.testing.assert_array_equal(res, expected)
#        mmap.flush()
#        del mmap
#        
#        runner.run_fancy()
#        mmap = open_memmap(
#            filename=os.path.join(self.mproc_dir, mmap_file), 
#            mode='r')
#        fancy = np.empty((cols, ), bool)
#        fancy[:] = 1 + (-1)**np.arange(cols)
#        
#        expected = np.empty(cols, dtype=np.int32)
#        expected[:] = -200
#        expected[fancy] = 100
#        mid_count = int(count * 0.5)
#        res = mmap[mid_count, :]
#        np.testing.assert_array_equal(res, expected)
#        del mmap
        
        


if __name__ == "__main__":

    fssettings.multiprocessing_start_method = "fork"
    full_test = False
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_mproc]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_mproc("test_memmap"))
        runner.run(suite)
