# -*- coding: utf-8 -*-
import os
import sys
import unittest

import numpy as np
from numpy.lib.format import open_memmap

import fractalshades as fs
from fractalshades.mthreading import Multithreading_iterator
import test_config

fs.settings.enable_multithreading = True


class Test_mproc(unittest.TestCase):
    
    def setUp(self):
        mproc_dir = os.path.join(test_config.temporary_data_dir, "_mproc_dir")
        fs.utils.mkdir_p(mproc_dir)
        self.mproc_dir = mproc_dir
    
    def test_memmap(self):
        """
        Testing that a np.memmap array can be filled by multithreading batches
        /!\ Note that we do not use mmap.flush() in each subprocess
        """
        # Create a 200 Mb file
        count = 10000
        cols = 5000 # > 3

        class Runner:
            def __init__(self, count, directory, mmap_file):
                # OK with C order
                self.count = count
                self.directory = directory
                self.filepath = os.path.join(directory, mmap_file)
                self.multiprocess_dir = os.path.join(directory,
                                                     "multiprocess_dir")

            def crange(self):
                for item in range(self.count):
                    yield item

            @Multithreading_iterator(iterable_attr="crange",
                                     iter_kwargs="item")
            def run(self, item):
                # OK with C order
#                print("in run", item)
                sys.stdout.flush()
                mmap = open_memmap(
                    filename=os.path.join(self.directory, mmap_file), 
                    mode='r+')
                nx, ny = mmap.shape
                mmap[item, 1:ny] = item * 10
                mmap[item, -1] = item * 100
                # DO NOT mmap.flush() here - performance hit and no benefit
                #del mmap
            
            @Multithreading_iterator(iterable_attr="crange",
                                     iter_kwargs="item")
            def run_fancy(self, item):
                # OK with C order
                mmap = open_memmap(
                    filename=os.path.join(self.directory, mmap_file), 
                    mode='r+')
                nx, ny = mmap.shape
                fancy = np.empty((ny, ), bool)
                fancy[:] = 1 + (-1)**np.arange(ny)
                mmap[item, fancy] = 100
                mmap[item, ~fancy] = -200
                #del mmap


        mmap_file = "runners"
        mmap = open_memmap(
            filename=os.path.join(self.mproc_dir, mmap_file), 
            mode='w+',
            dtype=np.int32,
            shape=(count, cols),
            fortran_order=False,
            version=None)
        # mmap.flush()
        del mmap

        print("runner")
        runner = Runner(count, self.mproc_dir, mmap_file)
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
        del mmap
#        
        runner.run_fancy()
        mmap = open_memmap(
            filename=os.path.join(self.mproc_dir, mmap_file), 
            mode='r')
        fancy = np.empty((cols, ), bool)
        fancy[:] = 1 + (-1)**np.arange(cols)
        
        expected = np.empty(cols, dtype=np.int32)
        expected[:] = -200
        expected[fancy] = 100
        mid_count = int(count * 0.5)
        res = mmap[mid_count, :]
        np.testing.assert_array_equal(res, expected)
        del mmap
        
        
        



if __name__ == "__main__":
    full_test = False
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_mproc]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_mproc("test_memmap"))
        runner.run(suite)
