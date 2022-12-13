# -*- coding: utf-8 -*-
import os
import unittest

#import numpy as np
#from numpy.lib.format import open_memmap

import fractalshades.models as fsm
import fractalshades.utils as fsutils
#from fractalshades.mprocessing import Multiprocess_filler
import test_config



class Test_core(unittest.TestCase):
    
    def setUp(self):
        core_dir = os.path.join(test_config.temporary_data_dir, "_core_dir")
        fsutils.mkdir_p(core_dir)
        self.core_dir = core_dir

        x = '-1.0'
        y = '-0.0'
        dx = '5.0'

        xy_ratio = 1.0
        dps = 77
        nx = 600

        fractal = fsm.Perturbation_mandelbrot(core_dir)
        fractal.zoom(precision=dps, x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=0., projection="cartesian")
        self.fractal = fractal

    
    def test_chunk_count(self):
        """ Test the various Fractal methods linked to chunk indexing"""
        fractal = self.fractal

        for (nx, xy_ratio) in [
                (600, 1.0),
                (601, 1.0),
                (601, 1.1),
                (600, 1.1),
                (600, 0.9),
                (601, 0.9),
                (6000, 3.0),
                (6000, 0.3),
                (6000, 3.05),
                (6000, 0.333),
                (50, 0.333),
                ]:
            fractal.nx = nx
            fractal.xy_ratio = xy_ratio
            with self.subTest(nx=nx, xy_ratio=xy_ratio):
                counter = 0
                for _ in fractal.chunk_slices():
                    counter += 1
                self.assertTrue(counter == fractal.chunks_count)

                counter = 0
                for chunk_slice in fractal.chunk_slices():
                    rank = fractal.chunk_rank(chunk_slice)
                    self.assertTrue(counter == rank)
                    self.assertTrue(chunk_slice == fractal.chunk_from_rank(
                                    rank))
                    counter += 1
        
        




if __name__ == "__main__":
    full_test = False
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_core]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_core("test_chunk_count"))
        runner.run(suite)
