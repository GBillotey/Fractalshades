# -*- coding: utf-8 -*-
"""
This contains the tests for the various graphical exports option,
including complex layering etc.
"""
import os
import unittest
import pickle

import numpy as np

import fractalshades as fs
import fractalshades.utils as fsutils
import fractalshades.models as fsm
from fractalshades.postproc import (
    Postproc_batch,
    Fractal_array
)
import test_config


class Test_pickle(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        fs.settings.enable_multiprocessing = True

        pickle_dir = os.path.join(test_config.temporary_data_dir, "_pickle_dir")
        fsutils.mkdir_p(pickle_dir)
        cls.pickle_dir = pickle_dir
        cls.calc_name = "pickle"
        cls.dir_ref = os.path.join(test_config.ref_data_dir, "pickle_REF")

        x = -0.5
        y = 0.
        dx = 5.
        nx = 400
        f = fsm.Mandelbrot(pickle_dir)
        f.clean_up(cls.calc_name)

        f.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0,
               theta_deg=0., projection="cartesian")
        f.calc_std_div(
            calc_name=cls.calc_name,
            subset=None,
            max_iter=1000,
            M_divergence=100.,
            epsilon_stationnary= 0.001,
        )
        cls.f = f
        
    def test_pickle_fractal_array(self):
        """ Check that is it possible to pickle / unpickle a Fractal Array
        """
        f = self.f
        pp = Postproc_batch(f, "pickle")
        plotter = fs.Fractal_plotter(pp)
        plotter.plot()

        f_arr = Fractal_array(
            f, self.calc_name, "stop_reason", func="x ! =1"
        )

        chunk_slice = next(f.chunk_slices())
        expected = f_arr[chunk_slice]
        
        save_path = os.path.join(self.pickle_dir, "test_arr.pickle") 
        with open(save_path, 'wb+') as tmpfile:
            pickle.dump(f_arr, tmpfile, pickle.HIGHEST_PROTOCOL)

        with open(save_path, "rb") as tmpfile:
            f_arr2 = pickle.load(tmpfile)
        
        chunk_slice = next(f.chunk_slices())
        f_arr2.fractal = f
        expected2 = f_arr2[chunk_slice]
        
        self.assertEqual(np.sum(expected), np.sum(expected2))


    def test_pickle_fractal(self):
        """ Just basic pickling / unpickling Fractal object without a full
        equality test """
        f = self.f
        save_path = os.path.join(self.pickle_dir, "test_fractal.pickle") 
        with open(save_path, 'wb+') as tmpfile:
            pickle.dump(f, tmpfile, pickle.HIGHEST_PROTOCOL)

        with open(save_path, "rb") as tmpfile:
            pickle.load(tmpfile)




if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    full_test = True
    if full_test:
        runner.run(test_config.suite([Test_pickle]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_pickle("test_pickle_fractal_array"))
        runner.run(suite)
