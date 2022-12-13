# -*- coding: utf-8 -*-
import os
import unittest
import shutil

import numpy as np
import numba

import fractalshades as fs
import fractalshades.utils as fsutils
import fractalshades.postproc as fspp
#from fractalshades.mprocessing import Multiprocess_filler
import test_config

from fractalshades.postproc import (
    Postproc_batch,
)

class Dummy_Fractal(fs.Fractal):
    """ Minimal test implementation """

    @fsutils.calc_options
    def calc1(self, calc_name, subset=None):
        complex_codes = ["z", "_z2", "z3"]
        int_codes = ["int_1"]  # reference FP
        stop_codes = ["max_iter"]

        def set_state():
            def impl(instance):
                instance.codes = (complex_codes, int_codes, stop_codes)
                instance.complex_type = np.complex128
            return impl

        def initialize():
            @numba.njit
            def numba_impl(Z, U, c):
                Z[2] = c ** 3
                Z[1] = c ** 2
                Z[0] = c
                U[0] = 1
            return numba_impl
        
        def iterate():
            @numba.njit
            def numba_impl(c, Z, U, stop_reason):
                n_iter = 0
                while True:
                    n_iter += 1
                    if np.abs(c) > 1:
                        Z[0] = 0.
                        Z[1] = 0.
                        Z[2] = 0.
                        U[0] = 0
                    if n_iter >= 0:
                        stop_reason[0] = 0
                        break
                return n_iter
            return numba_impl
        

        return {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }


class Test_postproc(unittest.TestCase):
    
    def setUp(self):
        pp_dir = os.path.join(test_config.temporary_data_dir, "_postproc_dir")
        fsutils.mkdir_p(pp_dir)
        self.pp_dir = pp_dir
        self.fractal = Dummy_Fractal(pp_dir)
    
    def test_arr(self):
        """
        Testing Fractal simple calc, storing of result arrays, re-reading them
        through a Fractal_array object.
        """
        f = self.fractal
        f.clean_up("test")

        f.zoom(x=0., y=0., dx=3., nx=800, xy_ratio=1, theta_deg=0.)
        f.calc1(calc_name="test")

        pp_z = fspp.Fractal_array(f, "test", "z", func=None)
        pp_z3 = fspp.Fractal_array(f, "test", "z3", func=None)
        pp_i1 = fspp.Fractal_array(f, "test", "int_1", func=None)
        stop_reason = fspp.Fractal_array(f, "test", "stop_reason", func=None)
        stop_iter = fspp.Fractal_array(f, "test", "stop_iter", func=None)

        pp = Postproc_batch(f, "test")
        plotter = fs.Fractal_plotter(pp)  
        plotter.plot()

        for chunk in f.chunk_slices():
            c_pix = np.ravel(
                self.fractal.chunk_pixel_pos(
                    chunk, jitter=False, supersampling=None
                )
            )
            c = 0. + 3. * c_pix

            invalid = (np.abs(c) > 1)

            expected_z = c.copy()
            expected_z[invalid] = 0.
            np.testing.assert_array_equal(expected_z, pp_z[chunk])

            expected_z3 = expected_z ** 3
            np.testing.assert_allclose(expected_z3, pp_z3[chunk])

            expected_i1 = np.where(invalid, 0, 1)
            np.testing.assert_array_equal(expected_i1, pp_i1[chunk])

            expected_stop = np.zeros(c.shape, np.int32)
            expected_stop_iter = np.ones(c.shape, np.int32)
            np.testing.assert_array_equal(expected_stop, stop_reason[chunk])
            np.testing.assert_array_equal(expected_stop_iter, stop_iter[chunk])
            

    def tearDown(self):
        pp_dir = self.pp_dir
        try:
            shutil.rmtree(os.path.join(pp_dir, "multiproc_calc"))
        except FileNotFoundError:
            pass

        


if __name__ == "__main__":
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_postproc]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_postproc("test_arr"))
        runner.run(suite)
