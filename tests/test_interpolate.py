# -*- coding: utf-8 -*-
"""
This contains the tests for 2d lienar interpolation routines
"""
import os
import unittest
import time

import numpy as np
import PIL

import fractalshades as fs
import fractalshades.numpy_utils.filters as fsfilters
import fractalshades.utils
import test_config
from fractalshades.numpy_utils.interp2d import (
    Grid_lin_interpolator as fsGrid_lin_interpolator
)
        


class Test_2dinterpolate(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Creates a grid with 'standard' and PIL indexing
        nx = 5
        stepx = 1.
        ny = 3 #5
        stepy = 1.#00.
        
        x_1d = np.linspace(0., stepx * nx, nx + 1)
        y_1d = np.linspace(0., stepy * ny, ny + 1)
        gridx, gridy = np.meshgrid(x_1d, y_1d, indexing='ij')
        gridx_PIL, gridy_PIL = np.meshgrid(x_1d, y_1d[::-1], indexing='xy')
        
        cls.gridx, cls.gridy = gridx, gridy
        cls.gridx_PIL, cls.gridy_PIL = gridx_PIL, gridy_PIL
        cls.a = (0., 0.)   # lower bound
        cls.b = (stepx * nx, stepx * ny) # upper bound
        cls.h = (stepx, stepy) # grid spacing


    def test_patch(self):
        """ Testing the class with a simple 'linear patch test' """
        gridx, gridy = self.gridx, self.gridy
        gridx_PIL, gridy_PIL = self.gridx_PIL, self.gridy_PIL
        a, b, h = self.a, self.b, self.h
        
        # create linear patch surface
        ka = 0.5
        kb = 0.642545
        ka = 0.
        kb = 1.
        f = ka * gridx + kb * gridy
        f_PIL = ka * gridx_PIL + kb * gridy_PIL
        
        # Create 1d test vec
        seed = 1253
        nvec = 10000
        rg = np.random.default_rng(seed)
        pts_x = rg.random([nvec], dtype=np.float64) * self.b[0]
        pts_y = rg.random([nvec], dtype=np.float64) * self.b[1]

        for (PIL_order) in (True, False):
            with self.subTest(PIL_order=PIL_order):

                if PIL_order:
                    interpolator = fsGrid_lin_interpolator(
                            a, b, h, f_PIL, PIL_order=True
                    )

                else:
                    interpolator = fsGrid_lin_interpolator(
                            a, b, h, f, PIL_order=False
                    )

                expected = ka * pts_x + kb * pts_y
                res = interpolator(pts_x, pts_y)

                np.testing.assert_allclose(res, expected, rtol=1.e-12)



    def test_support(self):
        """ Testing that the support is correct ie that val = 0 outside of 4
        square for 1 impulse """
        nx, ny = self.gridx.shape
        a, b, h = self.a, self.b, self.h
        
        
        for (PIL_order) in (False, True):
            with self.subTest(PIL_order=PIL_order):

                for i in range(nx):
                    for j in range(ny):
                        

                        # Create impulse surface response
                        f = np.zeros((nx, ny))
                        f[i, j] = 1.
                        f_PIL = np.swapaxes(f[:, ::-1], 0, 1)

                        # create test vec
                        x_1d = np.arange(self.h[0] * 0.5, self.b[0], self.h[0])
                        y_1d = np.arange(self.h[0] * 0.5, self.b[1], self.h[1])
                        # note the order does not matter here
                        pts_x, pts_y = np.meshgrid(x_1d, y_1d, indexing='ij')
                        pts1, pts2 = pts_x.shape
                        assert pts1 == nx - 1
                        assert pts2 == ny - 1
                        

                        if PIL_order:
                            interpolator = fsGrid_lin_interpolator(
                                a, b, h, f_PIL, PIL_order=True #f_PIL, PIL_order=True
                            )
        
                        else:
                            interpolator = fsGrid_lin_interpolator(
                                a, b, h, f, PIL_order=False
                            )
                        
                        res = interpolator(pts_x.reshape(-1), pts_y.reshape(-1))
                        res = res.reshape((pts1, pts2))
                        
                        non_zero_k1, non_zero_k2 = np.nonzero(res)
                        len_non_zeros = len(non_zero_k1)
                        
                        for ik in range(len_non_zeros):
                            loc_k1 = non_zero_k1[ik]
                            loc_k2 = non_zero_k2[ik]
                            val_loc = res[loc_k1, loc_k2]
                            assert val_loc == 0.25
                            if loc_k1 != i or loc_k1 != i - 1:
                                print("loc_k1", loc_k1, "i", i)
                            assert loc_k1 == i or loc_k1 == i - 1
                            assert loc_k2 == j or loc_k2 == j - 1


    def test_equivalent(self):
        """ Testing the equivalence of the 2 PIL_order options """
        # gridx, gridy = self.gridx, self.gridy
        a, b, h = self.a, self.b, self.h
        nx, ny = self.gridx.shape

        # create linear patch surface
        seed = 1253
        nvec = nx * ny 
        rg = np.random.default_rng(seed)
        f = rg.random([nvec], dtype=np.float64).reshape(nx, ny)
        f_PIL = np.swapaxes(f[:, ::-1], 0, 1)

        # create interpolating test points
        rg = np.random.default_rng(seed)
        pts_x = rg.random([nvec], dtype=np.float64) * self.b[0]
        pts_y = rg.random([nvec], dtype=np.float64) * self.b[1]


        interpolator = fsGrid_lin_interpolator(
                a, b, h, f_PIL, PIL_order=True
        )
        expected = interpolator(pts_x, pts_y)

        interpolator_PIL = fsGrid_lin_interpolator(
                a, b, h, f, PIL_order=False
        )
        res = interpolator_PIL(pts_x, pts_y)
        
        np.testing.assert_allclose(res, expected, rtol=1.e-12)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    full_test = True
    if full_test:
        runner.run(test_config.suite([Test_2dinterpolate]))
    else:
        suite = unittest.TestSuite()
        # suite.addTest(Test_2dinterpolate("test_equivalent"))
        # suite.addTest(Test_2dinterpolate("test_patch"))
        suite.addTest(Test_2dinterpolate("test_support"))
        runner.run(suite)
