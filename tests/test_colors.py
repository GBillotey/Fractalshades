# -*- coding: utf-8 -*-
import os
import numpy as np
import unittest

import fractalshades as fs
import fractalshades.utils as fsutils
import test_config


class Test_Fractal_colormap(unittest.TestCase):
    
    def setUp(self):
        cmap_dir = os.path.join(test_config.temporary_data_dir,
                                "_cmap_comparison")
        fsutils.mkdir_p(cmap_dir)
        self.cmap_dir = cmap_dir

        cmap_dir_ref = os.path.join(test_config.ref_data_dir, "cmap_REF")
        fsutils.mkdir_p(cmap_dir_ref)
        self.cmap_dir_ref = cmap_dir_ref
    
    @test_config.no_stdout
    def test_print_cmap(self):
        """ Testing that Fractal_colormap _repr method returns a string
        which can be evaluated to generate the same cmap.
        """
        gold = np.array([255, 210, 66]) / 255.
        black = np.array([0, 0, 0]) / 255.
        purple = np.array([181, 40, 99]) / 255.
        citrus2 = np.array([103, 189, 0]) / 255.
        colors = np.vstack((citrus2[np.newaxis, :],
                             purple[np.newaxis, :]))
        colormap = fs.colors.Fractal_colormap(colors=colors, kinds="Lab",
             grad_npts=200, grad_funcs="x", extent="mirror")
        colormap_inp = eval(colormap.script_repr())
        # Tests that the 2 cmap are the same
        np.testing.assert_almost_equal(colormap.colors, colormap_inp.colors)
        np.testing.assert_almost_equal(colormap._probes, colormap_inp._probes)
        np.testing.assert_almost_equal(colormap._interp_colors,
                                       colormap_inp._interp_colors)

        colors2 = np.vstack((citrus2[np.newaxis, :],
                             black[np.newaxis, :],
                             gold[np.newaxis, :],
                             purple[np.newaxis, :]))
        colormap2 = fs.colors.Fractal_colormap(kinds="Lch", colors=colors2,
             grad_npts=[200, 20, 10], grad_funcs=["x", "x**6", "(1-x)"], extent="mirror")
        colormap_inp2 = eval(colormap2.script_repr())
        # Tests that the 2 cmap are the same
        np.testing.assert_almost_equal(colormap2.colors, colormap_inp2.colors)
        np.testing.assert_almost_equal(colormap2._probes, colormap_inp2._probes)
        np.testing.assert_almost_equal(colormap2._interp_colors,
                                       colormap_inp2._interp_colors)


    def test_cbar_im(self):
        """ Testing export of a Fractal_colormap to png
        """
        black = np.array([0, 0, 0]) / 255.
        red = np.array([255, 0, 0]) / 255.
        yellow = np.array([255, 255, 0]) / 255.

        cmap_name = "black_yellow_black"
        colors = np.vstack((black[np.newaxis, :],
                            yellow[np.newaxis, :],
                            black[np.newaxis, :]))
        fc = fs.colors.Fractal_colormap(colors, "Lch", grad_npts=100, 
                                       grad_funcs=["1.-(1.-x)**2", "x**2"])
        with self.subTest(cmap_name=cmap_name):
            fc.output_png(self.cmap_dir, cmap_name, 1000, 100)
            test_file = os.path.join(self.cmap_dir, cmap_name + ".cbar.png")
            ref_file = os.path.join(self.cmap_dir_ref, cmap_name + ".cbar.png")
            err = test_config.compare_png(ref_file, test_file)
            self.assertTrue(err < 0.0001)

        cmap_name = "red_yellow_black"
        colors = np.vstack((red[np.newaxis, :],
                            yellow[np.newaxis, :],
                            black[np.newaxis, :]))
        fc = fs.colors.Fractal_colormap(colors, "Lch", grad_npts=100, 
                                           grad_funcs=["1.-(1.-x)**0.5", "x**0.5"])
        with self.subTest(cmap_name=cmap_name):
            fc.output_png(self.cmap_dir, cmap_name, 1000, 100)
            test_file = os.path.join(self.cmap_dir, cmap_name + ".cbar.png")
            ref_file = os.path.join(self.cmap_dir_ref, cmap_name + ".cbar.png")
            err = test_config.compare_png(ref_file, test_file)
            self.assertTrue(err < 0.0001)


if __name__ == "__main__":
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_Fractal_colormap]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_Fractal_colormap("test_print_cmap"))
        runner.run(suite)
