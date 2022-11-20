# -*- coding: utf-8 -*-
"""
This contains the tests for the various graphical exports option,
including complex layering etc.
"""
import os
import unittest
import shutil
import copy
import pickle

import numpy as np

import fractalshades as fs
import fractalshades.utils as fsutils
import fractalshades.colors as fscolors
#import fractalshades.postproc as fspp
import fractalshades.models as fsm
from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    DEM_normal_pp,
    Raw_pp,
    Fieldlines_pp,
    Attr_normal_pp,
    Attr_pp,
    Fractal_array
)
import test_config
from fractalshades.colors.layers import (
    Color_layer,
    Grey_layer,
    Bool_layer,
    Normal_map_layer,
    Virtual_layer,
    Blinn_lighting,
    Overlay_mode
)


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
        f.base_calc(
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
        # f.run()
        # A Fractal_array with the interior points
        f_arr = Fractal_array(
            f, self.calc_name, "stop_reason", func="x ! =1"
        )

        chunk_slice = next(f.chunk_slices())
        expected = f_arr[chunk_slice]
        print("expected", expected, np.sum(expected))
        
        save_path = os.path.join(self.pickle_dir, "test_arr.pickle") 
        with open(save_path, 'wb+') as tmpfile:
            pickle.dump(f_arr, tmpfile, pickle.HIGHEST_PROTOCOL)
            
            
        print("##################### UNPICKLING")

        with open(save_path, "rb") as tmpfile:
            f_arr2 = pickle.load(tmpfile)
        print("UNPICKLED", f_arr)
        
        chunk_slice = next(f.chunk_slices())
        expected = f_arr2[chunk_slice]
        print("expected", expected, np.sum(expected))


    def test_pickle_fractal(self):
        f = self.f
        save_path = os.path.join(self.pickle_dir, "test_fractal.pickle") 
        with open(save_path, 'wb+') as tmpfile:
            pickle.dump(f, tmpfile, pickle.HIGHEST_PROTOCOL)
        
        with open(save_path, "rb") as tmpfile:
            f_res = pickle.load(tmpfile)
        print("UNPICKLED", f_res)
##    @test_config.no_stdout
#    def test_pickled_color_basic(self):
#
#
#        """ Testing basic pickling """
##        g = copy.copy(self.f)
#        save_path = os.path.join(self.pickle_dir, "test.pickle") 
#        with open(save_path, 'wb+') as tmpfile:
#            pickle.dump(self.f, tmpfile, pickle.HIGHEST_PROTOCOL)
#        
#        with open(save_path, 'rb') as tmpfile:
#            g = pickle.load(tmpfile)
#        g.clean_up(self.calc_name)
#        g.run()
#
#        pp = Postproc_batch(g, self.calc_name)
#        pp.add_postproc("cont_iter", Continuous_iter_pp())
#        pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
#        plotter = fs.Fractal_plotter(pp)   
#        plotter.add_layer(Bool_layer("interior", output=False))
#        plotter.add_layer(Color_layer(
#                "cont_iter",
#                func="np.log(x)",
#                colormap=self.colormap,
#                probes_z=[0., 0.2],
#                probes_kind="relative",
#                output=True
#        ))
#        plotter["cont_iter"].set_mask(
#                plotter["interior"],
#                mask_color=(0., 0., 0.)
#        )
#        plotter.plot()
#        self.layer = plotter["cont_iter"]
#        self.check_current_layer()



    def check_current_layer(self, err_max=0.01):
        """ Compare with stored reference image
        """
        layer = self.layer
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        test_file_path = os.path.join(self.layer_dir, file_name + ".png")
        ref_file_path = os.path.join(self.dir_ref, file_name + ".REF.png")
        err = test_config.compare_png(ref_file_path, test_file_path)
        self.assertTrue(err < err_max)



if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    full_test = False
    if full_test:
        runner.run(test_config.suite([Test_pickle]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_pickle("test_pickle_fractal_array"))
        runner.run(suite)
