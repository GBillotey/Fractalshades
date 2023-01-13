# -*- coding: utf-8 -*-
"""
This contains the tests for the various graphical exports option,
including complex layering etc.
"""
import os
import unittest

import numpy as np
import PIL

import fractalshades as fs
import fractalshades.numpy_utils.filters as fsfilters
import fractalshades.utils
#import fractalshades.models as fsm
#from fractalshades.postproc import (
#    Postproc_batch,
#    Fractal_array
#)
import test_config


class Test_filters(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
#        fs.settings.enable_multiprocessing = True
#
#        pickle_dir = os.path.join(test_config.temporary_data_dir, "_pickle_dir")
#        fsutils.mkdir_p(pickle_dir)
#        cls.pickle_dir = pickle_dir
#        cls.calc_name = "pickle"
        dir_ref = os.path.join(test_config.ref_data_dir, "perturb_REF")
        img_path = os.path.join(
                dir_ref,
                "Color_layer_test_supersampling_potential_2-2.REF.png"
        )
        pic = PIL.Image.open(img_path)
        cls.rgb_arr = np.array(pic).astype(np.float32)
        
        cls.filter_dir = os.path.join(
                test_config.temporary_data_dir, "_filter_dir")
        fs.utils.mkdir_p(cls.filter_dir)
        
        cls.decimator = fsfilters.Lanczos_decimator()

#
#        x = -0.5
#        y = 0.
#        dx = 5.
#        nx = 400
#        f = fsm.Mandelbrot(pickle_dir)
#        f.clean_up(cls.calc_name)
#
#        f.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0,
#               theta_deg=0., projection="cartesian")
#        f.calc_std_div(
#            calc_name=cls.calc_name,
#            subset=None,
#            max_iter=1000,
#            M_divergence=100.,
#            epsilon_stationnary= 0.001,
#        )
#        cls.f = f
        
    def test_Lanczos2_filter(self):
        """ Applying a Lanczos2 filter for decimation
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        (nx, ny, nchannel) = rgb_arr.shape
        decimator = self.decimator
        
#        for decimation in (2, 3, 4, 5, 6):
#            with self.subTest(decimation=decimation):
#                res_arr = np.empty(
#                    (nx // decimation, ny // decimation, nchannel),
#                    dtype=np.uint8
#                )
#                lf2 = decimator.get_impl(2, decimation)
#                for ic in range(nchannel):
#                    res_arr[:, :, ic] = lf2(rgb_arr[:, :, ic]).astype(np.uint8)

        for decimation in (2, 3, 4, 5, 6):
            with self.subTest(decimation=decimation):
                
                res_arr = np.empty(
                    (nx // decimation, ny // decimation, nchannel),
                    dtype=np.uint8
                )

                
                lf2 = decimator.get_impl(2, decimation)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = np.clip(
                        lf2(rgb_arr[:, :, ic]) + 0.5, 0., 255.
                    ).astype(np.uint8)
                
                res_im = PIL.Image.fromarray(res_arr)
                res_im.save(os.path.join(
                        filter_dir, "LF2_" + str(decimation) + ".png"
                ))

    def test_Lanczos3_filter(self):
        """ Applying a Lanczos3 filter for decimation
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        (nx, ny, nchannel) = rgb_arr.shape
        decimator = self.decimator

        for decimation in (2, 3, 4, 5, 6):
            with self.subTest(decimation=decimation):
                res_arr = np.empty(
                    (nx // decimation, ny // decimation, nchannel),
                    dtype=np.uint8
                )
                lf2 = decimator.get_impl(2, decimation)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = lf2(rgb_arr[:, :, ic]).astype(np.uint8)

        for decimation in (2, 3, 4, 5, 6):
            with self.subTest(decimation=decimation):

                res_arr = np.empty(
                    (nx // decimation, ny // decimation, nchannel),
                    dtype=np.uint8
                )

                lf2 = decimator.get_impl(2, decimation)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = np.clip(
                        lf2(rgb_arr[:, :, ic]) + 0.5, 0., 255.
                    ).astype(np.uint8)
                
                res_im = PIL.Image.fromarray(res_arr)
                res_im.save(os.path.join(
                        filter_dir, "LF3_" + str(decimation) + ".png"
                ))


    def test_Lanczos_filter_dtype(self):
        pass
        


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    full_test = False
    if full_test:
        runner.run(test_config.suite([Test_filters]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_filters("test_Lanczos2_filter"))
        suite.addTest(Test_filters("test_Lanczos3_filter"))
        # suite.addTest(Test_filters("test_Lanczos2_filter"))
        runner.run(suite)
