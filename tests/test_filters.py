# -*- coding: utf-8 -*-
"""
This contains the tests for the various graphical exports option,
including complex layering etc.
"""
import os
import unittest
import time

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
        
        test_dir = os.path.dirname(__file__)
        ref_img_dir = os.path.join(
            test_dir, "..", "docs", "_static"
            ""
        )

        # temporary_data_dir = os.path.join(test_dir, "_temporary_data")
#        ref_data_dir = os.path.join(test_dir, "REFERENCE_DATA")
#        
#        dir_ref = os.path.join(test_config.ref_data_dir, "perturb_REF")
#        dir_ref = os.path.join(test_config.ref_data_dir, "perturb_REF")
        
        img_path = os.path.join(
                ref_img_dir,
                "deep_julia_BS.jpg"
                # "gaia.jpg"
        )
        pic = PIL.Image.open(img_path)
        cls.rgb_arr = np.array(pic).astype(np.float32)
        
        cls.filter_dir = os.path.join(
                test_config.temporary_data_dir, "_filter_dir")
        fs.utils.mkdir_p(cls.filter_dir)
        
        cls.decimator = fsfilters.Lanczos_decimator()
        cls.upsampler = fsfilters.Lanczos_upsampler()


        
    def test_Lanczos2_decimate(self):
        """ Applying a Lanczos2 filter for decimation
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        (nx, ny, nchannel) = rgb_arr.shape
        decimator = self.decimator
        

        for decimation in (2, 4, 5):#, 6):
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

    def test_Lanczos3_decimate(self):
        """ Applying a Lanczos3 filter for decimation
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        (nx, ny, nchannel) = rgb_arr.shape
        decimator = self.decimator

        for decimation in (2, 4, 5): #, 6):
            with self.subTest(decimation=decimation):
                res_arr = np.empty(
                    (nx // decimation, ny // decimation, nchannel),
                    dtype=np.uint8
                )
                lf3 = decimator.get_impl(3, decimation)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = lf3(rgb_arr[:, :, ic]).astype(np.uint8)
                    
        

        for decimation in (2, 4, 5): #, 6):
            with self.subTest(decimation=decimation):
                t_filter = - time.time()

                res_arr = np.empty(
                    (nx // decimation, ny // decimation, nchannel),
                    dtype=np.uint8
                )

                lf3 = decimator.get_impl(3, decimation)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = np.clip(
                        lf3(rgb_arr[:, :, ic]) + 0.5, 0., 255.
                    ).astype(np.uint8)
                
                res_im = PIL.Image.fromarray(res_arr)
                res_im.save(os.path.join(
                        filter_dir, "LF3_" + str(decimation) + ".png"
                ))

                t_filter += time.time()
                print("t_filter", t_filter)
#                t_filter 0.15944552421569824
#                t_filter 0.16670966148376465
#                t_filter 0.13544988632202148


    def test_Lanczos2_upsampling(self):
        """ Applying a Lanczos2 filter for upsampling
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        (nx, ny, nchannel) = rgb_arr.shape
        upsampler = self.upsampler

        for usp in (2, 3): #, 6):
            with self.subTest(upsampling=usp):
                res_arr = np.empty(
                    ((nx - 1) * usp + 1, (ny - 1) * usp + 1, nchannel),
                    dtype=np.uint8
                )
                lf2 = upsampler.get_impl(2, usp)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = lf2(rgb_arr[:, :, ic]).astype(np.uint8)
                    
        

        for usp in (2, 3): #, 6):
            with self.subTest(upsampling=usp):
                t_filter = - time.time()

                res_arr = np.empty(
                    ((nx - 1) * usp + 1, (ny - 1) * usp + 1, nchannel),
                    dtype=np.uint8
                )

                lf2 = upsampler.get_impl(2, usp)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = np.clip(
                        lf2(rgb_arr[:, :, ic]) + 0.5, 0., 255.
                    ).astype(np.uint8)
                
                res_im = PIL.Image.fromarray(res_arr)
                res_im.save(os.path.join(
                        filter_dir, "LF2_ups_" + str(usp) + ".png"
                ))

                t_filter += time.time()
                print("t_filter", t_filter)
#                t_filter 1.6039433479309082
#                t_filter 4.082881450653076
#                t_filter 8.059937000274658
#                t_filter 13.8790123462677
        
    def test_Lanczos3_upsampling(self):
        """ Applying a Lanczos3 filter for upsampling
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        (nx, ny, nchannel) = rgb_arr.shape
        upsampler = self.upsampler

        for usp in (2, 3): #, 6):
            with self.subTest(upsampling=usp):
                res_arr = np.empty(
                    ((nx - 1) * usp + 1, (ny - 1) * usp + 1, nchannel),
                    dtype=np.uint8
                )
                lf3 = upsampler.get_impl(3, usp)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = lf3(rgb_arr[:, :, ic]).astype(np.uint8)
                    
        

        for usp in (2, 3): #, 6):
            with self.subTest(upsampling=usp):
                t_filter = - time.time()

                res_arr = np.empty(
                    ((nx - 1) * usp + 1, (ny - 1) * usp + 1, nchannel),
                    dtype=np.uint8
                )

                lf3 = upsampler.get_impl(3, usp)
                for ic in range(nchannel):
                    res_arr[:, :, ic] = np.clip(
                        lf3(rgb_arr[:, :, ic]) + 0.5, 0., 255.
                    ).astype(np.uint8)
                
                res_im = PIL.Image.fromarray(res_arr)
                res_im.save(os.path.join(
                        filter_dir, "LF3_ups_" + str(usp) + ".png"
                ))

                t_filter += time.time()
                print("t_filter", t_filter)
#                t_filter 1.6039433479309082
#                t_filter 4.082881450653076
#                t_filter 8.059937000274658
#                t_filter 13.8790123462677




if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    full_test = False
    if full_test:
        runner.run(test_config.suite([Test_filters]))
    else:
        suite = unittest.TestSuite()
        # suite.addTest(Test_filters("test_Lanczos2_filter"))
        # suite.addTest(Test_filters("test_Lanczos3_decimate"))
        suite.addTest(Test_filters("test_Lanczos2_upsampling"))
        suite.addTest(Test_filters("test_Lanczos3_upsampling"))
        # suite.addTest(Test_filters("test_Lanczos2_filter"))
        runner.run(suite)
