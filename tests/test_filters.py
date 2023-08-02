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
import test_config


class Test_filters(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        test_dir = os.path.dirname(__file__)
        ref_img_dir = os.path.join(
            test_dir, "..", "docs", "_static"
            ""
        )

        img_path = os.path.join(ref_img_dir, "gaia.jpg")
        pic = PIL.Image.open(img_path)
        cls.rgb_arr = np.array(pic)
        cls.float_arr = np.array(pic).astype(np.float32)
        (cls.nx, cls.ny, nchannel) = cls.rgb_arr.shape
        print(cls.nx, cls.ny)

        cls.filter_dir = os.path.join(
                test_config.temporary_data_dir, "_filter_dir"
        )
        fs.utils.mkdir_p(cls.filter_dir)
        
        cls.decimator = fsfilters.Lanczos_decimator()
        cls.decimations = (2, 3, 4, 5, 6)
        cls.REF_decimated_img = {}
        for decimation in cls.decimations:
            dec_nx = cls.nx // decimation + (cls.nx % decimation != 0)
            dec_ny = cls.ny // decimation + (cls.ny % decimation != 0)
            ref_img = pic.resize((dec_ny, dec_nx), PIL.Image.LANCZOS)
            save_path = os.path.join(cls.filter_dir, f"REF_{decimation}.png")
            ref_img.save(save_path)
            cls.REF_decimated_img[decimation] = save_path
        
        dec_stable_nx = cls.nx // 2 + 1
        dec_stable_ny = cls.ny // 2 + 1
        ref_img = pic.resize((dec_stable_ny, dec_stable_nx), PIL.Image.LANCZOS)
        save_path = os.path.join(cls.filter_dir, f"REF_{2}_stable.png")
        ref_img.save(save_path)
        cls.REF_decimated_img[(2, "stable")] = save_path


        cls.upsampler = fsfilters.Lanczos_upsampler()
        cls.ups = (2, 3)
        cls.REF_ups_img = {}
        for ups in cls.ups:
            ups_nx = (cls.nx - 1) * ups + 1
            ups_ny = (cls.ny - 1) * ups + 1
            ref_img = pic.resize((ups_ny, ups_nx), PIL.Image.LANCZOS)
            save_path = os.path.join(cls.filter_dir, f"REF_ups_{ups}.png")
            ref_img.save(save_path)
            cls.REF_ups_img[ups] = save_path



    def test_Lanczos2_decimate(self):
        """ Applying a Lanczos2 filter for decimation
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr #[:-1, :1596, :]
        float_arr = self.float_arr
        (nx, ny, nchannel) = rgb_arr.shape
        decimator = self.decimator

        gen = ((decimation, dtype)
            for decimation in self.decimations
            for dtype in ("rgb", "float")
        )

        for (decimation, dtype) in gen:
            with self.subTest(decimation=decimation, dtype=dtype):

                res_nx = nx // decimation + (nx % decimation != 0)
                res_ny = ny // decimation + (ny % decimation != 0)
                res_arr = np.empty((res_nx, res_ny, nchannel), dtype=np.uint8)

                lf2 = decimator.get_impl(2, decimation)
                for ic in range(nchannel):
                    if dtype == "rgb":
                        res_arr[:, :, ic] = lf2(rgb_arr[:, :, ic])
                    else:
                        res_arr[:, :, ic] = np.clip(
                            lf2(float_arr[:, :, ic]) + 0.5, 0., 255.
                        ).astype(np.uint8)

                res_im = PIL.Image.fromarray(res_arr)
                save_path = os.path.join(
                    filter_dir, f"LF2_{decimation}_{dtype}.png"
                )
                res_im.save(save_path)
                self.check_image(
                    self.REF_decimated_img[decimation], save_path, 0.03
                )


    def test_Lanczos3_decimate(self):
        """ Applying a Lanczos3 filter for decimation
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr #[:-1, :1596, :]
        float_arr = self.float_arr
        (nx, ny, nchannel) = rgb_arr.shape
        decimator = self.decimator
        
        gen = ((decimation, dtype)
            for decimation in self.decimations
            for dtype in ("rgb", "float")
        )

        for (decimation, dtype) in gen:
            with self.subTest(decimation=decimation, dtype=dtype):

                res_nx = nx // decimation + (nx % decimation != 0)
                res_ny = ny // decimation + (ny % decimation != 0)
                res_arr = np.empty((res_nx, res_ny, nchannel), dtype=np.uint8)

                lf3 = decimator.get_impl(3, decimation)
                for ic in range(nchannel):
                    if dtype == "rgb":
                        res_arr[:, :, ic] = lf3(rgb_arr[:, :, ic])
                    else:
                        res_arr[:, :, ic] = np.clip(
                            lf3(float_arr[:, :, ic]) + 0.5, 0., 255.
                        ).astype(np.uint8)

                res_im = PIL.Image.fromarray(res_arr)
                save_path = os.path.join(
                    filter_dir, f"LF3_{decimation}_{dtype}.png"
                )
                res_im.save(save_path)
                self.check_image(
                    self.REF_decimated_img[decimation], save_path, 0.03
                )

    def test_Lanczos_stable_decimate(self):
        """ Applying a Lanczos3 filter for decimation
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        float_arr = self.float_arr
        (nx, ny, nchannel) = rgb_arr.shape
        decimator = self.decimator
        
        gen = ((a, dtype)
            for a in (2, 3)
            for dtype in ("rgb", "float")
        )

        for (a, dtype) in gen:
            with self.subTest(a=a, dtype=dtype):

                res_nx = nx // 2 + 1
                res_ny = ny // 2 + 1
                res_arr = np.empty((res_nx, res_ny, nchannel), dtype=np.uint8)

                lf_stable = decimator.get_stable_impl(a)
                for ic in range(nchannel):
                    if dtype == "rgb":
                        arr, kx, ky = lf_stable(rgb_arr[:, :, ic])
                        res_arr[:, :, ic] = arr
                    else:
                        arr, kx, ky = lf_stable(float_arr[:, :, ic])
                        res_arr[:, :, ic] = np.clip(
                            arr + 0.5, 0., 255.
                        ).astype(np.uint8)

                res_im = PIL.Image.fromarray(res_arr)
                save_path = os.path.join(
                    filter_dir, f"LF_{a}_stable_{dtype}.png"
                )
                res_im.save(save_path)
                # Image is a little "zoom out" as expected, hence correlation
                # a little lower
                self.check_image(
                    self.REF_decimated_img[(2, "stable")], save_path, 0.05
                )


    def test_Lanczos2_upsampling(self):
        """ Applying a Lanczos2 filter for upsampling
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        float_arr = self.float_arr
        (nx, ny, nchannel) = float_arr.shape
        upsampler = self.upsampler
        
        gen = ((ups, dtype)
            for ups in (2, 3)
            for dtype in ("rgb", "float")
        )

        for (ups, dtype) in gen:
            with self.subTest(upsampling=ups, dtype=dtype):

                res_arr = np.empty(
                    ((nx - 1) * ups + 1, (ny - 1) * ups + 1, nchannel),
                    dtype=np.uint8
                )
                ups_2 = upsampler.get_impl(2, ups)
                for ic in range(nchannel):
                    if dtype == "rgb":
                        res_arr[:, :, ic] = ups_2(rgb_arr[:, :, ic])
                    else:
                        res_arr[:, :, ic] = np.clip(
                            ups_2(float_arr[:, :, ic]) + 0.5, 0., 255.
                        ).astype(np.uint8)

                res_im = PIL.Image.fromarray(res_arr)
                
                save_path = os.path.join(
                    filter_dir, f"LF2_ups_{ups}__{dtype}.png"
                )
                res_im.save(save_path)
                self.check_image(
                    self.REF_ups_img[ups], save_path, 0.03
                )


    def test_Lanczos3_upsampling(self):
        """ Applying a Lanczos3 filter for upsampling
        """
        filter_dir = self.filter_dir
        rgb_arr = self.rgb_arr
        float_arr = self.float_arr
        (nx, ny, nchannel) = float_arr.shape
        upsampler = self.upsampler
        
        gen = ((ups, dtype)
            for ups in (2, 3)
            for dtype in ("rgb", "float")
        )

        for (ups, dtype) in gen:
            with self.subTest(upsampling=ups, dtype=dtype):

                res_arr = np.empty(
                    ((nx - 1) * ups + 1, (ny - 1) * ups + 1, nchannel),
                    dtype=np.uint8
                )
                ups_3 = upsampler.get_impl(3, ups)
                for ic in range(nchannel):
                    if dtype == "rgb":
                        res_arr[:, :, ic] = ups_3(rgb_arr[:, :, ic])
                    else:
                        res_arr[:, :, ic] = np.clip(
                            ups_3(float_arr[:, :, ic]) + 0.5, 0., 255.
                        ).astype(np.uint8)

                res_im = PIL.Image.fromarray(res_arr)
                
                save_path = os.path.join(
                    filter_dir, f"LF3_ups_{ups}__{dtype}.png"
                )
                res_im.save(save_path)
                self.check_image(
                    self.REF_ups_img[ups], save_path, 0.03
                )
        


    def check_image(self, ref_file_path, test_file_path, err_max=0.01):
        """ Compare with stored reference image
        """
        err = test_config.compare_png(ref_file_path, test_file_path)
        print("err < err_max", err, err_max)
        self.assertTrue(err < err_max)




        



if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    full_test = True
    if full_test:
        runner.run(test_config.suite([Test_filters]))
    else:
        suite = unittest.TestSuite()
        # suite.addTest(Test_filters("test_Lanczos2_filter"))
        #suite.addTest(Test_filters("test_Lanczos3_decimate"))
#        suite.addTest(Test_filters("test_Lanczos2_decimate"))
#        suite.addTest(Test_filters("test_Lanczos3_decimate"))
#        suite.addTest(Test_filters("test_Lanczos_stable_decimate"))
        # suite.addTest(Test_filters("test_Lanczos2_upsampling"))
        # suite.addTest(Test_filters("test_Lanczos3_upsampling"))
        # suite.addTest(Test_filters("test_Lanczos2_filter"))
        suite.addTest(Test_filters("test_final_77"))

        runner.run(suite)
