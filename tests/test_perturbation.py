# -*- coding: utf-8 -*-
import numpy as np
import unittest
import os
import shutil
import PIL

# Allows relative imports when run locally as script
# https://docs.python-guide.org/writing/structure/
#if __name__ == "__main__":
#    import sys
#    sys.path.insert(0, os.path.abspath(
#            os.path.join(os.path.dirname(__file__), '..')))

#from fractalshades import (Color_tools, Fractal_colormap, Fractal_plotter,
#                                Fractal_Data_array,mkdir_p)
import fractalshades as fs
#from fractalshades.perturbation import Perturbation_mandelbrot
import fractalshades.models as fsmodels

import test_config


def compare_png(ref_file, test_file):
    ref_image = PIL.Image.open(ref_file)
    test_image = PIL.Image.open(test_file)
    
    root, ext = os.path.splitext(test_file)
    diff_file = root + ".diff" + ext
    diff_image = PIL.ImageChops.difference(ref_image, test_image)
    diff_image.save(diff_file)
    errors = np.asarray(diff_image) / 255.
    return np.mean(errors)


class Test_Perturbation_mandelbrot(unittest.TestCase):
        
    def setUp(self):
        image_dir = os.path.join(test_config.test_dir, "images_comparison")
#        if os.path.exists(image_dir):
#            shutil.rmtree(image_dir)
        fs.mkdir_p(image_dir)
        self.image_dir = image_dir

        image_dir_ref = os.path.join(test_config.test_dir, "images_REF")
        fs.mkdir_p(image_dir_ref)
        self.image_dir_ref = image_dir_ref

        purple = np.array([181, 40, 99]) / 255.
        gold = np.array([255, 210, 66]) / 255.
        color_gradient = fs.Color_tools.Lch_gradient(purple, gold,  200)
        self.colormap = fs.Fractal_colormap(color_gradient)
        self.colormap.extent = "mirror"

    @test_config.no_stdout
    def test_M2_E20(self):
        """
        Testing all datatype options, 5e-20 test case.
        """
        x, y = "-1.74928893611435556407228", "0."
        dx = "5.e-20"
        precision = 30
        nx = 600
        test_name = self.test_M2_E20.__name__
        #with tests.suppress_stdout():
        for complex_type in [#np.complex64, np.complex128,
                ("Xrange", np.complex64), ("Xrange", np.complex128)]:
            if type(complex_type) is tuple:
                _, base_complex_type = complex_type
                prefix = "Xr_" + np.dtype(base_complex_type).name
            else:
                prefix = np.dtype(complex_type).name 
            with self.subTest(complex_type=complex_type):


                test_file = self.make_M2_img(x, y, dx, precision, nx,
                                             complex_type, test_name, prefix)
                ref_file = os.path.join(self.image_dir_ref, test_name + ".png")
                err = compare_png(ref_file, test_file)
                self.assertTrue(err < 0.01)
    
    @test_config.no_stdout
    def test_M2_int_E11(self):
        """
        Testing interior detection, 5e-12 test case.
        """
        x, y = "-1.74920463345912691e+00", "-2.8684660237361114e-04"
        dx = "5e-12"
        precision = 16
        nx = 600
        test_name = self.test_M2_int_E11.__name__
        complex_type = np.complex128
        for SA_params in [{"cutdeg": 64, "cutdeg_glitch": 8}, None]:
            with self.subTest(SA_params=SA_params):
                if SA_params is None:
                    prefix = "noSA"
                else:
                    prefix = str(SA_params["cutdeg"]) + "SA"
                test_file = self.make_M2_img(x, y, dx, precision, nx,
                    complex_type, test_name, prefix, interior_detect=True,
                    mask_codes=[2], SA_params=SA_params)
                ref_file = os.path.join(self.image_dir_ref, test_name + ".png")
                err = compare_png(ref_file, test_file)
                self.assertTrue(err < 0.01)
                

    @test_config.no_stdout
    def test_M2_antialias_E0(self):
        """
        Testing field lines, and antialiasing. Full Mandelbrot
        """
        x, y = "-0.75", "0."
        dx = "5.e0"
        precision = 10
        nx = 1600
        test_name = self.test_M2_antialias_E0.__name__
        complex_type = np.complex128
        prefix = "antialiasing"

        gold = np.array([255, 210, 66]) / 255.
        black = np.array([0, 0, 0]) / 255.
        color_gradient = fs.Color_tools.Lch_gradient(gold, black, 200)
        colormap = fs.Fractal_colormap(color_gradient)

        test_file = self.make_M2_img(x, y, dx, precision, nx,
            complex_type, test_name, prefix, interior_detect=True,
            mask_codes=[2], antialiasing=True, colormap=colormap,
            layer="field_lines", probes_val=[0., 0.1])
        ref_file = os.path.join(self.image_dir_ref, test_name + ".png")
        err = compare_png(ref_file, test_file)
        self.assertTrue(err < 0.01)
        
        x, y = "-0.1", "0.975"
        dx = "0.8e0"
        prefix = "antialiasing_2"
        test_file = self.make_M2_img(x, y, dx, precision, nx,
            complex_type, test_name, prefix, interior_detect=True,
            mask_codes=[2], antialiasing=True, colormap=colormap,
            layer="field_lines", probes_val=[0., 0.1],
            field_lines=("field_lines", {"n_iter": 5, "swirl": 1.}))
        ref_file = os.path.join(self.image_dir_ref, test_name + "_2.png")
        err = compare_png(ref_file, test_file)
        self.assertTrue(err < 0.01)


    def make_M2_img(self, x, y, dx, precision, nx, complex_type, test_name,
                    prefix, interior_detect=False, mask_codes=[3, 4], 
                    SA_params={"cutdeg": 64, "cutdeg_glitch": 8},
                    antialiasing=False, colormap=None, layer="shade",
                    probes_val=[0., 0.25], 
                    field_lines=("field_lines", {"n_iter": 10, "swirl": 1.})):
        """
        """
        test_dir = os.path.join(self.image_dir, test_name)

        mandelbrot = fsmodels.Perturbation_mandelbrot(test_dir, x, y, dx, nx,
             xy_ratio=1.,
             theta_deg=0.,
             chunk_size=200,
             complex_type=complex_type,
             projection="cartesian",
             precision=precision,
             antialiasing=antialiasing)

        mandelbrot.full_loop(
            file_prefix=prefix,
            subset=None,
            max_iter=50000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            pc_threshold=0.1,
            SA_params=SA_params,
            glitch_eps=1.e-6,
            interior_detect=interior_detect)

        mask = fs.Fractal_Data_array(mandelbrot, file_prefix=prefix,
            postproc_keys=('stop_reason', lambda x: np.isin(x, mask_codes)),
            mode="r+raw")

        potential = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
        shade = ("DEM_shade", 
                 {"kind": "potential",
                  "theta_LS": 50.,
                  "phi_LS": 40.,
                  "shininess": 40.,
                  "ratio_specular": 8.})

        if colormap is None:
            colormap = self.colormap
        plotter = fs.Fractal_plotter(
            fractal=mandelbrot,
            base_data_key=potential,
            base_data_prefix=prefix,
            base_data_function=lambda x:x,
            colormap=colormap,
            probes_val=probes_val,
            probes_kind="qt",
            mask=mask)

        if layer == "shade":
            plotter.add_grey_layer(
                    postproc_key=shade,
                    intensity=0.95, 
                    blur_ranges=[],
                    hardness=0.75,  
                    shade_type={"Lch": 1.0, "overlay": 0., "pegtop": 4.})
        elif layer == "field_lines":
            plotter.add_grey_layer(
                    postproc_key=field_lines,
                    hardness=0.9, intensity=0.8,
                    blur_ranges=[[0.8, 0.95, 1.0]], 
                    shade_type={"Lch": 1., "overlay": 0., "pegtop": 4.})

        plotter.plot(prefix, mask_color=(0., 0., 1.))
        new_file = os.path.join(self.image_dir,
                                test_name + "_" + prefix + ".png")
        shutil.move(os.path.join(test_dir, prefix + ".png"), new_file)
        shutil.rmtree(test_dir)
        return new_file


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)

    full_test = False
    if full_test:
        runner.run(test_config.suite([Test_Perturbation_mandelbrot]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_Perturbation_mandelbrot("test_M2_antialias_E0"))
        runner.run(suite)
