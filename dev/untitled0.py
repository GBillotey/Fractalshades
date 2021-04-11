#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:41:47 2020

@author: geoffroy
"""

# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil

# Allows relative imports when run locally as script
# https://docs.python-guide.org/writing/structure/
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')))


from src.perturbation import Perturbation_mandelbrot
from src.fractal import (Color_tools,
                         Fractal_colormap,
                         Fractal_plotter,
                         Fractal_Data_array,
                         mkdir_p)

#
#def compare_png(ref_file, test_file):
#    ref_image = PIL.Image.open(ref_file)
#    test_image = PIL.Image.open(test_file)
#    
#    root, ext = os.path.splitext(test_file)
#    diff_file = root + ".diff" + ext
#    diff_image = PIL.ImageChops.difference(ref_image, test_image)
#    diff_image.save(diff_file)
#    errors = np.asarray(diff_image) / 255.
#    return np.mean(errors)


class Dev_image():
        
    def setUp(self):
        image_dir = os.path.join("/home/geoffroy/Pictures/math/github_fractal_rep/Fractal-shades/dev",
                                 "im_dev0")
        self.image_dir = image_dir

        purple = np.array([181, 40, 99]) / 255.
        gold = np.array([255, 210, 66]) / 255.
        black = np.array([0, 0, 0]) / 255.
        color_gradient = Color_tools.Lch_gradient(gold, black,  200)
        self.colormap = Fractal_colormap(color_gradient)
        self.colormap.extent = "clip"


    def dev_M2_E0(self):
        """
        Ran 1 test in 88.734s
        """
        x, y = "-0.75", "0."
        dx = "5.e0"
        precision = 10
        nx = 800
        test_name = self.dev_M2_E0.__name__
        complex_type = np.complex128
        self.make_M2_img(x, y, dx, precision, nx,
                                     complex_type, test_name)


    def dev_M2_E11(self):
        """
        Ran 1 test in 88.734s
        """
        x, y = "-1.74920463345912691e+00", "-2.8684660237361114e-04"
        dx = "2e-11"
        precision = 16
        nx = 600
        test_name = self.dev_M2_E11.__name__
        complex_type = np.complex128
        self.make_M2_img(x, y, dx, precision, nx,
                                     complex_type, test_name)



    def make_M2_img(self, x, y, dx, precision, nx, complex_type, test_name):
        """
        """
        test_dir = os.path.join(self.image_dir, test_name)
        if type(complex_type) is tuple:
            _, base_complex_type = complex_type
            prefix = "Xr_" + np.dtype(base_complex_type).name
        else:
            prefix = np.dtype(complex_type).name 

        mandelbrot = Perturbation_mandelbrot(test_dir, x, y, dx, nx,
             xy_ratio=1.,
             theta_deg=0.,
             chunk_size=200,
             complex_type=complex_type,
             projection="cartesian",
             precision=precision,
             antialiasing=True)

        mandelbrot.full_loop(
            file_prefix=prefix,
            subset=None,
            max_iter=50000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            pc_threshold=0.0,
            SA_params={"cutdeg": 64, "cutdeg_glitch": 8},
            glitch_eps=1.e-6,
            interior_detect=True)
        non_escaped = Fractal_Data_array(mandelbrot, file_prefix=prefix,
                postproc_keys=('stop_reason', lambda x: np.isin(x, [0, 2])), mode="r+raw")

        potential = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
        shade = ("DEM_shade", {"kind": "potential",
                                "theta_LS": 50.,
                                "phi_LS": 40.,
                                "shininess": 40.,
                                "ratio_specular": 8.})
        plotter = Fractal_plotter(
            fractal=mandelbrot,
            base_data_key=potential,
            base_data_prefix=prefix,
            base_data_function=lambda x:x,
            colormap=self.colormap,
            probes_val=[0., 0.04],
            probes_kind="qt",
            mask=non_escaped)

#        plotter.add_grey_layer(
#                postproc_key=shade,
#                intensity=0.95, 
#                blur_ranges=[],
#                hardness=0.85,  
#                shade_type={"Lch": 1.0, "overlay": 0., "pegtop": 4.})
        
        # shade layer based on field lines
        layer2_key = ("field_lines", {})
        Fourrier = ([1.,], [0, 0.,])
        plotter.add_grey_layer(postproc_key=layer2_key,# Fourrier=Fourrier,
                             hardness=0.9, intensity=0.7,
                             blur_ranges=[[0.8, 0.95, 1.0]], 
                             shade_type={"Lch": 1., "overlay": 0., "pegtop": 4.}) 

        plotter.plot(prefix, mask_color=(0., 0., 1.))
        new_file = os.path.join(self.image_dir,
                                test_name + "_" + prefix + ".png")
#        shutil.move(os.path.join(test_dir, prefix + ".png"), new_file)
#        shutil.rmtree(test_dir)
        return new_file


if __name__ == "__main__":
    d = Dev_image()
    d.setUp()
    d.dev_M2_E0()
