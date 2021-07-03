# -*- coding: utf-8 -*-
import unittest
from contextlib import contextmanager
from functools import wraps
import os
import sys

test_dir = os.path.dirname(__file__)

test_dir = "/home/geoffroy/Pictures/math/github_fractal_rep/Fractal-shades/tests/images_comparison/test_M2_int_E11"
file_prefix = "64SA" #"noSA"
mandelbrot = fsmodels.Perturbation_mandelbrot(test_dir)
mandelbrot.zoom(
             precision=1,
             x="0.0",
             y="0.0",
             dx="0.0",
             nx=600,
             xy_ratio=1.,
             theta_deg=0.)
tile = next(mandelbrot.chunk_slices())
params, codes, raw_data = mandelbrot.reload_data_chunk(tile, file_prefix)

if __name__ == "__main__":
    pass
