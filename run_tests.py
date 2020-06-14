# -*- coding: utf-8 -*-
import os
from run_mandelbrot import plot_classic_draft

def plot_tests():
    """
    Run a few sanity tests based on classical Mandelbrot.
    (image size, x / y ratio, rotation, ...)
    """
    ntests = 5
    test_params = [{"x": -0.75, "y": 0.0, "dx": 3., "xy_ratio": 1., 
                        "theta_deg": 0, "nx": 400},
                   {"x": -0.75, "y": 0.0, "dx": 3., "xy_ratio": 1., 
                        "theta_deg": 0, "nx": 357}, 
                   {"x": -0.75, "y": 0.0, "dx": 3., "xy_ratio": 1., 
                        "theta_deg": 45, "nx": 400},
                   {"x": -0.75, "y": 0.1, "dx": 3., "xy_ratio": 2., 
                        "theta_deg": 0, "nx": 357}, 
                   {"x": -0.75, "y": 0.1, "dx": 3., "xy_ratio": 0.5, 
                        "theta_deg": -45, "nx": 633}]
    test_main_dir = "/home/gby/Pictures/Mes_photos/math/fractal/tests"
    test_dirs = [os.path.join(test_main_dir,
                              str(itest)) for itest in range(ntests)]

    for itest in range(ntests):
        plot_classic_draft(test_dirs[itest], **test_params[itest])

if __name__ == "__main__":
    plot_tests()