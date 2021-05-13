# -*- coding: utf-8 -*-
import numpy as np
import os
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_agg import FigureCanvasAgg
import fractalshades.colors as fscolors

def test_cbar_im():
    gold = np.array([255, 210, 66]) / 255.
    black = np.array([0, 0, 0]) / 255.

    colors1 = np.vstack((gold[np.newaxis, :],
                         black[np.newaxis, :]))
    colors2 = np.vstack((black[np.newaxis, :],
                         gold[np.newaxis, :]))
    kinds = ["Lch", "Lch"]
    n = 100
    funcs = [lambda x: x**6, lambda x: 1.- (1. - x)**6]

    colormap = fscolors.Fractal_colormap(
            kinds, colors1, colors2, n, funcs, extent="clip")
    colormap.output_png(os.path.normpath("/home/geoffroy/Pictures/math/github_fractal_rep/Fractal-shades"),
                        "test", 1000, 100)

if __name__ == "__main__":
    test_cbar_im()
