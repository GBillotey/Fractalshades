# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools, Fractal_Data_array
from classical_mandelbrot import Classical_mandelbrot


choice = 1

def plot(choice):
    """========================================================================
    Run the script to generate 1 of 4 fractal images from the classic
    Mandelbrot set, z(n+1) <- zn**2 + c (c = pixel).
    User-input needed : *choice* and the target *directory*
    ========================================================================"""


    select = {1: plot_test_blender}
#              2: plot_billiard_game_blender,
#              3: plot_medallion_blender,
#              5: plot_atoll}

    directory = {1: "/home/geoffroy/Pictures/math/classic/test_order"}

    # run the selected plot
    select[choice](directory[choice], choice)


def plot_test_blender(directory, choice):
    """
    Plots the "Billiard game" image.
    """
    #==========================================================================
    # Parameters
    
    dx = 0.7
    x = -0.125
    y = 0.90
    x = -0.75
    y = 0.
    dx = 3.
    
    x = -1.74920463345912691e+00
    y = -2.8684660237361114e-04
    dx = 5.e-12 * 4500
    
    xy_ratio = 1.0
    theta_deg = 0.
    nx = 600
    #known_order = 1
    complex_type = np.complex128

    #==========================================================================
    # Calculations
    mandelbrot = Classical_mandelbrot(directory, x, y, dx, nx, xy_ratio,
                                      theta_deg, chunk_size=200,
                                      complex_type=complex_type,
                                      projection="cartesian")
    mandelbrot.clean_up("explore")
    mandelbrot.explore_loop(file_prefix="explore",
        subset=None,
        max_iter=2000000,
        M_divergence = 1.e3,
        epsilon_stationnary = 1e-3,
        pc_threshold=1.,
        px_ball=3.)

    colormap_ball = -Fractal_colormap((0.1, 0.99, 200), plt.get_cmap("magma"))
    colormap_ball.extent = "mirror"
    plotter = Fractal_plotter(
        fractal=mandelbrot,
        base_data_key=("raw", {"code": "ball_period"}),
        base_data_prefix="explore",
        base_data_function=lambda x: np.log(1. + x),
        colormap=colormap_ball,
        probes_val=[0., 1.],
        probes_kind="qt",
        mask=None)
    layer_key = ("DEM_explore",
        {"px_snap": 0.5, "potential_dic": {"kind": "infinity"}})
    plotter.add_grey_layer(postproc_key=layer_key, intensity=0.5, 
                           normalized=False, skewness=0.0, 
                           shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})
    plotter.plot("explore")


if __name__ == "__main__":
    plot(choice)