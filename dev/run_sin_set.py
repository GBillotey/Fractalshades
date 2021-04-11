# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools
from Sin_set import Sin_set


def plot():
    """========================================================================
    Run the script to generate 1 of 2 fractal images from the 'power tower'
    Mandelbrot set, z(n+1) <- c**zn (c = pixel).
    User-input needed : *choice* and the target *directory*
    ========================================================================"""
    choice = 1

    select = {1: plot_sin_set}

    directory = {1: "/home/gby/Pictures/Mes_photos/math/fractal/sin0"}   

    # run the selected plot
    select[choice](directory[choice])




def plot_sin_set(directory):
    """
  
    """
    #==========================================================================
    #  Parameters
    x = 0.
    y = 0.
    dx = 2 * np.pi
    xy_ratio = 1.0
    theta_deg = 0. 
    nx = 600
    pc = 0.2
    complex_type = np.complex128

    #==========================================================================
    #  Calculations
    mandelbrot = Sin_set(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type)
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 1000000,
                          epsilon_stationnary = 1e-3, epsilon_cv=None)
    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                  == 3)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.01 * dx / float(nx)
    epsilon_cv = eps_pixel
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
                                  subset=stationnary, max_iter=1000000,
                                  eps_pixel=eps_pixel, start_file="loop-1",
                                  k_power=1.5)
    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 1)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=1000000,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")
        
    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=16, max_cycle=100000, eps_cv=1.e-8,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=None)
    valid = np.abs(mandelbrot.raw_data('attractivity', file_prefix="newton")
               <= 1)
    convergence = (mandelbrot.raw_data("r", file_prefix="newton") == 1)

#    mandelbrot.first_loop(file_prefix="loop-2", subset=stationnary,
#                          max_iter=4000, epsilon_cv=epsilon_cv,
#                          epsilon_stationnary=None, pc_threshold=pc,
#                          zr_file_prefix="newton")
#    convergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-2")
#                   == 1)


    #==========================================================================
    #   Plots everything except the minibrot "convergent" part

#    base_data_prefix = "loop-2"
#    base_data_key = ("potential", 
#                     {"kind": "convergent", "epsilon_cv": epsilon_cv})
#
#    base_data_function = lambda x: np.log(x)
#    probe_values = [0.0005, 0.50,  0.9995]
#
#    copper = Fractal_colormap((0., .99, 200), plt.get_cmap("copper"))
#    base_colormap = copper - copper
#    
#    plotter_potential = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix, 
#                            base_data_function, base_colormap, probe_values,
#                            mask=None)
#
#    # shade layer based on field lines
#    layer2_key = ("field_lines", {})
#    Fourrier = ([0.] * 8 + [1.], [0, 0.,])
#    plotter_potential.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
#                         hardness= 0.25, intensity=0.6, skewness= -0.6,
#                         blur_ranges=[[0.99, 0.999, 1.0]], 
#                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 1.})  
#
#
#    # shade layer based on potential DEM
#    layer_key = ("DEM_shade", {"kind": "potential",
#                               "theta_LS": 135.,
#                                "phi_LS": 40.,
#                                "shininess": 100.,
#                                "ratio_specular": 1.4})
#    plotter_potential.add_NB_layer(postproc_key=layer_key,# Fourrier=Fourrier,
#                         hardness= 1.0, intensity=0.8, skewness= 0.0,
#                         blur_ranges=[[0.9999, 0.99999, 1.0]], 
#                         shade_type={"Lch": 4., "overlay": 1., "pegtop": 1.})  
#
#    black = np.array([0, 0, 0]) / 255.
#    plotter_potential.plot("conv", mask_color=black)

    #==========================================================================
    # Plot the divergent part, in theory shall be a sparse set. Here bailout is
    # based on numerical overflow, an limit the max reach of a limit cycle - 
    # which can be arbitrary large.
    base_data_prefix = "loop-1"
    base_data_key = ("phase", {"source": "zn"})
    base_data_function = lambda x: -np.cos(2* x - np.pi*0.1)
    probe_values = [0.0, 1.00]
    base_colormap = Fractal_colormap((0.9, 0.1, 200), plt.get_cmap("copper"))
    plotter_inf = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix, 
                                  base_data_function, base_colormap,
                                  probe_values, mask=~divergence)             
    black = np.array([255, 0, 0]) / 255.
    plotter_inf.plot("div", mask_color=black)


    #==========================================================================
    # plots the "reaching limit cycle" part (color based on cycle order)
    base_data_key =  ("raw", {"code": "r"})
    base_data_prefix = "newton"

#    base_data_key =  ("raw", {"code": "r_candidate"})
#    base_data_prefix = "r_candidate"

    base_data_function = lambda x: np.where(x > 10., 10., x)
    probe_values = [0.0, 0.13, 0.14, 0.40, 0.50, 0.55, 0.70, 0.75, 0.85]
    
    gold = np.array([255, 210, 66]) / 255.
    light_yellow = np.array([255, 252, 199]) / 255.
    purple = np.array([181, 40, 99]) / 255.
    dark_blue = np.array([32, 52, 164]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    black = np.array([0, 0, 0]) / 255.
    grey = np.array([66, 0, 31]) / 255.
    paon_green = np.array([82, 233, 104]) / 255.
    
    color_gradient1 = Color_tools.Lab_gradient(black, grey, 200)
    color_gradient2 = Color_tools.Lab_gradient(grey, purple, 200)
    color_gradient3 = Color_tools.Lch_gradient(purple, gold, 100) 
    color_gradient4 = Color_tools.Lch_gradient(gold, light_yellow, 100) 
    color_gradient5 = Color_tools.Lch_gradient(light_yellow, paon_green, 100) 
    color_gradient6 = Color_tools.Lch_gradient(paon_green, dark_cyan, 100)
    color_gradient7 = Color_tools.Lch_gradient(dark_cyan, dark_blue, 100)
    color_gradient8 = Color_tools.Lab_gradient(dark_blue, black, 100)
    colormap_newton = (Fractal_colormap(color_gradient1) + 
                       Fractal_colormap(color_gradient2) + 
                       Fractal_colormap(color_gradient3) + 
                       Fractal_colormap(color_gradient4) + 
                       Fractal_colormap(color_gradient5) + 
                       Fractal_colormap(color_gradient6) + 
                       Fractal_colormap(color_gradient7)+ 
                       Fractal_colormap(color_gradient8))

    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_newton,
                              probe_values, mask=~cycling)

    # shade layer based on normal vec with a little "tuning" to get finer
    # shade details
    layern_key2 = ("power_attr_shade", {"theta_LS": -45.,
                                        "phi_LS": 70.,
                                        "shininess": 100.,
                                        "ratio_specular": 1.})
    plotter.add_NB_layer(postproc_key=layern_key2, intensity=0.95,
                           skewness=0.20, hardness=1.0,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter.plot("cycling", mask_color=(0, 0, 0, 0))

    # ==== Composite plot : cycling + divergent part
    plotter.blend_plots("div.png", "cycling.png")


if __name__ == "__main__":
    plot()