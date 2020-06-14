# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools
from Nova import Nova


def plot():
    """========================================================================
    Run the script to generate 1 of 4 fractal images
    User-input needed : *choice* and the target *directory*
    ========================================================================"""
    choice = 1

    select = {1: plot_nova6_whole_set,
              2: plot_nova6_zoom}

    directory = {1: "/home/gby/Pictures/Mes_photos/math/fractal/nova0",
                 2: "/home/gby/Pictures/Mes_photos/math/fractal/nova2"}   

    # run the selected plot
    select[choice](directory[choice])


def plot_nova6_whole_set(directory):
    """
    Plots Nova Mandelbrot - degree 6 with R = 0.8
    """
    #==========================================================================
    #   Parameters
    x = -0.5
    y = 0.
    dx = 2.5
    
    xy_ratio = 1.0
    theta_deg = 0. 
    nx = 3200
    complex_type = np.complex128
    known_order = None

    R = 0.8
    p = 6
    z0 = 1.
    epsilon_cv = 1e-8

    #==========================================================================
    #   Calculations
    mandelbrot = Nova(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type)
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 100000,
                          epsilon_cv=None, epsilon_stationnary=1e-4,
                          R=R, p=p, z0=z0)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.1 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=400000,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05,
        R=R, p=p, z0=z0)
    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 1)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=300000,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate", R=R, p=p, z0=z0)

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=30, max_cycle=20000, eps_cv=1.e-12,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order, R=R, p=p, z0=z0)
    real_loop = (mandelbrot.raw_data("r", file_prefix="newton") > 1)
    convergence = (mandelbrot.raw_data("r", file_prefix="newton") == 1)

    mandelbrot.first_loop(file_prefix="loop-2", subset=convergence,
                          max_iter=100000, epsilon_cv=epsilon_cv,
                          epsilon_stationnary=None,
                          R=R, p=p, z0=z0, zr_file_prefix="newton")
    convergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-2")
                   == 1)

    #==========================================================================
    #   Plots everything except the minibrot "convergent" part

    base_data_prefix = "loop-2"
    base_data_key = ("potential", 
                     {"kind": "convergent", "epsilon_cv": epsilon_cv})

    base_data_function = lambda x: np.log(x)
    probe_values = [0.0, 0.80,  0.999]

    copper = Fractal_colormap((0., .99, 200), plt.get_cmap("copper"))
    base_colormap = copper - copper
    
    plotter_potential = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix, 
                            base_data_function, base_colormap, probe_values,
                            mask=None)

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([0.] * 8 + [1.], [0, 0.,])
    plotter_potential.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness= 0.25, intensity=0.6, skewness= -0.6,
                         blur_ranges=[[0.99, 0.999, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 1.})  


    # shade layer based on potential DEM
    layer_key = ("DEM_shade", {"kind": "potential",
                               "theta_LS": 135.,
                                "phi_LS": 40.,
                                "shininess": 100.,
                                "ratio_specular": 1.4})
    plotter_potential.add_NB_layer(postproc_key=layer_key,# Fourrier=Fourrier,
                         hardness= 1.0, intensity=0.8, skewness= 0.0,
                         blur_ranges=[[0.999, 0.9995, 1.0]], 
                         shade_type={"Lch": 4., "overlay": 1., "pegtop": 1.})  

    black = np.array([0, 0, 0]) / 255.
    plotter_potential.plot("conv", mask_color=black)
    
    #==========================================================================
    #   Plots the minibrot, then blends
    base_data_key =  ("abs", {"source": "attractivity"})
    base_data_prefix = "newton"
    base_data_function = lambda x: x
    probe_values = [0.0, 1.0]

    black = np.array([0, 0, 0]) / 255.
    silver = np.array([200, 220, 220])/255. 
    color_gradient = Color_tools.Lab_gradient(silver, black, 100,
                                          f=lambda x: x**6)   
    colormap_newton =  Fractal_colormap(color_gradient)

    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_newton, probe_values,
                              mask=~real_loop)

    # shade layer based on normal vec : 
    layern_key2 = ("attr_shade", {"theta_LS": 135.,
                                "phi_LS": 35.,
                                "shininess": 100.,
                                "ratio_specular": 1.})
    plotter.add_NB_layer(postproc_key=layern_key2, intensity=0.95,
                           skewness=0.20, hardness=1.0,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter.plot("minibrot", mask_color=(0, 0, 0, 0))

    # ==== Composite plot : Minibrot + 'convergent' part
    plotter.blend_plots("conv.png", "minibrot.png")


def plot_nova6_zoom(directory):
    """
    Plots an embedded Julia in an elephant valley of Nova 6
    """
    #==========================================================================
    #   Parameters
    x = -0.343915827897
    y = 0.000167563886
    dx = 1.92888900E-09
    xy_ratio = 1.0#1.0
    theta_deg = 0. 
    nx = 3200
    complex_type = np.complex128
    known_order = 82

    R = 1.0
    p = 6
    z0 = 1.
    epsilon_cv = 1e-8

    #==========================================================================
    #   Calculations
    mandelbrot = Nova(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type)
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 100000,
                          epsilon_cv=None, epsilon_stationnary=1e-4,
                          R=R, p=p, z0=z0)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.5 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=400000,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05,
        R=R, p=p, z0=z0)
    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 1)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=300000,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate", R=R, p=p, z0=z0)

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=30, max_cycle=20000, eps_cv=1.e-14,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order, R=R, p=p, z0=z0)
    real_loop = (mandelbrot.raw_data("r", file_prefix="newton") > 1)
    convergence = (mandelbrot.raw_data("r", file_prefix="newton") == 1)

    mandelbrot.first_loop(file_prefix="loop-2", subset=convergence,
                          max_iter=100000,
                          epsilon_cv=epsilon_cv, epsilon_stationnary=None,
                          R=R, p=p, z0=z0, zr_file_prefix="newton")
    convergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-2")
                   == 1)

    #==========================================================================
    #   Plots everything except the minibrot "convergent" part

    base_data_prefix = "loop-2"
    base_data_key = ("potential", 
                     {"kind": "convergent", "epsilon_cv": epsilon_cv})

    base_data_function = lambda x: np.log(x)
    probe_values = [0.0, 0.70, 0.85, 0.95, 0.99]

    copper = Fractal_colormap((0., .99, 200), plt.get_cmap("copper"))
    black = np.array([0, 0, 0]) / 255.
    silver = np.array([200, 220, 220])/255.
    color_gradient = Color_tools.Lab_gradient(black, silver, 100)   
    gr_colormap =  Fractal_colormap(color_gradient)
    base_colormap = copper - copper + gr_colormap - gr_colormap
    plotter_potential = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix, 
                                  base_data_function, base_colormap, probe_values,
                                  mask=real_loop)

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([0.] * 8 + [1.], [0, 0.,])
    plotter_potential.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness= 0.25, intensity=0.6, skewness= -0.6,
                         blur_ranges=[[0.98, 0.995, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 1.})  

    # shade layer based on potential DEM
    layer_key = ("DEM_shade", {"kind": "potential",
                               "theta_LS": 135.,
                                "phi_LS": 40.,
                                "shininess": 100.,
                                "ratio_specular": 1.4})
    plotter_potential.add_NB_layer(postproc_key=layer_key,# Fourrier=Fourrier,
                         hardness= 1.0, intensity=0.8, skewness= 0.0,
                         blur_ranges=[[0.95, 0.98, 1.0]], 
                         shade_type={"Lch": 4., "overlay": 1., "pegtop": 1.})  

    plotter_potential.plot("conv", mask_color=black)

    #==========================================================================
    # Plots the minibrot, then blends
    # colors based on cycle attractivity
    base_data_key =  ("abs", {"source": "attractivity"})#{"raw", }
    base_data_prefix = "newton"
    base_data_function = lambda x: x
    probe_values = [0.0, 1.0]

    light_emerauld = np.array([15, 230, 186]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_emerauld, black, 100,
                                          f=lambda x: x**6)   
    colormap_newton =  Fractal_colormap(color_gradient)

    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                             base_data_function, colormap_newton, probe_values,
                             mask=~real_loop)

    # shade layer based on normal vec
    layern_key2 = ("attr_shade", {"theta_LS": 135.,
                                "phi_LS": 35.,
                                "shininess": 100.,
                                "ratio_specular": 1.})
    plotter.add_NB_layer(postproc_key=layern_key2, intensity=0.95,
                           skewness=0.20, hardness=1.0,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter.plot("minibrot", mask_color=(0, 0, 0, 0))

    #==========================================================================
    #   Composite plot : Minibrot + 'convergent' part
    plotter.blend_plots("conv.png", "minibrot.png")


if __name__ == "__main__":
    plot()