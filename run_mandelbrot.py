# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools
from classical_mandelbrot import Classical_mandelbrot


def plot():
    """========================================================================
    Run the script to generate 1 of 4 fractal images from the classic
    Mandelbrot set, z(n+1) <- zn**2 + c (c = pixel).
    User-input needed : *choice* and the target *directory*
    ========================================================================"""
    choice = 2

    select = {1: plot_classic_draft,
              2: plot_billiard_game,
              3: plot_medallion,
              4: plot_emerauld_shield}

    directory = {1: "/home/gby/Pictures/Mes_photos/math/fractal/classic/1",
                 2: "/home/gby/Pictures/Mes_photos/math/fractal/classic/2",
                 3: "/home/gby/Pictures/Mes_photos/math/fractal/classic/3",
                 4: "/home/gby/Pictures/Mes_photos/math/fractal/classic/4"}   

    # run the selected plot
    select[choice](directory[choice])


def plot_classic_draft(directory, **params):
    """
    Plots classical Mandelbrot in 'draft' mode
    Use for exploration only
    """
    #==========================================================================
    #  Parameters - if not provided, default to a general view of the set
    x = params.get("x", -0.75)
    y = params.get("y", -0.0)
    dx = params.get("dx", 3)
    xy_ratio = params.get("xy_ratio", 1.0)
    theta_deg = params.get("theta_deg", 0.)
    nx = params.get("nx", 600)
    complex_type = np.complex128

    #==========================================================================
    #  Calculations
    mandelbrot = Classical_mandelbrot(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type)
    mandelbrot.draft_loop(file_prefix="draft", subset=None, max_iter = 1000000,
                          M_divergence = 1.e3, epsilon_stationnary = 1e-3)
    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="draft") == 1)

    #=========================================================================
    # Plot, colors based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    base_data_prefix = "draft"
    base_data_function = lambda x: np.log(x)
    probe_values = [0., 0.85, 0.998]
    magma = Fractal_colormap((1., 0.1, 200), plt.get_cmap("magma"))
    base_colormap = magma - magma
    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, base_colormap, probe_values,
                              mask=~divergence)

    # shade layer based on field lines
    layer_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.25,
                         blur_ranges=[[0.98, 0.995, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 8.})    
    plotter.plot("draft")


def plot_billiard_game(directory):
    """
    Plots the "Billiard game" image.
    """
    #==========================================================================
    # Parameters
    x = -0.125
    y = 0.90
    dx = 0.6
    xy_ratio = 1.0
    theta_deg = 0.
    nx = 1600
    known_order = 1
    complex_type = np.complex128

    #==========================================================================
    # Calculations
    mandelbrot = Classical_mandelbrot(directory, x, y, dx, nx, xy_ratio,
                                      theta_deg, chunk_size=200,
                                      complex_type=complex_type)
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter=2000000,
                          M_divergence = 1.e3, epsilon_stationnary = 1e-3)

    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.1 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05)

    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=16, max_cycle=40000, eps_cv=1.e-12,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order)

    #==========================================================================
    # Plot the mandelbrot interior
    # Color base on attractivity
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([0, 0, 0]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_emerauld, black, 100,
                                          f=lambda x: x**6)
    colormap_newton = Fractal_colormap(color_gradient)
    base_data_prefix = "newton"
    base_data_key = ("abs", {"source": "attractivity"})
    base_data_function = lambda x: x
    probe_values = [0., 1.]
    plotter_minibrot = Fractal_plotter(mandelbrot, base_data_key,
                                       base_data_prefix, base_data_function,
                                       colormap_newton, probe_values,
                                       calc_layers=[], mask=~cycling)

    # shade layer based on normal vec : 
    # normal = attr * np.conj(dattrdc) / np.abs(dattrdc)
    layern_key2 = ("attr_shade", {"theta_LS": 135.,
                                "phi_LS": 30.,
                                "shininess": 200.,
                                "ratio_specular": 1.})
    plotter_minibrot.add_NB_layer(postproc_key=layern_key2, intensity=0.95,
                           skewness=0.15,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.plot("minibrot", mask_color=(0., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    magma = Fractal_colormap((1.0, 0.0, 200), plt.get_cmap("magma"))
    colormap_div = magma
    base_data_prefix = "loop-1"
    base_data_function = lambda x: np.log(x)
    probe_values = [0., 0.985]
    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_div, probe_values,
                              mask=~divergence)         

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.7,
                         blur_ranges=[[0.85, 0.95, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 

    # shade layer based on normal from DEM 'Milnor'
    layer1_key = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": 135.,
                                "phi_LS": 35.,
                                "shininess": 40.,
                                "ratio_specular": 4.5})
    plotter.add_NB_layer(postproc_key=layer1_key, intensity=0.95, 
                         blur_ranges=[[0.97, 0.99, 1.0]], normalized=False, 
            skewness=-0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    plotter.plot("divergent")


    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")


def plot_medallion(directory):
    """
    Plots the "Medallion" image.
    This example is at the limit of reach for np.float64 however it performs
    well at least up to a resolution of 3200 x 3200 (nx = 3200, xy_ratio = 1.0)
    (I didn't test higher).
    """
    #==========================================================================
    #   Parameters
    nx = 3200
    x = -1.74920463345912691e+00
    y = -2.8684660237361114e-04
    dx = 5.e-12
    xy_ratio = 1.0
    theta_deg = 0.
    known_order = 134 # if don't know use None - speed up the calculation
    complex_type = np.complex128

    #==========================================================================
    #   Calculations
    mandelbrot = Classical_mandelbrot(directory, x, y, dx, nx, xy_ratio,
                                      theta_deg, chunk_size=200,
                                      complex_type=complex_type)
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter=2000000,
                          M_divergence = 1.e3, epsilon_stationnary = 1e-3)

    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 1.0 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05)

    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=16, max_cycle=40000, eps_cv=1.e-15,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order)

    #==========================================================================
    # Plot the minibrot
    # Color base on attractivity
    gold = np.array([255, 210, 66]) / 255.
    light_cyan = np.array([97, 214, 217]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_cyan, gold, 100,
                                          f=lambda x: x**6)   
    colormap_newton = Fractal_colormap(color_gradient)
    base_data_prefix = "newton"
    base_data_key = ("abs", {"source": "attractivity"})
    base_data_function = lambda x: np.where(x > 1., 1., x) # a few err. pix > 1
    probe_values = [0., 1.]
    plotter_minibrot = Fractal_plotter(mandelbrot, base_data_key,
                                       base_data_prefix, base_data_function,
                                       colormap_newton, probe_values,
                                       calc_layers=[], mask=~cycling)

    # shade layer based on normal vec : 
    layern_key2 = ("attr_shade", {"theta_LS": 135.,
                                "phi_LS": 30.,
                                "shininess": 40.,
                                "ratio_specular": 1.})
    plotter_minibrot.add_NB_layer(postproc_key=layern_key2, intensity=0.95,
                           skewness=0.15,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.plot("minibrot", mask_color=(0., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    black = np.array([0, 0, 0]) / 255.
    light_grey = np.array([215, 215, 215]) / 255.
    blue = np.array([3, 76, 160])/255. # blue
    color_gradient1 = Color_tools.Lab_gradient(blue, light_grey, 200)
    color_gradient2 = Color_tools.Lab_gradient(light_grey, gold, 200)
    color_gradient3 = Color_tools.Lch_gradient(gold, black, 50,
                                               f=lambda x: x**2 )
    colormap_div =  Fractal_colormap(color_gradient1) + Fractal_colormap(
                    color_gradient2) + Fractal_colormap(color_gradient3)
    base_data_prefix = "loop-1"
    base_data_function = lambda x: np.log(x)
    probe_values = [0., 0.76, 0.975, 0.998]
    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_div, probe_values,
                              mask=~divergence)         

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.7,
                         blur_ranges=[[0.85, 0.95, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 

    # shade layer based on normal from DEM 'Milnor'
    layer1_key = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": 135.,
                                "phi_LS": 35.,
                                "shininess": 40.,
                                "ratio_specular": 4.5})
    plotter.add_NB_layer(postproc_key=layer1_key, intensity=0.95, 
                         blur_ranges=[[0.98, 1.0, 1.0]], normalized=False, 
            skewness=-0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    plotter.plot("divergent")


    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")


def plot_emerauld_shield(directory):
    """
    Plot the "Emerauld Shield" image
    Note that this example shows the use of 'long doubles' (traditionally
    80 bits precision) and is hence system-dependant.
    You can check your parameters for long-double with:

                > print np.finfo(np.float128)

    For reference on my machine it yields :
    
                Machine parameters for float128
                ---------------------------------------------------------------
                precision= 18   resolution= 1e-18
                machep=   -63   eps=        1.08420217249e-19
                negep =   -64   epsneg=     5.42101086243e-20
                minexp=-16382   tiny=       3.36210314311e-4932
                maxexp= 16384   max=        1.18973149536e+4932
                nexp  =    15   min=        -max
                ---------------------------------------------------------------

    This will raise an error on system where this datatype is unsupported.
    """
    #==========================================================================
    #   Parameters
    nx = 1600
    dx = 1.10625e-13
    x = -0.7658502608035550
    y = -0.09799552764351510
    xy_ratio = 1.0
    theta_deg = 0.
    complex_type = np.complex256
    known_order = 1902. # if don't know use None
    
    #==========================================================================
    #   Calculations
    mandelbrot = Classical_mandelbrot(directory, x, y, dx, nx, xy_ratio, theta_deg,
                                     chunk_size=200, complex_type=complex_type)
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 2000000,
                          M_divergence = 1.e3, epsilon_stationnary = 1e-3)

    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1") 
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.1 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05)

    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=16, max_cycle=40000, eps_cv=1.e-18,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order)

    #==========================================================================
    # Plot the minibrot
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([0, 0, 0]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_emerauld, black,  200,
                                              f= lambda x:x**0.7)
    colormap_newton = Fractal_colormap(color_gradient)

    # color based on cycle order
    base_data_prefix = "newton"
    base_data_key = ("raw", {"code": "r"})
    if known_order is not None:
        base_data_function = lambda x: ((x / known_order) - 1.) % 5
    else:
        base_data_function = lambda x: x % 17
    probe_values = [0., 1.]
    plotter_minibrot = Fractal_plotter(mandelbrot, base_data_key,
                                       base_data_prefix, base_data_function,
                                       colormap_newton, probe_values,
                                       calc_layers=[], mask=~cycling)

    # shade layer based on attractivity phase
    layern_key1 = ("minibrot_phase", {"source": "attractivity"})
    Fourrier = ([0.]*24 + [1], [0., 0.])
    plotter_minibrot.add_NB_layer(postproc_key=layern_key1, intensity=-0.5,
                           skewness=-0.25, hardness=1.8,
                           shade_type={"Lch": 0., "overlay": 0., "pegtop": 1.})

    # shade layer based on normal vec : 
    # normal = attr * np.conj(dattrdc) / np.abs(dattrdc)
    layern_key2 = ("attr_shade", {"theta_LS": -30.,
                                "phi_LS": 70.,
                                "shininess": 250.,
                                "ratio_specular": 4.})
    Fourrier = ([1], [0., 0.])
    plotter_minibrot.add_NB_layer(postproc_key=layern_key2, intensity=0.95,
                           skewness=0.1,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.plot("minibrot", mask_color=(0., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    magma = Fractal_colormap((1.0, 0.0, 200), plt.get_cmap("magma"))
    colormap_div = magma - magma + magma - magma + magma - magma
    base_data_prefix = "loop-1"
    base_data_function = lambda x: np.log(x)
    probe_values = [0., 0.38, 0.62,  0.76, 0.82, 0.985, 0.998]
    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_div, probe_values,
                              mask=~divergence)

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.7,
                         blur_ranges=[[0.40, 0.55, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.})            

    # shade layer based on normal from DEM 'Milnor'
    layer1_key = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": -30.,
                                "phi_LS": 70.,
                                "shininess": 40.,
                                "ratio_specular": 4.5})
    plotter.add_NB_layer(postproc_key=layer1_key, intensity=0.80, 
                         blur_ranges=[[0.70, 0.85, 1.0]], normalized=False, 
            skewness=0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.4,
                         blur_ranges=[[0.40, 0.55, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 
    plotter.plot("divergent")

    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")


if __name__ == "__main__":
    plot()