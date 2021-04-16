# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools, Fractal_Data_array
from classical_mandelbrot import Classical_mandelbrot


def plot():
    """========================================================================
    Run the script to generate 1 of 4 fractal images from the classic
    Mandelbrot set, z(n+1) <- zn**2 + c (c = pixel).
    User-input needed : *choice* and the target *directory*
    ========================================================================"""
    choice = 4

    select = {1: plot_classic_draft,
              2: plot_billiard_game,
              3: plot_medallion,
              4: plot_emerauld_shield,
              5: plot_new,
              6: plot_new2}

    directory = {1: "/home/geoffroy/Pictures/math/classic_blender/test_proj",
                 2: "/media/HDD_1000/Mes_photos/math/fractal/classic/2",
                 3: "/home/geoffroy/Pictures/math/classic/3bis",
                 4: "/home/geoffroy/Pictures/math/classic/4",
                 5: "/media/HDD_1000/Mes_photos/math/fractal/classic/5",
                 6: "/media/HDD_1000/Mes_photos/math/fractal/classic/6"}   

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
    
    nx = 3200
    x = -1.74920463345912691e+00
    y = -2.8684660237361114e-04
    dx = 5.e-12
    xy_ratio = 0.16
    
    
    theta_deg = -140.#params.get("theta_deg", 0.)
#    nx = params.get("nx", 600)
    complex_type = np.complex128

    #==========================================================================
    #  Calculations
    mandelbrot = Classical_mandelbrot(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type, projection="exp_map")
    mandelbrot.draft_loop(file_prefix="draft", subset=None, max_iter = 1000000,
                          M_divergence = 1.e3, epsilon_stationnary = 1e-3)

    divergence = Fractal_Data_array(mandelbrot, file_prefix="draft",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")
    #=========================================================================
    # Plot, colors based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    base_data_prefix = "draft"
    base_data_function = lambda x: np.log(x)
    probe_values = [0., 0.5, 0.85, 0.95, 0.998]
    magma = Fractal_colormap((1., 0.1, 200), plt.get_cmap("magma"))
    base_colormap = - magma + magma - magma + magma
    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, base_colormap, probe_values,
                              mask=~divergence)

    # shade layer based on field lines
    layer_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_grey_layer(postproc_key=layer_key, Fourrier=Fourrier,
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
                                       mask=~cycling)

    # shade layer based on normal vec : 
    # normal = attr * np.conj(dattrdc) / np.abs(dattrdc)
    layern_key2 = ("attr_shade", {"theta_LS": 135.,
                                "phi_LS": 30.,
                                "shininess": 200.,
                                "ratio_specular": 1.})
    plotter_minibrot.add_grey_layer(postproc_key=layern_key2, intensity=0.95,
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
    nx = 1600 * 4
    x = -1.74920463345912691e+00
    y = -2.8684660237361114e-04
    dx = 5.e-11
    xy_ratio = 0.18
    theta_deg = -140.
    known_order = None #134 # if don't know use None - speed up the calculation
    complex_type = np.complex128

    #==========================================================================
    #   Calculations
    mandelbrot = Classical_mandelbrot(directory, x, y, dx, nx, xy_ratio,
                                      theta_deg, chunk_size=200,
                                      complex_type=complex_type, projection="exp_map")
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter=2000000,
                          M_divergence = 1.e3, epsilon_stationnary = 1e-3, pc_threshold=0.05)

    divergence = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")
    stationnary = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 1.0 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05, pc_threshold=0.05)

    cycling = Fractal_Data_array(mandelbrot, file_prefix="r_candidate",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate", pc_threshold=0.05)

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=30, max_cycle=40000, eps_cv=2.e-15,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order, pc_threshold=0.05)

    #==========================================================================
    # Plot the minibrot
    # Color base on attractivity
    gold = np.array([255, 210, 66]) / 255.
    blue_mid = np.array([173, 190, 255]) / 255.
    light_cyan = np.array([62, 249, 255]) / 255.
    green = np.array([62, 249, 200.]) / 255.
    red = np.array([255, 60, 118]) / 255.
    dark_cyan = np.array([39, 119, 120]) / 255.
    color_gradient = Color_tools.Lch_gradient(light_cyan, dark_cyan, 100,
                                          f=lambda x: x**12, long_path=False)   
    colormap_newton = Fractal_colormap(color_gradient)
    base_data_prefix = "newton"
    base_data_key = ("abs", {"source": "attractivity"})
    base_data_function = lambda x: np.where(x > 1., 1., x) # a few err. pix > 1
    probe_values = [0., 1.]
    plotter_minibrot = Fractal_plotter(mandelbrot, base_data_key,
                                       base_data_prefix, base_data_function,
                                       colormap_newton, probe_values,
                                       mask=~cycling)

    # shade layer based on normal vec : 
    layern_key2 = ("attr_shade", {"theta_LS": -75. + 150 + 90,
                                "phi_LS": 45.,
                                "shininess": 400.,
                                "ratio_specular": 4.,
                                "exp_map": True})
    plotter_minibrot.add_grey_layer(postproc_key=layern_key2, intensity=0.95,
                           skewness= -0.,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.plot("minibrot", mask_color=(0., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    black = np.array([0, 0, 0]) / 255.
    dred = np.array([68, 22, 33]) / 255.
    light_grey = np.array([215, 215, 215]) / 255.
    blue = np.array([3, 76, 160])/255. # blue
    color_gradient1 = Color_tools.Lch_gradient(blue, light_grey, 200)
    color_gradient2 = Color_tools.Lch_gradient(light_grey, gold, 200)
    color_gradient3 = Color_tools.Lch_gradient(gold, dred, 50,
                                               f=lambda x: x**2 )
    
    copper = Fractal_colormap((1., 0.1, 100), plt.get_cmap("Blues"))
    colormap_div =   Fractal_colormap(color_gradient1) + Fractal_colormap(
                    color_gradient2) + Fractal_colormap(color_gradient3)
    colormap_div = copper - copper + copper 
    colormap_div.extent = "mirror"
    base_data_prefix = "loop-1"
    base_data_function = lambda x: np.log(x)
    probe_values = [0., 0.76, 0.975, 0.998]
    probe_values = [6.23353338241577, 6.50293933477195, 7.0, 8.0]
    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_div, probe_values,
                              probes_kind="z", mask=~divergence)       

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_grey_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.7,
                         blur_ranges=[[6.57, 6.81, 20.]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 

    # shade layer based on normal from DEM 'Milnor'
    layer1_key = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": -75. + 150 + 90,
                                "phi_LS": 35.,
                                "shininess": 40.,
                                "ratio_specular": 400.,
                                "exp_map": True})
    plotter.add_grey_layer(postproc_key=layer1_key, intensity=0.95, 
                         blur_ranges=[[7.0, 14., 20.]], normalized=False, 
            skewness=0., shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

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
    nx = 600
    dx = 1.10625e-13
    x = -0.7658502608035550
    y = -0.09799552764351510
    xy_ratio = 1.0
    theta_deg = 0.
    complex_type = np.complex128#256
    known_order = 1902. # if don't know use None

    import mpmath
    mpmath.mp.dps = 20     # higher precision for demonstration
    x = mpmath.mpf("-0.7658502608035550")
    y = mpmath.mpf("-0.09799552764351510")

    
    #==========================================================================
    #   Calculations
    mandelbrot = Classical_mandelbrot(directory, x, y, dx, nx, xy_ratio, theta_deg,
                                     chunk_size=200, complex_type=complex_type)
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 200000,
                          M_divergence = 1.e3, epsilon_stationnary = 1e-3)

    divergence = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")
    stationnary = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.1 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=200000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05)

    cycling = Fractal_Data_array(mandelbrot, file_prefix="r_candidate",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=200000, M_divergence = 1.e3,
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
                                       mask=~cycling)

    # shade layer based on attractivity phase
    layern_key1 = ("minibrot_phase", {"source": "attractivity"})
    Fourrier = ([0.]*24 + [1], [0., 0.])
    plotter_minibrot.add_grey_layer(postproc_key=layern_key1, intensity=-0.5,
                           skewness=-0.25, hardness=1.8,
                           shade_type={"Lch": 0., "overlay": 0., "pegtop": 1.})

    # shade layer based on normal vec : 
    # normal = attr * np.conj(dattrdc) / np.abs(dattrdc)
    layern_key2 = ("attr_shade", {"theta_LS": -30.,
                                "phi_LS": 70.,
                                "shininess": 250.,
                                "ratio_specular": 4.})
    Fourrier = ([1], [0., 0.])
    plotter_minibrot.add_grey_layer(postproc_key=layern_key2, intensity=0.95,
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
    plotter.add_grey_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.7,
                         blur_ranges=[[0.40, 0.55, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.})            

    # shade layer based on normal from DEM 'Milnor'
    layer1_key = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": -30.,
                                "phi_LS": 70.,
                                "shininess": 40.,
                                "ratio_specular": 4.5})
    plotter.add_grey_layer(postproc_key=layer1_key, intensity=0.80, 
                         blur_ranges=[[0.70, 0.85, 1.0]], normalized=False, 
            skewness=0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_grey_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.4,
                         blur_ranges=[[0.40, 0.55, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 
    plotter.plot("divergent")

    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")







def plot_new(directory):
    """
    Plot the 5" image
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
    nx = 4000
    dx = 2.0e-12
    x = -1.3555907426729 + 24./800. * 1e-12
    y = 0.068410397524657
    xy_ratio = 1.0
    theta_deg = 0.
    complex_type = np.complex128
    known_order = 444#. # if don't know use None
    
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
    eps_pixel = 0.5 * dx / float(nx)
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
                             max_newton=16, max_cycle=40000, eps_cv=1.e-14,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order)

    #==========================================================================
    # Plot the minibrot
#    light_emerauld = np.array([15, 230, 186]) / 255.
#    black = np.array([0, 0, 0]) / 255.
#    color_gradient = Color_tools.Lab_gradient(light_emerauld, black,  200,
#                                              f= lambda x:x**0.7)
    light_emerauld = np.array([15, 230, 186]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    flash_rubis = np.array([255., 0., 121.]) / 255.
    flash_navy = np.array([0., 64., 255.]) / 255.
    flash_yellow = np.array([249., 255., 148.]) / 255.
    black = np.array([0, 0, 0]) / 255.
    silver = np.array([200, 220, 220])/255. 

    color_gradient = Color_tools.Lch_gradient(flash_rubis, black, 100,
                                          f=lambda x: 1 - x**2)   
    colormap_newton = Fractal_colormap(color_gradient)

    # color based on cycle order
    base_data_prefix = "newton"
    base_data_key =  ("abs", {"source": "attractivity"})#{"raw", }
    base_data_function = lambda x: np.where(x < 1., x, 1.)
#    if known_order is not None:
#        base_data_function = lambda x: ((x / known_order) - 1.) % 5
#    else:
#        base_data_function = lambda x: x % 17
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
    layern_key2 = ("attr_shade", {"theta_LS": 45.,
                                "phi_LS": 15.,
                                "shininess": 100.,
                                "ratio_specular": 1.})
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
    magma = -Fractal_colormap((1.0, 0.0, 200), plt.get_cmap("Blues"))
    viridis = -Fractal_colormap((1.0, 0.0, 200), plt.get_cmap("inferno"))
    colormap_div = magma - magma + magma - magma + magma - magma
    colormap_div = magma - magma + magma - magma + magma - magma + magma - magma
    base_data_prefix = "loop-1"
    base_data_function = lambda x: np.log(x)
    probe_values = [0., 0.3, 0.76,  0.86, 0.92, 0.96, 0.98, 0.992, 0.999]
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
                                "theta_LS": 45.,
                                "phi_LS": 15.,
                                "shininess": 40.,
                                "ratio_specular": 4.5})
    plotter.add_NB_layer(postproc_key=layer1_key, intensity=0.80, 
                         blur_ranges=[[0.99, 0.999, 1.0]], normalized=False, 
            skewness=0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.4,
                         blur_ranges=[[0.70, 0.85, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 
    plotter.plot("divergent")

    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")



def plot_new2(directory):
    """
    Plot the 5" image
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
    nx = 3600 
    dx = 2.375e-14 * 0.36 * 0.2
    x = -0.745822157832475 - 1./80. * 2.375e-14 - 35./3600. * 2.375e-14 * 0.36
    y = 0.109889508817093 - 5./800*2.375e-13 - 1./80. * 2.375e-14 - 25./3600. * 2.375e-14 * 0.36
    
#-0,745822157832475
#0,109889508817093
#2,375000E-13

    
    xy_ratio = 1.0
    theta_deg = 0.
    complex_type = np.complex256
    known_order = 1997#. # if don't know use None
    
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
    eps_pixel = 0.5 * dx / float(nx)
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
                             max_newton=16, max_cycle=40000, eps_cv=1.e-17,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order)

    #==========================================================================
    # Plot the minibrot
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([0, 0, 0]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_emerauld, black,  200,
                                              f= lambda x:x**0.7)
    light_emerauld = np.array([15, 230, 186]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    flash_rubis = np.array([255., 0., 121.]) / 255.
    flash_navy = np.array([0., 64., 255.]) / 255.
    flash_yellow = np.array([249., 255., 148.]) / 255.
    black = np.array([0, 0, 0]) / 255.
    silver = np.array([200, 220, 220])/255. 
    paon_green = np.array([82, 233, 104]) / 255.

    color_gradient = Color_tools.Lch_gradient(black, paon_green, 100,
                                          f=lambda x: 1 - x**3)   
    colormap_newton = Fractal_colormap(color_gradient)

    # color based on cycle order
    base_data_prefix = "newton"
    base_data_key =  ("abs", {"source": "attractivity"})#{"raw", }
    base_data_function = lambda x: np.where(x < 1., x, 1.)
#    if known_order is not None:
#        base_data_function = lambda x: ((x / known_order) - 1.) % 5
#    else:
#        base_data_function = lambda x: x % 17
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
    layern_key2 = ("attr_shade", {"theta_LS": 45.,
                                "phi_LS": 15.,
                                "shininess": 100.,
                                "ratio_specular": 1.})
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
    magma = -Fractal_colormap((1.0, 0.0, 100), plt.get_cmap("magma"))
    inferno = -Fractal_colormap((1.0, 0.0, 100), plt.get_cmap("inferno"))
    viridis = -Fractal_colormap((1.0, 0.0, 100), plt.get_cmap("copper"))
#    colormap_div = magma - magma + magma - magma + magma - magma
#    colormap_div =  - magma + magma - magma + magma - magma + magma - magma + magma - magma + magma

    
    gold = np.array([255, 210, 66]) / 255.
    light_yellow = np.array([255, 252, 199]) / 255.
    purple = np.array([181, 40, 99]) / 255.
    dark_blue = np.array([32, 52, 164]) / 255.
    green_cyan = np.array([71, 206, 176]) / 255.
    black = np.array([0, 0, 0]) / 255.
    grey = np.array([66, 0, 31]) / 255.
    grey80 = np.array([80, 80, 80]) / 255.
    paon_green = np.array([82, 233, 104]) / 255.
    cyan = np.array([52, 218, 208]) / 255.
    royalblue = np.array([65, 105, 225]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    deepskyblue = np.array([0, 191, 255]) / 255.
#    
    color_gradient1 = Color_tools.Lch_gradient(deepskyblue, royalblue, 40)
    color_gradient2 = Color_tools.Lch_gradient(royalblue, blueviolet, 40)
    color_gradient3 = Color_tools.Lch_gradient(blueviolet, flash_yellow, 40) 
    color_gradient4 = Color_tools.Lch_gradient(flash_yellow, blueviolet, 40)#, long_path=True) 
    if True:
        color_gradient2 = Color_tools.Lch_gradient(royalblue, deepskyblue, 40)
        color_gradient3 = Color_tools.Lch_gradient(deepskyblue, royalblue, 40) 
        color_gradient4 = Color_tools.Lch_gradient(royalblue, blueviolet, 40)#, long_path=True)     
    
    color_gradient5 = Color_tools.Lch_gradient(blueviolet, flash_rubis, 40) 
    color_gradient6 = Color_tools.Lch_gradient(flash_rubis, flash_yellow, 40)
    color_gradient7 = Color_tools.Lch_gradient(flash_yellow, flash_rubis, 40)
    color_gradient8 = Color_tools.Lab_gradient(flash_rubis, black, 40)
    color_gradient9 = Color_tools.Lab_gradient(black, grey80, 40)
    color_gradient10 = Color_tools.Lab_gradient(grey80, black, 40)
    color_gradient11 = Color_tools.Lab_gradient(black, flash_rubis, 40)
    color_gradient12 = Color_tools.Lab_gradient(flash_rubis, flash_yellow, 40)
    colormap_div = (Fractal_colormap(color_gradient1) +
                    Fractal_colormap(color_gradient2) +
                    Fractal_colormap(color_gradient3) +
                    Fractal_colormap(color_gradient4) +
                    Fractal_colormap(color_gradient5) +
                    Fractal_colormap(color_gradient6) +
                    Fractal_colormap(color_gradient7) +
                    Fractal_colormap(color_gradient8) +
                    Fractal_colormap(color_gradient9) +
                    Fractal_colormap(color_gradient10) +
                    Fractal_colormap(color_gradient11) +
                    Fractal_colormap(color_gradient12))
    
#    cmap1 = Fractal_colormap((1.0, 0.0, 50), plt.get_cmap("YlOrRd"))
#    cmap2 = -Fractal_colormap((1.0, 0.0, 50), plt.get_cmap("PuRd"))
#    cmap3 = Fractal_colormap((1.0, 0.0, 50), plt.get_cmap("Reds"))
#    cmap4 = -Fractal_colormap((1.0, 0.0, 50), plt.get_cmap("RdPu"))
#    cmap5 = Fractal_colormap((1.0, 0.0, 50), plt.get_cmap("BuPu"))
#    cmap6 = Fractal_colormap((1.0, 0.0, 50), plt.get_cmap("buPu"))
#    cmap7 = Fractal_colormap((1.0, 0.0, 50), plt.get_cmap("buPu"))
#    
#    color_gradient2 = Color_tools.Lch_gradient(black, silver, 10)  
#    color_gradient3 = Color_tools.Lch_gradient(black, silver, 10)  
#    color_gradient4 = Color_tools.Lch_gradient(black, silver, 10)  
#    color_gradient5 = Color_tools.Lch_gradient(black, silver, 10)  
#    color_gradient6 = Color_tools.Lch_gradient(black, silver, 10)  
#    color_gradient7 = Color_tools.Lch_gradient(black, silver, 10)  
#    color_gradient8 = Color_tools.Lch_gradient(black, silver, 10)  
#    color_gradient9 = Color_tools.Lch_gradient(black, silver, 10)    
#    color_gradient10 = Color_tools.Lch_gradient(black, silver, 10)    
    
    
#    colormap_div =  viridis - viridis + magma - magma + inferno - inferno + magma - magma + inferno - inferno
#    colormap_div =  viridis - viridis - magma + magma + inferno - inferno + magma - magma + inferno - inferno
    base_data_prefix = "loop-1"
    base_data_function = lambda x: np.log(x)
    probe_values = [0., 0.32, 0.55, 0.76,  0.86, 0.92, 0.96, 0.98, 0.992, 0.995, 0.998, 0.999, 0.9999]
    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_div, probe_values,
                              mask=~divergence)

    # shade layer based on field lines
#    layer2_key = ("field_lines", {})
#    Fourrier = ([1.,], [0, 0.,])
#    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
#                         hardness=1.5, intensity=0.7,
#                         blur_ranges=[[0.40, 0.55, 1.0]], 
#                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.})            

    # shade layer based on normal from DEM 'Milnor'
    layer1_key = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": 45.,
                                "phi_LS": 15.,
                                "shininess": 140.,
                                "ratio_specular": 4.5})
    plotter.add_NB_layer(postproc_key=layer1_key, intensity=0.95, 
                         blur_ranges=[[0.85, 0.95, 1.0]], normalized=False, 
            skewness=0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.4,
                         blur_ranges=[[0.3, 0.4, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 
    plotter.plot("divergent")

    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")





if __name__ == "__main__":
    plot()