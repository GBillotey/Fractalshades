# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools
from mandelbrotN import mandelbrotN


def plot():
    """========================================================================
    Run the script to generate 1 of 4 fractal images from the classic
    Mandelbrot set, z(n+1) <- zn**2 + c (c = pixel).
    User-input needed : *choice* and the target *directory*
    ========================================================================"""
    choice = 2

    select = {0: plot_test,
              1: plot_new2,
              2: plot_new3,
              3: plot_new4}

    directory = {0: "/home/gby/Pictures/Mes_photos/math/fractal/pN/test",
                 1: "/home/gby/Pictures/Mes_photos/math/fractal/pN/1",
                 2: "/home/gby/Pictures/Mes_photos/math/fractal/pN/4",
                 3: "/home/gby/Pictures/Mes_photos/math/fractal/pN/5"}

    # run the selected plot
    select[choice](directory[choice])


def plot_N_draft(directory, **params):
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
    mandelbrot = mandelbrotN(
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



def plot_test(directory):
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
    nx = 1800 
    dx = 1.e-13
    x = 0.64734298336077
    y = 0.49631348266355
    #dx = 5.
    x = 0.647342983360766
    y = 0.496313482663546
    dx = 5.


#0,647342983361
#0,496313482664
#2,083333E-14

#-0,745822157832475
#0,109889508817093
#2,375000E-13

    
    xy_ratio = 1.0
    theta_deg = 0.
    complex_type = np.complex128
    known_order = None # if don't know use None # newton_600-800_2400-2600.tmp # 4590
    min_order = 0

    #==========================================================================
    #   Calculations
    mandelbrot = mandelbrotN(directory, x, y, dx, nx, xy_ratio, theta_deg,
                                     chunk_size=200, complex_type=complex_type)
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 2000000,
                          M_divergence = 1.e1, epsilon_stationnary = 1e-3)

    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1") 
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.5 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e1,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.4, pc_threshold=0.5)

    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e1,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=10, max_cycle=80000, eps_cv=1.e-13, # newton_400-600_1000-1200.tmp
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order,
                             min_order=min_order)

    #==========================================================================
    # Plot the minibrot
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([0, 0, 0]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_emerauld, black,  200,
                                              f= lambda x:x**0.7)
    light_emerauld = np.array([15, 230, 186]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    grey200 = np.array([200, 200, 200]) / 255.
    grey150 = np.array([150, 150, 150]) / 255.
    grey80 = np.array([80, 80, 80]) / 255.
    flash_rubis = np.array([255., 0., 121.]) / 255.
    pink = np.array([255., 0., 164.]) / 255.
    flash_navy = np.array([0., 64., 255.]) / 255.
    flash_blue = np.array([176,224,230]) / 255.
    flash_yellow = np.array([249., 255., 148.]) / 255.
    black = np.array([0, 0, 0]) / 255.
    silver = np.array([200, 220, 220])/255. 
    paon_green = np.array([82, 233, 104]) / 255.
    garden_green = np.array([166, 255, 72]) / 255.
    copper_rgb = np.array([255, 123, 94]) / 255. 

    color_gradient = Color_tools.Lch_gradient(black, garden_green, 100,
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
    layern_key2 = ("power_attr_shade", {"theta_LS": 45.,
                                "phi_LS": 75.,
                                "shininess": 200.,
                                "ratio_specular": 10.})
    Fourrier = ([1], [0., 0.])
    plotter_minibrot.add_NB_layer(postproc_key=layern_key2, intensity=0.999,
                           skewness=0.1,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.plot("minibrot", mask_color=(0., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 6, "a_d": 1., "N": 1e1})
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
    paon_green = np.array([82, 233, 104]) / 255.
    cyan = np.array([52, 218, 208]) / 255.
    royalblue = np.array([65, 105, 225]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    deepskyblue = np.array([0, 191, 255]) / 255.
#    
    color_gradient1 = Color_tools.Lch_gradient(dark_blue, royalblue, 40)
    color_gradient2 = Color_tools.Lch_gradient(royalblue, dark_blue, 40)
    color_gradient3 = Color_tools.Lab_gradient(dark_blue, gold, 40) #, long_path=True)
    color_gradient4 = Color_tools.Lab_gradient(gold, blueviolet, 40)#, long_path=True) 
    if False:
        color_gradient2 = Color_tools.Lch_gradient(royalblue, deepskyblue, 40)
        color_gradient3 = Color_tools.Lch_gradient(deepskyblue, royalblue, 40) 
        color_gradient4 = Color_tools.Lch_gradient(royalblue, blueviolet, 40)#, long_path=True)     
    
    color_gradient5 = Color_tools.Lch_gradient(blueviolet, black, 40) 
    color_gradient6 = Color_tools.Lab_gradient(black, flash_yellow, 40)
    color_gradient7 = Color_tools.Lch_gradient(flash_yellow, flash_rubis, 40)
    color_gradient8 = Color_tools.Lab_gradient(flash_rubis, blueviolet, 40)
    color_gradient9 = Color_tools.Lab_gradient(blueviolet, black, 40)
    color_gradient10 = Color_tools.Lab_gradient(black, blueviolet, 40)
    color_gradient11 = Color_tools.Lab_gradient(blueviolet, flash_rubis, 40)
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
    probe_values = [0., 0.32, 0.55, 0.76,  0.88, 0.94, 0.97, 0.98, 0.99, 0.993, 0.996, 0.999, 0.9999]
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
                                "phi_LS": 45.,
                                "shininess": 400.,
                                "ratio_specular": 2.5})
    plotter.add_NB_layer(postproc_key=layer1_key, intensity=0.95, 
                         blur_ranges=[[0.95, 0.99, 1.0]], normalized=False, 
            skewness=0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.4,
                         blur_ranges=[[0.8, 0.95, 1.0]], 
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
    dx = 1.e-13
    x = 0.64734298336077
    y = 0.49631348266355
    #dx = 5.
    x = 0.647342983360766
    y = 0.496313482663546
    dx = 2.083333E-14


#0,647342983361
#0,496313482664
#2,083333E-14

#-0,745822157832475
#0,109889508817093
#2,375000E-13

    
    xy_ratio = 1.0
    theta_deg = 0.
    complex_type = np.complex256
    known_order = 5 * 17# if don't know use None # newton_600-800_2400-2600.tmp # 4590
    min_order = 4000
    # order spotted  867 ?? -> 3 * 17 * 17  # 255 - 289 - 867
    #               1445 ?? -> 5 x 17**2
    # 4335 ????
    # CONFIRMED 27356 With cycle order: 4335 ! sans test < 1
    #==========================================================================
    #   Calculations
    mandelbrot = mandelbrotN(directory, x, y, dx, nx, xy_ratio, theta_deg,
                                     chunk_size=200, complex_type=complex_type)
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 2000000,
                          M_divergence = 1.e1, epsilon_stationnary = 1e-3)

    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1") 
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 5. * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e1,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.4, pc_threshold=0.5)

    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e1,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=36, max_cycle=80000, eps_cv=1.e-17, # newton_400-600_1000-1200.tmp
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order,
                             min_order=min_order)

    #==========================================================================
    # Plot the minibrot
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([0, 0, 0]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_emerauld, black,  200,
                                              f= lambda x:x**0.7)
    light_emerauld = np.array([15, 230, 186]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    grey200 = np.array([200, 200, 200]) / 255.
    grey150 = np.array([150, 150, 150]) / 255.
    grey80 = np.array([80, 80, 80]) / 255.
    flash_rubis = np.array([255., 0., 121.]) / 255.
    pink = np.array([255., 0., 164.]) / 255.
    flash_navy = np.array([0., 64., 255.]) / 255.
    flash_blue = np.array([176,224,230]) / 255.
    flash_yellow = np.array([249., 255., 148.]) / 255.
    black = np.array([0, 0, 0]) / 255.
    silver = np.array([200, 220, 220])/255. 
    paon_green = np.array([82, 233, 104]) / 255.
    garden_green = np.array([166, 255, 72]) / 255.
    copper_rgb = np.array([255, 123, 94]) / 255. 

    color_gradient = Color_tools.Lch_gradient(black, light_emerauld, 100,
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
    layern_key2 = ("power_attr_shade", {"theta_LS": 45.,
                                "phi_LS": 75.,
                                "shininess": 200.,
                                "ratio_specular": 10.})
    Fourrier = ([1], [0., 0.])
    plotter_minibrot.add_NB_layer(postproc_key=layern_key2, intensity=0.999,
                           skewness=0.1,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.plot("minibrot", mask_color=(0., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 6, "a_d": 1., "N": 1e1})
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
    paon_green = np.array([82, 233, 104]) / 255.
    cyan = np.array([52, 218, 208]) / 255.
    royalblue = np.array([65, 105, 225]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    deepskyblue = np.array([0, 191, 255]) / 255.
#    
    color_gradient1 = Color_tools.Lch_gradient(royalblue, dark_blue, 40)
    color_gradient2 = Color_tools.Lch_gradient(dark_blue, blueviolet, 40)
    color_gradient3 = Color_tools.Lab_gradient(blueviolet, gold, 40) #, long_path=True)
    color_gradient4 = Color_tools.Lab_gradient(gold, blueviolet, 40)#, long_path=True) 
    if False:
        color_gradient2 = Color_tools.Lch_gradient(royalblue, deepskyblue, 40)
        color_gradient3 = Color_tools.Lch_gradient(deepskyblue, royalblue, 40) 
        color_gradient4 = Color_tools.Lch_gradient(royalblue, blueviolet, 40)#, long_path=True)     
    
    color_gradient5 = Color_tools.Lch_gradient(blueviolet, black, 40) 
    color_gradient6 = Color_tools.Lab_gradient(black, flash_yellow, 40)
    color_gradient7 = Color_tools.Lch_gradient(flash_yellow, flash_rubis, 40)
    color_gradient8 = Color_tools.Lab_gradient(flash_rubis, blueviolet, 40)
    color_gradient9 = Color_tools.Lab_gradient(blueviolet, black, 40)
    color_gradient10 = Color_tools.Lab_gradient(black, blueviolet, 40)
    color_gradient11 = Color_tools.Lab_gradient(blueviolet, flash_rubis, 40)
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
    probe_values = [0., 0.32, 0.55, 0.76,  0.88, 0.94, 0.97, 0.98, 0.99, 0.993, 0.996, 0.999, 0.9999]
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
                                "phi_LS": 45.,
                                "shininess": 400.,
                                "ratio_specular": 2.5})
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


def plot_new3(directory):
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
    nx = 3800
    dx = 1.e-13
    x = 0.64734298336077
    y = 0.49631348266355
    #dx = 5.
    x = 0.91507582974878800
    y = -0.66484121060400200
    dx = 2.e-14

    x = 0.9150758297487877
    y = -0.6648412106040023
    dx = 1.5e-15
    
    x = 1.00642730240118000 + 35. / 600. * 6e-14
    y = 0.39404799133561200 - 15. / 600. * 6e-14
    dx = 6e-14
#1,00642730240118000
#0,39404799133561200
#5,916667E-14



#0,647342983361
#0,496313482664
#2,083333E-14

#-0,745822157832475
#0,109889508817093
#2,375000E-13

    
    xy_ratio = 1.0
    theta_deg = 0.
    complex_type = np.complex256
    known_order = None# if don't know use None # newton_600-800_2400-2600.tmp # 4590
    min_order = 1
    d = 9
    # order spotted  867 ?? -> 3 * 17 * 17  # 255 - 289 - 867
    #               1445 ?? -> 5 x 17**2
    # 4335 ????
    # CONFIRMED 27356 With cycle order: 4335 ! sans test < 1
    #==========================================================================
    #   Calculations
    mandelbrot = mandelbrotN(directory, x, y, dx, nx, xy_ratio, theta_deg,
                                     chunk_size=200, complex_type=complex_type)
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 2000000,
                          M_divergence = 1.e1, epsilon_stationnary = 1e-3, d=d)

    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1") 
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 5. * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e1,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.4, pc_threshold=0.5, d=d)

    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e1,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate", d=d)

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=36, max_cycle=80000, eps_cv=1.e-18, # newton_400-600_1000-1200.tmp
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order,
                             min_order=min_order, d=d)

    #==========================================================================
    # Plot the minibrot
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([0, 0, 0]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_emerauld, black,  200,
                                              f= lambda x:x**0.5)
    light_emerauld = np.array([15, 230, 186]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    grey200 = np.array([200, 200, 200]) / 255.
    grey150 = np.array([150, 150, 150]) / 255.
    grey80 = np.array([80, 80, 80]) / 255.
    grey40 = np.array([40, 40, 40]) / 255.
    flash_rubis = np.array([255., 0., 121.]) / 255.
    pink = np.array([255., 0., 164.]) / 255.
    pink2 = np.array([255., 128., 210.]) / 255.
    flash_navy = np.array([0., 64., 255.]) / 255.
    flash_blue = np.array([176,224,230]) / 255.
    flash_yellow = np.array([249., 255., 148.]) / 255.
    black = np.array([0, 0, 0]) / 255.
    silver = np.array([200, 220, 220])/255. 
    paon_green = np.array([82, 233, 104]) / 255.
    garden_green = np.array([166, 255, 72]) / 255.
    copper_rgb = np.array([255, 123, 94]) / 255. 
    deepskyblue = np.array([0, 191, 255]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    purple = np.array([181, 40, 99]) / 255. 

    color_gradient = Color_tools.Lch_gradient(black, purple, 100,
                                          f=lambda x: 1 - x**0.9)   
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
    layern_key2 = ("power_attr_shade", {"theta_LS": 45.,
                                "phi_LS": 75.,
                                "shininess": 200.,
                                "ratio_specular": 10.})
    Fourrier = ([1], [0., 0.])
    plotter_minibrot.add_NB_layer(postproc_key=layern_key2, intensity=0.999,
                           skewness=0.1,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.plot("minibrot", mask_color=(0., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": d, "a_d": 1., "N": 1e1})
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
    light_cyan = np.array([190, 255, 243]) / 255.
    black = np.array([0, 0, 0]) / 255.
    grey = np.array([66, 0, 31]) / 255.
    paon_green = np.array([82, 233, 104]) / 255.
    cyan = np.array([52, 218, 208]) / 255.
    cyan2 = np.array([96, 215, 225]) / 255.
    royalblue = np.array([65, 105, 225]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    taupe_dark = np.array([92, 53, 102]) / 255.
    taupe = np.array([207, 120, 188]) / 255.
    brown = np.array([137, 68, 71]) / 255.
    deepskyblue = np.array([0, 191, 255]) / 255.
#    
    color_gradient1 = Color_tools.Lch_gradient(silver, silver, 40)
    color_gradient2 = Color_tools.Lch_gradient(blueviolet, blueviolet, 40)
    color_gradient3 = Color_tools.Lch_gradient(blueviolet, taupe, 40) 
    color_gradient4 = Color_tools.Lch_gradient(taupe, blueviolet, 40)#, long_path=True) 
    if True:
        #color_gradient1 = Color_tools.Lch_gradient(taupe_dark, taupe_dark, 40)
        color_gradient1 = Color_tools.Lch_gradient(taupe_dark, taupe_dark, 40)
        color_gradient2 = Color_tools.Lab_gradient(taupe_dark, light_emerauld, 40) 
        color_gradient3 = Color_tools.Lch_gradient(light_emerauld, dark_blue, 40)#, long_path=True)  
        color_gradient4 = Color_tools.Lch_gradient(dark_blue, blueviolet, 40)#, long_path=True)    
    
    color_gradient5 = Color_tools.Lch_gradient(blueviolet, dark_blue, 40) 
    color_gradient6 = Color_tools.Lch_gradient(dark_blue, deepskyblue, 40)
    color_gradient7 = Color_tools.Lch_gradient(deepskyblue, paon_green, 40)
    color_gradient8 = Color_tools.Lch_gradient(paon_green, deepskyblue, 40)
    color_gradient9 = Color_tools.Lch_gradient(deepskyblue, taupe_dark, 40)
    color_gradient10 = Color_tools.Lch_gradient(taupe_dark, deepskyblue, 40)
    color_gradient11 = Color_tools.Lch_gradient(deepskyblue, taupe_dark, 40)
    color_gradient12 = Color_tools.Lab_gradient(taupe_dark, flash_blue, 40)
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
                    
    blue = True
    if blue:
        gold = np.array([255, 210, 66]) / 255.
        purple = np.array([181, 40, 99]) / 255.       
        dark_blue = np.array([32, 52, 164]) / 255.
        color_gradient1 = Color_tools.Lch_gradient(royalblue, light_yellow, 100,
                    f=lambda x: x**0.7,
                   long_path=False)
        color_gradient12 = Color_tools.Lch_gradient(royalblue, light_yellow, 100,
                    long_path=False)
        color_gradient2 = Color_tools.Lab_gradient(gold, purple, 40)
        
        blues = Fractal_colormap((0.0, 1.0, 40), plt.get_cmap("magma"))
        oranges = Fractal_colormap((0.0, 1.0, 40), plt.get_cmap("inferno"))
        
        blues = Fractal_colormap(color_gradient1)
        blues2 = Fractal_colormap(color_gradient12)
        oranges = Fractal_colormap(color_gradient2)
        colormap_div = (blues - 
                        blues +
                        blues - 
                        blues +
                        blues - 
                        blues2 +
                        oranges - 
                        oranges +
                        oranges - 
                        oranges +
                        oranges - 
                        oranges)
    
    

    
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
    probe_values = [0., 0.35, 0.45, 0.67, 0.78,  0.88, 0.94, 0.97, 0.98, 0.99, 0.993, 0.996, 0.999]
    #probe_values = [0., 0.15, 0.32, 0.55, 0.76,  0.88, 0.94, 0.97, 0.98, 0.99, 0.993, 0.996, 0.999]
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
                                "phi_LS": 45.,
                                "inverse_n": False,  #nx * 0.5, nx * 0.5),
                                "shininess": 400.,
                                "ratio_specular": 2.5})
    plotter.add_NB_layer(postproc_key=layer1_key, intensity=0.9, 
                         blur_ranges=[[0.1, 0.15, 0.2], [0.40, 0.35, 0.15], [0.55, 0.60, 0.65], [0.75, 0.70, 0.65], [0.92, 0.95, 1.0]], #0.82, 0.85, 1.0]
                         normalized=False, 
            skewness=0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.55,
                         blur_ranges=[[0.9, 0.95, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 
    plotter.plot("divergent")

    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")



def plot_new4(directory):
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
    nx = 3800
    dx = 1.e-13
    x = 0.64734298336077
    y = 0.49631348266355
    #dx = 5.
    x = 0.91507582974878800
    y = -0.66484121060400200
    dx = 2.e-14

    x = 0.9150758297487877
    y = -0.6648412106040023
    dx = 1.5e-15
    
    x = 0.76665366600585
    y = 0.35322451025056
    dx = 1.e-13
    x += dx * (335.5 / 600. - 0.5)
    y -= dx * (278. / 600. - 0.5)
    dx = 5.e-15
#1,00642730240118000
#0,39404799133561200
#5,916667E-14



#0,647342983361
#0,496313482664
#2,083333E-14

#-0,745822157832475
#0,109889508817093
#2,375000E-13

    
    xy_ratio = 1.0
    theta_deg = 0.
    complex_type = np.complex256
    known_order = None# if don't know use None # newton_600-800_2400-2600.tmp # 4590
    min_order = 1
    d = 9
    # order spotted  867 ?? -> 3 * 17 * 17  # 255 - 289 - 867
    #               1445 ?? -> 5 x 17**2
    # 4335 ????
    # CONFIRMED 27356 With cycle order: 4335 ! sans test < 1
    #==========================================================================
    #   Calculations
    mandelbrot = mandelbrotN(directory, x, y, dx, nx, xy_ratio, theta_deg,
                                     chunk_size=200, complex_type=complex_type)
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 2000000,
                          M_divergence = 1.e1, epsilon_stationnary = 1e-3, d=d)

    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1") 
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 5. * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e1,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.4, pc_threshold=0.5, d=d)

    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e1,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate", d=d)

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=36, max_cycle=80000, eps_cv=1.e-18, # newton_400-600_1000-1200.tmp
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order,
                             min_order=min_order, d=d)

    #==========================================================================
    # Plot the minibrot
    light_emerauld = np.array([15, 230, 186]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    grey200 = np.array([200, 200, 200]) / 255.
    grey150 = np.array([150, 150, 150]) / 255.
    grey80 = np.array([80, 80, 80]) / 255.
    flash_rubis = np.array([255., 0., 121.]) / 255.
    pink = np.array([255., 0., 164.]) / 255.
    flash_navy = np.array([0., 64., 255.]) / 255.
    flash_blue = np.array([176,224,230]) / 255.
    flash_yellow = np.array([249., 255., 148.]) / 255.
    black = np.array([0, 0, 0]) / 255.
    silver = np.array([200, 220, 220])/255. 
    paon_green = np.array([82, 233, 104]) / 255.
    garden_green = np.array([166, 255, 72]) / 255.
    copper_rgb = np.array([255, 123, 94]) / 255. 

    color_gradient = Color_tools.Lch_gradient(black, flash_rubis, 100,
                                          f=lambda x: x**1.5)   
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
    layern_key2 = ("power_attr_shade", {"theta_LS": 45.,
                                "phi_LS": 85.,
                                "shininess": 200.,
                                "ratio_specular": 10.})
    Fourrier = ([1], [0., 0.])
    plotter_minibrot.add_NB_layer(postproc_key=layern_key2, intensity=0.5,
                           skewness=-0.1,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.plot("minibrot", mask_color=(0., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": d, "a_d": 1., "N": 1e1})
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
    light_cyan = np.array([190, 255, 243]) / 255.
    black = np.array([0, 0, 0]) / 255.
    grey = np.array([66, 0, 31]) / 255.
    paon_green = np.array([82, 233, 104]) / 255.
    cyan = np.array([52, 218, 208]) / 255.
    royalblue = np.array([65, 105, 225]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    deepskyblue = np.array([0, 191, 255]) / 255.
#    
    color_gradient1 = Color_tools.Lch_gradient(black, black, 40)
    color_gradient2 = Color_tools.Lch_gradient(black, blueviolet, 40)
    color_gradient3 = Color_tools.Lch_gradient(blueviolet, flash_yellow, 40) 
    color_gradient4 = Color_tools.Lch_gradient(flash_yellow, blueviolet, 40)#, long_path=True) 
    if True:
        color_gradient2 = Color_tools.Lch_gradient(black, black, 40)
        color_gradient3 = Color_tools.Lch_gradient(black, black, 40) 
        color_gradient4 = Color_tools.Lab_gradient(black, flash_yellow, 40)#, long_path=True)     
    
    color_gradient5 = Color_tools.Lch_gradient(flash_yellow, flash_rubis, 40) 
    color_gradient6 = Color_tools.Lch_gradient(flash_rubis, flash_yellow, 40)
    color_gradient7 = Color_tools.Lch_gradient(flash_yellow, flash_rubis, 40)
    color_gradient8 = Color_tools.Lab_gradient(flash_rubis, black, 40)
    color_gradient9 = Color_tools.Lch_gradient(black, blueviolet, 40)
    color_gradient10 = Color_tools.Lch_gradient(blueviolet, black, 40)
    color_gradient11 = Color_tools.Lch_gradient(black, flash_rubis, 40)
    color_gradient12 = Color_tools.Lab_gradient(flash_rubis, blueviolet, 40)
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
    probe_values = [0., 0.35, 0.42, 0.55, 0.76,  0.88, 0.94, 0.97, 0.98, 0.99, 0.993, 0.996, 0.999]
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
                                "phi_LS": 45.,
                                "LS_coords": (nx * 0.5, nx * 0.5),
                                "shininess": 400.,
                                "ratio_specular": 4.5})
    plotter.add_NB_layer(postproc_key=layer1_key, intensity=0.95, 
                         blur_ranges=[[0.76, 0.88, 1.0]], normalized=False, 
            skewness=0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [1, 0.,])
    plotter.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.4,
                         blur_ranges=[[0.85, 0.95, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 
    plotter.plot("divergent")

    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")




if __name__ == "__main__":
    plot()