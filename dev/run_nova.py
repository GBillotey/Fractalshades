# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools, Fractal_Data_array
from Nova import Nova


def plot():
    """========================================================================
    Run the script to generate 1 of 4 fractal images
    User-input needed : *choice* and the target *directory*
    ========================================================================"""
    choice = 1

    select = {1: plot_nova6_whole_set,
              2: plot_nova6_zoom,
              3: plot_alt_whole_set}

    directory = {1: "/home/geoffroy/Pictures/math/classic_blender/nova1b",#0_test",
                 2: "/home/geoffroy/Pictures/math/classic_blender/nova2",
                 3: "/home/geoffroy/Pictures/math/classic_blender/nova5"}   

    # run the selected plot
    select[choice](directory[choice])


def plot_nova6_whole_set(directory):
    """
    Plots Nova Mandelbrot
    """
    #==========================================================================
    #   Parameters
    x = -1.0
    y = 0.
    dx = 3.5 #* 2.
    
#    x = 0
#    y = 0.
#    dx = 0.5 #* 2.
    
    
    xy_ratio = 1.0
    theta_deg = 90. 
    nx = 8000
    complex_type = np.complex128
    known_order = None

    R = 0.1#0.8
    p = 12
    
    R = 0.4
    p = 8

    epsilon_cv = min(0.1 * dx / nx, 1.e-8)
    pc = 0.02

    #==========================================================================
    #   Calculations
    mandelbrot = Nova(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type, R=R, p=p)
#    z0 = mandelbrot.critical_point(R=R, p=p)
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 400000,
                          epsilon_cv=None, epsilon_stationnary=1e-3,
                           pc_threshold=pc)
#        divergence = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
#                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")
    stationnary = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")
#    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
#                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary,
        max_iter=400000,  pc_threshold=pc,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05)
#    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
#               == 1)
    cycling = Fractal_Data_array(mandelbrot, file_prefix="r_candidate",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=400000, pc_threshold=pc,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=40, max_cycle=20000, eps_cv=1.e-12, # 10 -> 14
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order)
#    real_loop = (mandelbrot.raw_data("r", file_prefix="newton") > 1)
    real_loop = Fractal_Data_array(mandelbrot, file_prefix="newton",
                postproc_keys=('r', lambda x: x > 1), mode="r+raw")
    #convergence = (mandelbrot.raw_data("r", file_prefix="newton") == 1)
    convergence = Fractal_Data_array(mandelbrot, file_prefix="newton",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")

    mandelbrot.first_loop(file_prefix="loop-2", subset=convergence,
                          max_iter=400000, epsilon_cv=epsilon_cv,
                          epsilon_stationnary=None, pc_threshold=pc,
                          zr_file_prefix="newton")
#    convergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-2")
#                   == 1)
    convergence = Fractal_Data_array(mandelbrot, file_prefix="loop",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")

    #==========================================================================
    #   Plots everything except the minibrot "convergent" part

    base_data_prefix = "loop-2"
    potential = ("potential", 
                     {"kind": "convergent", "epsilon_cv": epsilon_cv})
    
    nova_iteration = {}#"R": R, "p": p, "z0": z0}
    #nova_iteration = None
    height_map = ("DEM_height_map", {"iteration": nova_iteration, "px_snap": 0.005})

        
#    base_data_key = ("abs", {"source": "alpha"})
#    base_data_key = ("abs", {"source": "alpha"})

    base_data_function = lambda x: np.log(x + 1.)
    probe_values = [0.0005, 0.50,  0.9995]
    probe_values = [0.0, 0.07, 0.5]

    gold = np.array([255, 210, 66]) / 255.

    yellow = np.array([255, 253, 167]) / 255.
    purple = np.array([181, 40, 99]) / 255.
    blue = np.array([33, 77, 157]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    brown = np.array([147, 64, 93]) / 255.
    vertdeau = np.array([70, 160, 161])/255.  
    cmap = Fractal_colormap(Color_tools.Lch_gradient(purple,  yellow, 50,
                            f=lambda x: (1-x)**16)) + Fractal_colormap(Color_tools.Lch_gradient(purple, vertdeau, 100, f=lambda x: 1-(1-x)**3))
#    cmap = Fractal_colormap((0.0, .5, 100), plt.get_cmap("magma")) + Fractal_colormap((0.5, 1.0, 100), plt.get_cmap("magma"))
    base_colormap = cmap
    base_colormap.extent = "mirror"

    plotter_potential = Fractal_plotter(mandelbrot, height_map, base_data_prefix, 
                            base_data_function, base_colormap, probe_values,
                            probes_kind="qt", mask=real_loop)
    plotter_potential.add_calculation_layer(potential)

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([0., 0.], [1, 0.,])
    plotter_potential.add_grey_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness= 0.25, intensity=0.6, skewness=-0.6,
                         blur_ranges=[[0.99, 0.999, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 1.},
                         veto_blend=False)  


    # shade layer based on potential DEM
    layer_key = ("DEM_shade", {"kind": "potential",
                               "theta_LS": 135.,
                                "phi_LS": 40.,
                                "shininess": 100.,
                                "ratio_specular": 1.4})
    plotter_potential.add_grey_layer(postproc_key=layer_key,# Fourrier=Fourrier,
                         hardness= 1.0, intensity=0.8, skewness= 0.0,
                         blur_ranges=[[0.9999, 0.99999, 1.0]], 
                         shade_type={"Lch": 4., "overlay": 1., "pegtop": 1.},
                         veto_blend=True)  

    #height_map = ("DEM_height_map", {})
    layer2_key = ("potential_height_map", {})
#    layern_key2 = ("DEM_shade_normal_n", {"px_snap", 2.})
    plotter_potential.add_grey_layer(postproc_key=height_map, intensity=-1.0, 
                         skewness=0.0, disp_layer=True, layer_mask_color=(1.,),
                         clip_max=None, user_func=lambda x: np.log(x + 1.))
#    plotter.add_grey_layer(postproc_key=layer2_key, intensity=-1.0, 
#                         skewness=0.0, disp_layer=True, disp_mask_color=(1.,))

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
    vertdeau = np.array([70, 160, 161])/255.  
    color_gradient = Color_tools.Lab_gradient(gold, black, 100,
                                          f=lambda x: x*2) #**6)   
    colormap_newton =  Fractal_colormap(color_gradient)

    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_newton, probe_values,
                              mask=~real_loop)

    # shade layer based on normal vec : 
    layern_key2 = ("attr_shade", {"theta_LS": 135.,
                                "phi_LS": 35.,
                                "shininess": 100.,
                                "ratio_specular": 1.})
    plotter.add_grey_layer(postproc_key=layern_key2, intensity=0.95,
                           skewness=0.20, hardness=1.0,
                           shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.},
                         veto_blend=True)
    
    blender_mini_disp = ("attr_height_map", {})
    plotter.add_grey_layer(postproc_key=blender_mini_disp, intensity=1.0,
                           skewness=0.95, disp_layer=True, layer_mask_color=(0,))
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
    
    x = -0.343915827930756
    y = 0.000167563867917
    dx = 9.644445E-10

    xy_ratio = 1.0#1.0
    theta_deg = 0. 
    nx = 4000
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

#    base_data_prefix = "loop-2"
#    base_data_key = ("potential", 
#                     {"kind": "convergent", "epsilon_cv": epsilon_cv})
#
#    base_data_function = lambda x: np.log(x)
#    probe_values = [0.0, 0.24, 0.54, 0.82, 0.96, 0.98, 0.995, 0.9995]
#
#    magma = Fractal_colormap((0.0, 1., 200), plt.get_cmap("magma"))
#    inferno = Fractal_colormap((0.0, 1., 200), plt.get_cmap("inferno"))
#    viridis = Fractal_colormap((0.0, 1., 200), plt.get_cmap("viridis"))
#    Blues = Fractal_colormap((0.15, 1., 200), plt.get_cmap("Blues"))
#    Purples = Fractal_colormap((0.15, 1., 200), plt.get_cmap("Reds"))
#    black = np.array([0, 0, 0]) / 255.
#    white = np.array([100, 100, 100]) / 255.
#    black2 = np.array([50, 50, 50]) / 255.
#    silver = np.array([200, 220, 220])/255.
#    color_gradient = Color_tools.Lab_gradient(silver, white, 100)   
#    gr_colormap =  Fractal_colormap(color_gradient)
#    base_colormap = gr_colormap - gr_colormap - magma + magma - inferno + inferno - inferno
#    plotter_potential = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix, 
#                                  base_data_function, base_colormap, probe_values,
#                                  mask=real_loop)
#
#    # shade layer based on field lines
#    layer2_key = ("field_lines", {})
#    Fourrier = ([0.] * 8 + [1.], [0, 0.,])
#    plotter_potential.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
#                         hardness= 0.25, intensity=0.6, skewness= -0.6,
#                         blur_ranges=[[0.75, 0.9, 1.0]], 
#                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 1.})  
#
#    # shade layer based on potential DEM
#    layer_key = ("DEM_shade", {"kind": "potential",
#                               "theta_LS": 135.,
#                                "phi_LS": 40.,
#                                "shininess": 100.,
#                                "ratio_specular": 1.4})
#    plotter_potential.add_NB_layer(postproc_key=layer_key,# Fourrier=Fourrier,
#                         hardness= 1.0, intensity=0.8, skewness= 0.0,
#                         blur_ranges=[[0.85, 0.91, 1.0]], 
#                         shade_type={"Lch": 4., "overlay": 1., "pegtop": 1.})  
#
#    plotter_potential.plot("conv", mask_color=black)

    #==========================================================================
    # Plots the minibrot, then blends
    # colors based on cycle attractivity
    base_data_key =  ("abs", {"source": "attractivity"})#{"raw", }
    base_data_prefix = "newton"
    base_data_function = lambda x: x
    probe_values = [0.0, 1.0]

    light_emerauld = np.array([15, 230, 186]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    flash_rubis = np.array([255., 0., 121.]) / 255.
    flash_navy = np.array([0., 64., 255.]) / 255.
    black = np.array([0, 0, 0]) / 255.
    color_gradient = Color_tools.Lab_gradient(dark_cyan, black, 100,
                                          f=lambda x: 1. - x**4)   
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

def plot_alt_whole_set(directory):
    """
    Plots Nova Mandelbrot - degree 6 with R = 0.8
    """
    #==========================================================================
    #   Parameters
    x = 0.
    y = 0.
    dx = 5.2 
    
    #dx = 0.2
    
    xy_ratio = 1.0
    theta_deg = 0. 
    nx = 400
    complex_type = np.complex128
    known_order = None

    R = 1.
    p = 5 + 1j*np.pi*0.1
    z0 = "c"#1.
    epsilon_cv = 1e-7

    #==========================================================================
    #   Calculations
    mandelbrot = Nova(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type)
    z0 = mandelbrot.critical_point(R=R, p=p)
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 400000,
                          epsilon_cv=None, epsilon_stationnary=1e-4,
                          R=R, p=p, z0=z0, pc_threshold=0.5)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.1 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=400000,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05,
        R=R, p=p, z0=z0, pc_threshold=0.5)
    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 1)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=400000,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate", R=R, p=p, z0=z0, pc_threshold=0.2)

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=30, max_cycle=20000, eps_cv=1.e-12,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=known_order, R=R, p=p, z0=z0,
                             pc_threshold=0.5)
    real_loop = (mandelbrot.raw_data("r", file_prefix="newton") > 1)
    convergence = (mandelbrot.raw_data("r", file_prefix="newton") == 1)

    mandelbrot.first_loop(file_prefix="loop-2", subset=convergence,
                          max_iter=40000, epsilon_cv=epsilon_cv,
                          epsilon_stationnary=None, pc_threshold=0.5,
                          R=R, p=p, z0=z0, zr_file_prefix="newton")
    convergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-2")
                   == 1)

    #==========================================================================
    #   Plots everything except the minibrot "convergent" part

    base_data_prefix = "loop-2"
    base_data_key = ("potential", 
                     {"kind": "convergent", "epsilon_cv": epsilon_cv})

    base_data_function = lambda x: np.log(x)
    probe_values = [0.01, 0.50,  0.99]

    gold = np.array([255, 210, 66]) / 255.
    light_yellow = np.array([255, 252, 199]) / 255.
    purple = np.array([181, 40, 99]) / 255.
    dark_blue = np.array([32, 52, 164]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    black = np.array([0, 0, 0]) / 255.
    grey = np.array([66, 0, 31]) / 255.
    paon_green = np.array([82, 233, 104]) / 255.
    color_gradient = Color_tools.Lab_gradient(dark_cyan, paon_green, 100,
                                              f=lambda x: x)
    cmap = Fractal_colormap(color_gradient)             
    copper = Fractal_colormap((0., .99, 200), plt.get_cmap("copper"))
    base_colormap = -cmap + cmap
    
    plotter_potential = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix, 
                            base_data_function, base_colormap, probe_values,
                            mask=None)

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([0.] * 1 + [1.], [0, 0.,])
    plotter_potential.add_NB_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness= 2.25, intensity=0.3, skewness= -0.3,
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



if __name__ == "__main__":
    plot()