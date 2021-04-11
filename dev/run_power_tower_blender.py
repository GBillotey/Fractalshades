# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools
from Power_tower import Power_tower


def plot():
    """========================================================================
    Run the script to generate 1 of 2 fractal images from the 'power tower'
    Mandelbrot set, z(n+1) <- c**zn (c = pixel).
    User-input needed : *choice* and the target *directory*
    ========================================================================"""
    choice = 2

    select = {1: plot_tower_draft,
              2: plot_ankh,
              3: plot_ankh2,
              4: plot_ankh3}

    directory = {1: "/home/gby/Pictures/Mes_photos/math/fractal/draft_tower3",
                 2: "/home/geoffroy/Images/math/classic_blender/ankh",
                 3: "/home/gby/Pictures/Mes_photos/math/fractal/tower3",
                 4: "/home/gby/Pictures/Mes_photos/math/fractal/tower4"}   

    # run the selected plot
    select[choice](directory[choice])


def plot_tower_draft(directory):
    """
    Plots power tower Mandelbrot in 'draft' mode
    Use for exploration only
    """
    #==========================================================================
    # Parameters
    x = -4.125
    y = 0.0125
    dx = 0.0125
    
    x = -4.11850313
    y = 0.02096875
    dx = 0.0006563
    
    x = -4.118508873
    y = 0.020730021
    dx = 0.000008204

    x = -4.1185082368344
    y = 0.0207312924563
    dx = 1.02547E-07

    x = -4.118508273944
    y = 0.020731315722
    dx = 1.28184E-09

    x = -4.118508274085
    y = 0.020731315723
    dx = 3.20459E-11
    
    x = -4.11850827408001
    y = 0.02073131572533
    dx = 8.43313E-13 * 4 * 2 #* 0.1    
    
    x = -4.11850827407427 - 4.e-13
    y = 0.02073131571634
    dx = 4.00574E-12
    
    x = -4.11850827407432000
    y= 0.02073131571776570
    dx = 6.50932E-13


    xy_ratio = 1.0
    theta_deg = 0. 
    nx = 800
    complex_type = np.complex256

    #==========================================================================
    # Calculation
    mandelbrot = Power_tower(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type)
    mandelbrot.draft_loop(file_prefix="draft", subset=None, max_iter = 10000,
                          epsilon_stationnary = 1e-4)
    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="draft") == 1)

    #==========================================================================
    # Plot - colors based on potential
    base_data_key = ("potential", {"kind": "transcendent"})
    base_data_prefix = "draft"
    base_data_function = lambda x:  np.log(x)
    probe_values = [0.0, 0.5, 1.0]
    magma = Fractal_colormap((0.9, 0.1, 200), plt.get_cmap("magma"))
    base_colormap = - magma + magma
    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, base_colormap, probe_values,
                              mask=~divergence)
    plotter.plot("draft", mask_color=(1., 1., 1.))


def plot_ankh(directory):
    """
    Plots a Ankh-shaped detail in Power tower mandelbrot
    """
    #==========================================================================
    #  Parameters
    x = -2.2180444173011
    y = 0.4477385106434
    dx = 3.27863E-07

    xy_ratio = 1.0
    theta_deg = 0. 
    nx = 3600
    complex_type = np.complex128

    #==========================================================================
    #  Calculations
    mandelbrot = Power_tower(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type,
                 spherical=True)
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 100000,
                          epsilon_stationnary = 1e-3)
    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.1 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
                                  subset=stationnary, max_iter=400000,
                                  eps_pixel=eps_pixel, start_file="loop-1",
                                  k_power=1.05)
    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=300000,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")
        
    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=16, max_cycle=20000, eps_cv=1.e-12,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=None)
    valid = np.abs(mandelbrot.raw_data('attractivity', file_prefix="newton")
               <= 1)

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
    black = np.array([0, 0, 0]) / 255.
    plotter_inf.plot("div", mask_color=black)

    #==========================================================================
    # plots the "reaching limit cycle" part (color based on cycle order)
    base_data_key =  ("raw", {"code": "r"})
    base_data_prefix = "newton"
    base_data_function = lambda x: np.where(x > 1000., 1000., x)
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
                              probe_values, mask=~valid)

    # shade layer based on normal vec with a little "tuning" to get finer
    # shade details
    layer_key2 = ("power_attr_heightmap", {"theta_LS": -45.,
                                        "phi_LS": 70.,
                                        "shininess": 100.,
                                        "ratio_specular": 1.})
    layern_key2 = ("power_attr__normal_n", {})#, {"px_snap", 2.})
    plotter.add_NB_layer(postproc_key=layer_key2, intensity=1.0,
                           skewness=0.0, #hardness=1.0,
                           disp_mask_color=(1.,),
                           disp_layer=True)
    plotter.add_normal_map(postproc_key=layern_key2)

    plotter.plot("cycling", mask_color=(0, 0, 0, 0))

    # ==== Composite plot : cycling + divergent part
    plotter.blend_plots("div.png", "cycling.png")



def plot_ankh2(directory):
    """
    Plots a Ankh-shaped detail in Power tower mandelbrot
    """
    #==========================================================================
    #  Parameters
    x = -2.2180444173011
    y = 0.4477385106434
    dx = 3.27863E-07

    x = -4.11850827408001
    y = 0.02073131572533
    dx = 8.43313E-13 * 4 * 2

    x = -4.11850827407467
    y = 0.02073131571634
    dx = 4.00574E-12

    xy_ratio = 1.0
    theta_deg = 0. 
    nx = 6400
    complex_type = np.complex256

    #==========================================================================
    #  Calculations
    mandelbrot = Power_tower(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type)
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 50000,
                          epsilon_stationnary = 1e-3)
    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.8 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
                                  subset=stationnary, max_iter=400000,
                                  eps_pixel=eps_pixel, start_file="loop-1",
                                  k_power=1.05)
    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=300000,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")
        
    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=16, max_cycle=20000, eps_cv=1.e-12,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=None, min_order=100)
    valid = np.abs(mandelbrot.raw_data('attractivity', file_prefix="newton")
               <= 1)

    #==========================================================================
    # Plot the divergent part, in theory shall be a sparse set. Here bailout is
    # based on numerical overflow, an limit the max reach of a limit cycle - 
    # which can be arbitrary large.
    base_data_prefix = "loop-1"
    base_data_key = ("phase", {"source": "zn"})
    base_data_function = lambda x: -np.cos(2* x - np.pi*0.1)
    probe_values = [0.0, 1.00]
    base_colormap = Fractal_colormap((1.0, 0.2, 200), plt.get_cmap("copper"))
    plotter_inf = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix, 
                                  base_data_function, base_colormap,
                                  probe_values, mask=~divergence)             
    black = np.array([0, 0, 0]) / 255.
    plotter_inf.plot("div", mask_color=black)

    #==========================================================================
    # plots the "reaching limit cycle" part (color based on cycle order)
    base_data_key =  ("raw", {"code": "r"})
    base_data_prefix = "newton"
    base_data_function = lambda x: (np.log(x + 20.)) % np.log(100.)
    probe_values = [0.0, 0.13, 0.14, 0.40, 0.50, 0.55, 0.70, 0.75, 0.85]
    probe_values = np.linspace(0., 1., 17) 
#[0.0, 0.07, 0.13, 0.14, 0.18, 0.26, 0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    
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
    colormap_newton =  - colormap_newton + colormap_newton

    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_newton,
                              probe_values, mask=~valid)

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

def plot_ankh3(directory):
    """
    Plots a Ankh-shaped detail in Power tower mandelbrot
    """
    #==========================================================================
    #  Parameters
    x = -2.2180444173011
    y = 0.4477385106434
    dx = 3.27863E-07

    x = -4.11850827408001
    y = 0.02073131572533
    dx = 8.43313E-13 * 4 * 2

    x = -4.11850827407467
    y = 0.02073131571634
    dx = 4.00574E-12
    
  
    x = -4.11850827407432000
    y= 0.02073131571776570
    dx = 6.50932E-13

    xy_ratio = 1.0
    theta_deg = 0. 
    nx = 6400
    complex_type = np.complex256

    #==========================================================================
    #  Calculations
    mandelbrot = Power_tower(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type)
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter = 50000,
                          epsilon_stationnary = 1e-3)
    divergence = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                  == 1)
    stationnary = (mandelbrot.raw_data('stop_reason', file_prefix="loop-1")
                   == 2)

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.8 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
                                  subset=stationnary, max_iter=400000,
                                  eps_pixel=eps_pixel, start_file="loop-1",
                                  k_power=1.05)
    cycling = (mandelbrot.raw_data('stop_reason', file_prefix="r_candidate")
               == 2)

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=300000,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")
        
    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=16, max_cycle=20000, eps_cv=1.e-12,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             known_order=None, min_order=140)
    valid = np.abs(mandelbrot.raw_data('attractivity', file_prefix="newton")
               <= 1)

    #==========================================================================
    # Plot the divergent part, in theory shall be a sparse set. Here bailout is
    # based on numerical overflow, an limit the max reach of a limit cycle - 
    # which can be arbitrary large.
    base_data_prefix = "loop-1"
    base_data_key = ("phase", {"source": "zn"})
    base_data_function = lambda x: -np.cos(2* x - np.pi*0.1)
    probe_values = [0.0, 1.00]
    base_colormap = Fractal_colormap((0.8, 0.2, 200), plt.get_cmap("magma"))
    plotter_inf = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix, 
                                  base_data_function, base_colormap,
                                  probe_values, mask=~divergence)             
    black = np.array([0, 0, 0]) / 255.
    plotter_inf.plot("div", mask_color=black)

    #==========================================================================
    # plots the "reaching limit cycle" part (color based on cycle order)
    base_data_key =  ("raw", {"code": "r"})
    base_data_prefix = "newton"
    base_data_function = lambda x: (np.log(x - 140.)) 
    probe_values = [0.0, 0.13, 0.14, 0.40, 0.50, 0.55, 0.70, 0.75, 0.85]
    probe_values = np.linspace(0., 1., 15) 
#[0.0, 0.07, 0.13, 0.14, 0.18, 0.26, 0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    gold = np.array([255, 210, 66]) / 255.
    light_yellow = np.array([255, 252, 199]) / 255.
    purple = np.array([181, 40, 99]) / 255.
    dark_blue = np.array([32, 52, 164]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    ice = np.array([158, 228, 217]) / 255.
    black = np.array([0, 0, 0]) / 255.
    grey = np.array([66, 0, 31]) / 255.
    paon_green = np.array([82, 233, 104]) / 255.
    
    color_gradient1 = Color_tools.Lch_gradient(dark_blue, ice, 200)
    color_gradient2 = Color_tools.Lch_gradient(ice, dark_blue, 200)
    color_gradient3 = Color_tools.Lch_gradient(dark_blue, ice, 100) 
    color_gradient4 = Color_tools.Lch_gradient(ice, dark_blue, 100) 
    color_gradient5 = Color_tools.Lch_gradient(dark_blue, ice, 100) 
    color_gradient6 = Color_tools.Lch_gradient(ice, dark_blue, 100)
    color_gradient7 = Color_tools.Lch_gradient(dark_blue, ice, 100)
    color_gradient8 = Color_tools.Lch_gradient(ice, dark_blue, 100)
    colormap_newton = (Fractal_colormap(color_gradient1) + 
                       Fractal_colormap(color_gradient2) + 
                       Fractal_colormap(color_gradient3) + 
                       Fractal_colormap(color_gradient4) + 
                       Fractal_colormap(color_gradient5) + 
                       Fractal_colormap(color_gradient6) + 
                       Fractal_colormap(color_gradient7)+ 
                       Fractal_colormap(color_gradient8))
    colormap_newton =   colormap_newton - colormap_newton
    magma = Fractal_colormap((0.2, 0.8, 200), plt.get_cmap("viridis"))
    colormap_newton = magma - magma + magma - magma + magma - magma + magma - magma + magma - magma + magma - magma + magma - magma

    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_newton,
                              probe_values, mask=~valid)

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