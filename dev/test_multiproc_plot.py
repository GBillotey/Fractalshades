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

    directory = {1: "/home/geoffroy/Pictures/math/classic_blender/1_test",
                 2: "/home/geoffroy/Pictures/math/classic_blender/2b",
                 3: "/home/geoffroy/Pictures/math/classic_blender/3f",
                 4: "/home/geoffroy/Pictures/math/classic_blender/4",
                 5: "/home/geoffroy/Pictures/math/classic_blender/5f",
                 6: "/home/geoffroy/Pictures/math/classic_blender/6"}

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
    xy_ratio = 1.0
    theta_deg = 0.
    nx = 3200
    known_order = 1
    complex_type = np.complex128

    #==========================================================================
    # Calculations
    mandelbrot = Classical_mandelbrot(directory, x, y, dx, nx, xy_ratio,
                                      theta_deg, chunk_size=200,
                                      complex_type=complex_type,
                                      projection="cartesian")
    # Compute loop
    mandelbrot.first_loop(file_prefix="loop-1", subset=None, max_iter=2000000,
                          M_divergence = 1.e3, epsilon_stationnary = 1e-3)

    divergence = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")
    stationnary = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")

    #  We search for a quasi-cycle (Brent algorithm)
    eps_pixel = 0.1 * dx / float(nx)
    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
        subset=stationnary, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05)

    cycling = Fractal_Data_array(mandelbrot, file_prefix="r_candidate",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")

    # We iterate to the 1st cyle detection
    mandelbrot.teleportation(file_prefix="teleportation",
        subset=cycling, max_iter=2000000, M_divergence = 1.e3,
        eps_pixel=eps_pixel, start_file="loop-1",
        r_candidate_file="r_candidate")

    # now we do the newton iteration
    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
                             max_newton=60, max_cycle=40000, eps_cv=1.e-12,
                             start_file="teleportation",
                             r_candidate_file="r_candidate",
                             pc_threshold=0.0,
                             known_order=known_order)
    cycling = Fractal_Data_array(mandelbrot, file_prefix="newton",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")

    #==========================================================================
#     Plot the mandelbrot interior
#     Color base on attractivity
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([0, 0, 0]) / 255.
    color_gradient = Color_tools.Lab_gradient(light_emerauld, black, 100,
                                          f=lambda x: (1. - x)**10)
    colormap_newton = Fractal_colormap(color_gradient)
    base_data_prefix = "newton"
    base_data_key = ("attr_height_map", {})
    base_data_function = lambda x: x
    probe_values = [0., 1.]
    plotter_minibrot = Fractal_plotter(mandelbrot, base_data_key,
                                       base_data_prefix, base_data_function,
                                       colormap_newton, probe_values,
                                       mask=~cycling)

    # shade layer based on normal vec : 
    # normal = attr * np.conj(dattrdc) / np.abs(dattrdc)
    layern_key2 = ("attr_normal_n", {})
    blender_mini_disp = ("attr_height_map", {})
    layern_key3 = ("attr_shade", {"theta_LS": 45.,
                                "phi_LS": 30.,
                                "shininess": 200.,
                                "ratio_specular": 1.})
    plotter_minibrot.add_grey_layer(postproc_key=blender_mini_disp, intensity=1.0,
                           skewness=0.5, disp_layer=True, disp_mask_color=(0,))
    
    plotter_minibrot.add_grey_layer(postproc_key=layern_key3, intensity=1.0,
                skewness=0., shade_type={"Lch": 4., "overlay": 4., "pegtop": 1.})

    plotter_minibrot.add_normal_map(postproc_key=layern_key2)#, intensity=1.0,
    plotter_minibrot.add_mask_layer()
    

    plotter_minibrot.plot("minibrot", mask_color=(1., 0., 0., 0.))

    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    magma = Fractal_colormap((1.0, 0.0, 200), plt.get_cmap("magma"))
    colormap_div = magma
    base_data_prefix = "loop-1"
    base_data_function = lambda x: np.log(x)
    probes_qt = [0., 0.985]

    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_div, probes_qt,
                              mask=~divergence)    

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_grey_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.7,
                         blur_ranges=[[0.85, 0.95, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 


    layer1_key = ("Milnor_height_map", {"px_snap": 1.})
    layern_key2 = ("DEM_shade_normal_n", {})
    plotter.add_grey_layer(postproc_key=layer1_key, intensity=-1.0, 
                         skewness=0.0, disp_layer=True, disp_mask_color=(1.,),
                         vmax=0.1)
    plotter.add_normal_map(postproc_key=layern_key2)
    
    # shade layer based on normal from DEM 'Milnor'
    layer3_key = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": 45.,
                                "phi_LS": 35.,
                                "shininess": 40.,
                                "ratio_specular": 4.5})
    plotter.add_grey_layer(postproc_key=layer3_key, intensity=0.95, 
                         blur_ranges=[[0.97, 0.99, 1.0]], normalized=False, 
            skewness=-0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})
    
    
    plotter.add_zoning_layer()



    plotter.plot("divergent")


    # ==== Composite plot : Minibrot + divergent part
    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")
    plotter_minibrot.blend_plots("divergent_DEM_shade_normal_n.nmap3.png",
                                 "minibrot_attr_normal_n.nmap2.png",
                                 im_mask="minibrot.mask.png",
                                 output_file="composite_normal.png")




if __name__ == "__main__":
    plot(choice)