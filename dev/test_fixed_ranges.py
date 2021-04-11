# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import Fractal_plotter, Fractal_colormap, Color_tools, Fractal_Data_array
from classical_mandelbrot import Classical_mandelbrot
import os

import PIL


def plot(t_dx, t_dy, subdir):
    directory = os.path.join(
            "/home/geoffroy/Pictures/math/classic_blender/test/fixed", subdir)
    plot_test_fixed_range(t_dx, t_dy, directory)


def plot_test_fixed_range(t_dx, t_dy, directory):
    """
    Plots the "Billiard game" image.
    """
    #==========================================================================
    # Parameters
    nx = 1200
    x = -1.74920463345912691e+00 + t_dx
    y = -2.8684660237361114e-04 + t_dy
    dx = 5.e-12
    xy_ratio = 1.0
    theta_deg = 0.
    # known_order = 134 # if don't know use None - speed up the calculation
    complex_type = np.complex128
    
    # =========================
#    dx = 0.7
#    x = -0.125
#    y = 0.90
#    xy_ratio = 1.0
#    theta_deg = 0.
#    nx = 1600
##    known_order = 1
#    complex_type = np.complex128

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
#    stationnary = Fractal_Data_array(mandelbrot, file_prefix="loop-1",
#                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")

#    #  We search for a quasi-cycle (Brent algorithm)
#    eps_pixel = 0.1 * dx / float(nx)
#    mandelbrot.r_candidate_cycles(file_prefix="r_candidate",
#        subset=stationnary, max_iter=2000000, M_divergence = 1.e3,
#        eps_pixel=eps_pixel, start_file="loop-1", k_power=1.05)
#
#    cycling = Fractal_Data_array(mandelbrot, file_prefix="r_candidate",
#                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")
#
#    # We iterate to the 1st cyle detection
#    mandelbrot.teleportation(file_prefix="teleportation",
#        subset=cycling, max_iter=2000000, M_divergence = 1.e3,
#        eps_pixel=eps_pixel, start_file="loop-1",
#        r_candidate_file="r_candidate")
#
#    # now we do the newton iteration
#    mandelbrot.newton_cycles(file_prefix="newton", subset=stationnary,
#                             max_newton=60, max_cycle=40000, eps_cv=1.e-12,
#                             start_file="teleportation",
#                             r_candidate_file="r_candidate",
#                             pc_threshold=0.0,
#                             known_order=known_order)
#    cycling = Fractal_Data_array(mandelbrot, file_prefix="newton",
#                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")


    #==========================================================================
    # Plot the divergent part
    # Color based on potential
    base_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    magma = Fractal_colormap((0.1, 0.9, 200), plt.get_cmap("magma"))
    colormap_div = magma
    colormap_div.extent = "clip" # "mirror"
    
    base_data_prefix = "loop-1"
    base_data_function = lambda x: np.log(x)
    probes_qt = [0.1, 0.9] #0.45, 0.55
    probes_z = [6.25214168059345, 6.68992757797241]
    probes_z = [6.25214168059345, 6.68992757797241]
    probes_z = [6.34972512572658, 6.36105676309507]



    plotter = Fractal_plotter(mandelbrot, base_data_key, base_data_prefix,
                              base_data_function, colormap_div, probes_z,
                              probes_kind="z", mask=~divergence)    

    # shade layer based on field lines
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.,])
    plotter.add_grey_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.7,
                         blur_ranges=[[0.85, 0.95, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 


    height_map = ("Milnor_height_map", {"px_snap": 1.})
    normal_map = ("DEM_shade_normal_n", {})
    plotter.add_grey_layer(postproc_key=height_map, intensity=-1.0, 
                         skewness=0.0, disp_layer=True, layer_mask_color=(1.,),
                         clip_max=0.1)
    plotter.add_normal_map(postproc_key=normal_map)
    
    # shade layer based on normal from DEM 'Milnor'
    DEM_shade = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": 135.,
                                "phi_LS": 35.,
                                "shininess": 40.,
                                "ratio_specular": 4.5})
    plotter.add_grey_layer(postproc_key=DEM_shade, intensity=0.95, 
                         blur_ranges=[[0.97, 0.99, 1.0]], normalized=False, 
            skewness=-0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

    height_map2 = ("potential_height_map", {})
    plotter.add_grey_layer(postproc_key=height_map2, intensity=1.0, 
                         blur_ranges=[[0.97, 0.99, 1.0]], normalized=False,
                         veto_blend=True,
            skewness=-0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.},
            clip_min=6., clip_max=8.)


    plotter.add_zoning_layer()


    plotter.plot("divergent")


#    # ==== Composite plot : Minibrot + divergent part
#    plotter_minibrot.blend_plots("divergent.png", "minibrot.png")
#    plotter_minibrot.blend_plots("divergent_DEM_shade_normal_n.nmap2.png",
#                                 "minibrot_attr_normal_n.nmap1.png",
#           100                      im_mask="minibrot.mask.png",
#                                 output_file="composite_normal.png")

def tile(im_path1, im_path2, im_path, tw):
    im1 = PIL.Image.open(im_path1).convert('RGBA')
    im2 = PIL.Image.open(im_path2).convert('RGBA')
    w, h = im1.size
    im = PIL.Image.new('RGBA', size=(w + tw - 1, h))
    
    im.paste(im1, (0, 0))
    im.paste(im2, (tw - 1, 0))
    im.save(im_path)

if __name__ == "__main__":
    dx = 5.e-12
    plot(t_dx=0.75*dx, t_dy=0., subdir="t1")
    plot(t_dx=-0.25*dx, t_dy=0., subdir="t2")
    
    base_path = "/home/geoffroy/Pictures/math/classic_blender/test/fixed"
    im_path1 = os.path.join(base_path, "t2/divergent.png")
    im_path2 = os.path.join(base_path, "t1/divergent.png")
    im_path = os.path.join(base_path, "combined_divergent.png")
    tile(im_path1, im_path2, im_path, tw=1200)

    im_path1 = os.path.join(base_path, "t2/divergent_DEM_shade.layer2.png")
    im_path2 = os.path.join(base_path, "t1/divergent_DEM_shade.layer2.png")
    im_path = os.path.join(base_path, "combined_DEM_shade.png")
    tile(im_path1, im_path2, im_path, tw=1200)
    
    im_path1 = os.path.join(base_path, "t2/divergent_field_lines.layer0.png")
    im_path2 = os.path.join(base_path, "t1/divergent_field_lines.layer0.png")
    im_path = os.path.join(base_path, "combined_field_lines.png")
    tile(im_path1, im_path2, im_path, tw=1200)
    
    im_path1 = os.path.join(base_path, "t2/divergent_potential_height_map.layer3.png")
    im_path2 = os.path.join(base_path, "t1/divergent_potential_height_map.layer3.png")
    im_path = os.path.join(base_path, "combined_potential_height.png")
    tile(im_path1, im_path2, im_path, tw=1200)