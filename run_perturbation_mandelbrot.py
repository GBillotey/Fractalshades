# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from perturbation import Perturbation_mandelbrot
from fractal import Fractal_plotter, Fractal_colormap, Color_tools, Fractal_Data_array
import mpmath
from numpy_utils.xrange import Xrange_array, Xrange_polynomial


def plot():
    """
    Dev
    """
    directory = "/home/geoffroy/Pictures/math/perturb/dev_SA4"
    x = "-1.74928893611435556407228"
      #   -1.749288936114352993350644 
          #-1.7492889361143529933506439560953077
    # x = "-1.7492889361143"
         #"-1.749288936114355564073"
    y = "0."
    precision = 30
    dx = 8.e-20
    
    # emerauld shield
#    x = "-0.7658502608035550"
#    y = "-0.09799552764351510"
#    precision = 18
#    dx = 1.10625e-13
    
    # double embedded J set
    x = "-1.768667862837488812627419470"# + 0.001645580546820209430325900 i @ 2.7×10-22"
    y = "-0.001645580546820209430325900"
    precision = 30
    dx = 7.5e-22

    nx = 3200
    xy_ratio = 1.
    theta_deg = 0.
    complex_type = np.complex128
    
#    max_iter = 10000
#    M_divergence = 1.e3
#    epsilon_stationnary = 1.e-3
#    pc_threshold = 0.2

    mandelbrot = Perturbation_mandelbrot(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type, projection="cartesian",
                 precision=precision)
    
    mandelbrot.dev_loop(
            file_prefix="dev",
            subset=None,
            max_iter=400000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            pc_threshold=1.0,
            iref=0,
            SA_params=None, #{"cutdeg": 128},
            glitch_eps=1.e-3)

    stationnary = Fractal_Data_array(mandelbrot, file_prefix="dev",
                postproc_keys=('stop_reason', lambda x: x == 2), mode="r+raw")

#(self, file_prefix, subset, max_iter, M_divergence,
#                   epsilon_stationnary, pc_threshold=1.0, iref=0, 
#                   SA_params=None, glitch_eps=1.e-3):
#    c0 = mandelbrot.x + 1j * mandelbrot.y
#    order = mandelbrot.ball_method(c0, dx/nx * 10., 100000)
##    if order is None:
##        raise ValueError()
#    order = None
#    print("order", order)
#    newton_cv, nucleus = mandelbrot.find_nucleus(c0, order)
##    print("nucleus", newton_cv, nucleus)
##    print("shift", c0 - nucleus)
#    shift = c0 - nucleus
#    if (abs(shift.real) < mandelbrot.dx) and (abs(shift.imag) < mandelbrot.dy):
#        print("reference nucleus found at:\n", nucleus, order)
#        print("img coords:\n",
#              shift.real / mandelbrot.dx,
#              shift.imag / mandelbrot.dy)
#    else:
#        raise ValueError()
#    
#    z = mpmath.mp.zero
#    dzdc = mpmath.mp.zero
#    d2zdc2 = mpmath.mp.zero
#    z_tab = [dzdc]
#    dzdc_tab = [d2zdc2]
#    for i in range(1, order + 30):
#        dzdc = 2 * dzdc * z + 1.
#        z = z**2 + c0
#        z_tab += [z]
#        dzdc_tab +=[dzdc]
#        if   (i >= order):
#            print("z", i, z_tab[i], z_tab[i-order] - z_tab[i])
    

#
#    mandelbrot.dev_loop(
#        file_prefix="dev",
#        subset=None,
#        max_iter=max_iter,
#        M_divergence=M_divergence,
#        epsilon_stationnary=epsilon_stationnary,
#        pc_threshold=pc_threshold)
    potential_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    base_data_key = ("DEM_explore",
            {"px_snap": 0.5, "potential_dic": {"kind": "infinity"}})
    
    light_emerauld = np.array([15, 230, 186]) / 255.
    black = np.array([50, 50, 50]) / 255.
    gold = np.array([255, 210, 66]) / 255.
    purple = np.array([181, 40, 99]) / 255.
    royalblue = np.array([65, 105, 225]) / 255.
    blueviolet = np.array([138, 43, 226]) / 255.
    deepskyblue = np.array([0, 191, 255]) / 255.
    dark_blue = np.array([32, 52, 164]) / 255.
    navy_blue = np.array([0, 19, 95]) / 255.
    mint_green = np.array([112, 214, 203]) / 255.
    dark_cyan = np.array([71, 206, 176]) / 255.
    color_gradient = Color_tools.Lch_gradient(navy_blue, mint_green, 200,
                                              f= lambda x: x**2)
    color_gradient2 = Color_tools.Lch_gradient(purple, dark_blue,  200,
                                              f= lambda x:x**1.)
    colormap = Fractal_colormap(color_gradient, extent="mirror")
    colormap2 = Fractal_colormap(color_gradient2)
    
    plotter = Fractal_plotter(
        fractal=mandelbrot,
        base_data_key=potential_data_key,
        base_data_prefix="dev",
        base_data_function=lambda x: x,# np.log(1. + x),
        colormap=colormap,
        probes_val=[0., 0.05],
        probes_kind="qt",
        mask=stationnary)
    
    
    plotter.add_calculation_layer(postproc_key=potential_data_key)
    
    layer1_key = ("DEM_shade", {"kind": "Milnor",
                                "theta_LS": 45.,
                                "phi_LS": 75.,
                                "shininess": 40.,
                                "ratio_specular": 400.})
    plotter.add_grey_layer(postproc_key=layer1_key, intensity=0.85, 
                         blur_ranges=[],#[[0.99, 0.999, 1.0]],
                         normalized=False, 
            skewness=-0.1, shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})
    
    layer2_key = ("field_lines", {})
    Fourrier = ([1.,], [0, 0.2,])
    plotter.add_grey_layer(postproc_key=layer2_key, Fourrier=Fourrier,
                         hardness=1.5, intensity=0.12,
                         blur_ranges=[],#[[0.99, 0.99, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 0., "pegtop": 2.}) 
#
#    layer_key = ("DEM_explore",
#        {"px_snap": 0.5, "potential_dic": {"kind": "infinity"}})
#    plotter.add_grey_layer(
#        postproc_key=layer_key,
#        intensity=0.5, 
#        normalized=False,
#        skewness=0.0, 
#        shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})
#
    plotter.plot("dev")

if __name__ == "__main__":
    plot()
