# -*- coding: utf-8 -*-
import numpy as np

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.settings as settings
import fractalshades.colors as fscolors

def plot():
    """
    Example plot of "Dinkydau flake" location, classic test case for 
    perturbation techinque and glitch correction.
    """
    directory = "./perturb3"
    # A simple showcas using perturbation technique
    x, y = "-0.75", "0."
    dx = "5.e0"
    precision = 10
    nx = 1600


    # Set to True if you only want to rerun the post-processing part
    settings.skip_calc = False
    # Set to True to enable multi-processing
    settings.enable_multiprocessing = True

#    xy_ratio = 1.0
#    theta_deg = 0.
    # complex_type = np.complex128

    mandelbrot = fsm.Perturbation_mandelbrot(directory)
    mandelbrot.zoom(
            precision=precision,
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=1.0,
            theta_deg=0., 
            projection="cartesian",
            antialiasing=True)

    mandelbrot.calc_std_div(
            complex_type=np.complex128,
            file_prefix="dev",
            subset=None,
            max_iter=50000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            pc_threshold=0.1,
            SA_params={"cutdeg": 64,
                       "cutdeg_glitch": 8},
#                       "SA_err": 1.e-4,
#                       "use_Taylor_shift": True},
            glitch_eps=1.e-6,
            interior_detect=True,
            glitch_max_attempt=20)

    mandelbrot.run()

    cv = fs.Fractal_Data_array(mandelbrot, file_prefix="dev",
                postproc_keys=('stop_reason', lambda x: x != 2), mode="r+raw")
    potential_data_key = ("potential", {})


#    gold = np.array([255, 210, 66]) / 255.
#    black = np.array([0, 0, 0]) / 255.
#    color_gradient = fs.Color_tools.Lch_gradient(gold, black, 200)
#    colormap = fs.Fractal_colormap(color_gradient)
    gold = np.array([255, 210, 66]) / 255.
    black = np.array([0, 0, 0]) / 255.
    colors1 = np.vstack((gold[np.newaxis, :]))
    colors2 = np.vstack((black[np.newaxis, :]))
    colormap = fscolors.Fractal_colormap(kinds="Lch", colors1=colors1,
        colors2=colors2, n=200, funcs=None, extent="clip")

    plotter = fs.Fractal_plotter(
        fractal=mandelbrot,
        base_data_key=potential_data_key, # potential_data_key, #glitch_sort_key,
        base_data_prefix="dev",
        base_data_function=lambda x:x,# np.sin(x*0.0001),
        colormap=colormap,
        probes_val=[0., 0.25],# 200. + 200, #* 428  - 00.,#[0., 0.5, 1.], #phi * k * 2. + k * np.array([0., 1., 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) / 3.5,
        probes_kind="qt",#"z", "qt"
        mask=~cv)
    
    
    #plotter.add_calculation_layer(postproc_key=potential_data_key)
    
#    layer1_key = ("DEM_shade", {"kind": "potential",
#                                "theta_LS": 30.,
#                                "phi_LS": 50.,
#                                "shininess": 3.,
#                                "ratio_specular": 15000.})
#    plotter.add_grey_layer(postproc_key=layer1_key, intensity=0.75, 
#                         blur_ranges=[],#[[0.99, 0.999, 1.0]],
#                        disp_layer=False, #skewness=0.2,
#                         normalized=False, hardness=0.35,  
#            skewness=0.0, shade_type={"Lch": 1.0, "overlay": 1., "pegtop": 4.})
    
    layer2_key = ("field_lines", {})
    plotter.add_grey_layer(postproc_key=layer2_key,
                         hardness=1.0, intensity=0.68, skewness=0.0,
#                         blur_ranges=[[0.50, 0.60, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 2., "pegtop": 1.}) 


    plotter.plot("dev", mask_color=(0., 0., 1.))

if __name__ == "__main__":
    plot()
