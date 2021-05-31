# -*- coding: utf-8 -*-
import numpy as np

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.settings as settings

def plot():
    """
    Example plot of "Dinkydau flake" location, classic test case for 
    perturbation techinque and glitch correction.
    """
    directory = "./perturb2"
    # A simple showcas using perturbation technique
    x, y = "-1.74920463345912691e+00", "-2.8684660237361114e-04"
    dx = "5e-12"
    precision = 16
    nx = 3200


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
            antialiasing=False)

    mandelbrot.calc_std_div(
            complex_type=np.complex128,
            file_prefix="dev",
            subset=None,
            max_iter=50000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            pc_threshold=0.1,
            SA_params={"cutdeg": 8,
                       "cutdeg_glitch": 8,
                       "SA_err": 1.e-4,
                       "use_Taylor_shift": True},
            glitch_eps=1.e-6,
            interior_detect=True,
            glitch_max_attempt=20)

    mandelbrot.run()

    cv = fs.Fractal_Data_array(mandelbrot, file_prefix="dev",
                postproc_keys=('stop_reason', lambda x: x == 1), mode="r+raw")
    potential_data_key = ("potential", {})


    citrus2 = np.array([103, 189, 0]) / 255.
    citrus_white = np.array([252, 251, 226]) / 255.

    wheat1 = np.array([244, 235, 158]) / 255.
    wheat2 = np.array([246, 207, 106]) / 255.
    wheat3 = np.array([191, 156, 96]) / 255.

    lavender1 = np.array([154, 121, 144]) / 255.
    lavender2 = np.array([140, 94, 134]) / 255.
    lavender3 = np.array([18, 16, 58]) / 255.
    

    def wave(x):
        return 0.5 + (0.4 * (x - 0.5) - 0.6 * 0.5 * np.cos(x * np.pi * 3.))

    color_gradient4 = fs.Color_tools.Lch_gradient(citrus_white, wheat2,  200,
                                              f= lambda x: wave(x))
    color_gradient5 = fs.Color_tools.Lch_gradient(wheat2, wheat1,  200,
                                              f= lambda x: wave(x))
    color_gradient6 = fs.Color_tools.Lch_gradient(wheat1, wheat2,  200,
                                              f= lambda x: wave(x))
    color_gradient7 = fs.Color_tools.Lch_gradient(wheat2, wheat1,  200,
                                              f= lambda x: wave(x))
    color_gradient8 = fs.Color_tools.Lch_gradient(wheat1, wheat2, 200,
                                              f= lambda x: wave(x))
    color_gradient9 = fs.Color_tools.Lch_gradient(wheat2, wheat3, 200,
                                              f= lambda x: wave(x))
    color_gradient10 = fs.Color_tools.Lch_gradient(wheat3, wheat1,  200,
                                              f= lambda x: wave(x))
    color_gradient11 = fs.Color_tools.Lch_gradient(wheat1, lavender2,  200,
                                              f= lambda x: wave(x))
    color_gradient12 = fs.Color_tools.Lch_gradient(lavender2, wheat1,  200,
                                              f= lambda x: wave(x))
    color_gradient13 = fs.Color_tools.Lch_gradient(wheat1, wheat2,  200,
                                              f= lambda x: wave(x))
    color_gradient14 = fs.Color_tools.Lch_gradient(wheat2, wheat3, 200,
                                              f= lambda x: wave(x))
    color_gradient15 = fs.Color_tools.Lch_gradient(wheat3, wheat1, 200,
                                              f= lambda x: wave(x))
    color_gradient16 = fs.Color_tools.Lch_gradient(wheat1, lavender2, 200,
                                              f=  lambda x: wave(x))
    color_gradient17 = fs.Color_tools.Lch_gradient(lavender2, wheat1, 200,
                                              f= lambda x: wave(x))
    color_gradient18 = fs.Color_tools.Lch_gradient(wheat1, lavender3, 200,
                                              f= lambda x: wave(x))
    color_gradient19 = fs.Color_tools.Lch_gradient(lavender3, lavender2, 200,
                                              f= lambda x: wave(x))
    color_gradient20 = fs.Color_tools.Lch_gradient(lavender2, lavender3, 200,
                                              f= lambda x: wave(x))
    color_gradient21 = fs.Color_tools.Lch_gradient(lavender3, lavender1, 200,
                                              f= lambda x: wave(x))
    color_gradient22 = fs.Color_tools.Lch_gradient(lavender1, lavender3, 200,
                                              f= lambda x: wave(x))
    color_gradient23 = fs.Color_tools.Lch_gradient(lavender3, lavender2, 200,
                                              f= lambda x: wave(x))
    color_gradient24 = fs.Color_tools.Lch_gradient(lavender2, citrus2, 200,
                                              f= lambda x: wave(x))

    colormap = (fs.Fractal_colormap(color_gradient4)+
                fs.Fractal_colormap(color_gradient5) + 
                fs.Fractal_colormap(color_gradient6) + 
                fs.Fractal_colormap(color_gradient7) + 
                fs.Fractal_colormap(color_gradient8) + 
                fs.Fractal_colormap(color_gradient9) + 
                fs.Fractal_colormap(color_gradient10) + 
                fs.Fractal_colormap(color_gradient11) + 
                fs.Fractal_colormap(color_gradient12) + 
                fs.Fractal_colormap(color_gradient13) + 
                fs.Fractal_colormap(color_gradient14) + 
                fs.Fractal_colormap(color_gradient15) + 
                fs.Fractal_colormap(color_gradient16) + 
                fs.Fractal_colormap(color_gradient17) + 
                fs.Fractal_colormap(color_gradient18) + 
                fs.Fractal_colormap(color_gradient19) + 
                fs.Fractal_colormap(color_gradient20) + 
                fs.Fractal_colormap(color_gradient21) + 
                fs.Fractal_colormap(color_gradient22) + 
                fs.Fractal_colormap(color_gradient23) + 
                fs.Fractal_colormap(color_gradient24) )

    colormap.extent = "mirror" #"repeat"

    plotter = fs.Fractal_plotter(
        fractal=mandelbrot,
        base_data_key=potential_data_key, # potential_data_key, #glitch_sort_key,
        base_data_prefix="dev",
        base_data_function=lambda x:x,# np.sin(x*0.0001),
        colormap=colormap,
        probes_val=np.linspace(0., 1., 22) *0.5,# 200. + 200, #* 428  - 00.,#[0., 0.5, 1.], #phi * k * 2. + k * np.array([0., 1., 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) / 3.5,
        probes_kind="qt",#"z", "qt"
        mask=~cv)
    
    
    #plotter.add_calculation_layer(postproc_key=potential_data_key)
    
    layer1_key = ("DEM_shade", {"kind": "potential",
                                "theta_LS": 30.,
                                "phi_LS": 50.,
                                "shininess": 3.,
                                "ratio_specular": 15000.})
    plotter.add_grey_layer(postproc_key=layer1_key, intensity=0.75, 
                         blur_ranges=[],#[[0.99, 0.999, 1.0]],
                        disp_layer=False, #skewness=0.2,
                         normalized=False, hardness=0.35,  
            skewness=0.0, shade_type={"Lch": 1.0, "overlay": 1., "pegtop": 4.})
    
    layer2_key = ("field_lines", {})
    plotter.add_grey_layer(postproc_key=layer2_key,
                         hardness=1.0, intensity=0.68, skewness=0.4,
                         blur_ranges=[[0.50, 0.60, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 2., "pegtop": 1.}) 


    plotter.plot("dev", mask_color=(0., 0., 1.))

if __name__ == "__main__":
    plot()
