# -*- coding: utf-8 -*-
import numpy as np

import fractalshades as fs
import fractalshades.models as fsm

def plot():
    """
    Dev
    """
    directory = "./flake"

    # Dinkydau flake
    # http://www.fractalforums.com/announcements-and-news/pertubation-theory-glitches-improvement/msg73027/#msg73027
    # Ball method 1 found period: 7884
    x = "-1.99996619445037030418434688506350579675531241540724851511761922944801584242342684381376129778868913812287046406560949864353810575744772166485672496092803920095332"
    y = "0.00000000000000000000000000000000030013824367909383240724973039775924987346831190773335270174257280120474975614823581185647299288414075519224186504978181625478529"
    dx = "1.1e-157"
    precision = 200
    
    nx = 800
    xy_ratio = 0.5 
    theta_deg = 0.
    complex_type = ("Xrange", np.complex64)

    mandelbrot = fsm.Perturbation_mandelbrot(
                 directory, x, y, dx, nx, xy_ratio, theta_deg, chunk_size=200,
                 complex_type=complex_type, projection="cartesian",
                 precision=precision)
    mandelbrot.full_loop(
            file_prefix="dev",
            subset=None,
            max_iter=50000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            pc_threshold=0.1,
            SA_params={"cutdeg": 64, "cutdeg_glitch": 8},
            glitch_eps=1.e-6,
            interior_detect=False)

    glitched = fs.Fractal_Data_array(mandelbrot, file_prefix="dev",
                postproc_keys=('stop_reason', lambda x: x >= 3), mode="r+raw")

    potential_data_key = ("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3})
    

    

    citrus2 = np.array([103, 189, 0]) / 255.
    wheat1 = np.array([244, 235, 158]) / 255.
    wheat2 = np.array([246, 207, 106]) / 255.
    wheat3 = np.array([191, 156, 96]) / 255.
    lavender1 = np.array([154, 121, 144]) / 255.
    lavender2 = np.array([140, 94, 134]) / 255.
    lavender3 = np.array([18, 16, 58]) / 255.

    def wave(x):
        return 0.5 + (0.4 * (x - 0.5) - 0.6 * 0.5 * np.cos(x * np.pi * 3.))
        
    color_gradient1 = fs.Color_tools.Lch_gradient(wheat1, wheat2, 200,
                                              f= lambda x: wave(x))#x**2 * (3. - 2.*x))
    color_gradient2 = fs.Color_tools.Lch_gradient(wheat2, wheat1,  200,
                                              f= lambda x: wave(x))
    color_gradient3 = fs.Color_tools.Lch_gradient(wheat1, wheat2, 200,
                                              f= lambda x: wave(x))
    color_gradient4 = fs.Color_tools.Lch_gradient(wheat1, wheat2,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient5 = fs.Color_tools.Lch_gradient(wheat2, wheat1,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient6 = fs.Color_tools.Lch_gradient(wheat1, wheat2,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient7 = fs.Color_tools.Lch_gradient(wheat2, wheat1,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient8 = fs.Color_tools.Lch_gradient(wheat1, wheat2, 200,
                                              f= lambda x: wave(x)) #**1.)
    
    color_gradient9 = fs.Color_tools.Lch_gradient(wheat2, wheat3, 200,
                                              f= lambda x: wave(x)) #**1.)
    
    color_gradient10 = fs.Color_tools.Lch_gradient(wheat3, wheat1,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient11 = fs.Color_tools.Lch_gradient(wheat1, lavender2,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient12 = fs.Color_tools.Lch_gradient(lavender2, wheat1,  200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient13 = fs.Color_tools.Lch_gradient(wheat1, wheat2,  200,
                                              f= lambda x: wave(x)) #**1.)
    
    
    color_gradient14 = fs.Color_tools.Lch_gradient(wheat2, wheat3, 200,
                                              f= lambda x: wave(x))
    color_gradient15 = fs.Color_tools.Lch_gradient(wheat3, wheat1, 200,
                                              f= lambda x: wave(x))
    color_gradient16 = fs.Color_tools.Lch_gradient(wheat1, lavender1, 200,
                                              f=  lambda x: wave(x))
    color_gradient17 = fs.Color_tools.Lch_gradient(lavender1, wheat1, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient18 = fs.Color_tools.Lch_gradient(wheat1, lavender3, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient19 = fs.Color_tools.Lch_gradient(lavender3, lavender2, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient20 = fs.Color_tools.Lch_gradient(lavender2, lavender3, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient21 = fs.Color_tools.Lch_gradient(lavender3, lavender1, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient22 = fs.Color_tools.Lch_gradient(lavender1, lavender3, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient23 = fs.Color_tools.Lch_gradient(lavender3, lavender2, 200,
                                              f= lambda x: wave(x)) #**1.)
    color_gradient24 = fs.Color_tools.Lch_gradient(lavender2, citrus2, 200,
                                              f= lambda x: wave(x)) #**1.)

    
    colormap = (fs.Fractal_colormap(color_gradient4)+#, plt.get_cmap("magma")) + Fractal_colormap((0.1, 1.0, 200), plt.get_cmap("magma"))
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
    
#    colormap = Fractal_colormap((0.95, 0.15, 200), plt.get_cmap("magma"))
#    colormap = Fractal_colormap(color_gradient2c)
    colormap.extent = "mirror" #"repeat"

    plotter = fs.Fractal_plotter(
        fractal=mandelbrot,
        base_data_key=potential_data_key, # potential_data_key, #glitch_sort_key,
        base_data_prefix="dev",
        base_data_function=lambda x:x,# np.sin(x*0.0001),
        colormap=colormap,
        probes_val=np.linspace(0., 1., 22)**0.8, #* 428  - 00.,#[0., 0.5, 1.], #phi * k * 2. + k * np.array([0., 1., 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) / 3.5,
        probes_kind="qt",#"z",
        mask=glitched)
    
    
    #plotter.add_calculation_layer(postproc_key=potential_data_key)
    
    layer1_key = ("DEM_shade", {"kind": "potential",
                                "theta_LS": 50.,
                                "phi_LS": 40.,
                                "shininess": 40.,
                                "ratio_specular": 4000.})
    plotter.add_grey_layer(postproc_key=layer1_key, intensity=0.95, 
                         blur_ranges=[],#[[0.99, 0.999, 1.0]],
                        disp_layer=False,
                         normalized=False, hardness=0.15,  
            skewness=0.0, shade_type={"Lch": 1.0, "overlay": 0., "pegtop": 4.})
    
    layer2_key = ("field_lines", {})
    plotter.add_grey_layer(postproc_key=layer2_key,
                         hardness=0.5, intensity=0.38,
                         blur_ranges=[],#[[0.99, 0.99, 1.0]], 
                         shade_type={"Lch": 0., "overlay": 2., "pegtop": 0.}) 
#

#
    plotter.plot("dev", mask_color=(0., 0., 1.))

if __name__ == "__main__":
    plot()
