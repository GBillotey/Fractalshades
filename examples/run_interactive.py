# -*- coding: utf-8 -*-
import numpy as np
import os

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.settings as settings
import fractalshades.colors as fscolors
import fractalshades.gui as fsgui

def plot():
    """
    Example interactive
    """
    import mpmath
#    import numpy as np
#    import fractalshades as fs
#    import fractalshades.models as fsm
#    import fractalshades.colors as fscolors
    x = '-1.0'
    y = '-0.0'
    dx = '5.0'
    
#    x = '-1.24710405741042405360962651098294394779517220029152023311484191548274'
#    y = '0.404377520626112573015438541889013872729238186318787005674541172744775'
#    dx = '5.709248399870929e-64'
#    
##    fractal = <fractalshades.models.mandelbrot.Perturbation_mandelbrot object at 0x7f43b55a3970>
##file_prefix = 'test'
#    x = '-1.24710405741042405360962651098294394779517220029152023311484191535399137570594560782611307742355103703450012792393523736208965581219742075615943643313'
#    y = '0.404377520626112573015438541889013872729238186318787005674541172723337681034735258655310637272768389756452645231222132046734930367073008504918539074049'
#    dx = '1.35414499422036e-144'
#xy_ratio = 1.0
#dps = 150
    max_iter = 2500
#nx = 3200
#    
    
#    x = '-1.7413792275517676072893904237743226'
#    y = '0.00035405344491501011658346057198089323'
#    dx = '4.781449761087386e-29'
#    
#    x = '-1.74137922755176760728939042377712255'
#    y = '0.000354053444915010116583460574285668865'
#    dx = '1.715986446020114e-29'
#    
#    x = "-1.785105124108500275342045589035133634602693041579915280419774680163030730526246759348250007296604082"
#    y = "0.000000000000000000000000005326380701673882244449706206785580700667649125301227944428359283401632060406306534343857597559690475"
#    dx = "1.238826247496661e-25"
#xy_ratio = 1.0
#dps = 80
#max_iter = 100000
#nx = 600
    
#    x = '-1.7413792275517676072893904237737925222'
#    y = '0.00035405344491501011658346057080945946209'
#    dx = '8.409106050193e-34'
#    xy_ratio = 1.0
#    dps = 1100
#    max_iter = 100000
#    nx = 600
    # Set to True to enable multi-processing
    settings.enable_multiprocessing = True
    
    test_dir = os.path.dirname(__file__)
    directory = os.path.join(test_dir, "localtest_GUI")
    fractal = fsm.Perturbation_mandelbrot(directory)
    
    def func(fractal: fsm.Perturbation_mandelbrot= fractal,
             file_prefix: str= "test",
             x: mpmath.mpf= x,
             y: mpmath.mpf= y,
             dx: mpmath.mpf= dx,
             xy_ratio: float=1.0,
             dps: int= 100,
             max_iter: int=max_iter,
             nx: int=600):#
#             interior_detect: bool=True):

        interior_detect = False # True
        
        fractal.zoom(precision=dps, x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=0., projection="cartesian", antialiasing=False)
        fractal.calc_std_div(complex_type=np.complex128, file_prefix=file_prefix,
            subset=None, max_iter=max_iter, M_divergence=1.e3,
            epsilon_stationnary=1.e-4, pc_threshold=0.1,
            SA_params={"cutdeg": 8,
                       "cutdeg_glitch": 8,
                       "SA_err": 1.e-6,
                       "use_Taylor_shift": True},
            glitch_eps=1.e-6, interior_detect=interior_detect, glitch_max_attempt=20)
        fractal.run()
        
        
        gold = np.array([255, 210, 66]) / 255.
        black = np.array([0, 0, 0]) / 255.
        purple = np.array([181, 40, 99]) / 255.
        citrus2 = np.array([103, 189, 0]) / 255.
        colors1 = np.vstack((purple[np.newaxis, :]))
        colors2 = np.vstack((gold[np.newaxis, :]))
        colormap = fscolors.Fractal_colormap(kinds="Lch", colors1=colors1,
            colors2=colors2, n=200, funcs=None, extent="mirror")
        
        mask_codes = [2]#, 3, 4]
        mask = fs.Fractal_Data_array(fractal, file_prefix=file_prefix,
            postproc_keys=('stop_reason', lambda x: np.isin(x, mask_codes)),
            mode="r+raw")


        plotter = fs.Fractal_plotter(fractal=fractal,
            base_data_key=("potential", {}), # ("field_lines", {"n_iter": 10, "swirl": 1.}), ,
            base_data_prefix=file_prefix,
            base_data_function=lambda x:x,
            colormap=colormap,
            probes_val=[0.25, 0.35],
            probes_kind="qt",
            mask=mask)
       #  plotter.add_calculation_layer(("potential", {}))
        plotter.add_grey_layer(
                postproc_key=("DEM_shade", {"kind": "potential",
                                "theta_LS": 30.,
                                "phi_LS": 50.,
                                "shininess": 30.,
                                "ratio_specular": 15000.}),
                blur_ranges=[],
                hardness=0.9,
                intensity=0.8,
                shade_type={"Lch": 1.0, "overlay": 0., "pegtop": 4.})
        plotter.plot(file_prefix, mask_color=(0., 0., 1.))
#             x: =,
#             y: =):
        
#    view = "test"
    # func()
    
    gui = fsgui.Fractal_GUI(func) #, fractal_param="fractal")
    gui.connect_image(image_param="file_prefix")#=image_prefix)
    gui.connect_mouse(x="x", y="y", dx="dx", xy_ratio="xy_ratio", dps="dps")
    gui.show()

if __name__ == "__main__":
    plot()
