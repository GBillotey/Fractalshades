# -*- coding: utf-8 -*-
import numpy as np
import os
import mpmath

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.settings as settings
import fractalshades.colors as fscolors

# STATUS : solved

settings.enable_multiprocessing = True

fail_dir = os.path.dirname(__file__)
directory = os.path.join(fail_dir, "_data", "fail1")
fractal = fsm.Perturbation_mandelbrot(directory)


# fractal = <fractalshades.models.perturbation_mandelbrot.Perturbation_mandelbrot object at 0x7faec00b9eb0>
file_prefix = 'test'
x = '0.25411832016152502641905073941554840739168805719902'
y = '0.00041824774391017867724304142410708948184586737326948'
dx = '6.609027311521655e-42'
xy_ratio = 1.0
dps = 51
max_iter = 250000
nx = 600

def func(fractal: fsm.Perturbation_mandelbrot= fractal,
         file_prefix: str= "test",
         x: mpmath.mpf= x,
         y: mpmath.mpf= y,
         dx: mpmath.mpf= dx,
         xy_ratio: float=xy_ratio,
         dps: int= dps,
         max_iter: int=max_iter,
         nx: int=nx):#
#             interior_detect: bool=True):

    interior_detect = False # True
    
#    fractal.clean_up(file_prefix)
    
    fractal.zoom(precision=dps, x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
         theta_deg=0., projection="cartesian", antialiasing=False)
    fractal.calc_std_div(complex_type=np.complex128, file_prefix=file_prefix,
        subset=None, max_iter=max_iter, M_divergence=1.e3,
        epsilon_stationnary=1.e-4, pc_threshold=0.1,
        SA_params={"cutdeg": 8,
                   "cutdeg_glitch": 8,
                   "SA_err": 1.e-20, # "SA_err" 1e-6, "cutdeg": 8 fails
                                     # "SA_err" 1e-20, "cutdeg": 8 fails
                                     # "SA_err" 1e-20, "cutdeg": 64 fails
                                     # "SA_err" 1e-50, "cutdeg": 64 fails
                                     # "SA_err" 1e-200, "cutdeg": 64
                   "use_Taylor_shift": True},
        glitch_eps=1.e-6, interior_detect=interior_detect,
        glitch_max_attempt=20)
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
        probes_val=[0.25, 0.75],
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

if __name__ == "__main__":
    func()

