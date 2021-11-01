# -*- coding: utf-8 -*-
"""
=============================
DEM example with perturbation
=============================

This example shows how to create a color layer, displaying the 
distance estimation for Mandelbrot (power 2) fractal.

The location, at 16.e-22, is below the reach of double, pertubation theory must
be used.
"""

import os
import numpy as np

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.settings as settings
import fractalshades.colors as fscolors


from fractalshades.postproc import (
    Postproc_batch,
    DEM_pp,
    Continuous_iter_pp,
    Raw_pp,
    DEM_normal_pp,
    # Fieldlines_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    # Blinn_lighting,
    # Normal_map_layer,
    Virtual_layer,
)

def plot(directory):
    """
    Example plot of distance estimation method
    """
    # A simple showcas using perturbation technique
    x = "-1.768667862837488812627419470"
    y = "0.001645580546820209430325900"
    dx = "16.e-22"
    precision = 30
    nx = 1600

    colormap = fscolors.cmap_register["autumn"]

    # Set to True if you only want to rerun the post-processing part
    settings.skip_calc = False
    # Set to True to enable multi-processing
    settings.enable_multiprocessing = True


    f = fsm.Perturbation_mandelbrot(directory)
    f.zoom(precision=precision,
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=1.0,
            theta_deg=0., 
            projection="cartesian",
            antialiasing=False)

    f.calc_std_div(
            datatype=np.complex128,
            calc_name="div",
            subset=None,
            max_iter=50000, #00,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            SA_params={"cutdeg": 8,
                       "cutdeg_glitch": 8,
                       "SA_err": 1.e-4},
            glitch_eps=1.e-6,
            interior_detect=True,
            glitch_max_attempt=2)

    f.run()
    
    # Plot the image
    pp = Postproc_batch(f, "div")
    pp.add_postproc("potential", Continuous_iter_pp())
    pp.add_postproc("DEM", DEM_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
    
    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Virtual_layer("potential", func=None, output=False))
    plotter.add_layer(Color_layer(
            "DEM",
            func="x",
            colormap=colormap,
            probes_z=[0.5, 1.5],
            probes_kind="relative",
            output=True
    ))
    plotter["DEM"].set_mask(
            plotter["interior"],
            mask_color=(0., 0., 0.)
    )
    
    plotter.plot()

if __name__ == "__main__":
    # Some magic to get the directory for plotting: with a name that matches
    # the file or a temporary dir if we are building the documentation
    try:
        realpath = os.path.realpath(__file__)
        plot_dir = os.path.splitext(realpath)[0]
        plot(plot_dir)
    except NameError:
        import tempfile
        with tempfile.TemporaryDirectory() as plot_dir:
            plot(plot_dir)