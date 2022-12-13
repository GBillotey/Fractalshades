# -*- coding: utf-8 -*-
"""
==================================
01 - Full Mandelbrot basic example
==================================

This basic example shows how to create a color layer, displaying the 
"continuous iteration number" for Mandelbrot (power 2) fractal.

The full Mandelbrot set is diplayed here.

Reference:
`fractalshades.models.Mandelbrot`
"""
import os

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.colors as fscolors
from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    Raw_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
)

def plot(plot_dir):
    """
    A very simple example: full view of the Mandelbrot set with escape-time
    coloring
    """
    fs.settings.enable_multithreading = True
    # Define the parameters for this calculation
    x = -1.0
    y = -0.0
    dx = 5.
    nx = 2400
    
    calc_name="mandelbrot"
    colormap = fscolors.cmap_register["classic"]

    # Run the calculation
    f = fsm.Mandelbrot(plot_dir)
    f.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0,
           theta_deg=0., projection="cartesian")
    f.calc_std_div(
        calc_name=calc_name,
        subset=None,
        max_iter=1000,
        M_divergence=1000.,
        epsilon_stationnary= 0.001,
    )

    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("cont_iter", Continuous_iter_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))

    plotter = fs.Fractal_plotter(pp)
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Color_layer(
            "cont_iter",
            func="np.log(x)",
            colormap=colormap,
            probes_z=[1., 3.],
            output=True
    ))

    plotter["cont_iter"].set_mask(
            plotter["interior"],
            mask_color=(0.1, 0.1, 0.1)
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
            fs.utils.exec_no_output(plot, plot_dir)
