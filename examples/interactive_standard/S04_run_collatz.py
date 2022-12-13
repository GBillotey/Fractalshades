# -*- coding: utf-8 -*-
"""
=============================================================
S04 - Collatz explorer - Standard precision
=============================================================

This is a simple template to explore the Collatz set with
a GUI (example of "low-level API" for GUI implementation).
Resolution limited to approx 1.e-13 due to double (float64) precision

Reference:
`fractalshades.models.Collatz`
"""
import os

import numpy as np

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.settings as settings
import fractalshades.colors
import fractalshades.gui as fsgui
from fractalshades.gui.guimodel import Fractal_GUI

from fractalshades.postproc import (
    Postproc_batch,
    Raw_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
)


def plot(plot_dir):
    """
    Example of "low-level API" for GUI implementation
    """
    calc_name = 'test'
    x = 0.0
    y = 0.0
    dx = 8.
    xy_ratio = 1.7777
    theta_deg = 0.

    max_iter = 1000
    nx = 800
    M_divergence = 1000.0
    interior_color = (0.0, 0.0, 1.0)
    colormap = fs.colors.cmap_register["classic"]
    zmin = 0.0
    zmax = 0.5

    # Set to True to enable multi-processing
    settings.enable_multithreading = True
    # Set to True in case RAM issue (Memory error)
    settings.optimize_RAM = False
    settings.inspect_calc = True

    directory = plot_dir
    fractal = fsm.Collatz(directory)
    
    def func(
        fractal: fsm.Mandelbrot=fractal,
        calc_name: str= calc_name,
         _1: fsgui.separator="Zoom parameters",
         x: float= x,
         y: float= y,
         dx: float= dx,
         xy_ratio: float=xy_ratio,
         theta_deg: float=theta_deg,
         _2: fsgui.separator="Calculation parameters",
         max_iter: int=max_iter,
         nx: int=nx,
         _3: fsgui.separator="Plotting parameters",
         M_divergence: float=M_divergence,
         interior_color: fs.colors.Color=interior_color,
         colormap: fs.colors.Fractal_colormap=colormap,
         zmin: float=zmin,
         zmax: float=zmax,
         zshift: float=0.0
    ):


        fractal.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=0., projection="cartesian")

        fractal.base_calc(
            calc_name=calc_name,
            subset=None,
            max_iter=max_iter,
            M_divergence=M_divergence,
            epsilon_stationnary=1.e-4,
        )

        pp = Postproc_batch(fractal, calc_name)
        pp.add_postproc("n_iter", Raw_pp("stop_iter", func=None))
        pp.add_postproc("interior", Raw_pp("stop_reason",
                        func=lambda x: x != 1))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))

        plotter.add_layer(Color_layer(
                "n_iter",
                func=lambda x: np.log(x + 10.),
                colormap=colormap,
                probes_z=[zmin + zshift, zmax + zshift],
                output=True))
        plotter["n_iter"].set_mask(
            plotter["interior"], mask_color=interior_color
        )

        plotter.plot()

        # Renaming output to match expected from the Fractal GUI
        layer = plotter["n_iter"]
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        src_path = os.path.join(fractal.directory, file_name + ".png")
        dest_path = os.path.join(fractal.directory, calc_name + ".png")
        if os.path.isfile(dest_path):
            os.unlink(dest_path)
        os.link(src_path, dest_path)


    gui = Fractal_GUI(func)
    gui.connect_image(image_param="calc_name")
    gui.connect_mouse(x="x", y="y", dx="dx", xy_ratio="xy_ratio", dps=None)
    gui.show()


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
