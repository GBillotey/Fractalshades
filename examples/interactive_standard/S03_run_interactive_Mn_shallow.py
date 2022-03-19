# -*- coding: utf-8 -*-
"""
========================================================================
Mandelbrot power n explorer - Naïve algorithm with standard precision
========================================================================

This is a simple template to start exploring the power n > 2 Mandelbrot
set with the GUI.
Resolution limited to approx 1.e-13 due to double (float64) precision
"""
import typing
import os

import numpy as np
from PyQt6 import QtGui

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.settings as settings
import fractalshades.colors as fscolors
import fractalshades.gui as fsgui

from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    DEM_normal_pp,
    Raw_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Blinn_lighting
)


def plot(plot_dir):
    """
    Example interactive
    """
    calc_name = 'test'
    deg = 5
    x = -0.0
    y = -0.0
    dx = 5.0
    xy_ratio = 1.0
    theta_deg=0.

    max_iter = 15000
    nx = 800
    epsilon_stationnary = 0.0001
    
    colormap = fscolors.cmap_register["classic"]

    zmin = 0.00
    zmax = 0.15

    # Set to True to enable multi-processing
    settings.enable_multiprocessing = True
    # Set to True in case RAM issue (Memory error)
    settings.optimize_RAM = False

    directory = plot_dir
    fractal = fsm.Mandelbrot_N(directory, deg)
    
    def func(
        fractal: fsm.Mandelbrot_N=fractal,
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
         epsilon_stationnary: float=epsilon_stationnary,
         _3: fsgui.separator="Plotting parameters",
         interior_color: QtGui.QColor=(0.1, 0.1, 0.1),
         colormap: fscolors.Fractal_colormap=colormap,
         cmap_z_kind: typing.Literal["relative", "absolute"]="relative",
         zmin: float=zmin,
         zmax: float=zmax
    ):

        fractal.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=0., projection="cartesian", antialiasing=False)

        fractal.base_calc(
            calc_name=calc_name,
            subset=None,
            max_iter=max_iter,
            M_divergence=1.e3,
            epsilon_stationnary=epsilon_stationnary,
        )

        if fractal.res_available():
            print("RES AVAILABLE, no compute")
        else:
            print("RES NOT AVAILABLE, clean-up")
            fractal.clean_up(calc_name)

        fractal.run()

        layer_name = "continuous_iter"

        pp = Postproc_batch(fractal, calc_name)
        pp.add_postproc(layer_name, Continuous_iter_pp())
        pp.add_postproc("interior", Raw_pp("stop_reason",
                        func=lambda x: x != 1))
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=45, output=True))
        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[zmin, zmax],
                probes_kind="relative",
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                     mask_color=interior_color)

        light = Blinn_lighting(0.2, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=1.05,
            k_specular=.0,
            shininess=350.,
            angles=(50., 50.),
            coords=None,
            color=np.array([1.0, 1.0, 0.9]))

        plotter[layer_name].shade(plotter["DEM_map"], light)
        plotter.plot()
        
        # Renaming output to match expected from the Fractal GUI
        layer = plotter[layer_name]
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        src_path = os.path.join(fractal.directory, file_name + ".png")
        dest_path = os.path.join(fractal.directory, calc_name + ".png")
        if os.path.isfile(dest_path):
            os.unlink(dest_path)
        os.link(src_path, dest_path)


    gui = fsgui.Fractal_GUI(func)
    gui.connect_image(image_param="calc_name")
    gui.connect_mouse(dps=None)#x="x", y="y", dx="dx", xy_ratio="xy_ratio",
                      # dps=None)
    gui.show()


def _plot_from_data(plot_dir, static_im_link):
    # Private function only used when building fractalshades documentation
    # Output from GUI might fail for the runner building the doc on github.
    # -> Defaulting to a static image if one is provided
    import PIL
    import PIL.PngImagePlugin
    data_path = fs.settings.output_context["doc_data_dir"]
    im = PIL.Image.open(os.path.join(data_path, static_im_link))
    rgb_im = im.convert('RGB')
    tag_dict = {"Software": "fractalshades " + fs.__version__,
                "GUI_plot": static_im_link}
    pnginfo = PIL.PngImagePlugin.PngInfo()
    for k, v in tag_dict.items():
        pnginfo.add_text(k, str(v))
    if fs.settings.output_context["doc"]:
        fs.settings.add_figure(fs._Pillow_figure(rgb_im, pnginfo))
    else:
        # Should not happen
        raise RuntimeError()


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
            static_im_link=None
            if static_im_link is None:
                plot(plot_dir)
            else:
                _plot_from_data(plot_dir, static_im_link)
