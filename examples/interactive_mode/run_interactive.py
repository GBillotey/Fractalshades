# -*- coding: utf-8 -*-
"""
=======================================
Mandelbrot arbitrary-precision explorer
=======================================

This is a simple template to start exploring the Mandelbrot set with
the GUI.
Good exploration !
"""
import typing
import os

import numpy as np
from PyQt5 import QtGui

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
    import mpmath

    x = '-1.0'
    y = '-0.0'
    dx = '5.0'
    calc_name = 'test'
    
    xy_ratio = 1.0
    dps = 77
    max_iter = 150000
    nx = 800
    interior_detect = True
    epsilon_stationnary = 0.0001
    
    colormap = fscolors.cmap_register["classic"]

    zmin = 0.00
    zmax = 0.15

    # Set to True to enable multi-processing
    settings.enable_multiprocessing = True

    directory = plot_dir
    fractal = fsm.Perturbation_mandelbrot(directory)
    
    def func(fractal: fsm.Perturbation_mandelbrot=fractal,
             calc_name: str= calc_name,
             x: mpmath.mpf= x,
             y: mpmath.mpf= y,
             dx: mpmath.mpf= dx,
             xy_ratio: float=xy_ratio,
             dps: int= dps,
             max_iter: int=max_iter,
             nx: int=nx,
             interior_detect: bool=interior_detect,
             epsilon_stationnary: float=epsilon_stationnary,
             interior_color: QtGui.QColor=(0.1, 0.1, 0.1),
             colormap: fscolors.Fractal_colormap=colormap,
             cmap_z_kind: typing.Literal["relative", "absolute"]="relative",
             zmin: float=zmin,
             zmax: float=zmax):


        fractal.zoom(precision=dps, x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=0., projection="cartesian", antialiasing=False)

        fractal.calc_std_div(datatype=np.complex128, calc_name=calc_name,
            subset=None, max_iter=max_iter, M_divergence=1.e3,
            epsilon_stationnary=1.e-4,
            SA_params={"cutdeg": 8,
                       "SA_err": 1.e-6,
                       "cutdeg_glitch": 8,
                       "use_Taylor_shift": False},
            glitch_eps=1.e-6,
            interior_detect=interior_detect,
            glitch_max_attempt=10)

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
    gui.connect_mouse(x="x", y="y", dx="dx", xy_ratio="xy_ratio", dps="dps")
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
            static_im_link="sphx_glr_run_interactive_001.png"
            if static_im_link is None:
                plot(plot_dir)
            else:
                _plot_from_data(plot_dir, static_im_link)
