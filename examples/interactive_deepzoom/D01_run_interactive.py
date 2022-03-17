# -*- coding: utf-8 -*-
"""
=======================================
Mandelbrot arbitrary-precision explorer
=======================================

This is a template to start exploring the Mandelbrot set with
arbitrary precision through a GUI.
It features the main postprocessing options (continuous
iteration, distance estimation based shading, field-lines)

Good exploration !
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
    Fieldlines_pp,
    DEM_normal_pp,
    Raw_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Grey_layer,
    Virtual_layer,
    Overlay_mode,
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
    dps = 16
    max_iter = 15000
    nx = 600
    theta_deg = 0.
    interior_detect = True
    epsilon_stationnary = 0.0001
    eps = 1.e-6
    
    colormap = fscolors.cmap_register["classic"]
    cmap_z_kind = "relative"
    zmin = 0.00
    zmax = 0.50
    
    shade_kind="glossy"
    field_kind="None"

    # Set to True to enable multi-threading
    settings.enable_multithreading = True

    directory = plot_dir
    fractal = fsm.Perturbation_mandelbrot(directory)
    
    def func(
        fractal: fsm.Perturbation_mandelbrot=fractal,
         calc_name: str=calc_name,
         _1: fsgui.separator="Zoom parameters",
         x: mpmath.mpf=x,
         y: mpmath.mpf=y,
         dx: mpmath.mpf=dx,
         xy_ratio: float=xy_ratio,
         theta_deg: float=theta_deg,
         dps: int=dps,
         nx: int=nx,
         _2: fsgui.separator="Calculation parameters",
         max_iter: int=max_iter,
         interior_detect: bool=interior_detect,
         epsilon_stationnary: float=epsilon_stationnary,
         _3: fsgui.separator="Bilinear series parameters",
         eps: float=eps,
         _4: fsgui.separator="Plotting parameters: continuous iteration",
         interior_color: QtGui.QColor=(0.1, 0.1, 0.1),
         colormap: fscolors.Fractal_colormap=colormap,
         cmap_z_kind: typing.Literal["relative", "absolute"]=cmap_z_kind,
         zmin: float=zmin,
         zmax: float=zmax,
         _5: fsgui.separator="Plotting parameters: shading",
         shade_kind: typing.Literal["None", "standard", "glossy"]=shade_kind,
         gloss_intensity: float=10.,
         light_angle_deg: float=65.,
         light_color: QtGui.QColor=(1.0, 1.0, 1.0),
         gloss_light_color: QtGui.QColor=(1.0, 1.0, 1.0),
         _6: fsgui.separator="Plotting parameters: field lines",
         field_kind: typing.Literal["None", "overlay", "twin"]=field_kind,
         n_iter: int=3,
         swirl: float=0.,
         damping_ratio: float=0.8,
         twin_intensity: float=0.1
    ):


        fractal.zoom(precision=dps, x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=theta_deg, projection="cartesian", antialiasing=False)

        fractal.calc_std_div(
                calc_name=calc_name,
                subset=None,
                max_iter=max_iter,
                M_divergence=1.e3,
                epsilon_stationnary=1.e-3,
                SA_params=None,
                BLA_params={
                    "eps": eps
                },
                interior_detect=interior_detect,
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

        if field_kind != "None":
            pp.add_postproc(
                "fieldlines",
                Fieldlines_pp(n_iter, swirl, damping_ratio)
            )
        pp.add_postproc("interior", Raw_pp("stop_reason",
                        func=lambda x: x != 1))
        if shade_kind != "None":
            pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))

        if field_kind == "twin":
            plotter.add_layer(Virtual_layer(
                    "fieldlines", func=None, output=False
            ))
        elif field_kind == "overlay":
            plotter.add_layer(Grey_layer(
                    "fieldlines", func=None, output=False
            ))

        if shade_kind != "None":
            plotter.add_layer(Normal_map_layer(
                "DEM_map", max_slope=60, output=True
            ))

        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(x),
                colormap=colormap,
                probes_z=[zmin, zmax],
                probes_kind=cmap_z_kind,
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                     mask_color=interior_color)

        if field_kind == "twin":
            plotter[layer_name].set_twin_field(plotter["fieldlines"],
                   twin_intensity)
        elif field_kind == "overlay":
            overlay_mode = Overlay_mode("tint_or_shade", pegtop=1.0)
            plotter[layer_name].overlay(plotter["fieldlines"], overlay_mode)

        if shade_kind != "None":
            light = Blinn_lighting(0.4, np.array([1., 1., 1.]))
            light.add_light_source(
                k_diffuse=0.8,
                k_specular=.0,
                shininess=350.,
                angles=(light_angle_deg, 20.),
                coords=None,
                color=np.array(light_color))
    
            if shade_kind == "glossy":
                light.add_light_source(
                    k_diffuse=0.2,
                    k_specular=gloss_intensity,
                    shininess=1400.,
                    angles=(light_angle_deg, 20.),
                    coords=None,
                    color=np.array(gloss_light_color))
    
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
            static_im_link = "Screenshot_from_2022-02-04.png"
            if static_im_link is None:
                plot(plot_dir)
            else:
                _plot_from_data(plot_dir, static_im_link)
