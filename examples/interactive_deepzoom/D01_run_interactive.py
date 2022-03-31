# -*- coding: utf-8 -*-
"""
=============================================
D01 - Mandelbrot arbitrary-precision explorer
=============================================

This is a template to explore the Mandelbrot set with
arbitrary precision through a GUI.
It features the main postprocessing options (continuous
iteration, distance estimation based shading, field-lines)

Good exploration !


Reference:
`fractalshades.models.Perturbation_mandelbrot`
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
    DEM_pp,
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
    M_divergence = 1.e2
    nx = 800
    theta_deg = 0.
    interior_detect = True
    epsilon_stationnary = 0.0001
    eps = 1.e-6
    
    base_layer = "continuous_iter"
    colormap = fscolors.cmap_register["classic"]
    cmap_z_kind = "relative"
    zmin = 0.00
    zmax = 0.50
    
    shade_kind="glossy"
    field_kind="None"

    # Set to True to enable multi-threading
    settings.enable_multithreading = True
    settings.no_newton = False

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
         M_divergence: float=M_divergence,
         interior_detect: bool=interior_detect,
         epsilon_stationnary: float=epsilon_stationnary,

         _3: fsgui.separator="Bilinear series parameters",
         eps: float=eps,

         _4: fsgui.separator="Plotting parameters: base field",
         base_layer: typing.Literal[
                 "continuous_iter",
                 "distance_estimation"
         ]=base_layer,
         interior_color: QtGui.QColor=(0.1, 0.1, 0.1),
         colormap: fscolors.Fractal_colormap=colormap,
         invert_cmap: bool=False,
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
                M_divergence=M_divergence,
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

        pp = Postproc_batch(fractal, calc_name)
        
        if base_layer == "continuous_iter":
            pp.add_postproc(base_layer, Continuous_iter_pp())
        elif base_layer == "distance_estimation":
            pp.add_postproc("continuous_iter", Continuous_iter_pp())
            pp.add_postproc(base_layer, DEM_pp())

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

        if base_layer != 'continuous_iter':
            plotter.add_layer(
                Virtual_layer("continuous_iter", func=None, output=False)
            )

        sign = {False: 1., True: -1.}[invert_cmap]
        plotter.add_layer(Color_layer(
                base_layer,
                func=lambda x: sign * np.log(x),
                colormap=colormap,
                probes_z=[zmin, zmax],
                probes_kind=cmap_z_kind,
                output=True))
        plotter[base_layer].set_mask(
            plotter["interior"], mask_color=interior_color
        )
        if field_kind == "twin":
            plotter[base_layer].set_twin_field(plotter["fieldlines"],
                   twin_intensity)
        elif field_kind == "overlay":
            overlay_mode = Overlay_mode("tint_or_shade", pegtop=1.0)
            plotter[base_layer].overlay(plotter["fieldlines"], overlay_mode)

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
    
            plotter[base_layer].shade(plotter["DEM_map"], light)

        plotter.plot()
        
        # Renaming output to match expected from the Fractal GUI
        layer = plotter[base_layer]
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