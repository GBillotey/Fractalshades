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
#from PyQt6 import QtGui

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
    epsilon_stationnary = 0.001
    eps = 1.e-6
    
    base_layer = "continuous_iter"
    colormap = fscolors.cmap_register["classic"]
    lighting = fscolors.lighting_register["glossy"]

    zmin = 0.0
    zmax = 5.0

    field_kind="overlay"

    # Set to True to enable multi-threading
    settings.enable_multithreading = True

    directory = plot_dir
    fractal = fsm.Perturbation_mandelbrot(directory)
    
    def func(
        fractal: fsm.Perturbation_mandelbrot=fractal,
        calc_name: str=calc_name,

        _1: fsgui.collapsible_separator="Zoom parameters",
        x: mpmath.mpf=x,
        y: mpmath.mpf=y,
        dx: mpmath.mpf=dx,
        xy_ratio: float=xy_ratio,
        theta_deg: float=theta_deg,
        dps: int=dps,
        nx: int=nx,

        _2: fsgui.collapsible_separator="Calculation parameters",
        max_iter: int=max_iter,
        M_divergence: float=M_divergence,
        interior_detect: bool=interior_detect,
        epsilon_stationnary: float=epsilon_stationnary,

        _3: fsgui.collapsible_separator="Bilinear series parameters",
        use_BLA: bool=True,
        eps: float=eps,

        _4: fsgui.collapsible_separator="Plotting parameters: base field",
        base_layer: typing.Literal[
                 "continuous_iter",
                 "distance_estimation"
        ]=base_layer,
        interior_mask: typing.Literal[
                 "all",
                 "not_diverging",
                 "dzndz_detection",
        ]="all",
        colormap: fscolors.Fractal_colormap=colormap,
        invert_cmap: bool=False,
        zmin: float=zmin,
        zmax: float=zmax,

        _5: fsgui.collapsible_separator="Plotting parameters: shading",
        has_shading: bool=False,
        lighting: fscolors.Blinn_lighting=lighting,
        max_slope: float = 60.,

        _6: fsgui.collapsible_separator="Plotting parameters: field lines",
        has_fieldlines: bool=False,
        field_kind: typing.Literal["overlay", "twin"]=field_kind,
        n_iter: int = 3,
        swirl: float = 0.,
        damping_ratio: float = 0.8,
        twin_intensity: float = 0.1,
        
        _7: fsgui.collapsible_separator="Interior points",
        interior_color: fscolors.Color=(0.1, 0.1, 0.1, 1.0),

#        _8: fsgui.collapsible_separator="Blender output: Heightmap",
#        has_heightmap: bool=False,

        _8: fsgui.collapsible_separator="High-quality rendering options",
        final_render: bool=False,
        supersampling: fs.core.supersampling_type = "None",
        jitter: bool = False,
        reload: bool = False,

        _9: fsgui.collapsible_separator="General settings",
        log_verbosity: typing.Literal[fs.log.verbosity_enum
                                      ]="debug @ console + log",
        enable_multithreading: bool = True,
        inspect_calc: bool = False,
        
        
    ):


        fs.settings.log_directory = os.path.join(fractal.directory, "log")
        fs.set_log_handlers(verbosity=log_verbosity)
        fs.settings.enable_multithreading = enable_multithreading
        fs.settings.inspect_calc = inspect_calc

        fractal.zoom(
            precision=dps,
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=xy_ratio,
            theta_deg=theta_deg,
            projection="cartesian",
        )

        BLA_eps = eps if use_BLA else None

        fractal.calc_std_div(
                calc_name=calc_name,
                subset=None,
                max_iter=max_iter,
                M_divergence=M_divergence,
                epsilon_stationnary=epsilon_stationnary,
                BLA_eps=BLA_eps,
                interior_detect=interior_detect,
            )


        pp = Postproc_batch(fractal, calc_name)
        
        if base_layer == "continuous_iter":
            pp.add_postproc(base_layer, Continuous_iter_pp())

        elif base_layer == "distance_estimation":
            pp.add_postproc("continuous_iter", Continuous_iter_pp())
            pp.add_postproc(base_layer, DEM_pp())

        if has_fieldlines:
            pp.add_postproc(
                "fieldlines",
                Fieldlines_pp(n_iter, swirl, damping_ratio)
            )
        else:
            field_kind = "None"

        interior_func = {
            "all": lambda x: x != 1,
            "not_diverging": lambda x: x == 0,
            "dzndz_detection": lambda x: x == 2,
        }[interior_mask]
        pp.add_postproc("interior", Raw_pp("stop_reason", func=interior_func))

        if has_shading:
            pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

        plotter = fs.Fractal_plotter(
            pp,
            final_render=final_render,
            supersampling=supersampling,
            jitter=jitter,
            reload=reload
        )

        plotter.add_layer(Bool_layer("interior", output=False))

        if field_kind == "twin":
            plotter.add_layer(Virtual_layer(
                    "fieldlines", func=None, output=False
            ))
        elif field_kind == "overlay":
            plotter.add_layer(Grey_layer(
                    "fieldlines", func=None, output=False
            ))

        if has_shading:
            plotter.add_layer(Normal_map_layer(
                "DEM_map", max_slope=max_slope, output=True
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

        if has_shading:
            plotter[base_layer].shade(plotter["DEM_map"], lighting)

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