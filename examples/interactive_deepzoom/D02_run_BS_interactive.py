# -*- coding: utf-8 -*-
"""
===============================================
D02 - Burning Ship arbitrary-precision explorer
===============================================

This is a template to explore the Burning Ship set with
arbitrary precision through a GUI.
It features the main postprocessing options (continuous
iteration, distance estimation based shading)

As the Burning ship is a non-holomorphic fractal, some areas can exibit a heavy
skew. This explorer allows you to use an unskewing matrice and continue
the exploration.
A suitable  unskew matrice is usually given by the influencing mini-ship, which
you can get as part of a Newton search results : right click on the image and 
select "Newton search".
When the skew parameters are changed, hit rerun to continue the exploration.

Good exploration !

Reference:
`fractalshades.models.Perturbation_burning_ship`
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
    DEM_pp,
    DEM_normal_pp,
    Raw_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Virtual_layer,
    Blinn_lighting
)


def plot(plot_dir):
    """
    Example interactive
    """
    import mpmath

    x = '-0.5'
    y = '0.5'
    dx = '5.0'
    calc_name = 'test'
    
    xy_ratio = 1.0
    dps = 16
    max_iter = 1500
    nx = 800
    theta_deg = 0.
    has_skew = False
    eps = 1.e-6

    base_layer = "continuous_iter"
    colormap = fscolors.cmap_register["classic"]
    cmap_z_kind = "relative"
    zmin = 0.30
    zmax = 0.60
    
    shade_kind="glossy"

    # Set to True to enable multi-threading
    settings.enable_multithreading = True

    directory = plot_dir
    fractal = fsm.Perturbation_burning_ship(directory)

    def func(
        fractal: fsm.Perturbation_burning_ship=fractal,
         calc_name: str=calc_name,

         _1: fsgui.separator="Zoom parameters",
         x: mpmath.mpf=x,
         y: mpmath.mpf=y,
         dx: mpmath.mpf=dx,
         xy_ratio: float=xy_ratio,
         theta_deg: float=theta_deg,
         dps: int=dps,
         nx: int=nx,

         _1b: fsgui.separator="Skew parameters /!\ Re-run when modified!",
         has_skew: bool=has_skew,
         skew_00: float=1.,
         skew_01: float=0.,
         skew_10: float=0.,
         skew_11: float=1.,

         _2: fsgui.separator="Calculation parameters",
         max_iter: int=max_iter,

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
         DEM_min: float=1.e-6,
         cmap_z_kind: typing.Literal["relative", "absolute"]=cmap_z_kind,
         zmin: float=zmin,
         zmax: float=zmax,

         _5: fsgui.separator="Plotting parameters: shading",
         shade_kind: typing.Literal["None", "standard", "glossy"]=shade_kind,
         gloss_intensity: float=10.,
         light_angle_deg: float=65.,
         light_color: QtGui.QColor=(1.0, 1.0, 1.0),
         gloss_light_color: QtGui.QColor=(1.0, 1.0, 1.0),
    ):


        fractal.zoom(precision=dps, x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=theta_deg, projection="cartesian", antialiasing=False,
             has_skew=has_skew, skew_00=skew_00, skew_01=skew_01,
             skew_10=skew_10, skew_11=skew_11
        )

        fractal.calc_std_div(
            calc_name=calc_name,
            subset=None,
            max_iter=max_iter,
            M_divergence=1.e3,
            BLA_params={"eps": eps},
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

        pp.add_postproc("interior", Raw_pp("stop_reason",
                        func=lambda x: x != 1))
        if shade_kind != "None":
            pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))

        if shade_kind != "None":
            plotter.add_layer(Normal_map_layer(
                "DEM_map", max_slope=60, output=True
            ))

        if base_layer != 'continuous_iter':
            plotter.add_layer(
                Virtual_layer("continuous_iter", func=None, output=False)
            )

        sign = {False: 1., True: -1.}[invert_cmap]
        if base_layer == 'distance_estimation':
            cmap_func = lambda x: sign * np.where(
               np.isinf(x),
               np.log(DEM_min),
               np.log(np.clip(x, DEM_min, None))
            )
        else:
            cmap_func = lambda x: sign * np.log(x)

        plotter.add_layer(Color_layer(
                base_layer,
                func=cmap_func,
                colormap=colormap,
                probes_z=[zmin, zmax],
                probes_kind=cmap_z_kind,
                output=True))
        plotter[base_layer].set_mask(
            plotter["interior"], mask_color=interior_color
        )
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
                    shininess=400.,
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
    gui.connect_mouse(
        x="x", y="y", dx="dx", xy_ratio="xy_ratio", dps="dps",
        has_skew="has_skew", skew_00="skew_00", skew_01="skew_01",
        skew_10="skew_10", skew_11="skew_11"
    )
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

