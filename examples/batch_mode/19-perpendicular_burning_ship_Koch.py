# -*- coding: utf-8 -*-
"""
=========================================================
19 - "Perpendicular" Burning Ship: hidden Koch snowflakes
=========================================================

Another hidden feature in this fractal: here, a reminiscence of Koch snowflakes
close to a mini at a depth of 1.e-53.

Reference:
`fractalshades.models.Perturbation_perpendicular_burning_ship`
"""

import os
import numpy as np

import fractalshades as fs
import fractalshades.models as fsm

import fractalshades.colors as fscolors
from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    DEM_normal_pp,
    DEM_pp,
    Raw_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Virtual_layer,
    Blinn_lighting,
)


def plot(plot_dir):
    fs.settings.enable_multithreading = True
    fs.settings.inspect_calc = True

    # A simple showcase using perturbation technique
    calc_name = 'test'

    # _1 = 'Zoom parameters'
    x = '-1.47214775731981401403314210855688086942157169819838263666455234'
    y = '-0.000000821699995458791163928571540134217349033686644929036252293894057'
    dx = '1.00371044925664e-53'
    theta_deg = 90.0
    xy_ratio = 1.8
    dps = 60
    nx = 2400

    # _1b = 'Skew parameters /!\\ Re-run when modified!'
    has_skew = True
    skew_00 = 0.13539222675927937
    skew_01 = 0.8700072011411545
    skew_10 = -1.153798854345238
    skew_11 = -0.028164925269682

    # _2 = 'Calculation parameters'
    max_iter = 20000

    # _3 = 'Bilinear series parameters'
    eps = 1e-06

    # _4 = 'Plotting parameters: base field'
    base_layer = 'distance_estimation'
    interior_color = (0.6627451181411743, 0.4313725531101227, 0.0)
    colormap = fscolors.Fractal_colormap(
        colors=[
            [0.        , 0.02745098, 0.39215686],
            [1.        , 1.        , 0.37254903]
        ],
        kinds=['Lch'],
        grad_npts=[40],
        grad_funcs=['1-(1-x)**2'],
        extent='repeat'
    )
    invert_cmap = False
    DEM_min = 1e-07
    zmin = -16.11809539794922
    zmax = -6.8490519523620605

    # _5 = 'Plotting parameters: shading'
    shade_kind = 'glossy'
    gloss_intensity = 100.0
    light_angle_deg = 35.0
    light_color = (1.0, 1.0, 1.0)
    gloss_light_color = (1.0, 1.0, 1.0)

    # Run the calculation
    fractal = fsm.Perturbation_burning_ship(
            plot_dir,
            flavor="Perpendicular burning ship"
    )


    fractal.zoom(precision=dps, x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
                 theta_deg=theta_deg, projection="cartesian",
                 has_skew=has_skew, skew_00=skew_00, skew_01=skew_01,
                 skew_10=skew_10, skew_11=skew_11
            )

    fractal.calc_std_div(
        calc_name=calc_name,
        subset=None,
        max_iter=max_iter,
        M_divergence=1.e3,
        BLA_eps=eps,
    )

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
            "DEM_map", max_slope=60, output=False
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
            output=True))
    plotter[base_layer].set_mask(
        plotter["interior"], mask_color=interior_color
    )
    if shade_kind != "None":
        light = Blinn_lighting(0.6, np.array([1., 1., 1.]))
        light.add_light_source(
            k_diffuse=0.8,
            k_specular=.0,
            shininess=350.,
            polar_angle=light_angle_deg,
            azimuth_angle=10.,
            color=np.array(light_color))

        if shade_kind == "glossy":
            light.add_light_source(
                k_diffuse=0.2,
                k_specular=gloss_intensity,
                shininess=400.,
                polar_angle=light_angle_deg,
                azimuth_angle=10.,
                color=np.array(gloss_light_color))

        plotter[base_layer].shade(plotter["DEM_map"], light)

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
