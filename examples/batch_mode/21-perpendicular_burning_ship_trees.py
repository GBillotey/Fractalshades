# -*- coding: utf-8 -*-
"""
============================================================
21 - "Perpendicular" Burning Ship: tree structures
============================================================

Another hidden feature in this fractal: here, tree structures
at a depth of 1.e-29.

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
    x = '-1.60075649116104853234447567671822519294'
    y = '-0.00000585584069328913182973043272000146363667'
    dx = '1.345424030679299e-29'
    xy_ratio = 1.6
    theta_deg = 120.
    dps = 64
    nx = 2400

    # _1b = 'Skew parameters /!\\ Re-run when modified!'
    has_skew = True
    skew_00 = -0.985244568474214
    skew_01 = 0.6137988525
    skew_10 = 0.8089497623371
    skew_11 = -1.518945126681

    # _2 = 'Calculation parameters'
    max_iter = 6000

    # _3 = 'Bilinear series parameters'
    eps = 1e-06

    # _4 = 'Plotting parameters: base field'
    base_layer = 'continuous_iter'
    interior_color = (0.6627451181411743, 0.4313725531101227, 0.0)
    colormap = fscolors.Fractal_colormap(
        colors=[
            [1.        , 1.        , 0.        ],
            [0.01176471, 0.07450981, 0.41960785],
            [0.19215687, 0.65098041, 0.63529414],
            [0.26666668, 1.        , 0.15294118],
            [1.        , 1.        , 0.        ],
            [1.        , 1.        , 0.        ],
            [1.        , 1.        , 0.        ],
            [1.        , 1.        , 0.        ]
        ],
        kinds=['Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch', 'Lch'],
        grad_npts=[30, 32, 32, 32, 32, 32, 32, 32],
        grad_funcs=['x**0.5', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        extent='repeat'
    )
    invert_cmap = False
    DEM_min = 1e-06
    zmin = 7.516922950744629
    zmax = 8.069853782653809

    # _5 = 'Plotting parameters: shading'
    shade_kind = 'glossy'
    gloss_intensity = 10.0
    light_angle_deg = -120.
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
            output=True)
    )
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
