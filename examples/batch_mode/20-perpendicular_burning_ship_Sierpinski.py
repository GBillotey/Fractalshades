# -*- coding: utf-8 -*-
"""
============================================================
20 - "Perpendicular" Burning Ship: hidden Sierpinski carpets
============================================================

Another hidden feature in this fractal: here, Sierpinski triangular carpets
at a depth of 7.e-55. This area is not too skeewed but well hidden close to a
minibrot from the needle.

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
    x = '-1.929319698524937920226708049698305350754670432084006734339806946'
    y = '-0.0000000000000000007592779387989739090287550144163328879329853232537252481600401185'
    dx = '7.032184999234219e-55'
    xy_ratio = 1.6
    theta_deg = -26.0
    dps = 64
    nx = 2400

    # _1b = 'Skew parameters /!\\ Re-run when modified!'
    has_skew = True
    skew_00 = 1.05
    skew_01 = 0.0
    skew_10 = -0.1
    skew_11 = 0.9523809

    # _2 = 'Calculation parameters'
    max_iter = 20000

    # _3 = 'Bilinear series parameters'
    eps = 1e-06

    # _4 = 'Plotting parameters: base field'
    base_layer = 'continuous_iter'
    interior_color = (0.6627451181411743, 0.4313725531101227, 0.0)
    colormap = fscolors.Fractal_colormap(
    colors=[
        [1.        , 1.        , 0.49803922],
        [0.1254902 , 0.05098039, 0.36862746],
        [0.1254902 , 0.05098039, 0.36862746]
    ],
    kinds=['Lch', 'Lch', 'Lch'],
    grad_npts=[32, 32, 32],
    grad_funcs=['1-(1-x)**3', 'x', 'x'],
    extent='mirror'
    )
    invert_cmap = False
    DEM_min = 1e-04
    zmin = 8.713781356811523
    zmax = 9.903434753417969

    # _5 = 'Plotting parameters: shading'
    shade_kind = 'glossy'
    gloss_intensity = 30.0
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
