# -*- coding: utf-8 -*-
"""
===========================
12 - Burning Ship deep zoom
===========================

Example plot of a deep-zoom inside the Burning Ship set (power-2).

Like the Mandelbrot set, this set features "mini-ships", which are
smaller copies of the whole. The zoom proposed here displays the inner
decoration around a deep miniship (period 9622, size 3.40e-265).

Reference:
`fractalshades.models.Perturbation_burning_ship`
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
    Raw_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Blinn_lighting,
)


def plot(plot_dir):
    fs.settings.enable_multithreading = True
    fs.settings.inspect_calc = True

    # A simple showcase using perturbation technique
    x = "-1.996381122925929653294037828311253469966641852551354028279745849996384561656543300231734846372389200884473400326047402259215365299651000049307967220797174220920308250039037884064184875313550754352410343246674089430454249309097057276774162871180276065193880170743903875782"
    y = "0.000004763303728392770746014692362029865866711691612095974571331721507794709607604050981467065736920676037481378828276642944072069756835182953451109905077881061139887661561617020726899218853890425452233104923366681895875677542245236396274857406762304353824848557938906613374564"
    dx = "1.207782640899473e-261"
    precision = 271
    nx = 2400
    xy_ratio = 1.8
    
    # As this formula is non-analytic, we will 'unskew' based on the 
    # influencing miniship "size estimate" matrix.
    has_skew = True
    skew_00 = 0.659275816850581
    skew_01 = -0.5484220094485625
    skew_10 = 0.25710339801865756
    skew_11 = 1.3029430412388974

    calc_name="Burning_ship"
    colormap = fscolors.cmap_register["peacock"]
    colormap.clip = "repeat"

    # Run the calculation
    f = fsm.Perturbation_burning_ship(plot_dir)

    f.zoom(
        precision=precision,
        x=x,
        y=y,
        dx=dx,
        nx=nx,
        xy_ratio=xy_ratio,
        theta_deg=0., 
        projection="cartesian",
        has_skew=has_skew,
        skew_00=skew_00,
        skew_01=skew_01,
        skew_10=skew_10,
        skew_11=skew_11
    )

    f.calc_std_div(
        calc_name=calc_name,
        subset=None,
        max_iter=150000,
        M_divergence=1.e3,
        BLA_eps= 1.e-6,
        calc_hessian=True
    )

    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("cont_iter", Continuous_iter_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Normal_map_layer("DEM_map", max_slope=40, output=False))
    plotter.add_layer(Color_layer(
            "cont_iter",
            func=lambda x: np.log(x),
            colormap=colormap,
            probes_z=[10.975318, 10.977222],
            output=True
    ))

    plotter["cont_iter"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
    plotter["DEM_map"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))

    # This is where we define the lighting (here 2 ccolored light sources)
    # and apply the shading
    light = Blinn_lighting(0.4, np.array([1., 1., 1.]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=30.,
        shininess=400.,
        polar_angle=0.,
        azimuth_angle=10.,
        color=np.array([1.0, 1.0, 0.95]))
    light.add_light_source(
        k_diffuse=0.8,
        k_specular=0.,
        shininess=400.,
        polar_angle=0.,
        azimuth_angle=10.,
        color=np.array([1., 1., 1.]))
    plotter["cont_iter"].shade(plotter["DEM_map"], light)

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
