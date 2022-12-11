# -*- coding: utf-8 -*-
"""
=====================
13 - Burning ship DEM
=====================

Plotting of a distance estimation for the Burning ship (power-2).
This zoom is on the structure which gave the fractal its name. We use an
arbitrary-precision model, even if this is obviously not needed here.

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
    x = '-1.7579317963'
    y = '0.052705991307'
    dx = '0.181287312180757'
    precision = 30
    nx = 2400
    xy_ratio = 1.0
    
    sign = 1.0
    DEM_min = 1.e-4
    zmin = -9.21034049987793
    zmax = -0.3999025523662567
    
    # As this formula is non-analytic, we will 'unskew' based on the 
    # influencing miniship "size estimate" matrix.
    has_skew = False
    skew_00 = 1.0
    skew_01 = 0.0
    skew_10 = -0.1
    skew_11 = 1.1

    calc_name="Burning_ship"
    colormap = fscolors.cmap_register["classic"]

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
        max_iter=1500,
        M_divergence=1.e3,
        BLA_eps= 1.e-6,
    )


    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("continuous_iter", Continuous_iter_pp())
    pp.add_postproc("distance_estimation", DEM_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Normal_map_layer("DEM_map", max_slope=30, output=False))
    plotter.add_layer(
        Virtual_layer("continuous_iter", func=None, output=False)
    )
    
    cmap_func = lambda x: sign * np.where(
       np.isinf(x),
       np.log(DEM_min),
       np.log(np.clip(x, DEM_min, None))
    )
    plotter.add_layer(Color_layer(
            "distance_estimation",
            func=cmap_func,
            colormap=colormap,
            probes_z=[zmin, zmax],
            output=True
    ))

    plotter["distance_estimation"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
    plotter["DEM_map"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))

    # This is where we define the lighting (here 2 ccolored light sources)
    # and apply the shading
    light = Blinn_lighting(0.4, np.array([1., 1., 1.]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=300.,
        shininess=1400.,
        polar_angle=45.,
        azimuth_angle=10.,
        color=np.array([1.0, 1.0, 0.98]))
    light.add_light_source(
        k_diffuse=0.8,
        k_specular=2.,
        shininess=400.,
        polar_angle=45.,
        azimuth_angle=10.,
        color=np.array([1., 1., 1.]))
    plotter["distance_estimation"].shade(plotter["DEM_map"], light)

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
