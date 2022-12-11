# -*- coding: utf-8 -*-
"""
============================
14 - Burning ship deeper DEM
============================

Plotting of a distance estimation for the Burning ship (power-2).
This zoom is deeper, featuring a miniship at 1.e-101

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
    x = '0.533551593577038561769721161491702555962775680136595415306315189524970818968817900068355227861158570104764433694'
    y = '1.26175074578870311547721223871955368990255513054155186351034363459852900933566891849764050954410207620093433856'
    dx = '7.072814368784043e-101'
    precision = 150
    nx = 2400
    xy_ratio = 1.8
    
    sign = 1.0
    DEM_min = 5.e-5
    zmin = -9.903487205505371
    zmax = -4.948512077331543
    
    # As this formula is non-analytic, we will 'unskew' based on the 
    # influencing miniship "size estimate" matrix.
    has_skew = True
    skew_00 = 1.3141410612942215
    skew_01 = 0.8651590600810832
    skew_10 = 0.6372176654581702
    skew_11 = 1.1804627997751416

    calc_name="Burning_ship"
    colormap = fscolors.cmap_register["dawn"]

    # Run the calculation
    f = fsm.Perturbation_burning_ship(plot_dir)

    f.zoom(
        precision=precision,
        x=x,
        y=y,
        dx=dx,
        nx=nx,
        xy_ratio=xy_ratio,
        theta_deg=-2., 
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
        max_iter=50000,
        M_divergence=1.e3,
        BLA_eps=1.e-6,
    )

    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("continuous_iter", Continuous_iter_pp())
    pp.add_postproc("distance_estimation", DEM_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Normal_map_layer("DEM_map", max_slope=35, output=False))
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

    plotter["distance_estimation"].set_mask(plotter["interior"],
           mask_color=(0.0, 0.22745098173618317, 0.9803921580314636))
    plotter["DEM_map"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))

    # define the lighting and apply the shading
    light = Blinn_lighting(0.3, np.array([1., 1., 1.]))
    light.add_light_source(
        k_diffuse=0.4,
        k_specular=4.,
        shininess=300.,
        polar_angle=45.,
        azimuth_angle=20.,
        color=np.array([1.0, 1.0, 0.96]))

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
