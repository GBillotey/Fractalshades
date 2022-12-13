# -*- coding: utf-8 -*-
"""
==============================
10 - Double embedded Julia set
==============================

Example plot a double-embedded Julia set in mandelbrot power-2

Embedded Julia sets are structures that occur around certain minibrots.
When zooming deeper in the fractal, these structures stacks and become
more and more complex.


This example even if not too deep is beyond the separation power of double
precision data type,
[#f1]_, at *7e-22*. We will use a `Perturbation_mandelbrot` instance.

Reference:
`fractalshades.models.Perturbation_mandelbrot`

.. [#f1] **Credit:** Coordinates from Robert P. Munafo website:
        <https://mrob.com/pub/muency/secondorderembeddedjuliase.html>
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
    Fieldlines_pp,
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
    x = "-1.768667862837488812627419470"
    y = "0.001645580546820209430325900"
    dx = "12.e-22"
    precision = 30
    nx = 2400
    xy_ratio = 16. / 9.

    calc_name="mandelbrot"
    
    cmap_choice = 2
    if cmap_choice == 1:
        c3 = np.array([255, 215, 0]) / 255.
        c0 = np.array([212, 175, 55]) / 255.
        c1 = c0 * 0.25

        colors = np.vstack((c0[np.newaxis, :],
                            c1[np.newaxis, :],
                            c3[np.newaxis, :]))
        colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
             grad_npts=200, grad_funcs="x**0.5", extent="mirror")
    elif cmap_choice == 2:
        colormap = fscolors.cmap_register["classic"]

    # Run the calculation
    f = fsm.Perturbation_mandelbrot(plot_dir)
    f.zoom( precision=precision,
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=xy_ratio,
            theta_deg=0., 
            projection="cartesian",
    )

    f.calc_std_div(
            calc_name=calc_name,
            subset=None,
            max_iter=100000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            BLA_eps=1.e-6,
            interior_detect=False,
            calc_orbit=True,
            backshift=4
    )

    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("cont_iter", Continuous_iter_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
    pp.add_postproc("fieldlines",
                Fieldlines_pp(n_iter=3, swirl=0., endpoint_k=1.0))

    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Normal_map_layer("DEM_map", max_slope=35, output=False))
    plotter.add_layer(Virtual_layer("fieldlines", func=None, output=False))
    plotter.add_layer(Color_layer(
            "cont_iter",
            func="np.log(x)",
            colormap=colormap,
            probes_z=[9.015, 9.025],
            output=True
    ))

    plotter["cont_iter"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
    plotter["DEM_map"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))

    # This is the line where we indicate that coloring is a combination of
    # "Continuous iteration" and "fieldines values"
    plotter["cont_iter"].set_twin_field(plotter["fieldlines"], 0.0005)

    # This is where we define the lighting (here 2 ccolored light sources)
    # and apply the shading
    light = Blinn_lighting(0.4, np.array([1., 1., 1.]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=10.,
        shininess=400.,
        polar_angle=45.,
        azimuth_angle=20.,
        color=np.array([0.9, 0.9, 0.9]))
    light.add_light_source(
        k_diffuse=0.8,
        k_specular=0.,
        shininess=400.,
        polar_angle=55.,
        azimuth_angle=20.,
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
