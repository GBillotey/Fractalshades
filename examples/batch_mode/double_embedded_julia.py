# -*- coding: utf-8 -*-
"""
=========================
Double embedded Julia set
=========================
Example plot a double-embedded Julia set in mandelbrot power-2

This example even if not too deep is beyong the reach of double precision
[#f1]_, at *7e-22*. We will use a `Perturbation_mandelbrot` instance.


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
    # A simple showcase using perturbation technique
    x = "-1.768667862837488812627419470"
    y = "0.001645580546820209430325900"
    dx = "16.e-22"
    precision = 30
    nx = 1600
    xy_ratio = 2.0

    calc_name="mandelbrot"
    
    c3 = np.array([125, 225, 255]) / 255.
    c0 = np.array([1, 254, 255]) / 255.
    c1 = c0 * 0.01

    colors = np.vstack((c0[np.newaxis, :],
                        c1[np.newaxis, :],
                        c3[np.newaxis, :]))
    colormap = fscolors.Fractal_colormap(kinds="Lch", colors=colors,
         grad_npts=200, grad_funcs="x**0.5", extent="mirror")

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
            antialiasing=False)

    f.calc_std_div(
            datatype=np.complex128,
            calc_name=calc_name,
            subset=None,
            max_iter=200000,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            SA_params={"cutdeg": 64,
                       "cutdeg_glitch": 8,
                       "SA_err": 1.e-4},
            glitch_eps=1.e-6,
            interior_detect=False,
            glitch_max_attempt=2)
    f.run()

    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("cont_iter", Continuous_iter_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
    pp.add_postproc("fieldlines",
                Fieldlines_pp(n_iter=8, swirl=0.4, damping_ratio=0.1))

    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Normal_map_layer("DEM_map", max_slope=60, output=False))
    plotter.add_layer(Virtual_layer("fieldlines", func=None, output=False))
    plotter.add_layer(Color_layer(
            "cont_iter",
            func="np.log(x)",
            colormap=colormap,
            #probes_z=[0.095, 0.11],
            probes_z=[0.07, 0.075],
            probes_kind="absolute",
            output=True
    ))

    plotter["cont_iter"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
    plotter["DEM_map"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))

    # This is the line where we indicate that coloring is a combination of
    # "Continuous iteration" and "fieldines values"
    plotter["cont_iter"].set_twin_field(plotter["fieldlines"], 0.0001)

    plotter.plot()
    # This is where we define the lighting (here 3 ccolored light sources)
    # and apply the shading
    light = Blinn_lighting(0.2, np.array([1., 1., 1.]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=200.,
        shininess=1400.,
        angles=(45., 20.),
        coords=None,
        color=np.array([1.0, 1.0, 0.5]))
    light.add_light_source(
        k_diffuse=1.,
        k_specular=0.,
        shininess=0.,
        angles=(-905., 20.),
        coords=None,
        color=np.array([1., 1., 0.8]))
    light.add_light_source(
        k_diffuse=1.,
        k_specular=0.,
        shininess=0.,
        angles=(-15., 20.),
        coords=None,
        color=np.array([1., 1., 0.5]))
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
            plot(plot_dir)
