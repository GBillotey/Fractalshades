# -*- coding: utf-8 -*-
"""
=========================
Seahorse interior example
=========================

This example shows how to use several layers to plot both the divergent and
convergent part of a fractal, with fieldlines and scene lighting.

The location is a shallow one in the main Seahorse valley.
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
    Fractal_array,
    Attr_pp,
    Attr_normal_pp
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Grey_layer,
    Blinn_lighting,
    Overlay_mode
)


def plot(plot_dir=None):
    """
    Combining all : a shallow zoom in the Seahorses valley
    Coloring of escaping points based on continuous iteration + lighting with a
    normal maps from distance estimation method
    Coloring of interior points based on the attracting cycle attractivity
    """
    # Define the parameters for this calculation
    x = -0.746223962861
    y = -0.0959468433527
    dx = 0.00745
    nx = 800

    calc_name="escaping"
    colormap = fscolors.classic_colormap
    colormap_int = fscolors.classic_colormap

    # Run the calculation
    f = fsm.Mandelbrot(plot_dir)
    f.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=1.0,
           theta_deg=0., projection="cartesian", antialiasing=False)
    f.base_calc(
        calc_name=calc_name,
        subset=None,
        max_iter=5000,
        M_divergence=100.,
        epsilon_stationnary= 0.001,
        datatype=np.complex128)
    # f.clean_up("escaping") # keep this line if you want to force recalculate
    f.run()
    
    # Run the calculation for the interior points
    interior = Fractal_array(f, "escaping", "stop_reason",
                         func= lambda x: (x != 1))
    f.newton_calc(
        calc_name="interior",
        subset=interior,
        known_orders=None,
        max_order=1500,
        max_newton=20,
        eps_newton_cv=1.e-12,
        datatype=np.complex128)
    # f.clean_up("interior")
    f.run()

    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("cont_iter", Continuous_iter_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    pp.add_postproc("div", Raw_pp("stop_reason", func="x == 1."))
    pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))
    pp.add_postproc("fieldlines",
                Fieldlines_pp(n_iter=10, swirl=1., damping_ratio=0.1))
    
    # Defines a second pastproc batch for interior points
    pp_int = Postproc_batch(f, "interior")
    pp_int.add_postproc("attr_map", Attr_normal_pp())
    pp_int.add_postproc("attr", Attr_pp())

    plotter = fs.Fractal_plotter([pp, pp_int])  
    plotter.add_layer(Bool_layer("interior", output=False))
    plotter.add_layer(Bool_layer("div", output=False))
    plotter.add_layer(Normal_map_layer("DEM_map", max_slope=60, output=False))
    plotter.add_layer(Normal_map_layer("attr_map", max_slope=90, output=False))
    plotter.add_layer(Color_layer(
            "cont_iter",
            func="np.log(x)",
            colormap=colormap,
            probes_z=[1., 2.],
            probes_kind="absolute",
            output=True
    ))
    plotter.add_layer(Color_layer(
            "attr",
            func=None, #"x * -1. + 1.", #"np.log(x)",
            colormap=colormap_int,
            probes_z=[1., 2.],
            probes_kind="absolute",
            output=False))
    plotter.add_layer(Grey_layer("fieldlines", func=None,
                                 curve=lambda x: 0.5 + (x - 0.5) * 0.2,
                                 output=False))

    # plotter["cont_iter"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
    plotter["DEM_map"].set_mask(plotter["interior"], mask_color=(0., 0., 0.))
    plotter["attr"].set_mask(plotter["div"], mask_color=(0.9568, 0.8039, 0.9372))

    # This is where we define the lighting (here 3 ccolored light sources)
    # and apply the shading
    light = Blinn_lighting(0.2, np.array([1., 1., 1.]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=10.,
        shininess=400.,
        angles=(-135., 20.),
        coords=None,
        color=np.array([0.05, 0.05, 1.0]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=10.,
        shininess=400.,
        angles=(135., 20.),
        coords=None,
        color=np.array([0.5, 0.5, .4]))
    light.add_light_source(
        k_diffuse=1.3,
        k_specular=0.,
        shininess=0.,
        angles=(90., 40.),
        coords=None,
        color=np.array([1.0, 1.0, 1.0]))

    # Adds some shading based on the previouly defined normal maps
    plotter["cont_iter"].shade(plotter["DEM_map"], light)
    plotter["attr"].shade(plotter["attr_map"], light)

    # Overlay : tint or shade depending on fieldlines layer value
    overlay_mode = Overlay_mode("tint_or_shade", pegtop=1.)
    plotter["cont_iter"].overlay(plotter["fieldlines"], overlay_mode)

    # Overlay : alpha composite with "interior" layer ie, where it is not
    # masked, we take the value of the "attr" layer
    overlay_mode = Overlay_mode("alpha_composite")
    plotter["cont_iter"].overlay(plotter["attr"], overlay_mode=overlay_mode)

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