# -*- coding: utf-8 -*-
"""============================================================================
Auto-generated from fractalshades GUI.
Save to `<file>.py` and run as a python script:
    > python <file>.py
============================================================================"""
import os
import typing

import numpy as np
import mpmath
from PyQt6 import QtGui

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.gui as fsgui
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

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Parameters
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    fractal = fsm.Log_var1(plot_dir)
    calc_name = 'test'
    _1 = 'Zoom parameters'
    x = 0.0
    y = -0.0
    dx = 10.0
    xy_ratio = 1.0
    theta_deg = 0.0
    _2 = 'Calculation parameters'
    nx = 800
    compute_order = True
    max_order = 100
    eps_newton_cv = 1e-12
    _3 = 'Plotting parameters'
    interior_color = (0.1, 0.1, 0.1)
    colormap = fs.colors.cmap_register["classic"]
    cmap_z_kind = 'relative'
    zmin = 0.0
    zmax = 0.2
    attr_strength = 0.15

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Plotting function
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def func(
        fractal: fsm.Mandelbrot=fractal,
         calc_name: str= calc_name,
         _1: fsgui.separator="Zoom parameters",
         x: float= x,
         y: float= y,
         dx: float= dx,
         xy_ratio: float=xy_ratio,
         theta_deg: float=theta_deg,
         _2: fsgui.separator="Calculation parameters",
         nx: int=nx,
         compute_order: bool=compute_order,
         max_order: int=max_order,
         eps_newton_cv: float=eps_newton_cv,
         _3: fsgui.separator="Plotting parameters",
         interior_color: QtGui.QColor=(0.1, 0.1, 0.1),
         colormap: fscolors.Fractal_colormap=colormap,
         cmap_z_kind: typing.Literal["relative", "absolute"]="relative",
         zmin: float=zmin,
         zmax: float=zmax,
         attr_strength: float =attr_strength,
         
    ):

        fractal.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=theta_deg, projection="cartesian", antialiasing=False)

        fractal.newton_calc(
            calc_name=calc_name,
            subset=None,
            compute_order=compute_order,
            max_order=max_order,
            max_newton=20,
            eps_newton_cv=eps_newton_cv
        )

        if fractal.res_available():
            print("RES AVAILABLE, no compute")
        else:
            print("RES NOT AVAILABLE, clean-up")
            fractal.clean_up(calc_name)

        fractal.run()

        layer_name = "cycle_order"

        pp = Postproc_batch(fractal, calc_name)
        pp.add_postproc(layer_name, Raw_pp("order"))
        pp.add_postproc("attr", Raw_pp("dzrdz", func=lambda x: np.abs(x)))
        pp.add_postproc("interior", Raw_pp("stop_reason",
                        func=lambda x: x != 1)
        )
#        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

        plotter = fs.Fractal_plotter(pp)   
        plotter.add_layer(Bool_layer("interior", output=False))
#        plotter.add_layer(Normal_map_layer("DEM_map", max_slope=45, output=True))
        plotter.add_layer(Color_layer(
                layer_name,
                func=lambda x: np.log(np.log(x + 1.) + 1.),
                colormap=colormap,
                probes_z=[zmin, zmax],
                probes_kind="relative",
                output=True))
        plotter[layer_name].set_mask(plotter["interior"],
                                     mask_color=interior_color)
        plotter.add_layer(Virtual_layer("attr", func=None, output=False))
        
        plotter[layer_name].set_twin_field(plotter["attr"], attr_strength)

        plotter.plot()
        
        # Renaming output to match expected from the Fractal GUI
        layer = plotter[layer_name]
        file_name = "{}_{}".format(type(layer).__name__, layer.postname)
        src_path = os.path.join(fractal.directory, file_name + ".png")
        dest_path = os.path.join(fractal.directory, calc_name + ".png")
        if os.path.isfile(dest_path):
            os.unlink(dest_path)
        os.link(src_path, dest_path)


    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Plotting call
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    func(fractal,
        calc_name,
        _1,
        x,
        y,
        dx,
        xy_ratio,
        theta_deg,
        _2,
        nx,
        compute_order,
        max_order,
        eps_newton_cv,
        _3,
        interior_color,
        colormap,
        cmap_z_kind,
        zmin,
        zmax,
        attr_strength
    )


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Footer
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if __name__ == "__main__":
    # Some magic to get the directory for plotting: with a name that matches
    # the file 
    realpath = os.path.realpath(__file__)
    plot_dir = os.path.splitext(realpath)[0]
    plot(plot_dir)
 