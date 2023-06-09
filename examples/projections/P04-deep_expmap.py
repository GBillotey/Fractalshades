# -*- coding: utf-8 -*-
"""============================================================================
Auto-generated from fractalshades GUI, version 1.1.0.
Save to `<file>.py` and run as a python script:
    > python <file>.py
============================================================================"""

import os
import typing

import numpy as np
import mpmath
from PyQt6 import QtGui

import fractalshades
import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.gui as fsgui
import fractalshades.colors as fscolors
import fractalshades.projection

from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    DEM_normal_pp,
    Fieldlines_pp,
    DEM_pp,
    Raw_pp,
    Attr_pp,
    Attr_normal_pp,
    Fractal_array
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Normal_map_layer,
    Grey_layer,
    Disp_layer,
    Virtual_layer,
    Blinn_lighting,
    Overlay_mode
)

# Note: in batch mode, edit this line to change the base directory
plot_dir = os.path.splitext(os.path.realpath(__file__))[0]

# Note: in batch mode, edit this line to change the local projection
# you may also call `plot` with a modified `batch_params` parameters
# (the latter allows to call from another module)
projection = fs.projection.Expmap(0.0, np.log(1.e20) + 0.3, orientation="vertical")
# projection = fs.projection.Expmap(np.log(1.e40), np.log(1.e50))
# projection = fs.projection.Expmap(np.log(1.e0), np.log(1.e10))
ks = 1.e-40
#projection = fs.projection.Cartesian()

batch_params = {
    "projection": projection
}


#------------------------------------------------------------------------------
# Parameters - user editable 
#------------------------------------------------------------------------------
plot_kwargs = {
    "fractal": fractalshades.models.mandelbrot_M2.Perturbation_mandelbrot(
        directory=plot_dir,
    ),
    "calc_name": "std_zooming_calc",
    "_1": "Zoom parameters",
#    "x": '-1.99996619445037030418434688506350579675531241540724851511761922944801584242342684381376129778868913812287046406560949864353810575744772166485672496092803920095332',
#    "y": '-0.00000000000000000000000000000000030013824367909383240724973039775924987346831190773335270174257280120474975614823581185647299288414075519224186504978181625478529',
#    "dx": mpmath.mpf('1.7e-157') * ks,
#    "dps": 285,

    "x": "-0.18476527944640054234980108927",
    "y": "1.0532419344392547587734377701",
    "dx": "7.603772829116657e-20",
    "dps": 31,

#    std_zooming_calc@zoom_kwargs@precision = 35
#std_zooming_calc@zoom_kwargs@x = -1.7147107444189592520878056649232183
#std_zooming_calc@zoom_kwargs@y = 5.8259262289981436923262936622352879e-29
#std_zooming_calc@zoom_kwargs@dx = 2.966450021341041e-26
    

    "xy_ratio": 1.6666,
    "theta_deg": 0.0,
    "nx": 2400, # 2400 x 325
    "_2": "Calculation parameters",
    "max_iter": 500000,
    "M_divergence": 1000.0,
    "interior_detect": True,
    "epsilon_stationnary": 0.01,
    "calc_dzndc": True,
    "_4": "Plotting parameters: base field",
    "base_layer": "continuous_iter",
    "colormap": fs.colors.cmap_register["classic"],
    "cmap_func": fractalshades.numpy_utils.expr_parser.Numpy_expr(
        variables=[
            "x",
        ],
        expr="np.log(x)",
    ),
    "zmin": 0.0,
    "zmax": 0.1,
    "zshift": 0.05,
    "mask_color": (
        0.1,
        0.1,
        0.1,
        1.0,
    ),
    "_5": "Plotting parameters: shading",
    "has_shading": False,
    "shading_kind": "potential",
    "lighting": fs.colors.layers.Blinn_lighting(
        k_ambient=0.4,
        color_ambient=[1., 1., 1.],
        ls0={
            'k_diffuse': 1.8,
            'k_specular': 15.0,
            'shininess': 500.0,
            'polar_angle': 50.0,
            'azimuth_angle': 20.0,
            'color': [1.  , 1.  , 0.95],
            'material_specular_color': None
        },
    ),
    "max_slope": 60.0,
    "_6": "Plotting parameters: field lines",
    "has_fieldlines": True,
    "fieldlines_func": fractalshades.numpy_utils.expr_parser.Numpy_expr(
        variables=[
            "x",
        ],
        expr="x",
    ),
    "fieldlines_kind": "twin",
    "fieldlines_zmin": -1.0,
    "fieldlines_zmax": 1.0,
    "backshift": 14,
    "n_iter": 10,
    "swirl": 0.0,
    "damping_ratio": 0.8,
    "twin_intensity": 0.25,
    "_8": "High-quality rendering options",
    "final_render": False,
    "supersampling": "3x3",
    "jitter": False,
    "recovery_mode": False,
    "_9": "Extra outputs",
    "output_masks": False,
    "output_normals": False,
    "output_heightmaps": False,
    "hmap_mask": 0.0,
    "int_hmap_mask": 0.0,
    "_10": "General settings",
    "log_verbosity": "debug @ console + log",
    "enable_multithreading": True,
    "inspect_calc": False,
    "no_newton": True,
    "postproc_dtype": "float32",
    "compute_newton": False,
    "_3": None,
    "max_order": None,
    "max_newton": None,
    "eps_newton_cv": None,
    "_7": None,
    "int_layer": None,
    "colormap_int": None,
    "cmap_func_int": None,
    "zmin_int": None,
    "zmax_int": None,
    "lighting_int": None,
    "_1b": None,
    "has_skew": False,
    "skew_00": 1.0,
    "skew_01": 0.0,
    "skew_10": 0.0,
    "skew_11": 1.0,
}

#------------------------------------------------------------------------------
# Function - /!\ do not modify this section
#------------------------------------------------------------------------------
def plot(
    fractal: fs.Fractal= fractalshades.models.mandelbrot_M2.Perturbation_mandelbrot(
        directory=plot_dir,
    ),
    calc_name: str="std_zooming_calc",

    _1: fs.gui.collapsible_separator="Zoom parameters",
    x : mpmath.mpf = "0.0",
    y : mpmath.mpf = "0.0",
    dx : mpmath.mpf = "10.0",
    dps: int = 16,
    xy_ratio: float = 1.0,
    theta_deg: float = 0.0,
    nx: int = 600,

    _1b: fs.gui.collapsible_separator = None,
    has_skew: bool = False,
    skew_00: float = 1.0,
    skew_01: float = 0.0,
    skew_10: float = 0.0,
    skew_11: float = 1.0,

    _2: fs.gui.collapsible_separator="Calculation parameters",
    max_iter: int = 5000,
    M_divergence: float = 1000.,
    interior_detect: bool = True,
    epsilon_stationnary: float = 0.001,
    calc_dzndc: bool = True,

    _3: fs.gui.collapsible_separator = None,
    compute_newton: bool = False,
    max_order: int = None,
    max_newton: int = None,
    eps_newton_cv: float = None,

    _4: fs.gui.collapsible_separator="Plotting parameters: base field",
    base_layer: typing.Literal[
             "continuous_iter",
             "distance_estimation"
    ]="continuous_iter",

    colormap: fs.colors.Fractal_colormap=(
            fs.colors.cmap_register["classic"]
    ),
    cmap_func: fs.numpy_utils.Numpy_expr = (
            fs.numpy_utils.Numpy_expr("x", "np.log(x)")
    ),
    zmin: float = 0.0,
    zmax: float = 5.0,
    zshift: float = -1.0,
    mask_color: fs.colors.Color=(0.1, 0.1, 0.1, 1.0),

    _7: fs.gui.collapsible_separator= None,
    int_layer: typing.Literal[
        "attractivity", "order", "attr / order"
    ]= None,
    colormap_int: fs.colors.Fractal_colormap = None,
    cmap_func_int: fs.numpy_utils.Numpy_expr = None,
    zmin_int: float = None,
    zmax_int: float = None,

    _5: fs.gui.collapsible_separator = "Plotting parameters: shading",
    has_shading: bool = True,
    shading_kind: typing.Literal["potential"] = "potential", 
    lighting: Blinn_lighting = (
            fs.colors.lighting_register["glossy"]
    ),
    lighting_int: Blinn_lighting = None,
    max_slope: float = 60.,

    _6: fs.gui.collapsible_separator = "Plotting parameters: field lines",
    has_fieldlines: bool = False,
    fieldlines_func: fs.numpy_utils.Numpy_expr = (
            fs.numpy_utils.Numpy_expr("x", "x")
    ),
    fieldlines_kind: typing.Literal["overlay", "twin"] = "overlay",
    fieldlines_zmin: float = -1.0,
    fieldlines_zmax: float = 1.0,
    backshift: int = 3, 
    n_iter: int = 4,
    swirl: float = 0.,
    damping_ratio: float = 0.8,
    twin_intensity: float = 0.1,


    _8: fs.gui.collapsible_separator="High-quality rendering options",
    final_render: bool=False,
    supersampling: fs.core.supersampling_type = "None",
    jitter: bool = False,
    recovery_mode: bool = False,

    _9: fs.gui.collapsible_separator="Extra outputs",
    output_masks: bool = False,
    output_normals: bool = False,
    output_heightmaps: bool = False,
    hmap_mask: float = 0.,
    int_hmap_mask: float = 0.,

    _10: fs.gui.collapsible_separator="General settings",
    log_verbosity: typing.Literal[fs.log.verbosity_enum
                                  ] = "debug @ console + log",
    enable_multithreading: bool = True,
    inspect_calc: bool = False,
    no_newton: bool = False,
    postproc_dtype: typing.Literal["float32", "float64"] = "float32",
    batch_params=batch_params
):

    fs.settings.log_directory = os.path.join(fractal.directory, "log")
    fs.set_log_handlers(verbosity=log_verbosity)
    fs.settings.enable_multithreading = enable_multithreading
    fs.settings.inspect_calc = inspect_calc
    fs.settings.no_newton = no_newton
    fs.settings.postproc_dtype = postproc_dtype


    zoom_kwargs = {
        "x": x,
        "y": y,
        "dx": dx,
        "nx": nx,
        "xy_ratio": xy_ratio,
        "theta_deg": theta_deg,
        "has_skew": has_skew,
        "skew_00": skew_00,
        "skew_01": skew_01,
        "skew_10": skew_10,
        "skew_11": skew_11,
        "projection": batch_params.get(
            "projection", fs.projection.Cartesian()
        )
    }
    if fractal.implements_deepzoom:
        zoom_kwargs["precision"] = dps
    fractal.zoom(**zoom_kwargs)


    calc_std_div_kw = {
        "calc_name": calc_name,
        "subset": None,
        "max_iter": max_iter,
        "M_divergence": M_divergence,
#        "BLA_eps": None # 1.e-6,
    }


    if fractal.implements_dzndc == "user":
        calc_std_div_kw["calc_dzndc"] = calc_dzndc

    if shading_kind == "Milnor":
        calc_std_div_kw["calc_d2zndc2"] = True

    if has_fieldlines:
        calc_orbit = (backshift > 0)
        calc_std_div_kw["calc_orbit"] = calc_orbit
        calc_std_div_kw["backshift"] = backshift

    if fractal.implements_interior_detection == "always":
        calc_std_div_kw["epsilon_stationnary"] = epsilon_stationnary
    elif fractal.implements_interior_detection == "user":
        calc_std_div_kw["interior_detect"] = interior_detect
        calc_std_div_kw["epsilon_stationnary"] = epsilon_stationnary

    fractal.calc_std_div(**calc_std_div_kw)


    # Run the calculation for the interior points - if wanted
    if compute_newton:
        interior = Fractal_array(
                fractal, calc_name, "stop_reason", func= "x != 1"
        )
        fractal.newton_calc(
            calc_name="interior",
            subset=interior,
            known_orders=None,
            max_order=max_order,
            max_newton=max_newton,
            eps_newton_cv=eps_newton_cv,
        )


    pp = Postproc_batch(fractal, calc_name)

    if base_layer == "continuous_iter":
        pp.add_postproc(base_layer, Continuous_iter_pp())
        if output_heightmaps:
            pp.add_postproc("base_hmap", Continuous_iter_pp())

    elif base_layer == "distance_estimation":
        pp.add_postproc("continuous_iter", Continuous_iter_pp())
        pp.add_postproc(base_layer, DEM_pp())
        if output_heightmaps:
            pp.add_postproc("base_hmap", DEM_pp())

    if has_fieldlines:
        pp.add_postproc(
            "fieldlines",
            Fieldlines_pp(n_iter, swirl, damping_ratio)
        )
    else:
        fieldlines_kind = "None"

    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1"))

    if compute_newton:
        pp_int = Postproc_batch(fractal, "interior")
        if int_layer == "attractivity":
            pp_int.add_postproc(int_layer, Attr_pp())
            if output_heightmaps:
                pp_int.add_postproc("interior_hmap", Attr_pp())
        elif int_layer == "order":
            pp_int.add_postproc(int_layer, Raw_pp("order"))
            if output_heightmaps:
                pp_int.add_postproc("interior_hmap", Raw_pp("order"))
        elif int_layer == "attr / order":
            pp_int.add_postproc(int_layer, Attr_pp(scale_by_order=True))
            if output_heightmaps:
                pp_int.add_postproc(
                    "interior_hmap", Attr_pp(scale_by_order=True)
                )

        # Set of unknown points
        pp_int.add_postproc(
            "unknown", Raw_pp("stop_reason", func="x == 0")
        )
        pps = [pp, pp_int]
    else:
        pps = pp

    if has_shading:
        pp.add_postproc("DEM_map", DEM_normal_pp(kind=shading_kind))
        if compute_newton:
            pp_int.add_postproc("attr_map", Attr_normal_pp())

    plotter = fs.Fractal_plotter(
        pps,
        final_render=final_render,
        supersampling=supersampling,
        jitter=jitter,
        recovery_mode=recovery_mode
    )

    # The mask values & curves for heighmaps
    r1 =  min(hmap_mask, 0.)
    r2 =  max(hmap_mask, 1.)
    dr = r2 - r1
    hmap_curve = lambda x : (np.clip(x, 0., 1.) - r1) / dr 

    r1 =  min(int_hmap_mask, 0.)
    r2 =  max(int_hmap_mask, 1.)
    dr = r2 - r1
    int_hmap_curve = lambda x : (np.clip(x, 0., 1.) - r1) / dr 


    # The layers
    plotter.add_layer(Bool_layer("interior", output=output_masks))

    if compute_newton:
        plotter.add_layer(Bool_layer("unknown", output=output_masks))

    if fieldlines_kind == "twin":
        plotter.add_layer(Virtual_layer(
                "fieldlines", func=fieldlines_func, output=False
        ))
    elif fieldlines_kind == "overlay":
        plotter.add_layer(Grey_layer(
                "fieldlines", func=fieldlines_func,
                probes_z=[fieldlines_zmin, fieldlines_zmax],
                output=False
        ))

    if has_shading:
        plotter.add_layer(Normal_map_layer(
            "DEM_map", max_slope=max_slope, output=output_normals
        ))
        plotter["DEM_map"].set_mask(plotter["interior"])
        if compute_newton:
            plotter.add_layer(Normal_map_layer(
                "attr_map", max_slope=90, output=output_normals
            ))

    if base_layer != 'continuous_iter':
        plotter.add_layer(
            Virtual_layer("continuous_iter", func=None, output=False)
        )

    plotter.add_layer(Color_layer(
            base_layer,
            func=cmap_func,
            colormap=colormap,
            probes_z=[zmin + zshift, zmax + zshift],
            output=True)
    )
    if output_heightmaps:
        plotter.add_layer(Disp_layer(
                "base_hmap",
                func=cmap_func,
                curve=hmap_curve,
                probes_z=[zmin + zshift, zmax + zshift],
                output=True
        ))


    if compute_newton:
        plotter.add_layer(Color_layer(
            int_layer,
            func=cmap_func_int,
            colormap=colormap_int,
            probes_z=[zmin_int, zmax_int],
            output=False))
        plotter[int_layer].set_mask(plotter["unknown"],
                                    mask_color=mask_color)
        if output_heightmaps:
            plotter.add_layer(Disp_layer(
                    "interior_hmap",
                    func=cmap_func,
                    curve=int_hmap_curve,
                    probes_z=[zmin_int, zmax_int],
                    output=True
            ))
            plotter["interior_hmap"].set_mask(
                plotter["unknown"],
                mask_color=(int_hmap_mask,)
            )

    if fieldlines_kind == "twin":
        plotter[base_layer].set_twin_field(
                plotter["fieldlines"], twin_intensity
        )
    elif fieldlines_kind == "overlay":
        overlay_mode = Overlay_mode("tint_or_shade", pegtop=1.0)
        plotter[base_layer].overlay(plotter["fieldlines"], overlay_mode)

    if has_shading:
        plotter[base_layer].shade(plotter["DEM_map"], lighting)
        if compute_newton:
            plotter[int_layer].shade(plotter["attr_map"], lighting_int)
            plotter["attr_map"].set_mask(plotter["unknown"],
                                         mask_color=(0., 0., 0., 0.))

    if compute_newton:
        # Overlay : alpha composite with "interior" layer ie, where it is not
        # masked, we take the value of the "attr" layer
        overlay_mode = Overlay_mode(
                "alpha_composite",
                alpha_mask=plotter["interior"],
                inverse_mask=True
        )
        plotter[base_layer].overlay(plotter[int_layer], overlay_mode=overlay_mode)
    else:
        plotter[base_layer].set_mask(
            plotter["interior"], mask_color=mask_color
        )

    if output_heightmaps:
        plotter["base_hmap"].set_mask(
            plotter["interior"], mask_color=(hmap_mask,)
        )

    plotter.plot()

    # Renaming output to match expected from the Fractal GUI
    layer = plotter[base_layer]
    file_name = "{}_{}".format(type(layer).__name__, layer.postname)
    src_path = os.path.join(fractal.directory, file_name + ".png")
    dest_path = os.path.join(fractal.directory, calc_name + ".png")
    if os.path.isfile(dest_path):
        os.unlink(dest_path)
    os.link(src_path, dest_path)

if __name__ == "__main__":
    plot(**plot_kwargs, batch_params=batch_params)
