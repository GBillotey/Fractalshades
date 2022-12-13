# -*- coding: utf-8 -*-
"""
============================================================
22 - "Shark Fin" escape-time fractal
============================================================

Shark Fin is yet another `abs` variant, which features characteristic swirly
structures.
Here, an embeded Julia set at a depth of 4.73e-709

Reference:
`fractalshades.models.Perturbation_shark_fin`

"""
import os
import typing

import numpy as np
import mpmath

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
    fractal = fsm.Perturbation_burning_ship(
            plot_dir,
            flavor="Shark fin"
    )
    calc_name = 'test'
    _1 = 'Zoom parameters'
    x = '-0.51589620268627970432443972140466026618218061060835255763287480145041290920221218736270428245220746279618295095166798752620726428609211826274613117053264928657361477467234927967848753770166731064805065463419553477373233269591169614800220483380564022654439634169986994659158374011085207126571652818183508697495230387717223608030522514636708124947753134519935604246289612244797927127587933109583025775968223378485067259810292595540274443866416810747689111293571346662942340520825497910806468006677501978335474464494846513169232172970646598024886328933772273261902456292147592478159136755604313683591557546924552731872428993587388001253830610215019930920128066319351540606478859235188128959774129483187488417147576690962535'
    y = '-0.66245287866929999372606685018770977042741695086446507741696933368387025939056546346094155505184636005256929748124385822155668880701474708417984582422272710613772844669490417841138223472001735816407958513797076939299186620556823065549014718232791996093184517670844492605187891162197307252314413279558393819847771389763599825046453903796922405420550932465191160133017417295941396971611900071562514890047201047120102195410731923629108732235286316679314710301128526293690085499066374212739634013484802464773762266855871308625831096062738587245346740647723535571648059438999329620958795738080623529817798169870967107413952074751872860339286384952167472237089529593701827385040502112441301761365329882413960477126592518217051'
    dx = '4.730268188484633e-709'
    
    xy_ratio = 1.8
    theta_deg = 180.0
    dps = 719
    nx = 2400
    _1b = 'Skew parameters /!\\ Re-run when modified!'
    has_skew = True
    skew_00 = 1.2082219388541917
    skew_01 = 0.30450938998799054
    skew_10 = -0.24552000996844184
    skew_11 = 0.7657838529335976
    _2 = 'Calculation parameters'
    max_iter = 400000
    _3 = 'Bilinear series parameters'
    eps = 1e-06
    _4 = 'Plotting parameters: base field'
    base_layer = 'continuous_iter'
    interior_color = (0.3333333432674408, 1.0, 0.49803921580314636)
    colormap = fs.colors.cmap_register["classic"]
    invert_cmap = True
    DEM_min = 1e-10
    zmin = 0.8 * (-12.720348358154297) + 0.2 * (-12.7184476852417)
    zmax = -12.7184476852417
    _5 = 'Plotting parameters: shading'
    shade_kind = 'glossy'
    gloss_intensity = 10.0
    light_angle_deg = -135.0
    light_color = (1.0, 1.0, 1.0)
    gloss_light_color = (1.0, 1.0, 1.0)

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Plotting function
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def func(
         fractal,
         calc_name: str=calc_name,

         _1: fsgui.separator="Zoom parameters",
         x: mpmath.mpf=x,
         y: mpmath.mpf=y,
         dx: mpmath.mpf=dx,
         xy_ratio: float=xy_ratio,
         theta_deg: float=theta_deg,
         dps: int=dps,
         nx: int=nx,

         _1b: fsgui.separator="Skew parameters /!\ Re-run when modified!",
         has_skew: bool=has_skew,
         skew_00: float=1.,
         skew_01: float=0.,
         skew_10: float=0.,
         skew_11: float=1.,

         _2: fsgui.separator="Calculation parameters",
         max_iter: int=max_iter,

         _3: fsgui.separator="Bilinear series parameters",
         eps: float=eps,

         _4: fsgui.separator="Plotting parameters: base field",
         base_layer: typing.Literal[
                 "continuous_iter",
                 "distance_estimation"
         ]=base_layer,
         interior_color=(0.1, 0.1, 0.1),
         colormap: fscolors.Fractal_colormap=colormap,
         invert_cmap: bool=False,
         DEM_min: float=1.e-6,
         zmin: float=zmin,
         zmax: float=zmax,

         _5: fsgui.separator="Plotting parameters: shading",
         shade_kind: typing.Literal["None", "standard", "glossy"]=shade_kind,
         gloss_intensity: float=10.,
         light_angle_deg: float=65.,
         light_color=(1.0, 1.0, 1.0),
         gloss_light_color=(1.0, 1.0, 1.0),
    ):


        fractal.zoom(precision=dps, x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
             theta_deg=theta_deg, projection="cartesian",
             has_skew=has_skew, skew_00=skew_00, skew_01=skew_01,
             skew_10=skew_10, skew_11=skew_11
        )

        fractal.calc_std_div(
            calc_name=calc_name,
            subset=None,
            max_iter=max_iter,
            M_divergence=1.e70,
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
                "DEM_map", max_slope=40, output=False
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
        dps,
        nx,
        _1b,
        has_skew,
        skew_00,
        skew_01,
        skew_10,
        skew_11,
        _2,
        max_iter,
        _3,
        eps,
        _4,
        base_layer,
        interior_color,
        colormap,
        invert_cmap,
        DEM_min,
        zmin,
        zmax,
        _5,
        shade_kind,
        gloss_intensity,
        light_angle_deg,
        light_color,
        gloss_light_color
    )


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
