# -*- coding: utf-8 -*-
"""
===========================================
16 - Tetration (power tower) zoom: "Spring"
===========================================

Coloring based on attracting cycle order and attractivity.
This zoom is quite shallow however already features complex structures

Note: due to the long running time and high antialising needed, this image
has been precomputed.

Reference:
`fractalshades.models.Power_tower`
"""

import os
import numpy as np

import fractalshades as fs
import fractalshades.models as fsm

import fractalshades.colors as fscolors
from fractalshades.postproc import (
    Postproc_batch,
    Raw_pp,
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Virtual_layer,
)



def plot(plot_dir):

    fractal = fsm.Power_tower(plot_dir)
    calc_name = 'test'

    x = 1.4073514628685297
    y = -3.362771439936941
    dx = 0.0005297704717647524
    xy_ratio = 1.8
    theta_deg = 135.0

    nx = 800
    compute_order = True
    max_order = 10000
    eps_newton_cv = 1e-12

    interior_color = (
        0.40784314274787903,
        0.5882353186607361,
        0.2823529541492462
    )
    colormap = colormap = fscolors.cmap_register["lily"]
    zmin = 0.9 * 1.0376274585723877 + 0.1 * 1.2009856700897217
    zmax = 0.3 * 1.0376274585723877 + 0.7 * 1.2009856700897217


    fractal.zoom(x=x, y=y, dx=dx, nx=nx, xy_ratio=xy_ratio,
         theta_deg=theta_deg, projection="cartesian")

    fractal.newton_calc(
        calc_name=calc_name,
        subset=None,
        compute_order=compute_order,
        max_order=max_order,
        max_newton=20,
        eps_newton_cv=eps_newton_cv
    )


    layer_name = "cycle_order"

    pp = Postproc_batch(fractal, calc_name)
    pp.add_postproc(layer_name, Raw_pp("order"))
    pp.add_postproc("attr", Raw_pp("dzrdz", func=lambda x: np.abs(x)))
    pp.add_postproc("interior", Raw_pp("stop_reason",
                    func=lambda x: x != 1)
    )

    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))

    plotter.add_layer(Color_layer(
            layer_name,
            func=lambda x: np.log(np.log(np.log(x + 1.) + 1.) + 1.),
            colormap=colormap,
            probes_z=[zmin, zmax],
            output=True))
    plotter[layer_name].set_mask(plotter["interior"],
                                 mask_color=interior_color)
    plotter.add_layer(Virtual_layer("attr", func=None, output=False))
    
    plotter[layer_name].set_twin_field(plotter["attr"], 0.3)

    plotter.plot()
    
    # Renaming output to match expected from the Fractal GUI
    layer = plotter[layer_name]
    file_name = "{}_{}".format(type(layer).__name__, layer.postname)
    src_path = os.path.join(fractal.directory, file_name + ".png")
    dest_path = os.path.join(fractal.directory, calc_name + ".png")
    if os.path.isfile(dest_path):
        os.unlink(dest_path)
    os.link(src_path, dest_path) 


def _plot_from_data(plot_dir):
    # Private function only used when building fractalshades documentation
    # This example takes too long too run to autogenerate the image for the
    # gallery each - so just grabbing the file from the html doc static path
    import PIL

    data_path = fs.settings.output_context["doc_data_dir"]
    im = PIL.Image.open(os.path.join(data_path, "tetration_spring.jpg"))
    rgb_im = im.convert('RGB')
    tag_dict = {"Software": "fractalshades " + fs.__version__,
                "example_plot": "tetration_spring"}
    pnginfo = PIL.PngImagePlugin.PngInfo()
    for k, v in tag_dict.items():
        pnginfo.add_text(k, str(v))
    if fs.settings.output_context["doc"]:
        fs.settings.add_figure(fs._Pillow_figure(rgb_im, pnginfo))
    else:
        # Should not happen
        raise RuntimeError()


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
            fs.utils.exec_no_output(_plot_from_data, plot_dir)