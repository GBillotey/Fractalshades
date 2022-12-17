# -*- coding: utf-8 -*-
"""
============================
Ligthings: available effects
============================

A selection of the lightings effects which
can be obtained with the `fractalshades.colors.layers.Blinn_lighting` class
and its associated GUI editor. 
"""
import os
import sys

import numpy as np


if sys.version_info < (3, 9):
# See :
# https://discuss.python.org/t/deprecating-importlib-resources-legacy-api/11386/24
    import importlib_resources
else:
    import importlib.resources as importlib_resources 

import PIL
from PIL import ImageDraw, ImageFont, PngImagePlugin

import fractalshades as fs
import fractalshades.colors as fscolors


def plot_lighting(lighting_identifier, plot_dir, nx=600, ny=600):

    lighting = fscolors.lighting_register[lighting_identifier]
    B = lighting._output(nx, ny)
    B[:30, :, :] = np.minimum(255 - (255 - B[:30, :, :]) / 1.5, 255)
    im = PIL.Image.fromarray(B)
    draw = ImageDraw.Draw(im)

    fs_resources = importlib_resources.files("fractalshades")
    with importlib_resources.as_file(
        fs_resources / "data" / "GidoleFont" / "Gidole-Regular.ttf"
    ) as font_file:
        font = ImageFont.truetype(str(font_file.resolve()), size=26)

    draw.text((0,0), lighting_identifier, (0, 0, 0), font=font)
    fs.utils.mkdir_p(plot_dir)

    if fs.settings.output_context["doc"]:
        tag_dict = {
            "Software": "fractalshades " + fs.__version__,
            "lighting example": lighting_identifier
        }
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in tag_dict.items():
            pnginfo.add_text(k, str(v))
        fs.settings.add_figure(fs._Pillow_figure(im, pnginfo))

    else:
        im.save(os.path.join(plot_dir, lighting_identifier + ".png"))

def plot_lightings(plot_dir):

    lighting_register = fscolors.lighting_register
    for lighting_identifier in lighting_register.keys():
        plot_lighting(lighting_identifier, plot_dir)


if __name__ == "__main__":
    # Some magic to get the directory for plotting: with a name that matches
    # the file or a temporary dir if we are building the documentation
    try:
        realpath = os.path.realpath(__file__)
        plot_dir = os.path.splitext(realpath)[0]
        plot_lightings(plot_dir)
    except NameError:
        import tempfile
        with tempfile.TemporaryDirectory() as plot_dir:
            fs.utils.exec_no_output(plot_lightings, plot_dir)

