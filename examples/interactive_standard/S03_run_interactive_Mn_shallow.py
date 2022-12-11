# -*- coding: utf-8 -*-
"""
========================================================================
S03 - Mandelbrot power n explorer - Standard precision
========================================================================

This is a simple template to explore the power n > 2 Mandelbrot
set with the GUI.
Resolution limited to approx 1.e-13 due to double (float64) precision

Reference:
`fractalshades.models.Mandelbrot_N`
"""
import os

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.gui
import fractalshades.gui.guitemplates
import fractalshades.gui.guimodel

def plot(plot_dir):
    """
    Example interactive
    """
    fractal = fsm.Mandelbrot_N(plot_dir, exponent=6)
    zooming = fs.gui.guitemplates.std_zooming(fractal)
    gui = fs.gui.guimodel.Fractal_GUI(zooming)
    gui.show()


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
