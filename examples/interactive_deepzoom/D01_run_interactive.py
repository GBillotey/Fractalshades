# -*- coding: utf-8 -*-
"""
=============================================
D01 - Mandelbrot arbitrary-precision explorer
=============================================

This is a template to explore the Mandelbrot set with
arbitrary precision through a GUI.
It features the main postprocessing options (continuous
iteration, distance estimation based shading, field-lines)

Good exploration !


Reference:
`fractalshades.models.Perturbation_mandelbrot`
"""
import os

import fractalshades as fs
import fractalshades.gui.guitemplates
import fractalshades.models as fsm
import fractalshades.gui


def plot(plot_dir):
    """
    Example interactive
    """
    fractal = fsm.Perturbation_mandelbrot(plot_dir)
    deepzooming = fs.gui.guitemplates.deepzooming(fractal)

    gui = fs.gui.Fractal_GUI(deepzooming)
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
