# -*- coding: utf-8 -*-
"""
===============================================
D02 - Burning Ship arbitrary-precision explorer
===============================================

This is a template to explore the Burning Ship set with
arbitrary precision through a GUI.
It features the main postprocessing options (continuous
iteration, distance estimation based shading)

As the Burning ship is a non-holomorphic fractal, some areas can exibit a heavy
skew. This explorer allows you to use an unskewing matrice and continue
the exploration.
A suitable  unskew matrice is usually given by the influencing mini-ship, which
you can get as part of a Newton search results : right click on the image and 
select "Newton search".
When the skew parameters are changed, hit rerun to continue the exploration.

(This class also implements Burning ship variants)

Good exploration !

Reference:
`fractalshades.models.Perturbation_burning_ship`
"""
import os

import fractalshades as fs
import fractalshades.models as fsm
import fractalshades.gui
import fractalshades.gui.guitemplates
import fractalshades.gui.guimodel


def plot(plot_dir):
    """
    GUI-interactive Burning-ship deepzoom example
    """
    fractal = fsm.Perturbation_burning_ship(plot_dir)
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
