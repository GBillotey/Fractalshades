# -*- coding: utf-8 -*-
"""
General settings at application-level
"""
import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

enable_multithreading = True
"""Turn on or off multithreading (for debugging purpose)"""

optimize_RAM = False
"""If true, memory mappings will be used not only during calculation but also
at post-processing stage. This obviously is at the detriment of speed
but will avoid memory errors for very large images or on system with low RAM.
"""

skip_calc = False
"""If True, no calculation "loop" will be performed. Only the data already
calculated will be post-processed"""

newton_zoom_level = 1.e-5
""" Minimal zoom level for activating Newton search. For lower zooms, a fixed
critical point will be used (0. for mandelbrot)"""

std_zoom_level = 1.e-8
""" Zoom level at which we drop some perturbation techniques optimisation 
Chosen as this level squared is ~ precision of float32
"""

xrange_zoom_level = 1.e-300
""" Minimal zoom level (dx) for activating Xrange special branches
"""

no_newton = False
""" Veto all Newton iterations- keep the ref point as is for the full precision
iterations (usually the center of the image). Useful when the center
coordinates of a deep zoom are already known."""

inspect_calc = False
""" Outputs a synthesis files of the calculation done. Useful for debugging"""

#glitch_off_last_iref = True / not needed in new implementation
#"""Turns glitch correction off for last reference point run."""

chunk_size = 200
"""The size for the basic calculation tile is chunk_size x chunk_size"""

BLA_compression = 3
""" number of BLA levels which are dropped (not stored) """


# output_context: "doc" True if we are building the doc (Pillow output)
# "gui_iter" 0 if not in GUI loop otherwise, start at 1 and increment
# at each GUI plot
output_context = {
    "doc": False,
    "gui_iter": 0,
    "doc_max_width": 800,
    "doc_data_dir": None
}

# Signal to interrupt a calculation (during GUI)
interrupted = False

# figures : module-level container used if output_mode == "Pillow"
# -> the output images are stagged and not directly written to disk.
# This is used when building the documentation
figures = list()

def add_figure(fig):
    """ add a figure to the stagged images """
    this.figures += [fig]

def get_figures():
    """ push the stagged images """
    return this.figures

def close(which):
    """ delete the stagged images """
    if which == "all":
        this.figures = list()
    else:
        raise ValueError(which)
