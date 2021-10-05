# -*- coding: utf-8 -*-
"""
General settings for the package
"""
import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]


enable_multiprocessing = True # True
skip_calc = False

# Glitch correction off for last reference point run.
glitch_off_last_iref = True

# size of the basic calculation 'tile' is chunk_size x chunk_size
chunk_size = 200

# Number of bins for plots
HIST_BINS = 100

# output_context: "doc" True if we are building the doc (Pillow output)
# "gui_iter" 0 if not in GUI loop otherwise, start at 1 and increment
# at each GUI plot
output_context = {
    "doc": False,
    "gui_iter": 0
}

# Signal to interrupt a calculation
interrupted = False

# Minimal zoom level for activating Newton search
newton_zoom_level = 1.e-8

# figures : module-level container used if output_mode == "Pillow"
# -> the output images are stagged and not directly written to disk.
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
