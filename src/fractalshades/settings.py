# -*- coding: utf-8 -*-
"""
General settings for the package
"""
import sys
import multiprocessing

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]


enable_multiprocessing = True # True
multiprocessing_start_method = multiprocessing.get_start_method()
skip_calc = False

# Glitch correction off for last reference point run.
glitch_off_last_iref = True

# size of the basic calculation 'tile' is chunk_size x chunk_size
chunk_size = 200

# Number of bins for plots
HIST_BINS = 100

# Output mode : "png" | "Pillow"
# if "Pillow" 
output_mode = "png"

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
