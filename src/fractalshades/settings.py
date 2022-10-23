# -*- coding: utf-8 -*-
"""
General settings at application-level
"""
# import os
import sys
# this_module is a pointer to the module object instance itself.
this_module = sys.modules[__name__]

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
Chosen as this level squared is ~ precision of float64"""

xrange_zoom_level = 1.e-300
"""Zoom level (dx) which triggers Xrange special branches
"""

no_newton = False
"""Veto all Newton iterations- keep the ref point as is for the full precision
iterations (usually the center of the image). Useful when the center
coordinates of a deep zoom are already known."""

inspect_calc = False
"""Outputs a synthesis files of the calculation done. Useful for debugging"""

chunk_size = 200
"""The size for the basic calculation tile is chunk_size x chunk_size"""

BLA_compression = 3
""" number of BLA levels which are dropped (not stored) """

GUI_image_Mblimit = 0
"""Maximal size of image that will be displayed in the GUI, in Mb - use 0 for
no limit (default). If limit exceeded, the image is still written to disk but 
will not be interactively displayed in the GUI"""

Disk_image_pixlimit = "xxx"
"""PIL sets a default output limit to 89478485 pixels and will raise a 
DecompressionBombWarning if it is exceeded.
Image size (144000000 pixels) exceeds limit of 89478485 pixels, could be
decompression bomb DOS attack.
"""

verbosity = 2
"""
Controls the verbosity for the log messages:
    0: WARNING & higher severity, output to stdout
    1: INFO & higher severity, output to stdout
    2 (default): INFO & higher severity, output to stdout
                 + DEBUG & higher severity, output to a log file
    3: INFO & higher severity, output to stdout
       + all message (incl. NOTSET), output to a log file

Note: Severities in descending order:
CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
"""

postproc_dtype = "float32"
""" The float datatype used for post-processing (float32 | float64)"""

def set_RAM_limit(RAM_limit_Gb):
    """
    Limits the amount of RAM used (mostly for debugging / test purpose)

    Parameters
    ----------
    RAM_limit_Gb
        RAM limit in Gb, or None for no limit
    """
    import resource
    rsrc = resource.RLIMIT_AS
    if RAM_limit_Gb is None:
        byte_limit = resource.RLIM_INFINITY
    else:
        byte_limit =  (2**30) * RAM_limit_Gb  # Gbyte to byte
    resource.setrlimit(rsrc, (byte_limit, byte_limit))


#class Working_directory(os.PathLike):
#    # https://docs.python.org/3/library/os.html#os.PathLike
#    def __init__(self, str_path=None):
#        self.path = str_path
#
#    def __fspath__(self):
#        if self.path is None:
#            raise RuntimeError(
#                "Working directory not specified, please define it through:\n"
#                "fs.working_directory = Working_directory(str_path)"
#            )
#        return self.path

log_directory = None # Working_directory(None)
""" The logging directory for this session """

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
    this_module.figures += [fig]

def get_figures():
    """ push the stagged images """
    return this_module.figures

def close(which):
    """ delete the stagged images """
    if which == "all":
        this_module.figures = list()
    else:
        raise ValueError(which)
