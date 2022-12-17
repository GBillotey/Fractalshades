# -*- coding: utf-8 -*-
"""
General settings at application-level
"""
import sys

enable_multithreading: bool = True
"""Turn on or off multithreading (for debugging purpose)"""

skip_calc: bool = False
"""If True, no calculation "loop" will be performed. Only the data already
calculated will be post-processed"""

newton_zoom_level: float = 1.e-5
""" Minimal zoom level for activating Newton search. For shallower zoom, a
fixed critical point will be used (usually 0.)"""

std_zoom_level: float = 1.e-8
""" Zoom level at which we drop some perturbation techniques optimisation 
Chosen as this level squared is ~ precision of float64 """

xrange_zoom_level: float = 1.e-300
"""Zoom level (dx) which triggers Xrange special branches"""

no_newton: bool = False
"""Veto all Newton iterations- keep the ref point as is for the full precision
iterations (usually the center of the image). Useful when the center
coordinates of a deep zoom are already known."""

inspect_calc: bool = False
"""Outputs a synthesis report file of the calculation done by tiles.
Useful for debugging"""

chunk_size: int  = 200
"""The size for the basic calculation tile is chunk_size x chunk_size"""

BLA_compression: int  = 3
""" number of BLA levels which are dropped (not stored) """

GUI_image_Mblimit = 0
"""Maximal size of image that will be displayed in the GUI, in Mb - use 0 for
no limit (default). If limit exceeded, the image is still written to disk but 
will not be interactively displayed in the GUI"""

def remove_decompression_size_check():
    """PIL sets a default output limit to 89478485 pixels
    (1024 * 1024 * 1024 // 4 // 3) and will raise a 
    DecompressionBombWarning if it is exceeded, and a DecompressionBombError
    for images that are twice this size.

    This function removes this check, use at your own risk. It will not be
    made accessible from the GUI and shall be used in batch mode.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

verbosity: int = 2
"""
Controls the verbosity for the log messages:

    - 0: WARNING & higher severity, output to stdout
    - 1: INFO & higher severity, output to stdout
    - 2 (default): 

        - INFO & higher severity, output to stdout
        - DEBUG & higher severity, output to a log file

    - 3 (highest verbosity):

        - INFO & higher severity, output to stdout
        - ALL message (incl. NOTSET), output to a log file

Note: Severities in descending order:
CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET """

log_directory: str = None
""" The logging directory for this session - as str"""

postproc_dtype: str = "float32"
"""The float datatype used for post-processing ("float32" | "float64").

For the vast majority of output images, the default "float32" should be
enough. However, for image with very high iteration number (usually several
millions of iterations) banding may occur when plotting the continuous
iteration numbers - as the resolution of float32 format is hit.
In this case, the solution is to switch to "float64".
"""

def set_RAM_limit(RAM_limit_Gb: int):
    """
    Limits the amount of RAM used by the program
    (mostly for debugging / test purpose)

    Parameters
    ----------
    RAM_limit_Gb: int
        RAM limit in Gb, or None for no limit
    """
    import resource
    rsrc = resource.RLIMIT_AS
    if RAM_limit_Gb is None:
        byte_limit = resource.RLIM_INFINITY
    else:
        byte_limit =  (2**30) * RAM_limit_Gb  # Gbyte to byte
    resource.setrlimit(rsrc, (byte_limit, byte_limit))


# output_context: "doc" = True if we are building the doc (Pillow output)
# "gui_iter" 0 if not in GUI loop otherwise, start at 1 and increment
# at each GUI plot - (internal).
output_context = {
    "doc": False,
    "gui_iter": 0,
    "doc_max_width": 800,
    "doc_data_dir": None
}

# Signal to interrupt a calculation (during GUI)
interrupted: bool = False

# figures : module-level container used if output_mode == "Pillow"
# -> the output images are stagged and not directly written to disk.
# This is used when building the documentation
# ``this_module`` is a pointer to the module object instance itself.
this_module = sys.modules[__name__]
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
