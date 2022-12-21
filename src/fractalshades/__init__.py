# -*- coding: utf-8 -*-
__author__ = "G. Billotey"
__license__ = "MIT"
__version__ = "1.0.3"

import numpy as np
import warnings

from .settings import (
    get_figures, close, verbosity, log_directory
)
from .utils import zoom_options, calc_options, interactive_options
from .core import Fractal, Fractal_plotter, _Pillow_figure
from .perturbation import PerturbationFractal
from .log import set_log_handlers

# Disable numpy warnings
if verbosity < 3:
    np.seterr(all="ignore")
    warnings.filterwarnings(
        action="ignore",
        message="invalid value encountered in"
    )
    warnings.filterwarnings(
        action="ignore",
        message="overflow encountered in"
    )
