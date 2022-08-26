# -*- coding: utf-8 -*-
__author__ = "G. Billotey"
__license__ = "MIT"
__version__ = "0.5.6"

import numpy as np
import warnings

from .settings import get_figures, close, verbosity
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

# Starts logging
import logging
logger = logging.getLogger(__name__)
set_log_handlers(verbosity, version_info=__version__)
