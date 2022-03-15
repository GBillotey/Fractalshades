# -*- coding: utf-8 -*-
__author__ = "G. Billotey"
__license__ = "MIT"
__version__ = "0.5.0"

from .settings import get_figures, close
from .utils import zoom_options, calc_options, interactive_options
from .core import Fractal, Fractal_plotter, _Pillow_figure
from .perturbation import PerturbationFractal
