# -*- coding: utf-8 -*-
import enum

from .mandelbrot_M2 import Mandelbrot, Perturbation_mandelbrot
from .mandelbrot_Mn import Mandelbrot_N
from .burning_ship import (
    Burning_ship,
    Perturbation_burning_ship,
    Perpendicular_burning_ship,
    Perturbation_perpendicular_burning_ship,
    Shark_fin,
    Perturbation_shark_fin
)
from .collatz import Collatz
from .Power_tower import Power_tower

# Enumeration of the available factals (used e.g., for GUI selection lists)
FRACTAL_CLASSES_ENUM = enum.Enum(
    "fractal_classes",
    ("Perturbation_mandelbrot",
     "Perturbation_burning_ship",
     "Perturbation_perpendicular_burning_ship",
     "Perturbation_shark_fin",
     "Mandelbrot",
     "Mandelbrot_N",
     "Burning_ship",
     "Perpendicular_burning_ship",
     "Shark_fin",
     "Power_tower",
     "Collatz"
    ),
    module=__name__
)
