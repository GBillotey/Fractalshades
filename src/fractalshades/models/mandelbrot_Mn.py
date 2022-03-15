# -*- coding: utf-8 -*-
import numpy as np
import numba

import fractalshades as fs
import fractalshades.utils as fsutils


class Mandelbrot_N(fs.Fractal):
    def __init__(self, directory, exponent):
        """
A standard power-n Mandelbrot Fractal set implementation.

Parameters
==========
directory : str
    Path for the working base directory
        """
        super().__init__(directory)
        # default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = exponent
        self.potential_a_d = 1.

    @fsutils.calc_options
    def base_calc(self, *,
            calc_name: str,
            subset,
            max_iter: int,
            M_divergence: float,
            epsilon_stationnary: float,
    ):
        """
Basic iterations for the Mandelbrot standard set (power n).

Parameters
==========
calc_name : str
     The string identifier for this calculation
subset : `fractalshades.postproc.Fractal_array`
    A boolean array-like, where False no calculation is performed
    If `None`, all points are calculated. Defaults to `None`.
max_iter : int
    the maximum iteration number. If reached, the loop is exited with
    exit code "max_iter".
M_divergence : float
    The diverging radius. If reached, the loop is exited with exit code
    "divergence"
epsilon_stationnary : float
    A small float to early exit non-divergent cycles (based on
    cumulated dzndz product). If reached, the loop is exited with exit
    code "stationnary" (Those points should belong to Mandelbrot set
    interior). A typical value is 1.e-3

Notes
=====
The following complex fields will be calculated: *zn* and its
derivatives (*dzndz*, *dzndc*).
Exit codes are *max_iter*, *divergence*, *stationnary*.
        """
        complex_codes = ["zn", "dzndz", "dzndc"]
        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(np.complex128)

        self.potential_M = M_divergence

        def initialize():
            @numba.njit
            def numba_init_impl(Z, U, c):
                # Not much to do here
                pass
            return numba_init_impl
        self.initialize = initialize

        def iterate():
            Mdiv_sq = self.M_divergence ** 2
            epscv_sq = self.epsilon_stationnary ** 2
            max_iter = self.max_iter
            deg = self.potential_d 
            deg_m1 = deg - 1 

            @numba.njit
            def numba_impl(Z, U, c, stop_reason, n_iter):
                while True:
                    n_iter += 1

                    if n_iter >= max_iter:
                        stop_reason[0] = 0
                        break
                    Z0_m1 = Z[0] ** deg_m1
                    Z[2] = deg * Z[2] * Z0_m1 + 1.
                    Z[1] = deg * Z[1] * Z0_m1 
                    Z[0] = Z0_m1 * Z[0] + c
                    if n_iter == 1:
                        Z[1] = 1.

                    if Z[0].real**2 + Z[0].imag ** 2 > Mdiv_sq:
                        stop_reason[0] = 1
                        break

                    if Z[1].real**2 + Z[1].imag ** 2 < epscv_sq:
                        stop_reason[0] = 2
                        break

                # End of while loop
                return n_iter

            return numba_impl
        self.iterate = iterate


