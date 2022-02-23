# -*- coding: utf-8 -*-
import numpy as np
import numba

import fractalshades as fs
import fractalshades.utils as fsutils

@numba.njit
def sgn(x):
    if x < 0:
        return -1
    return 1


class Burning_ship(fs.Fractal):
    def __init__(self, directory):
        """
        A standard power-2 Mandelbrot Fractal. 
        
        Parameters
        ==========
        directory : str
            Path for the working base directory
        """
        super().__init__(directory)
        # default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = 2
        self.potential_a_d = 1.

    @fsutils.calc_options
    def base_calc(self, *,
            calc_name: str,
            subset,
            max_iter: int,
            M_divergence: float,
            epsilon_stationnary: float
):
        """
    Basic iterations for Burning ship standard set (power 2).

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
    derivatives (*dzndz*, *dzndc*, *d2zndc2*).
    Exit codes are *max_iter*, *divergence*, *stationnary*.
        """
        complex_codes = ["xn", "yn", "dxnda", "dxndb", "dynda", "dyndb"]
        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(np.float64)

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

            @numba.njit
            def numba_impl(Z, U, c, stop_reason, n_iter):
                while True:
                    n_iter += 1

                    if n_iter >= max_iter:
                        stop_reason[0] = 0
                        break

                    a = c.real
                    b = c.imag
                    X = Z[0]
                    Y = Z[1]
                    dXdA = Z[2]
                    dXdB = Z[3]
                    dYdA = Z[4]
                    dYdB = Z[5]

                    Z[0] = X ** 2 - Y ** 2 + a
                    Z[1] = 2. * np.abs(X * Y) - b  # was : -b 
                    # Jacobian
                    Z[2] = 2 * (X * dXdA - Y * dYdA) + 1.
                    Z[3] = 2 * (X * dXdB - Y * dYdB)
                    Z[4] = 2 * (np.abs(X) * sgn(Y) * dYdA + sgn(X) * dXdA * np.abs(Y))
                    Z[5] = 2 * (np.abs(X) * sgn(Y) * dYdB + sgn(X) * dXdB * np.abs(Y)) - 1. # was : - 1.

                    if Z[0] ** 2 + Z[1] ** 2 > Mdiv_sq:
                        stop_reason[0] = 1
                        break

#                    if Z[1].real**2 + Z[1].imag ** 2 < epscv_sq:
#                        stop_reason[0] = 2
#                        break

                # End of while loop
                return n_iter

            return numba_impl
        self.iterate = iterate
