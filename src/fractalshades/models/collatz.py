# -*- coding: utf-8 -*-
import numpy as np
import numba

import fractalshades as fs
import fractalshades.utils as fsutils


class Collatz(fs.Fractal):
    def __init__(self, directory):
        """
The Collatz fracal:

.. math::

    z_0 &= c \\\\
    z_{n+1} &= 0.25 (2 + 7 z_n - (2+5z_n) \cos(\pi z_n))
        
This fractal is linked to the Syracuse conjecture.

Parameters
==========
directory : str
    Path for the working base directory
        """
        super().__init__(directory)
        self.potential_kind = "transcendent"

    @fsutils.calc_options
    def base_calc(self, *,
            calc_name: str,
            subset,
            max_iter: int,
            M_divergence: float,
            epsilon_stationnary: float,
    ):
        """
    Basic iterations for Mandelbrot standard set (power 2).

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
        complex_codes = ["zn", "dzndz"]
        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]
        
        def set_state():
            def impl(instance):
                instance.codes = (complex_codes, int_codes, stop_codes)
                instance.complex_type = np.complex128
                instance.potential_M = M_divergence

            return impl

        def initialize():
            @numba.njit
            def numba_init_impl(Z, U, c):
                Z[1] = 1.
                Z[0] = c
            return numba_init_impl

        def iterate():
            Mdiv_sq = self.M_divergence ** 2
            epscv_sq = self.epsilon_stationnary ** 2
            max_iter = self.max_iter

            @numba.njit
            def numba_impl(c, Z, U, stop_reason):
                n_iter = 0

                while True:
                    n_iter += 1

                    if n_iter >= max_iter:
                        stop_reason[0] = 0
                        break

                    cos0 = np.cos(np.pi * Z[0])
                    sin0 = np.sin(np.pi * Z[0])

                    Z[1] = 0.25 * Z[1] * (
                          7. - 5. * cos0 + (2. + 5. * Z[0]) * np.pi * sin0
                    )
                    Z[0] = 0.25 * (2. + 7. * Z[0] - (2. + 5. * Z[0]) * cos0)
    
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

        return {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }
