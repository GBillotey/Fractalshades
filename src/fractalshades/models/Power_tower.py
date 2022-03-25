# -*- coding: utf-8 -*-
import numpy as np
import numba

import fractalshades as fs

#==============================================================================
#==============================================================================

class Power_tower(fs.Fractal):
    def __init__(self, directory):
        """
The tetration fractal - standard precision implementation

.. math::

    z_0 &= 1 \\\\
    z_{n+1} &= c^{z_n} \\\\
            &= \exp(z_n \log(c))

This class implements limit cycle calculation through Newton search

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


    @fs.utils.calc_options
    def newton_calc(self, *,
            calc_name: str,
            subset=None,
            compute_order=True,
            max_order,
            max_newton,
            eps_newton_cv,
    ):
        """
    Newton iterations for the tetration fractal

    Parameters
    ==========
    calc_name : str
         The string identifier for this calculation
    subset : 
        A boolean array-like, where False no calculation is performed
        If `None`, all points are calculated. Defaults to `None`.
    compute_order : bool
        If True, the period of the limit cycle will be computed
        If `None` all order between 1 and `max_order` will be considered.
    max_order : int
        The maximum value for cycle order
    eps_newton_cv : float
        A small float to qualify the convergence of the Newton iteration
        Usually a fraction of a view pixel.
        
    Notes
    =====
    
    .. note::

        The following complex fields will be calculated:
    
        *zr*
            A (any) point belonging to the attracting cycle

        *dzrdz*
            The cycle attractivity (for a convergent cycle it is a complex
            of norm < 1.)

        The following integer field will be calculated:
    
        *order*
            The cycle order
    
        Exit codes are *max_order*, *order_confirmed*.


        """
        complex_codes = ["zr", "dzrdz", "_zn", "_dzndz", "_partial1"]
        zr = 0
        dzrdz = 1
        _zn = 2
        _dzndz = 3
        _partial1 = 4

        int_codes = ["order"]
        stop_codes = ["max_order", "order_confirmed", "overflow"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(np.complex128)

        def initialize():
            @numba.njit
            def numba_init_impl(Z, U, c):
                Z[_partial1] = 1.e6 # bigger than any reasonnable np.abs(z0)
                Z[_zn] = 1.
                Z[_dzndz] = 1.
            return numba_init_impl
        self.initialize = initialize

        def iterate():
            @numba.njit
            def numba_impl(Z, U, c, stop_reason, n_iter):

                logc = np.log(c)

                while True:
                    n_iter += 1

                    if n_iter > max_order:
                        stop_reason[0] = 0
                        break

                    # If n is not a 'partial' for this point it cannot be the 
                    # cycle order : early exit
                    cPzn = np.exp(Z[_zn] * logc)
#                    Z[_dzndz] = Z[_dzndz] * logc * cPzn
                    Z[_zn] = cPzn

                    if np.isinf(Z[_zn]):
                        stop_reason[0] = 2
                        break

                    if not(compute_order):
                        continue

                    m = np.abs(Z[_zn] - 1.)
                    if m <= Z[_partial1].real:
                        Z[_partial1] = m # Cannot assign to the real part
                    else:
                        continue

                    # This is a valid candidate n : 
                    # Run Newton search to find z0 so that f^n(z0) = z0
                    zr_loop = zr0_loop = Z[_zn]

                    for i_newton in range(max_newton):
                        dzrdz_loop = 1.
                        for i in range(n_iter):
                            cPzn = np.exp(zr_loop * logc)
                            dzrdz_loop = dzrdz_loop * logc * cPzn
                            zr_loop = cPzn

                        if np.isinf(zr_loop) or (dzrdz_loop == 1.):
                            newton_cv = False
                            break

                        # zr_loop = zr_loop - 1.
                        delta = (zr_loop - zr0_loop) / (dzrdz_loop - 1.)
                        newton_cv = (np.abs(delta) < eps_newton_cv)
                        zr_loop = (dzrdz_loop * zr0_loop - zr_loop) / (dzrdz_loop - 1.)
                        zr0_loop = zr_loop

                        if newton_cv:
                            break

                    # print("NEWTON", newton_cv, np.abs(dzrdz_loop))
                    # We have a candidate but is it the good one ?
                    is_confirmed = (np.abs(dzrdz_loop) <= 1.) & newton_cv
                    if not(is_confirmed): # not found, same player shoot again
                        continue
                    
                    Z[zr] = zr_loop
                    Z[dzrdz] = dzrdz_loop # attr (cycle attractivity)

                    U[0] = n_iter
                    stop_reason[0] = 1
                    break

                # End of while loop
                return n_iter

            return numba_impl
        self.iterate = iterate

    @fs.utils.interactive_options
    def coords(self, x, y, pix, dps):
        return super().coords(x, y, pix, dps)

#==============================================================================
#==============================================================================

