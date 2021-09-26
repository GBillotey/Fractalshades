# -*- coding: utf-8 -*-
import numpy as np
import numba

import fractalshades as fs
import fractalshades.utils as fsutils


class Mandelbrot(fs.Fractal):
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
            epsilon_stationnary: float,
            datatype):
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
    datatype :
        The dataype for operation on complex. Usually `np.complex128`
        
    Notes
    =====
    The following complex fields will be calculated: *zn* and its
    derivatives (*dzndz*, *dzndc*, *d2zndc2*).
    Exit codes are *max_iter*, *divergence*, *stationnary*.
        """
        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2"]
        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(datatype)

        self.potential_M = M_divergence

        def initialize():
            def func(Z, U, c, chunk_slice):
                Z[3, :] = 0.
                Z[2, :] = 0.
                Z[1, :] = 0.
                Z[0, :] = 0.
            return func
        self.initialize = initialize

        def iterate():
            Mdiv_sq = self.M_divergence ** 2
            epscv_sq = self.epsilon_stationnary ** 2
            max_iter = self.max_iter
            @numba.njit
            def numba_impl(Z, U, c, stop_reason, n_iter):
                if n_iter >= max_iter:
                    stop_reason[0] = 0

                Z[3] = 2 * (Z[3] * Z[0] + Z[2]**2)
                Z[2] = 2 * Z[2] * Z[0] + 1.
                Z[1] = 2 * Z[1] * Z[0] 
                Z[0] = Z[0]**2 + c
                if n_iter == 1:
                    Z[1] = 1.

                if Z[0].real**2 + Z[0].imag ** 2 > Mdiv_sq:
                    stop_reason[0] = 1
                if Z[1].real**2 + Z[1].imag ** 2 < epscv_sq:
                    stop_reason[0] = 2

            return numba_impl
        self.iterate = iterate


    @fsutils.calc_options
    def newton_calc(self, *,
            calc_name: str,
            subset=None,
            known_orders=None,
            max_order,
            max_newton,
            eps_newton_cv,
            datatype):
        """
    Newton iterations for Mandelbrot standard set interior (power 2).

    Parameters
    ==========
    calc_name : str
         The string identifier for this calculation
    subset : 
        A boolean array-like, where False no calculation is performed
        If `None`, all points are calculated. Defaults to `None`.
    known_orders : None | int list 
        If not `None`, only the integer listed of their multiples will be 
        candidates for the cycle order, the other rwill be disregarded.
        If `None` all order between 1 and `max_order` will be considered.
    max_order : int
        The maximum value for ccyle 
    eps_newton_cv : float
        A small float to qualify the convergence of the Newton iteration
        Usually a fraction of a view pixel.
    datatype :
        The dataype for operation on complex. Usually `np.complex128`
        
    Notes
    =====
    
    .. note::

        The following complex fields will be calculated:
    
        *zr*
            A (any) point belonging to the attracting cycle
        *attractivity*
            The cycle attractivity (for a convergent cycle it is a complex
            of norm < 1.)
        *dzrdc*
            Derivative of *zr*
        *dattrdc*
            Derivative of *attractivity*
    
        The following integer field will be calculated:
    
        *order*
            The cycle order
    
        Exit codes are *max_order*, *order_confirmed*.

    References
    ==========
    .. [1] <https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html>

    .. [2] <https://mathr.co.uk/blog/2014-11-02_practical_interior_distance_rendering.html>
        """
        complex_codes = ["zr", "attractivity", "dzrdc", "dattrdc",
                         "_partial", "_zn"]
        int_codes = ["order"]
        stop_codes = ["max_order", "order_confirmed"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(datatype)

        def initialize():
            def func(Z, U, c, chunk_slice):
                Z[5, :] = 0.
                Z[4, :] = 1.e6 # bigger than any reasonnable np.abs(zn)
                Z[3, :] = 0.
                Z[2, :] = 0.
                Z[1, :] = 0.
                Z[0, :] = 0.
            return func
        self.initialize = initialize

        def iterate():
            @numba.njit
            def numba_impl(Z, U, c, stop_reason, n_iter):

                if n_iter > max_order:
                    stop_reason[0] = 0
                    return

                # If n is not a 'partial' for this point it cannot be the 
                # cycle order : early exit
                Z[5] = Z[5]**2 + c
                m = np.abs(Z[5])
                if m < Z[4].real:
                    Z[4] = m # Cannot assign to the real part 
                else:
                    return

                # Early exit if n it is not a multiple of one of the
                # known_orders (provided by the user)
                if known_orders is not None:
                    valid = False
                    for order in known_orders:
                        if  n_iter % order == 0:
                            valid = True
                    if not valid:
                        return

                z0_loop = Z[0]
                dz0dc_loop = Z[2]

                for i_newton in range(max_newton):
                    zr = z0_loop
                    dzrdz = zr * 0. + 1.
                    d2zrdzdc = dzrdz * 0. # == 1. hence : constant wrt c...
                    dzrdc = dz0dc_loop
                    for i in range(n_iter):
                        d2zrdzdc = 2 * (d2zrdzdc * zr + dzrdz * dzrdc)
                        dzrdz = 2. * dzrdz * zr
                        dzrdc = 2 * dzrdc * zr + 1.
                        zr = zr**2 + c

                    delta = (zr - z0_loop) / (dzrdz - 1.)
                    newton_cv = (np.abs(delta) < eps_newton_cv)
                    zz = z0_loop - delta
                    dz0dc_loop = dz0dc_loop - (
                                 (dzrdc - dz0dc_loop) / (dzrdz - 1.) -
                                 (zr - z0_loop) * d2zrdzdc / (dzrdz - 1.)**2)
                    z0_loop = zz
                    if newton_cv:
                        break

                # We have a candidate but is it the good one ?
                is_confirmed = (np.abs(dzrdz) <= 1.) & newton_cv
                if not(is_confirmed): # not found, early exit
                    return

                Z[0] = zr
                Z[1] = dzrdz # attr (cycle attractivity)
                Z[2] = dzrdc
                Z[3] = d2zrdzdc # dattrdc
                U[0] = n_iter
                stop_reason[0] = 1

            return numba_impl
        self.iterate = iterate
