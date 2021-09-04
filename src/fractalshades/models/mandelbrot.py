# -*- coding: utf-8 -*-
import numpy as np
import numba

import fractalshades as fs
import fractalshades.utils as fsutils


class Mandelbrot(fs.Fractal):
    def __init__(self, *args, **kwargs):
        # default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = 2
        self.potential_a_d = 1.
        super().__init__(*args, **kwargs)

    @fsutils.calc_options
    def base_calc(self, *,
            calc_name: str,
            subset,
            max_iter: int,
            M_divergence: float,
            epsilon_stationnary: float,
            datatype):
        """
        Basic iterations for Mandelbrot standard set
        """
        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2"]
        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(np.complex128)

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
                    stop_reason[0] = 0 # TODO should we return here

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
        Newton iterations for Mandelbrot standard set interior
        
        References:
          https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html
          https://mathr.co.uk/blog/2014-11-02_practical_interior_distance_rendering.html
          https://en.wikibooks.org/wiki/Fractals/Iterations_in_the_complex_plane/Mandelbrot_set_interior
          https://fractalforums.org/fractal-mathematics-and-new-theories/28/interior-colorings/3352
        """
        complex_codes = ["zr", "attractivity", "dzrdc", "dattrdc", "d2attrdc2"]
        int_codes = ["order"]
        stop_codes = ["max_order", "order_confirmed"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(np.complex128)

        def initialize():
            def func(Z, U, c, chunk_slice):
                print("#################### initialize")
                Z[4, :] = 0.
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

                # n is the candidate cycle order. Early exit if it is not a
                # multiple of one of the known_orders
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
