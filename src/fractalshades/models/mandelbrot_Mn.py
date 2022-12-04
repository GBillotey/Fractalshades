# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import numba

import fractalshades as fs
import fractalshades.utils as fsutils
from fractalshades.postproc import Fractal_array


class Mandelbrot_N(fs.Fractal):
    def __init__(self, directory: str, exponent: int):
        """
A standard power-N Mandelbrot Fractal set implementation.

.. math::

    z_0 &= 0 \\\\
    z_{n+1} &= {z_{n}}^N + c

Parameters
==========
directory : str
    Path for the working base directory
        """
        super().__init__(directory)
        self.exponent = exponent # Needed for serialization

        # default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = exponent
        self.potential_a_d = 1.
        
        # GUI 'badges'
        self.holomorphic = True
        self.implements_fieldlines = True
        self.implements_newton = True
        self.implements_Milnor = True
        self.implements_interior_detection = True


        @numba.njit
        def _zn_iterate(zn, c):
            return zn ** exponent + c
        self.zn_iterate = _zn_iterate


    @fsutils.calc_options
    def calc_std_div(self, *,
            calc_name: str,
            subset: Optional[Fractal_array] = None,
            max_iter: int,
            M_divergence: float,
            epsilon_stationnary: float,
            calc_d2zndz2: bool = False,
            calc_orbit: bool = False,
            backshift: int = 0
    ):
        """
    Basic iterations for the Mandelbrot standard set (power n).
    
    Parameters
    ==========
    calc_name : str
         The string identifier for this calculation
    subset : Optional `fractalshades.postproc.Fractal_array`
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
    calc_d2zndz2: bool
        If True, activates the additional computation of *d2zndz2*, needed 
        only for the alternative normal map shading 'Milnor'.
    calc_orbit: bool
        If True, stores the value of an orbit point @ n of exit - backshift
    backshift: (> 0)
        The number of iteration backward for the stored orbit starting point 
    
    Notes
    =====
    The following complex fields will be calculated: *zn* and its
    derivatives (*dzndz*, *dzndc*).
    If calc_orbit is activated, *zn_orbit* will also be stored
    Exit codes are *max_iter*, *divergence*, *stationnary*.
        """
        complex_codes = ["zn", "dzndz", "dzndc"]
        zn = 0
        dzndz = 1
        dzndc = 2

        # Optionnal output fields
        icode = 3
        d2zndc2 = -1 
        if calc_d2zndz2:
            complex_codes += ["d2zndc2"]
            d2zndc2 = icode
            icode += 1

        i_znorbit = -1
        if calc_orbit:
            complex_codes += ["zn_orbit"]
            i_znorbit = icode
            icode += 1

        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]

        def set_state():
            def impl(instance):
                instance.codes = (complex_codes, int_codes, stop_codes)
                instance.complex_type = np.complex128
                instance.potential_M = M_divergence
                if calc_orbit:
                    instance.backshift = backshift
                else:
                    instance.backshift = None
            return impl

        def initialize():
            @numba.njit
            def numba_init_impl(Z, U, c):
                # Not much to do here
                pass
            return numba_init_impl


        Mdiv_sq = M_divergence ** 2
        epscv_sq = epsilon_stationnary ** 2
        deg = self.exponent 
        deg_m1 = deg - 1 
        deg_m2 = deg - 2

        @numba.njit
        def iterate_once(c, Z, U, stop_reason, n_iter):
            if n_iter >= max_iter:
                stop_reason[0] = 0
                return 1

            _zn = Z[zn]
            if calc_d2zndz2:
                zn_m2 = _zn ** deg_m2
                zn_m1 = zn_m2 * _zn
                zn_m = zn_m1 * _zn
                Z[d2zndc2] = deg * (
                    Z[d2zndc2] * zn_m1
                    + deg_m1 * Z[dzndz] * Z[dzndc] * zn_m2
                )
            else:
                zn_m1 = _zn ** deg_m1
                zn_m = zn_m1 * _zn

            Z[dzndc] = deg * Z[dzndc] * zn_m1 + 1.
            Z[dzndz] = deg * Z[dzndz] * zn_m1 
            Z[zn] = zn_m + c

            if n_iter == 1:
                Z[dzndz] = 1.

            if Z[zn].real ** 2 + Z[zn].imag ** 2 > Mdiv_sq:
                stop_reason[0] = 1
                return 1

            if Z[dzndz].real ** 2 + Z[dzndz].imag ** 2 < epscv_sq:
                stop_reason[0] = 2
                return 1

            return 0

        zn_iterate = self.zn_iterate

        def iterate():
            return fs.core.numba_iterate(
                calc_orbit, i_znorbit, backshift, zn,
                iterate_once, zn_iterate
            )

        return {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }

    @fs.utils.calc_options
    def newton_calc(self, *,
            calc_name: str = "newton_calc",
            subset: Optional[Fractal_array] = None,
            known_orders: Optional[int] = None,
            max_order: int = 500,
            max_newton: int = 20,
            eps_newton_cv: float = 1.e-12
    ):
        """
    Newton iterations for Mandelbrot standard set interior (power n).

    Parameters
    ==========
    calc_name : str
         The string identifier for this calculation
    subset : Optional `fractalshades.postproc.Fractal_array`
        A boolean array-like, where False no calculation is performed
        If `None`, all points are calculated. Defaults to `None`.
    known_orders : None | int list 
        If not `None`, only the integer listed of their multiples will be 
        candidates for the cycle order, the other rwill be disregarded.
        If `None` all order between 1 and `max_order` will be considered.
    max_order : int
        The maximum value tested for cycle order
    eps_newton_cv : float
        A small float to qualify the convergence of the Newton iteration
        Usually a fraction of a view pixel.
        
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
    
        Exit codes are 0: *max_order*, 1: *order_confirmed*.

    References
    ==========
    .. [1] <https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html>

    .. [2] <https://mathr.co.uk/blog/2014-11-02_practical_interior_distance_rendering.html>
        """
        complex_codes = [
            "zr", "attractivity", "dzrdc", "dattrdc", "_partial", "_zn"
        ]
        int_codes = ["order"]
        stop_codes = ["max_order", "order_confirmed"]
        
        deg = self.exponent 
        deg_m1 = deg - 1 
        deg_m2 = deg - 2

        izr = 0
        idzrdz = 1 # attractivity
        idzrdc = 2
        id2zrdzdc = 3 # dattrdc
        i_partial = 4
        i_zn = 5
        iorder = 0

        reason_max_order = 1
        reason_order_confirmed = 2

        def set_state():
            def impl(instance):
                instance.codes = (complex_codes, int_codes, stop_codes)
                instance.complex_type = np.complex128
            return impl

        def initialize():
            @numba.njit
            def numba_init_impl(Z, U, c):
                Z[4] = 1.e6 # Partial, bigger than any reasonnable np.abs(zn)
            return numba_init_impl

        zn_iterate = self.zn_iterate

        @numba.njit
        def iterate_newton_search(d2zrdzdc, dzrdc, dzrdz, zr, c):

            zr_m2 = zr ** deg_m2
            zr_m1 = zr_m2 * zr
            zr_m = zr_m1 * zr

            d2zrdzdc = deg * (
                    d2zrdzdc * zr_m1
                    + deg_m1 * dzrdz * dzrdc * zr_m2
            )
            dzrdz = deg * dzrdz * zr_m1
            dzrdc = deg * dzrdc * zr_m1 + 1.
            zr = zr_m + c

            return (d2zrdzdc, dzrdc, dzrdz, zr)

        def iterate():
            return fs.core.numba_Newton(
                known_orders, max_order, max_newton, eps_newton_cv,
                reason_max_order, reason_order_confirmed,
                izr, idzrdz, idzrdc,  id2zrdzdc, i_partial,  i_zn, iorder,
                zn_iterate, iterate_newton_search
            )
        
        return {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }