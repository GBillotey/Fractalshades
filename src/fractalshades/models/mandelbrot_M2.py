# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import mpmath
import numba

import fractalshades as fs
import fractalshades.mpmath_utils.FP_loop as fsFP
from fractalshades.postproc import Fractal_array


@numba.njit
def M2_iterate(zn, c):
    return zn * zn + c

class Mandelbrot(fs.Fractal):

    def __init__(self, directory: str):
        """
A standard power-2 Mandelbrot Fractal. 

.. math::

    z_0 &= 0 \\\\
    z_{n+1} &= z_{n}^2 + c

Parameters
==========
directory : str
    Path for the working base directory
"""
        super().__init__(directory)

        # Default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = 2
        self.potential_a_d = 1.
        self.potential_M_cutoff = 1000. # Minimum M for valid potential

        # GUI 'badges'
        self.holomorphic = True
        self.implements_dzndc = "always"
        self.implements_fieldlines = True
        self.implements_newton = True
        self.implements_Milnor = True
        self.implements_interior_detection = "always"
        self.implements_deepzoom = False

        # Orbit 'on the fly' calculation
        self.zn_iterate = M2_iterate

        # Cache for numba compiled implementations
        self._numba_cache = {}
        self._numba_iterate_cache = {
            "calc_std_div": ((), None),
            "newton_calc": ((), None),
        } # args, numba_impl


    @fs.utils.calc_options
    def calc_std_div(self, *,
            calc_name: str = "base_calc",
            subset: Optional[Fractal_array] = None,
            max_iter: int = 10000,
            M_divergence: float = 1000.,
            epsilon_stationnary: float = 0.01,
            calc_d2zndc2: bool = False,
            calc_orbit: bool = False,
            backshift: int = 0
    ):
        """
    Basic iterations for Mandelbrot standard set (power 2).

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
        A small criteria (typical range 0.01 to 0.001) used to detect earlier
        points belonging to a minibrot, based on dzndz1 value.
        If reached, the loop is exited with exit code "stationnary"
    calc_d2zndc2:
        If True, activates the additional computation of *d2zndc2*, needed 
        only for the alternative normal map shading 'Milnor'.
    calc_orbit: bool
        If True, stores the value of an orbit point @ exit - orbit_shift
    backshift: int (> 0)
        The number of iteration backward for the stored orbit starting point

    Notes
    =====
    The following complex fields will be calculated: *zn* and its
    derivatives (*dzndc*, *d2zndc2*[optionnal], *dzndz* [optionnal]).
    If calc_orbit is activated, *zn_orbit* will also be stored
    Exit codes are 0: *max_iter*, 1: *divergence*, 2: *stationnary*.
"""
        complex_codes = ["zn", "dzndz", "dzndc"]
        zn = 0
        dzndz = 1
        dzndc = 2

        # Optionnal output fields
        icode = 3
        d2zndc2 = -1 
        if calc_d2zndc2:
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
            def numba_impl(Z, U, c):
                # Not much to do here
                pass
            return numba_impl


        Mdiv_sq = M_divergence ** 2
        epscv_sq = epsilon_stationnary ** 2

        iterate_once = self.from_numba_cache(
            "iterate_once",
            zn, dzndz, dzndc, d2zndc2, calc_d2zndc2,
            max_iter, Mdiv_sq, epscv_sq
        )
        zn_iterate = self.zn_iterate

        def iterate():
            new_args = (
                calc_orbit, i_znorbit, backshift, zn,
                iterate_once, zn_iterate
            )
            args, numba_impl = self._numba_iterate_cache["calc_std_div"]
            if new_args == args:
                return numba_impl

            numba_impl = fs.core.numba_iterate(*new_args)
            self._numba_iterate_cache["calc_std_div"] = (new_args, numba_impl)
            return numba_impl


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
    Newton iterations for Mandelbrot standard set interior (power 2).

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
        iterate_newton_search = self.from_numba_cache(
            "iterate_newton_search",
            None, None, None, None, None, None, None, None
        )

        def iterate():
            new_args = (
                known_orders, max_order, max_newton, eps_newton_cv,
                reason_max_order, reason_order_confirmed,
                izr, idzrdz, idzrdc,  id2zrdzdc, i_partial, i_zn, iorder,
                zn_iterate, iterate_newton_search
            )
            args, numba_impl = self._numba_iterate_cache["calc_std_div"]
            if new_args == args:
                return numba_impl

            numba_impl = fs.core.numba_Newton(*new_args)
            self._numba_iterate_cache["calc_std_div"] = (new_args, numba_impl)
            return numba_impl


        return {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }


    def from_numba_cache(self, key, zn, dzndz, dzndc, d2zndc2, calc_d2zndc2,
                         max_iter, Mdiv_sq, epscv_sq):
        """ Returns the numba implementation if exists to avoid unnecessary
        recompilation"""
        cache = self._numba_cache
        full_key = (key, zn, dzndz, dzndc, d2zndc2, calc_d2zndc2,
                    max_iter, Mdiv_sq, epscv_sq)

        try:
            return cache[full_key]

        except KeyError:
            if key == "iterate_once":
                @numba.njit
                def numba_impl(c, Z, U, stop_reason, n_iter):
                    if n_iter >= max_iter:
                        stop_reason[0] = 0
                        return 1

                    if calc_d2zndc2:
                       Z[d2zndc2] = 2 * (Z[d2zndc2] * Z[zn] + Z[dzndc] ** 2)

                    Z[dzndc] = 2 * Z[dzndc] * Z[zn] + 1.
                    Z[dzndz] = 2 * Z[dzndz] * Z[zn] 
                    Z[zn] = Z[zn] ** 2 + c

                    if n_iter == 1:
                        Z[dzndz] = 1.

                    if Z[zn].real ** 2 + Z[zn].imag ** 2 > Mdiv_sq:
                        stop_reason[0] = 1
                        return 1

                    if Z[dzndz].real ** 2 + Z[dzndz].imag ** 2 < epscv_sq:
                        stop_reason[0] = 2
                        return 1

                    return 0

            elif key == "iterate_newton_search":
                @numba.njit
                def numba_impl(d2zrdzdc, dzrdc, dzrdz, zr, c):
                    d2zrdzdc = 2 * (d2zrdzdc * zr + dzrdz * dzrdc)
                    dzrdz = 2. * dzrdz * zr
                    dzrdc = 2 * dzrdc * zr + 1.
                    zr = zr * zr + c
                    return (d2zrdzdc, dzrdc, dzrdz, zr)

            else:
                raise NotImplementedError(key)

            cache[full_key] = numba_impl
            return numba_impl


    @fs.utils.interactive_options
    def coords(self, x, y, pix, dps):
        return super().coords(x, y, pix, dps)

#==============================================================================
#==============================================================================

class Perturbation_mandelbrot(fs.PerturbationFractal):
    
    def __init__(self, directory: str):
        """
An arbitrary precision power-2 Mandelbrot Fractal. 

.. math::

    z_0 &= 0 \\\\
    z_{n+1} &= z_{n}^2 + c

This class implements arbitrary precision for the reference orbit, ball method
period search, newton search, perturbation method, chained billinear
approximations.

Parameters
----------
directory : str
    Path for the working base directory
        """
        super().__init__(directory)
        # Sets default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = 2
        self.potential_a_d = 1.
        self.potential_M_cutoff = 1000. # Minimum M for valid potential

        # Set parameters for the full precision orbit
        self.critical_pt = 0.
        self.FP_code = "zn"

        # GUI 'badges'
        self.holomorphic = True
        self.implements_dzndc = "user"
        self.implements_fieldlines = True
        self.implements_newton = False
        self.implements_Milnor = False
        self.implements_interior_detection = "user"
        self.implements_deepzoom = True

        # Orbit 'on the fly' calculation
        self.zn_iterate = M2_iterate
        
        # Cache for numba compiled implementations
        self._numba_cache = {}
        self._numba_initialize_cache = ((), None) # args, numba_impl
        self._numba_iterate_cache = ((), None) # args, numba_impl


    def FP_loop(self, NP_orbit, c0):
        """
        The full precision loop ; fills in place NP_orbit
        """
        xr_detect_activated = self.xr_detect_activated

        max_orbit_iter = (NP_orbit.shape)[0] - 1
        M_divergence = self.M_divergence

        x = c0.real
        y = c0.imag
        seed_prec = mpmath.mp.prec
        (i, partial_dict, xr_dict
         ) = fsFP.perturbation_mandelbrot_FP_loop(
            NP_orbit.view(dtype=np.float64),
            xr_detect_activated,
            max_orbit_iter,
            M_divergence * 2, # to be sure ref exit after close points
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec
        )
        return i, partial_dict, xr_dict

    @fs.utils.calc_options
    def calc_std_div(self, *,
        calc_name: str,
        subset,
        max_iter: int,
        M_divergence: float,
        epsilon_stationnary: float,
        BLA_eps: float=1e-6,
        interior_detect: bool=False,
        calc_dzndc: bool=True,
        calc_orbit: bool = False,
        backshift: int = 0
    ):
        """
Perturbation iterations (arbitrary precision) for Mandelbrot standard set
(power 2).

Parameters
==========
calc_name : str
     The string identifier for this calculation
subset : 
    A boolean array-like, where False no calculation is performed
    If `None`, all points are calculated. Defaults to `None`.
max_iter : int
    the maximum iteration number. If reached, the loop is exited with
    exit code "max_iter".
M_divergence : float
    The diverging radius. If reached, the loop is exited with exit code
    "divergence"
epsilon_stationnary : float
    Used only if `interior_detect` parameter is set to True
    A small criteria (typical range 0.01 to 0.001) used to detect earlier
    points belonging to a minibrot, based on dzndz1 value.
    If reached, the loop is exited with exit code "stationnary"
BLA_eps : None | float
    Relative error criteriafor BiLinear Approximation (default: 1.e-6)
    if `None` BLA is not activated.
interior_detect : bool
    If True, activates interior points early detection.
    This will trigger the computation of an additionnal quantity (`dzndz`), so
    shall be activated only when a mini fills a significative part of the view.
calc_dzndc: bool
    If True, activates the computation of an additionnal quantities (`dzndc`),
    used for distance estimation plots and for shading (normal map calculation)
calc_orbit: bool
    If True, stores the value of an orbit point @ exit - backshift
backshift: int (> 0)
    The number of iteration backward for the stored orbit starting point
"""
        complex_codes = ["zn"]
        zn = 0
        code_int = 0

        calc_dzndz = interior_detect
        if calc_dzndz:
            code_int += 1
            complex_codes += ["dzndz"]
            dzndz = code_int
        else:
            dzndz = -1

        if calc_dzndc:
            code_int += 1
            complex_codes += ["dzndc"]
            dzndc = code_int
        else:
            dzndc = -1

        if calc_orbit:
            code_int += 1
            complex_codes += ["zn_orbit"]
            i_znorbit = code_int
        else:
            i_znorbit = -1

        # Integer int32 fields codes "U" 
        int_codes = ["ref_cycle_iter"] # Position in ref orbit

        # Stop codes
        stop_codes = ["max_iter", "divergence", "stationnary"]
        reason_max_iter = 0
        reason_M_divergence = 1
        reason_stationnary = 2


        #----------------------------------------------------------------------
        # Define the functions used for BLA approximation
        # BLA triggered ?
        BLA_activated = (
            (BLA_eps is not None)
            and (self.dx < fs.settings.newton_zoom_level)
        )

        dfdz = self.from_numba_cache("dfdz", zn, dzndz, dzndc)
        dfdc = self.from_numba_cache("dfdc", zn, dzndz, dzndc)

        def set_state():
            def impl(instance):
                instance.complex_type = np.complex128
                instance.potential_M = M_divergence
                instance.codes = (complex_codes, int_codes, stop_codes)

                instance.calc_dZndz = interior_detect
                instance.calc_dZndc = calc_dzndc or BLA_activated
                instance.dfdz = dfdz
                instance.dfdc = dfdc
            return impl

        #----------------------------------------------------------------------
        # Defines initialize - jitted implementation
        def initialize():
            new_args = (zn, dzndc, dzndz)
            args, numba_impl = self._numba_initialize_cache
            if new_args == args:
                return numba_impl

            numba_impl = fs.perturbation.numba_initialize(*new_args)
            self._numba_initialize_cache = (new_args, numba_impl)
            return numba_impl

        #----------------------------------------------------------------------
        # Defines iterate - jitted implementation
        M_divergence_sq = M_divergence ** 2
        epsilon_stationnary_sq = epsilon_stationnary ** 2
        # Xr triggered for ultra-deep zoom
        xr_detect_activated = self.xr_detect_activated

        p_iter_zn = self.from_numba_cache("p_iter_zn", zn, dzndz, dzndc)
        p_iter_dzndz = self.from_numba_cache("p_iter_dzndz", zn, dzndz, dzndc)
        p_iter_dzndc = self.from_numba_cache("p_iter_dzndc", zn, dzndz, dzndc)
        zn_iterate = self.zn_iterate


        def iterate():
            new_args = (
                max_iter, M_divergence_sq, epsilon_stationnary_sq,
                reason_max_iter, reason_M_divergence, reason_stationnary,
                xr_detect_activated, BLA_activated,
                zn, dzndc, dzndz,
                p_iter_zn, p_iter_dzndz, p_iter_dzndc,
                calc_dzndc, calc_dzndz,
                calc_orbit, i_znorbit, backshift, zn_iterate
            )
            args, numba_impl = self._numba_iterate_cache
            if new_args == args:
                return numba_impl

            numba_impl = fs.perturbation.numba_iterate(*new_args)
            self._numba_iterate_cache = (new_args, numba_impl)
            return numba_impl

        return {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }


    def from_numba_cache(self, key, zn, dzndz, dzndc):
        """ Returns the numba implementation if exists to avoid unnecessary
        recompilation"""
        cache = self._numba_cache
        try:
            return cache[(key, zn, dzndz, dzndc)]

        except KeyError:
            if key == "dfdz":
                @numba.njit
                def numba_impl(z):
                    return 2. * z
            elif key == "dfdc":
                @numba.njit
                def numba_impl(z):
                    return 1.
            elif key == "p_iter_zn":
                @numba.njit
                def numba_impl(Z, ref_zn, c):
                    Z[zn] = Z[zn] * (Z[zn] + 2. * ref_zn) + c
            elif key == "p_iter_dzndz":
                @numba.njit
                def numba_impl(Z, ref_zn, ref_dzndz):
                    Z[dzndz] = 2. * (
                        (ref_zn + Z[zn]) * Z[dzndz] + ref_dzndz * Z[zn]
                    )
            elif key == "p_iter_dzndc":
                @numba.njit
                def numba_impl(Z, ref_zn, ref_dzndc):
                    Z[dzndc] = 2. * (
                        (ref_zn + Z[zn]) * Z[dzndc] + ref_dzndc * Z[zn]
                    )
            else:
                raise NotImplementedError(key)

            cache[(key, zn, dzndz, dzndc)] = numba_impl
            return numba_impl


#------------------------------------------------------------------------------
# Newton search & other related methods

    @staticmethod
    def _ball_method(c, px, maxiter, M_divergence):
        """ Order 1 ball method: Cython wrapper"""
        x = c.real
        y = c.imag
        seed_prec = mpmath.mp.prec
        
        order = fsFP.perturbation_mandelbrot_ball_method(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            str(px).encode('utf8'),
            maxiter,
            M_divergence
        )
        if order == -1:
            return None
        return order


    @staticmethod
    def find_nucleus(c, order, eps_pixel, max_newton=None, eps_cv=None):
        """
        Run Newton search to find z0 so that f^n(z0) == 0 : Cython wrapper

        Includes a "divide by undesired roots" technique so that solutions
        using divisors of n are disregarded.
        
        # eps_pixel = self.dx * (1. / self.nx)
        """
        if order is None:
            raise ValueError("order shall be defined for Newton method")

        x = c.real
        y = c.imag
        seed_prec = mpmath.mp.prec
        if max_newton is None:
            max_newton = 80
        if eps_cv is None:
            eps_cv = mpmath.mpf(val=(2, -seed_prec))

        is_ok, val = fsFP.perturbation_mandelbrot_find_nucleus(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            order,
            max_newton,
            str(eps_cv).encode('utf8'),
            str(eps_pixel).encode('utf8'),
        )

        return is_ok, val


    @staticmethod
    def find_any_nucleus(c, order, eps_pixel, max_newton=None, eps_cv=None):
        """
        Run Newton search to find z0 so that f^n(z0) == 0 : Cython wrapper
        """
        if order is None:
            raise ValueError("order shall be defined for Newton method")

        x = c.real
        y = c.imag
        seed_prec = mpmath.mp.prec
        if max_newton is None:
            max_newton = 80
        if eps_cv is None:
            eps_cv = mpmath.mpf(val=(2, -seed_prec))
        
        is_ok, val = fsFP.perturbation_mandelbrot_find_any_nucleus(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            order,
            max_newton,
            str(eps_cv).encode('utf8'),
            str(eps_pixel).encode('utf8'),
        )

        return is_ok, val


    @staticmethod
    def _nucleus_size_estimate(c0, order):
        """
Nucleus size estimate

Parameters
----------
c0 :
    position of the nucleus
order :
    cycle order

Returns
-------
nucleus_size : 
    size estimate of the nucleus
julia_size : 
    size estimate of the Julian embedded set

https://mathr.co.uk/blog/2016-12-24_deriving_the_size_estimate.html

Structure in the parameter dependence of order and chaos for the quadratic map
Brian R Hunt and Edward Ott
J. Phys. A: Math. Gen. 30 (1997) 7067–7076

https://fractalforums.org/fractal-mathematics-and-new-theories/28/miniset-and-embedded-julia-size-estimates/912/msg4805#msg4805
    julia size estimate : r_J = r_M ** ((n+1)*(n-1)/n**2)
"""
        x = c0.real
        y = c0.imag
        seed_prec = mpmath.mp.prec
        nucleus_size = fsFP.perturbation_mandelbrot_nucleus_size_estimate(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            order
        )
        nucleus_size = np.abs(nucleus_size)

        # r_J = r_M ** 0.75 for power 2 Mandelbrot
        sqrt = np.sqrt(nucleus_size)
        sqrtsqrt = np.sqrt(sqrt)
        julia_size = sqrtsqrt * sqrt

        return nucleus_size, julia_size

#==============================================================================
# GUI : "interactive options"
#==============================================================================
    @fs.utils.interactive_options
    def coords(self, x, y, pix, dps):
        return super().coords(x, y, pix, dps)

    @fs.utils.interactive_options
    def ball_method_order(self, x, y, pix, dps, maxiter: int=100000,
                          radius_pixels: int=25):
        return super().ball_method_order(x, y, pix, dps, maxiter,
                    radius_pixels)

    @fs.utils.interactive_options
    def newton_search(self, x, y, pix, dps, maxiter: int=100000,
                      radius_pixels: int=3):
        return super().newton_search(x, y, pix, dps, maxiter, radius_pixels)
