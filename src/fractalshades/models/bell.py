# -*- coding: utf-8 -*-
import numpy as np
import numba
import mpmath

import fractalshades as fs
import fractalshades.mpmath_utils.FP_loop as fsFP

#==============================================================================
#==============================================================================


@numba.njit(fastmath=True, error_model="numpy")
def f_df_bell(z):
    if z == 0.:
        return z, z
    else:
        zm2 = 1. / (z * z) # np.reciprocal(z * z)
        f = np.exp(-zm2)
        df = 2. * zm2 / z * f
        return f, df

@numba.njit(fastmath=True, error_model="numpy")
def f_bell(z):
    if z == 0.:
        return z
    else:
        zm2 = 1. / (z * z) # np.reciprocal(z * z)
        return np.exp(-zm2)


class Bellbrot(fs.Fractal):
    def __init__(self, directory):
        """
A standard power-N Mandelbrot Fractal set implementation.

.. math::

    z_0 &= 0 \\\\
    z_{n+1} &= \exp {1 / z_{n}}^2 + c

Parameters
==========
directory : str
    Path for the working base directory
        """
        super().__init__(directory)
        # default values used for postprocessing (potential)
        self.potential_kind = "transcendent"
        self.potential_d = None
        self.potential_a_d = None

    @fs.utils.calc_options
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
            @numba.njit(fastmath=True, error_model="numpy")
            def numba_init_impl(Z, U, c):
                # Not much to do here
                pass
            return numba_init_impl
        self.initialize = initialize

        def iterate():
            Mdiv_sq = self.M_divergence ** 2
            epscv_sq = self.epsilon_stationnary ** 2
            max_iter = self.max_iter

            @numba.njit(fastmath=True, error_model="numpy")
            def numba_impl(Z, U, c, stop_reason, n_iter):
                while True:
                    n_iter += 1

                    if n_iter >= max_iter:
                        stop_reason[0] = 0
                        break
                    f, df  = f_df_bell(Z[0])
                    Z[2] = df * Z[2] + 1.
                    Z[1] = df * Z[1]
                    Z[0] = f + c
                    if n_iter == 1:
                        Z[1] = 1.

                    if Z[0].real**2 + Z[0].imag ** 2 > Mdiv_sq:
                        stop_reason[0] = 1
                        break

                    if Z[1].real**2 + Z[1].imag ** 2 < epscv_sq:
                        stop_reason[0] = 2
                        break

#                    if Z[0].real**2 + Z[0].imag ** 2 < epscv_sq:
#                        stop_reason[0] = 2
#                        break

                # End of while loop
                return n_iter

            return numba_impl
        self.iterate = iterate

#==============================================================================
#==============================================================================
class Perturbation_bellbrot(fs.PerturbationFractal):
    
    def __init__(self, directory):
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
        self.potential_kind = "transcendent"
        # Set parameters for the full precision orbit
        self.critical_pt = 0.
        self.FP_code = "zn"
        self.holomorphic = True

    def FP_loop(self, NP_orbit, c0):
        """
        The full precision loop ; fills in place NP_orbit
        """
        xr_detect_activated = self.xr_detect_activated
        # Parameters borrowed from last "@fs.utils.calc_options" call
        calc_options = self.calc_options
        max_orbit_iter = (NP_orbit.shape)[0] - 1
        M_divergence = calc_options["M_divergence"]

        x = c0.real
        y = c0.imag
        seed_prec = mpmath.mp.prec
        (i, partial_dict, xr_dict
         ) = fsFP.perturbation_bellbrot_FP_loop(
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
        BLA_params={"eps": 1e-6},
        interior_detect: bool=False,
        calc_dzndc: bool=True
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
        EXPERIMENTAL for perturbation.
        A small float to early exit non-divergent cycles (based on
        cumulated dzndz product). If reached, the loop is exited with exit
        code "stationnary" (Those points should belong to Mandelbrot set)
        Used only if interior_detect is True
    BLA_params :
        The dictionnary of parameters for Bilinear Approximations :

        .. list-table:: 
           :widths: 20 80
           :header-rows: 1

           * - keys
             - values 
           * - SA_err
             - float: relative error criteria (default: 1.e-6)

        if `None` BLA is not activated.
    interior_detect : bool
        EXPERIMENTAL for perturbation.
        If True activates interior point detection

        """
        self.init_data_types(np.complex128)

        # used for potential post-processing
        self.potential_M = M_divergence

        # Complex complex128 fields codes "Z" 
        complex_codes = ["zn"]
        zn = 0
        code_int = 0

        interior_detect_activated = (
            interior_detect 
            and (self.dx > fs.settings.newton_zoom_level)
        )
        if interior_detect_activated:
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

        # Integer int32 fields codes "U" 
        int_codes = ["ref_cycle_iter"] # Position in ref orbit

        # Stop codes
        stop_codes = ["max_iter", "divergence", "stationnary"]
        reason_max_iter = 0
        reason_M_divergence = 1
        reason_stationnary = 2

        self.codes = (complex_codes, int_codes, stop_codes)
        print("###self.codes", self.codes)

        #----------------------------------------------------------------------
        # Define the functions used for BLA approximation
        # BLA triggered ?
        BLA_activated = (
            (BLA_params is not None) 
            and (self.dx < fs.settings.newton_zoom_level)
        )

        @numba.njit(fastmath=True, error_model="numpy")
        def _dfdz(z): # TODO (z, c)
            f, df = f_df_bell(z)
            return df

        self.dfdz = _dfdz

        @numba.njit(fastmath=True, error_model="numpy")
        def _dfdc(z): # TODO (z, c)
            return 1.
        self.dfdc = _dfdc

        #----------------------------------------------------------------------
        # Defines SA_loop via a function factory - jitted implementation
        # SA triggered ?
        SA_activated = False
        self.SA_loop = None

        #----------------------------------------------------------------------
        # Defines initialize - jitted implementation
        def initialize():
            return fs.perturbation.numba_initialize(zn, dzndz, dzndc)
        self.initialize = initialize

        #----------------------------------------------------------------------
        # Defines iterate - jitted implementation
        M_divergence_sq = self.M_divergence ** 2
        epsilon_stationnary_sq = self.epsilon_stationnary ** 2
        # Xr triggered for ultra-deep zoom
        xr_detect_activated = self.xr_detect_activated

        @numba.njit(fastmath=True, error_model="numpy")
        def p_iter_zn(Z, ref_zn, c):
            # Handle special case with null reference
            if ref_zn == 0.:
                Z[zn] = f_bell(Z[zn]) + c
#                zm2 = 1. / (Z[zn] * Z[zn])
#                Z[zn] = np.exp(-zm2) + c
                return

            F = f_bell(ref_zn)
            num = Z[zn] * (2 * ref_zn + Z[zn])
            denom = (ref_zn * (ref_zn + Z[zn])) ** 2
            delta_f = np.expm1(num / denom)
            Z[zn] = F * delta_f + c
            # Z[zn] = Z[zn] * (Z[zn] + 2. * ref_zn) + c

#        @numba.njit
#        def p_iter_dzndz(Z):
#            # Only used at low zoom - assumes dZndz == 0 
#            Z[dzndz] = 2. * (Z[zn] * Z[dzndz])
        p_iter_dzndz = None

        @numba.njit(fastmath=True, error_model="numpy")
        def p_iter_dzndc(Z, ref_zn, ref_dzndc):
            # Z[dzndc] = 2. * ((ref_zn + Z[zn]) * Z[dzndc] + ref_dzndc * Z[zn])
#            print("enter p_iter_dzndc")

            # Handle special case with null reference
            if ref_zn == 0.:
#                print("fflag 001")
                f, df = f_df_bell(Z[zn])
#                zm2 = 1. / (Z[zn] * Z[zn])
#                print("fflag 002")
#                f = np.exp(-zm2)
#                print("fflag 003")
#                df = 2. * zm2 / Z[zn] * f
#                print("fflag 004")
                Z[dzndc] = Z[dzndc] * df
#                print("fflag 005")
                return

            F, dF = f_df_bell(ref_zn)
            dFdc = dF * ref_dzndc

            num = Z[zn] * (2 * ref_zn + Z[zn])
#            print("fflag 1")
            dnumdc = 2. * ((ref_zn + Z[zn]) * Z[dzndc] + ref_dzndc * Z[zn])
            denom = (ref_zn * (ref_zn + Z[zn])) ** 2
#            print("fflag 2")
            ddenomdc = (
                2 * (ref_zn * (ref_zn + Z[zn]))
                * ( 2 * ref_dzndc * ref_zn
                   + ref_dzndc * Z[zn]
                   + ref_zn * Z[dzndc])
            )
#            print("fflag 3")
            op = num / denom
#            print("fflag 4")
            dopdc = dnumdc / denom - ddenomdc * num / (denom * denom)
#            print("fflag 5")
            delta_f = np.expm1(op)
#            print("fflag 6")
            ddelta_f_dc = (delta_f + 1.) * dopdc
#            print("fflag 7")

            Z[dzndc] = dFdc * delta_f + F * ddelta_f_dc
#            print("fflag 8")

        def iterate():
            return fs.perturbation.numba_iterate(
                M_divergence_sq, max_iter, reason_max_iter, reason_M_divergence,
                epsilon_stationnary_sq, interior_detect_activated, reason_stationnary,
                SA_activated, xr_detect_activated, BLA_activated,
                calc_dzndc,
                zn, dzndz, dzndc,
                p_iter_zn, p_iter_dzndz, p_iter_dzndc
            )
        self.iterate = iterate


#------------------------------------------------------------------------------
# Newton search & other related methods

    @staticmethod
    def _ball_method(c, px, maxiter, M_divergence):
        """ Order 1 ball method: Cython wrapper"""
        return None


#    @staticmethod
#    def find_nucleus(c, order, eps_pixel, max_newton=None, eps_cv=None):
#        """
#        Run Newton search to find z0 so that f^n(z0) == 0 : Cython wrapper
#
#        Includes a "divide by undesired roots" technique so that solutions
#        using divisors of n are disregarded.
#        
#        # eps_pixel = self.dx * (1. / self.nx)
#        """
#        if order is None:
#            raise ValueError("order shall be defined for Newton method")
#
#        x = c.real
#        y = c.imag
#        seed_prec = mpmath.mp.prec
#        if max_newton is None:
#            max_newton = 80
#        if eps_cv is None:
#            eps_cv = mpmath.mpf(val=(2, -seed_prec))
#
#        is_ok, val = fsFP.perturbation_mandelbrot_find_nucleus(
#            str(x).encode('utf8'),
#            str(y).encode('utf8'),
#            seed_prec,
#            order,
#            max_newton,
#            str(eps_cv).encode('utf8'),
#            str(eps_pixel).encode('utf8'),
#        )
#
#        return is_ok, val


#    @staticmethod
#    def find_any_nucleus(c, order, eps_pixel, max_newton=None, eps_cv=None):
#        """
#        Run Newton search to find z0 so that f^n(z0) == 0 : Cython wrapper
#        """
#        if order is None:
#            raise ValueError("order shall be defined for Newton method")
#
#        x = c.real
#        y = c.imag
#        seed_prec = mpmath.mp.prec
#        if max_newton is None:
#            max_newton = 80
#        if eps_cv is None:
#            eps_cv = mpmath.mpf(val=(2, -seed_prec))
#        
#        is_ok, val = fsFP.perturbation_mandelbrot_find_any_nucleus(
#            str(x).encode('utf8'),
#            str(y).encode('utf8'),
#            seed_prec,
#            order,
#            max_newton,
#            str(eps_cv).encode('utf8'),
#            str(eps_pixel).encode('utf8'),
#        )
#
#        return is_ok, val


#    @staticmethod
#    def _nucleus_size_estimate(c0, order):
#        """
#Nucleus size estimate
#
#Parameters:
#-----------
#c0 :
#    position of the nucleus
#order :
#    cycle order
#
#Returns:
#--------
#nucleus_size : 
#    size estimate of the nucleus
#julia_size : 
#    size estimate of the Julian embedded set
#
#https://mathr.co.uk/blog/2016-12-24_deriving_the_size_estimate.html
#
#Structure in the parameter dependence of order and chaos for the quadratic map
#Brian R Hunt and Edward Ott
#J. Phys. A: Math. Gen. 30 (1997) 7067–7076
#
#https://fractalforums.org/fractal-mathematics-and-new-theories/28/miniset-and-embedded-julia-size-estimates/912/msg4805#msg4805
#    julia size estimate : r_J = r_M ** ((n+1)*(n-1)/n**2)
#"""
#        x = c0.real
#        y = c0.imag
#        seed_prec = mpmath.mp.prec
#        nucleus_size = fsFP.perturbation_mandelbrot_nucleus_size_estimate(
#            str(x).encode('utf8'),
#            str(y).encode('utf8'),
#            seed_prec,
#            order
#        )
#        print("raw nucleus_size", nucleus_size)
#        nucleus_size = np.abs(nucleus_size)
#
#        # r_J = r_M ** 0.75 for power 2 Mandelbrot
#        sqrt = np.sqrt(nucleus_size)
#        sqrtsqrt = np.sqrt(sqrt)
#        julia_size = sqrtsqrt * sqrt
#
#        return nucleus_size, julia_size

#==============================================================================
# GUI : "interactive options"
#==============================================================================
    @fs.utils.interactive_options
    def coords(self, x, y, pix, dps):
        return super().coords(x, y, pix, dps)

#    @fs.utils.interactive_options
#    def ball_method_order(self, x, y, pix, dps, maxiter: int=100000,
#                          radius_pixels: int=25):
#        return super().ball_method_order(x, y, pix, dps, maxiter,
#                    radius_pixels)
#
#    @fs.utils.interactive_options
#    def newton_search(self, x, y, pix, dps, maxiter: int=100000,
#                      radius_pixels: int=3):
#        return super().newton_search(x, y, pix, dps, maxiter, radius_pixels)