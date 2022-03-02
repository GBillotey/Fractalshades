# -*- coding: utf-8 -*-
import numpy as np
import mpmath
import numba

import fractalshades as fs
import fractalshades.mpmath_utils.FP_loop as fsFP

@numba.njit
def sgn(x):
    # Sign with no 0 case (0 mapped to 1)
    if x < 0:
        return -1.
    return 1.

@numba.njit
def diffabs(X, x):
    # Evaluates |X + x| - |X|
    if X >= 0.:
        if (X + x) >= 0.:
            return x
        else:
            return -(2 * X + x)
    else:
        if (X + x) <= 0.:
            return -x
        else:
            return (2 * X + x)

@numba.njit
def ddiffabsdx(X, x):
    # Evaluates |X + x| - |X|
    if X >= 0.:
        if (X + x) >= 0.:
            return 1.
        else:
            return -1.
    else:
        if (X + x) <= 0.:
            return -1.
        else:
            return 1.

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

    @fs.utils.calc_options
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
                    Z[1] = 2. * np.abs(X * Y) - b
                    # Jacobian
                    Z[2] = 2 * (X * dXdA - Y * dYdA) + 1.
                    Z[3] = 2 * (X * dXdB - Y * dYdB)
                    Z[4] = 2 * (np.abs(X) * sgn(Y) * dYdA + sgn(X) * dXdA * np.abs(Y))
                    Z[5] = 2 * (np.abs(X) * sgn(Y) * dYdB + sgn(X) * dXdB * np.abs(Y)) - 1.

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


#==============================================================================
#==============================================================================

class Perturbation_burning_ship(fs.PerturbationFractal):
    
    def __init__(self, directory):
        """
        An arbitrary precision power-2 Mandelbrot Fractal. 

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
        # Set parameters for the full precision orbit
        self.critical_pt = 0.
        self.FP_code = ["xn", "yn"]


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
         ) = fsFP.perturbation_BS_FP_loop(
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
        datatype,
        subset,
        max_iter: int,
        M_divergence: float,
        epsilon_stationnary: float,
        SA_params=None,
        BLA_params={"eps": 1e-6},
#        interior_detect: bool=False,
        calc_dzndc: bool=True
):
        """
    Perturbation iterations (arbitrary precision) for Mandelbrot standard set
    (power 2).

    Parameters
    ==========
    calc_name : str
         The string identifier for this calculation
    datatype :
        The dataype for operation on complex. Usually `np.complex128`
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
    SA_params :
        The dictionnary of parameters for Series-Approximation :

        .. list-table:: 
           :widths: 20 80
           :header-rows: 1

           * - keys
             - values 
           * - cutdeg
             - int: polynomial degree used for approximation (default: 32)
           * - SA_err
             - float: relative error criteria (default: 1.e-6)

        if `None` SA is not activated.
    interior_detect : bool
        EXPERIMENTAL for perturbation.
        If True activates interior point detection
        
    References
    ==========
    .. [1] <https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html>

    .. [2] <http://www.fractalforums.com/announcements-and-news>
        
        """
        self.init_data_types(datatype)

        # used for potential post-processing
        self.potential_M = M_divergence

        # Complex complex128 fields codes "Z" 
        complex_codes = ["xn", "yn"]
        xn = 0
        yn = 1
        code_int = 2

        if calc_dzndc:
            code_int += 1
            complex_codes += ["dxnda", "dxndb", "dynda", "dyndb"]
            dxnda = code_int
            dxndb = code_int
            dynda = code_int
            dyndb = code_int
        else:
            dxnda = -1
            dxndb = -1
            dynda = -1
            dyndb = -1


        # Integer int32 fields codes "U" 
        int_codes = ["ref_cycle_iter"] # Position in ref orbit

        # Stop codes
        stop_codes = ["max_iter", "divergence"] #, "stationnary"]
        reason_max_iter = 0
        reason_M_divergence = 1
        # reason_stationnary = 2

        self.codes = (complex_codes, int_codes, stop_codes)
        print("###self.codes", self.codes)

        #----------------------------------------------------------------------
        # Define the functions used for BLA approximation
        # BLA triggered ?
        BLA_activated = (
            (BLA_params is not None) 
            and (self.dx < fs.settings.newton_zoom_level)
        )

#        @numba.njit
#        def _f(x, y):
#            return z * z
#        self.f = _f

        @numba.njit
        def _dfdx(x, y):
            return (2. * x, 2. * sgn(x) * abs(y))
        self.dfdx = _dfdx

        @numba.njit
        def _dfdy(x, y):
            return (-2. * y, 2. * sgn(y) * abs(x))
        self.dfdy = _dfdy


        #----------------------------------------------------------------------
        # Defines initialize - jitted implementation
        def initialize():
            return fs.perturbation.numba_initialize_BS(
                    xn, yn,
                    dxnda, dxndb, dynda, dyndb,
            )
        self.initialize = initialize

        #----------------------------------------------------------------------
        # Defines iterate - jitted implementation
        M_divergence_sq = self.M_divergence ** 2
        epsilon_stationnary_sq = self.epsilon_stationnary ** 2
        # Xr triggered for ultra-deep zoom
        xr_detect_activated = self.xr_detect_activated


        @numba.njit
        def p_iter_zn(Z, ref_zn, c):
            ref_xn = ref_zn.real
            ref_yn = ref_zn.imag
            ref_xyn = ref_zn.real * ref_zn.imag
            return (
                (
                    Z[xn] * (Z[xn] + 2. * ref_xn)
                    - Z[yn] * (Z[yn] + 2. * ref_yn)
                    + c.real
                ),
                    2 * diffabs(
                        ref_xyn,
                        ref_xyn + Z[xn] * ref_xn + Z[yn] * ref_yn
                ) + c.imag
            )

        @numba.njit
        def p_iter_dzndc(Z, ref_zn, ref_dzndx, ref_dzndy):
            ref_xn = ref_zn.real
            ref_yn = ref_zn.imag
            ref_xyn = ref_zn.real * ref_zn.imag
            return (
                
            )
                    
                    
                    
                    #2. * ((ref_zn + Z[zn]) * Z[dzndc] + ref_dzndc * Z[zn])

        def iterate():
            return fs.perturbation.numba_iterate_BS(
                M_divergence_sq, max_iter, reason_max_iter, reason_M_divergence,
                epsilon_stationnary_sq, interior_detect_activated, reason_stationnary,
                SA_activated, xr_detect_activated, BLA_activated,
                calc_dzndc,
                zn, dzndz, dzndc,
                p_iter_zn, p_iter_dzndc
            )
        self.iterate = iterate


#------------------------------------------------------------------------------
# Newton search & other related methods

    @staticmethod
    def _ball_method(c, px, maxiter, M_divergence):
        """ Order 1 ball method: Cython wrapper"""
        x = c.real
        y = c.imag
        seed_prec = mpmath.mp.prec

        order = fsFP.perturbation_BS_ball_method(
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
        raise NotImplementedError("Divide by undesired roots technique "
                                  "not implemented for burning ship")


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
        
        is_ok, val = fsFP.perturbation_BS_find_any_nucleus(
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

Parameters:
-----------
c0 :
    position of the nucleus
order :
    cycle order

Returns:
--------
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
        nucleus_size, skew = fsFP.perturbation_BS_nucleus_size_estimate(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            order
        )
        print("raw nucleus_size", nucleus_size)
        print("raw skew", skew)
        return nucleus_size, skew

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
