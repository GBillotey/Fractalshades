# -*- coding: utf-8 -*-
import numpy as np
import mpmath
import numba

import fractalshades as fs
import fractalshades.mpmath_utils.FP_loop as fsFP

@numba.njit
def sgn(x):
    # Sign with no 0 case (0 mapped to 1)
    if x < 0.:
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
def ddiffabsdX(X, x):
    # Evaluates |X + x| - |X|
    if X >= 0.:
        if (X + x) >= 0.:
            return 0.
        else:
            return -2.
    else:
        if (X + x) <= 0.:
            return 0.
        else:
            return 2.

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
A standard Burning Ship Fractal (power 2). 

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
):
        """
Basic iterations for Burning ship set.

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

Notes
=====
The following complex fields will be calculated: *xn* *yn* and its
derivatives (*dxnda*, *dxndb*, *dynda*, *dyndb*).
Exit codes are *max_iter*, *divergence*.
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

                # End of while loop
                return n_iter

            return numba_impl
        self.iterate = iterate


#==============================================================================
#==============================================================================

class Perturbation_burning_ship(fs.PerturbationFractal):
    
    def __init__(self, directory):
        """
An arbitrary-precision implementation for the Burning ship set (power-2).

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
        self.holomorphic = False


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
        print("** FP_loop")
        print("partial_dict", partial_dict)
        print("xr_dict", xr_dict)
        print("NP_orbit", NP_orbit.dtype, NP_orbit.shape)
        print(NP_orbit)
        
        
        return i, partial_dict, xr_dict


    @fs.utils.calc_options
    def calc_std_div(self, *,
        calc_name: str,
        subset,
        max_iter: int,
        M_divergence: float,
        BLA_params={"eps": 1e-6},
        calc_hessian: bool=True
):
        """
    Perturbation iterations (arbitrary precision) for Burning ship standard set
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
    BLA_params :
        The dictionnary of parameters for Series-Approximation :

        .. list-table:: 
           :widths: 20 80
           :header-rows: 1

           * - keys
             - values 
           * - eps
             - float: relative error criteria (default: 1.e-6)

        if `None`, BLA is not activated.

    calc_hessian: bool
        if True, the derivatives will be claculated allowing distance
        estimation and shading.

    Notes
    -----
        Implementation based on [1]_.

        .. [1] At the Helm of the Burning Ship - Claude Heiland-Allen, 2019
               Proceedings of EVA London 2019 (EVA 2019) 
               <http://dx.doi.org/10.14236/ewic/EVA2019.74>
        """
        self.init_data_types(np.float64)

        # used for potential post-processing
        self.potential_M = M_divergence

        # Complex complex128 fields codes "Z" 
        complex_codes = ["xn", "yn"]
        xn = 0
        yn = 1
        code_int = 2

        if calc_hessian:
            complex_codes += ["dxnda", "dxndb", "dynda", "dyndb"]
            dxnda = code_int + 0
            dxndb = code_int + 1
            dynda = code_int + 2
            dyndb = code_int + 3
            code_int += 4
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
        # H = [dfxdx dfxdy]    [dfx] = H x [dx]
        #     [dfydx dfydy]    [dfy]       [dy]

        @numba.njit
        def _dfxdx(x, y):
            return 2. * x
        self.dfxdx = _dfxdx

        @numba.njit
        def _dfxdy(x, y):
            return -2. * y
        self.dfxdy = _dfxdy

        @numba.njit
        def _dfydx(x, y):
            return 2. * sgn(x) * np.abs(y)
        self.dfydx = _dfydx

        @numba.njit
        def _dfydy(x, y):
            return 2. * sgn(y) * np.abs(x)
        self.dfydy = _dfydy

        #----------------------------------------------------------------------
        # Defines initialize - jitted implementation
        def initialize():
            return fs.perturbation.numba_initialize_BS(
                xn, yn, dxnda, dxndb, dynda, dyndb
            )
        self.initialize = initialize

        #----------------------------------------------------------------------
        # Defines iterate - jitted implementation
        M_divergence_sq = self.M_divergence ** 2

        # Xr triggered for ultra-deep zoom
        xr_detect_activated = self.xr_detect_activated


        @numba.njit
        def p_iter_zn(Z, ref_xn, ref_yn, a, b):
            # Modifies in-place xn, yn
            # print("in p_iter_zn", Z, ref_xn, ref_yn, a, b)
            ref_xyn = ref_xn * ref_yn
            new_xn = (
                Z[xn] * (Z[xn] + 2. * ref_xn) - Z[yn] * (Z[yn] + 2. * ref_yn)
                + a
            )
            new_yn = (
                2. * diffabs(
                    ref_xyn,
                    Z[xn] * Z[yn] + Z[xn] * ref_yn + Z[yn] * ref_xn  # ****
                ) - b
            )
            Z[xn] = new_xn
            Z[yn] = new_yn
            # print("out p_iter_zn", Z, new_xn, new_yn, xn, yn)

        @numba.njit
        def p_iter_hessian(
            Z, ref_xn, ref_yn,            
            ref_dxnda, ref_dxndb, ref_dynda, ref_dyndb
        ):
            """
https://fractalforums.org/fractal-mathematics-and-new-theories/28/perturbation-theory/487/msg3226#msg3226
            """
#            print("in hessian",  Z, Z.shape, xn, yn, dxnda, dxndb, dynda, dyndb)
#            print("ref",  ref_xn, ref_yn, ref_dxnda, ref_dxndb, ref_dynda, ref_dyndb)
            # Modifies in-place the Hessian matrix
            ref_xyn = ref_xn * ref_yn

            _opX = ref_xyn
            d_opX_da = ref_dxnda * ref_yn + ref_xn * ref_dynda
            d_opX_db = ref_dxndb * ref_yn + ref_xn * ref_dyndb

            _opx = Z[xn] * Z[yn] + Z[xn] * ref_yn + Z[yn] * ref_xn
            d_opx_da = (
                Z[dxnda] * Z[yn] + Z[xn] * Z[dynda]
                + Z[dxnda] * ref_yn + Z[xn] * ref_dynda
                + Z[dynda] * ref_xn + Z[yn] * ref_dxnda
            )
            d_opx_db = (
                Z[dxndb] * Z[yn] + Z[xn] * Z[dyndb]
                + Z[dxndb] * ref_yn + Z[xn] * ref_dyndb
                + Z[dyndb] * ref_xn + Z[yn] * ref_dxndb
            )
            _ddiffabsdX = ddiffabsdX(_opX, _opx)
            _ddiffabsdx = ddiffabsdx(_opX, _opx)

            new_dxnda = (
                2. * ((ref_xn + Z[xn]) * Z[dxnda] + ref_dxnda * Z[xn])
                -2. * ((ref_yn + Z[yn]) * Z[dynda] + ref_dynda * Z[yn])
            )
            new_dxndb = (
                2. * ((ref_xn + Z[xn]) * Z[dxndb] + ref_dxndb * Z[xn])
                -2. * ((ref_yn + Z[yn]) * Z[dyndb] + ref_dyndb * Z[yn])
            )
            new_dynda = 2. * (_ddiffabsdX * d_opX_da + _ddiffabsdx * d_opx_da)
            new_dyndb = 2. * (_ddiffabsdX * d_opX_db + _ddiffabsdx * d_opx_db)

            Z[dxnda] = new_dxnda
            Z[dxndb] = new_dxndb
            Z[dynda] = new_dynda
            Z[dyndb] = new_dyndb
#            assert False


        def iterate():
            return fs.perturbation.numba_iterate_BS(
                M_divergence_sq, max_iter, reason_max_iter, reason_M_divergence,
                xr_detect_activated, BLA_activated,
                calc_hessian,
                xn, yn, dxnda, dxndb, dynda, dyndb,
                p_iter_zn, p_iter_hessian
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

        # r_J = r_M ** 0.75 for power 2 Mandelbrot
        sqrt = np.sqrt(nucleus_size)
        sqrtsqrt = np.sqrt(sqrt)
        julia_size = sqrtsqrt * sqrt

        return nucleus_size, julia_size, skew

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
        (
            x_str, y_str, maxiter, radius_pixels, radius_str, dps, order,
            xn_str, yn_str, size_estimates
        ) = self._newton_search(
            x, y, pix, dps, maxiter, radius_pixels
        )
        if size_estimates is not None:
            (nucleus_size, julia_size, skew) = size_estimates
        else:
            nucleus_size = None
            julia_size = None
            skew = np.array(((np.nan, np.nan), (np.nan, np.nan)))

        res_str = f"""
newton_search = {{
    "x_start": "{x_str}",
    "y_start": "{y_str}",
    "maxiter": {maxiter},
    "radius_pixels": {radius_pixels},
    "radius": "{radius_str}",
    "calculation dps": {dps}
    "order": {order}
    "x_nucleus": "{xn_str}",
    "y_nucleus": "{yn_str}",
    "nucleus_size": "{nucleus_size}",
    "julia_size": "{julia_size}",
    "skew_00": "{skew[0, 0]}",
    "skew_01": "{skew[0, 1]}",
    "skew_10": "{skew[1, 0]}",
    "skew_11": "{skew[1, 1]}",
}}
"""
        return res_str
