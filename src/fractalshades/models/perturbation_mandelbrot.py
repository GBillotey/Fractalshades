# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
import mpmath
import numba

import fractalshades as fs
#import fractalshades.numpy_utils.xrange as fsx
import fractalshades.numpy_utils.numba_xr as fsxn
import fractalshades.utils as fsutils
import fractalshades.settings as fssettings
import fractalshades.mpmath_utils.FP_loop as fsFP


class Perturbation_mandelbrot(fs.PerturbationFractal):
    
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
        self.FP_code = "zn"


    def FP_loop(self, NP_orbit, c0):
        """
        The full precision loop ; fills in place NP_orbit
        """
        xr_detect_activated = self.xr_detect_activated
        # Parameters borrowed from last "@fsutils.calc_options" call
        calc_options = self.calc_options
        max_orbit_iter = (NP_orbit.shape)[0] - 1# calc_options["max_iter"]
        M_divergence = calc_options["M_divergence"]

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


    @fsutils.calc_options
    def calc_std_div(self, *,
        calc_name: str,
        datatype,
        subset,
        max_iter: int,
        M_divergence: float,
        epsilon_stationnary: float,
        SA_params={"cutdeg": 32, "eps": 1e-6},
        interior_detect: bool=False,
        calc_dzndc: bool=True):
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
        complex_codes = ["zn"]
        zn = 0
        code_int = 0

        interior_detect_activated = (
            interior_detect 
            and (self.dx > fssettings.newton_zoom_level)
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


        # Defines SA_loop via a function factory - jitted implementation
        def SA_loop():
            @numba.njit
            def impl(Pn, n_iter, Zn_xr, kcX):
                """ Series Approximation loop
                Note that derivatives w.r.t dc will be deduced directly from the
                S.A polynomial.
                """
                return Pn * (Pn + 2. * Zn_xr) + kcX
            return impl
        self.SA_loop = SA_loop

        # Defines initialize via a function factory
        def initialize():

            @numba.njit
            def numba_init_impl(c_xr, Z, Z_xr, Z_xr_trigger, U, P, kc, dx_xr,
                                n_iter_init):
                """
                ... mostly the SA
                Initialize 'in place' at n_iter_init :
                    Z[zn], Z[dzndz], Z[dzndc]
                """
                if P is not None:
                    # Apply the  Series approximation step
                    U[0] = n_iter_init
                    c_scaled = c_xr / kc[0]
                    Z[zn] = fsxn.to_standard(P.__call__(c_scaled))

                    if fs.perturbation.need_xr(Z[zn]):
                        Z_xr_trigger[zn] = True
                        Z_xr[zn] = P.__call__(c_scaled)

                    if dzndc != -1:
                        P_deriv = P.deriv()
                        deriv_scale =  dx_xr[0] / kc[0]
                        Z[dzndc] = fsxn.to_standard(
                            P_deriv.__call__(c_scaled) * deriv_scale
                        )
                if (dzndz != -1):
                    Z[dzndz] = 1.
            return numba_init_impl
        self.initialize = initialize

        # Defines iterate via a function factory - jitted implementation
        def iterate():
            M_divergence_sq = self.M_divergence ** 2
            epsilon_stationnary_sq = self.epsilon_stationnary ** 2

            # Interior detection for very shallow zoom
            interior_detect_activated = (
                interior_detect 
                and (self.dx > fssettings.newton_zoom_level)
            )
            # SA triggered for deeper zoom
            SA_activated = (
                (SA_params is not None) 
                and (self.dx < fssettings.newton_zoom_level)
            )
            # Xr triggered for ultra-deep zoom
            xr_detect_activated = self.xr_detect_activated
            

            @numba.njit
            def numba_impl(
                c, c_xr, Z, Z_xr, Z_xr_trigger, U, stop, n_iter,
                Zn_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
            ):
                """
                dz(n+1)dc   <- 2. * dzndc * zn + 1.
                dz(n+1)dz   <- 2. * dzndz * zn
                z(n+1)      <- zn**2 + c 
                
                Termination codes
                0 -> max_iter reached ('interior')
                1 -> M_divergence reached by np.abs(zn)
                2 -> dzndz stationnary ('interior detection')
                """
                # print("in numba_impl", ref_div_iter, max_iter, ref_order, len(Zn_path))
                # in numba_impl 100001 100000 4611686018427387904
                # in numba_impl 100001 100000 2123
                # Usually len(Zn_path) == ref_order

                refpath_ptr = np.zeros((2,), dtype=np.int32)
                ref_is_xr = np.zeros((2,), dtype=numba.bool_)
                ref_zn_xr = fs.perturbation.Xr_template.repeat(2)

                # Wrapping if we reach the cycle order
                if U[0] >= ref_order:
                    U[0] = U[0] % ref_order

                while True:
                    n_iter += 1

                    #==============================================================
                    # Load reference point value @ U[0]
                    # refpath_ptr = [prev_idx, curr_xr]
                    if xr_detect_activated:
                        ref_zn = fs.perturbation.ref_path_get(
                            Zn_path, U[0],
                            has_xr, ref_index_xr, ref_xr, refpath_ptr,
                            ref_is_xr, ref_zn_xr, 0
                        )
                    else:
                        ref_zn = Zn_path[U[0]]

                    #==============================================================
                    # Pertubation iter block
                    #--------------------------------------------------------------
                    # dzndc subblock
                    if calc_dzndc:
                    # This is an approximation as we do not store the ref pt
                    # derivatives 
                    # Full term would be :
                    # Z[dzndc] =  2 * (
                    #    (ref_path_zn + Z[zn]) * Z[dzndc]
                    #    + Z[zn] * ref_path_dzndc
                        Z[dzndc] = 2. * (ref_zn + Z[zn]) * Z[dzndc]
                        if not(SA_activated) and (n_iter == 1):
                            # Non-null term needed to 'kick-off'
                            Z[dzndc] = 1.

                    #--------------------------------------------------------------
                    # Interior detection - Used only at low zoom level
                    if interior_detect_activated and (n_iter > 1):
                        Z[dzndz] = 2. * (Z[zn] * Z[dzndz])

                    #--------------------------------------------------------------
                    # zn subblok
                    if xr_detect_activated:
                        # We shall pay attention to overflow
                        if not(Z_xr_trigger[0]):
                            # try a standard iteration
                            old_zn = Z[zn]
                            Z[zn] = Z[zn] * (Z[zn] + 2. * ref_zn)
                            if fs.perturbation.need_xr(Z[zn]):
                                # Standard iteration underflows
                                # standard -> xrange conversion
                                Z_xr[zn] = fsxn.to_Xrange_scalar(old_zn)
                                Z_xr_trigger[0] = True

                        if Z_xr_trigger[0]:
                            if (ref_is_xr[0]):
                                Z_xr[zn] = Z_xr[zn] * (Z_xr[zn] + 2. * ref_zn_xr[0])
                            else:
                                Z_xr[zn] = Z_xr[zn] * (Z_xr[zn] + 2. * ref_zn)
                            # xrange -> standard conversion
                            Z[zn] = fsxn.to_standard(Z_xr[zn])

                            # Unlock trigger if we can...
                            Z_xr_trigger[0] = fs.perturbation.need_xr(Z[zn])
                    else:
                        # No risk of underflow, normal perturbation interation
                        Z[zn] = Z[zn] * (Z[zn] + 2. * ref_zn) + c

                    #==============================================================
                    if n_iter >= max_iter:
                        stop[0] = reason_max_iter
                        break

                    # Interior points detection
                    if interior_detect_activated:
                        bool_stationnary = (
                            Z[dzndz].real ** 2 + Z[dzndz].imag ** 2
                                < epsilon_stationnary_sq)
                        if bool_stationnary:
                            stop[0] = reason_stationnary
                            break

                    #==============================================================
                    # ZZ = "Total" z + dz
                    U[0] += 1
                    if U[0] >= ref_order:
                        U[0] = U[0] % ref_order



                    if xr_detect_activated:
                        ref_zn_next = fs.perturbation.ref_path_get(
                            Zn_path, U[0],
                            has_xr, ref_index_xr, ref_xr, refpath_ptr,
                            ref_is_xr, ref_zn_xr, 1
                        )
                    else:
                        ref_zn_next = Zn_path[U[0]]


                    # computation involve doubles only
                    ZZ = Z[zn] + ref_zn_next
                    full_sq_norm = ZZ.real**2 + ZZ.imag**2
    
                    # Flagged as 'diverging'
                    bool_infty = (full_sq_norm > M_divergence_sq)
                    if bool_infty:
                        stop[0] = reason_M_divergence
                        break

                    # Glitch correction - reference point diverging
                    if (U[0] >= ref_div_iter - 1):
                        # Rebasing - we are already big no underflow risk
                        U[0] = 0
                        Z[zn] = ZZ
                        continue

                    # Glitch correction -  "dynamic glitch"
                    bool_dyn_rebase = (
                        (abs(ZZ.real) <= abs(Z[zn].real))
                        and (abs(ZZ.imag) <= abs(Z[zn].imag))
                    )
                    if bool_dyn_rebase:
                        if xr_detect_activated and Z_xr_trigger[0]:
                            # Can we *really* rebase ??
                            # Note: if Z[zn] underflows we might miss a rebase
                            # So we cast everything to xr
                            Z_xrn = Z_xr[zn]
                            if ref_is_xr[1]:
                                # Reference underflows, use available xr ref
                                ZZ_xr = Z_xrn + ref_zn_xr[1]
                            else:
                                ZZ_xr = Z_xrn + ref_zn_next
    
                            bool_dyn_rebase_xr = (
                                fsxn.extended_abs2(ZZ_xr)
                                <= fsxn.extended_abs2(Z_xrn)   
                            )
                            if bool_dyn_rebase_xr:
                                U[0] = 0
                                Z_xr[zn] = ZZ_xr
                                Z[zn] = fsxn.to_standard(ZZ_xr)
                        else:
                            # No risk of underflow - safe to rebase
                            U[0] = 0
                            Z[zn] = ZZ
    
                # End of while loop
                return n_iter

            return numba_impl

        self.iterate = iterate



#==============================================================================
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
        nucleus_size = fsFP.perturbation_mandelbrot_nucleus_size_estimate(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            order
        )
        print("raw nucleus_size", nucleus_size)
        nucleus_size = np.abs(nucleus_size)

        # r_J = r_M ** 0.75 for power 2 Mandelbrot
        sqrt = np.sqrt(nucleus_size)
        sqrtsqrt = np.sqrt(sqrt)
        julia_size = sqrtsqrt * sqrt

        return nucleus_size, julia_size


#==============================================================================
# GUI : "interactive options"

    @fsutils.interactive_options
    def coords(self, x, y, pix, dps):
        """ x, y : coordinates of the event """
        x_str = str(x)
        y_str = str(y)
        res_str = f"""
coords = {{
    "x": "{x_str}"
    "y": "{y_str}"
}}
"""
        return res_str
        

    @fsutils.interactive_options
    def ball_method_order(self, x, y, pix, dps,
                          maxiter: int=100000,
                          radius_pixels: int=25):
        """ x, y : coordinates of the event """
        c = x + 1j * y
        radius = pix * radius_pixels
        M_divergence = 1.e3
        order = self._ball_method(c, radius, maxiter, M_divergence)

        x_str = str(x)
        y_str = str(y)
        radius_str = str(radius)
        res_str = f"""
ball_order = {{
    "x": "{x_str}",
    "y": "{y_str}",
    "maxiter": {maxiter},
    "radius_pixels": {radius_pixels},
    "radius": "{radius_str}",
    "M_divergence": {M_divergence},
    "order": {order}
}}
"""
        return res_str


    @fsutils.interactive_options
    def newton_search(self, x, y, pix, dps,
                          maxiter: int=100000,
                          radius_pixels: int=3):
        """ x, y : coordinates of the event """
        c = x + 1j * y

        radius = pix * radius_pixels
        radius_str = str(radius)
        M_divergence = 1.e3
        order = self._ball_method(c, radius, maxiter, M_divergence)

        newton_cv = False
        max_attempt = 2
        attempt = 0
        while not(newton_cv) and attempt < max_attempt:
            if order is None:
                break
            attempt += 1
            dps = int(1.5 * dps)
            print("Newton, dps boost to: ", dps)
            with mpmath.workdps(dps):
                newton_cv, c_newton = self.find_nucleus(
                        c, order, pix, max_newton=None, eps_cv=None)
                if newton_cv:
                    xn_str = str(c_newton.real)
                    yn_str = str(c_newton.imag)

        if newton_cv:
            nucleus_size, julia_size = self._nucleus_size_estimate(
                c_newton, order
            )
        else:
            nucleus_size = None
            julia_size = None
            xn_str = ""
            yn_str = ""

        x_str = str(x)
        y_str = str(y)

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
}}
"""
        return res_str
