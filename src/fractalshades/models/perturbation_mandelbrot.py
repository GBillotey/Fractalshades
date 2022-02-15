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
#        SA_params={"cutdeg": 32, "eps": 1e-6},
        BLA_params={"eps": 1e-6},
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
        nz = len(complex_codes)

        # Integer int32 fields codes "U" 
        int_codes = ["ref_cycle_iter"] # Position in ref orbit

        # Stop codes
        stop_codes = ["max_iter", "divergence", "stationnary"]
        reason_max_iter = 0
        reason_M_divergence = 1
        reason_stationnary = 2

        self.codes = (complex_codes, int_codes, stop_codes)
        print("###self.codes", self.codes)

        
        def dZndz_iter():
            """ Elementary dzndz iteration - used for reference cycle"""
            @numba.njit
            def impl(zn, dzndz):
                return 2. * zn * dzndz
            return impl
        self.dZndz_iter = dZndz_iter

        def dZndc_iter():
            """ Elementary dzndz iteration - used for reference cycle"""
            @numba.njit
            def impl(zn, dzndc):
                return 2. * zn * dzndc + 1.
            return impl
        self.dZndc_iter = dZndc_iter

        def dfdz():
            """ The tangent approximation - used for BLA approx"""
            @numba.njit
            def impl(z):
                return 2. * z
            return impl
        self.dfdz = dfdz

        # Initialise output arrays
        def initialize():
            @numba.njit
            def numba_init_impl(Z, U, c_xr):
                """ dzndz shall be filled with 1 to avoid cancellation """
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
            # Xr triggered for ultra-deep zoom
            xr_detect_activated = self.xr_detect_activated
            
            @numba.njit
            def numba_impl(
                c, c_xr, Z, U, stop,
                Zn_path, ref_order, ref_index_xr, ref_xr,
                A_bla, B_bla, r_bla, dZndz_path, dZndc_path
            ):
                """
Parameters
----------
c, c_xr: scalars
     values of c (resp. as complex 128 and as Xrange)
Z, U, stop: arrays
     arrays, raw outputs for this pixel
n_iter: int
     current iteration
Zn_path
     'full precision' orbit, stored at standard precision
ref_order: int
     The full precision orbit order (if no known order, this is simply a very
     large number 2**62)
ref_index_xr :
    The orbit indices for which eXtended range scalars shall be used to avoid
    cancellation
ref_xr :
    The above-mentionned Xr values in the orbit
dZndz_path, dZndc_path :
    the reference orbit derivatives, if available

Return
------
niter: int
    the exit iteration
"""
                Z_xr_trigger = np.zeros((1,), dtype=np.bool_)
                Z_xr = fs.perturbation.Xr_template.repeat(nz)
                ref_orbit_len = Zn_path.shape[0] # /!\ this only the ref pt,
                                                   # not max_iter
                refpath_ptr = np.zeros((2,), dtype=np.int32)

                has_xr = (len(ref_index_xr) > 0)
                ref_is_xr = np.zeros((1,), dtype=numba.bool_)
                ref_zn_xr = fs.perturbation.Xr_template.repeat(1)

                # Number of stages in BLA tree
                stages_bla = fs.perturbation.stages_bla(ref_orbit_len)

                n_iter = 0
                # Wrapping if we reach the cycle order
                if U[0] >= ref_order:
                    U[0] = U[0] % ref_order

                while True:
                    #==========================================================
                    # Try a BLA_step
                    step = 0
                    # if (U[0] != 0) and (U[0] %8 == 0):
                    (Ai, Bi, step) = fs.perturbation.ref_BLA_get(
                        A_bla, B_bla, r_bla, stages_bla, Z[zn], U[0]
                    )
                        # print("--> BLA step", n_iter, U[0], step, Ai, Bi)

                    if step != 0:
                        n_iter += step
                        U[0] = (U[0] + step) % ref_order
                        if calc_dzndc:
                            Z[dzndz] = Ai * Z[dzndz]
                        if calc_dzndc:
                            Z[dzndc] = Ai * Z[dzndc] + Bi
                        
                        Z[zn] = Ai * Z[zn] + Bi * c
                        continue

                    #==========================================================
                    # BLA failed, launching a full perturbation iteration
                    n_iter += 1
                    # Load reference point value @ U[0] taking into account
                    # potential xr case
                    # refpath_ptr = [prev_idx, curr_xr]
                    if xr_detect_activated:
                        ref_zn = fs.perturbation.ref_path_get(
                            Zn_path, U[0],
                            has_xr, ref_index_xr, ref_xr, refpath_ptr,
                            ref_is_xr, ref_zn_xr, 0
                        )
                    else:
                        ref_zn = Zn_path[U[0]]

                    #==========================================================
                    # Pertubation iter block
                    #----------------------------------------------------------
                    # dzndc subblock
                    if calc_dzndc:
                    # This is an approximation as we do not store the ref pt
                    # derivatives 
                    # Full term would be :
                    # Z[dzndc] =  2 * (
                    #    (ref_path_zn + Z[zn]) * Z[dzndc]
                    #    + Z[zn] * ref_path_dzndc
                        ref_dzndc = dZndc_path[U[0]]
                        Z[dzndc] = (
                            2. * (ref_zn + Z[zn]) * Z[dzndc]
                            + ref_dzndc * Z[zn]
                        )
#                        if (n_iter == 1): 
#                            # Non-null term needed to 'kick-off'
#                            # we do not use anymore not(SA_activated)
#                            Z[dzndc] = 1.

                    #----------------------------------------------------------
                    # Interior detection - Used only at low zoom level
                    if interior_detect_activated and (n_iter > 1):
                        Z[dzndz] = 2. * (Z[zn] * Z[dzndz])

                    #----------------------------------------------------------
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

                    #==========================================================
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

                    #==========================================================
                    # ZZ = "Total" z + dz
                    U[0] = (U[0] + 1) % ref_order

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
                    full_sq_norm = ZZ.real ** 2 + ZZ.imag ** 2
    
                    # Flagged as 'diverging'
                    bool_infty = (full_sq_norm > M_divergence_sq)
                    if bool_infty:
                        stop[0] = reason_M_divergence
                        break

                    # Glitch correction - reference point diverging
                    # Note: should never be triggered if ref pt is a cycle
                    if (U[0] >= ref_orbit_len):
                        assert ref_order >= 2**62 # debug 
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
                            Z[dzndc] += dZndc_path[U[0]]
    
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
