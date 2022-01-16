# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
import mpmath
import numba

import fractalshades as fs
import fractalshades.numpy_utils.xrange as fsx
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
        max_iter = calc_options["max_iter"]
        M_divergence = calc_options["M_divergence"]

        x = c0.real
        y = c0.imag
        seed_prec = mpmath.mp.prec
        (i, partial_dict, xr_dict
         ) = fsFP.perturbation_mandelbrot_FP_loop(
            NP_orbit,
            xr_detect_activated,
            max_iter,
            M_divergence * 2., # to be sure ref exit after close points
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
        SA_params={"cutdeg": 2, "eps": 1e-6},
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
             - int: polynomial degree used for first iteration  
           * - cutdeg_glitch
             - int: polynomial degree used for glitch correction 
           * - SA_err
             - float: maximal relative error before stopping SA 

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
        # dx for numba: use 1d array not 0d !
        dx = fsx.mpf_to_Xrange(self.dx, dtype=np.float64).ravel()

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
            def impl(Pn, n_iter, ref_path_xr, kcX): #, ref_path_xr, is_xr):
                """ Series Approximation loop
                Note that derivatives w.r.t dc will be deduced directly from the
                S.A polynomial.
                """
                return Pn * (Pn + 2. * ref_path_xr) + kcX
            return impl
        self.SA_loop = SA_loop

        # Defines initialize via a function factory
        def initialize():

            @numba.njit
            def numba_init_impl(Z, U, c_xr, Z_xr_trigger, Z_xr, P, kc, dx_xr, n_iter):
                """
                ... mostly the SA
                """

                if P is not None:
                    # Apply the  Series approximation step
                    U[0] = n_iter
                    c_scaled = c_xr / kc[0]
#                    print("P", P)
#                    print("c_scaled", c_scaled, P.__call__(c_scaled), fsxn.to_standard(P.__call__(c_scaled)))
#                    print("dzndc", dzndc)
                    Z[zn] = fsxn.to_standard(P.__call__(c_scaled))
                    if fs.perturbation.need_xr(Z[zn]):
                        Z_xr_trigger[zn] = True
                        Z_xr[zn] = P.__call__(c_scaled)
                    if dzndc != -1:
                        P_deriv = P.deriv()
#                        print("P_deriv", P_deriv)
                        deriv_scale =  dx_xr[0] / kc[0]
                        Z[dzndc] = fsxn.to_standard(
                            P_deriv.__call__(c_scaled) * deriv_scale
                        )
#                        print("Z[dzndc]", Z[dzndc], P_deriv.__call__(c_scaled), deriv_scale)
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
                c, Z, U, stop, n_iter,
                ref_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, drift_xr, dx_xr,
                Z_xr_trigger, Z_xr, c_xr, refpath_ptr, ref_is_xr, ref_zn_xr
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
                
                while True:
                    n_iter += 1

                    #==============================================================
                    # Load reference point value @ U[0]
                    # refpath_ptr = [prev_idx, curr_xr]
                    if xr_detect_activated:
                        ref_zn = fs.perturbation.ref_path_get(
                            ref_path, U[0],
                            has_xr, ref_index_xr, ref_xr, refpath_ptr,
                            ref_is_xr, ref_zn_xr, 0
                        )

                    else:
                        # ref_is_xr = False
                        ref_zn = ref_path[U[0]]

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
                    U[0] = U[0] + 1

                    if xr_detect_activated:
                        ref_zn_next = fs.perturbation.ref_path_get(
                            ref_path, U[0],
                            has_xr, ref_index_xr, ref_xr, refpath_ptr,
                            ref_is_xr, ref_zn_xr, 1
                        )
                    else:
                        ref_zn_next = ref_path[U[0]]


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
                        return

                    # Glitch correction -  "dynamic glitch"
                    bool_dyn_rebase = (
                        (abs(ZZ.real) <= abs(Z[zn].real))
                        and (abs(ZZ.imag) <= abs(Z[zn].imag))
                    )
                    if bool_dyn_rebase:# and not(xr_detect_activated): # and not(xr_detect_activated): # bool_dyn_rebase # debug and not(xr_detect_activated)
                        if xr_detect_activated and Z_xr_trigger[0]:
                            # Can we *really* rebase ??
                            # Note: if Z[zn] underflows we might miss a rebase...
                            # So we cast everything to xr
                            Z_xrn = Z_xr[zn]
                            if ref_is_xr[1]:
                                # Reference underflows, use available xr reference
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
    def _ball_method1(c, px, maxiter, M_divergence):#, M_divergence):
        #c = x + 1j * y
        z = mpmath.mpc(0.)
        r0 = px      # first radius
        r = r0 * 1.
        az = abs(z)
        dzdc = mpmath.mpc(0.)

        for i in range(1, maxiter + 1):
            if i%10000 == 0:
                print("Ball method", i, r)
            r = (az  + r)**2 - az**2 + r0
            z = z**2 + c
            dzdc =  2. * z * dzdc +  1.
            az = abs(z)
            if az > M_divergence:
                return None
            if (r > az):
                print("Ball method 1 found period:", i)
                return i #, z, dzdc

    @staticmethod
    def find_nucleus(c, order, max_newton=None, eps_cv=None):
        """
        https://en.wikibooks.org/wiki/Fractals/Mathematics/Newton_method#center
        https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html
        https://mathr.co.uk/blog/2018-11-17_newtons_method_for_periodic_points.html
        https://code.mathr.co.uk/mandelbrot-numerics/blob/HEAD:/c/lib/m_d_nucleus.c
        Run Newton search to find z0 so that f^n(z0) == 0
        """
        if order is None:
            return False, c
        if max_newton is None:
            max_newton = 80 # max(mpmath.mp.dps, 50)
#        if eps_cv is None:
#            eps_cv = mpmath.mpf(2.)**(-mpmath.mp.prec)
        c_loop = c

        hit = False
        for i_newton in range(max_newton): 
            print("Newton iteration", i_newton, "order", order)
            zr = mpmath.mp.zero
            dzrdc = mpmath.mp.zero
            h = mpmath.mp.one
            dh = mpmath.mp.zero
            for i in range(1, order + 1):# + 1):
                dzrdc = 2. * dzrdc * zr + 1. #  mpmath.mpf("2.")
                zr = zr * zr + c_loop
                # divide by unwanted periods
                if i < order and order % i == 0:
                    h *= zr
                    dh += dzrdc / zr
            f = zr / h
            df = (dzrdc * h - zr * dh) / (h * h)
            cc = c_loop - f / df
            newton_cv = mpmath.almosteq(cc, c_loop) #abs(cc - c_loop ) <= eps_cv
            c_loop = cc
            if newton_cv and (i_newton > 0):
                if hit:
                    print("Newton iteration cv @ ", i_newton)
                    break
                else:
                    hit = True
        return newton_cv, c_loop

    @staticmethod
    def find_any_nucleus(c, order, max_newton=None, eps_cv=None):
        """
        https://en.wikibooks.org/wiki/Fractals/Mathematics/Newton_method#center
        https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html
        https://mathr.co.uk/blog/2018-11-17_newtons_method_for_periodic_points.html
        https://code.mathr.co.uk/mandelbrot-numerics/blob/HEAD:/c/lib/m_d_nucleus.c
        Run Newton search to find z0 so that f^n(z0) == 0
        """
        if order is None:
            return False, c
        if max_newton is None:
            max_newton = 80 #max(mpmath.mp.dps, 50)
#        if eps_cv is None:
#            eps_cv = mpmath.mpf(2.)**(-mpmath.mp.prec)
        c_loop = c

        for i_newton in range(max_newton): 
            print("Newton iteration ANY", i_newton, "order", order)
            zr = mpmath.mp.zero
            dzrdc = mpmath.mp.zero
#            h = mpmath.mp.one
#            dh = mpmath.mp.zero
            for i in range(1, order + 1):# + 1):
                dzrdc = 2. * dzrdc * zr + 1. #  mpmath.mpf("2.")
                zr = zr * zr + c_loop
                # divide by unwanted periods
#                if i < order and order % i == 0:
#                    h *= zr
#                    dh += dzrdc / zr
            cc = c_loop - zr / dzrdc
            newton_cv = mpmath.almosteq(cc, c_loop) #abs(cc - c_loop ) <= eps_cv
            c_loop = cc
            if newton_cv and (i_newton > 0):
                print("Newton iteration cv @ ", i_newton)
                break
        return newton_cv, c_loop

    @staticmethod
    def find_any_attracting(c, order, max_newton=None, eps_cv=None):
        """
        https://en.wikibooks.org/wiki/Fractals/Mathematics/Newton_method#center
        https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html
        https://mathr.co.uk/blog/2018-11-17_newtons_method_for_periodic_points.html
        https://code.mathr.co.uk/mandelbrot-numerics/blob/HEAD:/c/lib/m_d_nucleus.c
        Run Newton search to find z0 so that f^n(z0) == 0
        """
        if max_newton is None:
            max_newton = max(mpmath.mp.dps, 50)
#        if eps_cv is None:
#            eps_cv = mpmath.mpf(2.)**(-mpmath.mp.prec)
#            print("eps_cv", eps_cv)
        c_loop = c

        for i_newton in range(max_newton):     
            if (i_newton % 5 == 0):
                print("attracting ANY", i_newton, " / ", max_newton)
            zr = c_loop #mpmath.mp.zero
            dzrdc = mpmath.mp.one# mpmath.mp.zero
            # d2zrdzdc = 0.   # == 1. hence : constant wrt c...
            for i in range(1, order + 1):
                # d2zrdzdc = 2 * (d2zrdzdc * zr + dzrdz * dzrdc)
                dzrdc = 2. * dzrdc * zr + 1. #mpmath.mp.one
                zr = zr * zr + c_loop
#                    zr = zr / 
#                if abs(zr) < mpmath.mpf("0.0001"):
#                    print("zr", zr, i)
            cc = c_loop - (zr - c_loop) / (dzrdc - 1.)#

            newton_cv = mpmath.almosteq(cc, c_loop)
            #abs(cc - c_loop ) <= eps_cv
            c_loop = cc
            if newton_cv:
                print("attracting ANY iteration cv @ ", i_newton)
                break
            
        return newton_cv, c_loop


    # TODO implement atom domain size estimate
    # https://mathr.co.uk/blog/2013-12-10_atom_domain_size_estimation.html


    @staticmethod
    def nucleus_size_estimate(c0, order):
        """
        Nucleus size estimate
        
        Parameters
        ----------
        c0 : position of the nucleus
        order : cycle order

https://mathr.co.uk/blog/2016-12-24_deriving_the_size_estimate.html

Structure in the parameter dependence of order and chaos for the quadratic map
Brian R Hunt and Edward Ott
J. Phys. A: Math. Gen. 30 (1997) 7067–7076
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
        return nucleus_size
    
    @staticmethod
    def julia_set_size_estimate(nucleus_size):
        """
        Julia set size estimate - knowing the nucleus_size
        
https://fractalforums.org/fractal-mathematics-and-new-theories/28/miniset-and-embedded-julia-size-estimates/912/msg4805#msg4805
    # r_J = r_M ** ((n+1)*(n-1)/n**2)
        """
        # r_J = r_M ** ((n+1)*(n-1)/n**2)   n = 2 -> r_M ** 0.75
        raise NotImplementedError()   # TODO
#        with mpfr.
#        r_m_mantissa = nucleus_size
#        r_m_exp = nucleus_size
#        
#        return nucleus_size

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
        order = self._ball_method1(c, radius, maxiter, M_divergence)

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
                          order: int=1):
        """ x, y : coordinates of the event """
        c = x + 1j * y
        
        newton_cv = False
        xn_str = ""
        yn_str = ""
        max_attempt = 2
        attempt = 0
        while not(newton_cv) and attempt < max_attempt:
            attempt += 1
            dps = int(1.5 * dps)
            print("Newton, dps boost to: ", dps)
            with mpmath.workdps(dps):
                newton_cv, c_loop = self.find_nucleus(
                        c, order, max_newton=None, eps_cv=None)
                if newton_cv:
                    xn_str = str(c_loop.real)
                    yn_str = str(c_loop.imag)

        x_str = str(x)
        y_str = str(y)

        res_str = f"""
newton_search = {{
    "x_start": "{x_str}",
    "y_start": "{y_str}",
    "order": {order}
    "x_nucleus": "{xn_str}",
    "y_nucleus": "{yn_str}",
}}
"""
        return res_str


