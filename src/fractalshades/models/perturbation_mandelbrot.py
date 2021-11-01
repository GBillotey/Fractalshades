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
        self.critical_pt = 0.

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

    # TODO implement nucleus size estimate
    # https://mathr.co.uk/blog/2016-12-24_deriving_the_size_estimate.html
    @staticmethod
    def nucleus_size_estimate(c, order):
        z = mpmath.mpc(0., 0.)
        l = mpmath.mpc(1., 0.)
        b = mpmath.mpc(1., 0.)
        for i in range(1, order):
            z = z * z + c
            l = 2. * z * l
            b = b + 1. / l
        return 1. / (b * l * l)

    def series_approx(self, SA_init, SA_loop, SA_params, iref, ref_div_iter,
                      calc_name):
        """
        Zk, zk
        C, c
        
        SA_params keys:
            init_P
            kc
            iref
            n_iter
            P
        """
        SA_stop =  SA_params.get("SA_stop", -1)
        if SA_stop is None:
            SA_stop = -1

        SA_err = SA_params.get("SA_err", 1.e-4)
        print("SA_err", SA_err)
        SA_err_sq = SA_err**2

        cutdeg = SA_params["cutdeg"]
        P = SA_init(cutdeg)
        SA_params["iref"] = iref
        
        # Ensure corner points strictly included in convergence disk :
        kc = self.ref_point_scaling(iref, calc_name)

        # Convert kc into a Xrange array (in all cases the SA iterations use
        # extended range)
        kc = fsx.mpf_to_Xrange(kc, dtype=self.base_float_type)
        SA_params["kc"] = kc
        print('SA_params["kc"]', kc)
        kcX = np.insert(kc, 0, 0.)
        kcX = fsx.Xrange_SA(kcX, cutdeg)

        _, ref_path = self.reload_ref_point(iref, calc_name)
        n_iter = 0
        P0 = P[0]
        
        # Call jitted function for the loop
        P0, n_iter, err = SA_run(SA_loop, P0, n_iter, ref_path, kcX, SA_err_sq,
                                 SA_stop, ref_div_iter)

        # Storing results
        print("SA stop", n_iter, err)
        deriv_scale = fsx.mpc_to_Xrange(self.dx) / SA_params["kc"]
        if not(self.Xrange_complex_type):
            deriv_scale = deriv_scale.to_standard()
        P1 = fsx.Xrange_polynomial([complex(1.)], cutdeg=cutdeg)
        # We derive the polynomial wrt to c. Note the 1st coefficent
        # should be set to 0 if we compute these with FP...
        # Here, to the contrary we use it to skip the FP iter for all 
        # derivatives.
        P_deriv = P0.deriv() * deriv_scale
        # P_deriv.coeffs[0] = 0.
        # P_deriv2 = P_deriv.deriv() * deriv_scale
        # P_deriv2.coeffs[0] = 0.

        P = [P0, P1, P_deriv]
        SA_params["n_iter"] = n_iter
        SA_params["P"] = P

        return SA_params



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



    @fsutils.calc_options
    def calc_std_div(self, *,
        calc_name: str,
        datatype,
        subset,
        max_iter: int,
        M_divergence: float,
        epsilon_stationnary: float,
        SA_params=None,
        glitch_eps=None,
        glitch_max_attempt: int=0,
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
    glitch_eps : float
        A small float to qualify the glitched pixel. Typical value 1.e-4
    interior_detect : bool
        EXPERIMENTAL for perturbation.
        If True activates interior point detection
    glitch_max_attempt: int
        The maximal number of attempt for glitched pixel correction
        
    References
    ==========
    .. [1] <https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html>

    .. [2] <http://www.fractalforums.com/announcements-and-news>
        
        """
        self.init_data_types(datatype)
        
        # used for potential post-processing
        self.potential_M = M_divergence

        if glitch_eps is None:
            glitch_eps = (1.e-6 if self.base_complex_type == np.float64
                          else 1.e-3)
        self.glitch_eps = glitch_eps

        complex_codes = ["zn", "dzndz"]
        if calc_dzndc:
            complex_codes += ["dzndc"]

        int_codes = ["iref"]  # reference FP
        stop_codes = ["max_iter", "divergence", "stationnary",
                      "dyn_glitch", "divref_glitch"]
        self.codes = (complex_codes, int_codes, stop_codes)
        

        if SA_params is None:
            FP_fields = [0] #ß, 1, 2]
        else:
            # If SA activated, derivatives will be deduced - no need to compute
            # with FP.
            FP_fields = [0]
        self.FP_codes = [complex_codes[f] for f in FP_fields]

        # Defines FP_init via a function factory
        def FP_init():
            SA_params = self.SA_params
            def func():
                if SA_params is None:
                    return [mpmath.mp.zero] #, mpmath.mp.zero, mpmath.mp.zero]
                else:
                    return [mpmath.mp.zero]
            return func
        self.FP_init = FP_init

        # Defines FP_loop via a function factory
        def FP_loop():
            M_divergence = self.M_divergence
            def func(FP_array, c0, n_iter):
                """ Full precision loop
                derivatives corrected by lenght kc
                """
                FP_array[0] = FP_array[0]**2 + c0
                # If FP iteration is divergent, raise the semaphore n_iter
                # We use the 'infinite' norm not the disc for obvious calc saving
                if ((abs((FP_array[0]).real) > M_divergence)
                    or (abs((FP_array[0]).imag) > M_divergence)):
                    print("Reference point FP iterations escaping at", n_iter)
                    return n_iter
            return func
        self.FP_loop = FP_loop

        # Defines SA_init via a function factory
        def SA_init():
            # cutdeg = self.SA_params["cutdeg"]
            def func(cutdeg):
                # Typing as complex for numba
                return [fsx.Xrange_SA([0j], cutdeg=cutdeg)]
            return func
        self.SA_init = SA_init

        # Defines SA_loop via a function factory - jitted implementation
        def SA_loop():
            @numba.njit
            def impl(P0, n_iter, ref_path, kcX):
                """ Series Approximation loop
                Note that derivatives w.r.t dc will be deduced directly from the
                S.A polynomial.
                """
                xr_2 = fsxn.Xrange_scalar(1., numba.int32(1))
                P0 = P0 * (P0 + xr_2 * ref_path[0]) + kcX
                return P0
            return impl
        self.SA_loop = SA_loop

        # Defines initialize via a function factory
        def initialize():
            def func(Z, U, c, chunk_slice, iref):
                if calc_dzndc:
                    Z[2, :] = 0.
                Z[1, :] = 1.
                Z[0, :] = 0.
                U[0, :] = iref
            return func
        self.initialize = initialize

        # Defines iterate via a function factory - jitted implementation
        def iterate():
            M_divergence_sq = self.M_divergence ** 2
            epsilon_stationnary_sq = self.epsilon_stationnary ** 2
            glitch_eps_sq = self.glitch_eps ** 2
            Xrange_complex_type = self.Xrange_complex_type

            zn = 0
            dzndz = 1
            dzndc = 2
            reason_max_iter = 0
            reason_M_divergence = 1
            reason_stationnary = 2
            reason_dyn_glitch = 3
            reason_div_glitch = 4
            glitch_off_last_iref = fssettings.glitch_off_last_iref
            no_SA = (SA_params is None)

            interior_detect_activated = (
                interior_detect 
                and (self.dx > fssettings.newton_zoom_level)
            )

            @numba.njit
            def numba_impl(Z, U, c, stop_reason, n_iter, SA_iter,
                        ref_div_iter, ref_path, ref_path_next,
                        last_iref):
                """
                dz(n+1)dc   <- 2. * dzndc * zn + 1.
                dz(n+1)dz   <- 2. * dzndz * zn
                z(n+1)      <- zn**2 + c 
                
                Termination codes
                0 -> max_iter reached ('interior')
                1 -> M_divergence reached by np.abs(zn)
                2 -> dzn stationnary ('interior early detection')
                3 -> glitched (Dynamic glitches...)
                4 -> glitched (Ref point diverging  ...)
                """
                # This is an approximation as we do not store the ref pt
                # derivatives - though, we have it through SA iterations
                if calc_dzndc:
                    Z[dzndc] = 2. * (
                            ref_path[zn] * Z[dzndc] + Z[zn] * Z[dzndc]
                    )
                    if no_SA and (n_iter == 1):
                        # Non-null term needed to 'kick-off'
                        Z[dzndc] = 1.

                # Interior detection - Currently only at low zoom level
                if interior_detect_activated and (n_iter > 1):
                    Z[dzndz] = 2. * (Z[zn] * Z[dzndz])

                Z[zn] = Z[zn] * (Z[zn] + 2. * ref_path[zn]) + c

                if n_iter >= max_iter:
                    stop_reason[0] = reason_max_iter
                    return

                # Flagged as 'diverging ref pt glitch'
                if n_iter >= ref_div_iter:
                    stop_reason[0] = reason_div_glitch
                    return

                # Interior points detection
                if interior_detect_activated: # and (n_iter > SA_iter):
                    bool_stationnary = (
                            (Z[dzndz].real)**2  # + ref_path_next[1].real
                            + (Z[dzndz].imag)**2 # + ref_path_next[1].imag
                            < epsilon_stationnary_sq)
                    if bool_stationnary:
                        stop_reason[0] = reason_stationnary

                ZZ = Z[zn] + ref_path_next[zn]
                if Xrange_complex_type:
                    full_sq_norm = fsxn.extended_abs2(ZZ)
                else:
                    full_sq_norm = ZZ.real**2 + ZZ.imag**2

                # Flagged as 'diverging'
                bool_infty = (full_sq_norm > M_divergence_sq)
                if bool_infty:
                    stop_reason[0] = reason_M_divergence

                # Glitch detection
                if glitch_off_last_iref and last_iref:
                    return

                if Xrange_complex_type:
                    ref_sq_norm = fsxn.extended_abs2(ref_path_next[zn])
                else:
                    ref_sq_norm = (ref_path_next[zn].real**2
                                   + ref_path_next[zn].imag**2)

                # Flagged as "dynamic glitch"
                bool_glitched = (full_sq_norm  < (ref_sq_norm * glitch_eps_sq))
                if bool_glitched:
                    stop_reason[0] = reason_dyn_glitch
                    # We generate a glitch_sort_key based on 'close to secondary
                    # nucleus' criteria
                    # We use dzndz field to save it, as specified by 
                    # glitch_sort_key parameter of self.cycles call.
                    Z[1] = full_sq_norm

            return numba_impl


        self.iterate = iterate
        # Parameters for glitch detection and solving
        self.glitch_stop_index = 3 #reason_dyn_glitch
        self.glitch_sort_key = "dzndz"



#    @fsutils.calc_options
#    def calc_fast(self, *,
#        calc_name: str,
#        datatype,
#        subset,
#        max_iter: int,
#        M_divergence: float,
#        epsilon_stationnary: float,
#        SA_params=None,
#        glitch_eps=None,
#        glitch_max_attempt: int=0):
#        
#        """
#        Faster options for perturbation iterations (arbitrary precision) of
#        Mandelbrot standard set (power 2).
#        
#        Refer to `calc_std_div` for the explanation of input parameters used.
#        Compared to `calc_std_div` this method should preferably be used for
#        exploration as it makes a few approximations and does not computes
#        derivatives
#        (Plotting options likes scene-lighting will not be available ;
#        field lines might show noticeable banding)
#        """
#
#        self.init_data_types(datatype)
#        
#        # used for potential post-processing
#        self.potential_M = M_divergence
#
#        if glitch_eps is None:
#            glitch_eps = (1.e-6 if self.base_complex_type == np.float64
#                          else 1.e-3)
#        self.glitch_eps = glitch_eps
#
#
#        complex_codes = ["zn", "glitch_sort"]
#        int_codes = ["iref"]  # reference FP
#        stop_codes = ["max_iter", "divergence",
#                      "dyn_glitch", "divref_glitch"]
#        self.codes = (complex_codes, int_codes, stop_codes)
#
#
#        if SA_params is None:
#            FP_fields = [0] #ß, 1, 2]
#        else:
#            # If SA activated, derivatives will be deduced - no need to compute
#            # with FP.
#            FP_fields = [0]
#        self.FP_codes = [complex_codes[f] for f in FP_fields]
#
#        # Defines FP_init via a function factory
#        def FP_init():
#            SA_params = self.SA_params
#            def func():
#                if SA_params is None:
#                    return [mpmath.mp.zero] #, mpmath.mp.zero, mpmath.mp.zero]
#                else:
#                    return [mpmath.mp.zero]
#            return func
#        self.FP_init = FP_init
#
#        # Defines FP_loop via a function factory
#        def FP_loop():
#            M_divergence = self.M_divergence
#            def func(FP_array, c0, n_iter):
#                """ Full precision loop
#                derivatives corrected by lenght kc
#                """
#                FP_array[0] = FP_array[0]**2 + c0
#                # If FP iteration is divergent, raise the semaphore n_iter
#                # We use the 'infinite' norm not the disc for obvious calc saving
#                if ((abs((FP_array[0]).real) > M_divergence)
#                    or (abs((FP_array[0]).imag) > M_divergence)):
#                    print("Reference point FP iterations escaping at", n_iter)
#                    return n_iter
#            return func
#        self.FP_loop = FP_loop
#
#        # Defines SA_init via a function factory
#        def SA_init():
#            # cutdeg = self.SA_params["cutdeg"]
#            def func(cutdeg):
#                # Typing as complex for numba
#                return [fsx.Xrange_SA([0j], cutdeg=cutdeg)]
#            return func
#        self.SA_init = SA_init
#
#        # Defines SA_loop via a function factory - jitted implementation
#        def SA_loop():
#            @numba.njit
#            def impl(P0, n_iter, ref_path, kcX):
#                """ Series Approximation loop
#                Note that derivatives w.r.t dc will be deduced directly from the
#                S.A polynomial.
#                """
#                xr_2 = fsxn.Xrange_scalar(1., numba.int32(1))
#                P0 = P0 * (P0 + xr_2 * ref_path[0]) + kcX
#                return P0
#            return impl
#        self.SA_loop = SA_loop
#
#        # Defines initialize via a function factory
#        def initialize():
#            def func(Z, U, c, chunk_slice, iref):
#                Z[0, :] = 0.
#                Z[1, :] = 0.
#                U[0, :] = iref
#            return func
#        self.initialize = initialize
#
#        # Defines iterate via a function factory - jitted implementation
#        def iterate():
#            M_divergence = self.M_divergence
#            glitch_eps = self.glitch_eps
##            Xrange_complex_type = self.Xrange_complex_type
#
#            zn = 0
#            reason_max_iter = 0
#            reason_M_divergence = 1
#            # reason_stationnary = 2
#            reason_dyn_glitch = 2
#            reason_div_glitch = 3
#            glitch_off_last_iref = fssettings.glitch_off_last_iref
#
#
#            @numba.njit
#            def numba_impl(Z, U, c, stop_reason, n_iter, SA_iter,
#                        ref_div_iter, ref_path, ref_path_next,
#                        last_iref):
#                """
#                dz(n+1)dc   <- 2. * dzndc * zn + 1.
#                dz(n+1)dz   <- 2. * dzndz * zn
#                z(n+1)      <- zn**2 + c 
#                
#                Termination codes
#                0 -> max_iter reached ('interior')
#                1 -> M_divergence reached by np.abs(zn)
#                2 -> dzn stationnary ('interior early detection')
#                3 -> glitched (Dynamic glitches...)
#                4 -> glitched (Ref point diverging  ...)
#                """
#
#                Z[zn] = Z[zn] * (Z[zn] + 2. * ref_path[zn]) + c
#
#                if n_iter >= max_iter:
#                    stop_reason[0] = reason_max_iter
#                    return
#
#                # Flagged as 'diverging ref pt glitch'
#                if n_iter >= ref_div_iter:
#                    stop_reason[0] = reason_div_glitch
#                    return
#
##                ZZ = Z[zn] + ref_path_next[zn]
##                if Xrange_complex_type:
##                    full_sq_norm = ZZ.abs2()
##                else:
##                    full_sq_norm = ZZ.real**2 + ZZ.imag**2
##                sq_norm = Z[zn].real**2 + Z[zn].imag**2
##                z_real = Z[zn].real
##                z_imag = Z[zn].imag
#
#                ref_real = ref_path_next[zn].real
#                ref_imag = ref_path_next[zn].imag
#                ref_norm = (np.abs(ref_real) + np.abs(ref_imag)) * glitch_eps
#
#                full_real = ref_real + Z[zn].real
#                full_imag = ref_imag + Z[zn].imag
#                full_norm = np.abs(full_real) + np.abs(full_imag)
#
#                # Flagged as 'diverging'
#                if (full_norm > M_divergence):
#                    stop_reason[0] = reason_M_divergence
#
#                # Glitch detection
#                if glitch_off_last_iref and last_iref:
#                    return
#
#                # Flagged as "dynamic glitch"
#                if (full_norm < ref_norm):
#                    stop_reason[0] = reason_dyn_glitch
#                    # We generate a glitch_sort_key based on 'close to secondary
#                    # nucleus' criteria
#                    # We use dzndz field to save it, as specified by 
#                    # glitch_sort_key parameter of self.cycles call.
#                    Z[1] = np.abs(full_real) + np.abs(full_imag)
#
#            return numba_impl
#
#
#        self.iterate = iterate
#        # Parameters for glitch detection and solving
#        self.glitch_stop_index = 2 #reason_dyn_glitch
#        self.glitch_sort_key = "glitch_sort"




@numba.njit
def SA_run(SA_loop, P0, n_iter, ref_path, kcX, SA_err_sq, SA_stop,
           ref_div_iter):
    """
    SA_stop : user-provided max SA iter. If -1, will default to ref_path length
    ref_div_iter : point where Reference point DV
    """
    if SA_stop == -1:
        SA_stop = ref_div_iter
    else:
        SA_stop = min(ref_div_iter, SA_stop)
        
    print_freq = max(5, int(SA_stop / 100000.))
    print_freq *= 1000
    print("Output every", print_freq)

    SA_valid = True
    while SA_valid:
        n_iter +=1
        # keep a copy in case this iter is invalidated
        P_old0 = P0.coeffs.copy()
        P0 = SA_loop(P0, n_iter, ref_path[n_iter - 1, :], kcX)
        
        coeffs_sum = fsxn.Xrange_scalar(0., numba.int32(0))
        for i in range(len(P0.coeffs)):
            coeffs_sum = coeffs_sum + fsxn.extended_abs2(P0.coeffs[i])
        err_abs2 = P0.err[0] * P0.err[0]

        SA_valid = (
            (err_abs2  <= SA_err_sq * coeffs_sum)
            and (coeffs_sum <= 1.e6) # 1e6 to allow 'low zoom'
            and (n_iter < SA_stop)
        )
        if not(SA_valid):
            P0_ret = fsx.Xrange_polynomial(P_old0, P0.cutdeg)
            n_iter -= 1

        if n_iter % print_freq == 0 and SA_valid:
            ssum = np.sqrt(coeffs_sum)
            print("SA running", n_iter, "err: ", P0.err,
                  "<< [(", ssum.mantissa, ",", ssum.exp, ")]")
    return P0_ret, n_iter, P0.err
