# -*- coding: utf-8 -*-
import numpy as np
import mpmath
import numba

import fractalshades as fs
import fractalshades.mpmath_utils.FP_loop as fsFP

#==============================================================================
#==============================================================================

class Mandelbrot(fs.Fractal):
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
            epsilon_stationnary: float,
    ):
        """
    Basic iterations for Mandelbrot standard set (power 2).

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
        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2"]
        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(np.complex128)

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

                    Z[3] = 2 * (Z[3] * Z[0] + Z[2]**2)
                    Z[2] = 2 * Z[2] * Z[0] + 1.
                    Z[1] = 2 * Z[1] * Z[0] 
                    Z[0] = Z[0]**2 + c
                    if n_iter == 1:
                        Z[1] = 1.

                    if Z[0].real**2 + Z[0].imag ** 2 > Mdiv_sq:
                        stop_reason[0] = 1
                        break

                    if Z[1].real**2 + Z[1].imag ** 2 < epscv_sq:
                        stop_reason[0] = 2
                        break

                # End of while loop
                return n_iter

            return numba_impl
        self.iterate = iterate


    @fs.utils.calc_options
    def newton_calc(self, *,
            calc_name: str,
            subset=None,
            known_orders=None,
            max_order,
            max_newton,
            eps_newton_cv,
    ):
        """
    Newton iterations for Mandelbrot standard set interior (power 2).

    Parameters
    ==========
    calc_name : str
         The string identifier for this calculation
    subset : 
        A boolean array-like, where False no calculation is performed
        If `None`, all points are calculated. Defaults to `None`.
    known_orders : None | int list 
        If not `None`, only the integer listed of their multiples will be 
        candidates for the cycle order, the other rwill be disregarded.
        If `None` all order between 1 and `max_order` will be considered.
    max_order : int
        The maximum value for ccyle 
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
    
        Exit codes are *max_order*, *order_confirmed*.

    References
    ==========
    .. [1] <https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html>

    .. [2] <https://mathr.co.uk/blog/2014-11-02_practical_interior_distance_rendering.html>
        """
        complex_codes = ["zr", "attractivity", "dzrdc", "dattrdc",
                         "_partial", "_zn"]
        int_codes = ["order"]
        stop_codes = ["max_order", "order_confirmed"]
        self.codes = (complex_codes, int_codes, stop_codes)
        self.init_data_types(np.complex128)

        def initialize():
            @numba.njit
            def numba_init_impl(Z, U, c):
                Z[4] = 1.e6 # bigger than any reasonnable np.abs(zn)
            return numba_init_impl
        self.initialize = initialize

        def iterate():
            @numba.njit
            def numba_impl(Z, U, c, stop_reason, n_iter):
                while True:
                    n_iter += 1

                    if n_iter > max_order:
                        stop_reason[0] = 0
                        break

                    # If n is not a 'partial' for this point it cannot be the 
                    # cycle order : early exit
                    Z[5] = Z[5]**2 + c
                    m = np.abs(Z[5])
                    if m < Z[4].real:
                        Z[4] = m # Cannot assign to the real part 
                    else:
                        continue

                    # Early exit if n it is not a multiple of one of the
                    # known_orders (provided by the user)
                    if known_orders is not None:
                        valid = False
                        for order in known_orders:
                            if  n_iter % order == 0:
                                valid = True
                        if not valid:
                            continue
    
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
                        continue
    
                    Z[0] = zr
                    Z[1] = dzrdz # attr (cycle attractivity)
                    Z[2] = dzrdc
                    Z[3] = d2zrdzdc # dattrdc
                    U[0] = n_iter
                    stop_reason[0] = 1
                    break

                # End of while loop
                return n_iter

            return numba_impl
        self.iterate = iterate

#==============================================================================
#==============================================================================

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
        SA_params=None,
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
    SA_params :
        The dictionnary of parameters for Series-Approximations :

        .. list-table:: 
           :widths: 20 80
           :header-rows: 1

           * - keys
             - values 
           * - cutdeg
             - int: polynomial degree used for approximation (default: 32)
           * - SA_err
             - float: relative error criteria (default: 1.e-6)

        if `None` SA is not activated. This option is kept for
        backward-compatibility, use of Bilinear Approximations is recommended.
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

        @numba.njit
        def _dfdz(z):
            return 2. * z
        self.dfdz = _dfdz

        #----------------------------------------------------------------------
        # Defines SA_loop via a function factory - jitted implementation
        # SA triggered ?
        SA_activated = (
            (SA_params is not None) 
            and (self.dx < fs.settings.newton_zoom_level)
        )
        def SA_loop():
            @numba.njit
            def impl(Pn, n_iter, Zn_xr, kcX):
                # Series Approximation loop
                # mostly: pertubation iteration applied to a polynomial
                return Pn * (Pn + 2. * Zn_xr) + kcX
            return impl
        self.SA_loop = SA_loop

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

        @numba.njit
        def p_iter_zn(Z, ref_zn, c):
            Z[zn] = Z[zn] * (Z[zn] + 2. * ref_zn) + c

        @numba.njit
        def p_iter_dzndz(Z):
            # Only used at low zoom - assumes dZndz == 0 
            Z[dzndz] = 2. * (Z[zn] * Z[dzndz])

        @numba.njit
        def p_iter_dzndc(Z, ref_zn, ref_dzndc):
            Z[dzndc] = 2. * ((ref_zn + Z[zn]) * Z[dzndc] + ref_dzndc * Z[zn])

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
