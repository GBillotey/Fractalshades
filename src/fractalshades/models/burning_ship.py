# -*- coding: utf-8 -*-
import typing
import enum

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
            # /!\ Record types need to be unified: hence the * 1.
            return 1. * x
        else:
            return -(2. * X + x)
    else:
        if (X + x) <= 0.:
            return -x
        else:
            return (2. * X + x)

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


BS_flavor_list = (
    "Burning ship",
    "Perpendicular burning ship",
    "Shark fin",
)

BS_flavor_enum =  enum.Enum(
    "BS_flavor_enum",
    BS_flavor_list,
    module=__name__
)

def get_flavor_int(flavor: str):
    # Using auto with IntEnum results in integers of increasing value,
    # starting with 1.
    return getattr(BS_flavor_enum, flavor).value

def BS_iterate(flavor_int):
    if flavor_int == 1:
        @numba.njit
        def numba_impl(xn, yn, a, b):
            # Burning ship
            return (
                xn ** 2 - yn ** 2 + a,
                2. * np.abs(xn * yn) - b
            )
    elif flavor_int == 2:
        @numba.njit
        def numba_impl(xn, yn, a, b):
            # Perpendicular burning ship
            return (
                xn ** 2 - yn ** 2 + a,
                2. * xn * np.abs(yn) - b
            )
    elif flavor_int == 3:
        @numba.njit
        def numba_impl(xn, yn, a, b):
            # Shark Fin
            return (
                xn ** 2 - yn * np.ans(yn) + a,
                2. * xn * yn - b
            )
    return numba_impl


class Burning_ship(fs.Fractal):
    def __init__(
        self,
        directory: str,
        flavor: typing.Literal[BS_flavor_enum]= "Burning ship"
    ):
        """
A Burning Ship Fractal (power 2). The basic equation a detailed below:

.. math::

    x_0 &= 0 \\\\
    y_0 &= 0 \\\\
    x_{n+1} &= x_n^2 - y_n^2 + a \\\\
    y_{n+1} &= 2 |x_n y_n| - b

where:

.. math::

    z_n &= x_n + i y_n \\\\
    c &= a + i b


Parameters
==========
directory : str
    Path for the working base directory
flavor : str
    The variant of Burning Ship detailed implementation, defaults to
    "Burning Ship". 

Notes
=====

.. note::

  Several variants (`flavor` parameter) are implemented with small
  differences in the iteration formula ; among them:
    
    - "Perpendicular burning ship" variant of the Burning Ship Fractal.
    
      .. math::
    
        x_{n+1} &= x_n^2 - y_n^2 + a \\\\
        y_{n+1} &= 2 x_n |y_n| - b
    
    - "Shark fin" variant
    
      .. math::
    
        x_{n+1} &= x_n^2 - y_n |y_n| + a \\\\
        y_{n+1} &= 2 x_n y_n - b
"""
        super().__init__(directory)
        self.flavor = flavor
        flavor_int = get_flavor_int(flavor)

        # default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = 2
        self.potential_a_d = 1.
        # Minimum M for valid potential estimation
        dic_cutoff = {
            "Burning ship": 1000.,
            "Perpendicular burning ship": 1000,
            "Shark fin": 1.e20,
        }
        self.potential_M_cutoff = dic_cutoff[flavor]

        # GUI 'badges'
        self.holomorphic = False
        self.implements_dzndc = "always"
        self.implements_fieldlines = True
        self.implements_newton = False
        self.implements_Milnor = False
        self.implements_interior_detection = "no"
        self.implements_deepzoom = False

        self.xnyn_iterate = BS_iterate(flavor_int)
        
        # Cache for numba compiled implementations
        self._numba_cache = {}
        self._numba_iterate_cache = {
            "calc_std_div": ((), None),
        } # args, numba_impl


    @fs.utils.calc_options
    def calc_std_div(self, *,
            calc_name: str,
            subset,
            max_iter: int,
            M_divergence: float,
            calc_orbit: bool = False,
            backshift: int = 0
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
    calc_orbit: bool
        If True, stores the value of an orbit point @ exit - orbit_shift
    backshift: int (> 0)
        The number of iteration backward for the stored orbit starting point

    Notes
    =====
    The following complex fields will be calculated: *xn* *yn* and its
    derivatives (*dxnda*, *dxndb*, *dynda*, *dyndb*).
    If calc_orbit is activated, *zn_orbit* will also be stored
    Exit codes are *max_iter*, *divergence*.
"""
        complex_codes = ["xn", "yn", "dxnda", "dxndb", "dynda", "dyndb"]
        xn = 0
        yn = 1
        dxnda = 2
        dxndb = 3
        dynda = 4
        dyndb = 5

        i_xnorbit = -1
        i_ynorbit = -1
        if calc_orbit:
            complex_codes += ["xn_orbit", "yn_orbit"]
            i_xnorbit = 6
            i_ynorbit = 7

        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]

        def set_state():
            def impl(instance):
                instance.codes = (complex_codes, int_codes, stop_codes)
                instance.complex_type = np.float64
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
        flavor_int = getattr(BS_flavor_enum, self.flavor).value

        iterate_once = self.from_numba_cache(
            "iterate_once", flavor_int, xn, yn, dxnda, dxndb, dynda, dyndb,
            max_iter, Mdiv_sq
        )
        xnyn_iterate = self.xnyn_iterate

        def iterate():
            
            new_args = (
                calc_orbit, i_xnorbit, i_ynorbit, backshift, xn, yn,
                iterate_once, xnyn_iterate
            )
            args, numba_impl = self._numba_iterate_cache["calc_std_div"]
            if new_args == args:
                return numba_impl

            numba_impl = fs.core.numba_iterate_BS(*new_args)
            self._numba_iterate_cache["calc_std_div"] = (new_args, numba_impl)
            return numba_impl


        return {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }



    def from_numba_cache(
        self, key, flavor_int, xn, yn, dxnda, dxndb, dynda, dyndb,
        max_iter, Mdiv_sq
    ):
        """ Returns the numba implementation if exists to avoid unnecessary
        recompilation"""
        cache = self._numba_cache
        full_key = (key, flavor_int, xn, yn, dxnda, dxndb, dynda, dyndb,
                    max_iter, Mdiv_sq)

        try:
            return cache[full_key]

        except KeyError:

            if key == "iterate_once":
                @numba.njit
                def numba_impl(c, Z, U, stop_reason, n_iter):
                    if n_iter >= max_iter:
                        stop_reason[0] = 0
                        return 1
        
                    a = c.real
                    b = c.imag
                    X = Z[xn]
                    Y = Z[yn]
                    dXdA = Z[dxnda]
                    dXdB = Z[dxndb]
                    dYdA = Z[dynda]
                    dYdB = Z[dyndb]
        
                    if flavor_int == 1:
                        # Standard BS implementation
                        Z[xn] = X ** 2 - Y ** 2 + a
                        Z[yn] = 2. * np.abs(X * Y) - b
                        # Jacobian
                        Z[dxnda] = 2 * (X * dXdA - Y * dYdA) + 1.
                        Z[dxndb] = 2 * (X * dXdB - Y * dYdB)
                        Z[dynda] = 2 * (np.abs(X) * sgn(Y) * dYdA + sgn(X) * dXdA * np.abs(Y))
                        Z[dyndb] = 2 * (np.abs(X) * sgn(Y) * dYdB + sgn(X) * dXdB * np.abs(Y)) - 1.
        
                    elif flavor_int == 2:
                        # Perpendicular BS implementation
                        Z[xn] = X ** 2 - Y ** 2 + a
                        Z[yn] = 2. * X * np.abs(Y) - b
                        # Jacobian
                        Z[dxnda] = 2 * (X * dXdA - Y * dYdA) + 1.
                        Z[dxndb] = 2 * (X * dXdB - Y * dYdB)
                        Z[dynda] = 2 * (X * sgn(Y) * dYdA + dXdA * np.abs(Y))
                        Z[dyndb] = 2 * (X * sgn(Y) * dYdB + dXdB * np.abs(Y)) - 1.
        
                    elif flavor_int == 3:
                        # Shark Fin
                        Z[xn] = X ** 2 - Y * np.abs(Y) + a
                        Z[yn] = 2. * X * Y - b
                        # Jacobian
                        Z[dxnda] = 2 * (X * dXdA - np.abs(Y) * dYdA) + 1.
                        Z[dxndb] = 2 * (X * dXdB - np.abs(Y) * dYdB)
                        Z[dynda] = 2 * dXdA * Y + 2 * X * dYdA
                        Z[dyndb] = 2 * dXdB * Y + 2 * X * dYdB - 1.
        
                    if Z[xn] ** 2 + Z[yn] ** 2 > Mdiv_sq:
                        stop_reason[0] = 1
                        return 1
        
                    return 0

            else:
                raise NotImplementedError(key)

            cache[full_key] = numba_impl
            return numba_impl



    @fs.utils.interactive_options
    def coords(self, x, y, pix, dps):
        return super().coords(x, y, pix, dps)

#==============================================================================
#==============================================================================
# Utility functions used in perturbation burning ship

def _dfxdx(flavor_int):
    if flavor_int == 1:
        @numba.njit
        def numba_impl(x, y):
            return 2. * x
    elif flavor_int == 2:
        @numba.njit
        def numba_impl(x, y):
            return 2. * x
    elif flavor_int == 3:
        @numba.njit
        def numba_impl(x, y):
            return 2. * x

    return numba_impl


def _dfxdy(flavor_int):
    if flavor_int == 1:
        @numba.njit
        def numba_impl(x, y):
            return -2. * y
    elif flavor_int == 2:
        @numba.njit
        def numba_impl(x, y):
            return -2. * y
    elif flavor_int == 3:
        @numba.njit
        def numba_impl(x, y):
            return -2. * np.abs(y)

    return numba_impl


def _dfydx(flavor_int):
    if flavor_int == 1:
        @numba.njit
        def numba_impl(x, y):
            return 2. * sgn(x) * np.abs(y)
    elif flavor_int == 2:
        @numba.njit
        def numba_impl(x, y):
            return 2. * np.abs(y)
    elif flavor_int == 3:
        @numba.njit
        def numba_impl(x, y):
            return 2. * y

    return numba_impl


def _dfydy(flavor_int):
    if flavor_int == 1:
        @numba.njit
        def numba_impl(x, y):
            return 2. * sgn(y) * np.abs(x)
    elif flavor_int == 2:
        @numba.njit
        def numba_impl(x, y):
            return 2. * sgn(y) * x
    elif flavor_int == 3:
        @numba.njit
        def numba_impl(x, y):
            return 2. * x

    return numba_impl


def _p_iter_zn(flavor_int, xn, yn):
    """ Perturbation iteration for different burning ship 'flavors'"""
    if flavor_int == 1:
        @numba.njit
        def numba_impl(Z, ref_xn, ref_yn, a, b):
            # Modifies in-place xn, yn
            ref_xyn = ref_xn * ref_yn
            new_xn = (
                Z[xn] * (Z[xn] + 2. * ref_xn) - Z[yn] * (Z[yn] + 2. * ref_yn)
                + a
            )
            new_yn = (
                2. * diffabs(
                    ref_xyn,
                    Z[xn] * Z[yn] + Z[xn] * ref_yn + Z[yn] * ref_xn
                ) - b
            )
            Z[xn] = new_xn
            Z[yn] = new_yn

    elif flavor_int == 2:
        @numba.njit
        def numba_impl(Z, ref_xn, ref_yn, a, b):
            new_xn = (
                Z[xn] * (Z[xn] + 2. * ref_xn) - Z[yn] * (Z[yn] + 2. * ref_yn)
                + a
            )
            new_yn = (
                2. * (
                    ref_xn * diffabs(ref_yn, Z[yn])
                    + Z[xn] * np.abs(ref_yn + Z[yn])
                ) - b
            )
            Z[xn] = new_xn
            Z[yn] = new_yn

    elif flavor_int == 3:
        @numba.njit
        def numba_impl(Z, ref_xn, ref_yn, a, b):
            new_xn = (
                Z[xn] * (Z[xn] + 2. * ref_xn)
                - ref_yn * diffabs(ref_yn, Z[yn])
                - Z[yn] * np.abs(ref_yn + Z[yn])
                + a
            )
            new_yn = 2. * (ref_xn * Z[yn] + ref_yn * Z[xn] + Z[xn] * Z[yn]) - b
            Z[xn] = new_xn
            Z[yn] = new_yn


    return numba_impl


def _p_iter_hessian(flavor_int, xn, yn, dxnda, dxndb, dynda, dyndb):
    """ Hessian matrix perturbation iteration for different Burning ship
        'flavors'
https://fractalforums.org/fractal-mathematics-and-new-theories/28/perturbation-theory/487/msg3226#msg3226
"""
    if flavor_int == 1:
        @numba.njit
        def numba_impl(
            Z, ref_xn, ref_yn, ref_dxnda, ref_dxndb, ref_dynda, ref_dyndb
        ):
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

    elif flavor_int == 2:
        @numba.njit
        def numba_impl(
            Z, ref_xn, ref_yn, ref_dxnda, ref_dxndb, ref_dynda, ref_dyndb
        ):
            _diffabs = diffabs(ref_yn, Z[yn])
            _ddiffabsdX = ddiffabsdX(ref_yn, Z[yn])
            _ddiffabsdx = ddiffabsdx(ref_yn, Z[yn])

            Y_y = ref_yn + Z[yn]
            _abs = np.abs(Y_y)
            _sgn = sgn(Y_y)

            new_dxnda = 2. * (
                ((ref_xn + Z[xn]) * Z[dxnda] + ref_dxnda * Z[xn])
                -((ref_yn + Z[yn]) * Z[dynda] + ref_dynda * Z[yn])
            )
            new_dxndb = 2. * (
                ((ref_xn + Z[xn]) * Z[dxndb] + ref_dxndb * Z[xn])
                -((ref_yn + Z[yn]) * Z[dyndb] + ref_dyndb * Z[yn])
            )
            new_dynda = 2. * (
                ref_dxnda * _diffabs + ref_xn * (
                    _ddiffabsdX * ref_dynda
                    + _ddiffabsdx * Z[dynda]
                )
                + Z[dxnda] * _abs + Z[xn] * _sgn * (ref_dynda + Z[dynda])
            )
            new_dyndb = 2. * (
                ref_dxndb * _diffabs + ref_xn * (
                        _ddiffabsdX * ref_dyndb
                        + _ddiffabsdx * Z[dyndb]
                )
                + Z[dxndb] * _abs + Z[xn] * _sgn * (ref_dyndb + Z[dyndb])
            )

            Z[dxnda] = new_dxnda
            Z[dxndb] = new_dxndb
            Z[dynda] = new_dynda
            Z[dyndb] = new_dyndb

    elif flavor_int == 3:
        @numba.njit
        def numba_impl(
            Z, ref_xn, ref_yn, ref_dxnda, ref_dxndb, ref_dynda, ref_dyndb
        ):
            _diffabs = diffabs(ref_yn, Z[yn])
            _ddiffabsdX = ddiffabsdX(ref_yn, Z[yn])
            _ddiffabsdx = ddiffabsdx(ref_yn, Z[yn])

            Y_y = ref_yn + Z[yn]
            _abs = np.abs(Y_y)
            _sgn = sgn(Y_y)

            new_dxnda = (
                Z[dxnda] * (Z[xn] + 2. * ref_xn)
                + Z[xn] * (Z[dxnda] + 2. * ref_dxnda)
                - ref_dynda * _diffabs
                - ref_yn * (ref_dynda * _ddiffabsdX + Z[dynda] * _ddiffabsdx)
                - Z[dynda] * _abs - Z[yn] * _sgn * (ref_dynda + Z[dynda])
            )
            new_dxndb = (
                Z[dxndb] * (Z[xn] + 2. * ref_xn)
                + Z[xn] * (Z[dxndb] + 2. * ref_dxndb)
                - ref_dyndb * _diffabs
                - ref_yn * (ref_dyndb * _ddiffabsdX + Z[dyndb] * _ddiffabsdx)
                - Z[dyndb] * _abs - Z[yn] * _sgn * (ref_dyndb + Z[dyndb])
            )
            new_dynda = 2. * (
                ref_dxnda * Z[yn] + ref_xn * Z[dynda]
                + ref_dynda * Z[xn] + ref_yn * Z[dxnda]
                + Z[dxnda] * Z[yn] + Z[xn] * Z[dynda]
            )
            new_dyndb = 2. * (
                ref_dxndb * Z[yn] + ref_xn * Z[dyndb]
                + ref_dyndb * Z[xn] + ref_yn * Z[dxndb]
                + Z[dxndb] * Z[yn] + Z[xn] * Z[dyndb]
            )

            Z[dxnda] = new_dxnda
            Z[dxndb] = new_dxndb
            Z[dynda] = new_dynda
            Z[dyndb] = new_dyndb
            
    return numba_impl

#==============================================================================
#==============================================================================
class Perturbation_burning_ship(fs.PerturbationFractal):

    def __init__(
        self,
        directory: str,
        flavor: typing.Literal[BS_flavor_enum]= "Burning ship"
):
        """
An arbitrary-precision implementation for the Burning ship set (power-2).
The Burning Ship fractal, first described by Michael Michelitsch
and Otto E. Rössler in 1992, is a variant of the mandelbrot fractal which 
involve the absolute value function, making the formula non-analytic:

.. math::

    x_0 &= 0 \\\\
    y_0 &= 0 \\\\
    x_{n+1} &= x_n^2 - y_n^2 + a \\\\
    y_{n+1} &= 2 |x_n y_n| - b

where:

.. math::

    z_n &= x_n + i y_n \\\\
    c &= a + i b

For a more comprehensive introduction, we recommend the paper 
`At the Helm of the Burning Ship`_.

This class implements arbitrary precision for the reference orbit, ball method
period search, newton search, perturbation method, chained billinear
approximations.


Parameters
----------
directory: str
    Path for the working base directory
flavor: str
    The variant of Burning Ship detailed implementation, defaults to
    "Burning Ship". 

Notes
-----
Implementation based on :

.. _At the Helm of the Burning Ship:

    **At the Helm of the Burning Ship** - Claude Heiland-Allen, 2019
    Proceedings of EVA London 2019 (EVA 2019) 
    <http://dx.doi.org/10.14236/ewic/EVA2019.74>

Several variants (`flavor` parameter) are implemented with small
differences in the iteration formula ; among them:
    
    - "Perpendicular burning ship" variant of the Burning Ship Fractal.
    
      .. math::
    
        x_{n+1} &= x_n^2 - y_n^2 + a \\\\
        y_{n+1} &= 2 x_n |y_n| - b
    
    - "Shark fin" variant
    
      .. math::
    
        x_{n+1} &= x_n^2 - y_n |y_n| + a \\\\
        y_{n+1} &= 2 x_n y_n - b
"""
        super().__init__(directory)
        self.flavor = flavor
        flavor_int = get_flavor_int(flavor)

        # Sets default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = 2
        self.potential_a_d = 1.
        self.potential_M_cutoff = 1000. # Minimum M for valid potential

        # Set parameters for the full precision orbit
        self.critical_pt = 0.
        self.FP_code = ["xn", "yn"]
        self.holomorphic = False


        # GUI 'badges'
        self.holomorphic = False
        self.implements_dzndc = "always"
        self.implements_fieldlines = False
        self.implements_newton = False
        self.implements_Milnor = False
        self.implements_interior_detection = "no"
        self.implements_deepzoom = True

        # Orbit 'on the fly' calculation
        self.xnyn_iterate = BS_iterate(flavor_int)
        
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

        flavor_int = get_flavor_int(self.flavor)

        x = c0.real
        y = c0.imag
        seed_prec = mpmath.mp.prec

        (i, partial_dict, xr_dict
         ) = fsFP.perturbation_nonholomorphic_FP_loop(
            NP_orbit.view(dtype=np.float64),
            xr_detect_activated,
            max_orbit_iter,
            M_divergence * 2, # to be sure ref exit after close points
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            kind=flavor_int
        )

        return i, partial_dict, xr_dict


    @fs.utils.calc_options
    def calc_std_div(self, *,
        calc_name: str,
        subset,
        max_iter: int,
        M_divergence: float,
        BLA_eps: float = 1e-6,
        calc_hessian: bool = True
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
    BLA_eps: float 
        Relative error criteria for BLA (default: 1.e-6)
        If `None`, BLA is not activated.
    calc_hessian: bool
        if True, the derivatives will be caculated allowing distance
        estimation and shading.
        """
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
        stop_codes = ["max_iter", "divergence"]
        reason_max_iter = 0
        reason_M_divergence = 1


        #----------------------------------------------------------------------
        # Define the functions used for BLA approximation
        # BLA triggered ?
        BLA_activated = (
            (BLA_eps is not None) 
            and (self.dx < fs.settings.newton_zoom_level)
        )
        # H = [dfxdx dfxdy]    [dfx] = H x [dx]
        #     [dfydx dfydy]    [dfy]       [dy]

        flavor_int = get_flavor_int(self.flavor)
        cache_args = (flavor_int, xn, yn, dxnda, dxndb, dynda, dyndb)

        dfxdx = self.from_numba_cache("dfxdx", *cache_args)
        dfxdy = self.from_numba_cache("dfxdy", *cache_args)
        dfydx = self.from_numba_cache("dfydx", *cache_args)
        dfydy = self.from_numba_cache("dfydy", *cache_args)


        def set_state():
            def impl(instance):
                instance.complex_type = np.float64
                instance.potential_M = M_divergence
                instance.codes = (complex_codes, int_codes, stop_codes)
                instance.dfxdx = dfxdx # (flavor_int)
                instance.dfxdy = dfxdy # (flavor_int)
                instance.dfydx = dfydx # (flavor_int)
                instance.dfydy = dfydy # (flavor_int)
            return impl


        #----------------------------------------------------------------------
        # Defines initialize - jitted implementation
        def initialize():
            new_args = (xn, yn, dxnda, dxndb, dynda, dyndb)
            args, numba_impl = self._numba_initialize_cache
            if new_args == args:
                return numba_impl

            numba_impl = fs.perturbation.numba_initialize_BS(*new_args)
            self._numba_initialize_cache = (new_args, numba_impl)
            return numba_impl

        #----------------------------------------------------------------------
        # Defines iterate - jitted implementation
        M_divergence_sq = M_divergence ** 2

        # Xr triggered for ultra-deep zoom
        xr_detect_activated = self.xr_detect_activated


        p_iter_zn = self.from_numba_cache("p_iter_zn", *cache_args)
        p_iter_hessian = self.from_numba_cache("p_iter_hessian", *cache_args)


        def iterate():
            new_args = (
                M_divergence_sq, max_iter, reason_max_iter,
                    reason_M_divergence,
                xr_detect_activated, BLA_activated,
                calc_hessian,
                xn, yn, dxnda, dxndb, dynda, dyndb,
                p_iter_zn, p_iter_hessian
            )
            args, numba_impl = self._numba_iterate_cache
            if new_args == args:
                return numba_impl

            numba_impl = fs.perturbation.numba_iterate_BS(*new_args)
            self._numba_iterate_cache = (new_args, numba_impl)
            return numba_impl


        return {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }


    def from_numba_cache(self, key, flavor_int,
                         xn, yn, dxnda, dxndb, dynda, dyndb):
        """ Returns the numba implementation if exists to avoid unnecessary
        recompilation"""
        cache = self._numba_cache
        full_key = (key, flavor_int, xn, yn, dxnda, dxndb, dynda, dyndb)
        
        try:
            return cache[full_key]

        except KeyError:
            if key == "dfxdx":
                numba_impl = _dfxdx(flavor_int)
            elif key == "dfxdy":
                numba_impl = _dfxdy(flavor_int)
            elif key == "dfydx":
                numba_impl = _dfydx(flavor_int)
            elif key == "dfydy":
                numba_impl = _dfydy(flavor_int)
            elif key == "p_iter_zn":
                numba_impl = _p_iter_zn(flavor_int, xn, yn)
            elif key == "p_iter_hessian":
                numba_impl = _p_iter_hessian(
                        flavor_int, xn, yn, dxnda, dxndb, dynda, dyndb
                )
            else:
                raise NotImplementedError(key)

            cache[full_key] = numba_impl
            return numba_impl


#------------------------------------------------------------------------------
# Newton search & other related methods

    def _ball_method(self, c, px, maxiter, M_divergence):
        """ Order 1 ball method: Cython wrapper"""
        x = c.real
        y = c.imag
        seed_prec = mpmath.mp.prec
        flavor_int = get_flavor_int(self.flavor)

        order = fsFP.perturbation_nonholomorphic_ball_method(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            str(px).encode('utf8'),
            maxiter,
            M_divergence,
            kind=flavor_int
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


    def find_any_nucleus(
        self, c, order, eps_pixel, max_newton=None, eps_cv=None
    ):
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
        flavor_int = get_flavor_int(self.flavor)
        
        is_ok, val = fsFP.perturbation_nonholomorphic_find_any_nucleus(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            order,
            max_newton,
            str(eps_cv).encode('utf8'),
            str(eps_pixel).encode('utf8'),
            kind=flavor_int
        )

        return is_ok, val


    def _nucleus_size_estimate(self, c0, order):
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
        flavor_int = get_flavor_int(self.flavor)

        (nucleus_size, skew
         ) = fsFP.perturbation_nonholomorphic_nucleus_size_estimate(
            str(x).encode('utf8'),
            str(y).encode('utf8'),
            seed_prec,
            order,
            kind=flavor_int
        )

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

    @fs.utils.interactive_options
    def quick_skew_estimate(self, x, y, pix, dps, maxiter: int=100000):
        flavor_int = get_flavor_int(self.flavor)

        seed_prec = mpmath.mp.prec
        skew = fsFP.perturbation_nonholomorphic_skew_estimate(
                str(x).encode('utf8'),
                str(y).encode('utf8'),
                seed_prec,
                maxiter,
                100000.,
                kind=flavor_int
        )

        res_str = f"""
quick_skew_estimate = {{
    "skew_00": "{skew[0, 0]}",
    "skew_01": "{skew[0, 1]}",
    "skew_10": "{skew[1, 0]}",
    "skew_11": "{skew[1, 1]}",
}}
"""
        return res_str
