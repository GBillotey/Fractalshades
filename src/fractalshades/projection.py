# -*- coding: utf-8 -*-
import inspect
#import enum

import numpy as np
import numba
import mpmath

import fractalshades.numpy_utils.xrange as fsx


# Log attributes
# https://docs.python.org/2/library/logging.html#logrecord-attributes

#PROJECTION_ENUM = enum.Enum(
#    "PROJECTION_ENUM",
#    ("cartesian", "spherical", "expmap"),
#    module=__name__
#)
#projection_type = typing.Literal[PROJECTION_ENUM]


class Projection:
    """
    A Projection instance is a container for:
        - numba-compiled mappings relative to the screen-coords
          [0.5, 0.5] x [-0.5..0.5]
          ie mapping x_pix, y_pix --> dx, dy = phi(x_pix, y_pic) ; 
          implemented for both
          complex (z) and 2 x real (x, y) coords
        - a setter for fractal property (e.g, for exmap xy_ratio is imposed to
          the value needed for y coords to span 2 * pi)

    (X, Y) = center + (dx, dy) = center + phi(dx, dy)
    """
    @property
    def init_kwargs(self):
        """ Return a dict of parameters used during __init__ call"""
        init_kwargs = {}
        for (
            p_name, param
        ) in inspect.signature(self.__init__).parameters.items():
            # By contract, __init__ params shall be stored as instance attrib.
            # ('Projection' interface)
            init_kwargs[p_name] = getattr(self, p_name)
        return init_kwargs

    def __eq__(self, other):
        """ 2 Projections are equal if they are instances from the SAME
        subclass and have been created with the same init_kwargs
        """
        return (
            other.__class__ == self.__class__
            and other.init_kwargs == self.init_kwargs
        )

    def adjust_to_zoom(self, fractal):
        """ Some projection might impose constraints on the xy_ratio"""
        # Default implementation: does nothing
        pass

    def make_impl(self):
        self.make_f_impl()
        self.make_df_impl()
        self.make_dfBS_impl()

    def get_impl(self):
        """ complex to complex mapping """
        raise NotImplementedError("Derived classes shall implement")

    def get_df_impl(self):
        """ complex to complex mapping """
        raise NotImplementedError("Derived classes shall implement")

    def get_dfBS_impl(self):
        """ complex to 4 floats mapping """
        raise NotImplementedError("Derived classes shall implement")

    @property
    def scale(self):
        """ Passed to Perturbation fractal as a corrective factor for the
        scaling used in dZndC computation (and consequently dzndc, distance
        estimation, ...)
        Default implementation returns 1
        """
        return 1.

# Development notes: Projections and derivatives / normals
# For perturbation fractals, the derivatives are relative to a "window" scale
# dx (in Cartesian mode). This scale become dx * projection.scale (constant
# real scalar).
# On top of that several "local" modifiers can be implemented 
# - total derivative (account for variable local scaling + rotation + skew)
# - derivative "without rotation" (useful if the final aim is to unwrap, for
#   instance to make a video)


#==============================================================================
class Cartesian(Projection):
    def __init__(self):
        """ Creates a Cartesian projection """
        self.make_impl()

    def make_f_impl(self):
        """ A cartesian projection just let pass-through the coordinates"""
        @numba.njit(numba.complex128(numba.complex128))
        def numba_impl(dz):
            return dz
        self.f = numba_impl

    def make_df_impl(self):
        self.df = None

    def make_dfBS_impl(self):
        self.dfBS = None

#==============================================================================
class Expmap(Projection):
    def __init__(self, hmin, hmax, is_xrange=False, rotates_df=True):
        """ 
        An exponential mapping will map :
            - x axis to r=exp(hmin)..exp(hmax) with nx points
            - y axis to theta=0..2pi with ny points
        xy_ratio of the fractal will be adjusted to ensure an almost conformal
        mapping

        Parameters:
        -----------
        hmin: str or float or mpmath.mpf
            scaling for the lower end of the y axis (absolute value is
            dx * hmin) 
        hmax: str or float or mpmath.mpf
            scaling for the higher end of the y axis (absolute value is
            dx * hmax)
        is_xrange: bool
            If true, hmin / hmax should be input as mpf or floats and are
            stored internally as Xrange
        rotates_df: bool
            If true, the derivative will be scaled but also rotated according
            to the scaling. A rule of thumb is this value shall be set to True
            for a standalone picture, and to False if used as input for a video
            making tool.
        """
        self.is_xrange = is_xrange
        self.rotates_df = rotates_df

        if is_xrange:
            # We store as xrange to allow use in numba function
            # str -> mpf -> Xrange conversion
            self.hmin = fsx.mpf_to_Xrange(mpmath.mpf(hmin))
            self.hmax = fsx.mpf_to_Xrange(mpmath.mpf(hmax))
        else:
            # str -> float64
            self.hmin = float(hmin)
            self.hmax = float(hmax)

        self.hmoy = (hmin + hmax) * 0.5
        self.dh = hmax - hmin

        self.make_impl()

    def adjust_to_zoom(self, fractal):
        """ We need to adjust the fractal xy_ratio in order to
        match hmax - hmin """
        # target: dh = 2. * np.pi * xy_ratio 
        fractal.xy_ratio = self.dh / (np.pi * 2.)

    def make_f_impl(self):
        hmoy = self.hmoy
        dh = self.dh

        @numba.njit(numba.complex128(numba.complex128),
                    nogil=True, fastmath=False)
        def numba_impl(z):
            # h linearly interpolated between hmin and hmax
            h = hmoy + dh * z.real
            # t linearly interpolated between -pi and pi
            t = dh * z.imag
            return  np.exp(complex(h, t))

        self.f = numba_impl

    def make_df_impl(self):
        dh = self.dh

        if self.rotates_df:
            @numba.njit(numba.complex128(numba.complex128),
                        nogil=True, fastmath=False)
            def numba_impl(z):
                return  np.exp(dh * z)

        else:
            @numba.njit(numba.complex128(numba.complex128),
                        nogil=True, fastmath=False)
            def numba_impl(z):
                return  np.exp(dh * z.real)

        self.df = numba_impl


    def make_dfBS_impl(self):
        dh = self.dh

        if self.rotates_df:
            @numba.njit(nogil=True, fastmath=False)
            def numba_impl(z):
                zhr = dh * z.real
                zhi = dh * z.imag
                r = np.exp(zhr)
                cr = np.cos(zhi) * r
                sr = np.sin(zhi) * r
                return  cr, -sr, sr, cr

        else:
            @numba.njit(nogil=True, fastmath=False)
            def numba_impl(z):
                zhr = dh * z.real
                r = np.exp(zhr)
                return  r, 0., 0., r

        self.dfBS = numba_impl

    @property
    def scale(self):
        return np.exp(self.hmoy)
