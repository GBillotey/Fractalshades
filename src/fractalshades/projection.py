# -*- coding: utf-8 -*-
#import typing
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
    
    def __eq__(self, other):
        """ 2 Projections are equal if they are instances from the SAME
        subclass and share the same attributes (as exposed by
        internal __dict__)
        """
        return (
            other.__class__ == self.__class__
            and other.__dict__ == self.__dict__
        )


class Cartesian(Projection):
    def __init__(self):
        """ A Cartesian projection just let pass-through the screen
        coordinates """
        pass

    def adjust_to_zoom(self, fractal):
        pass

    def get_impl(self):

        @numba.njit(numba.complex128(numba.complex128))
        def numba_impl(dz):
            return dz

        return numba_impl

    @property
    def name(self):
        return "cartesian"


class Expmap(Projection):
    def __init__(self, hmin, hmax, is_xrange=False):
        """ 
        Parameters:
        -----------
        hmin: str or float
            scaling for the lower end of the y axis (absolute value is
            dx * hmin) 
        hmax: str or float
            scaling for the higher end of the y axis (absolute value is
            dx * hmax)
        
        An exponential mapping will map :
            - x axis to 0..2pi with nx points
            - y axis to hmin..hmax with ny points
        """
        if is_xrange:
            # We store as xrange to allow use in numba function
            # str -> mpf -> Xrange conversion
            self.hmin = fsx.mpf_to_Xrange(mpmath.mpf(hmin))
            self.hmax = fsx.mpf_to_Xrange(mpmath.mpf(hmax))
        else:
            # str -> float64
            self.hmin = float(hmin)
            self.hmax = float(hmax)

    def adjust_to_zoom(self, fractal):
        """ We need to adjust the fractal xy_ratio in order to
        match hmax - hmin """
        # target: (h_max - hmin) = 2. * np.pi * xy_ratio 
        fractal.xy_ratio = (self.hmax - self.hmin) / (np.pi * 2.)

    def get_impl(self):
        hmin = self.hmin
        hmax = self.hmax

        @numba.njit(numba.complex128(numba.complex128))
        def numba_impl(z):
            # h linearly interpolated between hmin and hmax
            h = hmin * (0.5 - z.real) + hmax * (0.5 + z.real)
            # t linearly interpolated between -pi and pi
            t = z.imag * (hmax - hmin)
            return  np.exp(complex(h, t))

        return numba_impl

    @property
    def name(self):
        return "expmap"

    
#class Inverse(Projection):
#    def __init__(self, rlim):
#        self.args = None
#    
#    def get_impl(self):
        

    