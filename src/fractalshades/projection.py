# -*- coding: utf-8 -*-
import inspect
#import enum

import numpy as np
import numba
import mpmath

import fractalshades as fs
import fractalshades.settings
import fractalshades.numpy_utils.xrange as fsx


# Log attributes
# https://docs.python.org/2/library/logging.html#logrecord-attributes

class Projection:
    """
    A Projection defines a mapping acting on the screen-pixels before
    iteration.
    It implements the associated modifications of the derivatives, when
    needed (for distance estimation or shading).

    For a `xy_ratio` - zoom parameter - equal to 1, screen pixels are defined
    in :math:`[0.5, 0.5] \\times [-0.5, 0.5]`
    
    More generally, the screen is a
    rectangle and pixel coordinates take values in:
    
    .. math::

        [0.5, 0.5] \\times \\left[ -\\frac{0.5}{xy\\_ratio},
        \\frac{0.5}{xy\\_ratio} \\right]

    The projection is defined as:
        
    .. math::

        (\\bar{x}_{pix}, \\bar{y}_{pix}) =  f(x_{pix}, y_{pix})
    
    or alternatively with complex notations:
        
    .. math::

        \\bar{z}_{pix} =  f(z_{pix})

    Derived classes shall implement the actual  :math:`f` function
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
        """
        2 `Projections` are equal if:

            - they are instances from the same subclass
            - they have been created with the same init_kwargs
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
        """
        A Cartesian projection. This is simply the identity function:
        
        .. math::

            \\bar{z}_{pix} =  z_{pix}
        """
        self.make_impl()

    def make_f_impl(self):
        """ A cartesian projection just let pass-through the coordinates"""
        @numba.njit(numba.complex128(numba.complex128))
        def numba_impl(z):
            return z
        self.f = numba_impl

    def make_df_impl(self):
        self.df = None

    def make_dfBS_impl(self):
        self.dfBS = None

#==============================================================================
class Expmap(Projection):
    def __init__(self, hmin, hmax, rotates_df=True):
        """ 
        An exponential mapping will map :math:`z_{pix}` as follows:

        .. math::

            \\bar{z}_{pix} = \\exp(h_{moy}) \\cdot \\exp(dh \\cdot z_{pix}) 

        where:
            
        .. math::

            h_{moy} &= \\frac{1}{2} \\cdot (h_{min} + h_{max}) \\\\
            dh &= h_{max} - h_{min}

        `xy_ratio` of the fractal will be adjusted to ensure that
        :math:`\\bar{y}_{pix}` extends from :math:`- \\pi` to  :math:`\\pi`.

        Parameters
        ==========
        hmin: str or float or mpmath.mpf
            scaling at the lower end of the x-axis
        hmax: str or float or mpmath.mpf
            scaling at the higher end of the x-axis
        rotates_df: bool
            If ``True``, the derivative will be scaled but also rotated
            according to the mapping. Otherwise, only the scaling will be taken
            into account.
            A rule of thumb is this value shall be set to ``True``
            for a standalone picture, and to ``False`` if used as input for a
            movie making tool.
        """
        self.rotates_df = rotates_df

        if mpmath.exp(hmax) > (1. / fs.settings.xrange_zoom_level):
            # Or ~ hmax > 690... We store internally as Xrange
            self.hmin = fsx.mpf_to_Xrange(hmin)
            self.hmax = fsx.mpf_to_Xrange(hmax)
        else:
            # We store internally as float
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
        fractal.zoom_kwargs["xy_ratio"] = fractal.xy_ratio

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
            return np.exp(complex(h, t))

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
