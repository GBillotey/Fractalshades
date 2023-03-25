# -*- coding: utf-8 -*-
import inspect
import logging
import copy

import numpy as np
import numba
import mpmath

import fractalshades as fs
import fractalshades.settings
import fractalshades.numpy_utils.xrange as fsx


logger = logging.getLogger(__name__)

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
        """ Used in to Perturbation fractal as a corrective
        scaling factor used in dZndC computation (and consequently distance
        estimation, ...)
        Projections supporting arbitray precision shall implement.

        Returns:
        --------
        scale: mpmath.mpf or float, the scale factor
        """
        raise NotImplementedError(
            f"Arbitray precision not supported by {self.__class__.__name__}"
        )

    @property
    def bounding_box(self, xy_ratio):
        """ Passed to Perturbation fractal as an enveloppe of the mapping of
        the pixels (before general scaling `scale`)
        Projections supporting arbitray precision shall implement.

        Returns:
        --------
        w, h: floats, width and height of the bounding box
        """
        raise NotImplementedError(
            f"Arbitray precision not implemented for {self.__class__.__name__}"
        )
    
    @property
    def min_local_scale(self):
        """ Used in to Perturbation fractal todefine min pix size. Minimum of
        local |df| (before general scaling `scale`)
        Projections supporting arbitray precision shall implement.

        Returns:
        --------
        scale: mpmath.mpf or float, the scale factor
        """
        raise NotImplementedError(
            f"Arbitray precision not supported by {self.__class__.__name__}"
        )

# Development notes: Projections and derivatives / normals
# For perturbation fractals, the derivatives are relative to a "window" scale
# dx (in Cartesian mode). This scale become dx * projection.scale (constant
# real scalar) for non-cartesian projections.
# On top of this general scaling, 2 "local" modifiers may be implemented 
# - total derivative (account for variable local scaling + rotation + skew)
#   (mandatory)
# - derivative "without rotation" useful if the final aim is to unwrap, for
#   instance to make a videofrom a expmap (optionnal)

#==============================================================================
class Cartesian(Projection):
    def __init__(self):
        """
        A Cartesian projection. This is simply the identity function:

        .. math::

            \\bar{z}_{pix} =  z_{pix}

        This class can be used with arbitrary-precision deep zooms.
        """
        self.make_impl()

    def make_f_impl(self):
        """ A cartesian projection just let pass-through the coordinates"""
        self.f = cartesian_numba_impl

    def make_df_impl(self):
        self.df = None

    def make_dfBS_impl(self):
        self.dfBS = None

    @property
    def scale(self):
        return 1.

    def bounding_box(self, xy_ratio):
        return 1., 1. / xy_ratio
    
    @property
    def min_local_scale(self):
        return 1.

@numba.njit(nogil=True, fastmath=True)
def cartesian_numba_impl(z):
    return z

#==============================================================================
class Expmap(Projection):
    def __init__(self, hmin, hmax, rotates_df=True):
        """ 
        An exponential projection will map :math:`z_{pix}` as follows:

        .. math::

            \\bar{z}_{pix} = \\exp(h_{moy}) \\cdot \\exp(dh \\cdot z_{pix}) 

        where:

        .. math::

            h_{moy} &= \\frac{1}{2} \\cdot (h_{min} + h_{max}) \\\\
            dh &= h_{max} - h_{min}

        The `xy_ratio` of the zoom will be adjusted (during run time) to ensure
        that :math:`\\bar{y}_{pix}` extends from :math:`- \\pi`
        to :math:`\\pi`.

        This class can be used with arbitrary-precision deep zooms.

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
#         Development notes: TODO REVIEW THESE NOTES
#         ------------------
#         For arbitrary deep zooms this projection is implemented as the
#         non-evaluated composition of a scaling np.exp(self.hmoy) and a
#         local exponential mapping.
#         For stadard precision this is simply the mapping as described in the
#         docstring above.

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
        # We need to adjust the fractal xy_ratio in order to match hmax - hmin
        # target: dh = 2. * np.pi * xy_ratio 
        fractal.xy_ratio = self.dh / (np.pi * 2.)
        fractal.zoom_kwargs["xy_ratio"] = fractal.xy_ratio

        logger.info(
            "Adjusted parameters for Expmap projection:\n"
            f"zoom parameter xy_ratio: {fractal.xy_ratio}"
        )


    def make_f_impl(self):
        hmoy = self.hmoy
        dh = self.dh

        @numba.njit(
            numba.complex128(numba.complex128), nogil=True, fastmath=False)
        def numba_impl(z):
            return np.exp(hmoy + dh * z)

        self.f = numba_impl


    def make_df_impl(self):
        dh = self.dh

        if self.rotates_df:
            @numba.njit(
                numba.complex128(numba.complex128), nogil=True, fastmath=False)
            def numba_impl(z):
                return  np.exp(dh * z)

        else:
            @numba.njit(
                numba.complex128(numba.complex128), nogil=True, fastmath=False)
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
    def scale_mpf(self):
        return mpmath.exp(self.hmoy)

    @property
    def scale(self):
        # Returns the scaling - for perturbation implementation
        return mpmath.exp(self.hmoy)
    
    def bounding_box(self, xy_ratio):
        # Returns a bounding box - for perturbation implementation only
        w = np.exp(0.5 * self.dh)
        return w, w

    @property
    def min_local_scale(self):
        return  np.exp(-0.5 * self.dh)


#@numba.njit(nogil=True, fastmath=False) #, cache=True)
#def expmap_numba_impl(z, hmoy, dh):
##    # h linearly interpolated between hmin and hmax
##    h = hmoy + dh * z.real
##    # t linearly interpolated between -pi and pi
##    t = dh * z.imag
##    return np.exp(complex(h, t))
#    return np.exp(hmoy + dh * z)
#
#@numba.njit(nogil=True, fastmath=False) #, cache=True)
#def expmap_numba_perturb_impl(z, dh):
#    # Local mapping only (without exp(hmoy scaling))
#    return np.exp(dh * z)

#============================================================================== 

class Generic_mapping(Projection):
    def __init__(self, f, df):
        """ 
        This class allows the user to provide a custom mapping defined
        on the complex plane.

        The transformation between the complex plane and local pixel
        coordinates (map :math:`z_{pix}`, according to the zoom level)
        is managed internally.

        Known limitations:

         - currently only differentiable mappings of the
           complex variable (aka holomorphic / meromorphic functions) are
           supported.
         - not suitable for arbitray precision fractals.

        Parameters
        ==========
        f: numba jitted function numba.complex128 -> numba.complex128
            The complex function defining the mapping. If a tuple (f1, f2) or
            (f1, f2, f3) is provided the composition will be applied (f1 o f2)
        df: numba jitted function numba.complex128 -> numba.complex128
            The differential of f. If a tuple (df1, df2) or (df1, df2, df3) is
            provided the differential of the composition will be applied
            according to the differentiation chain rule.
        """
        self.f = f
        self.df = df

        _f = f if fs.utils.is_iterable(f) else (f,)
        _df = df if fs.utils.is_iterable(df) else (df,)

        if (len(_df) != len(_f)):
            raise ValueError("len(f) and len(df) shall match")
        self.n_func = len(_f)

        self.P_f_numba = copy.copy(_f)
        self.P_df_numba = copy.copy(_df)


    def make_phi(self):
        """
        phi : pixel to C-plane
        phi_inv: C-plane to pixel
        """
        z_center = self.z_center 
        dx = self.dx

        @numba.njit(nogil=True, fastmath=False)
        def numba_phi_impl(zpix):
            return  z_center + zpix * dx

        @numba.njit(nogil=True, fastmath=False)
        def numba_phi_inv_impl(z):
            return  (z - z_center) / dx

        self.phi = numba_phi_impl
        self.phi_inv = numba_phi_inv_impl


    def adjust_to_zoom(self, fractal):
        """ JIT compilation is delayed until we know the zoom
        parameters"""
        self.z_center = complex(fractal.x, fractal.y)
        self.dx = fractal.dx

        self.make_phi()
        self.make_impl()


    def make_f_impl(self):
        n_func = self.n_func
        phi = self.phi
        phi_inv = self.phi_inv
        P_f_numba = self.P_f_numba
        
        if n_func == 1:
            f0 = P_f_numba[0]
            @numba.njit(numba.complex128(numba.complex128),
                        nogil=True, fastmath=False)
            def numba_impl(z):
                zz = phi(z) # pix -> P
                zz = f0(zz)
                zz = phi_inv(zz) # P -> pix
                return zz

        elif n_func == 2:
            f0 = P_f_numba[0]
            f1 = P_f_numba[1]
            @numba.njit(numba.complex128(numba.complex128),
                        nogil=True, fastmath=False)
            def numba_impl(z):
                zz = phi(z) # pix -> P
                zz = f0(zz)
                zz = f1(zz)
                zz = phi_inv(zz) # P -> pix
                return zz
            
        elif n_func == 3:
            f0 = P_f_numba[0]
            f1 = P_f_numba[1]
            f2 = P_f_numba[2]
            @numba.njit(numba.complex128(numba.complex128),
                        nogil=True, fastmath=False)
            def numba_impl(z):
                zz = phi(z) # pix -> P
                zz = f0(zz)
                zz = f1(zz)
                zz = f2(zz)
                zz = phi_inv(zz) # P -> pix
                return zz

        else:
            raise NotImplementedError("Composition allowed up to 3 funcs only")

        self.f = numba_impl

    def make_df_impl(self):
        n_func = self.n_func
        phi = self.phi
        P_f_numba = self.P_f_numba
        P_df_numba = self.P_df_numba

        if n_func == 1:
            df0 = P_df_numba[0]
            @numba.njit(numba.complex128(numba.complex128),
                        nogil=True, fastmath=False)
            def numba_impl(z):
                zz = phi(z) # pix -> P
                dzz = df0(zz)
                return dzz

        elif n_func == 2:
            df0 = P_df_numba[0]
            f0 = P_f_numba[0]
            df1 = P_df_numba[1]
            @numba.njit(numba.complex128(numba.complex128),
                        nogil=True, fastmath=False)
            def numba_impl(z):
                zz = phi(z) # pix -> P
                dzz = df0(zz)
                zz = f0(zz)
                dzz *= df1(zz)
                return dzz

        elif n_func == 3:
            df0 = P_df_numba[0]
            f0 = P_f_numba[0]
            df1 = P_df_numba[1]
            f1 = P_f_numba[1]
            df2 = P_df_numba[2]
            @numba.njit(numba.complex128(numba.complex128),
                        nogil=True, fastmath=False)
            def numba_impl(z):
                zz = phi(z) # pix -> P
                dzz = df0(zz)
                zz = f0(zz)
                dzz *= df1(zz)
                zz = f1(zz)
                dzz *= df2(zz)
                return dzz

        else:
            raise NotImplementedError("Composition allowed up to 3 funcs only")

        self.df = numba_impl


    def make_dfBS_impl(self):
        """ Currently not implemented, will raise an assertion error if 
        called in LLVM """

        @numba.njit(nogil=True, fastmath=False)
        def numba_impl(z):
            assert False
            return z

        self.dfBS = numba_impl

































