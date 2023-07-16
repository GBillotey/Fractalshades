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
        self.make_impl()

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
    def df_scale(self):
        """ Used in to Perturbation fractal as a corrective
        scaling factor used in dZndC computation (and consequently distance
        estimation, ...)
        Projections supporting arbitray precision shall implement.

        Returns:
        --------
        df_scale: numba compiled function pix -> scaling or None
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
        # proj_dx = self.dx * self.projection.scale
        # corner_a = lin_proj_impl_noscale(0.5 * (w + 1j * h)) * proj_dx
        # corner_a is absolute the distance to center
        raise NotImplementedError(
            f"Arbitray precision not implemented for {self.__class__.__name__}"
        )
    
    @property
    def min_local_scale(self):
        """ Used in to Perturbation fractal to define min pix size. Minimum of
        local |df| (before general scaling `scale`)
        Projections supporting arbitray precision shall implement.

        Returns:
        --------
        scale: mpmath.mpf or float, the scale factor
        """
        # pix = proj.min_local_scale * proj.scale * (f.dx / f.nx)
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
        pass

    def make_f_impl(self):
        """ A cartesian projection just let pass-through the coordinates"""
        self.f = cartesian_numba_impl

    def make_df_impl(self):
        self.df = None

    def make_dfBS_impl(self):
        self.dfBS = None

    @property
    def df_scale(self):
        return None

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
    def __init__(self, hmin, hmax, rotates_df=True, orientation="horizontal"):
        """ 
        An exponential projection will map :math:`z_{pix}` as follows:

        .. math::

            \\bar{z}_{pix} = \\exp(h_{moy}) \\cdot \\exp(dh \\cdot z_{pix}) 

        where:

        .. math::

            h_{moy} &= \\frac{1}{2} \\cdot (h_{min} + h_{max}) \\\\
            dh &= h_{max} - h_{min}

        Notes
        =====
        Adjustment of zoom parameters:
        The `xy_ratio` of the zoom will be adjusted (during run time) to ensure
        that :math:`\\bar{y}_{pix}` extends from :math:`- \\pi`
        to :math:`\\pi`. `nx` is interpreted as `nh` be the  `direction`
        "horizontal" or "vertical".

        This class can be used with arbitrary-precision deep zooms.

        Parameters
        ==========
        hmin: str or float or mpmath.mpf
            scaling at the lower end of the h-axis, hmin >= 0.
        hmax: str or float or mpmath.mpf
            scaling at the higher end of the h-axis, hmax > hmin
        rotates_df: bool
            If ``True`` (default), the derivative will be scaled but also
            rotated according to the mapping. If ``False``, only the scaling
            will be taken into account.
            A rule of thumb is this value shall be set to ``True``
            for a standalone picture, and to ``False`` if used as input for a
            movie making tool.
        orientation: "horizontal" | "vertical"
            The direction for the h axis. Defaults to "horizontal".
        """
        if not(0 <= hmin < hmax):
            raise ValueError(
                "Provide hmin, hmax with:  0 <= hmin < hmax for Expmap"
            )
        self.use_step = False
        self.rotates_df = rotates_df
        self.orientation = orientation
        self.premul_1j = {"horizontal": False, "vertical": True}[orientation]

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

    def nh(self, fractal):
        return fractal.ny if self.premul_1j else fractal.nx

    def nt(self, fractal):
        return fractal.nx if self.premul_1j else fractal.ny

    @property
    def pix_to_ht(self):
        """ Factor that transforms a pixel to ht coordinates
        Note: 0 (scling 1.) is centered
        """
        dh = self.dh
        premul_1j = self.premul_1j
        xy_ratio = self.xy_ratio
        return (1j * dh * xy_ratio) if premul_1j else dh

#    @property
#    def pix_to_t(self):  #   pix_to_t = self.pix_to_t
#        premul_1j = self.premul_1j
#        xy_ratio = self.xy_ratio
#        return (1j * xy_ratio) if premul_1j else 1.

# /!\ Warning of the part with xyratio
#    @property
#    def kz(self):
#        dh = self.dh
#        premul_1j = self.premul_1j
#        xy_ratio = self.xy_ratio
#        return (1j * dh * xy_ratio) if premul_1j else dh




    def set_exp_zoom_step(self, exp_step_hmax, exp_step_hmin):
        # property defined in ``save_db`` with exp_zoom_step set
        logger.debug(
                f"set_exp_zoom_step: {exp_step_hmin} {exp_step_hmax}\n"
                f"for a full range of {self.hmin} {self.hmax}"
        )
        self.exp_step_hmax = exp_step_hmax
        self.exp_step_hmin = exp_step_hmin
        self.exp_step_hmoy = 0.5 * (exp_step_hmax + exp_step_hmin)
        self.exp_step_dh = (exp_step_hmax - exp_step_hmin)
        self.use_step = True

    def del_exp_zoom_step(self):
        """ property used in ``save_db`` with exp_zoom_step set """
        self.use_step = False

    def adjust_to_zoom(self, fractal):
        # We need to adjust the fractal xy_ratio in order to match hmax - hmin
        # target: dh = 2. * np.pi * xy_ratio 
        if self.premul_1j:
            xy_ratio = (np.pi * 2.) / self.dh
            nx = int(fractal.nx * xy_ratio + 0.5)
        else:
            xy_ratio = self.dh / (np.pi * 2.)
            nx = fractal.nx

        fractal.xy_ratio = self.xy_ratio = xy_ratio
        fractal.zoom_kwargs["xy_ratio"] = xy_ratio
        fractal.nx = nx
        fractal.zoom_kwargs["nx"] = nx

        logger.info(
            "Adjusted parameters for Expmap projection:\n"
            f"zoom parameter nx: {fractal.nx}\n"
            f"zoom parameter xy_ratio: {fractal.xy_ratio}"
        )
        self.make_impl()


    def make_f_impl(self):
        """ apply the full transform """
        hmoy = self.hmoy
        pix_to_ht = self.pix_to_ht

        @numba.njit(
            numba.complex128(numba.complex128), nogil=True, fastmath=False)
        def numba_impl(z):
            return np.exp(hmoy + pix_to_ht * z)

        self.f = numba_impl


    def make_df_impl(self):
        """ Rotate the derivatives of the transform, according to the options
        """
        pix_to_ht = self.pix_to_ht

        if self.rotates_df:
            @numba.njit(
                numba.complex128(numba.complex128), nogil=True, fastmath=False)
            def numba_impl(z):
                return np.exp(np.real(pix_to_ht * z.real))

        else:
            @numba.njit(
                numba.complex128(numba.complex128), nogil=True, fastmath=False)
            def numba_impl(z):
                return 1.

        self.df = numba_impl


    def make_dfBS_impl(self):
        premul_1j = self.premul_1j
        pix_to_ht = self.pix_to_ht

        if self.rotates_df:
            @numba.njit(nogil=True, fastmath=False)
            def numba_impl(z):
                if premul_1j:
#                    zhr = z.imag
                    zhi = pix_to_ht * z.real
                else:
#                    zhr = z.real
                    zhi = pix_to_ht * z.imag
#                r = np.exp(zhr)
                c = np.cos(zhi)
                s = np.sin(zhi)
                if premul_1j:
                    return -s, -c, c, -s
                else:
                    return c, -s, s, c

        else:
            @numba.njit(nogil=True, fastmath=False)
            def numba_impl(z):
                if premul_1j:
                    return  0., -1., 1., 0.
                else:
                    return  1., 0., 0., 1.

        self.dfBS = numba_impl

#    @property
#    def df_scale_factor(self):
#        """ The scaling factor applied to derivative iteration """
#        if self.use_step:
#            hmoy = self.hmoy
#            exp_step_hmoy = self.exp_step_hmoy
#            return np.exp(hmoy - exp_step_hmoy)
#        return 1.
#
#
#    @property
#    def df_scale_func(self):
#        """ The scaling factor applied to derivative iteration """
#        pix_to_ht = self.pix_to_ht
#        if self.use_step:
#            hmoy_df = self.exp_step_hmoy
#        else:
#            hmoy_df = self.hmoy
#            # return mpmath.exp(self.exp_step_hmoy)
#
#        @numba.njit(nogil=True, fastmath=False)
#        def numba_impl(z):
#            return np.exp(hmoy_df + np.real(pix_to_ht * z))
#
#        return numba_impl #mpmath.exp(self.hmoy)

    @property # TODO: suppress this
    def scale(self):
        # Returns the scaling - for perturbation implementation
        if self.use_step:
            hmoy_df = self.exp_step_hmoy
        else:
            hmoy_df = self.hmoy
        return mpmath.exp(hmoy_df)

    def bounding_box(self, xy_ratio):
        """
        This is the scaling of the bounding box for this calculation step
        ie either the whole image or the current exmpap step sub-image.
        """
        if self.use_step:
            print("******* BOUNDING BOX with hstep_max", self.exp_step_hmax)
            wh = self.exp_step_hmax
        else:
            wh = self.hmax

        w = mpmath.exp(wh)
        return w, w

    @property
    def min_local_scale(self):
        """ This is the scaling of the minimum pixel due to the projection
        for the *whole* image (accumulated by steps if activated)
        """
        return mpmath.exp(self.hmin)

    @property
    def min_step_scale(self):
        """ This is the scaling of the minimum pixel due to the projection
        for the *whole* image (accumulated by steps if activated)
        """
        if self.use_step:
            return mpmath.exp(self.exp_step_hmin)
        else:
            return mpmath.exp(self.hmin)

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

































