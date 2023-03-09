# -*- coding: utf-8 -*-
import numpy as np
import numba

import fractalshades.numpy_utils.filters as fsfilters
"""
This module implements interpolation routines
"""

CHECK_BOUNDS = False


class Grid_lin_interpolator:
    
    def __init__(self, a, b, h, f):
        """
        Linear interpolation inside a regular 2d grid

        Parameters:
        -----------
        a: 2-uplet, float
            The lower bounds of the interpolation region
        b: 2-uplet, float
            The upper bounds of the interpolation region
        h: 2-uplet, float
            The grid-spacing at which f is given (2-uplet)
        f: 2d-array
            The base data to be interpolated
        """
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.h = np.asarray(h)
        self.f = np.asarray(f)
        # The numba interpolating implementations
        self.numba_impl = self.get_impl()
        self.parallel_impl = self.get_parallel_impl()


    def get_impl(self):
        """
        Return a numba-jitted function for bilinear interpolation of 1 point
        """
        a = self.a
        b = self.b
        h = self.h
        f = self.f
        
        @numba.njit(nogil=True, parallel=False)
        def numba_impl(x_out, y_out):
            # Interpolation: f_out = finterp(x_out, y_out)
            if CHECK_BOUNDS:
                x_out = min(max(x_out, a[0]), b[0])
                y_out =  min(max(y_out, a[1]), b[1])

            ix, ratx = divmod(x_out - a[0], h[0])
            iy, raty = divmod(y_out - a[1], h[1])
            ix = np.intp(ix)
            iy = np.intp(iy)
            ratx /= h[0]
            raty /= h[1]

            cx0 = 1. - ratx
            cx1 = ratx
            cy0 = 1. - raty
            cy1 = raty

            f_out = (
                (cx0 * cy0 * f[ix, iy])
                + (cx0 * cy1 * f[ix, iy + 1])
                + (cx1 * cy0 * f[ix + 1, iy])
                + (cx1 * cy1 * f[ix + 1, iy + 1])
            )
            return f_out

        return numba_impl


    def get_parallel_impl(self):
        """
        Return a numba-jitted function for bilinear interpolation of 1d vecs 
        """
        impl = self.numba_impl

        @numba.njit(nogil=True, parallel=True)
        def serial_impl(pts_x, pts_y, pts_res):
            m = pts_res.shape[0]
            for i in numba.prange(m):
                pts_res[i] = impl(pts_x[i], pts_y[i])

        return serial_impl


    def __call__(self, pts_x, pts_y, pts_res=None):
        """
        Interpolates at pts_x, pts_y

        Parameters:
        ----------
        pts_x: 1d-array
            x-coord of interpolating point location
        pts_y: 1d-array
            y-coord of interpolating point location
        pts_res: 1d-array, Optionnal
            Out array handle - if not provided, will be created. 
        """
        assert np.ndim(pts_x) == 1

        if pts_res is None:
            pts_res = np.empty_like(pts_x)

        interp = self.parallel_impl
        interp(pts_x, pts_y, pts_res)

        return pts_res

#
##------------------------------------------------------------------------------
#def lanczos2_kernel(n, a=-4., b=4.):
#    """ Returns a discretization of Lanczos-2 kernel with n +1 discrete points
#    """
#    # Note: np.sinc(x) is np.sin(np.pi * x) / (np.pi * x) for numpy
#    x = np.linspace(a, b, n + 1)
#    return np.where(
#        np.abs(x) < 2., np.sinc(x) * np.sinc(x / 2.), 0.
#    )
#
#
#class Grid_multilevel_lf2_interpolator:
#    
#    def __init__(self, a, b, h, f):
#        """
#        Lanczos-2 nterpolation inside a regular 2d grid
#
#        Parameters:
#        -----------
#        a: 2-uplet, float
#            The lower bounds of the interpolation region
#        b: 2-uplet, float
#            The upper bounds of the interpolation region
#        h: 2-uplet, float
#            The grid-spacing at which f is given (2-uplet). Note that it is 
#            assumed that h[0] ~ h[1] / except fo rounding errors.
#        f: 2d-array
#            The base data to be interpolated
#        """
#        self.a = np.asarray(a)
#        self.b = np.asarray(b)
#        self.h = np.asarray(h)
#
#        # The numba multilevel_arrays
#        self.make_multilevel(np.asarray(f))
#        
#        
#        # The numba interpolating implementations
#        self.numba_impl = self.get_impl()
#        self.parallel_impl = self.get_parallel_impl()
#
#    def make_multilevel(self, f):
#        """ Return a list of Lanczos-2 level2 decimated array for subsequent
#        interpolation, with a Lf2 filter 
#        
#        """
#        # Lanczos2 decimator
#        lf2 = fsfilters.Lanczos_decimator().get_impl(2, 2)
#        # Taking into account the Lanczos2 2-decimantion kernel wr need a 
#        # border of 3 
#        border = 3
#        multi_f = [np.pad(f, border, mode="edge")]
#        ff = f
#        nx, ny = np.shape(f)
#        nm = max(nx, ny) 
#        lvls = (nm - 1).bit_length()
#        # 4 -> 2 [2, *1]    5 -> 3 [3, 2, *1]  9 -> 4 [5, 3, 2, *1]
#        # Note that the first is already done
#        dpix = 1
#        for i in range(lvls - 1):
#            dpix <<= 1 # TODO keep track of the low-low corner positions
#            nnx = (nx + 1) // 2
#            nny = (ny + 1) // 2
#            # Padding
#            padx = (nx % 2)
#            pady = (ny % 2)
#            high = (i + 1)%4 // 2
#            right = (i + 1)%4 % 2
#            pad_width = (
#                (padx * high, padx * (1 - high)),
#                (pady * right, pady * (1 - right))
#            )
#            padded = np.pad(ff, pad_width, mode="edge")
#            ff = lf2(padded)
#            multi_f.append(np.pad(ff, border, mode="edge"))
#        self.lvls = lvls
#        self.border = border
#        self.multi_f = multi_f
#        
#
#
#    def get_impl(self):
#        """
#        Return a numba-jitted function for bilinear interpolation of 1 point
#        """
#        a = self.a
#        b = self.b
#        h = self.h
#        multi_f = self.multi_f # Note: need boder-filling
#        lvls = self.lvls
#        border = self.border
#
#
#        @numba.njit(nogil=True, parallel=False)
#        def numba_impl(x_out, y_out, r_out):
#            # Interpolation: f_out = finterp(x_out, y_out)
#            # r_out is the pixel scale -> scale of the support
#            
#            r_out / h[0]
#            
#            if CHECK_BOUNDS:
#                x_out = min(max(x_out, a[0]), b[0])
#                y_out =  min(max(y_out, a[1]), b[1])
#            
#
#
#            ix, ratx = divmod(x_out - a[0], h[0])
#            iy, raty = divmod(y_out - a[1], h[1])
#            ix = np.intp(ix)
#            iy = np.intp(iy)
#            ratx /= h[0]
#            raty /= h[1]
#
#
#            # L2 convolution Kernel support width
#            kwx = max(r_out / h[0], 1.) * 2.
#            kwy = max(r_out / h[1], 1.) * 2.
#            min_ik = int(kwx - ratx) + 1
#            max_ik = int(kwx + ratx) + 1
#            min_jk = int(kwy - raty) + 1
#            max_jk = int(kwy + raty) + 1
#            
#            for ik in range(2 * nkx):
#                kernelx[ik] = l2k[]
#            for jk in range(2 * nky):
#                kernely[jk] = l2k[]
#
#
#            cx0 = 1. - ratx
#            cx1 = ratx
#            cy0 = 1. - raty
#            cy1 = raty
#
#            f_out = (
#                (cx0 * cy0 * f[ix, iy])
#                + (cx0 * cy1 * f[ix, iy + 1])
#                + (cx1 * cy0 * f[ix + 1, iy])
#                + (cx1 * cy1 * f[ix + 1, iy + 1])
#            )
#            f_out = 0.
#            for di in range(-support, support):
#                fir dj in range(-support, support):
#                    f_out += kernelx[di] * kernely[dj] * f[ix + di, iy + dj]
#            
#            return f_out
#
#        return numba_impl
#
#
#    def get_parallel_impl(self):
#        """
#        Return a numba-jitted function for bilinear interpolation of 1d vecs 
#        """
#        impl = self.numba_impl
#
#        @numba.njit(nogil=True, parallel=True)
#        def serial_impl(pts_x, pts_y, pts_res):
#            m = pts_res.shape[0]
#            for i in numba.prange(m):
#                pts_res[i] = impl(pts_x[i], pts_y[i])
#
#        return serial_impl
#
#
#    def __call__(self, pts_x, pts_y, pts_res=None):
#        """
#        Interpolates at pts_x, pts_y
#
#        Parameters:
#        ----------
#        pts_x: 1d-array
#            x-coord of interpolating point location
#        pts_y: 1d-array
#            y-coord of interpolating point location
#        pts_res: 1d-array, Optionnal
#            Out array handle - if not provided, will be created. 
#        """
#        assert np.ndim(pts_x) == 1
#
#        if pts_res is None:
#            pts_res = np.empty_like(pts_x)
#
#        interp = self.parallel_impl
#        interp(pts_x, pts_y, pts_res)
#
#        return pts_res
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#class Gridexpmap_interpolator:
#
#    def __init__(self, a, b, dh, f):
#        """
#        Interpolation inside an exponential 2d grid ([h1, h2] x [-pi, pi])
#
#        Parameters:
#        -----------
#        a: float
#            The lower h-bounds of the interpolation region - usually [h1, -pi]
#        b: 2-uplet, float
#            The upper h-bounds of the interpolation region - usually [h2, +pi]
#        dh: 2-uplet, float
#            The h grid-spacing at which f is given (2-uplet)
#        f: 2d-array
#            The base data to be interpolated
#        """
#        self.a = np.asarray(a)
#        self.b = np.asarray(b)
#        self.dh = np.asarray(dh)
#        self.f = np.asarray(f)
#        # The numba interpolating implementations
#        self.numba_impl = self.get_impl()
#
#
#    def get_impl(self):
#        """
#        Return a numba-jitted function for expmap interpolation of 1 point
#        """
#        a = self.a
#        b = self.b
#        dh = self.dh
#        f = self.f
#
#        @numba.njit(nogil=True, parallel=False)
#        def numba_impl(h_out, t_out):
#            # Interpolation: f_out = finterp(h_out, t_out)
#            # this is basically a bilinear interpolation
#            if CHECK_BOUNDS:
#                h_out = min(max(h_out, a[0]), b[0])
#                t_out =  min(max(t_out, a[1]), b[1])
#
#            ix, ratx = divmod(h_out - a[0], dh[0])
#            iy, raty = divmod(t_out - a[1], dh[1])
#            ix = np.intp(ix)
#            iy = np.intp(iy)
#            ratx /= dh[0]
#            raty /= dh[1]
#
#            cx0 = 1. - ratx
#            cx1 = ratx
#            cy0 = 1. - raty
#            cy1 = raty
#
#            f_out = (
#                (cx0 * cy0 * f[ix, iy])
#                + (cx0 * cy1 * f[ix, iy + 1])
#                + (cx1 * cy0 * f[ix + 1, iy])
#                + (cx1 * cy1 * f[ix + 1, iy + 1])
#            )
#            return f_out
#
#        return numba_impl
#
#
#class Gridexpmap_layered_interpolator:
#
#    def __init__(self, a_s, b_s, h_s, f_s, validity_pix_radius):
#        """
#        Interpolation inside an exponential map 2d grid ([0, h] x [-pi, pi])
#
#        This is a multi-layered interpolation, featuring :
#            - several concentric exponential-mapping interpolators
#              (with Lanczos-2 data filtering): valid for a given zoom range
#            - a final cartesian interpolator
#
#        Parameters:
#        -----------
#        a_s: list of 2-uplet, float
#            The lower bounds of the interpolation region, in exp coordinates
#            except the last, in cartesian
#        b_s: list of 2-uplet, float
#            The upper bounds of the interpolation region, in exp coordinates
#        h_s: list of 2-uplet, float
#            The grid-spacing at which f is given (2-uplet), in exp coordinates
#        f_s: 2d-array
#            The base data to be interpolated
#        validity_pix_radius: float
#            The max pix width "zoom level" for an accurate interpolation 
#        """
#        
#
#    def get_impl(self):
#        """
#        Return a numba-jitted function for expmap interpolation of 1 point
#        """
#
#
#        @numba.njit(nogil=True, parallel=False)
#        def numba_impl(x_out, y_out):
#
#            # early exit if in the final image
#            if x_out < x_final and y_out < y_final:
#                return xy_interp(x_out, y_out)
#
#            h_out = np.log(np.hypot(x_out, y_out))
#            t_out = np.arctan2(y_out, x_out)
#            
#            for i in range(ninterp_exmpap):
#                if h_out < b_s[i]: #  a_s[i] < h_out < b_s[i] == a_s[i + 1]
#                    f_out = (interp[ninterp_exmpap])(h_out, t_out)
#
#            return f_out
#
#        return numba_impl