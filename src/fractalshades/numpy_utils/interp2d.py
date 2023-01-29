# -*- coding: utf-8 -*-
import numpy as np
import numba
"""
This module implements 2d bilinear interpolation routine on a rectangular grid

"""

CHECK_BOUNDS = False


class Grid_lin_interpolator:
    
    def __init__(self, a, b, h, f):
        """
        Interpolation inside a regular 2d grid

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
        
        


