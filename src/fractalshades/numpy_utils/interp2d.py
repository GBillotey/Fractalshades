# -*- coding: utf-8 -*-
import numpy as np
import numba

"""
This module implements interpolation routines
"""

CHECK_BOUNDS = False

class Grid_lin_interpolator:
    
    def __init__(self, a, b, h, f, PIL_order=False):
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
            The base data to be interpolated, in order as specified by
            `PIL_order`
        PIL_order: bool
            If True, use pillow array indexing : y, x with y in reversed order
        """
        print("a", a)
        print("b", b)
        print("h", h)
        print("PIL_order", PIL_order)
        self.interp_args = tuple(np.asarray(p) for p in (a, b, h, f))
        self.PIL_order = PIL_order


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
        interp_args = self.interp_args
        PIL_order = self.PIL_order

        if pts_res is None:
            pts_res = np.empty_like(pts_x)
        
        if PIL_order:
            interpolate_2d_PIL(
                pts_x, pts_y, pts_res, *interp_args
            )
        else:
            interpolate_2d(
                pts_x, pts_y, pts_res, *interp_args
            )

        return pts_res

@numba.njit(nogil=True, parallel=False)
def interpolate_2d(pts_x, pts_y, pts_res, a, b, h, f):
    """  In place filling of pts_res array
    """

    m = pts_res.shape[0]
    max_ix = f.shape[0] - 2
    max_iy = f.shape[1] - 2


    for mi in range(m):
        x_out = pts_x[mi]
        y_out = pts_y[mi]
    
        if CHECK_BOUNDS:
            x_out = min(max(x_out, a[0]), b[0])
            y_out =  min(max(y_out, a[1]), b[1])
    
        ix_float, ratx = divmod(x_out - a[0], h[0])
        iy_float, raty = divmod(y_out - a[1], h[1])
        ix_float = min(ix_float, max_ix)
        iy_float = min(iy_float, max_iy)
        
#        assert(ix_float) > 0
#        assert(iy_float) > 0
#        assert(ix_float + 1) < fnx
#        assert(iy_float + 1) < fny
    
        ix = np.intp(ix_float)
        iy = np.intp(iy_float)
        ratx /= h[0]
        raty /= h[1]
    
        cx0 = 1. - ratx
        cx1 = ratx
        cy0 = 1. - raty
        cy1 = raty
    
        pts_res[mi] = (
            (cx0 * cy0 * f[ix, iy])
            + (cx0 * cy1 * f[ix, iy + 1])
            + (cx1 * cy0 * f[ix + 1, iy])
            + (cx1 * cy1 * f[ix + 1, iy + 1])
        )


@numba.njit(nogil=True, parallel=False)
def interpolate_2d_PIL(pts_x, pts_y, pts_res, a, b, h, f):
    """ In place filling of pts_res array
    """

    m = pts_res.shape[0]
    max_ix = f.shape[1] - 2
    max_iy = f.shape[0] - 2


    for mi in range(m):
        x_out = pts_x[mi]
        y_out = pts_y[mi]
    
        if CHECK_BOUNDS:
            x_out = min(max(x_out, a[0]), b[0])
            y_out =  min(max(y_out, a[1]), b[1])

        ix_float, ratx = divmod(x_out - a[0], h[0])
        iy_float, raty = divmod(b[1] - y_out, h[1])
        ix_float = min(ix_float, max_ix)
        iy_float = min(iy_float, max_iy)

        ix = np.intp(ix_float)
        iy = np.intp(iy_float)
        ratx /= h[0]
        raty /= h[1]
    
        cx0 = 1. - ratx
        cx1 = ratx
        cy0 = 1. - raty
        cy1 = raty

        pts_res[mi] = (
            (cy1 * cx0 * f[iy + 1, ix])
            + (cy1 * cx1 * f[iy + 1, ix + 1])
            + (cy0 * cx0 * f[iy, ix])
            + (cy0 * cx1 * f[iy, ix + 1])
        )
