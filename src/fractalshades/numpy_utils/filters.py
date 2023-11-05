# -*- coding: utf-8 -*-
import numpy as np
import numba
from numba.extending import overload
"""
This module implements a few 2d resampling tasks:
    - integer-factor decimation
    - integer-factor supersampling

Reference : Filters for Common Resampling Tasks - Ken Turkowski, Apple Computer
"""

def lanczos_decimation_kernel(a: int, decimation: int, phase: float=None):
    """
    Returns Lanczos-a 1d kernel coefficients for a decimation filter.

    Parameters
    ----------
    a: int
        The number of lobes for the filter (usually, 2 or 3) - Lanczos support
    decimation: int
        The decimation factor
    phase: float, Optional
        The phase for the kernel (if provided, shall be 0.0 of 0.5). If None,
        will default to 0.0 or 0.5 depending on ``decimation`` parity.

    Return
    ------
    Lanczos kernel coeff in float64 format
    """
    dec = decimation

    if phase is None:
        phase = 0.0 if (dec % 2) else 0.5

    def lanczos_filter(x):
        # Note: np.sinc(x) is np.sin(np.pi * x) / (np.pi * x) for numpy
        return np.where(
            np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.
        )

    if phase == 0.5: # We use phase = 0.5
        k = lanczos_filter((np.arange(-a * dec, a * dec) + 0.5) / dec)
    else: # We use phase = 0
        k = lanczos_filter(np.arange(-a * dec + 1., a * dec) / dec)
    return k # Not normallized, as we normalize later.


class Lanczos_decimator:
    
    def __init__(self):
        """ Class for 2d-arrays decimation with Lanczos filters """
        self.numba_cache = {}

    def get_impl(self, a: int, decimation: int):
        """
        Return a numba implementation for Lanczos-a decimation function
        (usually a = 2 or 3), where decimation factor is an integer.

        Usage, where ``arr`` is a 2d numpy array:

        .. code-block:: python

            decimator = Lanczos_decimator().get_impl(a=2, decimation=3)
            decimated_arr = decimator(arr)

        If one ``arr`` axis size is not a multiple of ``decimation``, it will
        be border-padded first (on the axis higher end) up to the next
        multiple, so that the final image size is snapped to:
        (ceil(nx / decimation), ceil(ny / decimation))
        Note that the latter is a fall-back case which is normally not used by
        the programm.

        Parameters
        ----------
        a: int
            The number of lobes for Lanczos kernel
        decimation: int
            the decimation coefficient (will keep one pixel for each set of
            ``decimate`` pixels)
        """
        cache_key = (a, decimation)
        if cache_key in self.numba_cache.keys():
            return self.numba_cache[cache_key]

        coeffs = lanczos_decimation_kernel(a, decimation)
        nc = coeffs.shape[0]
        n_border = (nc - decimation) // 2
        coeffs_2d = np.outer(coeffs, coeffs)
        coeffs_2d = coeffs_2d / np.sum(coeffs_2d)

        def np_impl(arr):
            # Implementation for a 2d array
            nx, ny = arr.shape
            dnx = (decimation - nx % decimation) % decimation
            dny = (decimation - ny % decimation) % decimation
            nx += dnx
            ny += dny
            pad_width = (
                (n_border, n_border + dnx), (n_border, n_border + dny)
            )
            padded = np.pad(arr, pad_width, mode="edge")
            out = np.zeros(
                (nx // decimation, ny // decimation),
                dtype=arr.dtype
            )
            return decimate_impl(
                decimation, nc, coeffs_2d, padded, out
            )

        self.numba_cache[cache_key] = np_impl
        return np_impl


    def get_masked_impl(self, a: int, decimation: int):
        """
        Return a numba implementation for Lanczos-a decimation function
        (usually a = 2 or 3), where decimation factor is an integer.

        Usage, where ``arr`` and ``mask_arr`` are 2d numpy array:

        .. code-block:: python

            decimator = Lanczos_decimator().get_impl(a=2, decimation=3)
            decimated_arr = decimator(arr, mask_arr)

        If one ``arr`` axis size is not a multiple of ``decimation``, it will
        be border-padded first (on the axis higher end) up to the next
        multiple, so that the final image size is snapped to:
        (ceil(nx / decimation), ceil(ny / decimation))
        Note that the latter is a fall-back case which is normally not used by
        the programm.

        Parameters
        ----------
        a: int
            The number of lobes for Lanczos kernel
        decimation: int
            the decimation coefficent (will keep one pixel for each set of
            ``decimate`` pixels)
        """
        cache_key = (a, decimation, True)
        if cache_key in self.numba_cache.keys():
            return self.numba_cache[cache_key]

        coeffs = lanczos_decimation_kernel(a, decimation)
        nc = coeffs.shape[0]
        n_border = (nc - decimation) // 2
        coeffs_2d = np.outer(coeffs, coeffs)
        coeffs_2d = coeffs_2d / np.sum(coeffs_2d)

        def np_impl(arr, mask_arr):
            # Implementation for a 2d array
            nx, ny = arr.shape
            dnx = (decimation - nx % decimation) % decimation
            dny = (decimation - ny % decimation) % decimation
            nx += dnx
            ny += dny
            pad_width = (
                (n_border, n_border + dnx), (n_border, n_border + dny)
            )
            padded = np.pad(arr, pad_width, mode="edge")

            pmask = np.pad(1. - mask_arr, pad_width, mode="edge")
            out = np.zeros(
                (nx // decimation, ny // decimation), dtype=arr.dtype
            )
            return decimate_masked_impl(
                decimation, nc, coeffs_2d, padded, pmask, out
            )

        self.numba_cache[cache_key] = np_impl
        return np_impl


    def get_stable_impl(self, a: int):
        """
        Return a numba implementation for Lanczos-a decimation function.

       Its main properties are as follows:
            - decimation factor is 2
            - startpoint position is kept (the filter phase is always 0.)
            - endpoint position - as returned - is >= initial endpoint
        This implementation is hence stabilizing the [startpoint, endpoint]
        interval (in the sense that the returned interval contains the initial
        one)

        Usage, where ``arr`` is a 2d numpy array:

        .. code-block:: python

            decimator = Lanczos_decimator().get_stable_impl(a=2)
            decimated_arr, k_spanx, k_spany = decimator(arr)

        The size of the returned array is: (nx // 2 + 1, ny // 2 + 1) where
        nx, ny = arr.shape
        k_spanx is the factor between initial and returned x range (>= 1.0)
        k_spany is the factor between initial and returned y range (>= 1.0)

        Parameters
        ----------
        a: int
            The number of lobes for Lanczos kernel
        
        Returns
        -------
        decimated_arr: np array
            The decimated array
        k_spanx: float (>= 1.0)
            The x-range interval growth coefficient
        k_spany: float (>= 1.0)
            The y-range interval growth coefficient
        """
        decimation = dec = 2

        cache_key = (a, "stable")
        if cache_key in self.numba_cache.keys():
            return self.numba_cache[cache_key]

        coeffs = lanczos_decimation_kernel(a, decimation=2, phase=0.0)
        nc = coeffs.shape[0]
        n_border = nc // 2  # i.e.: Lanczos 2:  3, Lanczos 3: 5
        coeffs_2d = np.outer(coeffs, coeffs)
        coeffs_2d = coeffs_2d / np.sum(coeffs_2d)

        def np_impl(arr):
            # Implementation for a 2d array
            nx, ny = arr.shape
            out_nx = nx // dec + 1
            out_ny = ny // dec + 1
            dnx = ((out_nx - 1) * dec + 1) - nx
            dny = ((out_ny - 1) * dec + 1) - ny

            k_spanx = (out_nx - 1.) * dec / (nx - 1.)
            k_spany = (out_ny - 1.) * dec / (ny - 1.)

            # nx = 15 -> outx = 8 -> dnx = 0 -> k_spanx = 1.0
            # nx = 16 -> outx = 9 -> dnx = 1 -> k_spanx = 16. / 15.

            out = np.zeros((out_nx, out_ny), dtype=arr.dtype)
            pad_width = (
                (n_border, n_border + dnx), (n_border, n_border + dny)
            )
            padded = np.pad(arr, pad_width, mode="edge")

            return (
                decimate_impl(decimation, nc, coeffs_2d, padded, out),
                k_spanx, k_spany
            )

        self.numba_cache[cache_key] = np_impl
        return np_impl

def _decimate_impl(decimation, nc, coeffs_2d, padded, out):
    pass
@overload(_decimate_impl, nopython=True, fastmath=True, nogil=True, cache=True)
def ov__decimate_impl(decimation, nc, coeffs_2d, padded, out):
    """
    Implemented as a pure jitted function through `overload`
    """
    # the actual numba implementation
    if out.dtype != numba.uint8:
        def impl(decimation, nc, coeffs_2d, padded, out):
            out_nx, out_ny = out.shape
            # implementation without clipping
            for ix in range(out_nx):
                for iy in range(out_ny):
                    tmp = 0.
                    for cx in range(nc):
                        for cy in range(nc):
                            iix = ix * decimation + cx
                            iiy = iy * decimation + cy
                            tmp += coeffs_2d[cx, cy] * padded[iix, iiy]
                    out[ix, iy] = tmp
            return out
        return impl
    else:
        def impl(decimation, nc, coeffs_2d, padded, out):
            out_nx, out_ny = out.shape
            # implementation with clipping to [0 - 255]
            for ix in range(out_nx):
                for iy in range(out_ny):
                    tmp = 0.
                    for cx in range(nc):
                        for cy in range(nc):
                            iix = ix * decimation + cx
                            iiy = iy * decimation + cy
                            tmp += coeffs_2d[cx, cy] * padded[iix, iiy]
                    out[ix, iy] = max(0, min(255, tmp + 0.5))
            return out
        return impl
@numba.njit
def decimate_impl(decimation, nc, coeffs_2d, padded, out):
    # See https://github.com/numba/numba/issues/8897
    return _decimate_impl(decimation, nc, coeffs_2d, padded, out)

def _decimate_masked_impl(decimation, nc, coeffs_2d, padded, pmask, out):
    pass
@overload(
    _decimate_masked_impl, nopython=True, fastmath=True, nogil=True, cache=True
)
def ov__decimate_masked_impl(decimation, nc, coeffs_2d, padded, pmask, out):
    """
    Implemented as a pure jitted function through `overload`
    """
    # the actual numba implementation
    if out.dtype != numba.uint8:
        def impl(decimation, nc, coeffs_2d, padded, pmask, out):
            out_nx, out_ny = out.shape
            # implementation without clipping
            for ix in range(out_nx):
                for iy in range(out_ny):
                    tmp = 0.
                    f_val = 0.
                    for cx in range(nc):
                        for cy in range(nc):
                            iix = ix * decimation + cx
                            iiy = iy * decimation + cy
                            lpm = pmask[iix, iiy]
                            if lpm > 0.:
                                lval = coeffs_2d[cx, cy] * lpm
                                f_val += lval
                                tmp += lval * padded[iix, iiy]
                    if f_val == 0.: # All values masked, defaults to 0.
                        out[ix, iy] = 0. # padded[ix, iy]
                    else:
                        out[ix, iy] = tmp / f_val
            return out
        return impl
    else:
        def impl(decimation, nc, coeffs_2d, padded, pmask, out):
            out_nx, out_ny = out.shape
            # implementation with clipping to [0 - 255]
            for ix in range(out_nx):
                for iy in range(out_ny):
                    tmp = 0.
                    f_val = 0.
                    for cx in range(nc):
                        for cy in range(nc):
                            iix = ix * decimation + cx
                            iiy = iy * decimation + cy
                            lpm = pmask[iix, iiy]
                            if lpm > 0.:
                                lval = coeffs_2d[cx, cy] * lpm
                                f_val += lval
                                tmp += lval * padded[iix, iiy]
                    if f_val == 0.: # All values masked, defaults to 0
                        out[ix, iy] = 0 # padded[ix, iy]
                    else:
                        out[ix, iy] = max(0, min(255, tmp / f_val + 0.5))
            return out
        return impl
@numba.njit
def decimate_masked_impl(decimation, nc, coeffs_2d, padded, pmask, out):
    # See https://github.com/numba/numba/issues/8897
    return _decimate_masked_impl(decimation, nc, coeffs_2d, padded, pmask, out)

def lanczos_upsampling_kernel(a: int, upsampling: int):
    """
    Return Lanczos-a 1d kernel coefficients for an upsampling filter

    Parameters
    ----------
    a: int
        the number of lobes for the filter (usually, 2 or 3) - lanczos support
    upsampling: int
        the upsampling factor

    Return
    ------
    Lanczos kernel coeff in float64 format
    """
    usp = upsampling
    def lanczos_filter(x):
        # Note: np.sinc(x) is np.sin(np.pi * x) / (np.pi * x) for numpy
        return np.where(
            np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.
        )
    # Note: len is 2 * a * usp - 1 / We split in usp * (2 * a)
    pts = np.arange(-a * usp, a * usp)
    return lanczos_filter(pts / usp)  # shape (usp, 2 * a)


class Lanczos_upsampler:
    
    def __init__(self):
        """ Class for 2d-arrays upsampling with Lanczos filters """
        self.numba_cache = {}

    def get_impl(self, a: int, upsampling: int):
        """
        Returns a numba implementation for Lanczos-a upsampling function (`a`
        is the lanczos support, usually a = 2 or 3), where `upsampling` factor
        is an integer.

        Parameters
        ----------
        a: int
            The number of lobes for Lanczos kernel support
        upsampling: int
            the upsampling factor (the final array will have ``upsampling``
            pixels in both direction for each pixel of the original image,
            except the last pixel which is kept as-is
        """
        usp = upsampling
        if (a, usp) in self.numba_cache.keys():
            return self.numba_cache[(a, usp)]

        coeffs = lanczos_upsampling_kernel(a, usp)
        coeffs_2d =  np.outer(coeffs, coeffs)

        for iphase_x in range(usp):
            for iphase_y in range(usp):
                tmp = coeffs_2d[iphase_x::usp, iphase_y::usp]
                coeffs_2d[iphase_x::usp, iphase_y::usp] = tmp / np.sum(tmp)

        nc = 2 * a
        n_border = a


        def np_impl(arr):
            # Implementation for a 2d array
            nx, ny = arr.shape
            padded = np.pad(arr, n_border, mode="edge")
            out = np.zeros(
                ((nx - 1) * usp + 1, (ny - 1) * usp + 1), dtype=arr.dtype
            )
            return upsampling_impl(padded, coeffs_2d, nx, ny, usp, nc, out)

        self.numba_cache[(a, upsampling)] = np_impl
        return np_impl

def _upsampling_impl(padded, coeffs_2d, nx, ny, usp, nc, out):
    pass
@overload(
    _upsampling_impl, nopython=True, fastmath=True, nogil=True, cache=True
)
def ov__upsampling_impl(padded, coeffs_2d, nx, ny, usp, nc, out):
    """
    Implemented as a pure jitted function through `overload`
    """
    # the actual numba implementation
    if out.dtype != numba.uint8:
        def impl(padded, coeffs_2d, nx, ny, usp, nc, out):
            for ix in range((nx - 1) * usp + 1):
                intx, iphase_x = divmod(-ix, -usp)
                for iy in range((ny - 1) * usp + 1):
                    inty, iphase_y = divmod(-iy, -usp)
                    tmp = 0. # float64
                    for cx in range(nc):
                        for cy in range(nc):
                            kx = cx * usp + iphase_x
                            ky = cy * usp + iphase_y
                            iix = intx + cx
                            iiy = inty + cy
                            tmp += coeffs_2d[kx, ky] * padded[iix, iiy]
                    out[ix, iy] = tmp # casting
            return out 
        return impl
    else:
        def impl(padded, coeffs_2d, nx, ny, usp, nc, out):
            for ix in range((nx - 1) * usp + 1):
                intx, iphase_x = divmod(-ix, -usp)
                for iy in range((ny - 1) * usp + 1):
                    inty, iphase_y = divmod(-iy, -usp)
                    tmp = 0. # float64
                    for cx in range(nc):
                        for cy in range(nc):
                            kx = cx * usp + iphase_x
                            ky = cy * usp + iphase_y
                            iix = intx + cx
                            iiy = inty + cy
                            tmp += coeffs_2d[kx, ky] * padded[iix, iiy]
                    out[ix, iy] = max(0, min(255, tmp + 0.5)) # casting
            return out 
        return impl
@numba.njit
def upsampling_impl(padded, coeffs_2d, nx, ny, usp, nc, out):
    # See https://github.com/numba/numba/issues/8897
    return _upsampling_impl(padded, coeffs_2d, nx, ny, usp, nc, out)
