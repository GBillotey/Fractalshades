# -*- coding: utf-8 -*-
import numpy as np
import numba
"""
This module implements a few 2d resampling tasks:
    - integer-factor decimation
    - integer-factor  supersampling

Reference : Filters for Common Resampling Tasks - Ken Turkowski, Apple Computer
"""

def lanczos_decimation_kernel(a: int, decimation: int):
    """
    Returns Lanczos-a 1d kernel coefficients for a decimation filter.

    Parameters
    ----------
    a: int
        The number of lobes for the filter (usually, 2 or 3) - Lanczos support
    decimation: int
        The decimation factor

    Return
    ------
    Lanczos kernel coeff in float64 format
    """
    dec = decimation

    def lanczos_filter(x):
        # Note: np.sinc(x) is np.sin(np.pi * x) / (np.pi * x) for numpy
        return np.where(
            np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.
        )

    if dec % 2 == 0: # We use phase = 0.5
        k = lanczos_filter((np.arange(-a * dec, a * dec) + 0.5) / dec)
    else: # We use phase = 0
        k = lanczos_filter(np.arange(-a * dec + 1, a * dec) / dec)
    return k # Not normallized, as we normalize later.


class Lanczos_decimator:
    
    def __init__(self):
        """ Class for 2d-arrays decimation with Lanczos filters """
        self.numba_cache = {}

    def get_impl(self, a: int, decimation: int):
        """
        Returns a numba implementation for Lanczos-a decimation function
        (usually a = 2 or 3), where decimation factor is an integer
        
        Parameters
        ----------
        a: int
            The number of lobes for Lanczos kernel
        decimation: int
            the decimation coefficent (will keep one pixel for each set of
            ``decimate`` pixels)
        """
        if (a, decimation) in self.numba_cache.keys():
            return self.numba_cache[(a, decimation)]

        coeffs = lanczos_decimation_kernel(a, decimation)
        nc = coeffs.shape[0]
        n_border = (nc - decimation) // 2

        coeffs_2d = np.outer(coeffs, coeffs)
        coeffs_2d = coeffs_2d / np.sum(coeffs_2d)

        @numba.njit(fastmath=True, nogil=True)
        def decimate_impl(padded, nx, ny, out):

            assert nx % decimation == 0
            assert ny % decimation == 0
            tmp = np.zeros(out.shape, dtype=np.float64)

            for ix in range(nx // decimation):
                for iy in range(ny // decimation):
                    for cx in range(nc):
                        for cy in range(nc):
                            iix = ix * decimation + cx
                            iiy = iy * decimation + cy
                            tmp[ix, iy] += coeffs_2d[cx, cy] * padded[iix, iiy]
            out[:] = tmp

        def np_impl(arr):
            # Implementation for a 2d array
            nx, ny = arr.shape
            padded = np.pad(arr, n_border, mode="edge")
            ret = np.zeros(
                (nx // decimation, ny // decimation), dtype=arr.dtype
            )
            decimate_impl(padded, nx, ny, out=ret)
            return ret

        self.numba_cache[(a, decimation)] = np_impl
        return np_impl


def lanczos_upsampling_kernel(a: int, upsampling: int):
    """
    Returns Lanczos-a 1d kernel coefficients for an upsampling filter

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
    #  Note: len is 2 * a * usp - 1 / We split in usp * (2 * a)
    pts = np.arange(-a * usp, a * usp)
    return lanczos_filter(pts / usp)  # shape (usp, 2 * a)


class Lanczos_upsampler:
    
    def __init__(self):
        """ Class for 2d-arrays decimation with Lanczos filters """
        self.numba_cache = {}

    def get_impl(self, a: int, upsampling: int):
        """
        Returns a numba implementation for Lanczos-a upsampling function
        (a lanczos support, usually a = 2 or 3), where decimation factor is an
        integer.
        The purpose is to built a fast surface-response linear interpolator

        Parameters
        ----------
        a: int
            The number of lobes for Lanczos kernel
        upsampling: int
            the upsampling factor (the final array will have ``decimate``
            pixels in both direction for each pixel of the original image)
            note: except the last pixel which is kept as-is
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

        @numba.njit(fastmath=True, nogil=True)
        def upsampling_impl(padded, nx, ny, out):

            tmp = np.zeros(out.shape, dtype=np.float64)

            for ix in range((nx - 1) * usp + 1):
                intx, iphase_x = divmod(ix, usp)

                for iy in range((ny - 1) * usp + 1):
                    inty, iphase_y = divmod(iy, usp)

                    for cx in range(nc):
                        for cy in range(nc):
                            kx = cx * usp + iphase_x
                            ky = cy * usp + iphase_y
                            iix = intx - cx
                            iiy = inty - cy
                            tmp[ix, iy] += (
                                coeffs_2d[kx, ky] * padded[iix, iiy]
                            )

            out[:] = tmp # casting

        def np_impl(arr):
            # Implementation for a 2d array
            nx, ny = arr.shape
            padded = np.pad(arr, n_border, mode="edge")
            ret = np.zeros(
                ((nx - 1) * usp + 1, (ny - 1) * usp + 1), dtype=arr.dtype
            )
            upsampling_impl(padded, nx, ny, out=ret)
            return ret

        self.numba_cache[(a, upsampling)] = np_impl
        return np_impl



if __name__ == "__main__":
    print("Coeffs:\n", lanczos_upsampling_kernel(2, 2))

    
