# -*- coding: utf-8 -*-
import sys
import numpy as np
import time
import unittest
import numba
import operator

from fractalshades.numpy_utils.xrange import Xrange_array
import fractalshades.numpy_utils.numba_xr as numba_xr


from_complex = {np.complex64: np.float32,
                np.complex128: np.float64}
crossed_dtypes = [
    (np.float64, np.float64),
    (np.complex128, np.complex128),
    (np.float64, np.complex128),
    (np.complex128, np.float64)
]

# testing binary operation of reals extended arrays
def generate_random_xr(dtype, nvec=500, max_bin_exp=200, seed=100):
    """
    Generates a random Xrange array
    dtype: mantissa dtype
    nvec: number of pts
    max_bin_exp max of base 2 exponent abs
    seed : random seed
    
    Return
    Xrange array, standard array
    """
    rg = np.random.default_rng(seed)
    if dtype in from_complex.keys():
        r_dtype = from_complex[dtype]
        mantissa = ((rg.random([nvec], dtype=r_dtype) * 2. - 1.)
                    + 1j * (2. * rg.random([nvec], dtype=r_dtype) - 1.))
    else:
        mantissa = rg.random([nvec], dtype=dtype) * 2. - 1.

    exp = rg.integers(low=-max_bin_exp, high=max_bin_exp, size=[nvec])

    xr = Xrange_array(mantissa, exp=exp)
    std = mantissa.copy() * (2. ** exp)
    return xr, std

def _matching(res, expected, almost=False, dtype=None, cmp_op=False, ktol=1.5):
    if not cmp_op:
        res = res.to_standard()
    if almost:
        np.testing.assert_allclose(res, expected,
                                   rtol= ktol * np.finfo(dtype).eps)
    else:
        np.testing.assert_array_equal(res, expected)

@numba.njit
def numba_test_setitem(arr, idx, val_tuple):
    arr[idx] = numba_xr.Xrange_scalar(*val_tuple)

@numba.njit
def numba_test_add(a, b, out):
    n, = a.shape
    for i in range(n):
        out[i] = a[i] + b[i]

@numba.njit
def numba_test_iadd(a, b):
    n, = a.shape
    for i in range(n):
        a[i] += b[i]
    return a

@numba.njit
def numba_test_sub(a, b, out):
    n, = a.shape
    for i in range(n):
        out[i] = a[i] - b[i]

@numba.njit
def numba_test_mul(a, b, out):
    n, = a.shape
    for i in range(n):
        out[i] = a[i] * b[i]

@numba.njit#(parallel=True)
def numba_test_div(a, b, out):
    n, = a.shape
    for i in range(n):
        out[i] = a[i] / b[i]

@numba.njit
def numba_test_ldexp(m, exp, out):
    n, = m.shape
    for i in range(n):
        out[i] = numba_xr._exp2_shift(m[i], exp[i])

@numba.njit
def numba_test_frexp(m, out_m, out_exp):
    n, = m.shape
    for i in range(n):
        out_m[i], out_exp[i] = numba_xr._frexp(m[i])

@numba.njit
def numba_test_normalize(m, exp, out_m, out_exp):
    n, = m.shape
    for i in range(n):
        out_m[i], out_exp[i] = numba_xr._normalize(m[i], exp[i])
        

@numba.njit
def numba_to_standard(xa, out_std):
    n, = xa.shape
    for i in range(n):
        out_std[i] = numba_xr.to_standard(xa[i])

@numba.njit
def numba_test_sqrt(xa, out):
    n, = xa.shape
    for i in range(n):
        out[i] = np.sqrt(xa[i])

@numba.njit
def numba_test_power(xa, power_exp, out):
    n, = xa.shape
    for i in range(n):
        out[i] = np.power(xa[i], power_exp)


@numba.njit
def numba_test_abs(xa, out):
    n, = xa.shape
    for i in range(n):
        out[i] = np.abs(xa[i])

@numba.njit
def numba_test_abs2(xa, out):
    n, = xa.shape
    for i in range(n):
        out[i] = numba_xr.extended_abs2(xa[i])

@numba.njit
def numba_geom_mean(vec):
    return numba_xr.geom_mean(vec[0], vec[1])

@numba.njit
def numba_test_constant():
    zero = numba_xr.zero()
    one = numba_xr.one()
    ok = (zero == 0.) and (one == 1.)
    return ok

@numba.njit
def numba_toXr_conversion():
    # Conversion of a float64
    real = 1.5e218
    real_xr = numba_xr.to_Xrange_scalar(real)
    real_xr = np.sqrt(real_xr * real_xr)
    ok = (real_xr == real)
    # Conversion of a complex128
    z = (1.5 + 0.5j) * 1.e218 # Note : random might fail equality test
    z_xr = numba_xr.to_Xrange_scalar(z)
    z_xr = np.sqrt(z_xr * z_xr)
    ok = ok and (z_xr == z)
    return ok

@numba.njit
def numba_tostandard_conversion():
    # Conversion of a float64
    real = 1.5e-155 #218
    real_xr = numba_xr.to_Xrange_scalar(real)
    real_expected = numba_xr.to_standard(real_xr)
    ok = (real == real_expected)

    real_xr_sq = real_xr * real_xr
    real_sq = numba_xr.to_standard(real_xr_sq)
    ok = ok and (real_sq == real ** 2)

    real_xr_sqsq = real_xr_sq * real_xr_sq
    real_sqsq = numba_xr.to_standard(real_xr_sqsq)
    ok = ok and (real_sqsq == real ** 4)

    real2 = 1.5e150
    real2_xr = numba_xr.to_Xrange_scalar(real2)
    real2_expected = numba_xr.to_standard(real2_xr)
    ok = ok and (real2 == real2_expected)

    real2_xr_sq = real2_xr * real2_xr
    real2_sq = numba_xr.to_standard(real2_xr_sq)
    ok = ok and (real2_sq == real2 ** 2)

    return ok

@numba.njit
def numba_unbox_xr_scalar(mantissa, exp):
    xr = numba_xr.Xrange_scalar(mantissa, exp)
    return xr

@numba.njit
def numba_test_add_smallstd(a, b):
    a_xr = numba_xr.to_Xrange_scalar(a)
    return  b + a_xr
@numba.njit
def numba_test_add_smallstd2(a, b):
    a_xr = numba_xr.to_Xrange_scalar(a)
    return  a_xr + b
@numba.njit
def numba_test_sub_smallstd(a, b):
    a_xr = numba_xr.to_Xrange_scalar(a)
    return  b - a_xr
@numba.njit
def numba_test_sub_smallstd2(a, b):
    a_xr = numba_xr.to_Xrange_scalar(a)
    return  a_xr - b

class Test_numba_xr(unittest.TestCase):

    def test_setitem(self):
        for dtype in [np.float64, np.complex128]:
            with self.subTest(dtype=dtype):
                nvec = 500
                xr, std = generate_random_xr(dtype, nvec=nvec)
                xr2 = Xrange_array.zeros(xr.shape, dtype)
                for i in range(nvec):
                    val_tuple = (xr._mantissa[i], xr._exp[i])
                    numba_test_setitem(xr2, i, val_tuple)
                _matching(xr2, std)

    def test_ldexp(self):
        dtype = np.float64
        nvec = 5000
        xr, std = generate_random_xr(dtype, nvec=nvec, max_bin_exp=200)
        exp = np.asarray(xr["exp"])
        out = np.empty(std.shape, dtype)
        numba_test_ldexp(std, exp, out)
        np.testing.assert_array_equal(out, np.ldexp(std, exp))
        numba_test_ldexp(std, -exp, out)
        np.testing.assert_array_equal(out, np.ldexp(std, -exp))

    def test_frexp(self):
        dtype = np.float64
        nvec = 5000
        xr, std = generate_random_xr(dtype, nvec=nvec, max_bin_exp=200)
        outm = np.empty(std.shape, dtype)
        outexp = np.empty(std.shape, np.int32)
        numba_test_frexp(std, outm, outexp)
        np.testing.assert_array_equal(std, outm * (2. ** outexp))

    def test_normalize(self):
        for dtype in (np.float64, np.complex128): #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtype=dtype):
                nvec = 5000
                xr, std = generate_random_xr(dtype, nvec=nvec, max_bin_exp=200)
        #        exp = np.asarray(xr["exp"])
                outm = np.empty(xr.shape, dtype)
                outexp = np.empty(xr.shape, np.int32)
                numba_test_normalize(xr["mantissa"], xr["exp"], outm, outexp)
                np.testing.assert_array_equal(std, outm * (2. ** outexp))
                # print("outm", outm)

    def test_to_standard(self):
        for dtype in (np.float64, np.complex128): #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtype=dtype):
                nvec = 5000
                xr, std = generate_random_xr(dtype, nvec=nvec, max_bin_exp=800)
                expected = np.empty_like(std)
                numba_to_standard(xr, expected)
                np.testing.assert_array_equal(std, expected)
                # print("expected", expected)

    def test_add(self):
        for (dtypea, dtypeb) in crossed_dtypes:# [(np.float64, np.float64), np.complex128]: #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 10000
                xa, stda = generate_random_xr(dtypea, nvec=nvec)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=800)
                res = Xrange_array.empty(xa.shape,
                    dtype=np.result_type(dtypea, dtypeb))
                expected = stda + stdb
                
                numba_test_add(xa, xb, res)
                # Numba timing without compilation
                t_numba = - time.time()
                numba_test_add(xa, xb, res)
                t_numba += time.time()
                # numpy timing 
                t_np = - time.time()
                res_np = xa + xb
                t_np += time.time()
                
                _matching(res, expected)
                _matching(res_np, expected)
                
                # Test add a scalar
                numba_test_add(xa, stdb, res)
                _matching(res, expected)
                numba_test_add(stda, xb, res)
                _matching(res, expected)


    def test_sub(self):
        for (dtypea, dtypeb) in crossed_dtypes: #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 10000
                xa, stda = generate_random_xr(dtypea, nvec=nvec)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=800)
                res = Xrange_array.empty(xa.shape, 
                    dtype=np.result_type(dtypea, dtypeb))
                expected = stda - stdb
                numba_test_sub(xa, xb, res)
                # Numba timing without compilation
                t_numba = - time.time()
                numba_test_sub(xa, xb, res)
                t_numba += time.time()
                # numpy timing 
                t_np = - time.time()
                res_np = xa - xb
                t_np += time.time()
                
                _matching(res, expected)
                _matching(res_np, expected)
                
                # Test substract a scalar
                numba_test_sub(xa, stdb, res)
                _matching(res, expected)
                numba_test_sub(stda, xb, res)
                _matching(res, expected)
    
    def test_mul(self):
        for (dtypea, dtypeb) in crossed_dtypes: #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100000
                xa, stda = generate_random_xr(dtypea, nvec=nvec,
                                              max_bin_exp=75)
                # Adjust the mantissa to be sure to trigger a renorm for some
                # (around 30 %) cases
                xa = np.asarray(xa)
                exp = np.copy(xa["exp"])
                xa["mantissa"] *= 2.**(2 * exp)
                xa["exp"] = -exp
                xa = xa.view(Xrange_array)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=7800)
                res = Xrange_array.empty(xa.shape,
                    dtype=np.result_type(dtypea, dtypeb))
                expected = stda * stdb
                numba_test_mul(xa, xb, res)
                # Numba timing without compilation
                t_numba = - time.time()
                numba_test_mul(xa, xb, res)
                t_numba += time.time()
                # numpy timing 
                t_np = - time.time()
                res_np = xa * xb
                t_np += time.time()
                
                _matching(res, expected)
                _matching(res_np, expected)

                # Test multiply by a scalar
                numba_test_mul(xa, stdb, res)
                _matching(res, expected)
                numba_test_mul(stda, xb, res)
                _matching(res, expected)


    def test_div(self):
        for (dtypea, dtypeb) in crossed_dtypes: #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100000
                xa, stda = generate_random_xr(dtypea, nvec=nvec,
                                              max_bin_exp=75)
                # Adjust the mantissa to be sure to trigger a renorm for some
                # (around 30 %) cases
                xa = np.asarray(xa)
                exp = np.copy(xa["exp"])
                xa["mantissa"] *= 2.**(2 * exp)
                xa["exp"] = -exp
                xa = xa.view(Xrange_array)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=7800)
                res = Xrange_array.empty(xa.shape, 
                    dtype=np.result_type(dtypea, dtypeb))
                expected = stda / stdb

                numba_test_div(xa, xb, res)
                # Numba timing without compilation
                t_numba = - time.time()
                numba_test_div(xa, xb, res)
                t_numba += time.time()
                # numpy timing 
                t_np = - time.time()
                res_np = xa / xb
                t_np += time.time()

                _matching(res, expected, almost=True, dtype=np.float64,
                          ktol=2.)
                _matching(res_np, expected, almost=True, dtype=np.float64,
                          ktol=2.)
                
                # Test divide by a scalar
                numba_test_div(xa, stdb, res)
                _matching(res, expected, almost=True, dtype=np.float64,
                          ktol=2.)
                numba_test_div(stda, xb, res)
                _matching(res, expected, almost=True, dtype=np.float64,
                          ktol=2.)
        
    def test_compare(self):
        for (dtypea, dtypeb) in crossed_dtypes: #, np.complex128]: # np.complex64 np.float32
            for compare_operator in (
                    operator.lt,
                    operator.le,
                    operator.eq,
                    operator.ne,
                    operator.ge,
                    operator.gt
                    ):
                with self.subTest(dtypea=dtypea, dtypeb=dtypeb, 
                                  operator=compare_operator):
                    # Only equality test with complex...
                    if not(compare_operator in (operator.ne, operator.eq)):
                        if ((dtypea == np.complex128)
                            or (dtypeb == np.complex128)):
                            continue

                    print(compare_operator, dtypea, dtypeb)
                    
                    nvec = 10000
                    xa, stda = generate_random_xr(
                        dtypea, nvec=nvec, max_bin_exp=75)
                    xb, stdb = generate_random_xr(
                        dtypeb, nvec=nvec, max_bin_exp=75)
                    
                    # Modify to allow precise 
                    if (dtypea == np.complex128) and (dtypeb == np.float64):
                        stdb[:3000] = stda[:3000].real
                        xb[:3000] = xa[:3000].real
                    else:
                        stdb[:3000] = stda[:3000]
                        xb[:3000] = xa[:3000]
                    xb[1000:2000] *= (1. + np.finfo(dtypeb).eps)
                    xb[2000:3000] *= (1. - np.finfo(dtypeb).eps)
                    stdb[1000:2000] *= (1. + np.finfo(dtypeb).eps)
                    stdb[2000:3000] *= (1. - np.finfo(dtypeb).eps)

                    expected = compare_operator(stda, stdb)
                    res = np.empty_like(expected)

                    t_np = - time.time()
                    res_np = compare_operator(xa, xb)
                    t_np += time.time()
                    
                    @numba.njit
                    def numba_cmp(xa, xb, out):
                        n, = xa.shape
                        for i in range(n):
                            out[i] = compare_operator(xa[i], xb[i])
                    
                    numba_cmp
                    numba_cmp(xa, xb, res)
                    t_numba = - time.time()
                    numba_cmp(xa, xb, res)
                    t_numba += time.time()

                    np.testing.assert_array_equal(res_np, expected)
                    np.testing.assert_array_equal(res, expected)
                    
                    # Test compare with a scalar
                    numba_cmp(xa, stdb, res)
                    np.testing.assert_array_equal(res, expected)
                    numba_cmp(stda, xb, res)
                    np.testing.assert_array_equal(res, expected)

    def test_sqrt(self):
        for dtype in (np.float64, np.complex128): # np.complex64 np.float32
            with self.subTest(dtype=dtype):
                nvec = 10000
                xa, stda = generate_random_xr(dtype, nvec=nvec, max_bin_exp=75)
                # sqrt not defined for negative reals
                if dtype == np.float64:
                    xa = np.abs(xa)
                    stda = np.abs(stda)
                # Adjust the mantissa to be sure to trigger a renorm for some
                # (around 30 %) cases
                xa = np.asarray(xa)
                exp = np.copy(xa["exp"])
                xa["mantissa"] *= 2.**(2 * exp)
                xa["exp"] = -exp
                xa = xa.view(Xrange_array)
                res = Xrange_array.empty(xa.shape, dtype=dtype)
                expected = np.sqrt(stda)

                numba_test_sqrt(xa, res)
                # Numba timing without compilation
                t_numba = - time.time()
                numba_test_sqrt(xa, res)
                t_numba += time.time()
                # numpy timing 
                t_np = - time.time()
                res_np = np.sqrt(xa)
                t_np += time.time()
                
                _matching(res, expected, almost=True, dtype=np.float64,
                          ktol=2.)
                _matching(res_np, expected, almost=True, dtype=np.float64,
                          ktol=2.)

    def test_power(self):
        # Testing integer path
        for power_exp in range(10): 
            for dtype in (np.float64, np.complex128): # np.complex64 np.float32
                k_tol = 2.0
                if dtype == np.complex128:
                    k_tol = 20.
                
                with self.subTest(power_exp=power_exp, dtype=dtype):
                    nvec = 1000
                    xa, stda = generate_random_xr(dtype, nvec=nvec,
                                                  max_bin_exp=15)

                    xa = xa.view(Xrange_array)
                    expected = np.power(stda, power_exp)
                    res = Xrange_array.empty(xa.shape, dtype=dtype)
                    res_np = np.power(xa, power_exp)
                    numba_test_power(xa, power_exp, res)
                    _matching(res, expected, almost=True, dtype=np.float64,
                              ktol=k_tol)
                    _matching(res_np, expected, almost=True, dtype=np.float64,
                              ktol=2.)

        # Testing float path
        for power_exp in (0.5, 0.122522, 1.554, 4.8962): #  -2.3): 
            for dtype in (np.float64, np.complex128): # np.complex64 np.float32
                k_tol = 70.
                if dtype == np.complex128:
                    k_tol = 70.
                
                with self.subTest(power_exp=power_exp, dtype=dtype):
                    nvec = 1000
                    xa, stda = generate_random_xr(
                            dtype, nvec=nvec, max_bin_exp=15)
                    
                    # fractionnal power not defined for negative reals
                    if dtype == np.float64:
                        xa = np.abs(xa)
                        stda = np.abs(stda)

                    xa = xa.view(Xrange_array)
                    expected = np.power(stda, power_exp)
                    res = Xrange_array.empty(xa.shape, dtype=dtype)
                    res_np = np.power(xa, power_exp)

                    numba_test_power(xa, power_exp, res)
                    _matching(res, expected, almost=True, dtype=np.float64,
                              ktol=k_tol)
                    _matching(res_np, expected, almost=True, dtype=np.float64,
                              ktol=k_tol)


                
    def test_abs(self):
        for dtype in (np.float64, np.complex128): # np.complex64 np.float32
            with self.subTest(dtype=dtype):
                nvec = 10000
                xa, stda = generate_random_xr(dtype, nvec=nvec, max_bin_exp=75)
                # Adjust the mantissa to be sure to trigger a renorm for some
                # (around 30 %) cases
                xa = np.asarray(xa)
                exp = np.copy(xa["exp"])
                xa["mantissa"] *= 2.**(2 * exp)
                xa["exp"] = -exp
                xa = xa.view(Xrange_array)
                res = Xrange_array.empty(xa.shape, dtype=dtype)
                expected = np.abs(stda)

                numba_test_abs(xa, res)
                # Numba timing without compilation
                t_numba = - time.time()
                numba_test_abs(xa, res)
                t_numba += time.time()
                # numpy timing 
                t_np = - time.time()
                res_np = np.abs(xa)
                t_np += time.time()
                
                _matching(res, expected, almost=True, dtype=np.float64,
                          ktol=4.)
                _matching(res_np, expected, almost=True, dtype=np.float64,
                          ktol=4.)


    def test_abs2(self):
        for dtype in (np.float64, np.complex128): # np.complex64 np.float32
            with self.subTest(dtype=dtype):
                nvec = 10000
                xa, stda = generate_random_xr(dtype, nvec=nvec, max_bin_exp=75)
                # Adjust the mantissa to be sure to trigger a renorm for some
                # (around 30 %) cases
                xa = np.asarray(xa)
                exp = np.copy(xa["exp"])
                xa["mantissa"] *= 2.**(2 * exp)
                xa["exp"] = -exp
                xa = xa.view(Xrange_array)
                res = Xrange_array.empty(xa.shape, dtype=dtype)
                expected = np.abs(stda) ** 2

                numba_test_abs(xa, res)
                # Numba timing without compilation
                t_numba = - time.time()
                numba_test_abs2(xa, res)
                t_numba += time.time()
                
                _matching(res, expected, almost=True, dtype=np.float64,
                          ktol=4.)

    def test_geom_mean(self):
        dtype = np.float64
        nvec = 10
        xa, stda = generate_random_xr(dtype, nvec=nvec * 2,
                                      max_bin_exp=75)
        xa = np.abs(xa).reshape([nvec, 2])
        stda = np.abs(stda).reshape([nvec, 2])

        for ivec in range(nvec):
            x_loc = xa[ivec,:]
            std_loc = stda[ivec,:]
            expected = np.sqrt(std_loc[0] *std_loc[1])
            res = numba_geom_mean(x_loc)
            _matching(res, expected, almost=True, dtype=np.float64,
                          ktol=4.)


    def test_expr(self):
        for (dtypea, dtypeb) in crossed_dtypes: #, np.complex128]: # np.complex64 np.float32

            dtype_res = np.result_type(dtypea, dtypeb)
            nvec = 10000
            xa, stda = generate_random_xr(dtypea, nvec=nvec)# , max_bin_exp=250)
            xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=800)
            res = Xrange_array.empty(xa.shape, dtype=dtype_res)
            
            
            def get_numba_expr(case):
                if case == 0:
                    def numba_expr(xa, xb, out):
                        n, = xa.shape
                        for i in range(n):
                            out[i] = xa[i] * xb[i] * xa[i]
                elif case == 1:
                    def numba_expr(xa, xb, out):
                        n, = xa.shape
                        for i in range(n):
                            out[i] = xa[i] * xb[i] + xa[i] - 7.8
                elif case == 2:
                    def numba_expr(xa, xb, out):
                        n, = xa.shape
                        for i in range(n):
                            out[i] = (xb[i] * 2.) * (xa[i] + xa[i] * xb[i])   + (xa[i] * xb[i] - 7.8 + xb[i])
                elif case == 3:
                    def numba_expr(xa, xb, out):
                        n, = xa.shape
                        for i in range(n):
                            out[i] = ((xb[i] * 2.) * (xa[i] + np.abs(xa[i] * xb[i]) + 1.)
                                      + (xa[i] * np.sqrt(np.abs(xb[i]) + 7.8) + xb[i]))
                else:
                    raise ValueError(case)
                return numba.njit(numba_expr)

            def get_std_expr(case):
                if case == 0:
                    def std_expr(xa, xb):
                        return xa * xb * xa
                elif case == 1:
                    def std_expr(xa, xb):
                        return xa * xb + xa - 7.8
                elif case == 2:
                    def std_expr(xa, xb):
                        return (xb * 2.) * (xa + xa * xb)  + (xa * xb - 7.8 + xb)
                elif case == 3:
                    def std_expr(xa, xb):
                        return ((xb * 2.) * (xa + np.abs(xa * xb) + 1.)
                                + (xa * np.sqrt(np.abs(xb) + 7.8) + xb))
                else:
                    raise ValueError(case)
                return std_expr
            
            n_case = 4
            for case in range(n_case):
                    
                with self.subTest(dtypea=dtypea, dtypeb=dtypeb, expr=case):
                    expected = get_std_expr(case)(stda, stdb)
            
                    # numpy timing 
                    t_np = - time.time()
                    res_np = get_std_expr(case)(xa, xb)
                    t_np += time.time()
                    
                    numba_expr = get_numba_expr(case)
                    numba_expr(xa, xb, res)
                    # Numba timing without compilation
                    t_numba = - time.time()
                    numba_expr(xa, xb, res)
                    t_numba += time.time()
            
                    _matching(res, expected, almost=True, dtype=np.float64,
                              ktol=4.)
                    _matching(res_np, expected, almost=True, dtype=np.float64,
                              ktol=4.)
                    

    def test_constant(self):
        ok = numba_test_constant()
        self.assertTrue(ok, msg="Numba constant test failed")

    def test_conversion(self):
        ok = numba_toXr_conversion()
        self.assertTrue(ok, msg="Numba conversion to Xr test failed")
        
        if sys.platform == "linux":
            ok = numba_tostandard_conversion()
            self.assertTrue(ok, msg="Numba conversion to standard test failed")
    
    def test_unbox_xr_scalar(self):
        res = numba_unbox_xr_scalar(1.0, np.int32(10))
        assert res == 1024.
        res = numba_unbox_xr_scalar(1.0j, np.int32(10))
        assert res == 1024.j
    
    def test_adding_smallstd(self):
        a = 1.e-200 * (2. + 1j)
        b = 2.e-200 * (2. + 1j)
        
        c = numba_test_add_smallstd(a, b)
        assert np.abs(c._mantissa) < 10.
        assert c == a + b

        c = numba_test_add_smallstd2(a, b)
        assert np.abs(c._mantissa) < 10.
        assert c == a + b

        c = numba_test_sub_smallstd(a, b)
        assert np.abs(c._mantissa) < 10.
        assert c == b - a

        c = numba_test_sub_smallstd2(a, b)
        assert np.abs(c._mantissa) < 10.
        assert c == a - b

        


if __name__ == "__main__":
    import test_config
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([
            Test_numba_xr,
        ]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_numba_xr("test_power"))
        runner.run(suite)












