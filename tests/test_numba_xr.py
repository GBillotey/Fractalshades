# -*- coding: utf-8 -*-
import os
import sys
import copy
import numpy as np
import time
import unittest
import numba
import operator
        
# Allows relative imports when run locally as script
# https://docs.python-guide.org/writing/structure/
#if __name__ == "__main__":
#    sys.path.insert(0, os.path.abspath(
#            os.path.join(os.path.dirname(__file__), '..')))


#import fractalshades.numpy_utils.xrange as fsx
from fractalshades.numpy_utils.xrange import (
        Xrange_array,
        Xrange_polynomial,
        Xrange_bivar_polynomial,
        Xrange_SA,
        Xrange_bivar_SA,
        Xrange_monome,
)
import fractalshades.numpy_utils.numba_xr as numba_xr
import fractalshades.models as fsmodels


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
    print("ok", ok)
    
    real_xr_sq = real_xr * real_xr
    real_sq = numba_xr.to_standard(real_xr_sq)
    ok = ok and (real_sq == real ** 2)
    print("real_sq", real_sq)
    print("ok", ok)
    
    real_xr_sqsq = real_xr_sq * real_xr_sq
    real_sqsq = numba_xr.to_standard(real_xr_sqsq)
    ok = ok and (real_sqsq == real ** 4)
    print("real_sq", real_sq)
    print("ok", ok, real ** 4)

    real2 = 1.5e150
    real2_xr = numba_xr.to_Xrange_scalar(real2)
    real2_expected = numba_xr.to_standard(real2_xr)
    ok = ok and (real2 == real2_expected)
    print("real2", real2_expected)
    print("ok", ok, real2)

    real2_xr_sq = real2_xr * real2_xr
    real2_sq = numba_xr.to_standard(real2_xr_sq)
    ok = ok and (real2_sq == real2 ** 2)
    print("real2_sq", real2_sq) 
    print("ok", ok, real2 ** 2)
    
#    real_xr_sqsq = real_xr_sq * real_xr_sq
#    real_sqsq = numba_xr.to_standard(real_xr_sqsq)
#    print("real_sqsq", real_sqsq)
#    ok = ok and (real_sqsq == real ** 4)
#    print("ok", ok, real ** 4)

    return ok

@numba.njit
def numba_unbox_xr_scalar(mantissa, exp):
    xr = numba_xr.Xrange_scalar(mantissa, exp)
    return xr
    
#    # Conversion of a complex128
#    z = (1.5 + 0.5j) * 1.e218 # Note : random might fail equality test
#    z_xr = numba_xr.to_Xrange_scalar(z)
#    z_xr = np.sqrt(z_xr * z_xr)
#    ok = ok and (z_xr == z)
#    return ok
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
                
                print("res", res.dtype, type(res))
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
                
                print("t_numba", t_numba)
                print("t_numpy", t_np)
                expr = (t_numba <=  t_np * 1.5)
                self.assertTrue(expr, msg="Numba speed below numpy")
                
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
                
                print("t_numba", t_numba)
                print("t_numpy", t_np)
                expr = (t_numba <=  t_np * 1.5)
                self.assertTrue(expr, msg="Numba speed below numpy")
                
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

                print("t_numba, numpy", t_numba, t_np)
                expr = (t_numba <=  t_np * 1.5)
                self.assertTrue(expr, msg="Numba speed below numpy")

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
                
                print("t_numba", t_numba)
                print("t_numpy", t_np)
                expr = (t_numba <=  t_np * 1.5)
                self.assertTrue(expr, msg="Numba speed below numpy")
                
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

                    print("t_numba *** test_compare", t_numba)
                    print("t_numpy *** test_compare", t_np) # t_numba/t_np)
                    expr = (t_numba <=  t_np * 1.5)
                    self.assertTrue(expr, msg="Numba speed below numpy")
                    
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
                
                print("t_numba", t_numba)
                print("t_numpy", t_np)
                expr = (t_numba <=  t_np * 1.5)
                self.assertTrue(expr, msg="Numba speed below numpy")
                
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

                print("t_numba", t_numba)
                if t_np == 0.:
                    return
                print("t_numpy", t_np)
                expr = (t_numba <=  t_np * 1.5)
                self.assertTrue(expr, msg="Numba speed below numpy")

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
                              ktol=2.)
                    _matching(res_np, expected, almost=True, dtype=np.float64,
                              ktol=2.)
                    
                    print("t_numba", t_numba)
                    print("t_numpy", t_np)
                    expr = (t_numba <=  t_np)
                    self.assertTrue(expr, msg="Numba speed below numpy")

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


@numba.njit
def numba_test_polyneg(poly):
    return -poly

@numba.njit
def numba_test_polyadd(polya, polyb):
    return polya + polyb
@numba.njit
def numba_test_polyadd_0(polya, polyb):
    return polya + polyb[0]
@numba.njit
def numba_test_polyadd0(polya, polyb):
    return polya[0] + polyb

@numba.njit
def numba_test_polycall(poly, val):
    return poly.__call__(val)
@numba.njit
def numba_test_polycall0(poly, val):
    return poly.__call__(val[0])

@numba.njit
def numba_test_polymul(polya, polyb):
    return polya * polyb
@numba.njit
def numba_test_polymul_0(polya, polyb):
    return polya * polyb[0]
@numba.njit
def numba_test_polymul0(polya, polyb):
    return polya[0] * polyb

@numba.njit
def numba_test_expr(polya, polyb):
#    print()
#    p = polya * polyb
#    q = p + polya
#    return q 
    return polya * polyb + polya #- polyb

class Test_poly_xr(unittest.TestCase):
    
    def test_neg(self):
        for dtype in (np.float64, np.complex128):
            with self.subTest(dtype=dtype):
                nvec = 100
                xa, stda = generate_random_xr(dtype, nvec=nvec)# , max_bin_exp=250)
                _P = Xrange_polynomial(xa, cutdeg=nvec-1)
                P = np.polynomial.Polynomial(stda)
                res = numba_test_polyneg(_P)
                expected = -P
                _matching(res.coeffs, expected.coef)
                # Check that the original array has not been modified
                _matching(_P.coeffs, P.coef)
                
    def test_add(self):
        for (dtypea, dtypeb) in crossed_dtypes: #((np.float64, np.float64),):#crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=510)# , max_bin_exp=250)
                _Pa = Xrange_polynomial(xa, cutdeg=nvec-1)
                Pa = np.polynomial.Polynomial(stda)
                _Pb = Xrange_polynomial(xb, cutdeg=nvec-1)
                Pb = np.polynomial.Polynomial(stdb)
                
                res = numba_test_polyadd(_Pa, _Pb)
                expected = Pa + Pb
                _matching(res.coeffs, expected.coef)
                # Check that the original array has not been modified
                _matching(_Pa.coeffs, Pa.coef)
                _matching(_Pb.coeffs, Pb.coef)

        for (dtypea, dtypeb) in crossed_dtypes: #((np.float64, np.float64),):#crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb, kind="scalar"):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=1, seed=5)# , max_bin_exp=250)
                _Pa = Xrange_polynomial(xa, cutdeg=nvec-1)
                Pa = np.polynomial.Polynomial(stda)              
                res = numba_test_polyadd_0(_Pa, xb)
                expected = Pa + stdb[0]
                _matching(res.coeffs, expected.coef)
                res = numba_test_polyadd0(xb, _Pa)
                _matching(res.coeffs, expected.coef)
                

    def test_op_partial(self):
        a = [1., 2., 5., 8.]
        _Pa = Xrange_polynomial(a, 10)
        Pa = np.polynomial.Polynomial(a)
        b = [1., 2.]
        _Pb = Xrange_polynomial(b, 10)
        Pb = np.polynomial.Polynomial(b)
        
        with self.subTest(op="+"):
            res = [numba_test_polyadd(_Pa, _Pb),
                   numba_test_polyadd(_Pb, _Pa),
                   numba_test_polyadd(_Pa, _Pa)]
            expected = [Pa + Pb , Pb + Pa, Pa + Pa]
            for i in range(len(res)):
                _matching(res[i].coeffs, expected[i].coef)

        with self.subTest(op="*"):
            res = [numba_test_polymul(_Pa, _Pb),
                   numba_test_polymul(_Pb, _Pa),
                   numba_test_polymul(_Pa, _Pa)]
            expected = [Pa * Pb , Pb * Pa, Pa * Pa]
            for i in range(len(res)):
                _matching(res[i].coeffs, expected[i].coef)


    def test_call(self):
        for (dtypea, dtypeb) in crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=10 , max_bin_exp=3)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec , max_bin_exp=5, seed=510)
                _Pa = Xrange_polynomial(xa, cutdeg=nvec-1)
                Pa = np.polynomial.Polynomial(stda)
                # Scalar call test
                res = numba_test_polycall0(_Pa, xb).view(Xrange_array)
                expected = Pa(stdb[0])
                _matching(res, expected)
                # Array call test
                res = numba_test_polycall(_Pa, xb).view(Xrange_array)
                expected = Pa(stdb)
                _matching(res, expected)

    def test_mul(self):
        for (dtypea, dtypeb) in crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec, seed=110,
                                              max_bin_exp=25)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=510,
                                              max_bin_exp=25)
                _Pa = Xrange_polynomial(xa, cutdeg=nvec-1)
                Pa = np.polynomial.Polynomial(stda)
                _Pb = Xrange_polynomial(xb, cutdeg=nvec-1)
                Pb = np.polynomial.Polynomial(stdb)
                
                res = numba_test_polymul(_Pa, _Pb)
                expected = Pa * Pb
                _matching(res.coeffs, expected.coef[:nvec], almost=True,
                          dtype=np.float64, ktol=10.)
                # Check that the original array has not been modified
                _matching(_Pa.coeffs, Pa.coef)
                _matching(_Pb.coeffs, Pb.coef)

        for (dtypea, dtypeb) in crossed_dtypes: #((np.float64, np.float64),):#crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb, kind="scalar"):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=1, seed=510)# , max_bin_exp=250)
                _Pa = Xrange_polynomial(xa, cutdeg=nvec-1)
                Pa = np.polynomial.Polynomial(stda)              
                res = numba_test_polymul_0(_Pa, xb)
                expected = Pa * stdb[0]
                _matching(res.coeffs, expected.coef)
                res = numba_test_polymul0(xb, _Pa)
                _matching(res.coeffs, expected.coef)

    def test_expr(self):
        for (dtypea, dtypeb) in crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec, seed=110,
                                              max_bin_exp=25)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=510,
                                              max_bin_exp=25)
                _Pa = Xrange_polynomial(xa, cutdeg=nvec-1)
                Pa = np.polynomial.Polynomial(stda)
                _Pb = Xrange_polynomial(xb, cutdeg=nvec-1)
                Pb = np.polynomial.Polynomial(stdb)
                
                res = numba_test_expr(_Pa, _Pb)
                expected = Pa * Pb + Pa #- Pb
                _matching(res.coeffs, expected.coef[:nvec], almost=True,
                          dtype=np.float64, ktol=10.)
        

@numba.njit
def numba_test_sa_neg(sa):
    return -sa

@numba.njit
def numba_test_sa_add(sa0, sa1):
    return sa0 + sa1
@numba.njit
def numba_test_sa_add_0(sa0, sa1):
    return sa0 + sa1[0]
@numba.njit
def numba_test_sa_add0(sa0, sa1):
    return sa0[0] + sa1

@numba.njit
def numba_test_sa_mul(sa0, sa1):
    return sa0 * sa1
@numba.njit
def numba_test_sa_mul_0(sa0, sa1):
    return sa0 * sa1[0]
@numba.njit
def numba_test_sa_mul0(sa0, sa1):
    return sa0[0] * sa1


@numba.njit
def numba_SA_loop(P0, n_iter, ref_path, kcX):
    xr_2 = numba_xr.Xrange_scalar(complex(2.), numba.int32(0))
    P0 = P0 * (P0 +  xr_2 * ref_path[0]) + kcX
    return P0

def std_SA_loop(P0, n_iter, ref_path, kcX):
    xr_2 = Xrange_array(2.) #(1j, numba.int32(1))
    P0 = P0 * (P0 + xr_2 * ref_path[0]) + kcX
    return P0

class Test_SA_xr(unittest.TestCase):
    
    def test_neg(self):
        for dtype in (np.float64, np.complex128):
            with self.subTest(dtype=dtype):
                nvec = 100
                xa, stda = generate_random_xr(dtype, nvec=nvec)# , max_bin_exp=250)
                xerr, stderr = generate_random_xr(np.float64, nvec=1, seed=480)# , max_bin_exp=250)
                sa = Xrange_SA(xa, cutdeg=nvec-1, err=abs(xerr))
                res = numba_test_sa_neg(sa)
                expected = -sa
                _matching(res.coeffs, expected.coeffs)
                _matching(res.err, expected.err)

    def test_add(self):
        for (dtypea, dtypeb) in crossed_dtypes: #((np.float64, np.float64),):#crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec, max_bin_exp=1)# , max_bin_exp=250)
                xerra, stderra = generate_random_xr(np.float64, nvec=1, max_bin_exp=1)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, max_bin_exp=1)# , max_bin_exp=250)
                xerrb, stderrb = generate_random_xr(np.float64, nvec=1, max_bin_exp=1)# , max_bin_exp=250)
                
                sa = Xrange_SA(xa, cutdeg=nvec-1, err=abs(xerra))
                sb = Xrange_SA(xb, cutdeg=nvec-1, err=abs(xerrb))
                res = numba_test_sa_add(sa, sb)
                expected = sa + sb
                _matching(res.coeffs, expected.coeffs)
                _matching(res.err, expected.err)
                
        for (dtypea, dtypeb) in crossed_dtypes: #((np.float64, np.float64),):#crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb, kind="scalar"):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec, max_bin_exp=3)# , max_bin_exp=250)
                xerra, stderra = generate_random_xr(np.float64, nvec=1, seed=512, max_bin_exp=3)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=1, seed=510)# , max_bin_exp=250)

                sa = Xrange_SA(xa, cutdeg=nvec-1)
                res = numba_test_sa_add_0(sa, xb)
                expected = sa + stdb[0]
                _matching(res.coeffs, expected.coeffs)
                _matching(res.err, expected.err)
                res = numba_test_sa_add0(xb, sa)
                expected = stdb[0] + sa 
                _matching(res.coeffs, expected.coeffs)
                _matching(res.err, expected.err)


    def test_mul(self):
        for (dtypea, dtypeb) in crossed_dtypes:
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 60
                xa, stda = generate_random_xr(dtypea, nvec=nvec, max_bin_exp=3)# , max_bin_exp=250)
                xerra, stderra = generate_random_xr(np.float64, nvec=1, seed=512, max_bin_exp=3)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec, seed=810, max_bin_exp=3)# , max_bin_exp=250)
                xerrb, stderrb = generate_random_xr(np.float64, nvec=1, seed=510, max_bin_exp=3)# , max_bin_exp=250)
                
                sa = Xrange_SA(xa, cutdeg=nvec-1, err=abs(xerra))
                sb = Xrange_SA(xb, cutdeg=nvec-1, err=abs(xerrb))
                res = numba_test_sa_mul(sa, sb)
                expected = sa * sb
                _matching(res.coeffs, expected.coeffs.to_standard(), almost=True,
                          dtype=np.float64, ktol=50.)
                _matching(res.err, expected.err.to_standard(), almost=True,
                          dtype=np.float64, ktol=50.)

        for (dtypea, dtypeb) in crossed_dtypes:
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb, kind="scalar"):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec, max_bin_exp=3)# , max_bin_exp=250)
                xerra, stderra = generate_random_xr(np.float64, nvec=1, seed=512, max_bin_exp=3)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=1, seed=510)# , max_bin_exp=250)

                sa = Xrange_SA(xa, cutdeg=nvec-1)
                res = numba_test_sa_mul_0(sa, xb)
                expected = sa * stdb[0]
                _matching(res.coeffs, expected.coeffs)
                _matching(res.err, expected.err)
                res = numba_test_sa_mul0(xb, sa)
                expected = stdb[0] * sa 
                _matching(res.coeffs, expected.coeffs)
                _matching(res.err, expected.err)

    def test_op_partial(self):
        a = [1., 2., 5., 8.]
        _Pa = Xrange_SA(a, 10)
        Pa = np.polynomial.Polynomial(a)
        b = [1., 2.]
        _Pb = Xrange_SA(b, 10)
        Pb = np.polynomial.Polynomial(b)
        
        with self.subTest(op="+"):
            res = [numba_test_sa_add(_Pa, _Pb),
                   numba_test_sa_add(_Pb, _Pa),
                   numba_test_sa_add(_Pa, _Pa)]
            expected = [Pa + Pb , Pb + Pa, Pa + Pa]
            for i in range(len(res)):
                _matching(res[i].coeffs, expected[i].coef)

        with self.subTest(op="*"):
            res = [numba_test_sa_mul(_Pa, _Pb),
                   numba_test_sa_mul(_Pb, _Pa),
                   numba_test_sa_mul(_Pa, _Pa)]
            expected = [Pa * Pb , Pb * Pa, Pa * Pa]
            for i in range(len(res)):
                _matching(res[i].coeffs, expected[i].coef)

    def test_SA_loop(self):

        n_iter = 6
        # Loader        
        test_dir = os.path.join(
                os.path.dirname(__file__),
                "REFERENCE_DATA",
                "xdata_REF"
        )
        
        file_prefix = "64SA" #"noSA"
        mandelbrot = fsmodels.Perturbation_mandelbrot(test_dir)
        mandelbrot.zoom(
            precision=16,
            x="-1.74920463345912691e+00",
            y="-2.8684660237361114e-04",
            dx="5e-12",
            nx=600,
            xy_ratio=1.,
            theta_deg=0.
        )
        FP_params, ref_path = mandelbrot.reload_ref_point() #0, file_prefix)
        cutdeg = 16
        kc = mandelbrot.ref_point_kc() #(0, file_prefix)

        kcX = np.insert(kc, 0, 0.)
        kcX = Xrange_SA(kcX, cutdeg=cutdeg)
#        print("kcX:\n", kcX, kcX.coeffs, kcX.coeffs.dtype, type(kcX.coeffs))
        P0 = Xrange_SA([0j], cutdeg=cutdeg)
#        print("P0:\n", P0, P0.coeffs, P0.coeffs.dtype, type(P0.coeffs))
        
#        ref_path = Xrange_array(ref_path)
        
        P0_std = copy.deepcopy(P0)
        for i_iter in range(n_iter):
            n_iter +=1
#            print("i_iter", i_iter)
            # keep a copy
            
            # P_old1 = P[1].coeffs.copy()
            P0 = numba_SA_loop(P0, i_iter, ref_path[i_iter - 1, :], kcX)
            P0_std = std_SA_loop(P0_std, i_iter, ref_path[i_iter - 1, :], kcX)
            _matching(P0.coeffs, P0_std.coeffs.to_standard(), almost=True,
                          dtype=np.float64, ktol=20.)
            _matching(P0.err, P0_std.err.to_standard(), almost=True,
                          dtype=np.float64, ktol=10.)
#
#@numba.njit
#def numba_test_bivar_polycall(poly, x, y):
#    return poly.__call__(x, y)
#@numba.njit
#def numba_test_bivar_polycall0(poly, x, y):
#    return poly.__call__(x[0], y[0])
#
#@numba.njit
#def numba_test_bivar_deriv(poly, direction):
#    return poly.deriv(direction)
#
#class Test_bivar_poly_xr(unittest.TestCase):
#
#    def test_neg(self):
#        for dtype in (np.float64, np.complex128):
#            with self.subTest(dtype=dtype):
#                nvec = 10
#                xa, stda = generate_random_xr(dtype, nvec=nvec**2)# , max_bin_exp=250)
#                xa = xa.reshape(nvec, nvec)
#                _P = Xrange_bivar_polynomial(xa, cutdeg=nvec-1)
#                old_coeffs = _P.coeffs.copy()
#                res = numba_test_polyneg(_P)
#                expected = - _P
#                _matching(res.coeffs, expected.coeffs.to_standard())
#                # Check that the original array has not been modified
#                _matching(_P.coeffs, old_coeffs)
#                
#    def test_add(self):
#        for (dtypea, dtypeb) in crossed_dtypes: #((np.float64, np.float64),):#crossed_dtypes: 
#            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
#                nvec = 10
#                xa, stda = generate_random_xr(dtypea, nvec=nvec**2, max_bin_exp=20)
#                xa = xa.reshape(nvec, nvec)
#                xb, stdb = generate_random_xr(dtypeb, nvec=nvec**2, seed=510, max_bin_exp=20)
#                xb = xb.reshape(nvec, nvec)
#
#                _Pa = Xrange_bivar_polynomial(xa, cutdeg=nvec-1)
#                _Pb = Xrange_bivar_polynomial(xb, cutdeg=nvec-1)
#
#                res = numba_test_polyadd(_Pa, _Pb)
#                expected = _Pa + _Pb
#                _matching(res.coeffs, expected.coeffs.to_standard())
#        
#        for (dtypea, dtypeb) in crossed_dtypes: #((np.float64, np.float64),):#crossed_dtypes: 
#            with self.subTest(dtypea=dtypea, dtypeb=dtypeb, kind="scalar"):
#                dtypec = np.result_type(dtypea, dtypeb)
#                nvec = 10
#                xa, stda = generate_random_xr(dtypea, nvec=nvec**2, max_bin_exp=20)
#                xa = xa.reshape(nvec, nvec)
#                xb, stdb = generate_random_xr(dtypeb, nvec=1, seed=5, max_bin_exp=2)
#                _Pa = Xrange_bivar_polynomial(xa, cutdeg=nvec-1)
#
#                res = numba_test_polyadd_0(_Pa, xb)
#                expected = _Pa + xb
#                _matching(res.coeffs, expected.coeffs.to_standard(),
#                          almost=True, ktol=20., dtype=dtypec)
#                res = numba_test_polyadd0(xb, _Pa)
#                _matching(res.coeffs, expected.coeffs.to_standard(),
#                          almost=True, ktol=20., dtype=dtypec)
#
#
#    def test_call(self):
#        for (dtypea, dtypeb) in crossed_dtypes: 
#            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
#                dtypec = np.result_type(dtypea, dtypeb)
#                # Test with arrays
#                npoly = 10
#                nvec = 100
#                c, stdc = generate_random_xr(dtypea, nvec=npoly**2, max_bin_exp=2)
#                c = c.reshape((npoly, npoly))
#                stdc = stdc.reshape((npoly, npoly))
#                for i in range(npoly):
#                    for j in range(npoly - i,npoly):
#                        c[i, j] = 0.
#                        stdc[i, j] = 0.
#                x, stdx = generate_random_xr(dtypeb, nvec=nvec , max_bin_exp=2, seed=10)
#                y, stdy = generate_random_xr(dtypeb, nvec=nvec , max_bin_exp=2, seed=810)
#                _P = Xrange_bivar_polynomial(c, npoly - 1)
#                expected = np.polynomial.polynomial.polyval2d(stdx, stdy, stdc)
#                
#                res = numba_test_bivar_polycall(_P, x, y).view(Xrange_array)
#                
#                # res = _P(x, y)
#                _matching(res, expected, almost=True, ktol=20., dtype=dtypec)
#                # print("matching arr", res, expected)
#
#                # Test with scalars
#                x = 1.
#                y = -1.5
#                Xx = Xrange_array(np.array([x,], dtype=dtypeb))
#                Xy = Xrange_array(np.array([y,], dtype=dtypeb))
#                expected = np.polynomial.polynomial.polyval2d(x, y, stdc)
#                res = numba_test_bivar_polycall0(_P, Xx, Xy).view(Xrange_array)
#                print("matching scalar", res, expected)
#                _matching(res, expected, almost=True, ktol=50., dtype=dtypec)
#        
#    # TODO test deriv("X") and  deriv("Y")
#    def test_deriv(self):
#        # Basic test of derivative - array size is fixed
#        for dtype in [np.float64, np.complex128]:
#            n_items = 10
#            p_arr = np.ones((n_items, n_items), dtype)
#            P = Xrange_bivar_polynomial(p_arr, cutdeg=n_items - 1)
#
#            expected = np.zeros(p_arr.shape, dtype)
#            for i in range(n_items - 1):
#                for j in range(n_items - i - 1):
#                    expected[i, j] = i + 1
#            _matching(numba_test_bivar_deriv(P, "X").coeffs, expected, almost=True, ktol=3.,
#                          dtype=dtype)
#
#            expected = np.zeros(p_arr.shape, dtype)
#            for j in range(n_items - 1):
#                for i in range(n_items - j - 1):
#                    expected[i, j] = j + 1
#            _matching(numba_test_bivar_deriv(P, "Y").coeffs, expected, almost=True, ktol=3.,
#                          dtype=dtype)
#
#
#    def test_mul(self):
#        for (dtypea, dtypeb) in crossed_dtypes: 
#            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
#                nvec = 10
#                xa, stda = generate_random_xr(dtypea, nvec=nvec**2)# , max_bin_exp=250)
#                xa = xa.reshape(nvec, nvec)
#                xb, stdb = generate_random_xr(dtypeb, nvec=nvec**2, seed=510)# , max_bin_exp=250)
#                xb = xb.reshape(nvec, nvec)
#
#                _Pa = Xrange_bivar_polynomial(xa, cutdeg=nvec-1)
#                _Pb = Xrange_bivar_polynomial(xb, cutdeg=nvec-1)
#
#                res = numba_test_polymul(_Pa, _Pb)
#                expected = _Pa * _Pb
#                _matching(res.coeffs, expected.coeffs.to_standard())
#
#        for (dtypea, dtypeb) in crossed_dtypes: #((np.float64, np.float64),):#crossed_dtypes: 
#            with self.subTest(dtypea=dtypea, dtypeb=dtypeb, kind="scalar"):
#                nvec = 10
#                xa, stda = generate_random_xr(dtypea, nvec=nvec**2)# , max_bin_exp=250)
#                xa = xa.reshape(nvec, nvec)
#                xb, stdb = generate_random_xr(dtypeb, nvec=1, seed=5)# , max_bin_exp=250)
#                _Pa = Xrange_bivar_polynomial(xa, cutdeg=nvec-1)
#           
#                res = numba_test_polymul_0(_Pa, xb)
#                expected = _Pa * stdb[0]
#                _matching(res.coeffs, expected.coeffs.to_standard())
#
#                res = numba_test_polymul0(xb, _Pa)
#                _matching(res.coeffs, expected.coeffs.to_standard())
#
#@numba.njit
#def SA_deriv(SA, direction):
#    return SA.to_polynomial().deriv(direction).to_SA()
#
#@numba.njit
#def SA_oper1(_P):
#    _QQ = (
#            ((_P + _P) * (_P + _P))
#            - 2. * (_P * _P) * 2. #2.
#    )
#    return _QQ
#
#@numba.njit
#def SA_oper2(_P, _DX, _DY):
#    _QQ = (
#        ((_P + _DX) * (_P + _DY)) 
#        -  ((_P * _P) + ((_DX + _DY) * _P) + (_DX * _DY))
#    )
#    return _QQ
#
#@numba.njit
#def SA_op_std(_P):
#    _QQ = (
#        ((5. + (1. * _P - _P * 2. + 3. * _P)) * 2. * (1. + 1j))
#        + (_P + (1. - 1j))
#    )
#    return _QQ
#
#class Test_bivar_SA_xr(unittest.TestCase):
#    def test_mul(self):
#        for (dtypea, dtypeb) in crossed_dtypes: 
#            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
#
#                arr = [
#                    [11., 12., 13. , 14., 15.],
#                    [21., 22., 23. , 24., 25.],
#                    [31., 32., 33. , 34., 35.],
#                    [41., 42., 43. , 44., 45.],
#                    [51., 52., 53. , 54., 55.]
#                ]
#                arr = np.array(arr, dtype=dtypea)
#                arr0 = [
#                    [11., 12., 13., 14., 15.],
#                    [21., 22., 23., 24.,  0.],
#                    [31., 32., 33.,  0.,  0.],
#                    [41., 42.,  0.,  0.,  0.],
#                    [51.,  0.,  0.,  0.,  0.]
#                ]
#                _P = Xrange_bivar_SA(arr, 4)
#        
#                _Q = numba_test_sa_mul(_P, _P)
#                _expected = [
#                    [121.,   264.,  430.,  620., 835.],
#                    [462.,   988., 1580., 2240.,   0.],
#                    [1123., 2372., 3750.,    0.,   0.],
#                    [2204., 4616.,    0.,    0.,   0.],
#                    [3805.,    0.,    0.,    0.,   0.]
#                ]
#                _matching(_Q.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
#            
#                # Checking error term :
#                # Checked with https://www.wolframalpha.com
#                assert abs(_Q.err - 25998.431510381546) < 0.01
#                _Perr = Xrange_bivar_SA(arr, 4)
#                _Perr.err = Xrange_array(1.0)
#                _Qerr1 = _Perr * _P
#                assert abs(_Qerr1.err - _Q.err - np.sqrt(np.sum(np.square(arr0)))
#                           ) < 0.01
#                _Qerr2 = _P * _Perr
#                assert abs(_Qerr2.err - _Q.err - np.sqrt(np.sum(np.square(arr0)))
#                           ) < 0.01
#                _Qerr3 = _Perr * _Perr
#                assert abs(_Qerr3.err - _Q.err - 2 * np.sqrt(np.sum(np.square(arr0)))
#                           - 1.
#                           ) < 0.01
#    
#                _expected = [
#                    [ 22.,  24.,  26.,  28.,  30.],
#                    [ 42.,  44.,  46.,  48.,  0.],
#                    [ 62.,  64.,  66.,  0.,  0.],
#                    [ 82.,  84.,  0.,  0.,  0.],
#                    [102., 0., 0., 0., 0.]
#                ]
#
#                _Qadd = numba_test_sa_add(_P, _P)
#                _matching((_Qadd).coeffs, _expected,
#                          almost=True, ktol=5., dtype=np.float64)
#                assert abs(_Qadd.err) == 0.0
#                _Qadd = numba_test_sa_add(_Perr, _Perr)
#                assert abs(_Qadd.err - 2.0) == 0.0
#
#                _DX = SA_deriv(_P, "X")
#                _expected = [
#                    [21., 22., 23. , 24., 0.],
#                    [62., 64., 66. , 0., 0.],
#                    [123., 126., 0. , 0., 0.],
#                    [204., 0., 0. , 0., 0.],
#                    [0., 0., 0. , 0., 0.]
#                ]
#                _matching(_DX.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
#
#                _DY = SA_deriv(_P, "Y")
#                _expected = [
#                    [12., 26., 42. , 60., 0.],
#                    [22., 46., 72. , 0., 0.],
#                    [32., 66., 0. , 0., 0.],
#                    [42., 0., 0. , 0., 0.],
#                    [0., 0., 0. , 0., 0.]
#                ]
#                _matching(_DY.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
#
#                _expected = [
#                   [ 32.,  34.,  36.,  38.,  15.],
#                   [ 83.,  86.,  89.,  24.,   0.],
#                   [154., 158.,  33.,   0.,   0.],
#                   [245.,  42.,   0.,   0.,   0.],
#                   [ 51.,   0.,   0.,   0.,   0.]
#               ]
#                _matching((numba_test_sa_add(_P, _DX)).coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
#                _QQ = SA_oper1(_P)
#                _expected = np.zeros((5, 5))
#                _matching(_QQ.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
#                _QQ = SA_oper2(_P, _DX, _DY)
#                _matching(_QQ.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
#
#    def test_op_with_std(self):
#        "Check operation with standard (float 64 / complex 128) numbers" 
#        arr0 = [
#            [11., 12., 13., 14., 15.],
#            [21., 22., 23., 24.,  0.],
#            [31., 32., 33.,  0.,  0.],
#            [41., 42.,  0.,  0.,  0.],
#            [51.,  0.,  0.,  0.,  0.]
#        ]
#        _P = Xrange_bivar_SA(arr0, 4)
#        res = SA_op_std(_P)
#
#        _P = Xrange_bivar_polynomial(arr0, 4)
#        res = SA_op_std(_P)
#
#        arr0 = [11., 12., 13., 14., 15.]
#        _P = Xrange_SA(arr0, 4)
#        res = SA_op_std(_P)
#
#        _P = Xrange_polynomial(arr0, 4)
#        res = SA_op_std(_P)
#        del res
#        
#        
#@numba.njit
#def monome_boxing(kX):
#    print(kX.k)
#    return kX
#
#@numba.njit
#def monome_neg(kX):
#    return (-kX)
#
#@numba.njit
#def monome_add(op0, op1):
#    return op0 + op1
#
#class Test_Xrange_monome(unittest.TestCase):
#
#    def test_box_unbox(self):
#        k = Xrange_array("1.e-8000")
#        kX = Xrange_monome(k)
#        print(kX)
#        ret = monome_boxing(kX)
#        print(ret)
#        print(ret.k)
#        
#    def test_neg(self):
#        k = Xrange_array("1.e-8000")
#        kX = Xrange_monome(k)
#        retk = monome_neg(kX).k.view(Xrange_array)
#        assert retk == - k
#        
#        k *= (1-1.j)
#        kX = Xrange_monome(k)
#        retk = monome_neg(kX).k.view(Xrange_array)
#        assert retk == - k
#        
#    def test_add(self):
#        k0 = Xrange_array("1.")
#        for k in (k0, (1.+1.j) * k0):
#            cast_complex = (k == (1.+1.j))
#            k = Xrange_array(k)
#            kX = Xrange_monome(k)
#            print("k for add", k)
#            
#            for dtype in (np.float64, np.complex128):
#                for n in range(12):
#                    arr = Xrange_array.ones(n, dtype=dtype)
#                    sa = Xrange_SA(arr, cutdeg=10)
#                    res = monome_add(kX, sa)
#                    res2 = monome_add(sa, kX)
#                    if n > 1:
#                        expected = arr.copy()[:max(n, 11)]
#                        if cast_complex:
#                            expected = complex(1.) * expected
#                        expected[1] += k
#                        assert np.all(expected == res.coeffs)
#                        assert np.all(expected == res2.coeffs)
#                
#                arr = Xrange_array.ones((11, 11), dtype=dtype)
#                expected = arr.copy()
#                for i in range(11):
#                    for j in range(11):
#                        if (i + j) > 10:
#                            expected[i, j] = 0
#                if cast_complex:
#                    expected = complex(1.) * expected
#                expected[1, 0] = expected[1, 0] + k
#                bivar_sa = Xrange_bivar_SA(arr, cutdeg=10)
#                res = monome_add(kX, bivar_sa)
#                res2 = monome_add(bivar_sa, kX)
#                assert np.all(expected == res.coeffs)
#                assert np.all(expected == res2.coeffs)
#
##    def test_add(self):
        


if __name__ == "__main__":
    import test_config
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([
            Test_numba_xr,
            Test_poly_xr,
            Test_SA_xr,
#            Test_bivar_poly_xr,
#            Test_bivar_SA_xr,
#            Test_Xrange_monome
        ]))
    else:
        suite = unittest.TestSuite()
#        suite.addTest(Test_poly_xr("test_call"))
        suite.addTest(Test_numba_xr("test_adding_smallstd"))
        # suite.addTest(Test_numba_xr("test_to_standard"))
        # suite.addTest(Test_poly_xr("test_expr"))
        # suite.addTest(Test_SA_xr("test_expr"))
        # suite.addTest(Test_numba_xr("test_conversion"))
#        suite.addTest(Test_bivar_poly_xr("test_neg"))
#        suite.addTest(Test_bivar_poly_xr("test_add"))
#        suite.addTest(Test_bivar_poly_xr("test_call"))
        #suite.addTest(Test_bivar_poly_xr("test_deriv"))
#        suite.addTest(Test_bivar_poly_xr("test_mul"))
#        # suite.addTest(Test_bivar_SA_xr("test_op_with_std"))
#        suite.addTest(Test_numba_xr("test_geom_mean"))
#        suite.addTest(Test_numba_xr("test_unbox_xr_scalar"))
#        suite.addTest(Test_Xrange_monome("test_box_unbox"))
#        suite.addTest(Test_Xrange_monome("test_neg"))
#        suite.addTest(Test_Xrange_monome("test_add"))

        runner.run(suite)












