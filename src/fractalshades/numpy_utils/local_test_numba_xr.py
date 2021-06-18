# -*- coding: utf-8 -*-
import numpy as np
import time
import unittest

import numba

# Allows relative imports when run locally as script
# https://docs.python-guide.org/writing/structure/
#if __name__ == "__main__":
#    sys.path.insert(0, os.path.abspath(
#            os.path.join(os.path.dirname(__file__), '..')))


#import fractalshades.numpy_utils.xrange as fsx
from xrange import (
   Xrange_array,
   Xrange_polynomial,
   Xrange_SA
)

import numba_xr
#from numba_xr import (
#    extended_setitem_tuple
#)


from_complex = {np.complex64: np.float32,
                np.complex128: np.float64}
crossed_dtypes = [
    (np.float64, np.float64),
    (np.complex128, np.complex128),
    (np.float64, np.complex128),
    (np.float64, np.complex128)
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
    std = mantissa.copy() * 2. ** exp
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
    arr[idx] = val_tuple

@numba.njit
def numba_test_add(a, b, out):
    n, = a.shape
    for i in range(n):
        out[i] = a[i] + b[i]

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

@numba.njit(parallel=True)
def numba_test_div(a, b, out):
    n, = a.shape
    for i in numba.prange(n):
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

class Test_numba_xr(unittest.TestCase):

    def test_setitem(self):
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            with self.subTest(dtype=dtype):
                nvec = 500
                xr, std = generate_random_xr(dtype, nvec=nvec)
                xr2 = Xrange_array.zeros(xr.shape, dtype)
                for i in range(nvec):
                    val_tuple = xr._mantissa[i], xr._exp[i]
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
#        exp = np.asarray(xr["exp"])
        outm = np.empty(std.shape, dtype)
        outexp = np.empty(std.shape, np.int32)
        numba_test_frexp(std, outm, outexp)
        expd_m, expd_exp = np.frexp(std, outm)
        np.testing.assert_array_equal(std, outm * 2. ** outexp)
        np.testing.assert_array_equal(outm, expd_m)
        np.testing.assert_array_equal(outexp, expd_exp)
        # np.testing.assert_array_equal(out, np.ldexp(std, -exp))

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
                
                print("t_numba", t_numba)
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
                self.assertTrue(expr, msg="Numba speed below numpy")

    def test_sub(self):
        for (dtypea, dtypeb) in crossed_dtypes: #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 10000
                xa, stda = generate_random_xr(dtypea, nvec=nvec)
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
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
                self.assertTrue(expr, msg="Numba speed below numpy")
    
    
    def test_mul(self):
        for (dtypea, dtypeb) in crossed_dtypes: #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100000
                xa, stda = generate_random_xr(dtypea, nvec=nvec)
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
                
                print("t_numba", t_numba)
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
                self.assertTrue(expr, msg="Numba speed below numpy")
    
    def test_div(self):
        for (dtypea, dtypeb) in crossed_dtypes: #, np.complex128]: # np.complex64 np.float32
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100000
                xa, stda = generate_random_xr(dtypea, nvec=nvec)
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
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
                self.assertTrue(expr, msg="Numba speed below numpy")
        
#    def test_op_with_scalar(self):
#        for dtype in [np.float64, np.complex128]: #, np.complex128]: # np.complex64 np.float32
#            with self.subTest(dtype=dtype):
#                nvec = 100000
#                xa, stda = generate_random_xr(dtype, nvec=nvec)
#                xb, stdb = generate_random_xr(dtype, nvec=nvec, seed=7800)
#                res = Xrange_array.empty(xa.shape, dtype)
#                expected = stda / stdb
#                numba_test_div(xa, xb, res)
#                # Numba timing without compilation
#                t_numba = - time.time()
#                numba_test_div(xa, xb, res)
#                t_numba += time.time()
#                # numpy timing 
#                t_np = - time.time()
#                res_np = xa / xb
#                t_np += time.time()
#                
#                _matching(res, expected, almost=True, dtype=np.float64,
#                          ktol=2.)
#                _matching(res_np, expected, almost=True, dtype=np.float64,
#                          ktol=2.)
#                
#                print("t_numba", t_numba)
#                print("t_numpy", t_np, t_numba/t_np)
#                expr = (t_numba <  t_np)
#                self.assertTrue(expr, msg="Numba speed below numpy")
        
if __name__ == "__main__":
    import test_config
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_numba_xr,]))
    else:
        suite = unittest.TestSuite()
        suite.addTest(Test_numba_xr("test_mul"))
        runner.run(suite)