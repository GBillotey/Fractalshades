# -*- coding: utf-8 -*-
import numpy as np
import time
import unittest
import operator
import numba

# Allows relative imports when run locally as script
# https://docs.python-guide.org/writing/structure/
#if __name__ == "__main__":
#    sys.path.insert(0, os.path.abspath(
#            os.path.join(os.path.dirname(__file__), '..')))


#import fractalshades.numpy_utils.xrange as fsx
from fractalshades.numpy_utils.xrange import (
   Xrange_array,
   Xrange_polynomial,
   Xrange_SA
)

# import fractalshades.numpy_utils.numba_xr as numba_xr
import numba_xr


from_complex = {np.complex64: np.float32,
                np.complex128: np.float64}
crossed_dtypes = [
    (np.float64, np.float64),
    (np.complex128, np.complex128),
    (np.float64, np.complex128),
    (np.complex128, np.float64)
]
#real_crossed_dtypes = [
#    (np.float64, np.float64)
#]

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
def numba_test_sqrt(xa, out):
    n, = xa.shape
    for i in range(n):
        out[i] = np.sqrt(xa[i])

@numba.njit
def numba_test_abs(xa, out):
    n, = xa.shape
    for i in range(n):
        out[i] = np.abs(xa[i])

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
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
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
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
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

                print("t_numba, numpy", t_numba, t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
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
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
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

                    print("t_numba", t_numba)
                    print("t_numpy", t_np, t_numba/t_np)
                    expr = (t_numba <  t_np)
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
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
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
                          ktol=2.)
                _matching(res_np, expected, almost=True, dtype=np.float64,
                          ktol=2.)

                print("t_numba", t_numba)
                print("t_numpy", t_np, t_numba/t_np)
                expr = (t_numba <  t_np)
                self.assertTrue(expr, msg="Numba speed below numpy")

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
                    print("t_numpy", t_np, t_numba/t_np)
                    expr = (t_numba <  t_np)
                    self.assertTrue(expr, msg="Numba speed below numpy")

@numba.njit
def numba_test_polyneg(poly):
    return -poly

@numba.njit
def numba_test_polyadd(polya, polyb):
    return polya + polyb

@numba.njit
def numba_test_polycall0(poly, val):
    return poly.__call__(val[0])

@numba.njit
def numba_test_polycall(poly, val):
    return poly.__call__(val)

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
        for (dtypea, dtypeb) in ((np.float64, np.float64),):#crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=nvec)# , max_bin_exp=250)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec)# , max_bin_exp=250)
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

    def test_op_partial(self):
        a = [1., 2., 5., 8.]
        _Pa = Xrange_polynomial(a, 10)
        Pa = np.polynomial.Polynomial(a)
        b = [1., 2.]
        _Pb = Xrange_polynomial(b, 10)
        Pb = np.polynomial.Polynomial(b)
        
        res = [numba_test_polyadd(_Pa, _Pb),
               numba_test_polyadd(_Pb, _Pa),
               numba_test_polyadd(_Pa, _Pa)]
        expected = [Pa + Pb , Pb + Pa, Pa + Pa]
        for i in range(len(res)):
            _matching(res[i].coeffs, expected[i].coef)

    def test_call(self):
        for (dtypea, dtypeb) in crossed_dtypes: 
            with self.subTest(dtypea=dtypea, dtypeb=dtypeb):
                nvec = 100
                xa, stda = generate_random_xr(dtypea, nvec=10 , max_bin_exp=3)
                xb, stdb = generate_random_xr(dtypeb, nvec=nvec , max_bin_exp=5)
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


if __name__ == "__main__":
    import test_config
    full_test = False
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_numba_xr,]))
        runner.run(test_config.suite([Test_poly_xr,]))
    else:
        suite = unittest.TestSuite()
        # suite.addTest(Test_numba_xr("test_expr"))
        suite.addTest(Test_poly_xr("test_add_partial"))
        runner.run(suite)