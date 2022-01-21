# -*- coding: utf-8 -*-
import numpy as np
import time
import unittest
import mpmath

import test_config


#import fractalshades.numpy_utils.xrange as fsx
from fractalshades.numpy_utils.xrange import (
        Xrange_array,
        mpf_to_Xrange,
        mpc_to_Xrange,
        Xrange_polynomial,
        Xrange_SA,
        Xrange_bivar_polynomial,
        Xrange_bivar_SA)




def _matching(res, expected, almost=False, dtype=None, cmp_op=False, ktol=1.5):
    if not cmp_op:
        res = res.to_standard()
    if almost:
        np.testing.assert_allclose(res, expected,
                                   rtol= ktol * np.finfo(dtype).eps)
    else:
        np.testing.assert_array_equal(res, expected)


def _test_op1(ufunc, almost=False, cmp_op=False, ktol=1.0):
    """
    General framework for testing unary operators on Xrange arrays
    """
#    print("testing function", ufunc)
    rg = np.random.default_rng(100)

    n_vec = 500
    max_bin_exp = 20
    
    # testing binary operation of reals extended arrays
    for dtype in [np.float64, np.float32]: 
#        print("dtype", dtype)
        op1 = rg.random([n_vec], dtype=dtype)
        op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                               size=[n_vec])
        expected = ufunc(op1)
        res = ufunc(Xrange_array(op1))

        _matching(res, expected, almost, dtype, cmp_op, ktol)

        # Checking datatype
        assert res._mantissa.dtype == dtype

        # with non null shift array # culprit
        exp_shift_array = rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                      size=[n_vec])
        expected = ufunc(op1 * (2.**exp_shift_array).astype(dtype))

        _matching(ufunc(Xrange_array(op1, exp_shift_array)),
                  expected, almost, dtype, cmp_op, ktol)        
        # test "scalar"
        _matching(ufunc(Xrange_array(op1, exp_shift_array)[0]),
                  expected[0], almost, dtype, cmp_op, ktol)
#        print("c2")

    # testing binary operation of reals extended arrays
    for dtype in [np.float32, np.float64]:
        op1 = (rg.random([n_vec], dtype=dtype) +
                   1j*rg.random([n_vec], dtype=dtype))
        op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                               size=[n_vec])
        expected = ufunc(op1)
        res = ufunc(Xrange_array(op1))
        _matching(res, expected, almost, dtype, cmp_op, ktol)

        # Checking datatype
        to_complex = {np.float32: np.complex64,
                 np.float64: np.complex128}
        if ufunc in [np.abs]:
            assert res._mantissa.dtype == dtype
        else:
            assert res._mantissa.dtype == to_complex[dtype]

        # with non null shift array
        exp_shift_array = rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                      size=[n_vec])
        expected = ufunc(op1 * (2.**exp_shift_array))
        _matching(ufunc(Xrange_array(op1, exp_shift_array)),
                  expected, almost, dtype, cmp_op, ktol)

def _test_op2(ufunc, almost=False, cmp_op=False):
    """
    General framework for testing operations between 2 Xrange arrays.
    """
#    print("testing operation", ufunc)
    rg = np.random.default_rng(100)
#    ea_type = (Xrange_array._FLOAT_DTYPES + 
#               Xrange_array._COMPLEX_DTYPES)
    n_vec = 500
    max_bin_exp = 20
    exp_shift = 2
    
    # testing binary operation of reals extended arrays
    for dtype in [np.float32, np.float64]:
        op1 = rg.random([n_vec], dtype=dtype)
        op2 = rg.random([n_vec], dtype=dtype)
        op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                               size=[n_vec])
        op2 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp,
                               size=[n_vec])

        # testing operation between 2 Xrange_arrays OR between ER_A and 
        # a standard np.array
        expected = ufunc(op1, op2)
        res = ufunc(Xrange_array(op1), Xrange_array(op2))
        _matching(res, expected, almost, dtype, cmp_op)
        
#        # testing operation between 2 Xrange_arrays OR between ER_A and 
#        # a standard np.array xith dim 2
        expected_2d = ufunc(op1.reshape(50, 10),
                         op2.reshape(50, 10))
        res_2d = ufunc(Xrange_array(op1.reshape(50, 10)),
                    Xrange_array(op2.reshape(50, 10)))

        _matching(res_2d, expected_2d, almost, dtype, cmp_op)

        # Checking datatype
        if ufunc in [np.add, np.multiply, np.subtract, np.divide]:
            assert res._mantissa.dtype == dtype

        if ufunc not in [np.equal, np.not_equal]:
            _matching(ufunc(op1, Xrange_array(op2)),
                      expected, almost, dtype, cmp_op)
            _matching(ufunc(Xrange_array(op1), op2),
                      expected, almost, dtype, cmp_op)
        # Testing with non-null exponent
        exp_shift_array = rg.integers(low=-exp_shift, high=exp_shift, 
                                      size=[n_vec])
        expected = ufunc(op1 * 2.**exp_shift_array, op2 * 2.**-exp_shift_array)

            

        _matching(ufunc(Xrange_array(op1, exp_shift_array),
                        Xrange_array(op2, -exp_shift_array)),
                  expected, almost, dtype, cmp_op)
        # testing operation of an Xrange_array with a scalar
        if ufunc not in [np.equal, np.not_equal]:
            expected = ufunc(op1[0], op2)
            _matching(ufunc(op1[0], Xrange_array(op2)),
                      expected, almost, dtype, cmp_op)
            expected = ufunc(op2, op1[0])
            _matching(ufunc(Xrange_array(op2), op1[0]),
                      expected, almost, dtype, cmp_op)
            
        # testing operation of an Xrange_array with a "Xrange" scalar
        if ufunc not in [np.equal, np.not_equal]:
            expected = ufunc(op1[0], op2)
            _matching(ufunc(Xrange_array(op1)[0], Xrange_array(op2)),
                      expected, almost, dtype, cmp_op)
            expected = ufunc(op2, op1[0])
            _matching(ufunc(Xrange_array(op2), Xrange_array(op1)[0]),
                      expected, almost, dtype, cmp_op)
            
    if cmp_op and (ufunc not in [np.equal, np.not_equal]):
        return
    if ufunc in [np.maximum]:
        return

    # testing binary operation of complex extended arrays
    for dtype in [np.float32, np.float64]:
        n_vec = 20
        max_bin_exp = 20
        rg = np.random.default_rng(1)
        
        op1 = (rg.random([n_vec], dtype=dtype) +
                   1j*rg.random([n_vec], dtype=dtype))
        op2 = (rg.random([n_vec], dtype=dtype) +
                   1j*rg.random([n_vec], dtype=dtype))
        op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                               size=[n_vec])
        op2 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp,
                               size=[n_vec])
        # testing operation between 2 Xrange_arrays OR between ER_A and 
        # a standard np.array
        expected = ufunc(op1, op2)
        res = ufunc(Xrange_array(op1), Xrange_array(op2))
        _matching(res, expected, almost, dtype, cmp_op)
        
        # Checking datatype
        if ufunc in [np.add, np.multiply, np.subtract, np.divide]:
            to_complex = {np.float32: np.complex64,
                 np.float64: np.complex128}
            assert res._mantissa.dtype == to_complex[dtype]

        _matching(ufunc(op1, Xrange_array(op2)),
                  expected, almost, dtype, cmp_op)
        _matching(ufunc(Xrange_array(op1), op2),
                  expected, almost, dtype, cmp_op)
        # Testing with non-null exponent (real and imag)
        expected = ufunc(op1 * 2.**exp_shift, op2 * 2.**-exp_shift)
        exp_shift_array = exp_shift * np.ones([n_vec], dtype=np.int32)
        _matching(ufunc(
                Xrange_array(op1, exp_shift_array),
                Xrange_array(op2, -exp_shift_array)),
            expected, almost, dtype, cmp_op)
        # Testing cross product of real with complex
        expected = ufunc(op1 * 2.**exp_shift, (op2 * 2.**-exp_shift).real)
        exp_shift_array = exp_shift * np.ones([n_vec], dtype=np.int32)
        _matching(ufunc(
                Xrange_array(op1, exp_shift_array),
                Xrange_array(op2, -exp_shift_array).real),
            expected, almost, dtype, cmp_op)
        expected = ufunc((op1 * 2.**exp_shift).imag, op2 * 2.**-exp_shift)
        _matching(ufunc(
                Xrange_array(op1, exp_shift_array).imag,
                Xrange_array(op2, -exp_shift_array)),
            expected, almost, dtype, cmp_op)
        # testing operation of an Xrange_array with a scalar
        expected = ufunc(op1[0], op2)
        _matching(ufunc(op1[0], Xrange_array(op2)),
                  expected, almost, dtype, cmp_op)
        expected = ufunc(op2, op1[0])
        _matching(ufunc(Xrange_array(op2), op1[0]),
                  expected, almost, dtype, cmp_op)





class Test_Xrange_array(unittest.TestCase):

    def test_sum(self):
        almost=True
        cmp_op=False
        for dtype in [np.float32, np.float64]:
            n_vec = 1000
            max_bin_exp = 20
            rg = np.random.default_rng(1)
            
            op1 = (rg.random([n_vec], dtype=dtype) +
                       1j*rg.random([n_vec], dtype=dtype))
            op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                   size=[n_vec])
            
            op2 = (rg.random([n_vec], dtype=dtype))
            exp_shift_array = rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                   size=[n_vec])
        
            expected = np.sum(op1)
            res = np.sum(Xrange_array(op1))
            _matching(res, expected, almost, dtype, cmp_op)
    
            expected = np.sum(op2 * 2.**exp_shift_array)
            res = np.sum(Xrange_array(op2, exp_shift_array))
            _matching(res, expected, almost, dtype, cmp_op)
            
            _op3 = Xrange_array(op2, exp_shift_array).reshape(10, 10, 10)
            op3 = (op2 * 2.**exp_shift_array).reshape(10, 10, 10)
            for axis in range(3):
                res = np.sum(_op3, axis=axis)
                expected = np.sum(op3, axis=axis)
                _matching(res, expected, almost, dtype, cmp_op)   
                
            expected = np.sum(op1 * 2.**exp_shift_array)
            res = np.sum(Xrange_array(op1, exp_shift_array))
            _matching(res, expected, almost, dtype, cmp_op)
            
            
            _op4 = Xrange_array(op1, exp_shift_array).reshape(10, 10, 10)
            op4 = (op1 * 2.**exp_shift_array).reshape(10, 10, 10)
            for axis in range(3):
                res = np.sum(_op4, axis=axis)
                expected = np.sum(op4, axis=axis)
                _matching(res, expected, almost, dtype, cmp_op)

    def test_cumprod(self):
        almost=True
        cmp_op=False
        for dtype in [np.float32, np.float64]:
            # We need to prevent overflow of 'standard numbers' for the test
            # to be meaningful...
            if dtype == np.float32:
                n_vec = 100
                max_bin_exp = 3
                resh = (4, 5, 5)
            else:
                n_vec = 200
                max_bin_exp = 10
                resh = (4, 10, 5)

            rg = np.random.default_rng(1)
            
            op1 = (rg.random([n_vec], dtype=dtype) +
                       1j*rg.random([n_vec], dtype=dtype))
            op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                   size=[n_vec])
            
            op2 = (rg.random([n_vec], dtype=dtype))
            exp_shift_array = rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                   size=[n_vec])
        
            expected = np.cumprod(op1)
            res = np.cumprod(Xrange_array(op1))
            _matching(res, expected, almost, dtype, cmp_op)
    
            expected = np.cumprod(op2 * 2.**exp_shift_array)
            res = np.cumprod(Xrange_array(op2, exp_shift_array))
            _matching(res, expected, almost, dtype, cmp_op, ktol=2.)
            
            
            _op3 = Xrange_array(op2, exp_shift_array).reshape(*resh)
            op3 = (op2 * 2.**exp_shift_array).reshape(resh)
            for axis in range(3):
                res = np.cumprod(_op3, axis=axis)
                expected = np.cumprod(op3, axis=axis)
                _matching(res, expected, almost, dtype, cmp_op)   
                
            expected = np.cumprod(op1 * 2.**exp_shift_array)
            res = np.cumprod(Xrange_array(op1, exp_shift_array))
            _matching(res, expected, almost, dtype, cmp_op, ktol=8)

            _op4 = Xrange_array(op1, exp_shift_array).reshape(*resh)
            op4 = (op1 * 2.**exp_shift_array).reshape(*resh)
            for axis in range(3):
                res = np.cumprod(_op4, axis=axis)
                expected = np.cumprod(op4, axis=axis)
                _matching(res, expected, almost, dtype, cmp_op)


    def test_angle(self):
        for dtype in [np.float32, np.float64]:
            n_vec = 1000
            max_bin_exp = 20
            rg = np.random.default_rng(1)
            op1 = (rg.random([n_vec], dtype=dtype) +
                       1j*rg.random([n_vec], dtype=dtype))
            exp_shift_array = rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                   size=[n_vec])
            _op1 = Xrange_array(op1, exp_shift_array)
            op1 *= 2.**exp_shift_array
            np.testing.assert_array_equal(np.angle(op1), np.angle(_op1))


    def test_add(self):
        _test_op2(np.add, almost=True)
    def test_multiply(self):
        _test_op2(np.multiply, almost=True)
    def test_subtract(self):
        _test_op2(np.subtract, almost=True)
    def test_divide(self):
        _test_op2(np.divide, almost=True)
    def test_maximum(self):
        _test_op2(np.maximum, almost=False)

    def test_greater(self):
        _test_op2(np.greater, cmp_op=True)
    def test_greater_equal(self):
        _test_op2(np.greater_equal, cmp_op=True)
    def test_less(self):
        _test_op2(np.less, cmp_op=True)
    def test_less_equal(self):
        _test_op2(np.less_equal, cmp_op=True)
    def test_equal(self):
        _test_op2(np.equal, cmp_op=True)
    def test_not_equal(self):
        _test_op2(np.not_equal, cmp_op=True)

    def test_abs(self):
        _test_op1(np.abs, almost=True, ktol=2.0)
    def test_sqrt(self):
        _test_op1(np.sqrt, almost=True)
    def test_square(self):
        _test_op1(np.square, almost=True)
    def test_conj(self):
        _test_op1(np.conj, almost=True)
    def test_log(self):
        _test_op1(np.log, almost=True, ktol=2.0)


    def test_edge_cases(self):
        _dtype = np.complex128
        base = np.linspace(0., 1500., 11, dtype=_dtype)
        base2 = np.linspace(-500., 500., 11, dtype=np.float64)
        # mul
        b = (Xrange_array((2. - 1.j) * base) * 
             Xrange_array((-1. + 1.j) * base2))
        expected = ((2. - 1j) * base) * ((-1. + 1.j) * base2)
        _matching(b, expected)
        # add
        b = (Xrange_array((2. - 1.j) * base) + 
             Xrange_array((-1. + 1.j) * base2))
        expected = ((2. - 1.j) * base) + ((-1. + 1j) * base2)
        _matching(b, expected)
        #  <=
        b = (Xrange_array((2. - 1j) * base).real <= 
             Xrange_array((-1. + 1j) * base2).real)
        expected = ((2. - 1j) * base).real <= ((-1. + 1j) * base2).real
        np.testing.assert_array_equal(b, expected)
        
        
        #   Testing equality with "almost close" floats
        base = - np.ones([40], dtype=np.float64)
        base = np.linspace(0., 1., 40, dtype=np.float64)
        base2 = base + np.linspace(-1., 1., 40) * np.finfo(np.float64).eps * 2.
        exp = np.zeros(40, dtype=np.int32)
    
        _base = Xrange_array(base, exp)
        _base2 = Xrange_array(base2 * 2., exp - 1)
    #    print("######################1")
    #    print("_base", _base)
    #    print("_base2", _base2)
    #    print("== ref: ", base == base2)
        np.testing.assert_array_equal(_base == _base2, base == base2)
        np.testing.assert_array_equal(_base == base2, base == base2)
       # print("######################1.1")
        np.testing.assert_array_equal(_base[:2] == _base2[:2], base[:2] == base2[:2])
    #    print("!=", _base != _base2)
    #    print("ref: ", base != base2)
        np.testing.assert_array_equal(_base != _base2, base != base2)
        np.testing.assert_array_equal(_base <= _base2, base <= base2)
        np.testing.assert_array_equal(_base >= _base2, base >= base2)
        np.testing.assert_array_equal(_base < _base2, base < base2)
        np.testing.assert_array_equal(_base > _base2, base > base2)

        shift = np.arange(40) - 10
        _base2 = Xrange_array(base2 / 2. ** shift, exp + shift)
        np.testing.assert_array_equal(_base != _base2, base != base2)
     #   print("######################2")
        np.testing.assert_array_equal(_base == _base2, base == base2)
        np.testing.assert_array_equal(_base > _base2, base > base2)
      #  print("######################exit")
        _base = _base * (1. +1.j)
        _base2 = _base2 * (1. +1.j)
        base = base * (1. +1.j)
        base2 = base2 * (1. +1.j)
        np.testing.assert_array_equal(_base == _base2, base == base2)
        np.testing.assert_array_equal(_base != _base2, base != base2)
        np.testing.assert_array_equal(_base[2] != _base2[2], base[2] != base2[2])
        np.testing.assert_array_equal(_base[20] != _base2[20],
                                      base[20] != base2[20])
        np.testing.assert_array_equal(_base[20] == _base2[20],
                                      base[20] == base2[20])
        np.testing.assert_array_equal(_base.real == _base2.real,
                                      base.real == base2.real)
        np.testing.assert_array_equal(_base.real != _base2.real,
                                      base.real != base2.real)
        np.testing.assert_array_equal(_base[20].real == _base2[20].real,
                                      base[20].real == base2[20].real)
        np.testing.assert_array_equal(_base.real[20] == _base2.real[20],
                                      base.real[20] == base2.real[20])
        np.testing.assert_array_equal(_base.real <= _base2.real,
                                      base.real <= base2.real)
        np.testing.assert_array_equal(_base[20].real <= _base2[20].real,
                                      base[20].real <= base2[20].real)
        np.testing.assert_array_equal(_base[2].real <= _base2[2].real,
                                      base[2].real <= base2[2].real)
    
        #   Testing complex equality logic
        a = np.array([1., 1., 1., 1.]) + 1.j * np.array([1., 1., 1., 1.])
        b = np.array([1., 1., -1., -1.]) + 1.j * np.array([1., -1., 1., -1.])
        a_ = Xrange_array(a)
        b_ = Xrange_array(b)
        np.testing.assert_array_equal(a_ == b_, a == b)
        np.testing.assert_array_equal(a_ == b, a == b)
        np.testing.assert_array_equal(a == b_, a == b)
        np.testing.assert_array_equal(a_ != b_, a != b)
        np.testing.assert_array_equal(a_ != b, a != b)
        np.testing.assert_array_equal(a != b_, a != b)


    def test_template_view(self):
        """
        Testing basic array capabilities
        Array creation via __new__, template of view
        real and imag are views
        """
        a = np.linspace(0., 5., 12, dtype=np.complex128)
        b = Xrange_array(a)
        # test shape of b and its mantissa / exponenent fields
        assert b.shape == a.shape
        assert b._mantissa.shape == a.shape
        assert b._exp.shape == a.shape
        # b is a full copy not a view
        b11_val = b[11]
        assert b[11] == b11_val#(5.0 + 0.j, 0, 0)
        m = b._mantissa
        assert m[11] == 5.
        assert a[11] != 10.
        a[11] = 10.
        assert b[11] == b11_val
        # you have to make a new instance to see the modification
        b = Xrange_array(a)
        assert b[11] != b11_val
        m = b._mantissa
        assert m[11] == 10.
    
        # Testing Xrange_array from template
        c = b[10:]
        # test shape Xrange_array subarray and its mantissa / exponenent
        assert c.shape == a[10:].shape
        assert c._mantissa.shape == a[10:].shape
        assert c._exp.shape == a[10:].shape
        # modifying subarray modifies array
        new_val = (12345.+0.j, 6)
        c[1] = Xrange_array(*new_val)
        assert b[11] == c[1]
        # modifying array modifies subarray
        new_val = (98765.+0.j, 4)
        b[10] = Xrange_array(*new_val)
        assert b[10] == c[0]
    
        # Testing Xrange_array from view
        d = a.view(Xrange_array)
        assert d.shape == a.shape
        assert d._mantissa.shape == a[:].shape
    
        # modifying array modifies view
        val = a[5]
        assert d._mantissa[5] == val
        val = 8888888.
        a[5] = val
        
        # Check that imag and real are views of the original array 
        e = Xrange_array(a + 2.j * a)
        assert e.to_standard()[4] == (20. + 40.j) / 11.
        re = (e.real).copy()
        re[4] = Xrange_array(np.pi, 0)
        e.real = re
        im = (e.imag).copy()
        im[4] = Xrange_array(-np.pi, 0)
        e.imag = im
    
        assert e.to_standard()[4] == (1. - 1.j) * np.pi
        bb = Xrange_array(np.linspace(0., 5., 12, dtype=np.float64))
        
        np.testing.assert_array_equal(bb.real, bb)
        bb.real[0] = Xrange_array(1.875, 6)  # 120...
        assert bb.to_standard()[0] == 120.
        np.testing.assert_array_equal(bb.imag.to_standard(), 0.)


    def test_print(self):
        """ Testing basic Xrange array prints """
        Xrange_array.MAX_COUNTER = 5
        a = np.array([1., 1., np.pi, np.pi], dtype=np.float64)
        Xa = Xrange_array(a)
        for exp10 in range(1001):
            Xa = Xa * [-10., 0.1, 10., -0.1]
        str8 = ("[-1.00000000e+1001  1.00000000e-1001"
               "  3.14159265e+1001 -3.14159265e-1001]")
        str8_m = ("[ 1.00000000e+1001 -1.00000000e-1001"
                       " -3.14159265e+1001  3.14159265e-1001]")
        str2 = ("[-1.00e+1001  1.00e-1001  3.14e+1001 -3.14e-1001]")
        with np.printoptions(precision=2, linewidth=100) as _:
            assert Xa.__str__() == str2
        with np.printoptions(precision=8, linewidth=100) as _:
            assert Xa.__str__() == str8
        with np.printoptions(precision=8, linewidth=100) as _:
            assert (-Xa).__str__() == str8_m
    
        a = np.array([0.999999, 1.00000, 0.9999996, 0.9999994], dtype=np.float64)
        str5 =  "[ 9.99999e-01  1.00000e+00  1.00000e+00  9.99999e-01]"
        for k in range(10):
            Xa = Xrange_array(a * 0.5**k, k * np.ones([4], dtype=np.int32))
            with np.printoptions(precision=5) as _:
                assert Xa.__str__() == str5
    
        a = 1.j * np.array([1., 1., np.pi, np.pi], dtype=np.float64)
        Xa = Xrange_array(a)
        for exp10 in range(1000):
            Xa = [-10., 0.1, 10., -0.1] * Xa
        str2 = ("[ 0.00e+00+1.00e+1000j  0.00e+00+1.00e-1000j"
                "  0.00e+00+3.14e+1000j  0.00e+00+3.14e-1000j]")
        with np.printoptions(precision=2, linewidth=100) as _:
            assert Xa.__str__() == str2
            
        a = np.array([[0.1, 10.], [np.pi, 1./np.pi]], dtype=np.float64)
        Xa = Xrange_array(a)
        Ya = np.copy(Xa).view(Xrange_array)
        for exp10 in range(21):
            Xa = np.sqrt(Xa * Xa * Xa * Xa)
        for exp10 in range(21):
            Ya = Ya * Ya
        str6 = ("[[ 1.000000e-2097152  1.000000e+2097152]\n"
                " [ 7.076528e+1042598  1.413122e-1042599]]")
        with np.printoptions(precision=6, linewidth=100) as _:
            assert Xa.__str__() == str6
            assert Ya.__str__() == str6
    
        Xa = Xrange_array([["123.456e-1789", "-.3e-7"], ["1.e700", "1.0"]])
        str6 = ("[[ 1.234560e-1787 -3.000000e-08]\n"
                " [ 1.000000e+700  1.000000e+00]]")
        str6_sq = ("[[ 1.524138e-3574  9.000000e-16]\n"
                   " [ 1.000000e+1400  1.000000e+00]]")
        Xb = Xa -1.j * Xa**2
        with np.printoptions(precision=6, linewidth=100) as _:
            assert Xa.__str__() == str6
            assert (Xa**2).__str__() == str6_sq

        # Testing accuracy of mantissa for highest exponents    
        Xa = Xrange_array([["1.0e+646456992", "1.23456789012345e+646456992"], 
                           ["1.0e+646456991", "1.23456789012345e+646456991"], 
                           ["1.0e+646456990", "1.23456789012345e+646456990"],
                           ["-1.0e-646456991", "1.23456789012345e-646456991"], 
                           ["1.0e-646456992", "1.23456789012345e-646456992"]])
        str_14 = ("[[ 1.00000000000000e+646456992  1.23456789012345e+646456992]\n"
            " [ 1.00000000000000e+646456991  1.23456789012345e+646456991]\n"
            " [ 1.00000000000000e+646456990  1.23456789012345e+646456990]\n"
            " [-1.00000000000000e-646456991  1.23456789012345e-646456991]\n"
            " [ 1.00000000000000e-646456992  1.23456789012345e-646456992]]")
        with np.printoptions(precision=14, linewidth=100) as _:
            assert Xa.__str__() == str_14
    
        Xb = np.array([1., -1.j]) * np.pi * Xrange_array(
                ["1.e+646456991","1.e-646456991" ])
        str_14 = ("[ 3.14159265358979e+646456991+0.00000000000000e+00j\n"
                 "  0.00000000000000e+00-3.14159265358979e-646456991j]")
        with np.printoptions(precision=14, linewidth=100) as _:
            assert Xb.__str__() == str_14
        
    def test_item_assignment(self):
        Xa = Xrange_array(["1.0e1002", "2.0e1000" ])
        with np.printoptions(precision=10, linewidth=100) as _:
            assert Xa[0].__str__() == " 1.0000000000e+1002"
            assert type(Xa[0]) is Xrange_array
        assert Xa[0] == Xrange_array("1.0e1002")
        Xa[1] = Xrange_array("9.876543e-999")
        assert Xa[1] == Xrange_array("9.876543e-999")
        Xb = Xa + 1.j * Xa
        assert Xb[0] == Xa[0] + 1.j * Xa[0]
        Xb[0] = Xa[0] + 3.14j * Xa[0]
        assert Xb[0] == Xa[0] + 3.14j * Xa[0]
        Xb[0] = Xa[0]
        assert Xb[0] == Xa[0]
        Xb = Xa + 2.j * Xa
        assert np.all(Xb.real == Xa)
        assert np.all(Xb.imag == 2 * Xa)
        Xb.real = -Xa
        assert np.all(Xb.real == -Xa)
        assert np.all(Xb.imag == 2 * Xa)
        Xb.imag = -2 * Xa
        assert np.all(Xb.real == -Xa)
        assert np.all(Xb.imag == -2. * Xa)

    def test_mpf_to_xrange(self):
        a_str = "1.e-2003"

        a_mpf = mpmath.mpf(a_str)
        a_xr = mpf_to_Xrange(a_mpf)
        self.assertTrue(a_xr == Xrange_array(a_str))

        a_mpc = mpmath.mpc("1.e-2003")
        a_xr = mpc_to_Xrange(a_mpc)
        self.assertTrue(a_xr == Xrange_array(a_str))

        b_mpc = 1j * mpmath.mpc(a_str)
        b_xr = mpc_to_Xrange(b_mpc)
        self.assertTrue(b_xr == 1j * Xrange_array(a_str))

        z_tab = [
            (1j + 0.00000001),
            (1j + 0.00000001),
            (1j + 1.),
            (1j + 1e-16),
            (1e-16 * 1j + 1.),
        ]
        for z in z_tab:
            print("z", z)
            b_mpc = z * mpmath.mpc(a_str)
            b_xr = mpc_to_Xrange(b_mpc)
            expected = z * Xrange_array(a_str)
            diff = b_xr - expected
            print(b_xr - (z * Xrange_array(a_str)))
            print("diff ratio", abs(diff) / abs(expected))
            self.assertTrue(abs(diff) < 1.e-18 * abs(expected))
            #_matching(b_xr, expected)


class Test_Xrange_timing(unittest.TestCase):
    
    def test_timing(self):
        self.assertTrue(self.timing_op1_complex(np.square) < 40)
        self.assertTrue(self.timing_op2_complex(np.add) < 80)
        self.assertTrue(self.timing_op2_complex(np.multiply) < 40)
        self.assertTrue(self.timing_abs2_complex() < 40)
    
    
    def timing_abs2_complex(self, dtype=np.float64):
        import time
        n_vec = 40000
        max_bin_exp = 20
        rg = np.random.default_rng(1) 
    
        op = rg.random([n_vec], dtype=dtype) + 1j*rg.random([n_vec], dtype=dtype)
        exp = rg.integers(-max_bin_exp, max_bin_exp)
        e_op = Xrange_array(op, exp)
        op = op * 2.**exp
        
        t0 = - time.time()
        e_res = e_op.abs2()
        t0 += time.time()
        
        t1 = - time.time()
        expected = op * np.conj(op)
        t1 += time.time()
    
        np.testing.assert_array_equal(e_res.to_standard(), expected)
        if t1 == 0.:
            return 0.
        return t0 / t1


    def timing_op1_complex(self, ufunc, dtype=np.float64):
        import time
        
        n_vec = 40000
        max_bin_exp = 20
        
        rg = np.random.default_rng(1) 
        
        op = rg.random([n_vec], dtype=dtype) + 1j*rg.random([n_vec], dtype=dtype)
        exp = rg.integers(-max_bin_exp, max_bin_exp)
        e_op = Xrange_array(op, exp)
        op = op * (2.**exp)
        
        t0 = - time.time()
        e_res = ufunc(e_op)
        t0 += time.time()
        
        t1 = - time.time()
        expected = ufunc(op)
        t1 += time.time()
    
        np.testing.assert_array_equal(e_res.to_standard(), expected)
        if t1 == 0.:
            return 0.
        return t0 / t1


    def timing_op2_complex(self, ufunc, dtype=np.float64):
        n_vec = 40000
        max_bin_exp = 200
        rg = np.random.default_rng(1) 
    
        op1 = rg.random([n_vec], dtype=dtype) + 1j*rg.random([n_vec], dtype=dtype)
        exp1 = rg.integers(-max_bin_exp, max_bin_exp)
        e_op1 = Xrange_array(op1, exp1)
        op1 = op1 * 2.**exp1
        
        op2 = rg.random([n_vec], dtype=dtype) + 1j*rg.random([n_vec], dtype=dtype)
        exp2 = rg.integers(-max_bin_exp, max_bin_exp)
        e_op2 = Xrange_array(op2, exp2)
        op2 = op2 * 2.**exp2
    
    
        t0 = - time.time()
        e_res = ufunc(e_op1, e_op2)
        t0 += time.time()
        
        t1 = - time.time()
        expected = ufunc(op1, op2)
        t1 += time.time()
    
        np.testing.assert_array_equal(e_res.to_standard(), expected)

        if t1 == 0.:
            return 0.
        print("\ntiming", ufunc, dtype, t0, t1, t0 / t1)
        return t0 / t1

def _matchingP(res, expected, **kwargs):
    if res.size > expected.size:
        _expected = np.zeros(res.shape, expected.dtype)
        _expected[:expected.size] = expected
        expected = _expected
    _matching(res, expected, **kwargs)


class Test_Xrange_polynomial(unittest.TestCase):
    def test_Xrange_polynomial(self):
        arr = [1., 2., 5.]
        _P = Xrange_polynomial(arr, 10)
        P = np.polynomial.Polynomial(arr)
        _matching(_P.coeffs, P.coef)
        _matching((_P * _P).coeffs, (P * P).coef)
        _matching((_P * 2).coeffs, (P * 2).coef)
        _matching((2 * _P).coeffs, (2 * P).coef)
        _matching((_P + _P).coeffs, (P + P).coef)
        _matching((_P + 2).coeffs, (P + 2).coef)
        _matching((2 + _P).coeffs, (2 + P).coef)

        two = Xrange_array([2.])
        _matching((_P * two).coeffs, (P * 2).coef)
        _matching((two * _P).coeffs, (2 * P).coef)
        _matching((_P + two).coeffs, (P + 2).coef)
        _matching((two + _P).coeffs, (2 + P).coef)
    
        _matching((_P - (2 * _P)).coeffs, (P - (2 * P)).coef)
        _matching((_P - 2).coeffs, (P - 2).coef)
    
        arr = [1. + 1.j, 1 - 1.j]
        _P = Xrange_polynomial(arr, 2)
        P = np.polynomial.Polynomial(arr)
        _matching((_P * _P).coeffs, (P * P).coef)
        
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            n_vec = 100
            rg = np.random.default_rng(101)
    
            if dtype in [np.float32, np.float64]:
                arr = rg.random([n_vec], dtype=dtype)
            else:
                real_dtype = np.float32 if dtype is np.complex64 else np.float64
                arr = rg.random([n_vec], dtype=real_dtype) + 1.j * (
                        rg.random([n_vec], dtype=real_dtype))
                
            _P = Xrange_polynomial(arr, 1000)
            P = np.polynomial.Polynomial(arr)
            _matching((_P * _P).coeffs, (P * P).coef, almost=True, ktol=3., dtype=dtype)
            
            n_vec2 = 83
            if dtype in [np.float32, np.float64]:
                arr2 = rg.random([n_vec2 ], dtype=dtype)
            else:
                real_dtype = np.float32 if dtype is np.complex64 else np.float64
                arr2 = rg.random([n_vec2 ], dtype=real_dtype) + 1.j * (
                        rg.random([n_vec2 ], dtype=real_dtype))
            
            _Q = Xrange_polynomial(arr2, 1000)
            Q = np.polynomial.Polynomial(arr2)
            _matching((_Q * _P).coeffs, (Q * P).coef, almost=True, ktol=3., dtype=dtype)
            _matching((_P * _Q).coeffs, (P * Q).coef, almost=True, ktol=3., dtype=dtype)
            
            _matching(_P([1.]), P(np.asarray([1.])), almost=True, ktol=3., dtype=dtype)
            _matching(_P([1.j]), P(np.asarray([1.j])), almost=True, ktol=3., dtype=dtype)
            _matching(_P([arr]), P(np.asarray([arr])), almost=True, ktol=3., dtype=dtype)
            
            print("_Q", _Q)
    
            # checking with cutdeg
            for cutdeg in range(1, 400, 10):
                _P = Xrange_polynomial(arr, cutdeg)
                _Q = Xrange_polynomial(arr2, cutdeg)
                _matching((_Q * _P).coeffs, (Q * P).cutdeg(cutdeg).coef,
                          almost=True, ktol=3., dtype=dtype)
                _P = Xrange_polynomial(arr, cutdeg=1000)
                
        coeff = Xrange_array("1.e-1000")
        arr = [1., 2., 5.]
        _P = Xrange_polynomial(arr, 10)
        P = np.polynomial.Polynomial(arr)
        _matching((2. * _P).coeffs, (2. * P).coef)
        _matching((_P * 2.).coeffs, (P * 2.).coef)
        _matching((2. + _P).coeffs, (2. + P).coef)
        _matching((_P + 2.).coeffs, (P + 2.).coef)
        _matching((coeff + _P).coeffs, (_P + coeff).coeffs)
        coeff * _P
        _P * coeff
        #_matching((coeff * _P).coeffs, (_P * coeff).coeffs)

    def test_shift(self):
        p_arr = [[0., 0., 0., 0., 1.],
                 [1., 0., 0., -7., 1., 21.]
                 ]
        q_arr = [[1., 4., 6., 4., 1.],
                 [16., 88., 195., 207., 106., 21.]
                 ]

        # testing _taylor_shift_one
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for i in range(len(p_arr)):
                P = Xrange_polynomial(np.array(p_arr[i], dtype), cutdeg=100)
                Q = P._taylor_shift_one()
                # print("\nQ", Q)
                _matching(Q.coeffs, np.array(q_arr[i], dtype), almost=True, ktol=3.,
                          dtype=dtype)


        p_arr = [[0., 0., 0., 0., 1.],
                 [1., 0., 0., -7., 1., 21.]
                 ]
        sc_arr = [10., -2.]
        q_arr = [[0., 0., 0., 0., 1.e4],
                 [1., 0., 0., 56., 16., -672.]
                 ]

        # testing scale_shift
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for i in range(len(p_arr)):
                P = Xrange_polynomial(np.array(p_arr[i], dtype), cutdeg=100)
                Q = P.scale_shift(sc_arr[i])
                _matching(Q.coeffs, np.array(q_arr[i], dtype), almost=True, ktol=3.,
                          dtype=dtype)

        # testing general taylor_shift
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            p_arr = [[0., 0., 0., 0., 1.],
                     [1., 0., 0., -7., 1., 21.]
                     ]
            sc_arr = [10., -7.]
            q_arr = [[10000., 4000., 600., 40., 1.],
                     [-348144., 249704., -71589., 10255., -734., 21.]
                     ]
            if dtype in [np.complex64, np.complex128]:
                p_arr += [[0., 0., 0., 0., 1. + 1.j]]
                sc_arr += [10.j]
                q_arr += [[10000. + 10000.j, (4000. - 4000.j), (-600. - 600.j),
                           (-40. + 40.j), (1. + 1.j)]]

            for i in range(len(p_arr)):
                P = Xrange_polynomial(np.array(p_arr[i], dtype), cutdeg=100)
                Q = P.taylor_shift(sc_arr[i])
                _matching(Q.coeffs, np.array(q_arr[i]), almost=True, ktol=3.,
                          dtype=dtype)

    def test_deriv(self):
        # Basic test of derivative - note the reduction of array size
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            n_items = 10
            p_arr = np.ones((10,), dtype)
            P = Xrange_polynomial(p_arr, cutdeg=100)

            expected = np.empty_like(p_arr)[:(n_items - 1)]
            for i in range(n_items - 1):
                expected[i] = i+1
            _matching(P.deriv().coeffs, expected, almost=True, ktol=3.,
                          dtype=dtype)
            
            
class Test_Xrange_SA(unittest.TestCase):
    def test_Xrange_SA(self):
        arr = [1., 2., 5.]
        _P = Xrange_SA(arr, 10)
        P = np.polynomial.Polynomial(arr)
        _matching(_P.coeffs, P.coef)
        _matching((_P * _P).coeffs, (P * P).coef)
        _matching((_P * 2.).coeffs, (P * 2.).coef)
        _matching((2. * _P).coeffs, (2. * P).coef)
        _matching((_P + _P).coeffs, (P + P).coef)
        _matching((_P + 2.).coeffs, (P + 2.).coef)
        _matching((2. + _P).coeffs, (2. + P).coef)
        _matching((_P - (2. * _P)).coeffs, (P - (2. * P)).coef)
        _matching((_P - 2.).coeffs, (P - 2.).coef)
        
        two = Xrange_array([2.])
        _matching((_P * two).coeffs, (P * 2.).coef)
        _matching((two * _P).coeffs, (2. * P).coef)
        _matching((_P + two).coeffs, (P + 2.).coef)
        _matching((two + _P).coeffs, (2. + P).coef)
        _matching((_P - (two * _P)).coeffs, (P - (2. * P)).coef)
        _matching((_P - two).coeffs, (P - 2.).coef)

        arrP = [1., 2., 3., 4.]
        arrQ = [4., 3., 2.]
        _P = Xrange_SA(arrP, 3)
        _Q = Xrange_SA(arrQ, 3)
        P = np.polynomial.Polynomial(arrP)
        Q = np.polynomial.Polynomial(arrQ)
        _prod = _P * _Q
        prod = P * Q
        res = prod - prod.cutdeg(3)
        _matching(_prod.err, np.sqrt(np.sum(np.abs(res.coef)**2)))

        arrP = [1. - 1.j, 2. + 4.j, 3. + 1.j, 4.-3.j]
        arrQ = [4.-7.j, 3.-1.j, 2.+2.j]
        _P = Xrange_SA(arrP, 3)
        _Q = Xrange_SA(arrQ, 3)
        P = np.polynomial.Polynomial(arrP)
        Q = np.polynomial.Polynomial(arrQ)
        _prod = _P * _Q
        prod = P * Q
        res = prod - prod.cutdeg(3)
        _matching(_prod.err, np.sqrt(np.sum(np.abs(res.coef)**2)), almost=True)

        arr = [1. + 1.j, 1 - 1.j]
        _P = Xrange_SA(arr, 10)
        P = np.polynomial.Polynomial(arr)
        _matching((_P * _P).coeffs, (P * P).coef)
        
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            with self.subTest(dtype=dtype):
                n_vec = 100
                rg = np.random.default_rng(101)
        
                if dtype in [np.float32, np.float64]:
                    arr = rg.random([n_vec], dtype=dtype)
                else:
                    real_dtype = (np.float32 if dtype is np.complex64 else
                                  np.float64)
                    arr = rg.random([n_vec], dtype=real_dtype) + 1.j * (
                            rg.random([n_vec], dtype=real_dtype))
                    
                _P = Xrange_SA(arr, 1000)
                P = np.polynomial.Polynomial(arr)
                _matching((_P * _P).coeffs, (P * P).coef, almost=True, ktol=3.,
                          dtype=dtype)
        
                n_vec2 = 83
                if dtype in [np.float32, np.float64]:
                    arr2 = rg.random([n_vec2], dtype=dtype)
                else:
                    real_dtype = (np.float32 if dtype is np.complex64 else
                                  np.float64)
                    arr2 = rg.random([n_vec2], dtype=real_dtype) + 1.j * (
                            rg.random([n_vec2], dtype=real_dtype))
        
                _Q = Xrange_polynomial(arr2, 1000)
                Q = np.polynomial.Polynomial(arr2)
                
                # checking with cutdeg - op_errT
                for cutdeg in range(120, 470, 10): # not testing below cutdeg...
                    _P = Xrange_SA(arr, cutdeg)
                    _Q = Xrange_SA(arr2, cutdeg)
                    _prod = _Q * _P
                    prod = Q * P
                    _matching(_prod.coeffs, prod.cutdeg(cutdeg).coef,
                              almost=True, ktol=3., dtype=dtype)
                    res =  prod - prod.cutdeg(cutdeg)
                    _matching(_prod.err, np.sqrt(np.sum(np.abs(res.coef)**2)), 
                              almost=True, ktol=10., dtype=dtype)

class Test_Xrange_bivar_polynomial(unittest.TestCase):
    def test_Xrange_bivar_polynomial(self):
        """
        11 + 21·X¹ + 31·X² + 41·X³ + 51·X⁴
        + 12·Y¹ + 22·X¹·Y¹ + 32·X²·Y¹ + 42·X³·Y¹
        + 13·Y² + 23·X¹·Y² + 33·X²·Y²
        + 14·Y³ + 24·X¹·Y³
        + 15·Y⁴
        
        (15 Y^4 + 24 X Y^3 + 14 Y^3 + 33 X^2 Y^2 + 23 X Y^2 + 13 Y^2 + 42 X^3 Y + 32 X^2 Y + 22 X Y + 12 Y + 51 X^4 + 41 X^3 + 31 X^2 + 21 X + 11)
        """
        arr = [
            [11., 12., 13. , 14., 15],
            [21., 22., 23. , 24., 25],
            [31., 32., 33. , 34., 35],
            [41., 42., 43. , 44., 45],
            [51., 52., 53. , 54., 55]
        ]
        _P = Xrange_bivar_polynomial(arr, 4)

        _Q = _P * _P
        _expected = [
            [121., 264., 430. , 620., 835.],
            [462., 988., 1580. , 2240., 0.],
            [1123., 2372., 3750. , 0., 0.],
            [2204., 4616., 0. , 0., 0.],
            [3805., 0., 0. , 0., 0.]
        ]
        _matching(_Q.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        _expected = [
            [ 22.,  24.,  26.,  28.,  30.],
            [ 42.,  44.,  46.,  48.,  0.],
            [ 62.,  64.,  66.,  0.,  0.],
            [ 82.,  84.,  0.,  0.,  0.],
            [102., 0., 0., 0., 0.]
        ]
        _matching((_P + _P).coeffs, _expected,
                  almost=True, ktol=5., dtype=np.float64)

        _DX = _P.deriv("X")
        _expected = [
            [21., 22., 23. , 24., 0.],
            [62., 64., 66. , 0., 0.],
            [123., 126., 0. , 0., 0.],
            [204., 0., 0. , 0., 0.],
            [0., 0., 0. , 0., 0.]
        ]
        _matching(_DX.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        _DY = _P.deriv("Y")
        _expected = [
            [12., 26., 42. , 60., 0.],
            [22., 46., 72. , 0., 0.],
            [32., 66., 0. , 0., 0.],
            [42., 0., 0. , 0., 0.],
            [0., 0., 0. , 0., 0.]
        ]
        _matching(_DY.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        # P + _DY 
        # (12 + 22 X + 32 X^2 + 42 X^3 + 26 Y + 46 X Y + 66 X^2 Y + 42 Y^2 + 72 X Y^2 + 60 Y^3)
        # (11 + 21 X + 31 X^2 + 41 X^3 + 12 Y + 22 X Y + 32 X^2 Y + 13 Y^2 + 23 X Y^2 + 14 Y^3) + 33 X^2 Y^2 + 42 X^3 Y + 24 X Y^3 + 51 X^4                                                  15 Y^4 
        # (15 Y^4 + 24 X Y^3 + 14 Y^3 + 33 X^2 Y^2 + 23 X Y^2 + 13 Y^2 + 42 X^3 Y + 32 X^2 Y + 22 X Y + 12 Y + 51 X^4 + 41 X^3 + 31 X^2 + 21 X + 11
        _expected = [
           [ 32.,  34.,  36.,  38.,  15.],
           [ 83.,  86.,  89.,  24.,   0.],
           [154., 158.,  33.,   0.,   0.],
           [245.,  42.,   0.,   0.,   0.],
           [ 51.,   0.,   0.,   0.,   0.]
       ]
        _matching((_P + _DX).coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
        # (32 + 83 X¹ + 154 ·X² + 245 X³ + 51 X⁴ + 34 Y¹ + 86 X¹·Y¹ + 158 X²·Y¹ + 42 X³·Y¹ + 36 Y² + 89 X¹·Y² + 33 X²·Y² + 38 Y³ + 24 X¹·Y³ + 15 Y⁴)

        _QQ = (((_P + _P) * (_P + _P)) - 4. * (_P * _P))#- (_DX * _P) - (_P * _DY) - (_DX * _DY))
        _expected = np.zeros((5, 5))
        _matching(_QQ.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        _QQ = (
            ((_P + _DX) * (_P + _DY)) 
            -  ((_P * _P) + ((_DX + _DY) * _P) + (_DX * _DY))
        )
        _matching(_QQ.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        m_arr = [
            [-11., -12., -13., -14., -15.],
            [-21., -22., -23., -24.,   0.],
            [-31., -32., -33.,   0.,   0.],
            [-41., -42.,   0.,   0.,   0.],
            [-51.,   0.,   0.,   0.,   0.]
        ]
        _M = -_P
        _expected = np.asarray(m_arr)
        _matching(_M.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
        
    
    def test_Xrange_bivar_polynomial_call(self):
        rg = np.random.default_rng(100)
        npoly = 10
        n_pts = 10
        c = rg.random([npoly, npoly])
        x = rg.random([n_pts])
        y = rg.random([n_pts])
        for i in range(npoly):
            for j in range(npoly - i,npoly):
                c[i, j] = 0.
        _P = Xrange_bivar_polynomial(c, npoly - 1)

        expected = np.polynomial.polynomial.polyval2d(x, y, c)
        res = _P(x, y)
        _matching(res, expected, almost=True, ktol=5., dtype=np.float64)
        
        x = 1.
        y = -1.5
        expected = np.polynomial.polynomial.polyval2d(x, y, c)
        res = _P(x, y)
        _matching(res, expected, almost=True, ktol=5., dtype=np.float64)
        
    def test_Xrange_bivar_polynomial_deriv(self):
        # Basic test of derivative - array size is fixed
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            n_items = 10
            p_arr = np.ones((n_items, n_items), dtype)
            P = Xrange_bivar_polynomial(p_arr, cutdeg=n_items - 1)

            expected = np.zeros(p_arr.shape, dtype)
            for i in range(n_items - 1):
                for j in range(n_items - i - 1):
                    expected[i, j] = i + 1
            _matching(P.deriv("X").coeffs, expected, almost=True, ktol=3.,
                          dtype=dtype)

            expected = np.zeros(p_arr.shape, dtype)
            for j in range(n_items - 1):
                for i in range(n_items - j - 1):
                    expected[i, j] = j + 1
            _matching(P.deriv("Y").coeffs, expected, almost=True, ktol=3.,
                          dtype=dtype)
            # print(P, "\n\n", P.deriv("X"), "\n\n",P.deriv("Y"), "\n\n")



class Test_Xrange_bivar_SA(unittest.TestCase):
    def test_Xrange_bivar_SA(self):
        """
        11 + 21·X¹ + 31·X² + 41·X³ + 51·X⁴
        + 12·Y¹ + 22·X¹·Y¹ + 32·X²·Y¹ + 42·X³·Y¹
        + 13·Y² + 23·X¹·Y² + 33·X²·Y²
        + 14·Y³ + 24·X¹·Y³
        + 15·Y⁴
        
        (15 Y^4 + 24 X Y^3 + 14 Y^3 + 33 X^2 Y^2 + 23 X Y^2 + 13 Y^2 + 42 X^3 Y + 32 X^2 Y + 22 X Y + 12 Y + 51 X^4 + 41 X^3 + 31 X^2 + 21 X + 11)
        
        
        2601 X^8 
        + 4284 X^7 Y  + 4182 X^7
        + 5130 X^6 Y^2 + 6708 X^6 Y + 4843 X^6 
        + 5220 X^5 Y^3 + 7740 X^5 Y^2 + 7472 X^5 Y + 4684 X^5 
        + 4635 X^4 Y^4 + 7440 X^4 Y^3 + 8130 X^4 Y^2 + 6776 X^4 Y 
        + 2844 X^3 Y^5 + 5460 X^3 Y^4 + 6652 X^3 Y^3 + 6294 X^3 Y^2 
        + 1566 X^2 Y^6 + 2988 X^2 Y^5  + 4269 X^2 Y^4 + 4512 X^2 Y^3
        + 720 X Y^7 + 1362 X Y^6 + 1928 X Y^5 + 2420 X Y^4
        + 225 Y^8 + 420 Y^7 + 586 Y^6 + 724 Y^5 
        
        -----------------------------------------
        + 4616 X^3 Y
        + 121 + 264 Y + 430 Y^2 + 620 Y^3 + 835 Y^4
        + 462 X+  988 X Y + 1580 X Y^2 + 2240 X Y^3 
        + 1123 X^2 + 2372 X^2 Y + 3750 X^2 Y^2
        + 2204 X^3 + 4616 X^3 Y
        + 3805 X^4 
        
        
        
        err = [
            2601,
            4284, 4182,
            5130, 6708, 4843,
            5220, 7740, 7472, 4684,
            4635, 7440, 8130, 6776,
            2844, 5460, 6652, 6294,
            1566, 2988, 4269, 4512,
            720,  1362, 1928, 2420,
            225,   420,  586,  724,
        ] # 122815
        np.sqrt(np.sum(np.array(err)**2)) # 25998.431510381546
        np.sum(np.array(err)) # 122815
        
        """
        arr = [
            [11., 12., 13. , 14., 15.],
            [21., 22., 23. , 24., 25.],
            [31., 32., 33. , 34., 35.],
            [41., 42., 43. , 44., 45.],
            [51., 52., 53. , 54., 55.]
        ]
        arr0 = [
            [11., 12., 13., 14., 15.],
            [21., 22., 23., 24.,  0.],
            [31., 32., 33.,  0.,  0.],
            [41., 42.,  0.,  0.,  0.],
            [51.,  0.,  0.,  0.,  0.]
        ]
        _P = Xrange_bivar_SA(arr, 4)

        _Q = _P * _P
        _expected = [
            [121., 264., 430. , 620., 835.],
            [462., 988., 1580. , 2240., 0.],
            [1123., 2372., 3750. , 0., 0.],
            [2204., 4616., 0. , 0., 0.],
            [3805., 0., 0. , 0., 0.]
        ]
        _matching(_Q.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
        
        # Checking error term :
        # Checked with https://www.wolframalpha.com
        assert abs(_Q.err - 25998.431510381546) < 0.01
        _Perr = Xrange_bivar_SA(arr, 4)
        _Perr.err = 1.0
        _Qerr1 = _Perr * _P
        assert abs(_Qerr1.err - _Q.err - np.sqrt(np.sum(np.square(arr0)))
                   ) < 0.01
        _Qerr2 = _P * _Perr
        assert abs(_Qerr2.err - _Q.err - np.sqrt(np.sum(np.square(arr0)))
                   ) < 0.01
        _Qerr3 = _Perr * _Perr
        assert abs(_Qerr3.err - _Q.err - 2 * np.sqrt(np.sum(np.square(arr0)))
                   - 1.
                   ) < 0.01

        _expected = [
            [ 22.,  24.,  26.,  28.,  30.],
            [ 42.,  44.,  46.,  48.,  0.],
            [ 62.,  64.,  66.,  0.,  0.],
            [ 82.,  84.,  0.,  0.,  0.],
            [102., 0., 0., 0., 0.]
        ]
        _matching((_P + _P).coeffs, _expected,
                  almost=True, ktol=5., dtype=np.float64)

        _DX = _P.deriv("X")
        _expected = [
            [21., 22., 23. , 24., 0.],
            [62., 64., 66. , 0., 0.],
            [123., 126., 0. , 0., 0.],
            [204., 0., 0. , 0., 0.],
            [0., 0., 0. , 0., 0.]
        ]
        _matching(_DX.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        _DY = _P.deriv("Y")
        _expected = [
            [12., 26., 42. , 60., 0.],
            [22., 46., 72. , 0., 0.],
            [32., 66., 0. , 0., 0.],
            [42., 0., 0. , 0., 0.],
            [0., 0., 0. , 0., 0.]
        ]
        _matching(_DY.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        # P + _DY 
        # (12 + 22 X + 32 X^2 + 42 X^3 + 26 Y + 46 X Y + 66 X^2 Y + 42 Y^2 + 72 X Y^2 + 60 Y^3)
        # (11 + 21 X + 31 X^2 + 41 X^3 + 12 Y + 22 X Y + 32 X^2 Y + 13 Y^2 + 23 X Y^2 + 14 Y^3) + 33 X^2 Y^2 + 42 X^3 Y + 24 X Y^3 + 51 X^4                                                  15 Y^4 
        # (15 Y^4 + 24 X Y^3 + 14 Y^3 + 33 X^2 Y^2 + 23 X Y^2 + 13 Y^2 + 42 X^3 Y + 32 X^2 Y + 22 X Y + 12 Y + 51 X^4 + 41 X^3 + 31 X^2 + 21 X + 11
        _expected = [
           [ 32.,  34.,  36.,  38.,  15.],
           [ 83.,  86.,  89.,  24.,   0.],
           [154., 158.,  33.,   0.,   0.],
           [245.,  42.,   0.,   0.,   0.],
           [ 51.,   0.,   0.,   0.,   0.]
       ]
        _matching((_P + _DX).coeffs, _expected, almost=True, ktol=5., dtype=np.float64)
        # (32 + 83 X¹ + 154 ·X² + 245 X³ + 51 X⁴ + 34 Y¹ + 86 X¹·Y¹ + 158 X²·Y¹ + 42 X³·Y¹ + 36 Y² + 89 X¹·Y² + 33 X²·Y² + 38 Y³ + 24 X¹·Y³ + 15 Y⁴)

        _QQ = (((_P + _P) * (_P + _P)) - 2. * (_P * _P) * 2.)#- (_DX * _P) - (_P * _DY) - (_DX * _DY))
        _expected = np.zeros((5, 5))
        _matching(_QQ.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        _QQ = (
            ((_P + _DX) * (_P + _DY)) 
            -  ((_P * _P) + ((_DX + _DY) * _P) + (_DX * _DY))
        )
        _matching(_QQ.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)

        m_arr = [
            [-11., -12., -13., -14., -15.],
            [-21., -22., -23., -24.,   0.],
            [-31., -32., -33.,   0.,   0.],
            [-41., -42.,   0.,   0.,   0.],
            [-51.,   0.,   0.,   0.,   0.]
        ]
        _M = -_P
        _expected = np.asarray(m_arr)
        _matching(_M.coeffs, _expected, almost=True, ktol=5., dtype=np.float64)


if __name__ == "__main__":
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_Xrange_array]))
        runner.run(test_config.suite([Test_Xrange_timing]))
        runner.run(test_config.suite([Test_Xrange_polynomial]))
        runner.run(test_config.suite([Test_Xrange_SA, Test_Xrange_bivar_SA]))
    else:
        suite = unittest.TestSuite()
        # suite.addTest(Test_Xrange_polynomial("test_deriv"))
        suite.addTest(Test_Xrange_bivar_polynomial("test_Xrange_bivar_polynomial_deriv"))
        suite.addTest(Test_Xrange_bivar_SA("test_Xrange_bivar_SA"))
        runner.run(suite)

