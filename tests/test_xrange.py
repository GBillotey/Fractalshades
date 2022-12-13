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
)


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
            _matching(res, expected, almost, dtype, cmp_op, ktol=4.)
            
            
            _op3 = Xrange_array(op2, exp_shift_array).reshape(*resh)
            op3 = (op2 * 2.**exp_shift_array).reshape(resh)
            for axis in range(3):
                res = np.cumprod(_op3, axis=axis)
                expected = np.cumprod(op3, axis=axis)
                _matching(res, expected, almost, dtype, cmp_op)   
                
            expected = np.cumprod(op1 * 2.**exp_shift_array)
            res = np.cumprod(Xrange_array(op1, exp_shift_array))
            _matching(res, expected, almost, dtype, cmp_op, ktol=8.)

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


if __name__ == "__main__":
    full_test = True
    runner = unittest.TextTestRunner(verbosity=2)
    if full_test:
        runner.run(test_config.suite([Test_Xrange_array]))
        runner.run(test_config.suite([Test_Xrange_timing]))
    else:
        suite = unittest.TestSuite()
        # suite.addTest(Test_Xrange_array("test_deriv"))
        runner.run(suite)

