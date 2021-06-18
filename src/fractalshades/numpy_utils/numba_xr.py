# -*- coding: utf-8 -*-
import numpy as np
import numbers
import re

import numba
from numba.core import types, utils, typing, errors, cgutils, extending
from numba import (
    njit,
    generated_jit
)
from numba.core.errors import TypingError
from numba.np import numpy_support
from numba.extending import (
    overload,
    overload_attribute,
    overload_method
)

import fractalshades.numpy_utils.xrange as fsx
import math
import operator

"""
The purpose of this module is to allow the use of Xrange_arrays inside numba
jitted functions.

By default, Numba will treat all numpy.ndarray subtypes as if they were of the
base numpy.ndarray type. On one side, ndarray subtypes can easily use all of
the support that Numba has for ndarray methods ; on the other side it is not
possible to fully customise the behavior.
(This is likely to change in future release of Numba, see 
https://github.com/numba/numba/pull/6148)

The workaround followed here is to provide ad-hoc implementation at datatype
level (in numba langage, for our specific numba.types.Record types). User code
in jitted function should fully expand the loops to work on individual array
elements - indeed numba is made for this.

As the extra complexity is not worth it, we drop support for float32, complex64
in numba: only float64, complex128 mantissa are currently supported.
"""

def np_xr_type(base_type):
    """ return the numba type for th 4 implemented base type
    float32 float64 complex64, complex128 """
    xr_type = np.dtype([("mantissa", base_type), ("exp", np.int32)],
                        align=False)
    return xr_type

def numba_xr_type(base_type):
    """ return the numba type for th 2 implemented base type
     float64, complex128 """
    xr_type = np.dtype([("mantissa", base_type), ("exp", np.int32)],
                        align=False)
    return numba.from_dtype(xr_type)

xr_type0 = np.dtype([("mantissa", np.float64), ("exp", np.int32)],
                    align=False)

np_base_types = (np.float64, np.complex128)
numba_base_types = (numba.float64, numba.complex128)
numba_xr_types = tuple(numba_xr_type(dtype) for dtype in np_base_types)
#numba_xr_types = numba_xr_type(np.float64) # for dtype in np_base_types)

np_float_types = (np.float64,)
numba_float_types = (numba.float64,)

np_complex_types = (np.complex128,)
numba_complex_types = (numba.complex128,)



@overload_attribute(numba.types.Record, "is_xr")
def is_xr(rec):
    ret = tuple(rec.fields.keys()) == ("mantissa", "exp")
    def impl(rec):
        return ret
    return impl


@overload(operator.setitem)
def extended_setitem_tuple(arr, idx, val_tuple):
    """
    Usage : if arr is an Xrange_array, then one will be able to do
        arr[i] = (mantissa, exp)
    """
    if (
        isinstance(val_tuple, types.Tuple)
        and isinstance(idx, types.Integer)
        and isinstance(arr, types.Array)
    ):
        if tuple(arr.dtype.fields.keys()) != ("mantissa", "exp"):
            raise TypingError("Only xrange dtype accepted")
        def impl(arr, idx, val_tuple):
            arr[idx]["mantissa"], arr[idx]["exp"] = val_tuple
        return impl


# https://github.com/numba/numba/blob/e314821f48bfc1678c9662584eef166fb9d5469c/numba/np/arrayobj.py






@overload_attribute(numba.types.Record, "is_complex")
def is_complex(rec):
    # _RecordField(type=, offset=, alignment=, title=)]
    dtype = rec.fields["mantissa"][0]
    is_complex = (dtype in numba_complex_types)
    def impl(rec):
        return is_complex
    return impl

# @overload_method(numba.types.Record, "_normalize")
# @numba.njit #((numba.float64, numba.int32))# ()
#def _normalize():
#    return None


@overload(operator.neg)
def extended_neg(op0):
    """ Change sign of a Record field """
    if (op0 in numba_xr_types):
        def impl(op0):
            return (-op0.mantissa, op0.exp)
        return impl
    else:
        raise TypingError("datatype not accepted {}".format(
            op0))

@overload(operator.add)
def extended_add(op0, op1):
    """ Add 2 Record fields """
    if (op0 in numba_xr_types) and (op1 in numba_xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1.mantissa, op1.exp)
            return (m0_out + m1_out, exp_out)
        return impl
    elif (op0 in numba_xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1, 0)
            return (m0_out + m1_out, exp_out)
        return impl
    elif (op0 in numba_base_types) and (op1 in numba_xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0, 0, op1.mantissa, op1.exp)
            return (m0_out + m1_out, exp_out)
        return impl
    else:
        raise TypingError("datatype not accepted xr_add({}, {})".format(
            op0, op1))

@overload(operator.sub)
def extended_sub(op0, op1):
    """ Substract 2 Record fields """
    if (op0 in numba_xr_types) and (op1 in numba_xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1.mantissa, op1.exp)
            return (m0_out - m1_out, exp_out)
        return impl
    elif (op0 in numba_xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1, 0)
            return (m0_out - m1_out, exp_out)
        return impl
    elif (op0 in numba_base_types) and (op1 in numba_xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0, 0, op1.mantissa, op1.exp)
            return (m0_out - m1_out, exp_out)
        return impl
    else:
        raise TypingError("datatype not accepted xr_sub({}, {})".format(
            op0, op1))

# @njit((numba.float64,))#, numba.int32))
@generated_jit(nopython=True)
def _need_renorm(m):
    """
    Returns True if abs(exponent) is above a given threshold
    """
    threshold = 100  # 2.**100 = 1.e30
    if (m in numba_float_types):
        def impl(m):
            bits = numba.cpython.unsafe.numbers.viewer(m, numba.int64)
            return abs(((bits >> 52) & 0x7ff) - 1023) > threshold
        return impl
    elif (m in numba_complex_types):
        def impl(m):
            bits = numba.cpython.unsafe.numbers.viewer(m.real, numba.int64)
            need1 = abs(((bits >> 52) & 0x7ff) - 1023) > threshold
            bits = numba.cpython.unsafe.numbers.viewer(m.imag, numba.int64)
            need2 = abs(((bits >> 52) & 0x7ff) - 1023) > threshold
            return (need1 or need2)
        return impl
    else:
        raise TypingError("datatype not accepted {}".format(m))

@overload(operator.mul)
def extended_mul(op0, op1):
    """ Multiply 2 Record fields """
    if (op0 in numba_xr_types) and (op1 in numba_xr_types):
        def impl(op0, op1):
            mul = op0.mantissa * op1.mantissa
            if _need_renorm(mul):
                return _normalize(mul, op0.exp + op1.exp)
            return (mul, op0.exp + op1.exp)
        return impl
    elif (op0 in numba_xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            mul = op0.mantissa * op1
            if _need_renorm(mul):
                return _normalize(mul, op0.exp)
            return (mul, op0.exp)
        return impl
    elif (op0 in numba_base_types) and (op1 in numba_xr_types):
        def impl(op0, op1):
            mul = op0 * op1.mantissa
            if _need_renorm(mul):
                return _normalize(mul, op1.exp)
            return (mul, op1.exp)
        return impl
    else:
        raise TypingError("datatype not accepted xr_mul({}, {})".format(
            op0, op1))

@overload(operator.truediv)
def extended_truediv(op0, op1):
    """ Divide 2 Record fields """
    if (op0 in numba_xr_types) and (op1 in numba_xr_types):
        def impl(op0, op1):
            mul = op0.mantissa / op1.mantissa
            if _need_renorm(mul):
                return _normalize(mul, op0.exp - op1.exp)
            return (mul, op0.exp - op1.exp)
        return impl
    elif (op0 in numba_xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            mul = op0.mantissa / op1
            if _need_renorm(mul):
                return _normalize(mul, op0.exp)
            return (mul, op0.exp)
        return impl
    elif (op0 in numba_base_types) and (op1 in numba_xr_types):
        def impl(op0, op1):
            mul = op0 / op1.mantissa
            if _need_renorm(mul):
                return _normalize(mul, -op1.exp)
            return (mul, op1.exp)
        return impl
    else:
        raise TypingError("datatype not accepted xr_mul({}, {})".format(
            op0, op1))



@generated_jit(nopython=True) #(_normalize)
def _normalize(m, exp):
    """ Returns a normalized couple """
    # Implementation for float
    # dtype = m.dtype
    if (m in numba_float_types):
        def impl(m, exp):
            return _normalize_real(m, exp)
    # Implementation for complex
    elif (m in numba_complex_types):
        def impl(m, exp):
            nm_re, nexp_re = _normalize_real(m.real, exp)
            nm_im, nexp_im = _normalize_real(m.imag, exp)
            co_nm_real, co_nm_imag, co_nexp = _coexp_ufunc(
                nm_re, nexp_re, nm_im, nexp_im)
            return (co_nm_real + 1j * co_nm_imag, co_nexp)
    else:
        raise TypingError("datatype not accepted {}".format(m))
    return impl

@njit((numba.float64,))#, numba.int32))
def _frexp(m):
    """ Faster equivalent for math.frexp(m) """
    # https://github.com/numba/numba/issues/3763
    # https://llvm.org/docs/LangRef.html#bitcast-to-instruction
    bits = numba.cpython.unsafe.numbers.viewer(m, numba.int64)
#    exp = ((bits >> 52) & 0x7ff)
    m = numba.cpython.unsafe.numbers.viewer(
        (bits & 0x8000000000000000) # signe
        + (0x3ff << 52) # exposant (bias) hex(1023) = 0x3ff
        + (bits & 0xfffffffffffff), numba.float64)
    exp = numba.int32(((bits >> 52)) & 0x7ff) - numba.int32(1022)
    return m, exp



#    val = np.asarray(val)
#    if val.dtype == np.float32:
#        bits = val.view(np.int32)
#        return np.any((np.abs(((bits >> 23) & 0xff) - 127) > 31)
#                      & (val != 0.))
#    elif val.dtype == np.float64:
#        bits = val.view(np.int64)
#        return np.any((np.abs(((bits >> 52) & 0x7ff) - 1023) > 255)
#                      & (val != 0.))


# @generated_jit(nopython=True) #(_normalize)
@njit((numba.float64, numba.int32))# ()
def _normalize_real(m, exp):
    """ Returns a normalized couple """
    # Implementation for float
    # dtype = m.dtype
#    if m in numba_float_types:
#        def impl(m, exp):
    if m == 0.:
        return (m, np.int32(0))
    else:
        nm, nexp = _frexp(m) # math.frexp(m) # 
        return (nm, exp + nexp)
#    else:
#        raise TypingError("datatype not accepted {}".format(m))
#    return impl



@njit((numba.float64, numba.int32))
def _exp2_shift(m, shift):
    """ faster equivalent for math.ldexp(m, shift) """
    # https://github.com/numba/numba/issues/3763
    # https://llvm.org/docs/LangRef.html#bitcast-to-instruction
    # math.
    bits = numba.cpython.unsafe.numbers.viewer(m, numba.int64)
    exp = max(((bits >> 52) & 0x7ff) + shift, 0)
    return numba.cpython.unsafe.numbers.viewer(
        (bits & 0x8000000000000000)
        + (exp << 52)
        + (bits & 0xfffffffffffff), numba.float64)

#@njit((numba.float32, numba.int32))
#def _exp2_shift(m, shift):
#    # https://github.com/numba/numba/issues/3763
#    # https://llvm.org/docs/LangRef.html#bitcast-to-instruction
#    bits = numba.cpython.unsafe.numbers.viewer(m, numba.int32)
#    exp = max(((bits >> 23) & 0xff) + shift, 0)  # (bits >> 23) & 0xff) - 127
#    return numba.cpython.unsafe.numbers.viewer(
#        (bits & 0x80000000)
#        + (exp << 23)
#        + (bits & 0xfffffffffffff), numba.float64)


@generated_jit(nopython=True) #(_normalize)
def _coexp_ufunc(m0, exp0, m1, exp1):
    """ Returns a co-exp couple of couples """
    # Implementation for real
    if (m0 in numba_float_types) and (m1 in numba_float_types):
        def impl(m0, exp0, m1, exp1):
            co_m0, co_m1 = m0, m1
            d_exp = exp0 - exp1
            if m0 == 0.:
                exp = exp1
            elif m1 == 0.:
                exp = exp0
            elif (exp1 > exp0):
                co_m0 = _exp2_shift(co_m0, d_exp) # _exp2_shift(co_m0, d_exp) #_exp2_shift(co_m0, d_exp)
                exp = exp1
            elif (exp0 > exp1):
                co_m1 = _exp2_shift(co_m1, -d_exp) #  _exp2_shift(co_m1, -d_exp) # _exp2_shift(co_m1, -d_exp)
                exp = exp0
            else: # exp0 == exp1
                exp = exp0
            return (co_m0, co_m1, exp)
    # Implementation for complex
    elif (m0 in numba_complex_types) or (m1 in numba_complex_types):
        def impl(m0, exp0, m1, exp1):
            co_m0, co_m1 = m0, m1
            d_exp = exp0 - exp1
            if m0 == 0.:
                exp = exp1
            elif m1 == 0.:
                exp = exp0
            elif (exp1 > exp0):
                co_m0 = (_exp2_shift(co_m0.real, d_exp) #_exp2_shift(co_m0, d_exp)
                         + 1j * _exp2_shift(co_m0.imag, d_exp)) #_exp2_shift(co_m0, d_exp)
                exp = exp1
            elif (exp0 > exp1):
                co_m1 = (_exp2_shift(co_m1.real, -d_exp)
                         + 1j * _exp2_shift(co_m1.imag, -d_exp))
                                # _exp2_shift(co_m1, -d_exp)
                # co_m1.imag = math.ldexp(co_m1.imag, -d_exp) # _exp2_shift(co_m1, -d_exp)
                exp = exp0
            else: # exp0 == exp1
                exp = exp0
            return (co_m0, co_m1, exp)
    else:
        raise TypingError("datatype not accepted {}{}".format(m0, m1))
    return impl




#@generated_jit(nopython=True) #(_normalize)
#def _exp2_shift(m, d_exp):
#    # https://docs.python.org/3/library/math.html
#    retrun math.ldexp(m, d_exp)
#    dtype = m.dtype
#    if m == numba.float32:# and (m in numba_float_types):
#        bits = m.view(np.int32)
#            # Need to take special care as casting to int32 a 0d array is only
#            # supported if the itemsize is unchanged. So we impose the res 
#            # dtype
#            res_32 = np.empty_like(bits)
#        exp = np.clip(((bits >> 23) & 0xff) + shift, 0, None)
#        out = (exp << 23) + (bits & 0x7fffff)
#        return np.copysign(res_32.view(np.float32), m)
#
#        elif dtype == np.float64:
#            bits = m.view(np.int64)
#            exp = np.clip(((bits >> 52) & 0x7ff) + shift, 0, None)
#            return np.copysign(((exp << 52) + (bits & 0xfffffffffffff)
#                                ).view(np.float64) , m)
#        else:
#            raise ValueError("Unsupported dtype {}".format(dtype))


@overload_method(numba.types.Record, "normalize")
def normalize(rec):
    """ Normalize in-place a xr Record """
    dtype = rec.fields["mantissa"][0]
    # Implementation for float
    if (dtype in numba_float_types):
        def impl(rec):
            m = rec.mantissa
            if m == 0.:
                rec.exp = 0
            else:
                nm, nexp = math.frexp(rec.mantissa) 
                rec.exp += nexp
                rec.mantissa = nm
    # Implementation for complex
    elif (dtype in numba_complex_types):
        def impl(rec):
            m = rec.mantissa
            if m == 0.:
                rec.exp = 0
            else:
                rec.mantissa, rec.exp = _normalize(m, rec.exp)
    else:
        raise TypingError("datatype not accepted {}".format(dtype))
    return impl

#        else:
#            nm, exp2 = np.frexp(m)
#            if m == 0.:
#                rec.exp = 0
#                rec.mantissa = 0
#            else:
#                rec.exp = 
#                rec.mantissa =
#    return impl

@njit# (np.int32(np.float64))
def frexp(my_float):
    # def impl(my_float):
#    bits = my_float.view(np.int64)
#    print("bits", bits, bits >> 52)
#    exp = (bits >> 52) & 0x7ff
    return math.frexp(my_float)
    # return impl


@njit
def numba_test1_xr(arr):
    rec = arr[0]
    return rec.is_complex

@njit
def numba_test_is_xr(arr):
    rec = arr[0]
    return rec.is_xr
    
def test_1_xr():
    arr = fsx.Xrange_array(["1.e12", "3.14"])
    # arr = arr * (1. + 0.j)
    print("arr1", arr)
    ret = arr[0].is_complex
    print("RET", ret)
    retn = numba_test1_xr(arr)
    print("RET", retn)
    arr = arr * (1. + 0.j)
    retn = numba_test1_xr(arr)
    print("RET", retn)
    
@numba.njit
def numba_test2_xr(arr):
    rec = arr[0]
    return _normalize(rec["mantissa"], rec["exp"])

def test_2_xr():
    arr = fsx.Xrange_array(["1.e12", "3.14"])
    print("arr2", arr)
    ret = numba_test2_xr(arr)
    print("normalized", ret)
    arr = arr * (1. + 1.j)
    ret = numba_test2_xr(arr)
    print("normalized", ret)

@numba.njit
def numba_test_add(arr0, arr1, out):
    print(arr0[0] + arr1[0])
    print(arr0[1] + arr1[1])
    # need to implement setitem for array(Record), int, tuple
    out[0] = arr0[0] + arr1[0]
    out[1] = arr0[1] + arr1[1]

def test_add_xr():
    arr0 = fsx.Xrange_array(["1.e1", "3.14"])
    arr1 = fsx.Xrange_array(["2.e2", "3.14159"])
    out = fsx.Xrange_array(["0.", "0."])
    numba_test_add(arr0, arr1, out)
    print("add", out)

@numba.njit
def numba_test_setitem(arr, idx, val_tuple):
    arr[idx] = val_tuple

#def test_add_xr():
@numba.njit
def numba_test_add2(arr):
    arr[2] = arr[0] + 1.# arr[1]

if __name__ == "__main__":
#    print("numba_xr_types", numba_xr_types)
#    print("numba_xr_types", numba_xr_types[np.float32])
#    print("numba_xr ##:\n", numba_xr_types.values())
#    print("\n\n test1\n")
#    test_1_xr()
#    res = frexp(np.array(1.))
    # print(res)
#    res = frexp(np.array(5235532552.))
    res = frexp(5235532552.)
    # a = np.array([4235532552.]).view(np.int64)
    # array([4751172963183624192])
    a = np.array([2.], dtype=np.float64)
    res = _exp2_shift(a[0], -2)
#    
    print(res)
    a[0] = -1.e-32
    need = _need_renorm(a[0])
    print(need)
#    a = np.array([.25], dtype=np.float64)
#    res = _frexp(a[0])
#    print(_frexp.signatures[0]) #inspect_disasm_cfg(signature=_frexp.signatures[0]))
#    
#    print(res) #, type(res[0]))
    
    
#    test_2_xr()
#    test_add_xr()
    
    arr = fsx.Xrange_array(["1.e1", "3.14", "0.0"])
#    numba_test_setitem(arr, 0, (1., 8))
    print("arr", arr)
    # arr[2] = arr[0] + arr[1]
    numba_test_add2(arr)
    print("sum", arr[2])
#    
#    print("is xr", numba_test_is_xr(arr))
    