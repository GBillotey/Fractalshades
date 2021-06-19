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
    overload_method,
    lower_builtin,
    typeof_impl,
    type_callable,
    models,
    register_model,
    make_attribute_wrapper, 
    unbox
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

NOte:
    https://numba.pydata.org/numba-doc/latest/proposals/extension-points.html
"""

# Implementation of 

numba_float_types = (numba.float64,)
numba_complex_types = (numba.complex128,)
numba_base_types = numba_float_types + numba_complex_types

def numba_xr_type(base_type):
    """ Return the numba "extended" Record type for the 2 implemented base type
    float64, complex128 """
    xr_type = np.dtype([("mantissa", base_type), ("exp", np.int32)],
                        align=False)
    return numba.from_dtype(xr_type)

numba_xr_types = tuple(numba_xr_type(dt) for dt in (np.float64, np.complex128))



# Create a datatype for Records 
# This datatype will only be used in numba jitted functions, so we do not
# expose a python implementation (class, boxing, unboxing)

class XrangeScalar():
    pass

class XrangeScalarType(types.Type):
    def __init__(self, base_type):
        super().__init__(name="{}_XrangeScalar".format(base_type))
        self.base_type = base_type

@type_callable(XrangeScalar)
def type_extended_item(context):
    def typer(mantissa, exp):
        if (mantissa in numba_base_types) and (exp == numba.int32):
            return XrangeScalarType(mantissa)
    return typer

@register_model(XrangeScalarType)
class XrangeScalar_Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('mantissa', fe_type.base_type),
            ('exp', numba.int32),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

for attr in ('mantissa', 'exp'):
    make_attribute_wrapper(XrangeScalarType, attr, attr)

@lower_builtin(XrangeScalar, types.Number, types.Integer)
def impl_xrange_scalar(context, builder, sig, args):
    typ = sig.return_type
    mantissa, exp = args
    xrange_scalar = cgutils.create_struct_proxy(typ)(context, builder)
    xrange_scalar.mantissa = mantissa
    xrange_scalar.exp = exp
    return xrange_scalar._getvalue()

# We will support operation between numba_xr_types and XrangeScalar instances
scalar_xr_types = tuple(XrangeScalarType(dt) for dt in numba_base_types)
xr_types = numba_xr_types + scalar_xr_types


        

@overload_attribute(numba.types.Record, "is_xr")
def is_xr(rec):
    ret = tuple(rec.fields.keys()) == ("mantissa", "exp")
    def impl(rec):
        return ret
    return impl


# Dedicated typing for Xrange_array adds some overhead with little benefit
# -> not implemented by default (passthrough as types.Array)
xrange_typing = False
xrange_type = types.Array

class XrangeArray(types.Array):
    # Array type source code
    # https://github.com/numba/numba/blob/39271156a52c58ca18b15aebcb1c85e4a07e49ed/numba/core/types/npytypes.py#L413
    # pd-like use case
    # https://github.com/numba/numba/blob/39271156a52c58ca18b15aebcb1c85e4a07e49ed/numba/tests/pdlike_usecase.py
    def __init__(self, dtype, ndim, layout, aligned):
        layout = 'C'
        type_name = "Xrange_array"
        name = "%s(%s, %sd, %s)" % (type_name, dtype, ndim, layout)
        super().__init__(dtype, ndim, layout, readonly=False, name=name,
                 aligned=aligned)

if xrange_typing:
    numba.extending.register_model(XrangeArray)(
        numba.extending.models.ArrayModel)

    @numba.extending.typeof_impl.register(fsx.Xrange_array)
    def typeof_xrangearray(val, c):
        arrty = numba.extending.typeof_impl(np.asarray(val), c)
        return XrangeArray(arrty.dtype, arrty.ndim, arrty.layout, arrty.aligned)
    xrange_type = XrangeArray


# Default typing for integer addition is int64(int32, int32)
# see https://numba.pydata.org/numba-doc/latest/proposals/integer-typing.html
# The typing of Python int values used as function arguments doesn’t change,
# as it works satisfyingly and doesn’t surprise the user.
# Here we need proper int32 addition, substraction...

@numba.extending.intrinsic
def add_int32(typingctx, src1, src2):
    # check for accepted types
    if (src1 == numba.int32) and (src2 == numba.int32):
        # create the expected type signature
        # result_type = types.int32
        sig = types.int32(types.int32, types.int32)
        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            # https://llvmlite.readthedocs.io/en/latest/
            (a ,b) = args
            return builder.add(a, b)
        return sig, codegen

@numba.extending.intrinsic
def neg_int32(typingctx, src1):
    if (src1 == numba.int32):
        sig = types.int32(types.int32)
        def codegen(context, builder, signature, args):
            (a,) = args
            return builder.neg(a)
        return sig, codegen

@numba.extending.intrinsic
def sub_int32(typingctx, src1, src2):
    if (src1 == numba.int32) and (src2 == numba.int32):
        sig = types.int32(types.int32, types.int32)
        def codegen(context, builder, signature, args):
            (a, b) = args
            return builder.sub(a, b)
        return sig, codegen

@numba.extending.intrinsic
def sdiv_int32(typingctx, src1, src2):
    if (src1 == numba.int32) and (src2 == numba.int32):
        sig = types.int32(types.int32, types.int32)
        def codegen(context, builder, signature, args):
            (a, b) = args
            return builder.sdiv(a, b)
        return sig, codegen


@overload(operator.setitem)
def extended_setitem_tuple(arr, idx, val):
    """
    Usage : if arr is an Xrange_array, then one will be able to do
        arr[i] = XrangeScalarType(mantissa, exp)
    """
    if (isinstance(val, XrangeScalarType) and isinstance(arr, xrange_type)):
        def impl(arr, idx, val):
            arr[idx]["mantissa"] = val.mantissa
            arr[idx]["exp"] = val.exp
        return impl

@overload_attribute(numba.types.Record, "is_complex")
def is_complex(rec):
    dtype = rec.fields["mantissa"][0]
    is_complex = (dtype in numba_complex_types)
    def impl(rec):
        return is_complex
    return impl

@overload(operator.neg)
def extended_neg(op0):
    """ Change sign of a Record field """
    if (op0 in xr_types):
        def impl(op0):
            return XrangeScalar(-op0.mantissa, op0.exp)
        return impl
    else:
        raise TypingError("datatype not accepted {}".format(
            op0))

@overload(operator.add)#, debug=True)
def extended_add(op0, op1):
    """ Add 2 Record fields """
    if (op0 in xr_types) and (op1 in xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1.mantissa, op1.exp)
            return XrangeScalar(m0_out + m1_out, exp_out)
        return impl
    elif (op0 in xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1, numba.int32(0))
            return XrangeScalar(m0_out + m1_out, exp_out)
        return impl
    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0, numba.int32(0), op1.mantissa, op1.exp)
            return XrangeScalar(m0_out + m1_out, exp_out)
        return impl
    else:
        raise TypingError("datatype not accepted xr_add({}, {})".format(
            op0, op1))

@overload(operator.sub)
def extended_sub(op0, op1):
    """ Substract 2 Record fields """
    if (op0 in xr_types) and (op1 in xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1.mantissa, op1.exp)
            return XrangeScalar(m0_out - m1_out, exp_out)
        return impl
    elif (op0 in xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1, numba.int32(0))
            return XrangeScalar(m0_out - m1_out, exp_out)
        return impl
    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0, numba.int32(0), op1.mantissa, op1.exp)
            return XrangeScalar(m0_out - m1_out, exp_out)
        return impl
    else:
        raise TypingError("datatype not accepted xr_sub({}, {})".format(
            op0, op1))

@generated_jit(nopython=True)
def _need_renorm(m):
    """
    Returns True if abs(exponent) is above a given threshold
    """
    threshold = 100  # as 2.**100 = 1.e30
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
    if (op0 in xr_types) and (op1 in xr_types):
        def impl(op0, op1):
            mul = op0.mantissa * op1.mantissa
            # Need to avoid casting to int64... ! 
            exp = add_int32(op0.exp, op1.exp)
            if _need_renorm(mul):
                return XrangeScalar(*_normalize(mul, exp))
            return XrangeScalar(mul, exp)
        return impl
    elif (op0 in xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            mul = op0.mantissa * op1
            if _need_renorm(mul):
                return XrangeScalar(*_normalize(mul, op0.exp))
            return XrangeScalar(mul, op0.exp)
        return impl
    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            mul = op0 * op1.mantissa
            if _need_renorm(mul):
                return XrangeScalar(*_normalize(mul, op1.exp))
            return XrangeScalar(mul, op1.exp)
        return impl
    else:
        raise TypingError("datatype not accepted xr_mul({}, {})".format(
            op0, op1))

@overload(operator.truediv)
def extended_truediv(op0, op1):
    """ Divide 2 Record fields """
    if (op0 in xr_types) and (op1 in xr_types):
        def impl(op0, op1):
            mul = op0.mantissa / op1.mantissa
            exp = sub_int32(op0.exp, op1.exp)
            if _need_renorm(mul):
                return XrangeScalar(*_normalize(mul, exp))
            return XrangeScalar(mul, exp)
        return impl
    elif (op0 in xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            mul = op0.mantissa / op1
            if _need_renorm(mul):
                return XrangeScalar(*_normalize(mul, op0.exp))
            return XrangeScalar(mul, op0.exp)
        return impl
    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            mul = op0 / op1.mantissa
            exp = neg_int32(op1.exp)
            if _need_renorm(mul):
                return XrangeScalar(*_normalize(mul, exp))
            return XrangeScalar(mul, exp)
        return impl
    else:
        raise TypingError("datatype not accepted xr_mul({}, {})".format(
            op0, op1))

def extended_overload(compare_operator):
    @overload(compare_operator)
    def extended_compare(op0, op1): 
        """ Compare 2 Record fields """
        if (op0 in xr_types) and (op1 in xr_types):
            def impl(op0, op1):
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0.mantissa, op0.exp, op1.mantissa, op1.exp)
                return compare_operator(m0_out, m1_out)
            return impl
        elif (op0 in xr_types) and (op1 in numba_base_types):
            def impl(op0, op1):
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0.mantissa, op0.exp, op1, 0)
                return compare_operator(m0_out, m1_out)
            return impl
        elif (op0 in numba_base_types) and (op1 in xr_types):
            def impl(op0, op1):
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0, 0, op1.mantissa, op1.exp)
                return compare_operator(m0_out, m1_out)
            return impl
        else:
            raise TypingError("datatype not accepted xr_add({}, {})".format(
                op0, op1))

for compare_operator in (
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt
        ):
    extended_overload(compare_operator)

@overload(np.sqrt)
def extended_sqrt(op0):
    """ sqrt of a Record fields """
    if op0 in xr_types:
        def impl(op0):
            exp = op0.exp
            if exp % 2:
                exp = sdiv_int32(sub_int32(exp, numba.int32(1)),
                                 numba.int32(2)) # // 2
                m = np.sqrt(op0.mantissa * 2.)
            else:
                exp = sdiv_int32(exp, numba.int32(2)) # // 2
                m = np.sqrt(op0.mantissa)
            return XrangeScalar(m, exp)
        return impl
    else:
        raise TypingError("Datatype not accepted xr_sqrt({})".format(
            op0))

@generated_jit(nopython=True)
def extended_abs2(op0):
    """ square of abs of a Record field """
    if op0 in xr_types:
        def impl(op0):
            return XrangeScalar(np.square(op0.mantissa), 2 * op0.exp)
        return impl
    else:
        raise TypingError("Datatype not accepted xr_sqrt({})".format(
            op0))

@overload(np.abs)
def extended_abs(op0):
    """ abs of a Record field """
    if op0 in xr_types:
        def impl(op0):
            return XrangeScalar(np.abs(op0.mantissa), op0.exp)
        return impl
    else:
        raise TypingError("Datatype not accepted xr_sqrt({})".format(
            op0))


@generated_jit(nopython=True)
def _normalize(m, exp):
    """ Returns a normalized couple """
    # Implementation for float
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

@njit(types.Tuple((numba.float64, numba.int32))(numba.float64,))
def _frexp(m):
    """ Faster unsafe equivalent for math.frexp(m) """
    # https://github.com/numba/numba/issues/3763
    # https://llvm.org/docs/LangRef.html#bitcast-to-instruction
    bits = numba.cpython.unsafe.numbers.viewer(m, numba.int64)
    m = numba.cpython.unsafe.numbers.viewer(
        (bits & 0x8000000000000000) # signe
        + (0x3ff << 0x34) # exposant (bias) hex(1023) = 0x3ff hex(52) = 0x34
        + (bits & 0xfffffffffffff), numba.float64)
    exp = (((bits >> 52)) & 0x7ff) - 0x3ff # numba.int32 ??
    return m, exp

@njit(types.Tuple((numba.float64, numba.int32))(numba.float64, numba.int32))
def _normalize_real(m, exp):
    """ Returns a normalized couple """
    if m == 0.:
        return (m, numba.int32(0))
    else:
        nm, nexp = _frexp(m)
        return (nm, exp + nexp)

@njit(numba.float64(numba.float64, numba.int32))
def _exp2_shift(m, shift):
    """ Faster unsafe equivalent for math.ldexp(m, shift) """
    # https://github.com/numba/numba/issues/3763
    # https://llvm.org/docs/LangRef.html#bitcast-to-instruction
    bits = numba.cpython.unsafe.numbers.viewer(m, numba.int64)
    exp = max(((bits >> 0x34) & 0x7ff) + shift, 0)
    return numba.cpython.unsafe.numbers.viewer(
        (bits & 0x8000000000000000)
        + (exp << 0x34)
        + (bits & 0xfffffffffffff), numba.float64)

@generated_jit(nopython=True)
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
                co_m0 = _exp2_shift(co_m0, d_exp)
                exp = exp1
            elif (exp0 > exp1):
                co_m1 = _exp2_shift(co_m1, -d_exp)
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
                co_m0 = (_exp2_shift(co_m0.real, d_exp)
                         + 1j * _exp2_shift(co_m0.imag, d_exp))
                exp = exp1
            elif (exp0 > exp1):
                co_m1 = (_exp2_shift(co_m1.real, -d_exp)
                         + 1j * _exp2_shift(co_m1.imag, -d_exp))
                exp = exp0
            else: # exp0 == exp1
                exp = exp0
            return (co_m0, co_m1, exp)
    else:
        raise TypingError("datatype not accepted {}{}".format(m0, m1))
    return impl

@overload_method(numba.types.Record, "normalize")
def normalize(rec):
    """ Normalize in-place a xr Record """
    dtype = rec.fields["mantissa"][0]
    # Implementation for float
    if (dtype in numba_float_types):
        def impl(rec):
            m = rec.mantissa
            if m == 0.:
                rec.exp = numba.int32(0)
            else:
                nm, nexp = _frexp(rec.mantissa) 
                rec.exp += nexp
                rec.mantissa = nm
    # Implementation for complex
    elif (dtype in numba_complex_types):
        def impl(rec):
            m = rec.mantissa
            if m == 0.:
                rec.exp = numba.int32(0)
            else:
                rec.mantissa, rec.exp = _normalize(m, rec.exp)
    else:
        raise TypingError("datatype not accepted {}".format(dtype))
    return impl

# Implementing the Xrange_polynomial class
# https://numba.pydata.org/numba-doc/latest/proposals/extension-points.html

class XrangePolynomialType(types.Type): # ArrayCompatible):
    def __init__(self, coeff, cutdeg):
        
        super().__init__(name="XrangePolynomial")#.format(base_type))

        

@typeof_impl.register(fsx.Xrange_polynomial)
def typeof_index(val, c):
    return XrangePolynomialType()

@register_model(fsx.Xrange_polynomial)
class XrangePolynomialModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        print("###", dmm, fe_type)
        members = [
            ('values', fe_type.as_array),
            ('cutdeg', numba.int64) # Not that we need, but int32 is painful
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

make_attribute_wrapper(XrangePolynomialType, 'coeffs', 'coeffs')

@lower_builtin(fsx.Xrange_polynomial, types.Array, types.Integer)
def impl_xrange_polynomial(context, builder, sig, args):
    typ = sig.return_type
    coeff, cutdeg = args
    xrange_polynomial = cgutils.create_struct_proxy(typ)(context, builder)
    xrange_polynomial.coeff = coeff
    xrange_polynomial.cutdeg = cutdeg
    return xrange_polynomial._getvalue()


@numba.njit
def test_poly(pol):
    p = fsx.Xrange_polynomial(pol, 2)

if __name__ == "__main__":
    pol = fsx.Xrange_array(["1.e2", "3.14", "2.0"])
#    tup = (1., numba.int32(1))
#    numba_test_tup2(tup)
##    print("numba_xr_types", numba_xr_types)
#
#    
#    arr = fsx.Xrange_array(["1.e70", "3.14", "2.0"])
#    arr = arr * (1. + 1.j)
#    out = arr.copy()
#    numba_test_neg(arr, out)
#    print("out", out)
#    
#    arr = fsx.Xrange_array(["1.e2", "3.14", "2.0"])
#    numba_test_add2(arr)
#    print("sum", arr[2])
#    arr *= (1. + 1.j)
#    numba_test_expr(arr)
#    print("expr", arr[0])
#    
##    tup = (1.23, np.int32(7))
##    numba_test_tup(tup)
#
#    
##    print("is xr", numba_test_is_xr(arr))
#    debug_code = False
#    if debug_code:
#        f = generated_jit(extended_mul)
#        f(np.asarray(arr)[0], np.asarray(arr)[1])
#        print(f.inspect_types(pretty=True))
