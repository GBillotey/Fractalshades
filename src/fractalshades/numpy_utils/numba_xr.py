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
    box,
    unbox,
    NativeValue
)
from numba.core.imputils import impl_ret_borrowed

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

numba_float_types = (numba.float64,)
numba_complex_types = (numba.complex128,)
numba_base_types = numba_float_types + numba_complex_types

def numpy_xr_type(base_type):
    return np.dtype([("mantissa", base_type), ("exp", np.int32)], align=False)
    

def numba_xr_type(base_type):
    """ Return the numba "extended" Record type for the 2 implemented base type
    float64, complex128 """
    return numba.from_dtype(numpy_xr_type(base_type))

numba_xr_types = tuple(numba_xr_type(dt) for dt in (np.float64, np.complex128))
#numba_xr_dict = {numba_xr_type(v): v for v in (np.float64, np.complex128)}


# Create a datatype for temporary manipulation of Xrange_array items. 
# This datatype will only be used in numba jitted functions, so we do not
# expose a full python implementation (e.g, boxing, unboxing)

class Xrange_scalar():
    def __init__(self, mantissa, exp):
        self.mantissa = mantissa
        self.exp = exp

class Xrange_scalar_Type(types.Type):
    def __init__(self, base_type):
        self.base_type = base_type
        super().__init__(name="{}_Xrange_scalar".format(base_type))
        self.base_type = base_type

@type_callable(Xrange_scalar)
def type_extended_item(context):
    def typer(mantissa, exp):
        if (mantissa in numba_base_types) and (exp == numba.int32):
            return Xrange_scalar_Type(mantissa)
    return typer

@register_model(Xrange_scalar_Type)
class Xrange_scalar_Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('mantissa', fe_type.base_type),
            ('exp', numba.int32),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

for attr in ('mantissa', 'exp'):
    make_attribute_wrapper(Xrange_scalar_Type, attr, attr)

@lower_builtin(Xrange_scalar, types.Number, types.Integer)
def impl_xrange_scalar(context, builder, sig, args):
    typ = sig.return_type
    mantissa, exp = args
    xrange_scalar = cgutils.create_struct_proxy(typ)(context, builder)
    xrange_scalar.mantissa = mantissa
    xrange_scalar.exp = exp
    return xrange_scalar._getvalue()

# We will support operation between numba_xr_types and Xrange_scalar instances
scalar_xr_types = tuple(Xrange_scalar_Type(dt) for dt in numba_base_types)
xr_types = numba_xr_types + scalar_xr_types

def is_xr_type(val):
    if isinstance(val, Xrange_scalar_Type):
        return (val.base_type in numba_base_types)
    if isinstance(val, numba.types.Record):
        return (
            len(val) == 2
            and "mantissa" in val.fields
            and "exp" in val.fields
            and val.fields["mantissa"][0] in numba_base_types)


@overload_attribute(numba.types.Record, "is_xr")
def is_xr(rec):
    ret = tuple(rec.fields.keys()) == ("mantissa", "exp")
    def impl(rec):
        return ret
    return impl


# Dedicated typing for Xrange_array adds some overhead with little benefit
# -> not implemented by default (passthrough as types.Array)
xrange_typing = False
xrange_arty = types.Array

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
    xrange_arty = XrangeArray


# The default numba typing for integer addition is int64(int32, int32) (?? ...)
# See https://numba.pydata.org/numba-doc/latest/proposals/integer-typing.html
# 'The typing of Python int values used as function arguments doesn’t change,
# as it works satisfyingly and doesn’t surprise the user.'
# Here we will need proper int32 addition, substraction...

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
        arr[i] = Xrange_scalar_Type(mantissa, exp)
    """
    if isinstance(arr, xrange_arty) and (val in xr_types):
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
    if (op0 in xr_types): #is_xr_type(op0)):# in xr_types):
        def impl(op0):
            return Xrange_scalar(-op0.mantissa, op0.exp)
        return impl
    else:
#        print(op0, op0 in xr_types, xr_types)
        raise TypingError("datatype not accepted {}".format(
            op0))

@overload(operator.add)#, debug=True)
def extended_add(op0, op1):
    """ Add 2 Record fields """
    if (op0 in xr_types) and (op1 in xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1.mantissa, op1.exp)
            return Xrange_scalar(m0_out + m1_out, exp_out)
        return impl
    elif (op0 in xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1, numba.int32(0))
            return Xrange_scalar(m0_out + m1_out, exp_out)
        return impl
    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0, numba.int32(0), op1.mantissa, op1.exp)
            return Xrange_scalar(m0_out + m1_out, exp_out)
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
            return Xrange_scalar(m0_out - m1_out, exp_out)
        return impl
    elif (op0 in xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0.mantissa, op0.exp, op1, numba.int32(0))
            return Xrange_scalar(m0_out - m1_out, exp_out)
        return impl
    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            m0_out, m1_out, exp_out = _coexp_ufunc(
                op0, numba.int32(0), op1.mantissa, op1.exp)
            return Xrange_scalar(m0_out - m1_out, exp_out)
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
                return Xrange_scalar(*_normalize(mul, exp))
            return Xrange_scalar(mul, exp)
        return impl
    elif (op0 in xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            mul = op0.mantissa * op1
            if _need_renorm(mul):
                return Xrange_scalar(*_normalize(mul, op0.exp))
            return Xrange_scalar(mul, op0.exp)
        return impl
    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            mul = op0 * op1.mantissa
            if _need_renorm(mul):
                return Xrange_scalar(*_normalize(mul, op1.exp))
            return Xrange_scalar(mul, op1.exp)
        return impl
    else:
#        print(op0 in numba_base_types, op0 in xr_types)
#        print(op1 in numba_base_types, op1 in xr_types)
        # TypingError: datatype not accepted xr_mul(float64_Xrange_scalar, Record(mantissa[type=float64;offset=0],exp[type=int32;offset=8];12;False))
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
                return Xrange_scalar(*_normalize(mul, exp))
            return Xrange_scalar(mul, exp)
        return impl
    elif (op0 in xr_types) and (op1 in numba_base_types):
        def impl(op0, op1):
            mul = op0.mantissa / op1
            if _need_renorm(mul):
                return Xrange_scalar(*_normalize(mul, op0.exp))
            return Xrange_scalar(mul, op0.exp)
        return impl
    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            mul = op0 / op1.mantissa
            exp = neg_int32(op1.exp)
            if _need_renorm(mul):
                return Xrange_scalar(*_normalize(mul, exp))
            return Xrange_scalar(mul, exp)
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
            return Xrange_scalar(m, exp)
        return impl
    else:
        raise TypingError("Datatype not accepted xr_sqrt({})".format(
            op0))

@generated_jit(nopython=True)
def extended_abs2(op0):
    """ square of abs of a Record field """
    if op0 in xr_types:
        def impl(op0):
            return Xrange_scalar(np.square(op0.mantissa), 2 * op0.exp)
        return impl
    else:
        raise TypingError("Datatype not accepted xr_sqrt({})".format(
            op0))

@overload(np.abs)
def extended_abs(op0):
    """ abs of a Record field """
    if op0 in xr_types:
        def impl(op0):
            return Xrange_scalar(np.abs(op0.mantissa), op0.exp)
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

# 
# Implementing the Xrange_polynomial class in numba
# https://numba.pydata.org/numba-doc/latest/proposals/extension-points.html
# 

class Xrange_polynomial_Type(types.Type):
    def __init__(self, dtype, cutdeg):
        self.dtype = dtype
        self.np_base_type = numba.np.numpy_support.as_dtype(
                dtype.fields["mantissa"][0])
#        self.cutdeg = cutdeg
        self.coeffs = types.Array(dtype, 1, 'C')
#        print("numba_xr_dict", numba_xr_dict)
#        print("numba_xr_dict", dtype.fields["mantissa"][0])
        # The name must be unique if the underlying model is different
        super().__init__(name="{}_Xrange_polynomial".format(
                dtype.fields["mantissa"][0])) # str(dtype.fields["mantissa"][0])))

#    @property
#    def as_array(self):
#        return self.coeffs
#
#    def copy(self, dtype=None, ndim=1, layout='C'):
#        assert ndim == 1
#        assert layout == 'C'
#        if dtype is None:
#            dtype = self.dtype
#        return type(self)(dtype, self.index)

@typeof_impl.register(fsx.Xrange_polynomial)
def typeof_xrange_polynomial(val, c):
    coeffs_arrty = typeof_impl(val.coeffs, c)
    return Xrange_polynomial_Type(coeffs_arrty.dtype, val.cutdeg)

@type_callable(fsx.Xrange_polynomial)
def type_xrange_polynomial(context):
    def typer(coeffs, cutdeg):
        if (isinstance(coeffs, types.Array)
              and (coeffs.dtype in numba_xr_types)
              and isinstance(cutdeg, types.Integer)):
            return Xrange_polynomial_Type(coeffs.dtype, cutdeg)
    return typer

@register_model(Xrange_polynomial_Type)
class XrangePolynomialModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('coeffs', fe_type.coeffs),
            ('cutdeg', numba.int64) # Not that we need, but int32 is painful
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

make_attribute_wrapper(Xrange_polynomial_Type, 'coeffs', 'coeffs')
make_attribute_wrapper(Xrange_polynomial_Type, 'cutdeg', 'cutdeg')

@lower_builtin(fsx.Xrange_polynomial, types.Array, types.Integer)
def impl_xrange_polynomial_constructor(context, builder, sig, args):
    typ = sig.return_type
    coeffs, cutdeg = args
    xrange_polynomial = cgutils.create_struct_proxy(typ)(context, builder)
    xrange_polynomial.coeffs = coeffs
    #  We do not copy !! following implementation in python
    xrange_polynomial.cutdeg = cutdeg
    return impl_ret_borrowed(context, builder, typ,
                             xrange_polynomial._getvalue())

@unbox(Xrange_polynomial_Type)
def unbox_xrange_polynomial(typ, obj, c):
    """
    Convert a fsx.Xrange_polynomial object to a native xrange_polynomial
    structure. """
    coeffs_obj = c.pyapi.object_getattr_string(obj, "coeffs")
    cutdeg_obj = c.pyapi.object_getattr_string(obj, "cutdeg")
    xrange_polynomial = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    xrange_polynomial.cutdeg = c.pyapi.long_as_longlong(cutdeg_obj) 
    xrange_polynomial.coeffs = c.unbox(typ.coeffs, coeffs_obj).value
    c.pyapi.decref(coeffs_obj)
    c.pyapi.decref(cutdeg_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(xrange_polynomial._getvalue(), is_error=is_error)

@box(Xrange_polynomial_Type)
def box_xrange_polynomial(typ, val, c):
    """
    Convert a native xrange_polynomial structure to a 
    fsx.Xrange_polynomial object """
    xrange_polynomial = cgutils.create_struct_proxy(typ
        )(c.context, c.builder, value=val)
    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(
        fsx.Xrange_polynomial))
    coeffs_obj = c.box(typ.coeffs, xrange_polynomial.coeffs)
    cutdeg_obj = c.pyapi.long_from_longlong(xrange_polynomial.cutdeg)
    xrange_polynomial_obj = c.pyapi.call_function_objargs(
        classobj, (coeffs_obj, cutdeg_obj))
    c.pyapi.decref(classobj)
    c.pyapi.decref(coeffs_obj)
    c.pyapi.decref(cutdeg_obj)
    return xrange_polynomial_obj

# 
# Implementing operations for Xrange_polynomial 
# 
@overload(operator.neg)
def poly_neg(op0):
    """ Copy of a polynomial with sign changed """
    if isinstance(op0, Xrange_polynomial_Type):
        def impl(op0):
            # assert op0.coeffs.size == op0.cutdeg + 1 
            coeffs = op0.coeffs
            new_coeffs = np.empty_like(op0.coeffs)
            for i in range(coeffs.size):
                new_coeffs[i] = - coeffs[i]
            return fsx.Xrange_polynomial(new_coeffs, op0.cutdeg)
        return impl

@overload(operator.add)
def poly_add(op0, op1):
    """ Add 2  polynomials with sign changed """
    if (isinstance(op0, Xrange_polynomial_Type)
            and isinstance(op0, Xrange_polynomial_Type)
            ):
        # There is no lowering implementation for a structured dtype ; so
        # we initiate a template of length 1 for the compilation.
        base_dtres = np.result_type(op0.np_base_type,
                                    op1.np_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.empty([1], dtype=res_dtype)

        def impl(op0, op1):
            assert op0.cutdeg == op1.cutdeg
            cutdeg = op0.cutdeg
            coeffs0 = op0.coeffs
            coeffs1 = op1.coeffs

            res_len = min(max(coeffs0.size, coeffs1.size), cutdeg + 1)
            r01 = min(min(coeffs0.size, coeffs1.size), cutdeg + 1)
            r0 = min(coeffs0.size, cutdeg + 1)
            r1 = min(coeffs1.size, cutdeg + 1)

            new_coeffs = res_template.repeat(res_len)
            for i in range(r01):
                new_coeffs[i] = coeffs0[i] + coeffs1[i]
            for i in range(r01, r0):
                new_coeffs[i] = coeffs0[i]
            for i in range(r01, r1):
                new_coeffs[i] = coeffs1[i]

            return fsx.Xrange_polynomial(new_coeffs, op0.cutdeg)
        return impl

@overload_method(Xrange_polynomial_Type, '__call__')
def xrange_polynomial_call(poly, val):
    # Implementation for scalars
    if (val in xr_types):
        if isinstance(val, Xrange_scalar_Type):
            base_type = val.base_type
        else:
            base_type = val.fields["mantissa"][0]
        base_dtres = numba.np.numpy_support.as_dtype(base_type)
        base_dtres = np.result_type(base_dtres, poly.np_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.empty([1], dtype=res_dtype)

        def call_impl(poly, val):
            res = res_template.repeat(1)
            coeffs = poly.coeffs
            n = coeffs.size
            res[0] = coeffs[n - 1]
            for i in range(2, coeffs.size + 1):
                res[0] = coeffs[n - i] + res[0] * val
            return res
        return call_impl
    # Implementation for arrays
    elif isinstance(val , xrange_arty):
        base_type = val.dtype.fields["mantissa"][0]
        base_dtres = numba.np.numpy_support.as_dtype(base_type)
        base_dtres = np.result_type(base_dtres, poly.np_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.empty([1], dtype=res_dtype)

        def call_impl(poly, val):
            res_len = val.size
            res = res_template.repeat(res_len)
            coeffs = poly.coeffs
            n = coeffs.size
            for j in range(res_len):
                res[j] = coeffs[n - 1]
                for i in range(2, coeffs.size + 1):
                    res[j] = coeffs[n - i] + res[j] * val[j]
            return res
        return call_impl
        

            
@overload_method(Xrange_polynomial_Type, 'deriv')
def xrange_polynomial_deriv(poly):
    # Call self as a function.
    def call_impl(poly, val, k=1.):
        coeffs = poly.coeffs
        n = coeffs.size
        deriv_coeffs = coeffs[1:] * np.arange(1, n)
        if k != 1.:
            mul = 1.
            for i in range(n - 1):
                deriv_coeffs[i] = deriv_coeffs[i] * mul
                mul *= k
        return fsx.Xrange_polynomial(deriv_coeffs, cutdeg=poly.cutdeg)

    return call_impl

#    def __call__(self, arg):
#        """ Call self as a function.
#        """
#        if not isinstance(arg, Xrange_array):
#            arg = Xrange_array(np.asarray(arg))
#
#        res_dtype = np.result_type(arg._mantissa, self.coeffs._mantissa)
#        res = Xrange_array.empty(arg.shape, dtype=res_dtype)
#        res.fill(self.coeffs[-1])
#
#        for i in range(2, self.coeffs.size + 1):
#            res = self.coeffs[-i] + res * arg
#        return res
    
#    def deriv(self, k=1.):
#        l = self.coeffs.size
#        coeffs = self.coeffs[1:] * np.arange(1, l)
#        if k != 1.:
#            mul = 1.
#            for i in range(l-1):
#                coeffs[i] *= mul
#                mul *= k
#        return Xrange_polynomial(coeffs, cutdeg=self.cutdeg)



#    @staticmethod
#    def _add(ufunc, inputs, cutdeg, out=None):
#        """ Add or Subtract 2 Xrange_polynomial """
#        op0, op1 = inputs
#        res_len = min(max(op0.size, op1.size), cutdeg + 1)
#        op0_len = min(op0.size, res_len)
#        op1_len = min(op1.size, res_len)
#
#        dtype=np.result_type(op0._mantissa, op1._mantissa)
#        res = Xrange_array(np.zeros([res_len], dtype=dtype))
#
#        res[:op0_len] += op0[:op0_len]
#        if ufunc is np.add:
#            res[:op1_len] += op1[:op1_len]
#        elif ufunc is np.subtract: 
#            res[:op1_len] -= op1[:op1_len]
#        return Xrange_polynomial(res, cutdeg=cutdeg)


@numba.njit
def test_poly(pol):
    print("in numba")
#    print("nb", pol, pol.coeffs.dtype)
    # pol.coeffs[1] = Xrange_scalar(45678., numba.int32(0))
    coeff2 = pol.coeffs.copy() # * Xrange_scalar(2., numba.int32(0))
#    print("nb", coeff2)
    for i in range(len(coeff2)):
        coeff2[i] = coeff2[i] * coeff2[i] 
    p2 = - fsx.Xrange_polynomial(coeff2, 2)
#    print("init", p2.cutdeg, p2.coeffs)
    print("p2 init", p2.cutdeg, p2.coeffs)
    return p2

@numba.njit
def test_poly_call(pol):
    print("in numba")
#    print("nb", pol, pol.coeffs.dtype)
    # pol.coeffs[1] = Xrange_scalar(45678., numba.int32(0))
#    coeff2 = pol.coeffs.copy() # * Xrange_scalar(2., numba.int32(0))
##    print("nb", coeff2)
#    for i in range(len(coeff2)):
#        coeff2[i] = coeff2[i] * coeff2[i] 
#    p2 = - fsx.Xrange_polynomial(coeff2, 2)
##    print("init", p2.cutdeg, p2.coeffs)
#    print("p2 init", p2.cutdeg, p2.coeffs)
    return pol.__call__(Xrange_scalar(2., np.int32(0)))


@numba.njit
def test_polyadd(pol1, pol2):
    print("in numba")
    return pol1 + pol2

if __name__ == "__main__":
#    arr0 = fsx.Xrange_array(["1.e100", "3.14", "2.0"]) #* (1. + 1j)
#    pol0 = fsx.Xrange_polynomial(arr0, 2)
#    print(pol0.coeffs)
#    print(np.asarray(pol0.coeffs).size)
#    print(pol0.coeffs.size, pol0.cutdeg + 1)
#    p_neg = test_poly(pol0)
#    print("p_neg", p_neg)

    arr0 = fsx.Xrange_array(["1.", "1.", "1.0"]) #* (1. + 1j)
    pol0 = fsx.Xrange_polynomial(arr0, 2)
    res = test_poly_call(pol0)
    print(res.view(fsx.Xrange_array))
    
#    arr1 = fsx.Xrange_array(["1.e100", "3.14", "2.0"]) #* (1. + 1j)
#    pol1 = fsx.Xrange_polynomial(arr1, 2)
#    
#    res =  test_polyadd(pol0, pol1)
#    print("res", res)
#    
#    arr0 = fsx.Xrange_array(["1.e100", "3.14", "2.0", "5.0"]) #* (1. + 1j)
#    pol0 = fsx.Xrange_polynomial(arr0, 3)
#    arr1 = fsx.Xrange_array(["1.e100", "3.14", "2.0", "6.4"]) #* (1. + 1j)
#    pol1 = fsx.Xrange_polynomial(arr1, 3)
#    
#    res =  test_polyadd(pol0, pol1)
#    print("res", res)