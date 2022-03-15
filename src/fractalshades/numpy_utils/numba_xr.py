# -*- coding: utf-8 -*-
import numpy as np
#import numbers
#import re

import numba
from numba.core import types, cgutils # utils, typing, errors, extending, sigutils
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
#    lower_getattr,
#    lower_setattr,
    typeof_impl,
    type_callable,
    models,
    register_model,
    make_attribute_wrapper,
    box,
    unbox,
    NativeValue
)
from numba.core.imputils import impl_ret_borrowed#, lower_setattr, lower_getattr
from llvmlite import ir

# from numba.core.typing.templates import (AttributeTemplate, infer_getattr)
#                                         AbstractTemplate, 
#                                         signature, Registry, infer_getattr)



import fractalshades.numpy_utils.xrange as fsx
# import math
import operator

"""
Its purpose is to allow the use of Xrange_arrays, polynomials and SA objects
inside jitted functions by defining mirrored low-level implementations.

By default, Numba will treat all numpy.ndarray subtypes as if they were of the
base numpy.ndarray type. On one side, ndarray subtypes can easily use all of
the support that Numba has for ndarray methods ; on the other side it is not
possible to fully customise the behavior. (This is likely to change in future
release of Numba, see https://github.com/numba/numba/pull/6148)

The workaround followed here is to provide ad-hoc implementation at datatype
level (in numba langage, for our specific numba.types.Record types). User code
in jitted function should fully expand the loops to work on individual array
elements - indeed numba is made for this.

As the extra complexity is not worth it, we drop support for float32, complex64
in numba: only float64, complex128 mantissa are currently supported.

NOte:
    https://numba.pydata.org/numba-doc/latest/proposals/extension-points.html

/!\ This submodule has side effects at import time (due to its use of
numba operators overload)

See https://github.com/pygae/clifford

Note : An alternative approach without numba:
https://numpy.org/doc/stable/user/c-info.ufunc-tutorial.html
https://jiffyclub.github.io/numpy/user/c-info.ufunc-tutorial.html

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
numba_real_xr_types = tuple(numba_xr_type(dt) for dt in (np.float64,))

# Create a datatype for temporary manipulation of Xrange_array items. 
# This datatype will only be used in numba jitted functions, so we do not
# expose a full python implementation (e.g, unboxing).
# However Xrange_scalar can be boxed to a python Xrange_array of shape (,) and
# then used in pure python code.

class Xrange_scalar():
    def __init__(self, mantissa, exp):
        self.mantissa = mantissa
        self.exp = exp

class Xrange_scalar_Type(types.Type):
    def __init__(self, base_type):
        super().__init__(name="{}_Xrange_scalar".format(base_type))
        self.base_type = base_type
        self.np_base_type = numba.np.numpy_support.as_dtype(base_type)

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

make_attribute_wrapper(Xrange_scalar_Type, 'mantissa', 'mantissa')
make_attribute_wrapper(Xrange_scalar_Type, 'exp', 'exp')

@lower_builtin(Xrange_scalar, types.Number, types.Integer)
def impl_xrange_scalar(context, builder, sig, args):
    typ = sig.return_type
    mantissa, exp = args
    xrange_scalar = cgutils.create_struct_proxy(typ)(context, builder)
    xrange_scalar.mantissa = mantissa
    xrange_scalar.exp = exp
    return xrange_scalar._getvalue()


#  @unbox(Xrange_scalar_Type) :
# Not implemented, use idiom : 
# scalar = fsxn.to_Xrange_scalar(arr.repeat(1))



@box(Xrange_scalar_Type)
def box_xrange_scalar(typ, val, c):
    """
    Convert a native xrange_scalar structure to a 
    fsx.Xrange_array object of shape (,)"""
    # https://github.com/numba/numba/blob/2776e1a7cf49aeb513e0319fe4a94a12836a995b/numba/core/pythonapi.py
    # See : call_method(self, callee, method, objargs=())
    # obj.name(arg1, arg2, ...)
    xrange_scalar = cgutils.create_struct_proxy(typ
        )(c.context, c.builder, value=val)
    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(
        fsx.Xrange_array))
    # Mantissa : float 64 or complex 128
    if typ is Xrange_scalar_Type(numba.float64):
        mantissa_obj = c.pyapi.float_from_double(xrange_scalar.mantissa)
    else:
        assert typ is Xrange_scalar_Type(numba.complex128)
        cval = c.context.make_complex(
                c.builder, numba.complex128, value=xrange_scalar.mantissa)
        mantissa_obj = c.pyapi.complex_from_doubles(cval.real, cval.imag)
    exp_obj = c.pyapi.long_from_signed_int(xrange_scalar.exp)
    xrange_obj = c.pyapi.call_method(
        classobj, "xr_scalar", (mantissa_obj, exp_obj))
    c.pyapi.decref(classobj)
    c.pyapi.decref(mantissa_obj)
    c.pyapi.decref(exp_obj)
    return xrange_obj



@numba.njit
def zero():
    return Xrange_scalar(0., numba.int32(0))
@numba.njit
def czero():
    return Xrange_scalar(0.j, numba.int32(0))
@numba.njit
def one():
    return Xrange_scalar(1., numba.int32(0))
@numba.njit
def two():
    return Xrange_scalar(1., numba.int32(1))

#xr_2 = 2. * fsxn.one() #Xrange_scalar(1., numba.int32(1))
    
#@overload_attribute(Xrange_scalar_Type, "np_base_type")
#def np_base_type(scalar):
#    ret = numba.np.numpy_support.as_dtype(
#                scalar.fields["mantissa"][0])
#    return lambda scalar: ret
    
#np_base_type = numba.np.numpy_support.as_dtype(
#                dtype.fields["mantissa"][0])

# We will support operation between numba_xr_types and Xrange_scalar instances
scalar_xr_types = tuple(Xrange_scalar_Type(dt) for dt in numba_base_types)
xr_types = numba_xr_types + scalar_xr_types
std_or_xr_types = xr_types + numba_base_types

scalar_real_xr_types = tuple(Xrange_scalar_Type(dt) for dt in numba_float_types)
real_xr_types = numba_real_xr_types + scalar_real_xr_types

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
            (a, b) = args
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

@numba.extending.intrinsic
def mul_int32(typingctx, src1, src2):
    if (src1 == numba.int32) and (src2 == numba.int32):
        sig = types.int32(types.int32, types.int32)
        def codegen(context, builder, signature, args):
            (a, b) = args
            return builder.mul(a, b)
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
    if (op0 in xr_types):
        def impl(op0):
            return Xrange_scalar(-op0.mantissa, op0.exp)
        return impl
    else:
        raise TypingError("datatype not accepted {}".format(
            op0))

@overload(operator.add)
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
            if _need_renorm(op1):
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0.mantissa, op0.exp, *_normalize(op1, numba.int32(0)))
            else:
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0.mantissa, op0.exp, op1, numba.int32(0))
            return Xrange_scalar(m0_out + m1_out, exp_out)
        return impl

    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            if _need_renorm(op0):
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    *_normalize(op0, numba.int32(0)), op1.mantissa, op1.exp)
            else:
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0, numba.int32(0), op1.mantissa, op1.exp)
            return Xrange_scalar(m0_out + m1_out, exp_out)
        return impl

    else:
        raise TypingError("xr_add: datatype not accepted({}, {})".format(
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
            if _need_renorm(op1):
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0.mantissa, op0.exp, *_normalize(op1, numba.int32(0)))
            else:
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0.mantissa, op0.exp, op1, numba.int32(0))
            return Xrange_scalar(m0_out - m1_out, exp_out)
        return impl

    elif (op0 in numba_base_types) and (op1 in xr_types):
        def impl(op0, op1):
            if _need_renorm(op0):
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    *_normalize(op0, numba.int32(0)), op1.mantissa, op1.exp)
            else:
                m0_out, m1_out, exp_out = _coexp_ufunc(
                    op0, numba.int32(0), op1.mantissa, op1.exp)
            return Xrange_scalar(m0_out - m1_out, exp_out)
        return impl

    else:
        raise TypingError("xr_sub: datatype not accepted ({}, {})".format(
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
        raise TypingError("xr_mul: datatype not accepted ({}, {})".format(
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
        raise TypingError("xr_div: datatype not accepted ({}, {})".format(
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
#            print(op0 in numba_base_types, op1 in xr_types)
            raise TypingError("datatype not accepted in compare({}, {})".format(
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
        raise TypingError("xr_sqrt: Datatype not accepted ({})".format(
            op0))

@generated_jit(nopython=True)
def extended_abs2(op0):
    """ square of abs of a Record field """
    if op0 in real_xr_types:
        def impl(op0):
            return Xrange_scalar(op0.mantissa ** 2,
                                 add_int32(op0.exp, op0.exp))
        return impl
    elif op0 in xr_types:
        def impl(op0):
            return Xrange_scalar(
                (op0.mantissa.real ** 2 + op0.mantissa.imag ** 2),
                add_int32(op0.exp, op0.exp))
        return impl
    else:
        raise TypingError("extended_abs2: Datatype not accepted ({})".format(
            op0))

@overload(np.abs)
def extended_abs(op0):
    """ abs of a Record field """
    if op0 in xr_types:
        def impl(op0):
            return Xrange_scalar(np.abs(op0.mantissa), op0.exp)
        return impl
    else:
        raise TypingError("extended_abs: Datatype not accepted ({})".format(
            op0))


@generated_jit(nopython=True)
def geom_mean(xr_min, xr_max):
    """ Returns the geometric mean of 2 real, positive Xrange scalars
    """
    if (xr_min in real_xr_types) and (xr_max in real_xr_types):
        def impl(xr_min, xr_max):
            xr_min_mantissa, xr_min_exp = _normalize(xr_min.mantissa, xr_min.exp)
            xr_max_mantissa, xr_max_exp = _normalize(xr_max.mantissa, xr_max.exp)
            assert xr_min_mantissa > 0.
            assert xr_max_mantissa > 0.
            exp_med, exp_rmder = numba.int32(divmod(xr_min_exp + xr_max_exp, 2))
            man_med = np.sqrt(xr_min_mantissa * xr_max_mantissa * (2. ** exp_rmder))
            return Xrange_scalar(man_med, exp_med)
        return impl
    else:
        raise TypingError("geom_mean: Expected real_xr_types({}, {})".format(
            xr_min, xr_max))


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
        (bits & 0x8000000000000000) # sign
        + (0x3ff << 0x34) # exposant (bias) hex(1023) = 0x3ff hex(52) = 0x34
        + (bits & 0xfffffffffffff), numba.float64)
    exp = (((bits >> 52)) & 0x7ff) - 0x3ff
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

# Conversion to / from Xrange_scalar in numba functions
#@numba.njit
#def to_Xrange_scalar(item):
#    return Xrange_scalar(*_normalize(item, numba.int32(0)))

@generated_jit(nopython=True)
def to_Xrange_scalar(item):
    """ Xrange Scalar from float / complex / Record """ 
    if isinstance(item, types.Number):
        def impl(item):
            return Xrange_scalar(*_normalize(item, numba.int32(0)))
    elif isinstance(item, types.Record):
        def impl(item):
            return Xrange_scalar(item.mantissa, item.exp)
    else:
        raise TypingError("to_Xrange_scalar : unexpected datatype {}".format(item))
    return impl
        

@generated_jit(nopython=True)
def to_standard(item):
    """ Returns a standard float / complex from a Xrange array or record """
    # Implementation for float
    if (item in real_xr_types):
        def impl(item):
            return np.ldexp(item.mantissa, item.exp)
    # Implementation for complex
    elif (item in xr_types):
        def impl(item):
            m = item.mantissa
            exp = item.exp
            nm_re, nexp_re = _normalize_real(m.real, exp)
            nm_im, nexp_im = _normalize_real(m.imag, exp)
            co_nm_real, co_nm_imag, co_nexp = _coexp_ufunc(
                nm_re, nexp_re, nm_im, nexp_im)
            return (
                (co_nm_real + 1j * co_nm_imag)
                * np.ldexp(1., co_nexp)
            )
    else:
        raise TypingError("datatype not accepted {}".format(item))
    return impl


#==============================================================================
# Implementing the Xrange_polynomial class in numba
# https://numba.pydata.org/numba-doc/latest/proposals/extension-points.html
# 

class Xrange_polynomial_Type(types.Type):
    def __init__(self, dtype, cutdeg):
        self.dtype = dtype
        self.np_base_type = numba.np.numpy_support.as_dtype(
                dtype.fields["mantissa"][0])
        self.coeffs = types.Array(dtype, 1, 'C')
        # The name must be unique if the underlying model is different
        super().__init__(name="{}_Xrange_polynomial".format(
                dtype.fields["mantissa"][0]))

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

def xr_type_to_base_type(val):
    if isinstance(val, Xrange_scalar_Type):
        base_type = val.base_type
    elif val in numba_xr_types:
        base_type = val.fields["mantissa"][0]
    else:
        base_type = val # .fields["mantissa"][0]
    return numba.np.numpy_support.as_dtype(base_type)


@overload(operator.add)
def poly_add(op0, op1):
    """ Add 2  polynomials or a polynomial and a scalar"""
    if (isinstance(op0, Xrange_polynomial_Type)
            and isinstance(op1, Xrange_polynomial_Type)
            ):
        # There is no lowering implementation for a structured dtype ; so
        # we initiate a template of length 1 for the compilation.
        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

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

            return fsx.Xrange_polynomial(new_coeffs, cutdeg)
        return impl

    elif (isinstance(op0, Xrange_polynomial_Type)
            and (op1 in std_or_xr_types)
            ):
        scalar_base_type = xr_type_to_base_type(op1)
        base_dtres = np.result_type(op0.np_base_type, scalar_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

        def impl(op0, op1):
            res_len = op0.coeffs.size
            new_coeffs = res_template.repeat(res_len)
            for i in range(res_len):
                new_coeffs[i] = op0.coeffs[i]
            new_coeffs[0] = new_coeffs[0] + op1
            return fsx.Xrange_polynomial(new_coeffs, op0.cutdeg)
        return impl

    elif (isinstance(op1, Xrange_polynomial_Type)
            and (op0 in std_or_xr_types)
            ):
        scalar_base_type = xr_type_to_base_type(op0)
        base_dtres = np.result_type(op1.np_base_type, scalar_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

        def impl(op0, op1):
            res_len = op1.coeffs.size
            new_coeffs = res_template.repeat(res_len)
            for i in range(res_len):
                new_coeffs[i] = op1.coeffs[i]
            new_coeffs[0] = new_coeffs[0] + op0
            return fsx.Xrange_polynomial(new_coeffs, op1.cutdeg)
        return impl


@overload_method(Xrange_polynomial_Type, '__call__')
def xrange_polynomial_call(poly, val):
    # Implementation for scalars
    if (val in xr_types):
        base_dtres = xr_type_to_base_type(val)
        base_dtres = np.result_type(base_dtres, poly.np_base_type)
        base_numba_type = numba.from_dtype(base_dtres)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros((1,), dtype=res_dtype)

        def call_impl(poly, val):
            res = Xrange_scalar(base_numba_type(0.), numba.int32(0))#= res_template.repeat(1)res_template.repeat(1)
            coeffs = poly.coeffs
            n = coeffs.size
            res = res + coeffs[n - 1]
            for i in range(2, coeffs.size + 1):
                res = coeffs[n - i] + res * val
            return res
        return call_impl
    # Implementation for arrays
    elif isinstance(val, xrange_arty):
        base_type = val.dtype.fields["mantissa"][0]
        base_dtres = numba.np.numpy_support.as_dtype(base_type)
        base_dtres = np.result_type(base_dtres, poly.np_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

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
    # Returns the derived polynomial
    deriv_dtype = numpy_xr_type(poly.np_base_type)
    deriv_template = np.zeros((1,), dtype=deriv_dtype)

    def call_impl(poly):
        coeffs = poly.coeffs
        n = coeffs.size
        deriv_coeffs = deriv_template.repeat(n - 1)
        for i in range(n - 1):
            deriv_coeffs[i] = coeffs[i + 1] * (i + 1.)

        return fsx.Xrange_polynomial(deriv_coeffs, cutdeg=poly.cutdeg)
    return call_impl


@overload(operator.sub)
def poly_sub(op0, op1):
    if (
        (isinstance(op0, Xrange_polynomial_Type) or (op0 in std_or_xr_types))
        or (isinstance(op1, Xrange_polynomial_Type) or (op1 in std_or_xr_types))
    ):
        def impl(op0, op1):
            return op0 + (- op1)
        return impl
    else:
        raise TypingError("sa_sub, not a Xrange_polynomial_Type ({}, {})".format(
            op0, op1))

@overload(operator.mul)
def poly_mul(op0, op1):
    """ Multiply 2  polynomials or a polynomial and a scalar"""
    if (isinstance(op0, Xrange_polynomial_Type)
            and isinstance(op1, Xrange_polynomial_Type)
            ):
        # There is no lowering implementation for a structured dtype ; so
        # we initiate a template of length 1 for the compilation.
        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

        def impl(op0, op1):
            assert op0.cutdeg == op1.cutdeg
            cutdeg = op0.cutdeg
            coeffs0 = op0.coeffs
            coeffs1 = op1.coeffs
            l0 = coeffs0.size
            l1 = coeffs1.size

            res_len = min(l0 + l1 - 1, cutdeg + 1)
            new_coeffs = res_template.repeat(res_len)

            for i in range(res_len):
                window_min = max(0, i - l1 + 1)
                window_max = min(l0 - 1, i)
                for k in range(window_min, window_max + 1):
                    new_coeffs[i] = new_coeffs[i] + coeffs0[k] * coeffs1[i - k]

            return fsx.Xrange_polynomial(new_coeffs, cutdeg)
        return impl

    elif (isinstance(op0, Xrange_polynomial_Type)
            and (op1 in std_or_xr_types)
            ):
        scalar_base_type = xr_type_to_base_type(op1)
        base_dtres = np.result_type(op0.np_base_type, scalar_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

        def impl(op0, op1):
            res_len = op0.coeffs.size
            new_coeffs = res_template.repeat(res_len)
            for i in range(res_len):
                new_coeffs[i] = op0.coeffs[i] * op1
            return fsx.Xrange_polynomial(new_coeffs, op0.cutdeg)
        return impl

    elif (isinstance(op1, Xrange_polynomial_Type)
            and (op0 in std_or_xr_types)
            ):
        scalar_base_type = xr_type_to_base_type(op0)
        base_dtres = np.result_type(op1.np_base_type, scalar_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

        def impl(op0, op1):
            res_len = op1.coeffs.size
            new_coeffs = res_template.repeat(res_len)
            for i in range(res_len):
                new_coeffs[i] = op0 * op1.coeffs[i]
            return fsx.Xrange_polynomial(new_coeffs, op1.cutdeg)
        return impl


#==============================================================================
# Implementing the Xrange_SA class in numba
# https://numba.pydata.org/numba-doc/latest/proposals/extension-points.html
        
class Xrange_SA_Type(types.Type):
    def __init__(self, dtype, cutdeg, err):
        self.dtype = dtype
        numba_base_type = dtype.fields["mantissa"][0]
        self.np_base_type = numba.np.numpy_support.as_dtype(numba_base_type)
        self.coeffs = types.Array(dtype, 1, 'C')
        err_dtype = numba_xr_type(np.float64)
        self.err = types.Array(err_dtype, 1, 'C')
        #self.err = Xrange_scalar_Type(numba.float64)
        prefix = "{}_Xrange_SA"
        super().__init__(name=prefix.format(numba_base_type))

@typeof_impl.register(fsx.Xrange_SA)
def typeof_xrange_SA(val, c):
    coeffs_arrty = typeof_impl(val.coeffs, c)
    return Xrange_SA_Type(coeffs_arrty.dtype, val.cutdeg, val.err)

@type_callable(fsx.Xrange_SA)
def type_xrange_SA(context):
    def typer(coeffs, cutdeg, err):
        if (isinstance(coeffs, types.Array)
              and (coeffs.dtype in numba_xr_types)
              and isinstance(cutdeg, types.Integer)
              and isinstance(err, types.Array)
        ):
            return Xrange_SA_Type(coeffs.dtype, cutdeg, err.dtype)
    return typer

@register_model(Xrange_SA_Type)
class XrangeSAModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('coeffs', fe_type.coeffs),
            ('cutdeg', numba.int64), # Not that we need, but int32 is painful
            ('err', fe_type.err)
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

make_attribute_wrapper(Xrange_SA_Type, 'coeffs', 'coeffs')
make_attribute_wrapper(Xrange_SA_Type, 'cutdeg', 'cutdeg')
make_attribute_wrapper(Xrange_SA_Type, 'err', 'err')

@lower_builtin(fsx.Xrange_SA, types.Array, types.Integer, types.Array)
def impl_xrange_SA_constructor(context, builder, sig, args):
    typ = sig.return_type
    coeffs, cutdeg, err = args
    xrange_SA = cgutils.create_struct_proxy(typ)(context, builder)
    #  We do not copy !! sticking to implementation in python
    xrange_SA.coeffs = coeffs
    xrange_SA.cutdeg = cutdeg
    xrange_SA.err = err
    return impl_ret_borrowed(context, builder, typ, xrange_SA._getvalue())


@unbox(Xrange_SA_Type)
def unbox_xrange_SA(typ, obj, c):
    """
    Convert a fsx.Xrange_polynomial object to a native xrange_polynomial
    structure. """
    coeffs_obj = c.pyapi.object_getattr_string(obj, "coeffs")
    cutdeg_obj = c.pyapi.object_getattr_string(obj, "cutdeg")
    err_obj = c.pyapi.object_getattr_string(obj, "err")

    xrange_sa = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    xrange_sa.coeffs = c.unbox(typ.coeffs, coeffs_obj).value
    xrange_sa.cutdeg = c.pyapi.long_as_longlong(cutdeg_obj)
    xrange_sa.err = c.unbox(typ.err, err_obj).value

    c.pyapi.decref(coeffs_obj)
    c.pyapi.decref(cutdeg_obj)
    c.pyapi.decref(err_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(xrange_sa._getvalue(), is_error=is_error)

@box(Xrange_SA_Type)
def box_xrange_SA(typ, val, c):
    """ Convert a native xrange_SA structure to a 
    fsx.Xrange_polynomial object """
    xrange_SA = cgutils.create_struct_proxy(typ
        )(c.context, c.builder, value=val)
    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(
        fsx.Xrange_SA))

    coeffs_obj = c.box(typ.coeffs, xrange_SA.coeffs)
    cutdeg_obj = c.pyapi.long_from_longlong(xrange_SA.cutdeg)
    err_obj = c.box(typ.err, xrange_SA.err)

    xrange_SA_obj = c.pyapi.call_function_objargs(
        classobj, (coeffs_obj, cutdeg_obj, err_obj))

    c.pyapi.decref(classobj)
    c.pyapi.decref(coeffs_obj)
    c.pyapi.decref(cutdeg_obj)
    c.pyapi.decref(err_obj)

    return xrange_SA_obj

@overload_method(Xrange_SA_Type, 'to_polynomial')
def xrange_SA_to_polynomial(sa):
    """ Convert a xrange_SA to a xrange_polynomial ; err is disregarded """
    def impl(sa):
        return fsx.Xrange_polynomial(sa.coeffs, cutdeg=sa.cutdeg)
    return impl

@overload_method(Xrange_polynomial_Type, 'to_SA')
def xrange_polynomial_to_SA(poly):
    """ Convert a xrange_polynomial to a xrange_SA with err = 0."""
    err_template = np.zeros((1,), dtype=numpy_xr_type(np.float64))
    def impl(poly):
        return fsx.Xrange_SA(poly.coeffs, cutdeg=poly.cutdeg, 
            err=err_template.copy())
    return impl

@overload_method(Xrange_SA_Type, 'ssum')
def xrange_SA_ssum(poly):
    # Returns the sum of abs2 of coeffs 
    def impl(poly):
        coeffs = poly.coeffs
        vec_len = poly.coeffs.size

        coeff_ssum = zero()
        for i in range(vec_len):
            coeff_ssum = coeff_ssum + extended_abs2(coeffs[i])
        return coeff_ssum
    return impl


@overload(operator.neg)
def sa_neg(op0):
    """ Copy of a polynomial with sign changed """
    if isinstance(op0, Xrange_SA_Type):
        def impl(op0):
            # assert op0.coeffs.size == op0.cutdeg + 1 
            coeffs = op0.coeffs
            new_coeffs = np.empty_like(op0.coeffs)
            for i in range(coeffs.size):
                new_coeffs[i] = - coeffs[i]
            return fsx.Xrange_SA(new_coeffs, op0.cutdeg, op0.err.copy())
        return impl

@overload(operator.add)
def sa_add(op0, op1):
    """ Add 2 SA or a SA and a scalar"""

    if (isinstance(op0, Xrange_SA_Type)
            and isinstance(op1, Xrange_SA_Type)
            ):
        # There is no lowering implementation for a structured dtype ; so
        # we initiate a template of length 1 for the compilation.
        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)
        err_template = np.zeros((1,), dtype=numpy_xr_type(np.float64))

        def impl(op0, op1):
            assert op0.cutdeg == op1.cutdeg
            cutdeg = op0.cutdeg
            coeffs0 = op0.coeffs
            coeffs1 = op1.coeffs
            err0 = op0.err
            err1 = op1.err

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
            err = err_template.copy()
            err[0] = err0[0] + err1[0]

            return fsx.Xrange_SA(new_coeffs, cutdeg, err)
        return impl

    elif (isinstance(op0, Xrange_SA_Type)
            and (op1 in std_or_xr_types)
            ):
        scalar_base_type = xr_type_to_base_type(op1)
        base_dtres = np.result_type(op0.np_base_type, scalar_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

        def impl(op0, op1):
            res_len = op0.coeffs.size
            new_coeffs = res_template.repeat(res_len)
            for i in range(res_len):
                new_coeffs[i] = op0.coeffs[i]
            new_coeffs[0] = new_coeffs[0] + op1
            return fsx.Xrange_SA(new_coeffs, op0.cutdeg, op0.err.copy())
        return impl

    elif (isinstance(op1, Xrange_SA_Type)
            and (op0 in std_or_xr_types)
            ):
        scalar_base_type = xr_type_to_base_type(op0)
        base_dtres = np.result_type(op1.np_base_type, scalar_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)

        def impl(op0, op1):
            res_len = op1.coeffs.size
            new_coeffs = res_template.repeat(res_len)
            for i in range(res_len):
                new_coeffs[i] = op1.coeffs[i]
            new_coeffs[0] = new_coeffs[0] + op0
            return fsx.Xrange_SA(new_coeffs, op1.cutdeg, op1.err.copy())
        return impl
    else:
#        print("!!!", op0.__class__, op1.__class__)
        raise TypingError("sa_add, not a Xrange_SA_Type ({}, {})".format(
            op0, op1))

@overload(operator.sub)
def sa_sub(op0, op1):
    if (
        (isinstance(op0, Xrange_SA_Type) or (op0 in std_or_xr_types))
        or (isinstance(op1, Xrange_SA_Type) or (op1 in std_or_xr_types))
    ):
        def impl(op0, op1):
            return op0 + (- op1)
        return impl
    else:
        raise TypingError("sa_sub, not a Xrange_SA_Type ({}, {})".format(
            op0, op1))

@overload(operator.mul)
def sa_mul(op0, op1):
    """ Multiply 2  SA or a SA and a scalar"""
    if (isinstance(op0, Xrange_SA_Type)
            and isinstance(op1, Xrange_SA_Type)
            ):
        # There is no lowering implementation for a structured dtype ; so
        # we initiate a template of length 1 before compilation.
        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)
        err_template = np.zeros((1,), dtype=numpy_xr_type(np.float64))

        def impl(op0, op1):
            assert op0.cutdeg == op1.cutdeg
            cutdeg = op0.cutdeg
            coeffs0 = op0.coeffs
            coeffs1 = op1.coeffs
            l0 = coeffs0.size
            l1 = coeffs1.size

            res_len = min(l0 + l1 - 1, cutdeg + 1)
            new_coeffs = res_template.repeat(res_len)

            for i in range(res_len):
                window_min = max(0, i - l1 + 1)
                window_max = min(l0 - 1, i)
                for k in range(window_min, window_max + 1):
                    new_coeffs[i] = new_coeffs[i] + coeffs0[k] * coeffs1[i - k]

            err0 = op0.err[0]
            err1 = op1.err[0]
            # 4 terms to store: err, op_err0, op_err1, err_trunc
            err = err_template.repeat(4)
            err_tmp = res_template.copy()
            # err[0] = err0 * err1
            #    We will use L2 norm to control truncature error term.
            # Heuristic based on random walk / magnitude of the sum of iud random
            # variables
            # Exact term is :
            #    op_err0 = err0 * np.sum(np.abs(op1))
            #    op_err1 = err1 * np.sum(np.abs(op0))
            # Approximated term :
            # op_err0 = err0 * np.sqrt(np.sum(op1.abs2()))
            # op_err1 = err1 * np.sqrt(np.sum(op0.abs2()))    
            # > op_err0 term
            for i in range(l1):
                err[1] = err[1] + extended_abs2(coeffs1[i])
            err[1] = np.sqrt(err[1])
            err[1] = err0 * err[1]
            # > op_err1 term
            for i in range(l0):
                err[2] = err[2] + extended_abs2(coeffs0[i])
            err[2] = np.sqrt(err[2])
            err[2] = err1 * err[2]

            # Truncature_term
            if cutdeg < (l0 + l1 - 2):
                # compute the missing terms by deg
                for i in range(res_len, l0 + l1 - 1):
                    window_min = max(0, i - l1 + 1)
                    window_max = min(l0 - 1, i)
                    err_tmp[0] = Xrange_scalar(0., numba.int32(0))
                    for k in range(window_min, window_max + 1):
                        err_tmp[0] = err_tmp[0] + coeffs0[k] * coeffs1[i - k]

                    err[3] = err[3] + extended_abs2(err_tmp[0])
                err[3] = np.sqrt(err[3])

            err[0] = (op0.err[0] * op1.err[0]) + err[1] + err[2] + err[3]
            return fsx.Xrange_SA(new_coeffs, cutdeg, err[0:1])
        return impl

    elif (isinstance(op0, Xrange_SA_Type)
            and (op1 in std_or_xr_types)
            ):
        scalar_base_type = xr_type_to_base_type(op1)
        base_dtres = np.result_type(op0.np_base_type, scalar_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)
        err_template = np.zeros((1,), dtype=numpy_xr_type(np.float64))

        def impl(op0, op1):
            res_len = op0.coeffs.size
            new_coeffs = res_template.repeat(res_len)
            new_err = err_template.copy()
            for i in range(res_len):
                new_coeffs[i] = op0.coeffs[i] * op1
            new_err[0] = new_err[0] * np.abs(op1)
            return fsx.Xrange_SA(new_coeffs, op0.cutdeg, new_err)
        return impl

    elif (isinstance(op1, Xrange_SA_Type)
            and (op0 in std_or_xr_types)
            ):
        scalar_base_type = xr_type_to_base_type(op0)
        base_dtres = np.result_type(op1.np_base_type, scalar_base_type)
        res_dtype = numpy_xr_type(base_dtres)
        res_template = np.zeros((1,), dtype=res_dtype)
        err_template = np.zeros((1,), dtype=numpy_xr_type(np.float64))

        def impl(op0, op1):
            res_len = op1.coeffs.size
            new_coeffs = res_template.repeat(res_len)
            new_err = err_template.copy()
            for i in range(res_len):
                new_coeffs[i] = op1.coeffs[i] * op0
            new_err[0] = new_err[0] * np.abs(op0)
            return fsx.Xrange_SA(new_coeffs, op1.cutdeg, new_err)
        return impl

    else:
#        print("!!!", isinstance(op0, Xrange_SA_Type), isinstance(op1, Xrange_SA_Type))
        raise TypingError("sa_mul, not a Xrange_SA_Type ({}, {})".format(
            op0, op1))





#==============================================================================
#        
#class Xrange_bivar_polynomial_Type(types.Type):
#    def __init__(self, dtype, cutdeg):
#        self.dtype = dtype
#        self.np_base_type = numba.np.numpy_support.as_dtype(
#                dtype.fields["mantissa"][0])
#        self.coeffs = types.Array(dtype, 2, 'C')
#        # The name must be unique if the underlying model is different
#        super().__init__(name="{}_Xrange_bivar_polynomial".format(
#                dtype.fields["mantissa"][0]))
#
#@typeof_impl.register(fsx.Xrange_bivar_polynomial)
#def typeof_xrange_bivar_polynomial(val, c):
#    coeffs_arrty = typeof_impl(val.coeffs, c)
#    return Xrange_bivar_polynomial_Type(coeffs_arrty.dtype, val.cutdeg)
#
#@type_callable(fsx.Xrange_bivar_polynomial)
#def type_xrange_bivar_polynomial(context):
#    def typer(coeffs, cutdeg):
#        if (isinstance(coeffs, types.Array)
#              and (coeffs.dtype in numba_xr_types)
#              and isinstance(cutdeg, types.Integer)):
#            return Xrange_bivar_polynomial_Type(coeffs.dtype, cutdeg)
#    return typer
#
#@register_model(Xrange_bivar_polynomial_Type)
#class XrangeBivarPolynomialModel(models.StructModel):
#    def __init__(self, dmm, fe_type):
#        members = [
#            ('coeffs', fe_type.coeffs),
#            ('cutdeg', numba.int64) # Not that we need, but int32 is painful
#            ]
#        models.StructModel.__init__(self, dmm, fe_type, members)
#
#make_attribute_wrapper(Xrange_bivar_polynomial_Type, 'coeffs', 'coeffs')
#make_attribute_wrapper(Xrange_bivar_polynomial_Type, 'cutdeg', 'cutdeg')
#
#@lower_builtin(fsx.Xrange_bivar_polynomial, types.Array, types.Integer)
#def impl_xrange_bivar_polynomial_constructor(context, builder, sig, args):
#    typ = sig.return_type
#    coeffs, cutdeg = args
#    xrange_bivar_polynomial = cgutils.create_struct_proxy(typ)(context, builder)
#    xrange_bivar_polynomial.coeffs = coeffs
#    #  We do not copy !! following implementation in python
#    xrange_bivar_polynomial.cutdeg = cutdeg
#    return impl_ret_borrowed(context, builder, typ,
#                             xrange_bivar_polynomial._getvalue())
#
#@unbox(Xrange_bivar_polynomial_Type)
#def unbox_xrange_bivar_polynomial(typ, obj, c):
#    """
#    Convert a fsx.Xrange_bivar_polynomial object to a native xrange_polynomial
#    structure. """
#    coeffs_obj = c.pyapi.object_getattr_string(obj, "coeffs")
#    cutdeg_obj = c.pyapi.object_getattr_string(obj, "cutdeg")
#    xrange_polynomial = cgutils.create_struct_proxy(typ)(c.context, c.builder)
#    xrange_polynomial.cutdeg = c.pyapi.long_as_longlong(cutdeg_obj) 
#    xrange_polynomial.coeffs = c.unbox(typ.coeffs, coeffs_obj).value
#    c.pyapi.decref(coeffs_obj)
#    c.pyapi.decref(cutdeg_obj)
#    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#    return NativeValue(xrange_polynomial._getvalue(), is_error=is_error)
#
#@box(Xrange_bivar_polynomial_Type)
#def box_xrange_bivar_polynomial(typ, val, c):
#    """
#    Convert a native xrange_polynomial structure to a 
#    fsx.Xrange_bivar_polynomial object """
#    xrange_polynomial = cgutils.create_struct_proxy(typ
#        )(c.context, c.builder, value=val)
#    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(
#        fsx.Xrange_bivar_polynomial))
#    coeffs_obj = c.box(typ.coeffs, xrange_polynomial.coeffs)
#    cutdeg_obj = c.pyapi.long_from_longlong(xrange_polynomial.cutdeg)
#    xrange_polynomial_obj = c.pyapi.call_function_objargs(
#        classobj, (coeffs_obj, cutdeg_obj))
#    c.pyapi.decref(classobj)
#    c.pyapi.decref(coeffs_obj)
#    c.pyapi.decref(cutdeg_obj)
#    return xrange_polynomial_obj
#
#
#@overload(operator.neg)
#def bivar_poly_neg(op0):
#    """ Copy of a polynomial with sign changed """
#    if isinstance(op0, Xrange_bivar_polynomial_Type):
#        res_dtype = numpy_xr_type(op0.np_base_type)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0):
#            coeffs = op0.coeffs
#            res_len = op0.cutdeg + 1
#            assert coeffs.shape[0] == res_len
#            assert coeffs.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len ** 2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = - coeffs[i, j]
#            return fsx.Xrange_bivar_polynomial(new_coeffs, op0.cutdeg)
#        return impl
#
#
#@overload(operator.add)
#def bivar_poly_add(op0, op1):
#    """ Add 2  polynomials or a polynomial and a scalar"""
#    if (isinstance(op0, Xrange_bivar_polynomial_Type)
#            and isinstance(op1, Xrange_bivar_polynomial_Type)
#            ):
#        # There is no lowering implementation for a structured dtype ; so
#        # we initiate a template of length 1 for the compilation.
#        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0, op1):
#            assert op0.cutdeg == op1.cutdeg
#            cutdeg = op0.cutdeg
#            res_len = cutdeg + 1
#            coeffs0 = op0.coeffs
#            coeffs1 = op1.coeffs
#            assert coeffs0.shape[0] == res_len
#            assert coeffs0.shape[1] == res_len
#            assert coeffs1.shape[0] == res_len
#            assert coeffs1.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = coeffs0[i, j] + coeffs1[i, j]
#
#            return fsx.Xrange_bivar_polynomial(new_coeffs, cutdeg)
#        return impl
#
#    elif (isinstance(op0, Xrange_bivar_polynomial_Type)
#            and (op1 in std_or_xr_types)
#            ):
#        scalar_base_type = xr_type_to_base_type(op1)
#        base_dtres = np.result_type(op0.np_base_type, scalar_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0, op1):
#            res_len = op0.cutdeg + 1
#            assert op0.coeffs.shape[0] == res_len
#            assert op0.coeffs.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = op0.coeffs[i, j]
#            new_coeffs[0, 0] = new_coeffs[0, 0] + op1
#
#            return fsx.Xrange_bivar_polynomial(new_coeffs, op0.cutdeg)
#        return impl
#
#    elif (isinstance(op1, Xrange_bivar_polynomial_Type)
#            and (op0 in std_or_xr_types)
#            ):
#        scalar_base_type = xr_type_to_base_type(op0)
#        base_dtres = np.result_type(op1.np_base_type,
#                                    scalar_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0, op1):
#            res_len = op1.cutdeg + 1
#            assert op1.coeffs.shape[0] == res_len
#            assert op1.coeffs.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = op1.coeffs[i, j]
#            new_coeffs[0, 0] = new_coeffs[0, 0] + op0
#
#            return fsx.Xrange_bivar_polynomial(new_coeffs, op1.cutdeg)
#        return impl
#
#
#@overload_method(Xrange_bivar_polynomial_Type, '__call__')
#def xrange_bivar_polynomial_call(poly, valX, valY):
#    # Implementation for scalars
#    if (valX in xr_types) and (valY in xr_types) :
#        base_dtres = xr_type_to_base_type(valX)
#        assert xr_type_to_base_type(valX) == base_dtres
#        base_dtres = np.result_type(base_dtres, poly.np_base_type)
#        base_numba_type = numba.from_dtype(base_dtres)
##        res_dtype = numpy_xr_type(base_dtres)
##        res_template = np.zeros([1], dtype=res_dtype)
#
#        def call_impl(poly, valX, valY):
#            res = Xrange_scalar(base_numba_type(0.), numba.int32(0))#= res_template.repeat(1)
#            coeffs = poly.coeffs
#            n = poly.cutdeg + 1
#            assert coeffs.shape[0] == n
#            assert coeffs.shape[1] == n
#
#            for i in range(0, n):
#                # https://github.com/numba/numba/issues/7469
#                # We need to use Xrange_scalar here due to lifetime of numba 
#                # record scalar not tied to the lifetime of record array
#                # In [4]: hex(np.uint32(-555819298))
#                # Out[4]: '0xdededede'
#                # 0xde is the recently destroyed object's memory marker 
#                resXi = Xrange_scalar(base_numba_type(0.), numba.int32(0))
#                for j in range(poly.cutdeg - i, n):
#                    # sum = 2 * poly.cutdeg -i -j <= n i.e. i + j >= poly.cutdeg 
#                    resXi = (
#                        resXi * valY
#                        + (coeffs[poly.cutdeg - i, poly.cutdeg - j])
#                    )
#                res = res  * valX  + resXi
#            return res
#        return call_impl
#
#    # Implementation for arrays
#    elif isinstance(valX , xrange_arty) and isinstance(valY , xrange_arty) :
#        base_type = valX.dtype.fields["mantissa"][0]
#        assert valY.dtype.fields["mantissa"][0] == base_type
#        base_dtres = numba.np.numpy_support.as_dtype(base_type)
#        base_dtres = np.result_type(base_dtres, poly.np_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def call_impl(poly, valX, valY):
#            res_len = valX.size
#            assert valX.ndim == 1
#            assert valY.ndim == 1
#            assert valY.size == res_len
#            res = res_template.repeat(res_len)
#            resX = res_template.repeat(res_len)
#
#            coeffs = poly.coeffs
#            n = poly.cutdeg + 1
#            assert poly.coeffs.shape[0] == n
#            assert poly.coeffs.shape[1] == n
#
#            # Double Horner rule
#            # [a11 a12 a13]  -> 1
#            # [a21 a22 a23]  -> X 
#            # [a31 a32 a33]  -> X2
#            for i in range(0, n):
#                resXi = resX.copy()
#                for j in range(poly.cutdeg - i, n):
#                    # sum = 2 * poly.cutdeg -i -j <= n i.e. i + j >= poly.cutdeg 
#                    for k in range(res_len):
#                        resXi[k] = (
#                            resXi[k] * valY[k]
#                            + (coeffs[poly.cutdeg - i, poly.cutdeg - j])
#                        )
#                for k in range(res_len):
#                    res[k] = res[k]  * valX[k]  + resXi[k]
#            return res
#        return call_impl
#
#
#@overload_method(Xrange_bivar_polynomial_Type, 'deriv')
#def xrange_bivar_polynomial_deriv(poly, direction):
#    res_template = np.zeros([1], dtype=poly.coeffs.dtype)
#
#    # Returns the derivative polynomial
#    def call_impl(poly, direction):
#        res_len = poly.cutdeg + 1
#        assert poly.coeffs.shape[0] == res_len
#        assert poly.coeffs.shape[1] == res_len
#        
#        new_coeffs = res_template.repeat(
#            res_len ** 2).reshape(res_len, res_len)
#
#        if direction == "X":
#            for j in range(res_len):
#                for i in range(res_len - j - 1):
#                    new_coeffs[i, j] = poly.coeffs[i + 1, j] * (i + 1.)
#        elif direction == "Y":
#            for i in range(res_len):
#                for j in range(res_len - i - 1):
#                    new_coeffs[i, j] = poly.coeffs[i, j + 1] * (j + 1.)
#        else:
#            raise ValueError("Invalid direction")
#
#        return fsx.Xrange_bivar_polynomial(new_coeffs, cutdeg=poly.cutdeg)
#    return call_impl
#
#
#@overload(operator.sub)
#def bivar_poly_sub(op0, op1):
#    if (
#        (isinstance(op0, Xrange_bivar_polynomial_Type) or (op0 in std_or_xr_types))
#        or (isinstance(op1, Xrange_bivar_polynomial_Type) or (op1 in std_or_xr_types))
#    ):
#        def impl(op0, op1):
#            return op0 + (- op1)
#        return impl
#    else:
#        raise TypingError("sa_sub, not a Xrange_bivar_polynomial ({}, {})".format(
#            op0, op1))
#
#@overload(operator.mul)
#def bivar_poly_mul(op0, op1):
#    """ Multiply 2  polynomials or a polynomial and a scalar"""
#    if (isinstance(op0, Xrange_bivar_polynomial_Type)
#            and isinstance(op1, Xrange_bivar_polynomial_Type)
#            ):
#        # There is no lowering implementation for a structured dtype ; so
#        # we initiate a template of length 1 for the compilation.
#        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0, op1):
#            assert op0.cutdeg == op1.cutdeg
#            cutdeg = op0.cutdeg
#            res_len = cutdeg + 1
#            coeffs0 = op0.coeffs
#            coeffs1 = op1.coeffs
#            assert coeffs0.shape[0] == res_len
#            assert coeffs0.shape[1] == res_len
#            assert coeffs1.shape[0] == res_len
#            assert coeffs1.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len **2 ).reshape(res_len, res_len)
#
#            for i in range(0, res_len):
#                for j in range(0, res_len - i):
#                    for k in range(0, i + 1):
#                        for l in range(0, j + 1):
#                            new_coeffs[i, j] = (
#                                new_coeffs[i, j]
#                                + coeffs0[k, l] * coeffs1[i - k, j - l]
#                            )
#
#            return fsx.Xrange_bivar_polynomial(new_coeffs, cutdeg)
#        return impl
#
#    elif (isinstance(op0, Xrange_bivar_polynomial_Type)
#            and (op1 in std_or_xr_types)
#            ):
#        scalar_base_type = xr_type_to_base_type(op1)
#        base_dtres = np.result_type(op0.np_base_type, scalar_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0, op1):
#            res_len = op0.cutdeg + 1
#            assert op0.coeffs.shape[0] == res_len
#            assert op0.coeffs.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len ** 2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = op0.coeffs[i, j] * op1
#            return fsx.Xrange_bivar_polynomial(new_coeffs, op0.cutdeg)
#        return impl
#
#    elif (isinstance(op1, Xrange_bivar_polynomial_Type)
#            and (op0 in std_or_xr_types)
#            ):
#        scalar_base_type = xr_type_to_base_type(op0)
#        base_dtres = np.result_type(op1.np_base_type,
#                                    scalar_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0, op1):
#            res_len = op1.cutdeg + 1
#            assert op1.coeffs.shape[0] == res_len
#            assert op1.coeffs.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len ** 2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = op0 * op1.coeffs[i, j]
#            return fsx.Xrange_bivar_polynomial(new_coeffs, op1.cutdeg)
#        return impl
#
#
#@generated_jit(nopython=True)
#def build_Xr_polynomial(arr, cutdeg):
#    # Generic implementation for poly & bivar_poly
#    if arr.ndim == 1:
#        def impl(arr, cutdeg):
#            return fsx.Xrange_polynomial(arr, cutdeg)
#        return impl
#    elif arr.ndim == 2:
#        def impl(arr, cutdeg):
#            return fsx.Xrange_bivar_polynomial(arr, cutdeg)
#        return impl
#    else:
#        raise NotImplementedError("No matching polynomial implementation for "
#                                  f"this ndim: {arr.ndim}")
#
##==============================================================================
## Implementing the Xrange_bivar_SA class in numba
## https://numba.pydata.org/numba-doc/latest/proposals/extension-points.html
## Caveat : unboxing is too complex and not implemented, so jitted function
## can return xrange_SA instances but not take xrange_SA as argument.
## workarounf is to pass separately xrange_polynomial and err (if not null)
## then use xrange_polynomial_to_SA
#        
#class Xrange_bivar_SA_Type(types.Type):
#    def __init__(self, dtype, cutdeg, err):
#        self.dtype = dtype
#        numba_base_type = dtype.fields["mantissa"][0]
#        self.np_base_type = numba.np.numpy_support.as_dtype(numba_base_type)
#        self.coeffs = types.Array(dtype, 2, 'C')
#        err_dtype = numba_xr_type(np.float64)
#        self.err = types.Array(err_dtype, 1, 'C')
#        #self.err = Xrange_scalar_Type(numba.float64)
#        prefix = "{}_Xrange_bivar_SA"
#        super().__init__(name=prefix.format(numba_base_type))
#
#@typeof_impl.register(fsx.Xrange_bivar_SA)
#def typeof_xrange_bivar_SA(val, c):
#    coeffs_arrty = typeof_impl(val.coeffs, c)
#    return Xrange_bivar_SA_Type(coeffs_arrty.dtype, val.cutdeg, val.err)
#
#@type_callable(fsx.Xrange_bivar_SA)
#def type_xrange_bivar_SA(context):
#    def typer(coeffs, cutdeg, err):
#        if (isinstance(coeffs, types.Array)
#              and (coeffs.dtype in numba_xr_types)
#              and isinstance(cutdeg, types.Integer)
#              and isinstance(err, types.Array)
#        ):
#            return Xrange_bivar_SA_Type(coeffs.dtype, cutdeg, err.dtype)
#    return typer
#
#@register_model(Xrange_bivar_SA_Type)
#class XrangebivarSAModel(models.StructModel):
#    def __init__(self, dmm, fe_type):
#        members = [
#            ('coeffs', fe_type.coeffs),
#            ('cutdeg', numba.int64), # Not that we need, but int32 is painful
#            ('err', fe_type.err)
#        ]
#        models.StructModel.__init__(self, dmm, fe_type, members)
#
#make_attribute_wrapper(Xrange_bivar_SA_Type, 'coeffs', 'coeffs')
#make_attribute_wrapper(Xrange_bivar_SA_Type, 'cutdeg', 'cutdeg')
#make_attribute_wrapper(Xrange_bivar_SA_Type, 'err', 'err')
#
#@lower_builtin(fsx.Xrange_bivar_SA, types.Array, types.Integer, types.Array)
#def impl_xrange_bivar_SA_constructor(context, builder, sig, args):
#    typ = sig.return_type
#    coeffs, cutdeg, err = args
#    xrange_bivar_SA = cgutils.create_struct_proxy(typ)(context, builder)
#    #  We do not copy !! sticking to implementation in python
#    xrange_bivar_SA.coeffs = coeffs
#    xrange_bivar_SA.cutdeg = cutdeg
#    xrange_bivar_SA.err = err
#    return impl_ret_borrowed(context, builder, typ, xrange_bivar_SA._getvalue())
#
#
#@unbox(Xrange_bivar_SA_Type)
#def unbox_xrange_bivar_SA(typ, obj, c):
#    """
#    Convert a fsx.xrange_bivar_sa object to a native Xrange_bivar_sa
#    structure. """
#    coeffs_obj = c.pyapi.object_getattr_string(obj, "coeffs")
#    cutdeg_obj = c.pyapi.object_getattr_string(obj, "cutdeg")
#    err_obj = c.pyapi.object_getattr_string(obj, "err")
#
#    xrange_bivar_sa = cgutils.create_struct_proxy(typ)(c.context, c.builder)
#    xrange_bivar_sa.coeffs = c.unbox(typ.coeffs, coeffs_obj).value
#    xrange_bivar_sa.cutdeg = c.pyapi.long_as_longlong(cutdeg_obj)
#    xrange_bivar_sa.err = c.unbox(typ.err, err_obj).value
#
#    c.pyapi.decref(coeffs_obj)
#    c.pyapi.decref(cutdeg_obj)
#    c.pyapi.decref(err_obj)
#
#    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#    return NativeValue(xrange_bivar_sa._getvalue(), is_error=is_error)
#
#@box(Xrange_bivar_SA_Type)
#def box_xrange_bivar_SA(typ, val, c):
#    """ Convert a native xrange_bivar_SA structure to a 
#    fsx.Xrange_polynomial object """
#    xrange_bivar_SA = cgutils.create_struct_proxy(typ
#        )(c.context, c.builder, value=val)
#    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(
#        fsx.Xrange_bivar_SA))
#
#    coeffs_obj = c.box(typ.coeffs, xrange_bivar_SA.coeffs)
#    cutdeg_obj = c.pyapi.long_from_longlong(xrange_bivar_SA.cutdeg)
#    err_obj = c.box(typ.err, xrange_bivar_SA.err)
#
#    xrange_bivar_SA_obj = c.pyapi.call_function_objargs(
#        classobj, (coeffs_obj, cutdeg_obj, err_obj))
#
#    c.pyapi.decref(classobj)
#    c.pyapi.decref(coeffs_obj)
#    c.pyapi.decref(cutdeg_obj)
#    c.pyapi.decref(err_obj)
#
#    return xrange_bivar_SA_obj
#
#@overload_method(Xrange_bivar_SA_Type, 'to_polynomial')
#def xrange_bivar_SA_to_polynomial(sa):
#    """ Convert a xrange_SA to a xrange_polynomial ; err is disregarded """
#    def impl(sa):
#        return fsx.Xrange_bivar_polynomial(sa.coeffs, cutdeg=sa.cutdeg)
#    return impl
#
#@overload_method(Xrange_bivar_polynomial_Type, 'to_SA')
#def xrange_bivar_polynomial_to_SA(poly):
#    """ Convert a xrange_polynomial to a xrange_SA with err = 0."""
#    err_template = np.zeros((1,), dtype=numpy_xr_type(np.float64))
#    def impl(poly):
#        return fsx.Xrange_bivar_SA(poly.coeffs, cutdeg=poly.cutdeg, 
#            err=err_template.copy())
#    return impl
#
#@overload_method(Xrange_bivar_SA_Type, 'ssum')
#def xrange_bivar_SA_ssum(poly):
#
#    # Returns the sum of obs2 of coeffs
#    def impl(poly):
#        coeffs = poly.coeffs
#        vec_len = poly.cutdeg + 1
#        assert coeffs.shape[0] == vec_len
#        assert coeffs.shape[1] == vec_len
#
#        coeff_ssum = zero()
#        for i in range(vec_len):
#            for j in range(vec_len - i):
#                coeff_ssum = coeff_ssum + extended_abs2(coeffs[i, j])
#        return coeff_ssum
#    return impl
#
#@overload(operator.neg)
#def bivar_sa_neg(op0):
#    """ Copy of a bivar_SA with sign changed """
#    if isinstance(op0, Xrange_bivar_SA_Type):
#        res_dtype = numpy_xr_type(op0.np_base_type)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0):
#            coeffs = op0.coeffs
#            res_len = op0.cutdeg + 1
#            assert coeffs.shape[0] == res_len
#            assert coeffs.shape[1] == res_len
#            
#            new_coeffs = res_template.repeat(
#                res_len ** 2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = - coeffs[i, j]
#
#            return fsx.Xrange_bivar_SA(new_coeffs, op0.cutdeg, op0.err.copy())
#        return impl
#
#
#@overload(operator.add)
#def bivar_sa_add(op0, op1):
#    """ Add 2 bivar_SA or a bivar_SA and a scalar"""
#    if (isinstance(op0, Xrange_bivar_SA_Type)
#            and isinstance(op1, Xrange_bivar_SA_Type)
#            ):
#        # There is no lowering implementation for a structured dtype ; so
#        # we initiate a template of length 1 for the compilation.
#        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#        err_template = np.zeros([1], dtype=numpy_xr_type(np.float64))
#
#        def impl(op0, op1):
#            assert op0.cutdeg == op1.cutdeg
#            cutdeg = op0.cutdeg
#            res_len = cutdeg + 1
#            coeffs0 = op0.coeffs
#            coeffs1 = op1.coeffs
#            err0 = op0.err
#            err1 = op1.err
#
#            assert coeffs0.shape[0] == res_len
#            assert coeffs0.shape[1] == res_len
#            assert coeffs1.shape[0] == res_len
#            assert coeffs1.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = coeffs0[i, j] + coeffs1[i, j]
#
#            err = err_template.copy()
#            err[0] = err0[0] + err1[0]
#
#            return fsx.Xrange_bivar_SA(new_coeffs, cutdeg, err)
#        return impl
#
#    elif (isinstance(op0, Xrange_bivar_SA_Type)
#            and (op1 in std_or_xr_types)
#            ):
#        scalar_base_type = xr_type_to_base_type(op1)
#        base_dtres = np.result_type(op0.np_base_type, scalar_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0, op1):
#            res_len = op0.cutdeg + 1
#            assert op0.coeffs.shape[0] == res_len
#            assert op0.coeffs.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = op0.coeffs[i, j]
#            new_coeffs[0, 0] = new_coeffs[0, 0] + op1
#
#            return fsx.Xrange_bivar_SA(new_coeffs, op0.cutdeg, op0.err.copy())
#        return impl
#
#    elif (isinstance(op1, Xrange_bivar_SA_Type)
#            and (op0 in std_or_xr_types)
#            ):
#        scalar_base_type = xr_type_to_base_type(op0)
#        base_dtres = np.result_type(op1.np_base_type, scalar_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0, op1):
#            res_len = op1.cutdeg + 1
#            assert op1.coeffs.shape[0] == res_len
#            assert op1.coeffs.shape[1] == res_len
#            
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = op1.coeffs[i, j]
#            new_coeffs[0, 0] = new_coeffs[0, 0] + op0
#
#            return fsx.Xrange_bivar_SA(new_coeffs, op1.cutdeg, op1.err.copy())
#        return impl
#    else:
##        print("!!!", op0.__class__, op1.__class__)
#        raise TypingError("sa_add, not a Xrange_bivar_SA_Type ({}, {})".format(
#            op0, op1))
#
#@overload(operator.sub)
#def bivar_sa_sub(op0, op1):
#    if (
#        (isinstance(op0, Xrange_bivar_SA_Type) or (op0 in std_or_xr_types))
#        or (isinstance(op1, Xrange_bivar_SA_Type) or (op1 in std_or_xr_types))
#    ):
#        def impl(op0, op1):
#            return op0 + (- op1)
#        return impl
#    else:
#        raise TypingError("sa_sub, not a Xrange_bivar_SA_Type ({}, {})".format(
#            op0, op1))
#
#
#@overload(operator.mul)
#def bivar_sa_mul(op0, op1):
#    """ Mul 2 bivar_SA or a bivar_SA and a scalar"""
#    if (isinstance(op0, Xrange_bivar_SA_Type)
#            and isinstance(op1, Xrange_bivar_SA_Type)
#            ):
#        # There is no lowering implementation for a structured dtype ; so
#        # we initiate a template of length 1 for the compilation.
#        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
#        base_numba_type = numba.from_dtype(base_dtres)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#        base_err_numba_type = numba.float64
#        err_template = np.zeros([1], dtype=numpy_xr_type(np.float64))
#
#        def impl(op0, op1):
#            assert op0.cutdeg == op1.cutdeg
#            cutdeg = op0.cutdeg
#            res_len = cutdeg + 1
#            coeffs0 = op0.coeffs
#            coeffs1 = op1.coeffs
#            err0 = op0.err
#            err1 = op1.err
#
#            assert coeffs0.shape[0] == res_len
#            assert coeffs0.shape[1] == res_len
#            assert coeffs1.shape[0] == res_len
#            assert coeffs1.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            # General term, the hard way
#            for i in range(0, res_len):
#                for j in range(0, res_len - i):
#                    for k in range(0, i + 1):
#                        for l in range(0, j + 1):
#                            new_coeffs[i, j] = (
#                                new_coeffs[i, j]
#                                + coeffs0[k, l] * coeffs1[i - k, j - l]
#                            )
#
#            # Truncature error term
#            errT = Xrange_scalar(base_err_numba_type(0.), numba.int32(0))
#            for i in range(0, 2 * cutdeg + 1):
#                for j in range(0, 2 * cutdeg + 1 - i):
#                    if (i + j) <= cutdeg: # Normal term, not an error
#                        continue
#                    ij_errT = Xrange_scalar(base_numba_type(0.),
#                                            numba.int32(0))
#                    for k in range(max(0, i - cutdeg), 
#                               min(i + 1, res_len)):
#                        for l in range(max(0, j - cutdeg),
#                                   min(j + 1, res_len)):
#                            ij_errT = ij_errT + coeffs0[k, l] * coeffs1[i - k, j - l]
#                    # print("Truncature error term", ij_errT.mantissa * 2 ** ij_errT.exp)
#                    errT = errT + extended_abs2(ij_errT)
#            errT = np.sqrt(errT)
#
#            # Sums0 and sums1 error term
#            sums0 = Xrange_scalar(base_err_numba_type(0.), numba.int32(0))
#            sums1 = Xrange_scalar(base_err_numba_type(0.), numba.int32(0))
#            for i in range(0, cutdeg + 1):
#                for j in range(0, cutdeg + 1 - i):
#                    sums0 = sums0 + extended_abs2(coeffs0[i, j])
#                    sums1 = sums1 + extended_abs2(coeffs1[i, j])
#            sums0 = np.sqrt(sums0)
#            sums1 = np.sqrt(sums1)
#
#            # Total err
#            err = err_template.copy()
#            err[0] = err0[0] * sums1 + err1[0] * sums0 + err0[0] * err1[0] + errT
#
#            return fsx.Xrange_bivar_SA(new_coeffs, cutdeg, err)
#        return impl
#
#    elif (isinstance(op0, Xrange_bivar_SA_Type)
#            and (op1 in std_or_xr_types)
#            ):
#        scalar_base_type = xr_type_to_base_type(op1)
#        base_dtres = np.result_type(op0.np_base_type, scalar_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#        err_template = np.zeros([1], dtype=numpy_xr_type(np.float64))
#
#        def impl(op0, op1):
#            res_len = op0.cutdeg + 1
#            assert op0.coeffs.shape[0] == res_len
#            assert op0.coeffs.shape[1] == res_len
#
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = op0.coeffs[i, j] * op1
#
#            # Total err
#            err = err_template.copy()
#            err[0] = op0.err[0] * np.abs(op1)
#
#            return fsx.Xrange_bivar_SA(new_coeffs, op0.cutdeg, err)
#        return impl
#
#    elif (isinstance(op1, Xrange_bivar_SA_Type)
#            and (op0 in std_or_xr_types)
#            ):
#        scalar_base_type = xr_type_to_base_type(op0)
#        base_dtres = np.result_type(op1.np_base_type, scalar_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#        err_template = np.zeros([1], dtype=numpy_xr_type(np.float64))
#
#        def impl(op0, op1):
#            res_len = op1.cutdeg + 1
#            assert op1.coeffs.shape[0] == res_len
#            assert op1.coeffs.shape[1] == res_len
#            
#            new_coeffs = res_template.repeat(
#                res_len **2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = op0 * op1.coeffs[i, j]
#
#            # Total err
#            err = err_template.copy()
#            err[0] = op1.err[0] * np.abs(op0)
#
#            return fsx.Xrange_bivar_SA(new_coeffs, op1.cutdeg, err)
#        return impl
#    else:
#        raise TypingError("sa_bivar_mul, not a Xrange_bivar_SA_Type ({}, {})".format(
#            op0, op1))
#
#
##==============================================================================
## Implementing a Xrange_monome class in numba "k * X" - for efficiency in 
## SA loops
## This is needed for mostly bivar due to the implementation details which always
## consider cutdeg x cutdeg arrays
## TODO : to be tested, operations x and + to be implemented (with SA)
#
#class Xrange_monome_Type(types.Type):
#    def __init__(self, dtype):
#        self.dtype = dtype
#        self.np_base_type = numba.np.numpy_support.as_dtype(
#                dtype.fields["mantissa"][0])
#        self.k = types.Array(dtype, 1, 'C')
#        # The name must be unique if the underlying model is different
#        super().__init__(name="{}_Xrange_monome".format(
#                dtype.fields["mantissa"][0]))
#
#@typeof_impl.register(fsx.Xrange_monome)
#def typeof_xrange_monome(val, c):
#    k_arrty = typeof_impl(val.k, c)
#    return Xrange_monome_Type(k_arrty.dtype)
#
#@type_callable(fsx.Xrange_monome)
#def type_xrange_monome(context):
#    def typer(k):
#        if (isinstance(k, types.Array)
#              and (k.dtype in numba_xr_types)):
#            return Xrange_monome_Type(k.dtype)
#    return typer
#
#@register_model(Xrange_monome_Type)
#class XrangeMonomeModel(models.StructModel):
#    def __init__(self, dmm, fe_type):
#        members = [
#            ('k', fe_type.k),
#        ]
#        models.StructModel.__init__(self, dmm, fe_type, members)
#
#make_attribute_wrapper(Xrange_monome_Type, 'k', 'k')
#
#@lower_builtin(fsx.Xrange_monome, types.Array)
#def impl_xrange_monome_constructor(context, builder, sig, args):
#    typ = sig.return_type
#    k, = args
#    xrange_monome = cgutils.create_struct_proxy(typ)(context, builder)
#    xrange_monome.k = k
#    #  We do not copy !! following implementation in python
#    return impl_ret_borrowed(context, builder, typ,
#                             xrange_monome._getvalue())
#
#@unbox(Xrange_monome_Type)
#def unbox_xrange_monome(typ, obj, c):
#    """
#    Convert a fsx.Xrange_monome object to a native xrange_monome
#    structure. """
#    k_obj = c.pyapi.object_getattr_string(obj, "k")
#    xrange_monome = cgutils.create_struct_proxy(typ)(c.context, c.builder)
#    xrange_monome.k = c.unbox(typ.k, k_obj).value
#    c.pyapi.decref(k_obj)
#    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#    return NativeValue(xrange_monome._getvalue(), is_error=is_error)
#
#@box(Xrange_monome_Type)
#def box_xrange_monome(typ, val, c):
#    """
#    Convert a native xrange_monome structure to a 
#    fsx.Xrange_monome object """
#    xrange_monome = cgutils.create_struct_proxy(typ
#        )(c.context, c.builder, value=val)
#    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(
#        fsx.Xrange_monome))
#    k_obj = c.box(typ.k, xrange_monome.k)
#    xrange_monome_obj = c.pyapi.call_function_objargs(
#        classobj, (k_obj,))
#    c.pyapi.decref(classobj)
#    c.pyapi.decref(k_obj)
#    return xrange_monome_obj
#
#@overload(operator.neg)
#def monome_neg(op0):
#    """ Copy of a Monome with sign changed """
#    if isinstance(op0, Xrange_monome_Type):
#        res_dtype = numpy_xr_type(op0.np_base_type)
#        res_template = np.zeros([1], dtype=res_dtype)
#
#        def impl(op0):
#            kX = op0.k
#            new_k = res_template.repeat(1)
#            new_k[0] = - kX[0]
#            return fsx.Xrange_monome(new_k)
#        return impl
#
#
#@overload(operator.add)
#def monome_add(op0, op1):
#    """ Add a Monome to a SA or a bivar_SA """
#    if (isinstance(op0, Xrange_monome_Type)
#            and isinstance(op1, Xrange_bivar_SA_Type)
#            ):
#        # There is no lowering implementation for a structured dtype ; so
#        # we initiate a template of length 1 for the compilation.
#        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#        err_template = np.zeros([1], dtype=numpy_xr_type(np.float64))
#
#        def impl(op0, op1):
#
#            cutdeg = op1.cutdeg
#            res_len = cutdeg + 1
#            kX = op0.k
#            coeffs1 = op1.coeffs
#            err1 = op1.err
#
#            assert res_len > 1
#
#            new_coeffs = res_template.repeat(
#                res_len ** 2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = coeffs1[i, j]
#            new_coeffs[1, 0] = new_coeffs[1, 0] + kX[0]
#
#            err = err_template.copy()
#            err[0] = err1[0]
#
#            return fsx.Xrange_bivar_SA(new_coeffs, cutdeg, err)
#        return impl
#    
#    elif (isinstance(op0, Xrange_bivar_SA_Type)
#            and isinstance(op1, Xrange_monome_Type)
#            ):
#        # There is no lowering implementation for a structured dtype ; so
#        # we initiate a template of length 1 for the compilation.
#        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#        err_template = np.zeros([1], dtype=numpy_xr_type(np.float64))
#
#        def impl(op0, op1):
#
#            cutdeg = op0.cutdeg
#            res_len = cutdeg + 1
#            kX = op1.k
#            coeffs0 = op0.coeffs
#            err0 = op0.err
#
#            assert res_len > 1
#
#            new_coeffs = res_template.repeat(
#                res_len ** 2).reshape(res_len, res_len)
#            for i in range(res_len):
#                for j in range(res_len - i):
#                    new_coeffs[i, j] = coeffs0[i, j]
#            new_coeffs[1, 0] = new_coeffs[1, 0] + kX[0]
#
#            err = err_template.copy()
#            err[0] = err0[0]
#
#            return fsx.Xrange_bivar_SA(new_coeffs, cutdeg, err)
#        return impl
#
#    elif (isinstance(op0, Xrange_monome_Type)
#            and isinstance(op1, Xrange_SA_Type)
#            ):
#        # There is no lowering implementation for a structured dtype ; so
#        # we initiate a template of length 1 for the compilation.
#        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#        err_template = np.zeros([1], dtype=numpy_xr_type(np.float64))
#
#        def impl(op0, op1):
#            cutdeg = op1.cutdeg
#            kX = op0.k
#            coeffs1 = op1.coeffs
#            err1 = op1.err
#
#            assert cutdeg >= 1
#
#            res_len = max(2, coeffs1.size)
#            new_coeffs = res_template.repeat(res_len)
#            for i in range(coeffs1.size):
#                new_coeffs[i] = coeffs1[i]
#
#            new_coeffs[1] = new_coeffs[1] + kX[0]
#
#            err = err_template.copy()
#            err[0] = err1[0]
#
#            return fsx.Xrange_SA(new_coeffs, cutdeg, err)
#        return impl
#
#    elif (isinstance(op0, Xrange_SA_Type)
#            and isinstance(op1, Xrange_monome_Type)
#            ):
#        # There is no lowering implementation for a structured dtype ; so
#        # we initiate a template of length 1 for the compilation.
#        base_dtres = np.result_type(op0.np_base_type, op1.np_base_type)
#        res_dtype = numpy_xr_type(base_dtres)
#        res_template = np.zeros([1], dtype=res_dtype)
#        err_template = np.zeros([1], dtype=numpy_xr_type(np.float64))
#
#        def impl(op0, op1):
#            cutdeg = op0.cutdeg
#            kX = op1.k
#            coeffs0 = op0.coeffs
#            err0 = op0.err
#
#            assert cutdeg >= 1
#
#            res_len = max(2, coeffs0.size)
#            new_coeffs = res_template.repeat(res_len)
#            for i in range(coeffs0.size):
#                new_coeffs[i] = coeffs0[i]
#
#            new_coeffs[1] = new_coeffs[1] + kX[0]
#
#            err = err_template.copy()
#            err[0] = err0[0]
#
#            return fsx.Xrange_SA(new_coeffs, cutdeg, err)
#        return impl
#
#    else:
##        print("!!!", op0.__class__, op1.__class__)
#        raise TypingError("monome_add, not a Xrange_monome_Type ({}, {})".format(
#            op0, op1))
#
#@overload(operator.sub)
#def monome_sub(op0, op1):
#    if isinstance(op1, Xrange_monome_Type):
#        def impl(op0, op1):
#            return op0 + (- op1)
#        return impl
#    else:
#        raise TypingError("monome_sub, not a Xrange_monome_Type ({}, {})".format(
#            op0, op1))
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##"""
##Implements atomic compare-and-swap in user-land using llvm_call() and use it to
##create a lock.
##Ref:
##https://gist.github.com/sklam/40f25167351832fe55b64232785d036d
##
##https://llvm.org/docs/LangRef.html#cmpxchg-instruction
##The ‘cmpxchg’ instruction is used to atomically modify memory. It loads a value
##in memory and compares it to a given value. If they are equal, it tries to
##store a new value into the memory.
##"""
##
##@numba.extending.intrinsic
##def atomic_xchg(context, ptr, cmp, val):
##    if isinstance(ptr, numba.types.CPointer):
##        valtype = ptr.dtype
##        sig = valtype(ptr, valtype, valtype)
##
##        def codegen(context, builder, signature, args):
##            [ptr, cmpval, value] = args
##            res = builder.cmpxchg(ptr, cmpval, value, ordering='monotonic')
##            oldval, succ = numba.core.cgutils.unpack_tuple(builder, res)
##            return oldval
##        return sig, codegen
##
##@numba.extending.intrinsic
##def cast_as_intp_ptr(context, ptrval):
##    ptrty = numba.types.CPointer(numba.intp)
##    sig = ptrty(numba.intp)
##
##    def codegen(context, builder, signature, args):
##        [val] = args
##        llrety = context.get_value_type(signature.return_type)
##        return builder.inttoptr(val, llrety)
##    return sig, codegen
##
##@numba.njit("intp(intp[:])")
##def try_lock(lock):
##    iptr = cast_as_intp_ptr(lock[0:].ctypes.data)
##    old = atomic_xchg(iptr, 0, 1)
##    return old == 0
##
##@numba.njit("void(intp[:])")
##def unlock(lock):
##    iptr = cast_as_intp_ptr(lock[0:].ctypes.data)
##    old = atomic_xchg(iptr, 1, 0)
##    assert old == 1
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##"""
##Implements Bivar Series approximations
##"""
###MIN_BIVAR_SPAN = 1024
###MIN_BIVAR_SEED = 1024
###MAX_FLOAT_XR = np.asarray(fsx.Xrange_array(1., 2**31-2)).ravel()
##
###zero = np.zeros((1,), dtype=numpy_xr_type(np.complex128))
##
###------------------------------------------------------------------------------
#class Ref_path:    
#    def __init__(self, ref_path, ref_index_xr, ref_xr, ref_div_iter,
#                 drift, dx):
#        """
#        Holds a standard precision orbit from full precision iterations + where
#        needed to avoid overflow, a set of 'extended range' orbit values.
#
#        ref_path: float64 np.array, the FP orbit stored in standard precision
#        ref_index_xr: int32 np.array, the indices for which we need an extended 
#            range to avoid underflow
#        ref_index_xr: int32 np.array, the extended range values at ref_index_xr
#        ref_div_iter: int, the first invalid iteration
#                      (diverging or not computed)
#        drift: c0 - ref where c0 is the image center, Xrange (scalar array).
#        dx: image width, Xrange (scalar array shape 1).
#
#        Note: read-only access ; class members should not be modified after
#        initialization
#        """
#        self.ref_path = ref_path
#        self.ref_index_xr = ref_index_xr
#        self.ref_xr = ref_xr
#        self.ref_div_iter = ref_div_iter
#        self.has_xr = (self.ref_index_xr.size > 0)
#        self.drift = drift.ravel()
#        self.dx = dx.ravel()
#
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Numba implementation - Note: the constructor is not implemented in Numba
## Only boxing / unboxing is implemented
## -> Ref_path object to be created in python and passed to numba
#
#class Ref_path_Type(types.Type):
#    def __init__(self):
#        self.ref_path = types.Array(numba.complex128, 1, 'C')
#        self.ref_index_xr = types.Array(numba.int32, 1, 'C')
#        self.ref_xr = types.Array(numba_xr_type(np.complex128), 1, 'C')
#        self.has_xr = types.boolean
#        self.ref_div_iter = types.int64
#        self.drift = types.Array(numba_xr_type(np.complex128), 1, 'C')
#        self.dx = types.Array(numba_xr_type(np.float64), 1, 'C')
#        # The name must be unique if the underlying model is different
#        super().__init__("Ref_path")
#
#@typeof_impl.register(Ref_path)
#def typeof_ref_path(val, c):
#    return Ref_path_Type()
#
#@register_model(Ref_path_Type)
#class RefPathModel(models.StructModel):
#    def __init__(self, dmm, fe_type):
#        members = [
#            ('ref_path', fe_type.ref_path),
#            ('ref_index_xr', fe_type.ref_index_xr),
#            ('ref_xr', fe_type.ref_xr),
#            ('has_xr', fe_type.has_xr),
#            ('ref_div_iter', fe_type.ref_div_iter),
#            ('drift', fe_type.drift),
#            ('dx', fe_type.dx),
#        ]
#        models.StructModel.__init__(self, dmm, fe_type, members)
#
#for attr in ['ref_path', 'ref_index_xr', 'ref_xr', 'has_xr', 'ref_div_iter',
#             'drift', 'dx']:
#    make_attribute_wrapper(Ref_path_Type, attr, attr)
#
#@unbox(Ref_path_Type)
#def unbox_ref_path(typ, obj, c):
#    """
#    Convert a Ref_path_Type object to a native structure
#    structure. """
#    ref_path_obj = c.pyapi.object_getattr_string(obj, "ref_path")
#    ref_index_xr_obj = c.pyapi.object_getattr_string(obj, "ref_index_xr")
#    ref_xr_obj = c.pyapi.object_getattr_string(obj, "ref_xr")
#    has_xr_obj = c.pyapi.object_getattr_string(obj, "has_xr")
#    ref_div_iter_obj = c.pyapi.object_getattr_string(obj, "ref_div_iter")
#    drift_obj = c.pyapi.object_getattr_string(obj, "drift")
#    dx_obj = c.pyapi.object_getattr_string(obj, "dx")
#
#    Ref_path = cgutils.create_struct_proxy(typ)(c.context, c.builder)
#    Ref_path.ref_path = c.unbox(typ.ref_path, ref_path_obj).value
#    Ref_path.ref_index_xr = c.unbox(typ.ref_index_xr, ref_index_xr_obj).value
#    Ref_path.ref_xr = c.unbox(typ.ref_xr, ref_xr_obj).value
#    # Boolean boxing following :
#    # https://github.com/numba/numba/blob/2a792155c3dce43f86b9ff93802f12d39a3752dc/numba/core/boxing.py
#    istrue = c.pyapi.object_istrue(has_xr_obj)
#    zero = ir.Constant(istrue.type, 0)
#    Ref_path.has_xr = c.builder.icmp_signed('!=', istrue, zero)
#    Ref_path.ref_div_iter = c.pyapi.long_as_longlong(ref_div_iter_obj)
#    Ref_path.drift = c.unbox(typ.drift, drift_obj).value
#    Ref_path.dx = c.unbox(typ.dx, dx_obj).value
#
#    c.pyapi.decref(ref_path_obj)
#    c.pyapi.decref(ref_index_xr_obj)
#    c.pyapi.decref(ref_xr_obj)
#    c.pyapi.decref(has_xr_obj)
#    c.pyapi.decref(ref_div_iter_obj)
#    c.pyapi.decref(drift_obj)
#    c.pyapi.decref(dx_obj)
#
#    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#    return NativeValue(Ref_path._getvalue(), is_error=is_error)
#
#
#@box(Ref_path_Type)
#def box_ref_path(typ, val, c):
#    """
#    Convert a native Ref_path_Type structure to a 
#    Ref_path_Type object
#    """
#    proxy = cgutils.create_struct_proxy(typ)(
#            c.context, c.builder, value=val)
#    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(Ref_path))
#
#    ref_path_obj = c.box(typ.ref_path, proxy.ref_path)
#    ref_index_xr_obj = c.box(typ.ref_index_xr, proxy.ref_index_xr)
#    ref_xr_obj = c.box(typ.ref_xr, proxy.ref_xr)
#    ref_div_iter_obj = c.pyapi.long_from_longlong(proxy.ref_div_iter)
#    drift_obj = c.box(typ.drift, proxy.drift)
#    dx_obj = c.box(typ.dx, proxy.dx)
#
#    Ref_path_obj = c.pyapi.call_function_objargs(
#        classobj,
#        (ref_path_obj, ref_index_xr_obj, ref_xr_obj, ref_div_iter_obj,
#         drift_obj, dx_obj)
#    )
#    c.pyapi.decref(classobj)
#    c.pyapi.decref(ref_path_obj)
#    c.pyapi.decref(ref_index_xr_obj)
#    c.pyapi.decref(ref_xr_obj)
#    c.pyapi.decref(ref_div_iter_obj)
#    c.pyapi.decref(drift_obj)
#    c.pyapi.decref(dx_obj)
#
#    return Ref_path_obj
#
#
#@overload_method(Ref_path_Type, 'get')
#def ref_path_get(path, idx, prev_idx, curr_xr):
#    """
#    Alternative to getitem which also takes as input prev_idx, curr_xr :
#    allows to optimize the look-up of Xrange values in case of successive calls
#    with strictly increasing idx.
#
#    idx :
#        index requested
#    (prev_idx, curr_xr) :
#        couple returned from last call, last index requested + next xr target
#        Contract : curr_xr the smallest integer that verify :
#            prev_idx <= ref_index_xr[curr_xr]
#            or curr_xr = ref_index_xr.size (No more xr)
#    Returns
#    -------
#    (val, xr_val, is_xr, prev_idx, curr_xr)
#        val : np.complex128
#        xr_val : complex128_Xrange_scalar
#        is_xr : bool
#        prev_idx : int
#        curr_xr : int (index in path.ref_xr)
#    """
#    def impl(path, idx, prev_idx, curr_xr):
#
#        # null_item = fsxn.czero
#        #null_item = None
#
#        if not(path.has_xr):
#            return (
#                path.ref_path[idx], czero(), False,
#                idx, curr_xr
#            )
#        ref_index_xr = path.ref_index_xr
#
#        # Not an increasing sequence, reset to restart a new sequence
#        if idx < prev_idx:
#            # Rewind to 0
#            curr_xr = 0
#            prev_idx = 0
#
#        # In increasing sequence (idx >= prev_idx)
#        if ((curr_xr >= ref_index_xr.size)
#                or (idx < ref_index_xr[curr_xr])):
#            return (
#                path.ref_path[idx], czero(), False,
#                idx, curr_xr
#            )
#        elif idx == ref_index_xr[curr_xr]:
#            xr = path.ref_xr[curr_xr] 
#            xr = Xrange_scalar(xr.mantissa, xr.exp)
#            return (
#                path.ref_path[idx], Xrange_scalar(xr.mantissa, xr.exp),
#                True, idx, curr_xr
#            )
#        else:  # idx > ref_index_xr[curr_xr]:
#            while ((idx > ref_index_xr[curr_xr])
#                   and (curr_xr < ref_index_xr.size)):
#                curr_xr += 1
#            if ((curr_xr == ref_index_xr.size)
#                    or (idx < ref_index_xr[curr_xr])):
#                return (
#                    path.ref_path[idx], czero(), False,
#                    idx, curr_xr
#                )
#            # Here idx == ref_index_xr[curr_xr]
#            xr = path.ref_xr[curr_xr]
#            xr = Xrange_scalar(xr.mantissa, xr.exp)
#            return (
#                path.ref_path[idx], Xrange_scalar(xr.mantissa, xr.exp),
#                True, idx, curr_xr
#            )
#
#    return impl
#
#@overload_method(Ref_path_Type, 'c_from_pix')
#def ref_path_c_from_pix(path, pix):
#    """
#    Returns the true c (coords from ref point) from the pixel coords
#    
#    Parameters
#    ----------
#    pix :  complex
#        pixel location in farction of dx
#        
#    Returns
#    -------
#    c, c_xr : c value as complex and as Xrange
#    """
#    def impl(path, pix):
#        c_xr = (pix * path.dx[0]) + path.drift[0]
#        return to_standard(c_xr), c_xr
#    return impl
##
##
###==============================================================================
###  We compute & store asynchronously the coefficients for SA intetpolation
###  for every multiple of the following seeds : 
###     - (2 ** min_exp, 2 ** (min_exp + 1), ...2 ** (max_exp)
###  For a location, knowing exp and multiple the storage index is:
###     - sto = stores[exp] + multiple
### Pushing to the array from a thread is protected by a lock to avoid race
### condition
##
##
##def make_Bivar_interpolator(Ref_path, SA_loop, kc, SA_params):
##    """
##    Builds the data
##    """
##    cutdeg = SA_params["cutdeg"]
##
##    min_seed_exp = fssettings.bivar_SA_min_seed_exp
##    max_jump = Ref_path.ref_div_iter - 1 # as path includes 0
##    max_seed_exp = max_jump.bit_length() - 1
##    # Note : max_seed = (1 << max_seed_exp)
##    bivar_SA_sto = np.zeros((max_seed_exp + 1,), dtype=np.int32)
##
##    print("max_jump", max_jump, max_seed_exp, 2**max_seed_exp, 1 << max_seed_exp)
##    for i in range(min_seed_exp, max_seed_exp + 1):
##        print("seed exp:", i)
##        bivar_SA_sto[i] = max_jump // (1 << i)
##    SA_sto_sum = np.cumsum(bivar_SA_sto)
##    n_sto = SA_sto_sum[-1]
##    bivar_SA_sto[1:] = SA_sto_sum[0:-1]
##
##    bivar_SA_lock = np.zeros((n_sto,), dtype=np.intp)
##    bivar_SA_coeffs = fsx.Xrange_array.zeros(
##            (n_sto, cutdeg, cutdeg), dtype=np.complex128)
##    bivar_SA_computed = np.zeros((n_sto, 2), dtype=np.bool_)
##    bivar_SA_zrad = fsx.Xrange_array.zeros(
##            (n_sto, 2), dtype=np.float64)
##    
##    print("Making a Bivar_interpolator with nsto:", n_sto)
##    print("bivar_SA_sto:", bivar_SA_sto)
##    print("with SA_params:", SA_params)
##
##    return Bivar_interpolator(
##        Ref_path=Ref_path,
##        SA_loop=SA_loop,
##        min_seed_exp=min_seed_exp,
##        max_seed_exp=max_seed_exp,
##        bivar_SA_cutdeg=SA_params["cutdeg"],
##        bivar_SA_kc=kc,
##        bivar_SA_eps=SA_params["eps"],
##        bivar_SA_lock=bivar_SA_lock, 
##        bivar_SA_sto=bivar_SA_sto,
##        bivar_SA_coeffs=bivar_SA_coeffs, 
##        bivar_SA_computed=bivar_SA_computed, 
##        bivar_SA_zrad=bivar_SA_zrad, 
##    )
##    
##
##class Bivar_interpolator:
##    
##    attr_list = (
##        "Ref_path",
##        "SA_loop",
##        "min_seed_exp",
##        "max_seed_exp",
##        "bivar_SA_cutdeg",
##        "bivar_SA_kc",
##        "bivar_SA_eps",
##        "bivar_SA_lock", 
##        "bivar_SA_sto",
##        "bivar_SA_coeffs",   # read-write
##        "bivar_SA_computed", # read-write
##        "bivar_SA_zrad",  # read-write
##    )
##    
##    def __init__(self, *args, **kwargs):
##        """
##        path: Ref_path object
##        min_exp: integer
##        SA_loop: numba function which take a 
##
##        """
##        for i, val in enumerate(args):
##            print("set", self.attr_list[i])
##            setattr(
##                self,
##                self.attr_list[i],
##                val
##            )
##        for key, val in kwargs.items():
##            setattr(self, key, val)
##
##
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Numba implementation
###
##class Bivar_interpolator_Type(types.Type):
##    def __init__(self, SA_loop):
##        self.Ref_path = Ref_path_Type()
##        self.SA_loop = numba.typeof(SA_loop)
##        self.min_seed_exp = types.int64
##        self.max_seed_exp = types.int64
##        self.bivar_SA_cutdeg = types.int64
##        self.bivar_SA_kc = types.Array(
##            numba_xr_type(np.float64), 1, 'C'
##        ) # kc -> kcX
##        self.bivar_SA_eps = types.float64
##        self.bivar_SA_lock = types.Array(numba.intp, 1, 'C')
##        self.bivar_SA_sto = types.Array(numba.int32, 1, 'C')
##        self.bivar_SA_coeffs = types.Array(
##            numba_xr_type(np.complex128), 3, 'C'
##        ) # store_address x nX x nY
##        self.bivar_SA_computed = types.Array(
##            numba.bool_, 2, 'C') # store_address x 2 (cv / div)
##        self.bivar_SA_zrad = types.Array(
##            numba_xr_type(np.float64), 2, 'C'
##        ) # store_address x 2 (cv / div)
##        # The name must be unique if the underlying model is different
##        super().__init__("Bivar_interpolator")#_{}".format(str(self.SA_loop)))
##
##@typeof_impl.register(Bivar_interpolator)
##def typeof_bivar_interpolator(val, c):
##    return Bivar_interpolator_Type(val.SA_loop)
##
##@register_model(Bivar_interpolator_Type)
##class BivarInterpolatorModel(models.StructModel):
##    def __init__(self, dmm, fe_type):
##        members = list(
##            (attr, getattr(fe_type, attr))
##            for attr in Bivar_interpolator.attr_list
##        )
##        models.StructModel.__init__(self, dmm, fe_type, members)
##
##for attr in Bivar_interpolator.attr_list:
##    make_attribute_wrapper(Bivar_interpolator_Type, attr, attr)
##
##@unbox(Bivar_interpolator_Type)
##def unbox_bivar_interpolator(typ, obj, c):
##    """ Convert a Bivar_interpolator object to a native structure """
##    Ref_path_obj = c.pyapi.object_getattr_string(obj, "Ref_path")
##    SA_loop_obj = c.pyapi.object_getattr_string(obj, "SA_loop")
##    min_seed_exp_obj = c.pyapi.object_getattr_string(obj, "min_seed_exp")
##    max_seed_exp_obj = c.pyapi.object_getattr_string(obj, "max_seed_exp")
##    bivar_SA_cutdeg_obj = c.pyapi.object_getattr_string(obj, "bivar_SA_cutdeg")
##    bivar_SA_kc_obj = c.pyapi.object_getattr_string(obj, "bivar_SA_kc")
##    bivar_SA_eps_obj = c.pyapi.object_getattr_string(obj, "bivar_SA_eps")
##    bivar_SA_lock_obj = c.pyapi.object_getattr_string(obj, "bivar_SA_lock")
##    bivar_SA_sto_obj = c.pyapi.object_getattr_string(obj, "bivar_SA_sto")
##    bivar_SA_coeffs_obj = c.pyapi.object_getattr_string(obj, "bivar_SA_coeffs")
##    bivar_SA_computed_obj = c.pyapi.object_getattr_string(obj, "bivar_SA_computed")
##    bivar_SA_zrad_obj = c.pyapi.object_getattr_string(obj, "bivar_SA_zrad")
##
##    proxy = cgutils.create_struct_proxy(typ)(c.context, c.builder)
##
##    proxy.Ref_path = c.unbox(typ.Ref_path, Ref_path_obj).value
##    proxy.SA_loop  = c.unbox(typ.SA_loop, SA_loop_obj).value
##    proxy.min_seed_exp = c.pyapi.long_as_longlong(min_seed_exp_obj)
##    proxy.max_seed_exp = c.pyapi.long_as_longlong(max_seed_exp_obj)
##    proxy.bivar_SA_cutdeg = c.pyapi.long_as_longlong(bivar_SA_cutdeg_obj)
##    proxy.bivar_SA_kc = c.unbox(typ.bivar_SA_kc, bivar_SA_kc_obj).value
##    proxy.bivar_SA_eps = c.pyapi.float_as_double(bivar_SA_eps_obj)
##    proxy.bivar_SA_lock = c.unbox(typ.bivar_SA_lock, bivar_SA_lock_obj).value
##    proxy.bivar_SA_sto = c.unbox(typ.bivar_SA_sto, bivar_SA_sto_obj).value
##    proxy.bivar_SA_coeffs = c.unbox(typ.bivar_SA_coeffs, bivar_SA_coeffs_obj).value
##    proxy.bivar_SA_computed = c.unbox(typ.bivar_SA_computed, bivar_SA_computed_obj).value
##    proxy.bivar_SA_zrad = c.unbox(typ.bivar_SA_zrad, bivar_SA_zrad_obj).value
##
##    # Free mem
##    c.pyapi.decref(Ref_path_obj)
##    c.pyapi.decref(SA_loop_obj)
##    c.pyapi.decref(min_seed_exp_obj)
##    c.pyapi.decref(max_seed_exp_obj)
##    c.pyapi.decref(bivar_SA_cutdeg_obj)
##    c.pyapi.decref(bivar_SA_kc_obj)
##    c.pyapi.decref(bivar_SA_eps_obj)
##    c.pyapi.decref(bivar_SA_lock_obj)
##    c.pyapi.decref(bivar_SA_sto_obj)
##    c.pyapi.decref(bivar_SA_coeffs_obj)
##    c.pyapi.decref(bivar_SA_computed_obj)
##    c.pyapi.decref(bivar_SA_zrad_obj)
##
##    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
##    return NativeValue(proxy._getvalue(), is_error=is_error)
##
##
##@box(Bivar_interpolator_Type)
##def box_bivar_interpolator(typ, val, c):
##    """
##    Convert a native Ref_path_Type structure to a 
##    Ref_path_Type object
##    """
##    proxy = cgutils.create_struct_proxy(typ
##        )(c.context, c.builder, value=val)
##    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(Bivar_interpolator))
##
##    # attr
##    Ref_path_obj = c.box(typ.Ref_path, proxy.Ref_path)
##    SA_loop_obj = c.box(typ.SA_loop, proxy.SA_loop)
##    min_seed_exp_obj = c.pyapi.long_from_longlong(proxy.min_seed_exp)
##    max_seed_exp_obj = c.pyapi.long_from_longlong(proxy.max_seed_exp)
##    bivar_SA_cutdeg_obj = c.pyapi.long_from_longlong(proxy.bivar_SA_cutdeg)
##    bivar_SA_kc_obj = c.box(typ.bivar_SA_kc, proxy.bivar_SA_kc)
##    bivar_SA_eps_obj = c.pyapi.float_from_double(proxy.bivar_SA_eps)
##    bivar_SA_lock_obj = c.box(typ.bivar_SA_lock, proxy.bivar_SA_lock)
##    bivar_SA_sto_obj = c.box(typ.bivar_SA_sto, proxy.bivar_SA_sto)
##    bivar_SA_coeffs_obj = c.box(typ.bivar_SA_coeffs, proxy.bivar_SA_coeffs)
##    bivar_SA_computed_obj = c.box(typ.bivar_SA_computed, proxy.bivar_SA_computed)
##    bivar_SA_zrad_obj = c.box(typ.bivar_SA_zrad, proxy.bivar_SA_zrad)
##
##    Bivar_interpolator_obj = c.pyapi.call_function_objargs(
##        classobj,
##        (Ref_path_obj,
##        SA_loop_obj,
##        min_seed_exp_obj,
##        max_seed_exp_obj,
##        bivar_SA_cutdeg_obj,
##        bivar_SA_kc_obj,
##        bivar_SA_eps_obj,
##        bivar_SA_lock_obj,
##        bivar_SA_sto_obj,
##        bivar_SA_coeffs_obj,
##        bivar_SA_computed_obj,
##        bivar_SA_zrad_obj,
##    ))
##    
##    # Free mem
##    c.pyapi.decref(Ref_path_obj)
##    c.pyapi.decref(SA_loop_obj)
##    c.pyapi.decref(min_seed_exp_obj)
##    c.pyapi.decref(max_seed_exp_obj)
##    c.pyapi.decref(bivar_SA_cutdeg_obj)
##    c.pyapi.decref(bivar_SA_kc_obj)
##    c.pyapi.decref(bivar_SA_eps_obj)
##    c.pyapi.decref(bivar_SA_lock_obj)
##    c.pyapi.decref(bivar_SA_sto_obj)
##    c.pyapi.decref(bivar_SA_coeffs_obj)
##    c.pyapi.decref(bivar_SA_computed_obj)
##    c.pyapi.decref(bivar_SA_zrad_obj)
##
##    return Bivar_interpolator_obj
##
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
###@numba.njit
###def enlarge_z_sq(z_sq):
###    m, e = _normalize(z_sq.mantissa, z_sq.exp)
###    assert e <= 0
###    return Xrange_scalar(
###            np.sm,
###            sdiv_int32(e, numba.int32(2))
###    )
##
###@overload_method(Bivar_interpolator_Type, 'is_seed')
###def bivar_is_seed(bi, iteration):
###    """
###    Returns True if idx is a multiple of 2 ** self.min_exp
###    """
###    def impl(bi, iteration):
###        return (iteration % (1 << bi.min_seed_exp)) == 0
###    return impl
##
##@overload_method(Bivar_interpolator_Type, 'seed_coords')
##def bivar_seed_coords(bi, idx):
##    """
##    Returns the couple (exp, factor)
##    exp is the largest integer for which holds: 
##        n = factor * 2 ** exp2
##    """
##    def impl(bi, idx):
##        print("entering seed_coords")
##        if idx == 0:
##            return bi.max_seed_exp, numba.int64(0)
##        exp = bi.min_seed_exp
##        assert (idx % numba.int64(1 << exp)) == 0
##        bits = numba.int64(idx >> exp)
##        print("bits", bits)
##        while not(bits & 1):
##            bits >>= 1
##            exp += 1
##        return numba.int64(exp), numba.int64(idx // (1 << exp))
##    return impl
##
##
##@overload_method(Bivar_interpolator_Type, 'null_poly_arr')
##def bivar_null_poly_arr(bi):
##    """ Compute and try to store
##    Returns
##    poly_incr, poly, kz
##    
##    -> to retrieve the values need to do P(z/z_sq, c/kc)
##    -> to retrieve the derivatives dc need to do dPdc()
##    """
##    arr_template = np.zeros((1,), dtype=numpy_xr_type(np.complex128))
##
##    def impl(bi):
##        arr_len = bi.bivar_SA_cutdeg + 1
##        coeffs = arr_template.repeat(arr_len ** 2).reshape(arr_len, arr_len)
##        return coeffs
##    return impl
##
##
##@overload_method(Bivar_interpolator_Type, 'get_poly')
##def bivar_get_poly(bi, iteration, z):
##    """ User-facing - Try to pull otherwise compute
##
##    iteration is fixed, but not the couple (exp, factor)
##    
##    Returns
##    0, None
##    poly_incr, poly, kz
##    """
##    print("cmp bivar_get_poly", iteration, z)
##
##    kz_template = np.zeros((1,), dtype=numpy_xr_type(np.float64))
##
##
##    if z in xr_types:
##        def impl(bi, iteration, z):
##            print("USER get_poly", iteration, z)
##            print("sto", bi.bivar_SA_sto.shape, "\n", bi.bivar_SA_sto)
##            (exp, factor) = bi.seed_coords(iteration)
##            print(exp, factor)
##            z_abs = np.abs(z)
##            # ptr to results
##            poly_incr = np.zeros((1,), dtype=np.int64)
##            poly_arr = bi.null_poly_arr()
##            kz_abs = kz_template.repeat(2)  # cv, div
##            print("call _get_poly")
##            bi._get_poly(exp, factor, z_abs, poly_incr, poly_arr, kz_abs)
##
##            # kz = np.sqrt(kz_sq[0])  #TODO
##            return poly_incr[0], poly_arr, to_Xrange_scalar(kz_abs[0])
##                     #poly_incr[0] #, poly#, kz_abs[0]
###    else:
###        def impl(bi, iteration, z):
###            print("USER get_poly", iteration, z)
###            print("sto", bi.bivar_SA_sto.shape, "\n", bi.bivar_SA_sto)
###            (exp, factor) = bi.seed_coords(iteration)
###            z_xr = to_Xrange_scalar(z)
###            z_sq = extended_abs2(z_xr)
###            poly_incr, poly, kz = bi._get_poly(exp, factor, z_sq)
###            return poly_incr, poly, kz
##    return impl
##
##@overload_method(Bivar_interpolator_Type, '_get_poly')
##def bivar__get_poly(bi, exp, factor, z_abs, poly_incr, poly_arr, kz_abs):
##    """ Internal - Try to pull otherwise compute
##
##    iteration is fixed, ALSO the couple (exp, factor)
##
##    Returns None
##    Modify in place
##    poly_incr, poly, kz_sq
##    """    
##    print("cmp bivar__get_poly", bi, exp, factor, z_abs)
##
##    computed_index = 0
##    valid_index = 1
##    div_computed_index = 2
##
##    zrad_cv_index = 0
##    zrad_div_index = 1
##    
##    def impl(bi, exp, factor, z_abs,
##             poly_incr, poly_arr, kz_abs):
##        
##
##        print("_get_poly", exp, factor, z_abs)
##
##        run_recursive = True
##        while run_recursive:
##            bool_arr = np.zeros((3,), dtype=numba.bool_) # computed, valid, div_computed
##            bi.pull_poly(  
##                exp, factor, z_abs,
##                poly_arr, kz_abs, bool_arr
##            )
##    
##            if (bool_arr[computed_index] * bool_arr[valid_index]):
##                # We validate poly_arr & kz_sq
##                poly_incr[0] = numba.int64(1 << exp)
##                return
##
##            elif (bool_arr[computed_index] * ~bool_arr[valid_index]):
##                print("computed and NOT valid")
##                zrad_div = to_Xrange_scalar(kz_abs[zrad_div_index])
##                # Try SA with a larger cv radius ?
##                z_abs_larger = np.sqrt(z_abs)
##
##                if (z_abs_larger > fssettings.std_zoom_level ** 2):
##                    # Too big, no SA possible
##                    poly_incr[0] = numba.int64(0)
##                    return
##
##                elif (bool_arr[div_computed_index]
##                      and (zrad_div <= z_abs_larger)):
##                    # We already know SA with this larger z fails. Try rather a 
##                    # smaller exp / smaller poly_incr
##                    if exp > bi.min_seed_exp:
##                        # Recursive call
##                        exp -= 1
##                        factor *= 2
##                    else:
##                        # End of recursive loop, min exp already reached
##                        run_recursive = False
##                else:
##                    # this z_sq_larger has not been computed, worth trying
##                    bi.compute_poly(
##                        exp, factor, z_abs_larger, 
##                        poly_incr, poly_arr, kz_abs
##                    )
##                    return
##
##            # else: not computed
##            else:
##                bi.compute_poly(exp, factor, z_abs,
##                                poly_incr, poly_arr, kz_abs)
##                return
##
##        # End of recursive loop : No SA possible
##        poly_incr[0] = numba.int64(0)
##        return
##
##    return impl
##
##
##@overload_method(Bivar_interpolator_Type, 'compute_poly')
##def bivar_compute_poly(bi, exp, factor, z_abs,
##                        poly_incr, poly_arr, kz_abs):
##    """ Compute and try to store
##    Returns
##    0, None
##    poly_incr, poly, kz
##    
##    -> to retrieve the values need to do P(z/z_sq, c/kc)
##    -> to retrieve the derivatives dc need to do dPdc()
##    """
##    print("cmp bivar_compute_poly")
##
##    arr_template = np.zeros((1,), dtype=numpy_xr_type(np.complex128))
##    err_template = np.zeros((1,), dtype=numpy_xr_type(np.float64))
##    
##    zrad_cv_index = 0
##    zrad_div_index = 1
##
##    def impl(bi, exp, factor, z_abs,
##                        poly_incr, poly_arr, kz_abs):
##
##        print("bivar_compute_poly", exp, factor, z_abs)
##        poly_incr[0] = (1 << exp)
##        iteration = factor * poly_incr[0] 
##
##        Ref_path = bi.Ref_path
##        cutdeg = bi.bivar_SA_cutdeg
##        arr_len = cutdeg + 1
##        # Builds a SA(X, Y) == Y/z_sq
##        SA_coeffs = arr_template.repeat(arr_len ** 2).reshape(arr_len, arr_len)
##        SA_coeffs[0, 1] = z_abs
##        SA = fsx.Xrange_bivar_SA(SA_coeffs, cutdeg, err=err_template.copy())
##        kcX = fsx.Xrange_monome(bi.bivar_SA_kc) 
##        SA_loop = bi.SA_loop
##        eps_sq = bi.bivar_SA_eps ** 2
##
##        prev_idx = 0
##        curr_xr = 0
##        for i in range(iteration, iteration + poly_incr[0]):
##            (val, xr_val, is_xr, prev_idx, curr_xr
##             ) = Ref_path.get(i, prev_idx, curr_xr)
##            if is_xr:
##                ref_path_item = xr_val
##            else:
##                ref_path_item = to_Xrange_scalar(val)
##
##            SA = SA_loop(SA, i, ref_path_item, kcX) # debug
##
##            ssum = SA.ssum() # sum of all coeff abs2
##            err_abs2 = SA.err[0] * SA.err[0]
##            SA_valid = ((err_abs2  <= eps_sq * ssum) and (ssum <= 1.e6))
##
##            if not(SA_valid):
##                break
##
##        if SA_valid:
##            # We store it and return
##            coeffs = SA.coeffs
##            for i in range(cutdeg + 1):
##                for j in range(cutdeg + 1 - i):
##                    poly_arr[i, j] = coeffs[i, j]
##
##            bi.push_poly(exp, factor, poly_arr, z_abs) 
##            kz_abs[zrad_cv_index] = z_abs
##            return
##
##        # SA invalid We store the dic radius and return invalid
##        bi.push_div(exp, factor, z_abs)
##        poly_incr[0] = 0
##        return
##
##    return impl
##
##
##
##@overload_method(Bivar_interpolator_Type, 'pull_poly')
##def bivar_pull_poly(bi, exp, factor, z_abs,
##                    poly_arr, kz_abs, bool_arr):
##    """
##    Return the stored SA data for this seed
##
##    Returns
##    -------
##    (computed, valid, sq_zrad_cv, div_computed, sq_zrad_div, bivar_poly)
##    """
##    print("cmp bivar_pull_poly", bi, exp, factor, z_abs)
##
##    computed_index = 0
##    valid_index = 1
##    div_computed_index = 2
##    
##    zrad_cv_index = 0
##    zrad_div_index = 1
##    
##    def impl(bi, exp, factor, z_abs,
##             poly_arr, kz_abs, bool_arr):
##        
##        print("bivar_pull_poly", exp, factor, z_abs)
##
##        cutdeg = bi.bivar_SA_cutdeg
##        sto = bi.bivar_SA_sto[exp] + factor
##        print("bivar_pull_poly STO", sto)
##        
##        # busywait to lock and do some work
##        lock_ptr = bi.bivar_SA_lock[sto:]
##        while not(try_lock(lock_ptr)):
##            pass
##
##        bool_arr[computed_index] = bi.bivar_SA_computed[sto, 0]
##        bool_arr[div_computed_index] = bi.bivar_SA_computed[sto, 1]
##        kz_abs[zrad_cv_index] = to_Xrange_scalar(bi.bivar_SA_zrad[sto, 0])
##        kz_abs[zrad_div_index] = to_Xrange_scalar(bi.bivar_SA_zrad[sto, 1])
##        bool_arr[valid_index] = (z_abs <= kz_abs[zrad_cv_index] )
##
##        if ~(bool_arr[computed_index] and bool_arr[valid_index]): 
##            # early exit, poly_arr is not useable
##            unlock(lock_ptr)
##            return
##
##        cutdeg = bi.bivar_SA_cutdeg
##        for i in range(cutdeg + 1):
##            for j in range(cutdeg + 1 - i):
##                 poly_arr[i, j] = bi.bivar_SA_coeffs[sto, i, j]
##        unlock(lock_ptr)
##        return
##
##    return impl
##
##@overload_method(Bivar_interpolator_Type, 'push_poly')
##def bivar_push_poly(bi, exp, factor, poly_arr, z_abs):
##    """
##    Try to store the newly computed SA.
##    As it is shared memory, a lock is used to avoid race conditions
##    If another thread already holds the lock, we simply do not save.
##    
##    note : z_sq should be one of the preset increment
##    """
##    print("cmp bivar_push_poly")
##
##    def impl(bi, exp, factor, poly_arr, z_abs):
##        sto = bi.bivar_SA_sto[exp] + factor
##        print("bivar_push_poly", exp, factor, z_abs, sto)
##
##        # busywait to lock and do some work
##        lock_ptr = bi.bivar_SA_lock[sto:]
##        while not(try_lock(lock_ptr)):
##            pass
##
##        # Here we check that the new SA is actually an improvement vs current
##        # situation
##        computed = bi.bivar_SA_computed[sto, 0]
##        div_computed = bi.bivar_SA_computed[sto, 1]
##        zrad_cv = to_Xrange_scalar(bi.bivar_SA_zrad[sto, 0])
##        zrad_div = to_Xrange_scalar(bi.bivar_SA_zrad[sto, 1])
##        valid = (z_abs <= zrad_cv)
##
##        improvement = not(computed and valid)
##        if improvement:
##            # update bivar_SA_coeffs, bivar_SA_zrad, bivar_SA_computed
##            cutdeg = bi.bivar_SA_cutdeg
##            for i in range(cutdeg + 1):
##                for j in range(cutdeg + 1 - i):
##                    bi.bivar_SA_coeffs[sto, i, j] = poly_arr[i, j]
##
##            bi.bivar_SA_zrad[sto, 0] = z_abs
##            if not computed:
##                bi.bivar_SA_computed[sto, 0] = True
##            # Sanity check : 
##            if div_computed:
##                assert z_abs < zrad_div
##
##        # unlock /!\ no early return here otherwise we will hold the lock
##        unlock(lock_ptr)
##    return impl
##
##@overload_method(Bivar_interpolator_Type, 'push_div')
##def bivar_push_div(bi, exp, factor, z_abs):
##    """
##    Try to store the newly computed radius for failure
##    As it is shared memory, a lock is used to avoid race conditions
##    If another thread already holds the lock, we simply do not save.
##    
##    note : z_sq should be one of the preset increment
##    """
##    print("cmp bivar_push_div")
##
##    def impl(bi, exp, factor, z_abs):
##        sto = bi.bivar_SA_sto[exp] + factor
##        print("bivar_push_div", exp, factor, z_abs, sto)
##
##        # busywait to lock and do some work
##        lock_ptr = bi.bivar_SA_lock[sto:]
##        while not(try_lock(lock_ptr)):
##            pass
##
##        # Here we check that the new invalid is actually an improvement vs
##        # current situation
##        computed = bi.bivar_SA_computed[sto, 0]
##        div_computed = bi.bivar_SA_computed[sto, 1]
##        zrad_cv = to_Xrange_scalar(bi.bivar_SA_zrad[sto, 0])
##        zrad_div = to_Xrange_scalar(bi.bivar_SA_zrad[sto, 1])
##        
##        improvement = False
##        if div_computed:
##            improvement = (z_abs < zrad_div)
##
##        if improvement:
##            # update bivar_SA_coeffs, bivar_SA_zrad, bivar_SA_computed
##            bi.bivar_SA_zrad[sto, 1] = z_abs
##            if not div_computed:
##                bi.bivar_SA_computed[sto, 1] = True
##            # Sanity check : 
##            if computed:
##                assert z_abs > zrad_cv
##
##        # unlock /!\ no early return here otherwise we will hold the lock
##        unlock(lock_ptr)
##    return impl
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
#
#
#
#
#
#
#
#
#
#
#
#
#
#print("======================================================================")
#print("IMPORTED NUMBA XR ====================================================")
#print("======================================================================")
##==============================================================================
## DEV
#
#
#@numba.njit
#def test_poly(pol):
#    print("in numba")
##    print("nb", pol, pol.coeffs.dtype)
#    # pol.coeffs[1] = Xrange_scalar(45678., numba.int32(0))
#    coeff2 = pol.coeffs.copy() # * Xrange_scalar(2., numba.int32(0))
##    print("nb", coeff2)
#    for i in range(len(coeff2)):
#        coeff2[i] = coeff2[i] * coeff2[i] 
#    p2 = - fsx.Xrange_polynomial(coeff2, 2)
##    print("init", p2.cutdeg, p2.coeffs)
#    print("p2 init", p2.cutdeg, p2.coeffs)
#    return p2
#
#@numba.njit
#def test_poly_call(pol):
#    print("in numba")
##    print("nb", pol, pol.coeffs.dtype)
#    # pol.coeffs[1] = Xrange_scalar(45678., numba.int32(0))
##    coeff2 = pol.coeffs.copy() # * Xrange_scalar(2., numba.int32(0))
###    print("nb", coeff2)
##    for i in range(len(coeff2)):
##        coeff2[i] = coeff2[i] * coeff2[i] 
##    p2 = - fsx.Xrange_polynomial(coeff2, 2)
###    print("init", p2.cutdeg, p2.coeffs)
##    print("p2 init", p2.cutdeg, p2.coeffs)
#    return pol.__call__(Xrange_scalar(2., np.int32(0)))
#
#
#@numba.njit
#def test_polyadd(pol1, pol2):
#    print("in numba")
#    return pol1 + pol2
#
##@numba.njit
##def test_iadd(arr, val):
##    print("in numba")
##    for i in range(len(arr)):
##        arr[i] += val[0]
##    return arr
#@numba.njit
#def test_sa(poly):
#    print("in numba")
#    sa = poly.to_SA()
##    print("nb", pol, pol.coeffs.dtype)
#    # pol.coeffs[1] = Xrange_scalar(45678., numba.int32(0))
#    coeff2 = sa.coeffs.copy() # * Xrange_scalar(2., numba.int32(0))
#    print("coeffs", coeff2)
##    print("err", sa.err.mantissa, sa.err.exp)
#    return sa
#
#@numba.njit
#def box_add_sa(poly1, poly2):
#    print("in numba")
#    sa1 = poly1.to_SA()
#    sa2 = poly2.to_SA()
#    err = Xrange_scalar(1., numba.int32(1))
#    print("err###", err.mantissa)
#    sa1.err[0] = err
#    print("err in SA ###", sa1.err)
##    print("nb", pol, pol.coeffs.dtype)
#    # pol.coeffs[1] = Xrange_scalar(45678., numba.int32(0))
#    sa_res = sa1 + sa2 # * Xrange_scalar(2., numba.int32(0))
#
#    #print("err", sa.err.mantissa, sa.err.exp)
#    return sa_res
#
#@numba.njit
#def box_add_bivar(poly1, poly2):
#    print("in numba")
##    sa1 = poly1.to_SA()
##    sa2 = poly2.to_SA()
##    err = Xrange_scalar(1., numba.int32(1))
##    print("err###", err.mantissa)
##    sa1.err[0] = err
##    print("err in SA ###", sa1.err)
##    print("nb", pol, pol.coeffs.dtype)
#    # pol.coeffs[1] = Xrange_scalar(45678., numba.int32(0))
#    bivar_res = poly1 # + poly2 # * Xrange_scalar(2., numba.int32(0))
#
#    #print("err", sa.err.mantissa, sa.err.exp)
#    return -bivar_res
#
#
#@numba.njit
#def test_saadd(sa1, sa2):
#    print("in numba")
#    return sa1 + sa2
#
#if __name__ == "__main__":
#    arr0 = fsx.Xrange_array(["1.e100", "3.14", "2.0"]) #* (1. + 1j)
#    pol0 = fsx.Xrange_polynomial(arr0, 2)
#    arr1 = fsx.Xrange_array(["2.e100", "-3.14", "-0.2"]) #* (1. + 1j)
#    pol1 = fsx.Xrange_polynomial(arr1, 2)
#    res = test_sa(pol0)
#    print("res", res)
#    res = box_add_sa(pol0, pol1)
#    print("res", res)
#    
#    sa0 = fsx.Xrange_SA(arr0, 2, fsx.Xrange_array(8.))
#    sa1 = fsx.Xrange_SA(arr1, 2, fsx.Xrange_array(8.))
#    res = test_saadd(sa0, sa1)
#    print("res", res)
#    
#    bivar_arr = [
#            [11., 12., 13. , 14., 15],
#            [21., 22., 23. , 24., 25],
#            [31., 32., 33. , 34., 35],
#            [41., 42., 43. , 44., 45],
#            [51., 52., 53. , 54., 55]
#        ]
#    bivar_pol0 = fsx.Xrange_bivar_polynomial(bivar_arr, 2)
#    res = box_add_bivar(bivar_pol0, bivar_pol0)
#    print("bivar", res)
#    
##    print(pol0.coeffs)
##    print(np.asarray(pol0.coeffs).size)
##    print(pol0.coeffs.size, pol0.cutdeg + 1)
##    p_neg = test_poly(pol0)
##    print("p_neg", p_neg)
#
##    arr0 = fsx.Xrange_array(["0.", "1.", "2."]) #* (1. + 1j)
##    val = fsx.Xrange_array(["1."]) #* (1. + 1j)
##    test = test_iadd(arr0, val)
##    print("test", test)
##    pol0 = fsx.Xrange_polynomial(arr0, 2)
##    res = test_poly_call(pol0)
##    print(res.view(fsx.Xrange_array))
#    
##    arr1 = fsx.Xrange_array(["1.e100", "3.14", "2.0"]) #* (1. + 1j)
##    pol1 = fsx.Xrange_polynomial(arr1, 2)
##    
##    res =  test_polyadd(pol0, pol1)
##    print("res", res)
##    
##    arr0 = fsx.Xrange_array(["1.e100", "3.14", "2.0", "5.0"]) #* (1. + 1j)
##    pol0 = fsx.Xrange_polynomial(arr0, 3)
##    arr1 = fsx.Xrange_array(["1.e100", "3.14", "2.0", "6.4"]) #* (1. + 1j)
##    pol1 = fsx.Xrange_polynomial(arr1, 3)
##    
##    res =  test_polyadd(pol0, pol1)
##    print("res", res)
