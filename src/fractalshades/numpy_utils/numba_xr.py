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


