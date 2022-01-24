# -*- coding: utf-8 -*-
import sys
import numpy as np
import numbers
import re
import mpmath

#import time

def mpc_to_Xrange(mpc, dtype=np.complex128):
    """ Convert a mpc complex to a Xrange array"""
    select = {np.dtype(np.complex64): np.float32,
              np.dtype(np.complex128): np.float64}
    float_type = select[np.dtype(dtype)]

    mpcx_m, mpcx_exp = mpmath.frexp(mpc.real)
    mpcy_m, mpcy_exp = mpmath.frexp(mpc.imag)
    
    print("mpcx_m", type(mpcx_m), mpcx_m)
    print("float_type", type(float_type), float_type)
    mpcx_m = float_type(float(mpcx_m))
    mpcy_m = float_type(float(mpcy_m))

    if (mpcx_exp >  mpcy_exp):
        case = 1
    elif (mpcx_exp <  mpcy_exp):
        case = 2
    else:
        case = 3

    # Need to handle 0. re / im as special cases to avoid cancellation
    if mpcx_exp == 0.:
        case = 2
    if mpcy_exp == 0.:
        case = 1

    if case == 1:
        m = mpcx_m + 1j * np.ldexp(mpcy_m, mpcy_exp - mpcx_exp)
        exp = mpcx_exp
    elif case == 2:
        m = 1j * mpcy_m + np.ldexp(mpcx_m, mpcx_exp - mpcy_exp)
        exp = mpcy_exp
    else:
        m = mpcx_m + 1j * mpcy_m
        exp = mpcx_exp

    return Xrange_array(m, np.int32(exp))


def mpf_to_Xrange(mpf, dtype=np.float64):
    """ Convert a mpc float to a Xrange array"""
    mpf_m, mpf_exp = mpmath.frexp(mpf)
    return Xrange_array(dtype(float(mpf_m)), np.int32(mpf_exp))
            # np.array(mpf_m, dtype), np.array(mpf_exp, np.int32))

def Xrange_to_mpfc(arr):
    """ Convert a Xrange array of size-1 to a mpf or mpc"""
    if arr.is_complex:
        return Xrange_to_mpfc(arr.real) + 1.j * Xrange_to_mpfc(arr.imag)
    else:
        m = arr._mantissa
        exp = arr._exp
        return mpmath.ldexp(float(m), int(exp))

def get_xr_dtype(dtype):
    return np.dtype([('mantissa', dtype),
                     ('exp', np.int32)], align=False)

class Xrange_array(np.ndarray):
    """
Arrays class for "extended range" floats or complex numbers.
    This class allows to represent floating points numbers in simple or double
    precision in the range [1e-646456992, 1e+646456992].

Parameters:
    mantissa :  Can be either
        - a nd.array of dtype: one of the supported dtype float32, float64,
            complex64, complex128,
        - a string array, each item reprenting a float in standard or
            e-notation e.g. ["123.456e789", "-.3e-7", "1.e-1000", "1.0"]
            Note that list inputs are accepted in both cases, and passed
            through np.asarray() method.
    exp :  int32 array of shape sh, if None will default to 0
        ignored if mantissa is provided as string array.
    str_input_dtype : np.float32 of np.float64, only used if mantissa
        provided as a string, to allow specification of dataype.
        (if None or not provided, will default to float64 / complex128)

Return:
    Xrange_array of same shape as parameter 'mantissa' representing
        (real case): mantissa * 2**exp  
        (complex case): (mantissa.real + 1.j * mantissa.imag) * 2**exp

Usage:
    >>> Xa = Xrange_array([["123.456e-1789", "-.3e-7"], ["1.e700", "1.0"]])
    >>> print(Xa**2)
    [[ 1.52413839e-3574  9.00000000e-16]
     [ 1.00000000e+1400  1.00000000e+00]]
    
    >>> b = np.array([1., 1., np.pi, np.pi], dtype=np.float32)
    >>> Xb = Xrange_array(b)
    >>> for exp10 in range(1001):
            Xb = Xb * [-10., 0.1, 10., -0.1]
    >>> Xb
    <class 'arrays.Xrange_array'>
    shape: (4,)
    internal dtype: [('mantissa', '<f8'), ('exp_re', '<i4')]
    base 10 representation:
    [-1.00000000e+1001  1.00000000e-1001  3.14159274e+1001 -3.14159274e-1001]
    >>> print(Xb)
    [-1.00000000e+1001  1.00000000e-1001  3.14159274e+1001 -3.14159274e-1001]

Implementation details:
    Each scalar in the array is stored as a couple: 1 real or complex and
    1 int32 integer for an extra base-2 exponent. 
    The overall array is stored as a structured array of type :
        - (float32, int32)
        - (float64, int32)
        - (complex64, int32)
        - (complex128, int32)
    Hence, the mantissa can be one of 4 supported types :
        float32, float64, complex64, complex128

    Each class instance exposes the following properties ("views" of the base
    data array):
        real    view of real part, as a real Xrange_array (read only)
        imag    view of imaginary part, as a real Xrange_array (read only)
        is_complex  Scalar boolean

    The binary operations implemented are:
        +, -, *, /, <, <=, >, >=
    and their matching 'assignment' operators:
        +=, -=, *=, /=

    The unary operations implemented are :
        as numpy unfunc : abs, sqrt, square, conj, log, angle (through arctan2)
        as instance method : abs2 (square of abs)

    Xrange_array may silently over/underflow, due to the implementation of its
    exponent as a np.int32 array. If needed, checks for overflow shall be
    implemented downstream in user code.
        >>> np.int32(2**31)
        -2147483648

Reference:
    https://numpy.org/devdocs/user/basics.subclassing.html
     
    """
    _FLOAT_DTYPES = [np.float32, np.float64]
    _COMPLEX_DTYPES = [np.complex64, np.complex128]
    _DTYPES = _FLOAT_DTYPES + _COMPLEX_DTYPES
    _STRUCT_DTYPES = [get_xr_dtype(dt)  for dt in _DTYPES]
    # types that can be 'viewed' as Xrange_array:
    _HANDLED_TYPES = (np.ndarray, numbers.Number, list)
    #__array_priority__ =  20
    def __new__(cls, mantissa, exp=None, str_input_dtype=None):
        """
        Constructor
        """
        mantissa = np.asarray(mantissa)
        if mantissa.dtype.type == np.str_:
            mantissa, exp = np.vectorize(cls._convert_from_string)(mantissa)
            if str_input_dtype is not None:
                mantissa = np.asarray(mantissa, dtype=str_input_dtype)

        data = cls._extended_data_array(mantissa, exp)
        return super().__new__(cls, data.shape, dtype=data.dtype, buffer=data)

    @staticmethod
    def xr_scalar(mantissa, exp):
        """ Builds a Xrange_array of shape (,) from mantissa and exp """
        if type(mantissa) is float:
            ret = np.empty([], get_xr_dtype(np.float64))
        elif type(mantissa) is complex:
            ret = np.empty([], get_xr_dtype(np.complex128))
        ret["mantissa"] = mantissa
        ret["exp"] = exp
        return ret.view(Xrange_array)

    @staticmethod
    def _convert_from_string(input_str):
        """
        Return mantissa and base 2 exponent from float input string
        
        Parameters
        input_str: string (see exp_pattern for accepted patterns)
        
        Return
        m  : mantissa
        exp_re : base 2 exponent
        """
        exp_pattern = ("^([-+]?[0-9]+\.[0-9]*|[-+]?[0-9]*\.[0-9]+|[-+]?[0-9]+)"
                       "([eE]?)([-+]?[0-9]*)$")
        err_msg = ("Unsupported Xrange_array string item: <{}>\n" +
            "(Examples of supported input items: " +
            "<123.456e789>, <-.123e-127>, <+1e77>, <1.0>, ...)")

        match = re.match(exp_pattern, input_str)
        if match:
            m = float(match.group(1))
            exp_10 = 0
            if match.group(2) in ["e", "E"]:
                try:
                    exp_10 = int(match.group(3))
                    if abs(exp_10) > 646456992:
                        raise ValueError("Overflow int string input, cannot "
                            "represent exponent with int32, maxint 2**31-1")
                except ValueError:
                    raise ValueError(err_msg.format(input_str))
    # We need 96 bits precision for accurate mantissa in this base-10 to base-2
    # conversion, will use native Python integers, as speed is not critical
    # here.
    # >>> import mpmath
    # >>> mpmath.mp.dps = 30
    # >>> mpmath.log("10.") / mpmath.log("2.") * mpmath.mpf("1.e25")
    # mpmath.log("10.") / mpmath.log("2.") * mpmath.mpf(2**96)
    # mpf('263190258962436467100402834429.2138584375862')
                rr_hex = 263190258962436467100402834429
                exp_10, mod = divmod(exp_10 * rr_hex, 2**96)
                m *= 2.**(mod * 2.**-96)
            return m, exp_10
        else:
            raise ValueError(err_msg.format(input_str))

    @staticmethod
    def _extended_data_array(mantissa, exp):#, counter):#_re, exp_im):
        """
        Builds the structured internal array.
        """
        mantissa_dtype = mantissa.dtype
        if mantissa_dtype not in Xrange_array._DTYPES:
            casted = False
            for cast_dtype in Xrange_array._DTYPES:
                if np.can_cast(mantissa_dtype, cast_dtype, "safe"):
                    mantissa = mantissa.astype(cast_dtype)
                    mantissa_dtype = cast_dtype
                    casted = True
                    break
            if not casted:
                if (mantissa_dtype in Xrange_array._STRUCT_DTYPES
                        ) and (exp is None):
                    return mantissa # Pass-through
                raise ValueError("Unsupported type for Xrange_array {}".format(
                    mantissa_dtype))

        # Builds the field-array
        sh = mantissa.shape
        if exp is None:
            exp = np.zeros(sh, dtype=np.int32)

        extended_dtype = get_xr_dtype(mantissa_dtype)

        data = np.empty(sh, dtype=extended_dtype)
        data['mantissa'] = mantissa
        data['exp'] = exp
        return data

    @property
    def is_complex(self):
        """ boolean scalar, True if Xrange_array is complex"""
        _dtype = self.dtype
        if len(_dtype) > 1:
            _dtype = _dtype[0]
        return _dtype in Xrange_array._COMPLEX_DTYPES

    @property
    def real(self):
        """
        Returns a view to the real part of self, as an Xrange_array.
        """
        if self.is_complex:
            if self.dtype.names is None:
                return np.asarray(self).real.view(Xrange_array)
            real_bytes = 4
            if self._mantissa.real.dtype == np.float64:
                real_bytes = 8
            data_dtype = np.dtype({'names': ['mantissa', 'exp'],
                                   'formats': ["f" + str(real_bytes), "i4"],
                                   'offsets': [0, real_bytes*2],
                                   'itemsize': real_bytes * 2 + 4})
            re = np.asarray(self).view(dtype=data_dtype).view(Xrange_array)
            # As exponent is shared between real and imaginary part, this 
            # view is read-only
            re.flags.writeable = False
            return re
        else:
            return self

    @real.setter
    def real(self, val):
        """
        Setting the real part
        note: impacts the imaginary through the exponent
        """
        val = val.view(Xrange_array)
        Xrange_array._coexp_ufunc(val._mantissa, val._exp,
                                         self._mantissa.imag, self._exp)
        arr = np.asarray(self)
        (arr["mantissa"].real, arr["mantissa"].imag, arr["exp"]
            )= Xrange_array._coexp_ufunc(val._mantissa, val._exp,
                                         self._mantissa.imag, self._exp)

    @property
    def imag(self):
        """
        Returns a view to the imaginary part of self, as an Xrange_array.
        """
        if self.is_complex:
            if self.dtype.names is None:
                return np.asarray(self).imag.view(Xrange_array)
            assert 'exp' in np.asarray(self).dtype.names
            real_bytes = 4
            if self._mantissa.real.dtype == np.float64:
                real_bytes = 8
            data_dtype = np.dtype({'names': ['mantissa', 'exp'],
                                   'formats': ["f" + str(real_bytes), "i4"],
                                   'offsets': [real_bytes, real_bytes*2],
                                   'itemsize': real_bytes * 2 + 4})
            im = np.asarray(self).view(dtype=data_dtype).view(Xrange_array)
            # As exponent is shared between real and imaginary part, this 
            # view is read-only
            im.flags.writeable = False
            return im
        else:
            return 0. * self

    @imag.setter
    def imag(self, val):
        """
        Setting the imaginary part
        note: impacts the real through the exponent
        """
        arr = np.asarray(self)
        (arr["mantissa"].real, arr["mantissa"].imag, arr["exp"]
            )= Xrange_array._coexp_ufunc(self._mantissa.real, self._exp,
                                         val._mantissa, val._exp)


    @staticmethod
    def empty(shape, dtype, asarray=False):
        """ Return a new Xrange_array of given shape and type, without
        initializing entries.

        if asarray is True, return a view as an array, otherwise (default)
        return a Xrange_array
        """
        extended_dtype = np.dtype([('mantissa', dtype),
                                   ('exp', np.int32)], align=False)
        if asarray:
            return np.empty(shape, dtype=extended_dtype)
        else:
            return np.empty(shape, dtype=extended_dtype).view(Xrange_array)

    @staticmethod
    def zeros(shape, dtype):
        """ Return a new Xrange_array of given shape and type, with all entries
        initialized with 0."""
        ret = Xrange_array.empty(shape, dtype, asarray=True)
        ret["mantissa"] = 0.
        ret["exp"] = 0
        return ret.view(Xrange_array)

    @staticmethod
    def ones(shape, dtype):
        """ Return a new Xrange_array of given shape and type, with all entries
        initialized with 1."""
        ret = Xrange_array.empty(shape, dtype, asarray=True)
        ret["mantissa"] = 1.
        ret["exp"] = 0
        return ret.view(Xrange_array)

    def fill(self, val):
        """ Fill the array with val.
        Parameter
        ---------
        val : numpy scalar of a Xrange_array of null shape
        """
        fill_dict = {"exp": 0}
        if np.isscalar(val):
            fill_dict["mantissa"] = val          
        elif isinstance(val, Xrange_array) and (val.shape == ()):
            fill_dict["mantissa"] = val._mantissa
            fill_dict["exp"] = val._exp
        else:
            raise ValueError("Invalid input to Xrange_array.fill, "
                    "expected a numpy scalar or a Xrange_array of dim 0.")
        for key in ["mantissa", "exp"]:
            (np.asarray(self)[key]).fill(fill_dict[key])

    def to_standard(self):
        """ Returns the Xrange_array downcasted to standard np.ndarray ;
        obviously, may overflow. """
        return self._mantissa * (2. ** self._exp)

    @staticmethod
    def _build_complex(re, im):
        """ Build a complex Xrange_array from 2 similar shaped and typed
        Xrange_array (imag and real parts)"""
        m_re, m_im, exp = Xrange_array._coexp_ufunc(
                re._mantissa, re._exp, im._mantissa, im._exp)
        dtype = np.complex64
        if (m_re.dtype == np.float64) or (m_im.dtype == np.float64):
            dtype = np.complex128
        c = np.empty(m_re.shape, dtype=dtype)
        c.real = m_re
        c.imag = m_im
        return Xrange_array(c, exp)

    @property
    def _mantissa(self):
        """ Returns the mantissa of Xrange_array"""
        try:
            return np.asarray(self["mantissa"])
        except IndexError: # Assume we are view casting a np.ndarray
            m = np.asarray(self)
            if m.dtype in Xrange_array._DTYPES:
                return m
            else:
                for cast_dtype in Xrange_array._DTYPES:
                    if np.can_cast(m.dtype, cast_dtype, "safe"):
                        return m.astype(cast_dtype)

    @property
    def _exp(self):
        """ Returns the exponent of Xrange_array"""
        try:
            return np.asarray(self["exp"])
        except IndexError: # We are view casting a np.ndarray
            return np.int32(0)


    def normalize(self):
        """
        Normalize in-place a Xrange_array
        """
        arr = np.asarray(self)
        arr["mantissa"], arr["exp"] = self._normalize(
                arr["mantissa"], arr["exp"])

    @staticmethod
    def _normalize(m, exp):
        """
        Parameters
        m :  np.array of supported type
        exp : int32 np.array

        Return
        nm : float32 or float64 np.array
        nexp : int32 np.array
            f * 2**exp == nf * 2**nexp
            .5 <= abs(nf) < 1.
        """
        if m.dtype in Xrange_array._FLOAT_DTYPES:
            nm, exp2 = np.frexp(m)
            nexp = np.where(m == 0., np.int32(0), exp + exp2)
            return nm, nexp
        elif m.dtype in Xrange_array._COMPLEX_DTYPES:
            nm = np.empty_like(m)
            nm_re, nexp_re = Xrange_array._normalize(m.real, exp)
            nm_im, nexp_im = Xrange_array._normalize(m.imag, exp)
            nm.real, nm.imag, nexp = Xrange_array._coexp_ufunc(
                nm_re, nexp_re, nm_im, nexp_im)
            return nm, nexp
        else:
            raise ValueError(m.dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        - *ufunc* is the ufunc object that was called.
        - *method* is a string indicating how the Ufunc was called, either
          ``"__call__"`` to indicate it was called directly, or one of its
          :ref:`methods<ufuncs.methods>`: ``"reduce"``, ``"accumulate"``,
          ``"reduceat"``, ``"outer"``, or ``"at"``.
        - *inputs* is a tuple of the input arguments to the ``ufunc``
        - *kwargs* contains any optional or keyword arguments passed to the
          function. This includes any ``out`` arguments, which are always
          contained in a tuple.
          
        see also:
    https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/mixins.py#L59-L176
        """
        out = kwargs.pop("out", None)
        if out is not None:
            if ufunc.nout == 1: # Only supported case to date
                out = np.asarray(out[0])

        casted_inputs = ()
        for x in inputs:
            # Only support operations with instances of _HANDLED_TYPES.
            if isinstance(x, Xrange_array):
                casted_inputs += (x,)
            elif isinstance(x, np.ndarray):
                casted_inputs += (x.view(Xrange_array),)
            elif isinstance(x, numbers.Number):
                casted_inputs += (Xrange_array(x),)
            elif isinstance(x, list):
                casted_inputs += (Xrange_array(x),)
            else:
                # Operation not supported (type not handled), return the
                # sentinel value NotImplemented
                return NotImplemented

        if method == "__call__":
            if ufunc in [np.add, np.subtract]:
                out = self._add(ufunc, casted_inputs, out=out)
            elif ufunc is np.negative:
                out = self._negative(casted_inputs, out=out)
            elif ufunc in [np.multiply, np.true_divide]:
                out = self._mul(ufunc, casted_inputs)
            elif ufunc in [np.greater, np.greater_equal, np.less,
                           np.less_equal, np.equal, np.not_equal]:
                # Not a Xrange array, returns bool array
                return self._compare(ufunc, casted_inputs, out=out)
            elif ufunc is np.maximum:
                out = self._maximum(casted_inputs, out=out)
            elif ufunc is np.absolute:
                out = self._abs(casted_inputs, out=out)
            elif ufunc is np.sqrt:
                out = self._sqrt(casted_inputs, out=out)
            elif ufunc is np.square:
                out = self._square(casted_inputs, out=out)
            elif ufunc is np.conj:
                out = self._conj(casted_inputs, out=out)
            elif ufunc is np.log:
                out = self._log(casted_inputs, out=out)
            elif ufunc is np.arctan2:
                # Not a Xrange array, returns a float array
                return self._arctan2(casted_inputs, out=out)
            else:
                out = None
        elif method in ["reduce", "accumulate"]:
            if ufunc is np.add:
                out = self._add_method(casted_inputs, method, out=out,
                                       **kwargs)
            elif ufunc is np.multiply:
                out = self._mul_method(casted_inputs, method, out=out,
                                       **kwargs)
            else:
                out = None

        if out is None:
            raise NotImplementedError("ufunc {} method {} not implemented for "
                                      "Xrange_array".format(ufunc, method))
        return out.view(Xrange_array)


    @staticmethod
    def _arctan2(inputs, out=None):
        """
        Return the arctan2 as a standard float array
        """
        op0, op1 = inputs

        if op0.shape == () and op0 == 0.:
            # As of numpy 1.19.3 'np.angle' is not a ufunc but wraps arctan2 ;
            # this branch will handle calls by np.angle with zimag = 0. and
            # zreal a complex Xrange_array
            return np.angle(op1._mantissa)

        m0 = op0._mantissa
        m1 = op1._mantissa
        if out is None:
            out =  np.empty(np.broadcast(m0, m1).shape,
                                   dtype=np.result_type(m0, m1))
        out, _ = Xrange_array._coexp_ufunc(
                m0, op0._exp, m1, op1._exp, ufunc=np.arctan2)
        return out

    def abs2(self, out=None):
        """
        Return the square of np.abs(self) (for optimisation purpose).
        """
        if out is None:
            out = Xrange_array.empty(self.shape,
                    dtype=self._mantissa.real.dtype, asarray=True)
        if self.is_complex:
            out["mantissa"] = self._mantissa.real**2 + self._mantissa.imag**2
            out["exp"] = 2 * self._exp
        else:
            out["mantissa"] = self._mantissa**2
            out["exp"] = 2 * self._exp
        # ! not a unfunc so need to keep the view
        return out.view(Xrange_array)

    @staticmethod
    def _conj(inputs, out=None):
        """ x -> np.conj(x) """
        op0, = inputs
        m0 = op0._mantissa
        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)
        out["mantissa"] = np.conj(op0._mantissa)
        out["exp"] = op0._exp
        return out

    @staticmethod
    def _square(inputs, out=None):
        """ x -> x**2  """
        op0, = inputs
        m0 = op0._mantissa
        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)        
        m =  np.square(m0)
        if Xrange_array._need_renorm(m):
            out["mantissa"], out["exp"] = Xrange_array._normalize(
                    m, 2 * op0._exp)
        else:
            out["mantissa"] = m
            out["exp"] = 2 * op0._exp
        return out

    @staticmethod
    def _log(inputs, out=None):
        """ x -> np.log(x)  """
        ln2 = 0.6931471805599453
        op0, = inputs
        m0 = op0._mantissa
        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)

        if op0.is_complex:
            m_re, exp_re = Xrange_array._normalize(m0.real, op0._exp)
            m_im, exp_im = Xrange_array._normalize(m0.imag, op0._exp)
            m_re *= 2.
            exp_re -= 1
            m_re, m_im, e = Xrange_array._coexp_ufunc(
                    m_re, exp_re, m_im, exp_im)
            m = m_re + 1.j * m_im
        else:
            m, e = Xrange_array._normalize(m0, op0._exp)
            m *= 2.
            e -= 1
            m_re = m
        # Avoid loss of significant digits if e * ln2 close to log(m)
        # ie m close to 2.0
        e_is_m1 = (e == -1)
        if np.isscalar(m):
            if e_is_m1:
                m[e_is_m1] *= 0.5
                e[e_is_m1] += 1
        else:
            m[e_is_m1] *= 0.5
            e[e_is_m1] += 1

        out["mantissa"] = np.log(m) + m_re.dtype.type(e * ln2)
        out["exp"] = 0
        return out

    @staticmethod
    def _sqrt(inputs, out=None):
        """ x -> np.sqrt(x)  """
        sqrt0, = inputs
        m0 = sqrt0._mantissa
        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)
        
        if sqrt0.is_complex:
            m_re, m_im, exp = Xrange_array._coexp_ufunc(
                    m0.real, sqrt0._exp,
                    m0.imag, sqrt0._exp, None)
            m = m_re + 1.j * m_im
            even_exp = ((exp % 2) == 0).astype(bool)
            exp = np.where(even_exp, exp // 2, (exp - 1) // 2)
            out["mantissa"] = np.sqrt(np.where(even_exp, m, m * 2.))
            out["exp"] = exp
        else:
            even_exp = ((sqrt0._exp % 2) == 0).astype(bool)
            out["mantissa"] = np.sqrt(np.where(even_exp, sqrt0._mantissa,
                                 sqrt0._mantissa * 2.))
            out["exp"] = np.where(even_exp, sqrt0._exp // 2,
                    (sqrt0._exp - 1) // 2)

        return out

    @staticmethod
    def _abs(inputs, out=None):
        """ x -> np.abs(x) """
        op0, = inputs
        m0 = op0._mantissa
        exp0 = op0._exp

        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.real.dtype,
                                     asarray=True)
        if op0.is_complex:
            Xrange_array._sqrt((op0.real * op0.real + op0.imag * op0.imag,),
                               out=out)
        else:
            out["mantissa"] = np.abs(m0)
            out["exp"] = exp0

        return out

    @staticmethod
    def _compare(ufunc, inputs, out=None):
        """ compare x and y """
        op0, op1 = inputs
        m0 = op0._mantissa
        m1 = op1._mantissa
        if out is None:
            out = np.empty(np.broadcast(m0, m1).shape, dtype=bool)

        if (op0.is_complex or op1.is_complex):
            if ufunc in [np.equal, np.not_equal]:
                re_eq = Xrange_array._coexp_ufunc(
                        m0.real, op0._exp, m1.real, op1._exp, ufunc)[0]
                im_eq = Xrange_array._coexp_ufunc(
                        m0.imag, op0._exp, m1.imag, op1._exp, ufunc)[0]
                if ufunc is np.equal:
                    out = re_eq & im_eq
                else:
                    out = re_eq | im_eq
            else:
                raise NotImplementedError(
                    "{} Not supported for complex".format(ufunc))
        else:
            out = Xrange_array._coexp_ufunc(m0, op0._exp, m1,
                                            op1._exp, ufunc)[0]
        return out

    @staticmethod
    def _maximum(inputs, out=None):
        op0, op1 = inputs
        m0 = op0._mantissa
        exp0 = op0._exp
        m1 = op1._mantissa
        exp1 = op1._exp
        if out is None:
            out = Xrange_array.empty(np.broadcast(m0, m1).shape,
                dtype=np.result_type(m0, m1), asarray=True)
        where1 = (op1 > op0)
        out["mantissa"] = np.where(where1, m1, m0)
        out["exp"] = np.where(where1, exp1, exp0)
        return out

    @staticmethod
    def _mul(ufunc, inputs, out=None):
        """ internal auxilliary function for * and / operators """
        op0, op1 = inputs
        m0 = op0._mantissa
        exp0 = op0._exp
        m1 = op1._mantissa
        exp1 = op1._exp
        if ufunc is np.true_divide:
            m1 = np.reciprocal(m1)
            exp1 = -exp1

        if out is None:
            out = Xrange_array.empty(np.broadcast(m0, m1).shape,
                dtype=np.result_type(m0, m1), asarray=True)

        m = m0 * m1
        if Xrange_array._need_renorm(m):
            out["mantissa"], out["exp"] = Xrange_array._normalize(m, exp0 + exp1)
        else:
            out["mantissa"] = m
            out["exp"] = exp0 + exp1
        return out

    @staticmethod
    def _negative(inputs, out=None):
        """ x -> -x """
        op0, = inputs
        m0 = op0._mantissa
        if out is None:
            out = Xrange_array.empty(op0.shape, dtype=m0.dtype, asarray=True)
        out["mantissa"] = -m0
        out["exp"] = op0._exp
        return out

    @staticmethod
    def _add(ufunc, inputs, out=None):
        """ internal auxilliary function for + and - operators """
        op0, op1 = inputs
        m0 = op0._mantissa
        m1 = op1._mantissa
        if out is None:
            out = Xrange_array.empty(np.broadcast(m0, m1).shape,
                dtype=np.result_type(m0, m1), asarray=True)

        if (op0.is_complex or op1.is_complex):
            # Packing together
            out["mantissa"], out["exp"] = Xrange_array._cplx_coexp_ufunc(
                    m0, op0._exp, m1, op1._exp, ufunc)
        else:
            out["mantissa"], out["exp"] = Xrange_array._coexp_ufunc(
                    m0, op0._exp, m1, op1._exp, ufunc)

        return out 

    @staticmethod
    def _coexp_ufunc(m0, exp0, m1, exp1, ufunc=None):
        """ 
        If ufunc is None :
        m0, exp0, m1, exp1, -> co_m0, co_m1, co_exp so that :
        (*)  m0 * 2**exp0 == co_m0 * 2**co_exp
        (*)  m1 * 2**exp1 == co_m1 * 2**co_exp
        (*)  co_exp is the "leading exponent" exp = np.maximum(exp0, exp1)
            except if one of m0, m1 is null.
        If ufunc is provided :
            m0, exp0, m1, exp1, -> ufunc(co_m0, co_m1), co_exp
   
        """
        co_m0, co_m1 = np.copy(np.broadcast_arrays(m0, m1))
        
        exp0 = np.broadcast_to(exp0, co_m0.shape)
        exp1 = np.broadcast_to(exp1, co_m0.shape)

        m0_null = (m0 == 0.)
        m1_null = (m1 == 0.)
        d_exp = exp0 - exp1

        if (co_m0.shape == ()):
            if ((exp1 > exp0) & ~m1_null):
                co_m0 = Xrange_array._exp2_shift(co_m0, d_exp)
            if ((exp0 > exp1) & ~m0_null):
                co_m1 = Xrange_array._exp2_shift(co_m1, -d_exp)
            exp = np.maximum(exp0, exp1)
            if m0_null:
                exp = exp1
            if m1_null:
                exp = exp0
        else:
            bool0 = ((exp1 > exp0) & ~m1_null)
            co_m0[bool0] = Xrange_array._exp2_shift(
                    co_m0[bool0], d_exp[bool0])
            bool1 = ((exp0 > exp1) & ~m0_null)
            co_m1[bool1] = Xrange_array._exp2_shift(
                    co_m1[bool1], -d_exp[bool1])
            exp = np.maximum(exp0, exp1)
            exp[m0_null] = exp1[m0_null]
            exp[m1_null] = exp0[m1_null]

        if ufunc is not None: 
            return (ufunc(co_m0, co_m1), exp)
        else:
            return (co_m0, co_m1, exp)
        
        
    @staticmethod
    def _cplx_coexp_ufunc(m0, exp0, m1, exp1, ufunc=None):
        """ 
        Idem with complex m0, m1
        """
        co_m0, co_m1 = np.copy(np.broadcast_arrays(m0, m1))
        exp0 = np.broadcast_to(exp0, co_m0.shape)
        exp1 = np.broadcast_to(exp1, co_m0.shape)

        m0_null = (m0 == 0.)
        m1_null = (m1 == 0.)
        d_exp = exp0 - exp1

        if (co_m0.shape == ()):
            if ((exp1 > exp0) & ~m1_null):
                co_m0 = (Xrange_array._exp2_shift(co_m0.real, d_exp)
                         + 1.j * Xrange_array._exp2_shift(co_m0.imag, d_exp))
            if ((exp0 > exp1) & ~m0_null):
                co_m1 = (Xrange_array._exp2_shift(co_m1.real, -d_exp)
                         + 1.j * Xrange_array._exp2_shift(co_m1.imag, -d_exp))
            exp = np.maximum(exp0, exp1)
            if m0_null:
                exp = exp1
            if m1_null:
                exp = exp0
        else:
            f_dtype = np.float32
            if (m0.dtype == np.complex128) or (m1.dtype == np.complex128):
                f_dtype = np.float64
            k0 = Xrange_array._exp2(-d_exp, dtype=f_dtype)
            k1 = Xrange_array._exp2(d_exp, dtype=f_dtype)
            
            bool0 = ((exp1 > exp0) & ~m1_null)
            bool1 = ((exp0 > exp1) & ~m0_null)

            co_m0[bool0] *= k0[bool0]
            co_m1[bool1] *= k1[bool1]

            exp = np.maximum(exp0, exp1)
            exp[m0_null] = exp1[m0_null]
            exp[m1_null] = exp0[m1_null]

        if ufunc is not None: 
            return (ufunc(co_m0, co_m1), exp)
        else:
            return (co_m0, co_m1, exp)

    @staticmethod
    def _add_method(inputs, method, out=None, **kwargs):
        """
        """
        if method == "accumulate":
            raise NotImplementedError("ufunc {} method {} not implemented for "
                                      "Xrange_array".format(np.add, method))
        if out is not None:
            raise NotImplementedError("`out` keyword not immplemented "
                "for ufunc {} method {} of Xrange_array".format(
                        np.add, "reduce"))

        op, = inputs

        axis = kwargs.get("axis", 0)
        broadcast_co_exp_acc = np.maximum.reduce(op._exp, axis=axis, 
                                                 keepdims=True)

        if op.is_complex:
            re = Xrange_array._exp2_shift(op._mantissa.real, 
                                        op._exp - broadcast_co_exp_acc)
            im = Xrange_array._exp2_shift(op._mantissa.imag, 
                                        op._exp - broadcast_co_exp_acc)
            co_m = re + 1.j * im
        else:
            co_m = Xrange_array._exp2_shift(op._mantissa, 
                                        op._exp - broadcast_co_exp_acc)

        res = Xrange_array(*Xrange_array._normalize(
                    np.add.reduce(co_m, axis=axis),
                    np.squeeze(broadcast_co_exp_acc, axis=axis)))
        return res

    @staticmethod
    def _mul_method(inputs, method, out=None, **kwargs):
        """
        methods implemented are reduce or accumulate
        """
        if out is not None:
            raise NotImplementedError("`out` keyword not immplemented "
                "for ufunc {} method {} of Xrange_array".format(
                        np.multiply, method))

        op, = inputs
        m0, exp0 = Xrange_array._normalize(op._mantissa, op._exp)
        # np.multiply.reduce(m0, axis=axis) shall remains bounded
        # Set m m between sqrt(0.5) and sqrt(2)
        # With float64 mantissa, in current implementation, mantissa is only
        # guaranteed to not overflow for arrays of less than 2000 elements
        # (because 1.41**2000 = 2.742996861934711e+298 < max float64)
        is_below = m0 < np.sqrt(0.5)
        m0[is_below] *= 2.
        exp0[is_below] -= 1 

        axis = kwargs.get("axis", 0)
        res = Xrange_array(*Xrange_array._normalize(
                    getattr(np.multiply, method)(m0, axis=axis),
                    getattr(np.add, method)(exp0, axis=axis)))
        return res

    @staticmethod
    def _need_renorm(val):
        """
        Returns True if val need renom
        """
        val = np.asarray(val)
        if val.dtype == np.float32:
            bits = val.view(np.int32)
            return np.any((np.abs(((bits >> 23) & 0xff) - 127) > 31)
                          & (val != 0.))
        elif val.dtype == np.float64:
            bits = val.view(np.int64)
            return np.any((np.abs(((bits >> 52) & 0x7ff) - 1023) > 255)
                          & (val != 0.))
        elif val.dtype in [np.complex64, np.complex128]:
            return np.logical_or(Xrange_array._need_renorm(val.real),
                                 Xrange_array._need_renorm(val.imag))
        else:
            raise ValueError("Unsupported dtype {}".format(val.dtype))

    @staticmethod
    def _xlog2(val):
        """
        Returns a rough evaluation of the exponent base 2
        """
        val = np.asarray(val)
        if val.dtype == np.float32:
            bits = val.view(np.int32)
            return np.where(val == 0., 0, np.abs(((bits >> 23) & 0xff) - 127)
                                                 ).astype(np.int16)
        elif val.dtype == np.float64:
            bits = val.view(np.int64)
            return np.where(val == 0., 0, np.abs(((bits >> 52) & 0x7ff) - 1023)
                                                 ).astype(np.int16)
        elif val.dtype in [np.complex64, np.complex128]:
            return np.maximum(Xrange_array._xlog2(val.real),
                              Xrange_array._xlog2(val.imag))
        else:
            raise ValueError("Unsupported dtype {}".format(val.dtype))

    @staticmethod
    def _exp2(exp, dtype):
        """
        Returns 2**-exp, exp np.int32 > 0
        """
        if dtype == np.float32:
            _exp = np.clip(127 - exp, 0, None)
            return (_exp << 23).view(np.float32)
        elif dtype == np.float64:
            _exp = np.clip(1023 - exp.astype(np.int64), 0, None)
            return (_exp << 52).view(np.float64)
        else:
            raise ValueError("Unsupported dtype {}".format(dtype))

    @staticmethod
    def _exp2_shift(m, shift):
        """
        Parameters
            m : float32 or float64 array, mantissa
            exp : int32 array, negative integers array

        Return
            res array of same type as m, shifted by 2**shift :
                res = m * 2**shift

        References:
        https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        s(1)e(8)m(23)
        (bits >> 23) & 0xff  : exponent with bias 127 (0x7f)
        (bits & 0x7fffff) : mantissa, implicit first bit of value 1

        https://en.wikipedia.org/wiki/Double-precision_floating-point_format
        s(1)e(11)m(52)
        (bits >> 52) & 0x7ff : exponent with bias 1023 (0x3ff)
        (bits & 0xfffffffffffff) : mantissa, implicit first bit of value 1
        """
        dtype = m.dtype
        if dtype == np.float32:
            bits = m.view(np.int32)
            # Need to take special care as casting to int32 a 0d array is only
            # supported if the itemsize is unchanged. So we impose the res 
            # dtype
            res_32 = np.empty_like(bits)
            exp = np.clip(((bits >> 23) & 0xff) + shift, 0, None)
            np.add((exp << 23), bits & 0x7fffff, out=res_32)
            return np.copysign(res_32.view(np.float32), m)

        elif dtype == np.float64:
            bits = m.view(np.int64)
            exp = np.clip(((bits >> 52) & 0x7ff) + shift, 0, None)
            return np.copysign(((exp << 52) + (bits & 0xfffffffffffff)
                                ).view(np.float64) , m)
        else:
            raise ValueError("Unsupported dtype {}".format(dtype))


    def __repr__(self):
        """ Detailed string representation of self """
        s = (str(type(self)) + "\nshape: " +str(self.shape) +
             "\ninternal dtype: " + str(self.dtype) + 
             "\nbase 10 representation:\n" +
             self.__str__())
        return s

    def __str__(self):
        """
        String representation of self. Takes into account the value of
        np.get_printoptions(precision)

        Usage :
        with np.printoptions(precision=2) as opts:
            print(extended_range_array)
        """
        # There is no easy way to impose the formatting of a structured array
        # Monkey patching np.core.arrayprint.StructuredVoidFormat
        orig = np.core.arrayprint.StructuredVoidFormat
        try:
            np.core.arrayprint.StructuredVoidFormat = _Xrange_array_format
            if self.shape == ():
                ret = np.array2string(self.reshape([1]))[1:-1]
            else:
                ret = np.array2string(self)
        finally:
            np.core.arrayprint.StructuredVoidFormat = orig
        return ret

    def _to_str_array(self, **options):
        """
        String representation of self. Takes into account the value of
        np.get_printoptions(precision)

        Usage :
        with np.printoptions(precision=2) as opts:
            print(extended_range_array)
        """
        if self.is_complex:
            s_re = Xrange_array._to_char(self.real, **options)
            s_im = Xrange_array._to_char(self.imag, im=True, **options)
            s = np.core.defchararray.add(s_re, s_im)
            s = np.core.defchararray.add(s, "j")
        else:
            s = Xrange_array._to_char(self, **options)
        return s

    @staticmethod
    def _to_char(arr, im=False, im_p_char = '+',
                       im_m_char = '-', **options):
        """
        Parameters:
            m2 base 2 real mantissa
            exp2 : base 2 exponent

        Return
            str_arr  string array of representations in base 10.

        Note: precisions according to:
            np.get_printoptions(precision)
        """
        def_opts = np.get_printoptions()
        precision = options.pop("precision", def_opts["precision"])
        nanstr = options.pop("nanstr", def_opts["nanstr"])
        infstr = options.pop("infstr", def_opts["infstr"])
        
        m2, exp2 = Xrange_array._normalize(arr._mantissa, arr._exp)
        m10, exp10 = Xrange_array._rebase_2to10(m2, exp2)

        if np.isscalar(m10): # scalar do not support item assignment
            if (np.abs(m10) < 1.0):
                m10 *= 10.
                exp10 -= 1
            exp10 = np.asarray(exp10, np.int32)
            _m10 = np.around(m10, decimals=precision)
            if (np.abs(_m10) >= 10.0):
                m10 *= 0.1
                exp10 += 1
            m10 = np.around(m10, decimals=precision)
            # Special case of 0.
            if (m2 == 0.):
                exp10 = 0
            # Special case of 0.
            if np.isnan(m2 == 0.):
                exp10 = 0
        else:
            m10_up = (np.abs(m10) < 1.0)
            m10[m10_up] *= 10.
            exp10[m10_up] -= 1
            exp10 = np.asarray(exp10, np.int32)
            _m10 = np.around(m10, decimals=precision)
            m10_down= (np.abs(_m10) >= 10.0)
            m10[m10_down] *= 0.1
            exp10[m10_down] += 1
            m10 = np.around(m10, decimals=precision)
            # Special case of 0.
            is_null = (m2 == 0.)
            exp10[is_null] = 0

        if im :
            p_char = im_p_char # '\u2795' bold +
            m_char = im_m_char # '\u2796' bold -
        else:
            p_char = " "
            m_char = "-"
        concat = np.core.defchararray.add
        exp_digits = int(np.log10(max([np.nanmax(np.abs(exp10)), 10.]))) + 1
        str_arr = np.where(m10 < 0., m_char, p_char)
        str_arr = concat(str_arr,
                         np.char.ljust(np.abs(m10).astype("|U" + 
                                       str(precision + 2)),
                                       precision + 2, "0"))
        str_arr = concat(str_arr, "e")
        str_arr = concat(str_arr, np.where(exp10 < 0, "-", "+"))
        str_arr = concat(str_arr,
            np.char.rjust(np.abs(exp10).astype("|U10"), exp_digits, "0"))

        # Handles nan and inf values
        np.putmask(str_arr, np.isnan(m2), nanstr)
        np.putmask(str_arr, np.isinf(m2), infstr)

        return str_arr

    @staticmethod
    def _rebase_2to10(m2, exp2):
        """
        Parameters:
        m2 mantissa in base 2
        exp2 int32 exponent in base 2

        Returns:
        m10 mantissa in base 10
        exp10 int32 exponent in base 10

        Note : 
        This is a high-precision version of:
            > r = math.log10(2)
            > exp10, mod = np.divmod(exp2 * r, 1.)
            > return m2 * 10.**mod, exp10

        In order to guarantee an accuracy > 15 digits (in reality, close to 16)
        for `mod` with the 9-digits highest int32 base 2 exponent (2**31 - 1)
        we use an overall precision of 96 bits for this divmod.
        """
        # We will divide by hand in base 2**32 (chosen so that exp2 * ri does
        # not overflow an int64 with the largest exp2 == 2**31-1), ri < 2**32.
        # >>> import mpmath
        # >>> mpmath.mp.dps = 35
        # >>> mpmath.log("2.") / mpmath.log("10.") * mpmath.mpf(2**96)
        # mpf('23850053418134191015272426710.02243475524574')
        r_96 = 23850053418134191015272426710
        mm = [None] * 3
        for i in range(3):
            ri = (r_96 >> (32 * (2 - i))) & 0xffffffff
            mm[i] = exp2.astype(np.int64) * ri
            if i == 0: # extract the integer `mod` part
                di, mm[i] = np.divmod(mm[i], 0x100000000)
                d = di.astype(np.int64)
        m = (mm[0] + (mm[1] + mm[2] * 2.**-32) * 2.**-32) * 2**-32
        return  m2 * 10.**m, d.astype(np.int32)


    def __setitem__(self, key, val):
        """ Can be given either a Xrange_array or a complex of float array-like
        (See 'supported types')
        """
        if type(val) is Xrange_array:
            if val.is_complex and not(self.is_complex):
                raise ValueError("Cant cast complex values to real")
            np.ndarray.__setitem__(self, key, val)
        else:
            val = np.asarray(val).view(Xrange_array)
            np.ndarray.__setitem__(self._mantissa, key, val._mantissa)
            np.ndarray.__setitem__(self._exp, key, val._exp)

    def __getitem__(self, key):
        """ For single item, return array of empty shape rather than a scalar,
        to allow pretty print and maintain assignment behaviour consistent.
        """
        res = np.ndarray.__getitem__(self, key)
        if np.isscalar(res):
            res = np.asarray(res).view(Xrange_array)
        return res

    def __eq__(self, other):
        """ Ensure that `!=` is handled by Xrange_array instance. """
        return np.equal(self, other)

    def __ne__(self, other):
        """ Ensure that `==` is handled by Xrange_array instance. """
        return np.not_equal(self, other)


class _Xrange_array_format():
    """ Formatter class for Xrange_array printing. """
    def __init__(self, **options):
        self.options = options
    @classmethod
    def from_data(cls, data, **options):
        return cls(**options)
    def __call__(self, x):
        return str(x._to_str_array(**self.options))


class Xrange_polynomial(np.lib.mixins.NDArrayOperatorsMixin):
    """
    One-dimensionnal polynomial class featuring extended-range coefficients
    which provides:
        - the standard Python numerical methods ‘+’, ‘-‘, ‘*' 
        - derivative
        - evaluation
        - pretty-print

    Parameters
    ----------
    coeffs: array_like - can be viewed as a Xrange_array
    Polynomial coefficients in order of increasing degree, i.e.,
    (1, 2, 3) give 1 + 2*x + 3*x**2.

    cutdeg : int, maximum degree coefficient. At instanciation but also for
    the subsequent operations, monomes of degree above cutdeg will be 
    disregarded.
    """  
    # Unicode character mappings for "pretty print" of the polynomial
    if sys.platform == "linux":
        _superscript_mapping = str.maketrans({
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹"
        })
    else:
        _superscript_mapping = str.maketrans({
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9"
        })

    def __init__(self, coeffs, cutdeg):
        if isinstance(coeffs, Xrange_array):
            self.coeffs = coeffs[0:cutdeg+1]
        else:
            self.coeffs = Xrange_array(np.asarray(coeffs)[0:cutdeg+1])#.view(Xrange_array)
        if self.coeffs.ndim != 1:
            raise ValueError("Only 1-d inputs for Xrange_polynomial")
        self.cutdeg = cutdeg


    def  __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        casted_inputs = ()
        casted_cutdegs = ()

        for x in inputs:
            # Only support operations with instances of 
            # Xrange_array._HANDLED_TYPES.
            if isinstance(x, Xrange_polynomial):
                casted_inputs += (x.coeffs,)
                casted_cutdegs += (x.cutdeg,)
            elif isinstance(x, Xrange_array):
                casted_inputs += (x.flatten(),)
            elif isinstance(x, np.ndarray):
                casted_inputs += (x.flatten().view(Xrange_array),)
            elif isinstance(x, numbers.Number):
                casted_inputs += (Xrange_array([x]),)
            elif isinstance(x, list):
                casted_inputs += (Xrange_array(x),)
            else:
                # Operation not supported (type not handled), return the
                # sentinel value NotImplemented
                return NotImplemented

        cutdeg = min(casted_cutdegs)
        if not all(item == cutdeg for item in casted_cutdegs):
            raise ValueError("Operation not supported, incompatible cutdegs {}"
                             .format(casted_cutdegs))

        out = kwargs.pop("out", None)

        if method == "__call__":
            if ufunc in [np.add, np.subtract]:
                return self._add(ufunc, casted_inputs, cutdeg=cutdeg, out=out)
            elif ufunc is np.negative:
                return self._negative(casted_inputs, cutdeg=cutdeg, out=out)
            elif ufunc is np.multiply:
                return self._mul(casted_inputs, cutdeg=cutdeg, out=out)
        # Other ufunc  not supported
        return NotImplemented

    @staticmethod
    def _add(ufunc, inputs, cutdeg, out=None):
        """ Add or Subtract 2 Xrange_polynomial """
        op0, op1 = inputs
        res_len = min(max(op0.size, op1.size), cutdeg + 1)
        op0_len = min(op0.size, res_len)
        op1_len = min(op1.size, res_len)

        dtype=np.result_type(op0._mantissa, op1._mantissa)
        res = Xrange_array(np.zeros([res_len], dtype=dtype))

        res[:op0_len] += op0[:op0_len]
        if ufunc is np.add:
            res[:op1_len] += op1[:op1_len]
        elif ufunc is np.subtract: 
            res[:op1_len] -= op1[:op1_len]
        return Xrange_polynomial(res, cutdeg=cutdeg)

    @staticmethod
    def _negative(inputs, cutdeg, out=None):
        """ Change sign of a Xrange_polynomial """
        op0, = inputs
        return Xrange_polynomial(-op0, cutdeg=cutdeg)

    @staticmethod
    def _mul(inputs, cutdeg, out=None):
        """ Product of 2 Xrange_polynomial """
        op0, op1 = inputs
        # This is a convolution, fix the window with the shortest poly op0,
        # swapping poly if needed. (We do not use fft but direct
        # calculation as number of terms stay low)
        if op0.size > op1.size:
            op0, op1 = op1, op0
        l0 = op0.size
        l1 = op1.size
        cutoff_res = min(l0 + l1 - 2, cutdeg) # the degree...
        op1 = np.pad(op1, (l0 - 1, cutoff_res - l1 + 1),
                     mode='constant').view(Xrange_array)
        shift = np.arange(0, cutoff_res + 1)
        take1 = shift[:, np.newaxis] + np.arange(l0 - 1 , -1, -1)
        return Xrange_polynomial(np.sum(op0 * np.take(op1, take1), axis=1),
                cutdeg=cutdeg) # /!\ and not cutoff_res

    def __call__(self, arg):
        """ Call self as a function.
        """
        if not isinstance(arg, Xrange_array):
            arg = Xrange_array(np.asarray(arg))

        res_dtype = np.result_type(arg._mantissa, self.coeffs._mantissa)
        res = Xrange_array.empty(arg.shape, dtype=res_dtype)
        res.fill(self.coeffs[-1])

        for i in range(2, self.coeffs.size + 1):
            res = self.coeffs[-i] + res * arg
        return res
    
    def deriv(self, k=1.):
        l = self.coeffs.size
        coeffs = self.coeffs[1:] * np.arange(1, l)
        if k != 1.:
            mul = 1.
            for i in range(l-1):
                coeffs[i] *= mul
                mul *= k
        return Xrange_polynomial(coeffs, cutdeg=self.cutdeg)

    def taylor_shift(self, x0):
        """
        Parameters
        ----------
        x0 : Xrange_array of shape (1,)

        Returns
        -------
        Q : Xrange_polynomial so that
            Q(X) = P(X + x0) 

        Implementation
        Q(X) = P(X + x0) transformation is accomplished by the three simpler
        transformation:
            g(X) = p(x0 * X)
            f(X) = g(X + 1)
            q(X) = f(1./x0 * X)

        References
        [1] Joachim von zur Gathen, Jürgen Gerhard Fast Algorithms for Taylor
        Shifts and Certain Difference Equations.
        [2] Mary Shaw, J.F. Traub On the number of multiplications for the
        evaluation of a polynomial and some of its derivatives.
        """
        if x0 == 0.:
            return Xrange_polynomial(self.coeffs, cutdeg=self.cutdeg)
        return self.scale_shift(x0)._taylor_shift_one().scale_shift(1. / x0)

    def _taylor_shift_one(self):
        """
        private auxilliary function, shift by 1.0 : return Q so that
        Q(X) = P(X + 1.0) where P is self
        """
        dtype = self.coeffs._mantissa.dtype
        pascalT = Xrange_array.zeros([self.coeffs.size], dtype)
        tmp = pascalT.copy()
        pascalT[0] = self.coeffs[-1]
        for i in range(2, self.coeffs.size + 1):
            # at each step P -> P + (ai + X P)
            tmp[1:] = pascalT[:-1]
            tmp[0] = self.coeffs[-i]
            pascalT += tmp
        return Xrange_polynomial(pascalT, cutdeg=self.cutdeg)

    def scale_shift(self, a):
        """
        Parameters
        ----------
        a : Xrange_array of shape (1,)
        
        Returns
        -------
        Q : Xrange_polynomial so that :
            Q(X) = P(a * X) where P is 'self'
        """
        dtype = self.coeffs._mantissa.dtype
        scaled = Xrange_array.ones([self.coeffs.size], dtype=dtype)
        scaled[1:] = a
        scaled = np.cumprod(scaled) * self.coeffs
        return Xrange_polynomial(scaled, cutdeg=self.cutdeg)

    def __repr__(self):
        return ("Xrange_polynomial(cutdeg="+ str(self.cutdeg) +",\n" +
                self.__str__() + ")")

    def __str__(self):
        return self._to_str()

    def _to_str(self):
        """
        Generate the full string representation of the polynomial, using
        `_monome_base_str` to generate each polynomial term.
        """
        if self.coeffs.is_complex:
            str_coeffs = self.coeffs._to_str_array()
        else:
            str_coeffs = np.abs(self.coeffs)._to_str_array()
        linewidth = np.get_printoptions().get('linewidth', 75)
        if linewidth < 1:
            linewidth = 1
        if self.coeffs.real[0] >= 0.:
            out = f"{str_coeffs[0][1:]}"
        else:
            out = f"-{str_coeffs[0][1:]}"
        for i, coef in enumerate(str_coeffs[1:]):
            out += " "
            power = str(i + 1)
            # 1st Polynomial coefficient
            if (self.coeffs.is_complex) or self.coeffs.real[i + 1] >= 0.:
                next_term = f"+ {coef}"
            else:
                next_term = f"- {coef}"
            # Polynomial term
            if sys.platform == "linux":
                next_term += self._monome_base_str(power, "X")
            else:
                next_term += self._monome_base_str(power, "X**")
            # Length of the current line with next term added
            line_len = len(out.split('\n')[-1]) + len(next_term)
            # If not the last term in the polynomial, it will be two           
            # characters longer due to the +/- with the next term
            if i < len(self.coeffs[1:]) - 1:
                line_len += 2
            # Handle linebreaking
            if line_len >= linewidth:
                next_term = next_term.replace(" ", "\n", 1)
            next_term = next_term.replace("  ", " ")
            out += next_term
        return out

    @classmethod
    def _monome_base_str(cls, i, var_str):
        if sys.platform == "linux":
            return f"·{var_str}{i.translate(cls._superscript_mapping)}"
        else:
            return f".{var_str}{i.translate(cls._superscript_mapping)}"


class Xrange_SA(Xrange_polynomial):
    """
    One-dimensionnal, extended-range serie approximation class based on
    Xrange_polynomial:
        - provides the same feature as Xrange_polynomial + control of a
            truncature error term
        - For the prupose of truncature error calculation, it is assumed that 
            the domain of convergence is enclosed in the unit circle.

    Parameters
    ----------
    coeffs: see Xrange_polynomial
    cutdeg: see Xrange_polynomial (Monomes of degree above cutoff will be 
            disregarded.)
    err : truncature error term, in X**(cutoff + 1). Default to 0.
    """  

    def __init__(self, coeffs, cutdeg, err=Xrange_array(0.)):
        self.err = err
        if not(isinstance(err, Xrange_array)):
            self.err = Xrange_array(err)
        super().__init__(coeffs, cutdeg)

    def  __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        casted_inputs = ()
        casted_cutdegs = ()
        casted_errs = ()

        for x in inputs:
            # Only support operations with instances of 
            # Xrange_array._HANDLED_TYPES.
            if isinstance(x, Xrange_SA):
                casted_cutdegs += (x.cutdeg,)
                casted_inputs += (x.coeffs,)
                casted_errs += (x.err,)
            else:
                casted_errs += (0.,)
                if isinstance(x, Xrange_polynomial):
                    casted_inputs += (x.coeffs,)
                    casted_cutdegs += (x.cutdeg,)
                elif isinstance(x, Xrange_array):
                    casted_inputs += (x.flatten(),)
                elif isinstance(x, np.ndarray):
                    casted_inputs += (x.flatten().view(Xrange_array),)
                elif isinstance(x, numbers.Number):
                    casted_inputs += (Xrange_array([x]),)
                elif isinstance(x, list):
                    casted_inputs += (Xrange_array(x),)
                else:
                    # Operation not supported (type not handled), return the
                    # sentinel value NotImplemented
                    return NotImplemented

        cutdeg = min(casted_cutdegs)
        if not all(item == cutdeg for item in casted_cutdegs):
            raise ValueError("Operation not supported, incompatible cutdegs {}"
                             .format(casted_cutdegs))

        out = kwargs.pop("out", None)

        if method == "__call__":
            if ufunc in [np.add, np.subtract]:
                return self._add(ufunc, casted_inputs, casted_errs,
                                 cutdeg=cutdeg, out=out)
            elif ufunc is np.negative:
                return self._negative(casted_inputs, casted_errs,
                                      cutdeg=cutdeg, out=out)
            elif ufunc is np.multiply:
                return self._mul(casted_inputs, casted_errs,
                                 cutdeg=cutdeg, out=out)
        # Other ufunc not supported
        return NotImplemented

    @staticmethod
    def _add(ufunc, inputs, errs, cutdeg, out=None):
        """ Add or Subtract 2 Xrange_SA """
        op0, op1 = inputs
        res_len = min(max(op0.size, op1.size), cutdeg + 1)
        op0_len = min(op0.size, res_len)
        op1_len = min(op1.size, res_len)

        dtype=np.result_type(op0._mantissa, op1._mantissa)
        res = Xrange_array(np.zeros([res_len], dtype=dtype))

        res[:op0_len] += op0[:op0_len]
        if ufunc is np.add:
            res[:op1_len] += op1[:op1_len]
        elif ufunc is np.subtract: 
            res[:op1_len] -= op1[:op1_len]

        return Xrange_SA(res, cutdeg=cutdeg, err=sum(errs))

    @staticmethod
    def _negative(inputs, errs, cutdeg, out=None):
        """ Change sign of a Xrange_SA """
        op0, = inputs
        err0, = errs
        return Xrange_SA(-op0, cutdeg=cutdeg, err=err0)

    @staticmethod
    def _mul(inputs, errs, cutdeg, out=None):
        """ Multiply 2 Xrange_SA """
        op0, op1 = inputs
        # Almost same as Xrange_polynomial but need to take care of the 
        # truncature error term
        if op0.size > op1.size:
            op0, op1 = op1, op0
        l0 = op0.size
        l1 = op1.size
        cutoff_res = min(l0 + l1 - 2, cutdeg) # the degree...
        op1 = np.pad(op1, (l0 - 1, l0 - 1),
                     mode='constant').view(Xrange_array)
        shift = np.arange(0, cutoff_res + 1)
        take1 = shift[:, np.newaxis] + np.arange(l0 - 1 , -1, -1)
        op_res = np.sum(op0 * np.take(op1, take1), axis=1)

        err0, err1 = errs
        # We will use L2 norm to control truncature error term.
        # Heuristic based on random walk / magnitude of the sum of iud random
        # variables
        # https://en.wikipedia.org/wiki/Law_of_the_iterated_logarithm
        # https://mathoverflow.net/questions/89478/magnitude-of-the-sum-of-complex-i-u-d-random-variables-in-the-unit-circle
        # O(sqrt(N ))
        # Exact term is :
        #    op_err0 = err0 * np.sum(np.abs(op1))
        #    op_err1 = err1 * np.sum(np.abs(op0))
        op_err0 = err0 * np.sqrt(np.sum(op1.abs2()))
        op_err1 = err1 * np.sqrt(np.sum(op0.abs2()))

        if cutdeg < (l0 + l1 - 2):
            # Truncature error term - L2 norm
            shift_errT = np.arange(cutoff_res + 1, l0 + l1 - 1)
            take1 = shift_errT[:, np.newaxis] + np.arange(l0 - 1 , -1, -1)
            op_errT = np.sum(op0 * np.take(op1, take1), axis=1)
            # We will use L2 norm to control truncature error term.
            # Exact term is :
            #    op_errT = np.sum(np.abs(op_errT))
            op_errT = np.sqrt(np.sum(op_errT.abs2()))
            err = op_err0 + op_err1 + op_errT + err0 * err1
        else:
            err = op_err0 + op_err1 + err0 * err1

        return Xrange_SA(op_res, cutdeg=cutdeg, err=err)


    def __repr__(self):
        return ("Xrange_SA(cutdeg="+ str(self.cutdeg) +",\n" +
                self.__str__() + ")")

    def __str__(self):
        return self._to_str()

    def _to_str(self):
        """
        Generate the full string representation of the SA, using
        `_monome_base_str` to generate each polynomial term.
        """
        out = super()._to_str()
        out += " // Res <= {}".format(self.err.__str__()
                ) + self._monome_base_str(str(self.cutdeg + 1), "X")
        return out


class Xrange_bivar_polynomial(Xrange_polynomial):
    """
    Two-dimensionnal polynomial class featuring extended-range coefficients
    which provides:
        - the standard Python numerical methods ‘+’, ‘-‘, ‘*' 
        - derivative
        - evaluation
        - pretty-print

    Parameters
    ----------
    coeffs: array_like - can be viewed as a Xrange_array
    Polynomial coefficients in order of increasing degree, i.e.,
    Xrange_array(np.array([
        [ 1.,  2.,  3.],
        [11., 12., 13.],
        [21., 22., 23.],
    ]))
    give 1.00000000e+00 + 1.10000000e+01·X¹ + 2.10000000e+01·X² +
         2.00000000e+00·Y¹ + 1.20000000e+01·X¹·Y¹ + 3.00000000e+00·Y²
    Note that below-diagonal terms are not taken into account as they exceed 
    the maximal degree (see below *cutdeg*)

    cutdeg : int, maximum degree coefficient. At instanciation but also for
    the subsequent operations, monomes of total degree above cutdeg will be 
    disregarded.
    """  

    def __init__(self, coeffs, cutdeg):
        # internal storage of size 'cutdeg * cutdeg'
        self.coeffs = self.get_coeffs(coeffs, cutdeg)
        self.cutdeg = cutdeg

    def get_coeffs(self, coeffs, cutdeg):
        """ internal storage of size 'cutdeg * cutdeg' """
        if isinstance(coeffs, Xrange_array):
            _coeffs = coeffs[0:(cutdeg + 1), 0:(cutdeg + 1)]
        else:
            _coeffs = Xrange_array(np.asarray(coeffs)[
                0:(cutdeg + 1), 0:(cutdeg + 1)
            ])#.view(Xrange_array)
        if _coeffs.ndim != 2:
            raise ValueError("Only 2-d inputs for Xrange_polynomial")

        nx, ny = _coeffs.shape
        dtype = _coeffs._mantissa.dtype

        # Internal storage should be of shape (cutdeg + 1), (cutdeg + 1)
        if nx < (cutdeg + 1):
            add_X = Xrange_array.zeros(((cutdeg + 1 - nx), ny), dtype)
            _coeffs = np.concatenate((_coeffs, add_X), axis=0).view(
                    Xrange_array)

        if ny < (cutdeg + 1):
            add_Y = Xrange_array.zeros(
                    ((cutdeg + 1), (cutdeg + 1 - nx)),
                    dtype
            )
            _coeffs = np.concatenate((_coeffs, add_Y), axis=1).view(
                    Xrange_array)

        # extra-diagonal terms 'over cutdeg' should be null
        for i in range(1, cutdeg + 1):
            for j in range(1 + cutdeg - i, cutdeg + 1):
                # i + j == deg >= cutdeg + 1
                _coeffs[i, j] = 0.
        return _coeffs
        

    def  __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        casted_inputs = ()
        casted_cutdegs = ()

        for x in inputs:
            # Only support operations with instances of 
            # Xrange_array._HANDLED_TYPES.
            if isinstance(x, Xrange_bivar_polynomial):
                casted_inputs += (x.coeffs,)
                casted_cutdegs += (x.cutdeg,)
            elif isinstance(x, Xrange_array):
                casted_inputs += (x.flatten(),)
            elif isinstance(x, np.ndarray):
                casted_inputs += (x.flatten().view(Xrange_array),)
            elif isinstance(x, numbers.Number):
                casted_inputs += (Xrange_array([x]),)
            elif isinstance(x, list):
                casted_inputs += (Xrange_array(x),)
            else:
                # Operation not supported (type not handled), return the
                # sentinel value NotImplemented
                return NotImplemented

        cutdeg = min(casted_cutdegs)
        if not all(item == cutdeg for item in casted_cutdegs):
            raise ValueError("Operation not supported, incompatible cutdegs {}"
                             .format(casted_cutdegs))

        out = kwargs.pop("out", None)

        if method == "__call__":
            if ufunc in [np.add, np.subtract]:
                return self._add(ufunc, casted_inputs, cutdeg=cutdeg, out=out)
            elif ufunc is np.negative:
                return self._negative(casted_inputs, cutdeg=cutdeg, out=out)
            elif ufunc is np.multiply:
                return self._mul(casted_inputs, cutdeg=cutdeg, out=out)
        # Other ufunc  not supported
        return NotImplemented

    @staticmethod
    def _add(ufunc, inputs, cutdeg, out=None):
        """ Add or Subtract 2 Xrange_bivar_polynomial """
        op0, op1 = inputs
        res_len = cutdeg + 1
        dtype=np.result_type(op0._mantissa, op1._mantissa)
        res = Xrange_array.zeros((res_len, res_len), dtype)

        if op0.size == 1:
            res[:] = op1
            res[0, 0] += op0[0]
            return Xrange_bivar_polynomial(res, cutdeg=cutdeg)
        if op1.size == 1:
            res[:] = op0 
            res[0, 0] += op1[0]
            return Xrange_bivar_polynomial(res, cutdeg=cutdeg)
 
        res[:] += op0
        if ufunc is np.add:
            res[:] += op1
        elif ufunc is np.subtract: 
            res[:] -= op1

        return Xrange_bivar_polynomial(res, cutdeg=cutdeg)

    @staticmethod
    def _negative(inputs, cutdeg, out=None):
        """ Change sign of a Xrange_bivar_polynomial """
        op0, = inputs
        return Xrange_bivar_polynomial(-op0, cutdeg=cutdeg)

    @staticmethod
    def _mul(inputs, cutdeg, out=None):
        """ Product of 2 Xrange_bivar_polynomial """
        op0, op1 = inputs

        if op0.size == 1:
            res = op0[0] * op1
            return Xrange_bivar_polynomial(res, cutdeg=cutdeg)
        if op1.size == 1:
            res = op0 * op1[0]
            return Xrange_bivar_polynomial(res, cutdeg=cutdeg)

        res_len = cutdeg + 1
        dtype=np.result_type(op0._mantissa, op1._mantissa)
        res = Xrange_array.zeros((res_len, res_len), dtype)

        # The hard way
        for i in range(0, cutdeg + 1):
            for j in range(0, cutdeg + 1 - i):
                for k in range(0, i + 1):
                    for l in range(0, j + 1):
                        res[i, j] += op0[k, l] * op1[i - k, j - l]

        return Xrange_bivar_polynomial(res, cutdeg=cutdeg)

    def __call__(self, argX, argY):
        """ Calls a Xrange_bivar_polynomial as a function.
        """
        if not isinstance(argX, Xrange_array):
            argX = Xrange_array(np.asarray(argX))
        if not isinstance(argY, Xrange_array):
            argY = Xrange_array(np.asarray(argY))
        if argX.shape != argY.shape:
            raise ValueError("argX and argY should have same shape, given"
                             + "{}, {}".format(argX.shape, argY.shape))

        res_dtype = np.result_type(argX._mantissa, self.coeffs._mantissa)
        res = Xrange_array.zeros(argX.shape, dtype=res_dtype)
        resX = Xrange_array.zeros(argX.shape, dtype=res_dtype)
        # Double Horner rule
        # [a11 a12 a13]  -> 1
        # [a21 a22 a23]  -> X 
        # [a31 a32 a33]  -> X2
        for i in range(0, self.cutdeg + 1):
            resXi = resX.copy()
            for j in range(0, self.cutdeg + 1):
                resXi = (
                    resXi * argY
                    + (self.coeffs[self.cutdeg - i, self.cutdeg - j])
                )
            res = res * argX + resXi
        return res


    def deriv(self, var):
        """
        var: "X" | "Y"
            str, the direction for the derivative
        """
        l = self.cutdeg + 1
        coeffs = Xrange_array.zeros((l, l), dtype=self.coeffs._mantissa.dtype)
        if var == "X":
            coeffs[:-1, :] = self.coeffs[1:, :] * np.arange(1, l)[:, np.newaxis]
        elif var == "Y":
            coeffs[:, :-1] = self.coeffs[:, 1:] * np.arange(1, l)[np.newaxis, :]
        else:
            raise ValueError("Invalid var")

        return Xrange_bivar_polynomial(coeffs, cutdeg=self.cutdeg)

    def taylor_shift(self, x0):#, quad_prec=False):
        raise NotImplementedError("Not implemented for bivar")

    def __repr__(self):
        return ("Xrange_polynomial(cutdeg="+ str(self.cutdeg) +",\n" +
                self.__str__() + ")")

    def __str__(self):
        return self._to_str()

    def _to_str(self):
        """
        Generate the full string representation of the polynomial, using
        `_monome_base_str` to generate each polynomial term.
        """
        if self.coeffs.is_complex:
            str_coeffs = self.coeffs._to_str_array()
        else:
            str_coeffs = np.abs(self.coeffs)._to_str_array()
        linewidth = np.get_printoptions().get('linewidth', 75)
        if linewidth < 1:
            linewidth = 1
        if self.coeffs.real[0, 0] >= 0.:
            out = f"{str_coeffs[0, 0][1:]}"
        else:
            out = f"-{str_coeffs[0, 0][1:]}"
        for j in range(self.cutdeg + 1):
            for i in range(self.cutdeg + 1):
                if (i + j) > self.cutdeg or (i == 0 and j == 0):
                    continue# , coef in enumerate(str_coeffs[1:]):
                coef = str_coeffs[i, j]
                out += " "
                poweri = str(i)
                powerj = str(j)
                # 1st Polynomial coefficient
                if (self.coeffs.is_complex) or self.coeffs.real[i, j] >= 0.:
                    next_term = f"+ {coef}"
                else:
                    next_term = f"- {coef}"
                # Polynomial term
                if i > 0:
                    next_term += self._monome_base_str(poweri, "X")
                if j > 0:
                    next_term += self._monome_base_str(powerj, "Y")
                # Length of the current line with next term added
                line_len = len(out.split('\n')[-1]) + len(next_term)
                # If not the last term in the polynomial, it will be two           
                # characters longer due to the +/- with the next term
                if i < len(self.coeffs[1:]) - 1:
                    line_len += 2
                # Handle linebreaking
                if line_len >= linewidth:
                    next_term = next_term.replace(" ", "\n", 1)
                next_term = next_term.replace("  ", " ")
                out += next_term
        return out

#    @classmethod
#    def _monome_base_str(cls, i, var_str):
#        return f"·{var_str}{i.translate(cls._superscript_mapping)}"


class Xrange_bivar_SA(Xrange_bivar_polynomial):
    """
    One-dimensionnal, extended-range serie approximation class based on
    Xrange_polynomial:
        - provides the same feature as Xrange_polynomial + control of a
            truncature error term
        - For the prupose of truncature error calculation, it is assumed that 
            the domain of convergence is enclosed in the unit circle.

    Parameters
    ----------
    coeffs: see Xrange_polynomial
    cutdeg: see Xrange_polynomial (Monomes of degree above cutoff will be 
            disregarded.)
    err : truncature error term, in X**(cutoff + 1). Default to 0.
    """  

    def __init__(self, coeffs, cutdeg, err=Xrange_array(0.)):
        self.err = err
        if not(isinstance(err, Xrange_array)):
            self.err = Xrange_array(err)
        super().__init__(coeffs, cutdeg)

    def  __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        casted_inputs = ()
        casted_cutdegs = ()
        casted_errs = ()

        for x in inputs:
            # Only support operations with instances of 
            # Xrange_array._HANDLED_TYPES.
            if isinstance(x, Xrange_bivar_SA):
                casted_cutdegs += (x.cutdeg,)
                casted_inputs += (x.coeffs,)
                casted_errs += (x.err,)
            else:
                casted_errs += (0.,)
                if isinstance(x, Xrange_bivar_polynomial):
                    casted_inputs += (x.coeffs,)
                    casted_cutdegs += (x.cutdeg,)
                elif isinstance(x, Xrange_array):
                    casted_inputs += (x.flatten(),)
                elif isinstance(x, np.ndarray):
                    casted_inputs += (x.flatten().view(Xrange_array),)
                elif isinstance(x, numbers.Number):
                    casted_inputs += (Xrange_array([x]),)
                elif isinstance(x, list):
                    casted_inputs += (Xrange_array(x),)
                else:
                    # Operation not supported (type not handled), return the
                    # sentinel value NotImplemented
                    return NotImplemented

        cutdeg = min(casted_cutdegs)
        if not all(item == cutdeg for item in casted_cutdegs):
            raise ValueError("Operation not supported, incompatible cutdegs {}"
                             .format(casted_cutdegs))

        out = kwargs.pop("out", None)

        if method == "__call__":
            if ufunc in [np.add, np.subtract]:
                return self._add(ufunc, casted_inputs, casted_errs,
                                 cutdeg=cutdeg, out=out)
            elif ufunc is np.negative:
                return self._negative(casted_inputs, casted_errs,
                                      cutdeg=cutdeg, out=out)
            elif ufunc is np.multiply:
                return self._mul(casted_inputs, casted_errs,
                                 cutdeg=cutdeg, out=out)
        # Other ufunc not supported
        return NotImplemented

    @staticmethod
    def _add(ufunc, inputs, errs, cutdeg, out=None):
        """ Add or Subtract 2 Xrange_bivar_SA """
        op0, op1 = inputs
        res_len = cutdeg + 1
        dtype=np.result_type(op0._mantissa, op1._mantissa)
        res = Xrange_array.zeros((res_len, res_len), dtype)

        if op0.size == 1:
            res[:] = op1
            res[0, 0] += op0[0]
            return Xrange_bivar_SA(res, cutdeg=cutdeg, err=sum(errs))
        if op1.size == 1:
            res[:] = op0 
            res[0, 0] += op1[0]
            return Xrange_bivar_SA(res, cutdeg=cutdeg, err=sum(errs))
 
        res[:] += op0
        if ufunc is np.add:
            res[:] += op1
        elif ufunc is np.subtract: 
            res[:] -= op1

        return Xrange_bivar_SA(res, cutdeg=cutdeg, err=sum(errs))

    @staticmethod
    def _negative(inputs, errs, cutdeg, out=None):
        """ Change sign of a Xrange_bivar_SA """
        op0, = inputs
        err0, = errs
        return Xrange_bivar_SA(-op0, cutdeg=cutdeg, err=err0)

    @staticmethod
    def _mul(inputs, errs, cutdeg, out=None):
        """ Multiply 2 Xrange_bivar_SA """
        op0, op1 = inputs
        err0, err1 = errs

        if op0.size == 1: # No trunc err term
            res = op0[0] * op1
            err = (
                op0[0] * err1
                + np.sqrt(np.sum(op1.abs2())) * err0
                + err1 * err0
            )
            return Xrange_bivar_SA(res, cutdeg=cutdeg, err=err)
        if op1.size == 1: # No trunc err term
            res = op0 * op1[0]
            err = (
                np.sqrt(np.sum(op0.abs2())) * err1
                + op1[0] * err0
                + err1 * err0
            )
            return Xrange_bivar_SA(res, cutdeg=cutdeg, err=err)

        res_len = cutdeg + 1
        dtype=np.result_type(op0._mantissa, op1._mantissa)
        res = Xrange_array.zeros((res_len, res_len), dtype)

        # The hard way
        for i in range(0, res_len):
            for j in range(0, res_len - i):
                for k in range(0, i + 1):
                    for l in range(0, j + 1):
                        res[i, j] += op0[k, l] * op1[i - k, j - l]

        # Truncature error term
        errT = Xrange_array.zeros([], dtype)
        for i in range(0, 2 * cutdeg + 1):
            for j in range(0, 2 * cutdeg + 1 - i):
                if (i + j) <= cutdeg: # Normal term, not an error
                    continue
                op_errT = Xrange_array.zeros([], dtype)
                for k in range(max(0, i - cutdeg), 
                               min(i + 1, res_len)):
                    for l in range(max(0, j - cutdeg),
                                   min(j + 1, res_len)):
                        op_errT += op0[k, l] * op1[i - k, j - l]
#                print("op_errT", op_errT)
                errT += op_errT.abs2()
        errT = np.sqrt(errT)

        # Sums0 and sums1 error term
        sums0 = np.sqrt(np.sum(op0.abs2()))
        sums1 = np.sqrt(np.sum(op1.abs2()))

        # Total err
        err = err0 * sums1 + err1 * sums0 + err0 * err1 + errT

        return Xrange_bivar_SA(res, cutdeg=cutdeg, err=err)


    def __repr__(self):
        return ("Xrange_bivar_SA(cutdeg="+ str(self.cutdeg) +",\n" +
                self.__str__() + ")")

    def __str__(self):
        return self._to_str()

    def _to_str(self):
        """
        Generate the full string representation of the SA, using
        `_monome_base_str` to generate each polynomial term.
        """
        out = super()._to_str()
        out += " // Res <= {}".format(self.err.__str__()
                ) + self._monome_base_str(str(self.cutdeg + 1), "[X|Y]")
        return out


class Xrange_monome:
    """
    Class for a monome in X.

    Multiplication with SA and bivar_SA is only implemented in numba

    Parameters
    ----------
    k : Xrange_array of size 1
        coefficient of the monome
    """
    def __init__(self, k):
        self.k = k.ravel()



if __name__ == "__main__":
    print("test")
    a = Xrange_array(np.array([
        [ 1.,  2.,  3.],
        [11., 12., 13.],
        [21., 22., 23.],
    ]))
    a = Xrange_array(np.array([
        [ 1.,  1.,  1.],
        [1., 1., 1.],
        [1., 1., 1.],
    ]))
    a = (1. ) * a
    print("a\n", a, a[1, 2], a[2, 0])
    bivar = Xrange_bivar_polynomial(a, cutdeg=2)
    print(bivar)
    print(bivar + bivar)
    print(bivar * bivar)
    print("eval")
    print(bivar(-1., 1.))
    print("deriv")
    print(bivar)
    print(bivar.deriv("Y"))