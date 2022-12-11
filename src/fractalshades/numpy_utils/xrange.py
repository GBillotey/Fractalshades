# -*- coding: utf-8 -*-
import numpy as np
import numbers
import re
import mpmath


def mpc_to_Xrange(mpc, dtype=np.complex128):
    """ Convert a mpc complex to a Xrange array"""
    select = {np.dtype(np.complex64): np.float32,
              np.dtype(np.complex128): np.float64}
    float_type = select[np.dtype(dtype)]

    mpcx_m, mpcx_exp = mpmath.frexp(mpc.real)
    mpcy_m, mpcy_exp = mpmath.frexp(mpc.imag)

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
            elif ufunc is np.power:
                # For power passing directly the inputs, as exponent shall not
                # be Xrange
                out = self._pow(inputs, out=out)
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
            # this branch will handle calls by np.angle with imag = 0. and
            # real a complex Xrange_array
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
    def _pow(inputs, out=None):
        """ x -> x ** alpha, alpha real """
        pow0, pow1 = inputs
        if isinstance(pow1, Xrange_array):
            raise NotImplementedError(
                "np.power called with a Xrange exponent, not implemented"
            )
        m0 = pow0._mantissa
        exp0 = pow0._exp

        exp_tot = exp0 * pow1
        exp_int = np.trunc(exp_tot)
        exp_frac = exp_tot - exp_int

        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)

        out["mantissa"], out["exp"] = Xrange_array._normalize(
            (m0 ** pow1) * (2. ** exp_frac),
            exp_int.astype(np.int32)
        )

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
