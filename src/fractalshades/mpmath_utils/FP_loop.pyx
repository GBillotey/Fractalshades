""""
Cython 

The types mpz, mpq, mpfr and mpc are declared as extension types in gmpy2.pxd.
They correspond respectively to the C structures MPZ_Object, MPQ_Object,
MPFR_Object and MPC_Object.

The gmpy2.pxd header also provides convenience macro to wrap a (copy of) a
mpz_t, mpq_t, mpfr_t or a mpc_t object into the corresponding gmpy2 type.
"""
import fractalshades.settings
import fractalshades.numpy_utils.xrange as fsx

import numpy as np

cimport numpy as np
from gmpy2 cimport *

# https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
np.import_array()

# Initialize the gmpy2 C-API - see some examples at
# https://github.com/aleaxit/gmpy/blob/master/test_cython/test_cython.pyx
import_gmpy2()

# Note : do NOT use complex.h it is a can of worms under windows as 
# Visual Studio is not C99 compatible :
# https://stackoverflow.com/questions/57837255/defining-dcomplex-externally-in-cython
# cdef extern from "complex.h":
#     pass

cdef extern from "math.h":
    cpdef double hypot(double x, double y)

# MPFR - https://machinecognitis.github.io/Math.Mpfr.Native/html/3a4a909f-0c87-700e-42d0-7159d6a6bf37.htm
cdef extern from "mpfr.h":
    void mpfr_init2 (mpfr_t x, mpfr_prec_t prec)
    void mpfr_clear (mpfr_t x)

    int mpfr_set_si (mpfr_t rop, long int op, mpfr_rnd_t rnd)
    int mpfr_set_str (mpfr_t rop, char* op_str, int base, mpfr_rnd_t rnd)
    
    double mpfr_get_d(mpfr_t op, mpfr_rnd_t rnd)
    double mpfr_get_d_2exp (long *exp, mpfr_t op, mpfr_rnd_t rnd)

# MPC - http://www.multiprecision.org/downloads/mpc-1.2.1.pdf
cdef extern from "mpc.h":
    void mpc_init2 (mpc_ptr x, mpfr_prec_t rnd);
    void mpc_clear (mpc_ptr x)
    void mpc_swap (mpc_t op1, mpc_t op2)

    int mpc_set_si_si (mpc_t rop, long int re, long int im, mpc_rnd_t rnd)
    int mpc_set_fr_fr (mpc_t rop, mpfr_t re, mpfr_t im, mpc_rnd_t rnd)

    # pointer to real and complex parts
    mpfr_t mpc_realref(mpc_t op)
    mpfr_t mpc_imagref(mpc_t op)

    void mpc_swap (mpc_t op1, mpc_t op2)

    int mpc_add (mpc_t rop, mpc_t op1, mpc_t op2, mpc_rnd_t rnd)

    int mpc_mul (mpc_t rop, mpc_t op1, mpc_t op2, mpc_rnd_t rnd)
    int mpc_mul_si (mpc_t rop, mpc_t op1, long int op2, mpc_rnd_t rnd)
    
    int mpc_div (mpc_t rop, mpc_t op1, mpc_t op2, mpc_rnd_t rnd)
    int mpc_div_ui (mpc_t rop, mpc_t op1, unsigned long int op2, mpc_rnd_t rnd)
    int mpc_ui_div (mpc_t rop, unsigned long int op1, mpc_t op2, mpc_rnd_t rnd)

    int mpc_sqr (mpc_t rop, mpc_t op, mpc_rnd_t rnd)
    # Fused multiply-add of three complex numbers - suboptimal here
    int mpc_fma (mpc_ptr rop, mpc_srcptr a, mpc_srcptr b, mpc_srcptr c,
                 mpc_rnd_t rnd)


DTYPE_INT = np.int32
ctypedef np.int32_t DTYPE_INT_t

DTYPE_COMPLEX = np.complex128
ctypedef np.complex128_t DTYPE_COMPLEX_t

DTYPE_FLOAT = np.float64
ctypedef np.float64_t DTYPE_FLOAT_t

ctypedef np.npy_bool DTYPE_BOOL_t

cdef:
    double PARTIAL_TSHOLD = fractalshades.settings.newton_zoom_level
    double XRANGE_TSHOLD = fractalshades.settings.xrange_zoom_level


def perturbation_mandelbrot_FP_loop(
        np.ndarray[DTYPE_FLOAT_t, ndim=1] orbit,
       # np.ndarray[DTYPE_BOOL_t, ndim=1] orbit_is_Xrange,
        bint need_Xrange,
        long max_iter,
        double M,
        char * seed_x,
        char * seed_y,
        long seed_prec
    ):
    """
    Full precision orbit for standard Mandelbrot
    
    Parameters
    ----------
    orbit arr
        low prec (np.complex 128 viewed as 2 np.float64 components)
        array which will be filled with the orbit pts
    need_Xrange bool
        bool - wether we shall worry about ultra low values (Xrange needed)
    max_iter : maximal iteration.
    M : radius
    seed_x : C char array representing screen pixel abscissa
        Note: use str(x).encode('utf8') if x is a python str
    seed_y : C char array representing screen pixel abscissa
        Note: use str(y).encode('utf8') if y is a python str
    int seed_prec
        Precision used for the full precision calculation
        Usually one should just use `mpmath.mp.prec`

    Return
    ------
    i int
        index of exit
    orbit_Xrange_register
        dictionnary containing the special iterations thats need Xrange
    orbit_partial_register
        dictionnary containing the partials
    """
    cdef:
        long max_len = orbit.shape[0]
        long i = 0
        long print_freq = 0
#        double complex tmp_dc = 0j
        double curr_partial = PARTIAL_TSHOLD
        double x = 0.
        double y = 0.

        mpc_t z_t
        mpc_t c_t
        mpc_t tmp_t
        mpfr_t x_t
        mpfr_t y_t

    assert orbit.dtype == DTYPE_FLOAT
    assert max_iter <= max_len + 1 # (NP_orbit starts at critical point)

    orbit_Xrange_register = dict()
    orbit_partial_register = dict()

    mpc_init2(z_t, seed_prec)
    mpc_init2(c_t, seed_prec)
    mpc_init2(tmp_t, seed_prec)
    mpfr_init2(x_t, seed_prec)
    mpfr_init2(y_t, seed_prec)

    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(y_t, seed_y, 10, MPFR_RNDN)
    mpc_set_fr_fr(c_t, x_t, y_t, MPC_RNDNN)
    
    # For standard Mandelbrot the critical point is 0. - initializing
    # the FP z and the NP_orbit
    mpc_set_si_si(z_t, 0, 0, MPC_RNDNN)
    # Complex 0. :
    orbit[0] = 0.
    orbit[1] = 0.

    print_freq = max(10000, (int(max_iter / 100.) // 10000 + 1) * 10000)
    print("============================================")
    print("Mandelbrot iterations, full precision: ", seed_prec)
    print("Output every: ", print_freq, flush=True)

    for i in range(1, max_iter + 1):

        mpc_sqr(tmp_t, z_t, MPC_RNDNN)
        mpc_add(z_t, tmp_t, c_t, MPC_RNDNN)

        # C _Complex type assignment to numpy complex128 array is not
        # straightforward, using 2 float64 components
        x = mpfr_get_d(mpc_realref(z_t), MPFR_RNDN)
        y = mpfr_get_d(mpc_imagref(z_t), MPFR_RNDN)
        orbit[2 * i] = x
        orbit[2 * i + 1] = y
        
        # take the norm 
        abs_i = hypot(x, y)

        if abs_i > M: # escaping
            break

        # Handles the special case where the orbit goes closer to the critical
        # point than a standard double can handle.
        if need_Xrange and (abs_i < XRANGE_TSHOLD):
            orbit_Xrange_register[i] = mpc_t_to_Xrange(z_t)

        # Hanldes the successive partials
        if abs_i <= curr_partial:
            # We need to check more precisely (due to the 'Xrange' cases)
            try:
                curr_index= next(reversed(orbit_partial_register.keys()))
                curr_partial_Xrange = orbit_partial_register[curr_index]
            except StopIteration:
                curr_partial_Xrange = fsx.Xrange_array([curr_partial])
            candidate_partial = mpc_t_to_Xrange(z_t)

            if candidate_partial.abs2() < curr_partial_Xrange.abs2():
                orbit_partial_register[i] = candidate_partial
                curr_partial = abs(candidate_partial.to_standard())

        # Outputs every print_freq
        if i % print_freq == 0:
            print("FP loop", i, flush=True)
    
    # If we did not escape from the last loop, first invalid iteration is i + 1
    div = True
    if (i == max_iter) and (abs_i <= M):
        div = False
        i += 1

    print("FP loop completed at iteration: ", i)
    print("Divergence ? : ", div)
    print("============================================", flush=True)

    mpc_clear(z_t)
    mpc_clear(c_t)
    mpc_clear(tmp_t)
    mpfr_clear(x_t)
    mpfr_clear(y_t)

    return i, orbit_partial_register, orbit_Xrange_register


cdef mpc_t_to_Xrange(mpc_t z_t):
    """
    Convert a mpc_t to a fsx.Xrange_array
    """
    cdef:
        long x_exp = 0
        long y_exp = 0
        long * x_exp_ptr = &x_exp
        long * y_exp_ptr = &y_exp
        double x_mantissa = 0.
        double y_mantissa = 0.

    x_mantissa = mpfr_get_d_2exp(x_exp_ptr, mpc_realref(z_t), MPFR_RNDN)
    y_mantissa = mpfr_get_d_2exp(y_exp_ptr, mpc_imagref(z_t), MPFR_RNDN)

    x_Xr = fsx.Xrange_array([x_mantissa], x_exp, DTYPE_COMPLEX)
    y_Xr = fsx.Xrange_array([y_mantissa], y_exp, DTYPE_COMPLEX)
    return (x_Xr + 1j * y_Xr)


def perturbation_mandelbrot_nucleus_size_estimate(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        long order
    ):
    """
    Hyperbolic component size estimate. Reference :
    https://mathr.co.uk/blog/2016-12-24_deriving_the_size_estimate.html
    
    Note : C-translation of the following python code

    def nucleus_size_estimate(c, order):
        z = mpmath.mpc(0., 0.)
        l = mpmath.mpc(1., 0.)
        b = mpmath.mpc(1., 0.)
        for i in range(1, order):
            z = z * z + c
            l = 2. * z * l
            b = b + 1. / l
        return 1. / (b * l * l)
    """
    cdef:
        unsigned long int ui_one = 1
        mpc_t z_t
        mpc_t c_t
        mpc_t l_t
        mpc_t b_t
        mpc_t tmp_t
        mpc_t tmp2_t
        mpfr_t x_t
        mpfr_t y_t

    mpc_init2(z_t, seed_prec)
    mpc_init2(c_t, seed_prec)
    mpc_init2(l_t, seed_prec)
    mpc_init2(b_t, seed_prec)
    mpc_init2(tmp_t, seed_prec)
    mpc_init2(tmp2_t, seed_prec)

    mpfr_init2(x_t, seed_prec)
    mpfr_init2(y_t, seed_prec)

    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(y_t, seed_y, 10, MPFR_RNDN)
    mpc_set_fr_fr(c_t, x_t, y_t, MPC_RNDNN)

    # Initializing z = 0 l = 1 b = 1
    mpc_set_si_si(z_t, 0, 0, MPC_RNDNN)
    mpc_set_si_si(l_t, 1, 0, MPC_RNDNN)
    mpc_set_si_si(b_t, 1, 0, MPC_RNDNN)
    mpc_set_si_si(tmp_t, 0, 0, MPC_RNDNN)
    mpc_set_si_si(tmp2_t, 0, 0, MPC_RNDNN)

    for i in range(1, order):
        #  z = z * z + c
        mpc_sqr(tmp_t, z_t, MPC_RNDNN)
        mpc_add(z_t, tmp_t, c_t, MPC_RNDNN)

        # l = 2. * z * l
        mpc_mul(tmp_t, z_t, l_t, MPC_RNDNN)
        mpc_mul_si(l_t, tmp_t, 2, MPC_RNDNN)

        # b = b + 1. / l
        mpc_ui_div(tmp_t, ui_one, l_t, MPC_RNDNN)
        mpc_add(tmp2_t, b_t, tmp_t, MPC_RNDNN)
        mpc_swap(tmp2_t, b_t)

    # return 1. / (b * l * l)
    mpc_sqr(tmp_t, l_t, MPC_RNDNN)
    mpc_mul(tmp2_t, b_t, tmp_t, MPC_RNDNN)
    mpc_ui_div(tmp_t, ui_one, tmp2_t, MPC_RNDNN)
    
    ret = mpc_t_to_Xrange(z_t)

    mpc_clear(z_t)
    mpc_clear(c_t)
    mpc_clear(l_t)
    mpc_clear(b_t)
    mpc_clear(tmp_t)
    mpc_clear(tmp2_t)

    mpfr_clear(x_t)
    mpfr_clear(y_t)

    return ret
    
    
    