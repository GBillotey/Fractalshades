""""
Source code for a Cython extension modules which calls directly the
MPFR and MPC libraries.

The types mpz, mpq, mpfr and mpc are declared as extension types in gmpy2.pxd.
They correspond respectively to the C structures MPZ_Object, MPQ_Object,
MPFR_Object and MPC_Object.

The gmpy2.pxd header also provides convenience macro to wrap a (copy of) a
mpz_t, mpq_t, mpfr_t or a mpc_t object into the corresponding gmpy2 type.
"""
import fractalshades.settings
import fractalshades.numpy_utils.xrange as fsx

import numpy as np
import mpmath

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
    cdef double hypot(double x, double y)

# MPFR - https://www.mpfr.org/mpfr-current/mpfr.pdf
cdef extern from "mpfr.h":
    void mpfr_init2(mpfr_t x, mpfr_prec_t prec)
    void mpfr_clear(mpfr_t x)
    void mpfr_free_cache()

    int mpfr_set_si(mpfr_t rop, long int op, mpfr_rnd_t rnd)

    int mpfr_set_str(mpfr_t rop, char *op_str, int base, mpfr_rnd_t rnd)
    char *mpfr_get_str(char *res_str, mpfr_exp_t *expptr, int base, size_t n,
                        mpfr_t op, mpfr_rnd_t rnd)
    void mpfr_free_str(char *res_str)
    
    double mpfr_get_d(mpfr_t op, mpfr_rnd_t rnd)
    double mpfr_get_d_2exp(long *exp, mpfr_t op, mpfr_rnd_t rnd)
    
    int mpfr_add(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)
    int mpfr_sub(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)
    int mpfr_mul(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)
    int mpfr_mul_si(mpfr_t rop, mpfr_t op1, long int op2, mpfr_rnd_t rnd)
    int mpfr_sqr(mpfr_t rop, mpfr_t op, mpfr_rnd_t rnd)
    
    int mpfr_cmp_d(mpfr_t op1, double op2)
    int mpfr_cmp(mpfr_t op1, mpfr_t op2)
    int mpfr_greater_p(mpfr_t op1, mpfr_t op2)
    int mpfr_greaterequal_p(mpfr_t op1, mpfr_t op2)
    
    void mpfr_reldiff(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)

# MPC - http://www.multiprecision.org/downloads/mpc-1.2.1.pdf
cdef extern from "mpc.h":
    void mpc_init2(mpc_ptr x, mpfr_prec_t rnd);
    void mpc_clear(mpc_ptr x)
    void mpc_swap(mpc_t op1, mpc_t op2)

    int mpc_set_si_si(mpc_t rop, long int re, long int im, mpc_rnd_t rnd)
    int mpc_set_fr_fr(mpc_t rop, mpfr_t re, mpfr_t im, mpc_rnd_t rnd)

    # pointer to real and complex parts
    mpfr_t mpc_realref(mpc_t op)
    mpfr_t mpc_imagref(mpc_t op)

    void mpc_swap(mpc_t op1, mpc_t op2)

    int mpc_add(mpc_t rop, const mpc_t op1, const mpc_t op2, mpc_rnd_t rnd)
    int mpc_add_ui(mpc_t rop, const mpc_t op1, unsigned long int op2,
                   mpc_rnd_t rnd)

    int mpc_sub(mpc_t rop, const mpc_t op1, const mpc_t op2, mpc_rnd_t rnd)

    int mpc_mul(mpc_t rop, const mpc_t op1, const mpc_t op2, mpc_rnd_t rnd)
    int mpc_mul_si(mpc_t rop, const mpc_t op1, long int op2, mpc_rnd_t rnd)
    
    int mpc_div(mpc_t rop, const mpc_t op1, const mpc_t op2, mpc_rnd_t rnd)
    int mpc_div_ui(mpc_t rop, const mpc_t op1, unsigned long int op2,
                   mpc_rnd_t rnd)
    int mpc_ui_div(mpc_t rop, unsigned long int op1, const mpc_t op2,
                   mpc_rnd_t rnd)

    int mpc_sqr(mpc_t rop, const mpc_t op, mpc_rnd_t rnd)
    # Fused multiply-add of three complex numbers - suboptimal here
    int mpc_fma(mpc_ptr rop, mpc_srcptr a, mpc_srcptr b, mpc_srcptr c,
                 mpc_rnd_t rnd)

    int mpc_abs(mpfr_t rop, const mpc_t op, mpfr_rnd_t rnd)
    
    char *mpc_get_str(int b , size_t n , const mpc_t op , mpc_rnd_t rnd)


cdef extern from "Python.h":
    object Py_BuildValue(const char* format, ...)


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
        Precision used for the full precision calculation (in bits)
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
        double curr_partial = PARTIAL_TSHOLD
        double x = 0.
        double y = 0.
        double abs_i = 0.

        mpc_t z_t
        mpc_t c_t
        mpc_t tmp_t
        mpfr_t x_t, y_t

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
    
    Parameters:
    -----------
    x: str
        real part of the starting reference point
    y: str
        imag part of the starting reference point
    seed_prec
        Precision used for the full precision calculation (in bits)
        Usually one should just use `mpmath.mp.prec`
    order: int
        The reference cycle order

    Returns:
    --------
    A size estimate of the nucleus described by the input parameters


    Note: 
    -----
    C-translation of the following python code

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
        long i = 0

        mpc_t z_t, c_t, l_t, b_t, tmp_t, tmp2_t
        mpfr_t x_t, y_t

    # initialisation
    mpc_init2(z_t, seed_prec)
    mpc_init2(c_t, seed_prec)
    mpc_init2(l_t, seed_prec)
    mpc_init2(b_t, seed_prec)
    mpc_init2(tmp_t, seed_prec)
    mpc_init2(tmp2_t, seed_prec)

    mpfr_init2(x_t, seed_prec)
    mpfr_init2(y_t, seed_prec)

    # set value of c
    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(y_t, seed_y, 10, MPFR_RNDN)
    mpc_set_fr_fr(c_t, x_t, y_t, MPC_RNDNN)

    # set z = 0 l = 1 b = 1
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
    

def perturbation_mandelbrot_ball_method(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        char * seed_px,
        long maxiter,
        double M_divergence
):
    """
    Cycle order estimation by ball-method (iterations of a ball untils it
    contains 0.)
    
    Note : this is the implementation for the so-called "first order" ball
    method.

    Parameters:
    -----------
    x: str
        real part of the starting reference point
    y: str
        imag part of the starting reference point
    seed_prec
        Precision used for the full precision calculation (in bits)
        Usually one should just use `mpmath.mp.prec`
    maxiter: int
        The maximum number of iteration
    M_divergence: double
        The criteria for divergence

    Returns:
    --------
    i: int

    Note: 
    -----
    C-translation of the following python code

    def _ball_method1(c, px, maxiter, M_divergence):

        z = mpmath.mpc(0.)
        dzdc = mpmath.mpc(0.)
        r0 = px      # first radius
        r = r0 * 1.
        az = abs(z)

        for i in range(1, maxiter + 1):
            if i%10000 == 0:
                print("Ball method", i, r)
            r = (az  + r)**2 - az**2 + r0
            z = z**2 + c
            az = abs(z)
            if az > M_divergence:
                print("Fail in ball method")
                return -1
            if (r > az):
                print("Ball method 1 found period:", i)
                return i
    """
    cdef:
        int cmp = 0
        long ret = -1
        long i = 0

        mpc_t c_t, z_t, tmp_t
        mpfr_t x_t, y_t, r0_t, r_t, az_t, tmp_real1, tmp_real2


    mpc_init2(c_t, seed_prec)
    mpc_init2(z_t, seed_prec)
    mpc_init2(tmp_t, seed_prec)

    mpfr_init2(x_t, seed_prec)
    mpfr_init2(y_t, seed_prec)
    mpfr_init2(r0_t, seed_prec)
    mpfr_init2(r_t, seed_prec)
    mpfr_init2(az_t, seed_prec)
    mpfr_init2(tmp_real1, seed_prec)
    mpfr_init2(tmp_real2, seed_prec)

    # from char: set value of c - and of r0 = r = px
    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(y_t, seed_y, 10, MPFR_RNDN)
    mpc_set_fr_fr(c_t, x_t, y_t, MPC_RNDNN)
    mpfr_set_str(r0_t, seed_px, 10, MPFR_RNDN)
    mpfr_set_str(r_t, seed_px, 10, MPFR_RNDN)

    # set z = 0 
    mpc_set_si_si(z_t, 0, 0, MPC_RNDNN)

    # set az = abs(z)
    mpc_abs(az_t, z_t, MPFR_RNDN)
    
    for i in range(1, maxiter + 1):

        # r = (az  + r) ** 2 - az ** 2 + r0
        mpfr_add(tmp_real1, az_t, r_t, MPFR_RNDN)
        mpfr_sqr(tmp_real2, tmp_real1, MPFR_RNDN)
        mpfr_sqr(tmp_real1, az_t, MPFR_RNDN)
        mpfr_sub(r_t, tmp_real2, tmp_real1, MPFR_RNDN)
        mpfr_add(r_t, r_t, r0_t, MPFR_RNDN)

        #  z = z * z + c
        mpc_sqr(tmp_t, z_t, MPC_RNDNN)
        mpc_add(z_t, tmp_t, c_t, MPC_RNDNN)

        # az = abs(z)
        mpc_abs(az_t, z_t, MPFR_RNDN)

        # if az > M_divergence:
        cmp = mpfr_cmp_d(az_t, M_divergence)
        # Return a positive value if op1 > op2, zero if op1 = op2, and a
        # negative value if op1 < op2.
        if cmp > 0:
            ret = -1
            break

        # if (r > az):
        cmp = mpfr_greater_p(r_t, az_t)
        # Return non-zero if op1 > op2, op1 ≥ op2, op1 < op2, op1 ≤ op2,
        # op1 = op2 respectively, and zero otherwise.
        if cmp != 0:
            ret = i
            break

    mpc_clear(c_t)
    mpc_clear(z_t)
    mpc_clear(tmp_t)

    mpfr_clear(x_t)
    mpfr_clear(y_t)
    mpfr_clear(r0_t)
    mpfr_clear(r_t)
    mpfr_clear(az_t)
    mpfr_clear(tmp_real1)
    mpfr_clear(tmp_real2)
    
    return ret

def perturbation_mandelbrot_find_nucleus(
    char *seed_x,
    char *seed_y,
    long seed_prec,
    long order,
    long max_newton,
    char *seed_eps_cv
):
    """
    Run Newton search to find z0 so that f^n(z0) == 0 (n being the order input)

    This implementation includes a "divide by undesired roots" technique so
    that solutions using divisors of n are disregarded.

    Parameters:
    -----------
    x: str
        real part of the starting reference point
    y: str
        imag part of the starting reference point
    seed_prec
        Precision used for the full precision calculation (in bits)
        Usually one should just use `mpmath.mp.prec`
    order: int
        The candidate order of the attracting cycle
    max_newton: int
        Maximal number of iteration for Newton method. 80 is a good first 
        estimate.
    eps_cv: str
        The criteria for convergence
        eps_cv = mpmath.mpf(2.)**(-mpmath.mp.prec) is a good first estimate

    Returns:
    --------
    newton_cv: bool
    c: mpmath.mpc
        The nucleus center found by Newton search

    Note: 
    -----
    C-translation of the following python code

    def find_nucleus(c, order, max_newton=None, eps_cv=None):

        c_loop = c

        for i_newton in range(max_newton): 
            print("Newton iteration", i_newton, "order", order)
            zr = mpmath.mp.zero
            dzrdc = mpmath.mp.zero
            h = mpmath.mp.one
            dh = mpmath.mp.zero
            for i in range(1, order + 1):# + 1):
                dzrdc = 2. * dzrdc * zr + 1. #  mpmath.mpf("2.")
                zr = zr * zr + c_loop
                # divide by unwanted periods
                if i < order and order % i == 0:
                    h *= zr
                    dh += dzrdc / zr
            f = zr / h
            df = (dzrdc * h - zr * dh) / (h * h)
            cc = c_loop - f / df
            # abs(cc - c_loop) <= eps_cv * 2**4
            newton_cv = mpmath.almosteq(cc, c_loop) 
            c_loop = cc
            if newton_cv:
                break 

        return newton_cv, c_loop
        
    See also:
    ---------
    https://mathr.co.uk/blog/2018-11-17_newtons_method_for_periodic_points.html
    """
    cdef:
        unsigned long int ui_one = 1
        bint newton_cv = False
        int cmp = 0
        long fail = -1
        long i_newton = 0
        long i = 0
        object gmpy_mpc

        mpc_t c_t
        mpc_t zr_t
        mpc_t dzrdc_t
        mpc_t h_t
        mpc_t dh_t
        mpc_t f_t
        mpc_t df_t
        mpc_t tmp_t1
        mpc_t tmp_t2
        mpc_t tmp_t3

        mpfr_t x_t
        mpfr_t y_t
        mpfr_t abs_diff
        mpfr_t eps_t

    
    mpc_init2(c_t, seed_prec)
    mpc_init2(zr_t, seed_prec)
    mpc_init2(dzrdc_t, seed_prec)
    mpc_init2(h_t, seed_prec)
    mpc_init2(dh_t, seed_prec)
    mpc_init2(f_t, seed_prec)
    mpc_init2(df_t, seed_prec)
    mpc_init2(tmp_t1, seed_prec)
    mpc_init2(tmp_t2, seed_prec)
    mpc_init2(tmp_t3, seed_prec)

    mpfr_init2(x_t, seed_prec)
    mpfr_init2(y_t, seed_prec)
    mpfr_init2(abs_diff, seed_prec)
    mpfr_init2(eps_t, seed_prec)
    
    # from char: set value of c - and of r0 = r = px
    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(y_t, seed_y, 10, MPFR_RNDN)
    mpc_set_fr_fr(c_t, x_t, y_t, MPC_RNDNN)
    mpfr_set_str(eps_t, seed_eps_cv, 10, MPFR_RNDN)

    # Value of epsilon
    # 64 = 2**4 * 4  - see the notes below on mpmath.almosteq
    mpfr_mul_si(eps_t, eps_t, 64, MPFR_RNDN) 

    for i_newton in range(max_newton):
        # zr = dzrdc = dh = 0 /  h = 1
        mpc_set_si_si(zr_t, 0, 0, MPC_RNDNN)
        mpc_set_si_si(dzrdc_t, 0, 0, MPC_RNDNN)
        mpc_set_si_si(h_t, 1, 0, MPC_RNDNN)
        mpc_set_si_si(dh_t, 0, 0, MPC_RNDNN)
        
        # Newton descent
        for i in range(1, order + 1):
            # dzrdc = 2. * dzrdc * zr + 1.
            mpc_mul_si(tmp_t1, dzrdc_t, 2, MPC_RNDNN)
            mpc_mul(tmp_t2, tmp_t1, zr_t, MPC_RNDNN)
            mpc_add_ui(dzrdc_t, tmp_t2, ui_one, MPC_RNDNN)

            # zr = zr * zr + c_loop
            mpc_sqr(tmp_t1, zr_t, MPC_RNDNN)
            mpc_add(zr_t, tmp_t1, c_t, MPC_RNDNN)

            # Divide by unwanted periods
            if (i < order) and (order % i == 0):
                # h *= zr
                mpc_mul(tmp_t1, h_t, zr_t, MPC_RNDNN)
                mpc_swap(tmp_t1, h_t)
                
                # dh += dzrdc / zr
                mpc_div(tmp_t1, dzrdc_t, zr_t, MPC_RNDNN)
                mpc_add(tmp_t2, tmp_t1, dh_t, MPC_RNDNN)
                mpc_swap(tmp_t2, dh_t)
                
        # f = zr / h
        mpc_div(f_t, zr_t, h_t, MPC_RNDNN)

        # df = (dzrdc * h - zr * dh) / (h * h)
        mpc_mul(tmp_t1, dzrdc_t, h_t, MPC_RNDNN)
        mpc_mul(tmp_t2, zr_t, dh_t, MPC_RNDNN)
        mpc_sub(tmp_t3, tmp_t1, tmp_t2, MPC_RNDNN)
        mpc_sqr(tmp_t1, h_t, MPC_RNDNN)
        mpc_div(df_t, tmp_t3, tmp_t1, MPC_RNDNN)

        # cc = c_loop - f / df
        mpc_div(tmp_t1, f_t, df_t, MPC_RNDNN)
        mpc_sub(tmp_t2, c_t, tmp_t1, MPC_RNDNN)
        mpc_swap(tmp_t2, c_t)

        # mpmath.almosteq(s, t, rel_eps=None, abs_eps=None)
        #
        #  """Determine whether the difference between s and t is smaller than
        #  a given epsilon, either relatively or absolutely.
        #  Both a maximum relative difference and a maximum difference
        #  (‘epsilons’) may be specified. The absolute difference is defined
        #  as |s−t| and the relative difference is defined as
        #  |s−t|/max(|s|,|t|)
        #  If only one epsilon is given, both are set to the same value.
        #  If none is given, both epsilons are set to 2**(−p+m)  where p is
        #  the current working precision and m is a small integer. The
        #  default setting allows almosteq() to be used to check for
        #  mathematical equality in the presence of small rounding errors."""

        # Notes: from mpmath source code, m = 4. We also know that
        # 0.25 <= |c| <= 2.0. Hence the implementation proposed here.

        mpc_abs(abs_diff, tmp_t1, MPFR_RNDN)
        cmp = mpfr_greaterequal_p(eps_t, abs_diff)
        if cmp != 0:
            newton_cv = True
            break

    gmpy_mpc = GMPy_MPC_From_mpc(c_t)

    mpc_clear(c_t)
    mpc_clear(zr_t)
    mpc_clear(dzrdc_t)
    mpc_clear(h_t)
    mpc_clear(dh_t)
    mpc_clear(f_t)
    mpc_clear(df_t)
    mpc_clear(tmp_t1)
    mpc_clear(tmp_t2)
    mpc_clear(tmp_t3)

    mpfr_clear(x_t)
    mpfr_clear(y_t)
    mpfr_clear(abs_diff)
    mpfr_clear(eps_t)

    with mpmath.workprec(seed_prec):
        mpmath_mpc = mpmath.mpc(
            mpmath.mpf(gmpy_mpc.real.as_mantissa_exp()),
            mpmath.mpf(gmpy_mpc.imag.as_mantissa_exp())
        )
    return (newton_cv, mpmath_mpc)


def _test_mpfr_to_python(
        char *seed_x,
        long seed_prec
):
    """ Debug / test function: mpfr_t to mpmath.mpf
    """
    cdef:
        mpfr_t x_t
        object ret
    
    mpfr_init2(x_t, seed_prec)

    # from char *: set value of x
    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_mul_si(x_t, x_t, 2, MPFR_RNDN)

    ret = GMPy_MPFR_From_mpfr(x_t)
    mpfr_clear(x_t)

    with mpmath.workprec(seed_prec):
        return mpmath.mpf(ret.as_mantissa_exp())


def _test_mpc_to_python(
        char *seed_x,
        char *seed_y,
        long seed_prec
):
    """ Debug / test function: mpc_t to mpmath.mpc
    """
    cdef:
        mpfr_t x_t
        mpfr_t y_t
        mpc_t c_t
        object ret
    
    mpfr_init2(x_t, seed_prec)
    mpfr_init2(y_t, seed_prec)
    mpc_init2(c_t, seed_prec)
    
    # from char: set value of c - and of r0 = r = px
    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(y_t, seed_y, 10, MPFR_RNDN)
    mpc_set_fr_fr(c_t, x_t, y_t, MPC_RNDNN)

    # from char *: set value of x
    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_mul_si(x_t, x_t, 2, MPFR_RNDN)

    ret = GMPy_MPC_From_mpc(c_t)

    mpfr_clear(x_t)
    mpfr_clear(y_t)
    mpc_clear(c_t)


    with mpmath.workprec(seed_prec):
        return mpmath.mpc(
            mpmath.mpf(ret.real.as_mantissa_exp()),
            mpmath.mpf(ret.imag.as_mantissa_exp())
        )



# mpmath.mpc(ret.as_mantissa_exp())
#
##
cdef mpfr_to_mpmath(mpfr_t x_t):
    """ Convert a mpfr_t to a mpmath.mpf """
    str_x, exp_x = mpfr_to_str(x_t)

    if str_x[0] == "-":
        str_x = "-." + str_x[1:]
    else:
        str_x = "." + str_x

    return mpmath.ldexp(mpmath.mpf(str_x), exp_x)


cdef mpfr_to_str(mpfr_t x_t):
    """ 
    Take a mpfr and returns a pair of
    mantissa as a string with implicit "." before first digit
    exp as a Python integer
    """
    cdef:
        object result_x
        char *buffer_x
        mpfr_exp_t exp_x

    # to Python str
    buffer_x = mpfr_get_str(NULL, &exp_x, 10, 0, x_t, MPFR_RNDN)
    result_x = Py_BuildValue("(si)", buffer_x, exp_x)

    mpfr_free_str(buffer_x)
    mpfr_clear(x_t)

    return result_x
