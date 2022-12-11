"""
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
import gmpy2
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
    void mpfr_inits2(mpfr_prec_t prec, mpfr_t x, ...)
    void mpfr_clear(mpfr_t x)
    void mpfr_clears(mpfr_t x, ...)
    void mpfr_swap(mpfr_t x, mpfr_t y)
    void mpfr_free_cache()

    int mpfr_set_si(mpfr_t rop, long int op, mpfr_rnd_t rnd)

    int mpfr_set_str(mpfr_t rop, char *op_str, int base, mpfr_rnd_t rnd)
    char *mpfr_get_str(char *res_str, mpfr_exp_t *expptr, int base, size_t n,
                        mpfr_t op, mpfr_rnd_t rnd)
    void mpfr_free_str(char *res_str)
    
    double mpfr_get_d(mpfr_t op, mpfr_rnd_t rnd)
    double mpfr_get_d_2exp(long *exp, mpfr_t op, mpfr_rnd_t rnd)

    int mpfr_add(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)
    int mpfr_add_si(mpfr_t rop, mpfr_t op1, long int op2, mpfr_rnd_t rnd)
    int mpfr_sub(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)
    int mpfr_mul(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)
    int mpfr_mul_si(mpfr_t rop, mpfr_t op1, long int op2, mpfr_rnd_t rnd)
    int mpfr_sqr(mpfr_t rop, mpfr_t op, mpfr_rnd_t rnd)
    int mpfr_ui_div(mpfr_t rop, unsigned long int op1, mpfr_t op2,
                    mpfr_rnd_t rnd)
    
    int mpfr_cmp_d(mpfr_t op1, double op2)
    int mpfr_cmp(mpfr_t op1, mpfr_t op2)
    int mpfr_greater_p(mpfr_t op1, mpfr_t op2)
    int mpfr_greaterequal_p(mpfr_t op1, mpfr_t op2)

    int mpfr_abs(mpfr_t rop, mpfr_t op, mpfr_rnd_t rnd)
    int mpfr_sgn(mpfr_t op)
    void mpfr_reldiff(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)

    int mpfr_rec_sqrt(mpfr_t rop, mpfr_t op, mpfr_rnd_t rnd)
    int mpfr_hypot(mpfr_t rop, mpfr_t x, mpfr_t y, mpfr_rnd_t rnd)


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
    int mpc_mul_fr(mpc_t rop, const mpc_t op1, const mpfr_t op2, mpc_rnd_t rnd)
    int mpc_mul_si(mpc_t rop, const mpc_t op1, long int op2, mpc_rnd_t rnd)
    int mpc_mul_ui(mpc_t rop, const mpc_t op1 , unsigned long int op2 ,
                   mpc_rnd_t rnd )

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
    
    int mpc_pow_ui(mpc_t rop, const mpc_t op1, unsigned long op2,
                   mpc_rnd_t rnd)
    int mpc_pow_d(mpc_t rop, const mpc_t op1, double op2 , mpc_rnd_t rnd)


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
    double XR_TSHOLD = fractalshades.settings.xrange_zoom_level


# Type for in-place iterations of Mandelbrot Mn or other holomorphic formula
# zn+1 = f(zn, c)
ctypedef void (*iterMn_zn_t)(
    unsigned long, mpc_t, mpc_t, mpc_t
)

# Type for in-place iterations for Mandelbrot Mn or other holomorphic formula
# dzndc = f(zn, dzndc) or dzndz = f(zn, dzndz)
ctypedef void (*iterMn_dzndz_t)(
    unsigned long, int, mpc_t, mpc_t, mpc_t
)

cdef void iter_M2(
    unsigned long exponent, mpc_t z_t, mpc_t c_t, mpc_t tmp_t
):
    # One standard Mandelbrot M2 iteration - here exponent == 2
    # zn <- zn **2 + c
    mpc_sqr(tmp_t, z_t, MPC_RNDNN)
    mpc_add(z_t, tmp_t, c_t, MPC_RNDNN)
    return 

cdef void iter_Mn(
    unsigned long exponent, mpc_t z_t, mpc_t c_t, mpc_t tmp_t
):
    # One standard Mandelbrot Mn iteration
    # zn <- zn ** exponent + c
    mpc_pow_ui(tmp_t, z_t, exponent, MPC_RNDNN)
    mpc_add(z_t, tmp_t, c_t, MPC_RNDNN)
    return


cdef void iter_deriv_M2(
    unsigned long exponent, int var_c_z,
    mpc_t z_t, mpc_t dzndz_t, mpc_t tmp_t
):
    # One derivative Mandelbrot M2 iteration - here exponent == 2
    # dzndz <- 2 * dzndz * zn * (+ 1)
    # if var_c_z == 0 : we compute the derivatives wrt c
    # if var_c_z == 1 : we compute the derivatives wrt z
    mpc_mul(tmp_t, z_t, dzndz_t, MPC_RNDNN)
    mpc_mul_si(dzndz_t, tmp_t, 2, MPC_RNDNN)

    if var_c_z == 0:
        mpc_add_ui(dzndz_t, dzndz_t, 1, MPC_RNDNN)

    return

cdef void iter_deriv_Mn(
    unsigned long exponent, int var_c_z,
    mpc_t z_t, mpc_t dzndz_t, mpc_t tmp_t
):
# iter_deriv(exponent, 0, z_t, dzndc_t, tmp_t)

    # One derivative Mandelbrot Mn iteration
    # dzndz <- n * dzndz * zn ** (n-1) * (+ 1)
    # if var_c_z == 0 : we compute the derivatives wrt c
    # if var_c_z == 1 : we compute the derivatives wrt z
    cdef unsigned long exponent_m1 = (exponent - 1)
    mpc_pow_ui(tmp_t, z_t, exponent_m1, MPC_RNDNN)
    mpc_mul(tmp_t, dzndz_t, tmp_t, MPC_RNDNN)
    mpc_mul_ui(dzndz_t, tmp_t, exponent, MPC_RNDNN)

    if var_c_z == 0:
        mpc_add_ui(dzndz_t, dzndz_t, 1, MPC_RNDNN)

    return


cdef iterMn_zn_t select_Mnfunc(unsigned long exponent):
    # Selects the right code path for orbit calculation
    cdef iterMn_zn_t iterMn_zn
    if exponent == 2:
        iterMn_zn = &iter_M2
    elif exponent > 2:
        iterMn_zn = &iter_Mn
    else:
        raise NotImplementedError(exponent)
    return iterMn_zn

cdef iterMn_dzndz_t select_Mn_deriv_func(unsigned long exponent):
    # Selects the right code path for orbit derivative calculation
    cdef iterMn_dzndz_t iterMn_dzndz
    if exponent == 2:
        iterMn_dzndz = &iter_deriv_M2
    elif exponent > 2:
        iterMn_dzndz = &iter_deriv_Mn
    else:
        raise NotImplementedError(exponent)
    return iterMn_dzndz


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
    Full precision orbit for standard Mandelbrot (power 2)
    """
    cdef unsigned long exponent = 2
    return perturbation_mandelbrotN_select_FP_loop(
        orbit, need_Xrange, max_iter, exponent, M, seed_x, seed_y, seed_prec
    )


def perturbation_mandelbrotN_FP_loop(
        np.ndarray[DTYPE_FLOAT_t, ndim=1] orbit,
        bint need_Xrange,
        long max_iter,
        unsigned long exponent,
        double M,
        char * seed_x,
        char * seed_y,
        long seed_prec
):
    """
    Full precision orbit for standard Mandelbrot with power `exponent`, 
    integer >= 2
    """
    return perturbation_mandelbrotN_select_FP_loop(
        orbit, need_Xrange, max_iter, exponent, M, seed_x, seed_y, seed_prec
    )


def perturbation_mandelbrotN_select_FP_loop(
        np.ndarray[DTYPE_FLOAT_t, ndim=1] orbit,
        bint need_Xrange,
        long max_iter,
        unsigned long exponent,
        double M,
        char * seed_x,
        char * seed_y,
        long seed_prec,
):
    """
    Full precision orbit for standard Mandelbrot (power = 2) or higher
    exponent (integer exponent > 2).

    Parameters
    ----------
    orbit arr
        low prec (np.complex 128 viewed as 2 np.float64 components)
        array which will be filled with the orbit pts
    need_Xrange bool
        bool - wether we shall worry about ultra low values (Xrange needed)
    max_iter : maximal iteration.
    exponent: int >= 2, exponent for this Mandelbrot fractal 
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
    # Internal private func, common code path for M2 and Mn variants
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
        iterMn_zn_t iter_func


    iter_func = select_Mnfunc(exponent)

    assert orbit.dtype == DTYPE_FLOAT
    print(max_iter, max_len, max_iter * 2)
    assert (max_iter + 1) * 2 == max_len # (NP_orbit starts at critical point)

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

        iter_func(exponent, z_t, c_t, tmp_t)

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
        if need_Xrange and (abs_i < XR_TSHOLD):
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

cdef mpfr_t_to_Xrange(mpfr_t x_t):
    """
    Convert a mpc_t to a fsx.Xrange_array
    """
    cdef:
        long x_exp = 0
        long * x_exp_ptr = &x_exp
        double x_mantissa = 0.

    x_mantissa = mpfr_get_d_2exp(x_exp_ptr, x_t, MPFR_RNDN)
    x_Xr = fsx.Xrange_array([x_mantissa], x_exp, DTYPE_COMPLEX)
    return x_Xr 


def perturbation_mandelbrot_nucleus_size_estimate(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        long order
):
    """
    Hyperbolic component size estimate for standard Mandelbrot (power 2)
    """
    cdef unsigned long exponent = 2
    return perturbation_mandelbrotN_select_nucleus_size_estimate(
        seed_x, seed_y, seed_prec, exponent, order
    )


def perturbation_mandelbrotN_nucleus_size_estimate(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        unsigned long exponent,
        long order
):
    """
    Hyperbolic component size estimate for standard Mandelbrot with power
    `exponent`, integer >= 2
    """
    return perturbation_mandelbrotN_select_nucleus_size_estimate(
        seed_x, seed_y, seed_prec, exponent, order
    )

def perturbation_mandelbrotN_select_nucleus_size_estimate(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        unsigned long exponent,
        long order
):
    """
    Hyperbolic component size estimate. Reference :
    https://mathr.co.uk/blog/2016-12-24_deriving_the_size_estimate.html
    
    For power n > 2:
https://fractalforums.org/fractal-mathematics-and-new-theories/28/miniset-and-embedded-julia-size-estimates/912/msg4815;topicseen#msg4815

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
        double map_normalisation_exponent
        long i = 0

        mpc_t z_t, c_t, l_t, b_t, tmp_t, tmp2_t
        mpfr_t x_t, y_t

        iterMn_zn_t iter_func
        iterMn_dzndz_t iter_deriv

        
    iter_func = select_Mnfunc(exponent)
    iter_deriv = select_Mn_deriv_func(exponent)

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

    for i in range(1, order):
        #  z = z * z + c
        iter_func(exponent, z_t, c_t, tmp_t)

        # l = 2. * z * l
        iter_deriv(exponent, 1, z_t, l_t, tmp_t)

        # b = b + 1. / l
        mpc_ui_div(tmp_t, ui_one, l_t, MPC_RNDNN)
        mpc_add(tmp2_t, b_t, tmp_t, MPC_RNDNN)
        mpc_swap(tmp2_t, b_t)

    # return 1. / (b * l * l)
    if exponent == 2:
        mpc_sqr(tmp_t, l_t, MPC_RNDNN)
    else:
        # For exponent > 2, ll is ll ** ((n)/(n - 1.))
        map_normalisation_exponent = exponent / (exponent - 1.)
        mpc_pow_d(tmp_t, l_t, map_normalisation_exponent, MPC_RNDNN)
    mpc_mul(tmp2_t, b_t, tmp_t, MPC_RNDNN)
    mpc_ui_div(tmp_t, ui_one, tmp2_t, MPC_RNDNN)

    ret = mpc_t_to_Xrange(tmp_t)

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
    cdef unsigned long exponent = 2
    return perturbation_mandelbrotN_select_ball_method(
        seed_x, seed_y, seed_prec, seed_px, exponent, maxiter, M_divergence
    )

def perturbation_mandelbrotN_ball_method(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        char * seed_px,
        unsigned long exponent,
        long maxiter,
        double M_divergence
        ):
    return perturbation_mandelbrotN_select_ball_method(
        seed_x, seed_y, seed_prec, seed_px, exponent, maxiter, M_divergence
    )

def perturbation_mandelbrotN_select_ball_method(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        char * seed_px,
        unsigned long exponent,
        long maxiter,
        double M_divergence
):
    """
    Cycle order estimation by ball-method (iterations of a ball untils it
    contains 0.)

    Note: this implementation uses the local differential dzdc ; see
    perturbation_mandelbrot_ball_method_legacy for an implementation based
    on ball arithmetic 

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
        The found order (or -1 if no order is found)
    """
    cdef:
        int cmp = 0
        long ret = -1
        long i = 0
        unsigned long int ui_one = 1

        mpc_t c_t, z_t, dzndc_t, tmp_t, r_t
        mpfr_t ar_t, x_t, y_t, pix_t, inv_pix_t
        
        iterMn_zn_t iter_func
        iterMn_dzndz_t iter_deriv

    iter_func = select_Mnfunc(exponent)
    iter_deriv = select_Mn_deriv_func(exponent)

    mpc_init2(c_t, seed_prec)
    mpc_init2(z_t, seed_prec)
    mpc_init2(dzndc_t, seed_prec)
    mpc_init2(tmp_t, seed_prec)
    mpc_init2(r_t, seed_prec)

    mpfr_init2(ar_t, seed_prec)
    mpfr_init2(x_t, seed_prec)
    mpfr_init2(y_t, seed_prec)
    mpfr_init2(pix_t, seed_prec)
    mpfr_init2(inv_pix_t, seed_prec)


    # from char: set value of c
    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(y_t, seed_y, 10, MPFR_RNDN)
    mpc_set_fr_fr(c_t, x_t, y_t, MPC_RNDNN)

    # from char: set value of pix_t, inv_pix_t
    mpfr_set_str(pix_t, seed_px, 10, MPFR_RNDN)
    mpfr_ui_div(inv_pix_t, ui_one, pix_t, MPFR_RNDN)

    # set z = 0, dz = 0
    mpc_set_si_si(z_t, 0, 0, MPC_RNDNN)
    mpc_set_si_si(dzndc_t, 0, 0, MPC_RNDNN)

    # Can't harm, but is it needed ...
    mpc_set_si_si(tmp_t, 0, 0, MPC_RNDNN)
    mpc_set_si_si(r_t, 0, 0, MPC_RNDNN)

    for i in range(1, maxiter + 1):
        
        iter_deriv(exponent, 0, z_t, dzndc_t, tmp_t)
        iter_func(exponent, z_t, c_t, tmp_t)

        # r_t <- z_t / dzndc_t
        mpc_div(r_t, z_t, dzndc_t, MPC_RNDNN)
        mpc_mul_fr(r_t, r_t, inv_pix_t, MPC_RNDNN)

        # C _Complex type assignment to numpy complex128 array is not
        # straightforward, using 2 float64 components
        x = mpfr_get_d(mpc_realref(z_t), MPFR_RNDN)
        y = mpfr_get_d(mpc_imagref(z_t), MPFR_RNDN)

        if hypot(x, y) > M_divergence: # escaping
            ret = -1
            break

        xprint = mpfr_get_d(mpc_realref(dzndc_t), MPFR_RNDN)
        yprint = mpfr_get_d(mpc_imagref(dzndc_t), MPFR_RNDN)

        # Or did we find a cycle |r_t| < 1
        mpc_abs(ar_t, r_t, MPFR_RNDN)
        cmp = mpfr_cmp_d(ar_t, 1.)
        
        ar_print = mpfr_get_d(ar_t, MPFR_RNDN)
        
        # Return a positive value if op1 > op2, zero if op1 = op2, and a
        # negative value if op1 < op2.
        if cmp < 0.:
            ret = i
            break

    mpc_clear(c_t)
    mpc_clear(z_t)
    mpc_clear(dzndc_t)
    mpc_clear(tmp_t)
    mpc_clear(r_t)

    mpfr_clear(x_t)
    mpfr_clear(y_t)
    mpfr_clear(ar_t)
    mpfr_clear(pix_t)
    mpfr_clear(inv_pix_t)
    
    return ret


def perturbation_mandelbrot_ball_method_legacy(
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
    char *seed_eps_cv,
    char *seed_eps_valid
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
                        # h = h * zr : unwanted period we need to divide by
                        h *= zr
                        # h = zr_i1 * zr_i2 * ... * zr_in  / (dh / h) = SUM(dzr_i / zr_i)
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
                # dh += dzrdc / zr - Note dh is in reality dhdc/h
                mpc_div(tmp_t1, dzrdc_t, zr_t, MPC_RNDNN)
                mpc_add(tmp_t2, tmp_t1, dh_t, MPC_RNDNN)
                mpc_swap(tmp_t2, dh_t)

        # f = zr / h
        mpc_div(f_t, zr_t, h_t, MPC_RNDNN)

        # df = (dzrdc - zr * (dh / h)) / h
        mpc_mul(tmp_t1, zr_t, dh_t, MPC_RNDNN)
        mpc_sub(tmp_t2, dzrdc_t, tmp_t1, MPC_RNDNN)
        mpc_div(df_t, tmp_t2, h_t, MPC_RNDNN)

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
            # Sanity check that c_t is not a 'red herring' - zr_t shall be null
            # with 'enough' precision - depend on local pixel size
            mpc_abs(abs_diff, zr_t, MPFR_RNDN)
            mpfr_set_str(eps_t, seed_eps_valid, 10, MPFR_RNDN)
            newton_cv  = (mpfr_greaterequal_p(eps_t, abs_diff) != 0)
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

    if newton_cv:
        return newton_cv, gmpy_to_mpmath_mpc(gmpy_mpc, seed_prec)
    return False, mpmath.mpc("nan", "nan")


def perturbation_mandelbrot_find_any_nucleus(
    char *seed_x,
    char *seed_y,
    long seed_prec,
    long order,
    long max_newton,
    char *seed_eps_cv,
    char *seed_eps_valid
):
    """
    Run Newton search to find z0 so that f^n(z0) == 0 (n being the order input)
    Applied to Mandelbrot M2
    """
    cdef unsigned long exponent = 2
    return perturbation_mandelbrotN_select_find_any_nucleus(
        seed_x, seed_y, seed_prec, exponent, order, max_newton, seed_eps_cv,
        seed_eps_valid
    )


def perturbation_mandelbrotN_find_any_nucleus(
    char *seed_x,
    char *seed_y,
    long seed_prec,
    unsigned long exponent,
    long order,
    long max_newton,
    char *seed_eps_cv,
    char *seed_eps_valid
):
    """
    Run Newton search to find z0 so that f^n(z0) == 0 (n being the order input)
    Applied to Mandelbrot Mn
    """
    return perturbation_mandelbrotN_select_find_any_nucleus(
        seed_x, seed_y, seed_prec, exponent, order, max_newton, seed_eps_cv,
        seed_eps_valid
    )
    

def perturbation_mandelbrotN_select_find_any_nucleus(
    char *seed_x,
    char *seed_y,
    long seed_prec,
    unsigned long exponent,
    long order,
    long max_newton,
    char *seed_eps_cv,
    char *seed_eps_valid
):
    """
    Run Newton search to find z0 so that f^n(z0) == 0 (n being the order input)

    This implementation does not includes a "divide by undesired roots"
    technique, so divisors of n are considered.

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
                zr = mpmath.mp.zero
                dzrdc = mpmath.mp.zero
                for i in range(1, order + 1):
                    dzrdc = 2. * dzrdc * zr + 1.
                    zr = zr * zr + c_loop
    
                cc = c_loop - zr / dzrdc
                newton_cv = mpmath.almosteq(cc, c_loop) 
                c_loop = cc
                if newton_cv and (i_newton > 0):
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
        mpc_t tmp_t1

        mpfr_t x_t
        mpfr_t y_t
        mpfr_t abs_diff
        mpfr_t eps_t
        
        iterMn_zn_t iter_func
        iterMn_dzndz_t iter_deriv

    iter_func = select_Mnfunc(exponent)
    iter_deriv = select_Mn_deriv_func(exponent)

    mpc_init2(c_t, seed_prec)
    mpc_init2(zr_t, seed_prec)
    mpc_init2(dzrdc_t, seed_prec)
    mpc_init2(tmp_t1, seed_prec)

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
        
        # Newton descent
        for i in range(1, order + 1):
            # dzrdc = 2. * dzrdc * zr + 1.
            iter_deriv(exponent, 0, zr_t, dzrdc_t, tmp_t1)

            # zr = zr * zr + c_loop
            iter_func(exponent, zr_t, c_t, tmp_t1)

        # cc = c_loop - zr / dzrdc
        mpc_div(tmp_t1, zr_t, dzrdc_t, MPC_RNDNN)
        mpc_sub(c_t, c_t, tmp_t1, MPC_RNDNN)

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
            # Sanity check that c_t is not a 'red herring' - zr_t shall be null
            # with 'enough' precision - depend on local pixel size
            mpc_abs(abs_diff, zr_t, MPFR_RNDN)
            mpfr_set_str(eps_t, seed_eps_valid, 10, MPFR_RNDN)
            newton_cv  = (mpfr_greaterequal_p(eps_t, abs_diff) != 0)
            break

    gmpy_mpc = GMPy_MPC_From_mpc(c_t)

    mpc_clear(c_t)
    mpc_clear(zr_t)
    mpc_clear(dzrdc_t)
    mpc_clear(tmp_t1)

    mpfr_clear(x_t)
    mpfr_clear(y_t)
    mpfr_clear(abs_diff)
    mpfr_clear(eps_t)

    if newton_cv:
        return newton_cv, gmpy_to_mpmath_mpc(gmpy_mpc, seed_prec)
    return False, mpmath.mpc("nan", "nan")

#==============================================================================
#==============================================================================
# The Burning ship & al !
# This implementation is largely following the following paper:
#
# At the Helm of the Burning Ship, Claude Heiland-Allen, 2019
# Proceedings of EVA London 2019 (EVA 2019)
# http://dx.doi.org/10.14236/ewic/EVA2019.74
    
# These constants shall match the enumeration define in Burning_Ship
# implementation: BS_flavor_enum
cdef int BURNING_SHIP = 1
cdef int PERPENDICULAR_BURNING_SHIP = 2
cdef int SHARK_FIN = 3

ctypedef void (*iter_func_t)(
    mpfr_t, mpfr_t, mpfr_t, mpfr_t, mpfr_t, mpfr_t, mpfr_t
)

ctypedef void (*iter_hessian_t)(
    int, mpfr_t, mpfr_t, mpfr_t, mpfr_t, mpfr_t, mpfr_t, mpfr_t, mpfr_t,
    mpfr_t, mpfr_t, mpfr_t, mpfr_t, mpfr_t
)

cdef void iter_BS(
    mpfr_t xn_t, mpfr_t yn_t, mpfr_t a_t, mpfr_t b_t,
    mpfr_t xsq_t, mpfr_t ysq_t, mpfr_t xy_t, 
):
    # One standard burning ship iteration
    # X <- X**2 - Y**2 + A
    # Y <- 2 * abs(X * Y) - B
    mpfr_sqr(xsq_t, xn_t, MPFR_RNDN)
    mpfr_sqr(ysq_t, yn_t, MPFR_RNDN)
    mpfr_mul(xy_t, xn_t, yn_t, MPFR_RNDN)

    mpfr_sub(xn_t, xsq_t, ysq_t, MPFR_RNDN)
    mpfr_add(xn_t, xn_t, a_t, MPFR_RNDN)

    mpfr_abs(xy_t, xy_t, MPFR_RNDN)
    mpfr_mul_si(xy_t, xy_t, 2, MPFR_RNDN)
    mpfr_sub(yn_t, xy_t, b_t, MPFR_RNDN)
    return

cdef void iter_pBS(
    mpfr_t xn_t, mpfr_t yn_t, mpfr_t a_t, mpfr_t b_t,
    mpfr_t xsq_t, mpfr_t ysq_t, mpfr_t xy_t, 
):
    # One standard perpendicular burning ship iteration
    # X <- X**2 - Y**2 + A
    # Y <- 2 * X * abs(Y) - B   # 2 * abs(X * Y) - B
    mpfr_sqr(xsq_t, xn_t, MPFR_RNDN)
    mpfr_sqr(ysq_t, yn_t, MPFR_RNDN)

    mpfr_abs(xy_t, yn_t, MPFR_RNDN)
    mpfr_mul(xy_t, xn_t, xy_t, MPFR_RNDN)
    mpfr_mul_si(xy_t, xy_t, 2, MPFR_RNDN)

    mpfr_sub(xn_t, xsq_t, ysq_t, MPFR_RNDN)
    mpfr_add(xn_t, xn_t, a_t, MPFR_RNDN)

    mpfr_sub(yn_t, xy_t, b_t, MPFR_RNDN)
    return

cdef void iter_sharkfin(
    mpfr_t xn_t, mpfr_t yn_t, mpfr_t a_t, mpfr_t b_t,
    mpfr_t xsq_t, mpfr_t ysq_t, mpfr_t xy_t, 
):
    # One standard shark fin iteration
    # X <- X**2 - Y |Y| + A
    # Y <- 2 * X * Y - B
    mpfr_sqr(xsq_t, xn_t, MPFR_RNDN)
    mpfr_abs(ysq_t, yn_t, MPFR_RNDN)
    mpfr_mul(ysq_t, ysq_t, yn_t, MPFR_RNDN)

    mpfr_mul(xy_t, xn_t, yn_t, MPFR_RNDN)
    mpfr_mul_si(xy_t, xy_t, 2, MPFR_RNDN)

    mpfr_sub(xn_t, xsq_t, ysq_t, MPFR_RNDN)
    mpfr_add(xn_t, xn_t, a_t, MPFR_RNDN)

    mpfr_sub(yn_t, xy_t, b_t, MPFR_RNDN)
    return

cdef void iter_J_BS(
    int var_ab_xy,
    mpfr_t xn_t, mpfr_t yn_t,
    mpfr_t dxndx_t, mpfr_t dxndy_t,mpfr_t dyndx_t, mpfr_t dyndy_t,
    mpfr_t abs_xn, mpfr_t abs_yn, mpfr_t tmp_xx, mpfr_t tmp_xy, 
    mpfr_t tmp_yx, mpfr_t tmp_yy, mpfr_t _tmp
):
    # if var_ab_xy == 0 : we compute the derivatives wrt a, b / c = a + i b
    # if var_ab_xy == 1 : we compute the derivatives wrt x, y / z1 = x + i y
    cdef int sgn_xn, sgn_yn
    # One Jacobian burning ship iteration
    # dxndx <- 2 * (xn * dxndx - yn * dyndx) [+ 1]
    # dxndy <- 2 * (xn * dxndy - yn * dyndy)
    # dyndx <- 2 * (abs_xn * sgn_yn * dyndx + sgn_xn * dxndx * abs_yn)
    # dyndy <- 2 * (abs_xn * sgn_yn * dyndy + sgn_xn * dxndy * abs_yn) [- 1]
    mpfr_abs(abs_xn, xn_t, MPFR_RNDN)
    mpfr_abs(abs_yn, yn_t, MPFR_RNDN)
    sgn_xn = sign(xn_t)
    sgn_yn = sign(yn_t)

    # dxndx <- 2 * (xn * dxndx - yn * dyndx) [+ 1]
    mpfr_mul(tmp_xx, xn_t, dxndx_t, MPFR_RNDN)
    mpfr_mul(_tmp, yn_t, dyndx_t, MPFR_RNDN)
    mpfr_sub(tmp_xx, tmp_xx, _tmp, MPFR_RNDN)

    # dxndy <- 2 * (xn * dxndy - yn * dyndy)
    mpfr_mul(tmp_xy, xn_t, dxndy_t, MPFR_RNDN)
    mpfr_mul(_tmp, yn_t, dyndy_t, MPFR_RNDN)
    mpfr_sub(tmp_xy, tmp_xy, _tmp, MPFR_RNDN)

    # dyndx <- 2 * (abs_xn * sgn_yn * dyndx + sgn_xn * dxndx * abs_yn)
    mpfr_mul(tmp_yx, abs_xn, dyndx_t, MPFR_RNDN)
    mpfr_mul_si(tmp_yx, tmp_yx, sgn_yn, MPFR_RNDN)
    mpfr_mul(_tmp, abs_yn, dxndx_t, MPFR_RNDN)
    mpfr_mul_si(_tmp, _tmp, sgn_xn, MPFR_RNDN)
    mpfr_add(tmp_yx, tmp_yx, _tmp, MPFR_RNDN)

    # dyndy <- 2 * (abs_xn * sgn_yn * dyndy + sgn_xn * dxndy * abs_yn) [-1]
    mpfr_mul(tmp_yy, abs_xn, dyndy_t, MPFR_RNDN)
    mpfr_mul_si(tmp_yy, tmp_yy, sgn_yn, MPFR_RNDN)
    mpfr_mul(_tmp, abs_yn, dxndy_t, MPFR_RNDN)
    mpfr_mul_si(_tmp, _tmp, sgn_xn, MPFR_RNDN)
    mpfr_add(tmp_yy, tmp_yy, _tmp, MPFR_RNDN)

    # push the result
    mpfr_mul_si(dxndx_t, tmp_xx, 2, MPFR_RNDN)
    mpfr_mul_si(dxndy_t, tmp_xy, 2, MPFR_RNDN)
    mpfr_mul_si(dyndx_t, tmp_yx, 2, MPFR_RNDN)
    mpfr_mul_si(dyndy_t, tmp_yy, 2, MPFR_RNDN)

    if var_ab_xy == 0:
        mpfr_add_si(dxndx_t, dxndx_t, 1, MPFR_RNDN)
        mpfr_add_si(dyndy_t, dyndy_t, -1, MPFR_RNDN)

    return

cdef void iter_J_pBS(
    int var_ab_xy,
    mpfr_t xn_t, mpfr_t yn_t,
    mpfr_t dxndx_t, mpfr_t dxndy_t,mpfr_t dyndx_t, mpfr_t dyndy_t,
    mpfr_t abs_xn, mpfr_t abs_yn, mpfr_t tmp_xx, mpfr_t tmp_xy, 
    mpfr_t tmp_yx, mpfr_t tmp_yy, mpfr_t _tmp
):
    # if var_ab_xy == 0 : we compute the derivatives wrt a, b / c = a + i b
    # if var_ab_xy == 1 : we compute the derivatives wrt x, y / z1 = x + i y
    cdef int sgn_xn, sgn_yn
    # One Jacobian burning ship iteration
    # dxndx <- 2 * (xn * dxndx - yn * dyndx) [+ 1]
    # dxndy <- 2 * (xn * dxndy - yn * dyndy)
    # dyndx <- 2 * (xn * sgn_yn * dyndx + dxndx * abs_yn)
    # dyndy <- 2 * (xn * sgn_yn * dyndy + dxndy * abs_yn) [- 1]
    # mpfr_abs(abs_xn, xn_t, MPFR_RNDN)
    mpfr_abs(abs_yn, yn_t, MPFR_RNDN)
    sgn_yn = sign(yn_t)

    # dxndx <- 2 * (xn * dxndx - yn * dyndx) [+ 1]
    mpfr_mul(tmp_xx, xn_t, dxndx_t, MPFR_RNDN)
    mpfr_mul(_tmp, yn_t, dyndx_t, MPFR_RNDN)
    mpfr_sub(tmp_xx, tmp_xx, _tmp, MPFR_RNDN)

    # dxndy <- 2 * (xn * dxndy - yn * dyndy)
    mpfr_mul(tmp_xy, xn_t, dxndy_t, MPFR_RNDN)
    mpfr_mul(_tmp, yn_t, dyndy_t, MPFR_RNDN)
    mpfr_sub(tmp_xy, tmp_xy, _tmp, MPFR_RNDN)

    # dyndx <- 2 * (abs_xn * sgn_yn * dyndx +  dxndx * abs_yn)
    mpfr_mul(tmp_yx, xn_t, dyndx_t, MPFR_RNDN)
    mpfr_mul_si(tmp_yx, tmp_yx, sgn_yn, MPFR_RNDN)
    mpfr_mul(_tmp, abs_yn, dxndx_t, MPFR_RNDN)
    mpfr_add(tmp_yx, tmp_yx, _tmp, MPFR_RNDN)

    # dyndy <- 2 * (xn * sgn_yn * dyndy + dxndy * abs_yn) [- 1]
    mpfr_mul(tmp_yy, xn_t, dyndy_t, MPFR_RNDN)
    mpfr_mul_si(tmp_yy, tmp_yy, sgn_yn, MPFR_RNDN)
    mpfr_mul(_tmp, abs_yn, dxndy_t, MPFR_RNDN)
    mpfr_add(tmp_yy, tmp_yy, _tmp, MPFR_RNDN)

    # push the result
    mpfr_mul_si(dxndx_t, tmp_xx, 2, MPFR_RNDN)
    mpfr_mul_si(dxndy_t, tmp_xy, 2, MPFR_RNDN)
    mpfr_mul_si(dyndx_t, tmp_yx, 2, MPFR_RNDN)
    mpfr_mul_si(dyndy_t, tmp_yy, 2, MPFR_RNDN)

    if var_ab_xy == 0:
        mpfr_add_si(dxndx_t, dxndx_t, 1, MPFR_RNDN)
        mpfr_add_si(dyndy_t, dyndy_t, -1, MPFR_RNDN)

    return

cdef void iter_J_sharkfin(
    int var_ab_xy,
    mpfr_t xn_t, mpfr_t yn_t,
    mpfr_t dxndx_t, mpfr_t dxndy_t,mpfr_t dyndx_t, mpfr_t dyndy_t,
    mpfr_t abs_xn, mpfr_t abs_yn, mpfr_t tmp_xx, mpfr_t tmp_xy, 
    mpfr_t tmp_yx, mpfr_t tmp_yy, mpfr_t _tmp
):
    # if var_ab_xy == 0 : we compute the derivatives wrt a, b / c = a + i b
    # if var_ab_xy == 1 : we compute the derivatives wrt x, y / z1 = x + i y
    
    # X <- X**2 - Y |Y| + A
    # Y <- 2 * X * Y - B

    # One Jacobian burning ship iteration
    # dxndx <- 2 * (xn * dxndx - |yn| * dyndx) [+ 1]
    # dxndy <- 2 * (xn * dxndy - |yn| * dyndy)
    # dyndx <- 2 * (xn * dyndx + dxndx * yn)
    # dyndy <- 2 * (xn * dyndy + dxndy * yn) [- 1]
    mpfr_abs(abs_yn, yn_t, MPFR_RNDN)

    # dxndx <- 2 * (xn * dxndx - |yn| * dyndx) [+ 1]
    mpfr_mul(tmp_xx, xn_t, dxndx_t, MPFR_RNDN)
    mpfr_mul(_tmp, abs_yn, dyndx_t, MPFR_RNDN)
    mpfr_sub(tmp_xx, tmp_xx, _tmp, MPFR_RNDN)

    # dxndy <- 2 * (xn * dxndy - |yn| * dyndy)
    mpfr_mul(tmp_xy, xn_t, dxndy_t, MPFR_RNDN)
    mpfr_mul(_tmp, abs_yn, dyndy_t, MPFR_RNDN)
    mpfr_sub(tmp_xy, tmp_xy, _tmp, MPFR_RNDN)

    # dyndx <- 2 * (xn * dyndx +  dxndx * yn)
    mpfr_mul(tmp_yx, xn_t, dyndx_t, MPFR_RNDN)
    mpfr_mul(_tmp, yn_t, dxndx_t, MPFR_RNDN)
    mpfr_add(tmp_yx, tmp_yx, _tmp, MPFR_RNDN)

    # dyndy <- 2 * (xn * dyndy + dxndy * yn) [-1]
    mpfr_mul(tmp_yy, xn_t, dyndy_t, MPFR_RNDN)
    mpfr_mul(_tmp, yn_t, dxndy_t, MPFR_RNDN)
    mpfr_add(tmp_yy, tmp_yy, _tmp, MPFR_RNDN)

    # push the result
    mpfr_mul_si(dxndx_t, tmp_xx, 2, MPFR_RNDN)
    mpfr_mul_si(dxndy_t, tmp_xy, 2, MPFR_RNDN)
    mpfr_mul_si(dyndx_t, tmp_yx, 2, MPFR_RNDN)
    mpfr_mul_si(dyndy_t, tmp_yy, 2, MPFR_RNDN)

    if var_ab_xy == 0:
        mpfr_add_si(dxndx_t, dxndx_t, 1, MPFR_RNDN)
        mpfr_add_si(dyndy_t, dyndy_t, -1, MPFR_RNDN)

    return


cdef iter_func_t select_func(const int kind):
    cdef iter_func_t iter_func
    if kind == BURNING_SHIP:
        iter_func = &iter_BS
    elif kind == PERPENDICULAR_BURNING_SHIP:
        iter_func = &iter_pBS
    elif kind == SHARK_FIN:
        iter_func = &iter_sharkfin
    else:
        raise NotImplementedError(kind)
    return iter_func

cdef iter_hessian_t select_hessian(const int kind):
    cdef iter_hessian_t iter_hessian
    if kind == BURNING_SHIP:
        iter_hessian = &iter_J_BS
    elif kind == PERPENDICULAR_BURNING_SHIP:
        iter_hessian = &iter_J_pBS
    elif kind == SHARK_FIN:
        iter_hessian = &iter_J_sharkfin
    else:
        raise NotImplementedError(kind)
    return iter_hessian


cdef inline int sign(mpfr_t op):
    # sign function which returns always 1 or -1 to avoid singular cases 
    ret = mpfr_sgn(op)
    if ret >= 0:
        return 1
    return -1


cdef inline void det(
    mpfr_t delta, mpfr_t a, mpfr_t b, mpfr_t c, mpfr_t d, mpfr_t _tmp
):
    # return the determinant of the 2 x 2 matrix:
    #  [a b]
    #  [c d]
    # delta = ad - cb
    mpfr_mul(delta, a, d, MPFR_RNDN)
    mpfr_mul(_tmp, c, b, MPFR_RNDN)
    mpfr_sub(delta, delta, _tmp, MPFR_RNDN)
    return


cdef void matsolve(
    mpfr_t x_res, mpfr_t y_res,
    mpfr_t a, mpfr_t b, mpfr_t c, mpfr_t d,
    mpfr_t e, mpfr_t f,
    mpfr_t delta, mpfr_t _tmp
):
    # return the (x, y) solution of the 2 x 2 linear problem:
    #  [a b] x [x] = [e] 
    #  [c d]   [y]   [f]

    cdef:
        unsigned long int ui_one = 1

    det(delta, a, b, c, d, _tmp)
    mpfr_ui_div(delta, ui_one, delta, MPFR_RNDN)
    #  [x] = delta x [ d -b] x [e]
    #  [y]           [-c  a]   [f]
    mpfr_mul(x_res, d, e, MPFR_RNDN)
    mpfr_mul(_tmp, b, f, MPFR_RNDN)
    mpfr_sub(x_res, x_res, _tmp, MPFR_RNDN)
    mpfr_mul(x_res, x_res, delta, MPFR_RNDN)

    mpfr_mul(y_res, a, f, MPFR_RNDN)
    mpfr_mul(_tmp, c, e, MPFR_RNDN)
    mpfr_sub(y_res, y_res, _tmp, MPFR_RNDN)
    mpfr_mul(y_res, y_res, delta, MPFR_RNDN)
    return


#def perturbation_BS_FP_loop(
#        np.ndarray[DTYPE_FLOAT_t, ndim=1] orbit,
#        bint need_Xrange,
#        long max_iter,
#        double M,
#        char * seed_x,
#        char * seed_y,
#        long seed_prec
#):
#    return perturbation_nonholomorphic_FP_loop(
#        orbit,
#        need_Xrange,
#        max_iter,
#        M,
#        seed_x,
#        seed_y,
#        seed_prec,
#        kind=BURNING_SHIP
#    )
#
#def perturbation_perpendicular_BS_FP_loop(
#        np.ndarray[DTYPE_FLOAT_t, ndim=1] orbit,
#        bint need_Xrange,
#        long max_iter,
#        double M,
#        char * seed_x,
#        char * seed_y,
#        long seed_prec
#):
#    return perturbation_nonholomorphic_FP_loop(
#        orbit,
#        need_Xrange,
#        max_iter,
#        M,
#        seed_x,
#        seed_y,
#        seed_prec,
#        kind=PERPENDICULAR_BURNING_SHIP
#    )
#
#def perturbation_shark_fin_FP_loop(
#        np.ndarray[DTYPE_FLOAT_t, ndim=1] orbit,
#        bint need_Xrange,
#        long max_iter,
#        double M,
#        char * seed_x,
#        char * seed_y,
#        long seed_prec
#):
#    return perturbation_nonholomorphic_FP_loop(
#        orbit,
#        need_Xrange,
#        max_iter,
#        M,
#        seed_x,
#        seed_y,
#        seed_prec,
#        kind=SHARK_FIN
#    )

def perturbation_nonholomorphic_FP_loop(
        np.ndarray[DTYPE_FLOAT_t, ndim=1] orbit,
        bint need_Xrange,
        long max_iter,
        double M,
        char * seed_x,
        char * seed_y,
        long seed_prec,
        const int kind
):
    """
    Full precision orbit for Burning ship
    
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
    seed_prec: int 
        Precision used for the full precision calculation (in bits)
        Usually one should just use `mpmath.mp.prec`
    kind: int
        The int flavor for this Burning ship implementation 

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
        iter_func_t iter_func

        mpfr_t xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t

    assert orbit.dtype == DTYPE_FLOAT
    print(max_iter, max_len, max_iter * 2)
    assert (max_iter + 1) * 2 == max_len # (NP_orbit starts at critical point)
    
    iter_func = select_func(kind)

    orbit_Xrange_register = dict()
    orbit_partial_register = dict()

    mpfr_inits2(seed_prec, xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t, NULL)

    # a + i b = c
    mpfr_set_str(a_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(b_t, seed_y, 10, MPFR_RNDN)

    # For standard Mandelbrot the critical point is 0. - initializing
    # the FP z and the NP_orbit
    #### mpc_set_si_si(z_t, 0, 0, MPC_RNDNN)
    mpfr_set_si(xn_t, 0, MPFR_RNDN)
    mpfr_set_si(yn_t, 0, MPFR_RNDN)

    # Complex 0. :
    orbit[0] = 0.
    orbit[1] = 0.

    print_freq = max(10000, (int(max_iter / 100.) // 10000 + 1) * 10000)
    print("============================================")
    print("Mandelbrot iterations, full precision: ", seed_prec)
    print("Output every: ", print_freq, flush=True)

    for i in range(1, max_iter + 1):
        # X <- X**2 - Y**2 + A
        # Y <- 2 * abs(X * Y) - B
        iter_func(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t)

        # Store in orbit as double
        x = mpfr_get_d(xn_t, MPFR_RNDN)
        y = mpfr_get_d(yn_t, MPFR_RNDN)
        orbit[2 * i] = x
        orbit[2 * i + 1] = y

        # take the norm 
        abs_i = hypot(x, y)
        abs_xi = abs(x)
        abs_yi = abs(y)

        if abs_i > M: # escaping
            break

        # Handles the special cases where the orbit goes closer to the critical
        # axes than a standard double can handle.
        if need_Xrange and ((abs_xi < XR_TSHOLD) or (abs_yi < XR_TSHOLD)):
            orbit_Xrange_register[i] = (
                mpfr_t_to_Xrange(xn_t),
                mpfr_t_to_Xrange(yn_t),
            )

        # Hanldes the successive partials
        if abs_i <= curr_partial:
            # We need to check more precisely (due to the 'Xrange' cases)
            try:
                curr_index= next(reversed(orbit_partial_register.keys()))
                curr_xrx, curr_xry = orbit_partial_register[curr_index]
                curr_xr_abs2 = curr_xrx ** 2 + curr_xry ** 2
            except StopIteration:
                curr_xr_abs2 = fsx.Xrange_array([curr_partial]).abs2()

            candidate_partial_x = mpfr_t_to_Xrange(xn_t)
            candidate_partial_y = mpfr_t_to_Xrange(yn_t)
            candidate_partial_abs2 = (
                candidate_partial_x ** 2
                + candidate_partial_y ** 2
            )
            if candidate_partial_abs2 < curr_xr_abs2:
                orbit_partial_register[i] = (
                    candidate_partial_x, candidate_partial_y
                )
                curr_partial = np.sqrt(
                    candidate_partial_x ** 2 + candidate_partial_y ** 2
                ).to_standard()

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

    mpfr_clears(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t, NULL)

    return i, orbit_partial_register, orbit_Xrange_register


#def perturbation_BS_nucleus_size_estimate(
#        char * seed_x,
#        char * seed_y,
#        long seed_prec,
#        long order
#):
#
#    return perturbation_nonholomorphic_nucleus_size_estimate(
#        seed_x,
#        seed_y,
#        seed_prec,
#        order,
#        kind=BURNING_SHIP
#    )
#
#def perturbation_perpendicular_BS_nucleus_size_estimate(
#        char * seed_x,
#        char * seed_y,
#        long seed_prec,
#        long order
#):
#    return perturbation_nonholomorphic_nucleus_size_estimate(
#        seed_x,
#        seed_y,
#        seed_prec,
#        order,
#        kind=PERPENDICULAR_BURNING_SHIP
#    )
#
#def perturbation_shark_fin_nucleus_size_estimate(
#        char * seed_x,
#        char * seed_y,
#        long seed_prec,
#        long order
#):
#    return perturbation_nonholomorphic_nucleus_size_estimate(
#        seed_x,
#        seed_y,
#        seed_prec,
#        order,
#        kind=SHARK_FIN
#    )

def perturbation_nonholomorphic_nucleus_size_estimate(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        long order,
        const int kind
):
    """
    Hyperbolic component size estimate. Reference :
    https://mathr.co.uk/blog/2016-12-24_deriving_the_size_estimate.html
    https://fractalforums.org/fractal-mathematics-and-new-theories/28/miniset-and-embedded-julia-size-estimates/912/msg4815#msg4815
    
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
    kind: int
        The int flavor for this Burning ship implementation 

    Return:
    -------
    size, skew
    A size estimate of the nucleus described by the input parameters, and the
    local skew matrice
    """
    
    cdef:
        unsigned long int ui_one = 1
        long i = 0
        iter_func_t iter_func
        iter_hessian_t iter_hessian

        mpfr_t xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t
        mpfr_t dxndx_t, dxndy_t, dyndx_t, dyndy_t, delta_t
        mpfr_t abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy
        mpfr_t bn_xx_t, bn_xy_t, bn_yx_t, bn_yy_t
        mpfr_t _tmp

    iter_func = select_func(kind)
    iter_hessian = select_hessian(kind)

    # initialisation
    mpfr_inits2(seed_prec, xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t, NULL)
    mpfr_inits2(seed_prec, dxndx_t, dxndy_t, dyndx_t, dyndy_t, delta_t, NULL)
    mpfr_inits2(seed_prec, abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, NULL)
    mpfr_inits2(seed_prec, bn_xx_t, bn_xy_t, bn_yx_t, bn_yy_t, NULL)
    mpfr_init2(_tmp, seed_prec)

    # set value of a + i b = c
    mpfr_set_str(a_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(b_t, seed_y, 10, MPFR_RNDN)

    # set z = 0 l = 1 b = 1
    mpfr_set_si(xn_t, 0, MPFR_RNDN)
    mpfr_set_si(yn_t, 0, MPFR_RNDN)

    # L0 = 1 (identity matrix)
    mpfr_set_si(dxndx_t, 1, MPFR_RNDN)
    mpfr_set_si(dxndy_t, 0, MPFR_RNDN)
    mpfr_set_si(dyndx_t, 0, MPFR_RNDN)
    mpfr_set_si(dyndy_t, 1, MPFR_RNDN)

    # B0 = 1 (identity matrix)
    mpfr_set_si(bn_xx_t, 1, MPFR_RNDN)
    mpfr_set_si(bn_xy_t, 0, MPFR_RNDN)
    mpfr_set_si(bn_yx_t, 0, MPFR_RNDN)
    mpfr_set_si(bn_yy_t, 1, MPFR_RNDN)


    for i in range(1, order):
        # X <- X**2 - Y**2 + A
        # Y <- 2 * abs(X * Y) - B
        iter_func(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t)

        # Ln is the Jacobian
        # Ln = [dxndx  dxndy]
        #      [dyndx  dyndy]
        # dxndx <- 2 * (xn * dxndx - yn * dyndx)
        # dxndy <- 2 * (xn * dxndy - yn * dyndy)
        # dyndx <- 2 * abs_xn * sgn_yn * dyndx + 2 * sgn_xn * dxndx * abs_yn
        # dyndy <- 2 * abs_xn * sgn_yn * dyndy + 2 * sgn_xn * dxndy * abs_yn
        iter_hessian(
            1,
            xn_t, yn_t, dxndx_t, dxndy_t, dyndx_t, dyndy_t,
            abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, _tmp
        )

        # Bn is the accumulation of Ln-1
        # delta = 1/det(Ln)
        det(delta_t, dxndx_t, dxndy_t, dyndx_t, dyndy_t, _tmp)
        mpfr_ui_div(delta_t, ui_one, delta_t, MPFR_RNDN)
        # Now, by components
        # Ln^-1 = 1/delta * [ dyndy  -dxndy]
        #                   [-dyndx   dxndx]
        mpfr_mul(_tmp, delta_t, dyndy_t, MPFR_RNDN)
        mpfr_add(bn_xx_t, bn_xx_t, _tmp, MPFR_RNDN)

        mpfr_mul(_tmp, delta_t, dxndy_t, MPFR_RNDN)
        mpfr_sub(bn_xy_t, bn_xy_t, _tmp, MPFR_RNDN)

        mpfr_mul(_tmp, delta_t, dyndx_t, MPFR_RNDN)
        mpfr_sub(bn_yx_t, bn_yx_t, _tmp, MPFR_RNDN)

        mpfr_mul(_tmp, delta_t, dxndx_t, MPFR_RNDN)
        mpfr_add(bn_yy_t, bn_yy_t, _tmp, MPFR_RNDN)


    # S = beta * Lp ** 2
    # det(S) =  det(beta) * det(Lp) ** 2
    det(delta_t, dxndx_t, dxndy_t, dyndx_t, dyndy_t, _tmp)
    mpfr_sqr(delta_t, delta_t, MPFR_RNDN)
    det(tmp_xx, bn_xx_t, bn_xy_t, bn_yx_t, bn_yy_t, _tmp)
    mpfr_mul(delta_t, tmp_xx, delta_t, MPFR_RNDN)

    # size = 1 / sqrt(abs(det(S)))
    mpfr_abs(_tmp, delta_t, MPFR_RNDN)
    mpfr_rec_sqrt(delta_t, _tmp, MPFR_RNDN)
    size = mpfr_t_to_Xrange(delta_t)

    # Skew : SKEW = normalized(beta)^-1
    # However we ALSO need to change basis (because of the minus sign before b)
    # so M -> P^-1 M P where :
    #   P = [1  0]   M = SKEW  so in the end SKEW = [yy xy]
    #       [0 -1]                                  [yx xx]
    mpfr_rec_sqrt(_tmp, tmp_xx, MPFR_RNDN) # Here tmp_xx is det(beta)
    mpfr_mul(bn_xx_t, bn_xx_t, _tmp, MPFR_RNDN)
    mpfr_mul(bn_xy_t, bn_xy_t, _tmp, MPFR_RNDN)
    mpfr_mul(bn_yx_t, bn_yx_t, _tmp, MPFR_RNDN)
    mpfr_mul(bn_yy_t, bn_yy_t, _tmp, MPFR_RNDN)
    skew = np.array((
            (mpfr_get_d(bn_yy_t, MPFR_RNDN), mpfr_get_d(bn_xy_t, MPFR_RNDN)),
            (mpfr_get_d(bn_yx_t, MPFR_RNDN), mpfr_get_d(bn_xx_t, MPFR_RNDN))
        ),
        dtype=np.float64
    )

    mpfr_clears(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t, NULL)
    mpfr_clears(dxndx_t, dxndy_t, dyndx_t, dyndy_t, delta_t, NULL)
    mpfr_clears(abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, NULL)
    mpfr_clears(bn_xx_t, bn_xy_t, bn_yx_t, bn_yy_t, NULL)
    mpfr_clear(_tmp)

    return size, skew


#def perturbation_BS_skew_estimate(
#        char * seed_x,
#        char * seed_y,
#        long seed_prec,
#        long max_iter,
#        double M,
#):
#    """
#    Quick estimation of skew for areas without minis
#    """
#    return perturbation_nonholomorphic_skew_estimate(
#        seed_x,
#        seed_y,
#        seed_prec,
#        max_iter,
#        M,
#        kind=BURNING_SHIP
#    )
#
#def perturbation_perpendicular_BS_skew_estimate(
#        char * seed_x,
#        char * seed_y,
#        long seed_prec,
#        long max_iter,
#        double M,
#):
#    return perturbation_nonholomorphic_skew_estimate(
#        seed_x,
#        seed_y,
#        seed_prec,
#        max_iter,
#        M,
#        kind=PERPENDICULAR_BURNING_SHIP
#    )
#
#def perturbation_shark_fin_skew_estimate(
#        char * seed_x,
#        char * seed_y,
#        long seed_prec,
#        long max_iter,
#        double M,
#):
#    return perturbation_nonholomorphic_skew_estimate(
#        seed_x,
#        seed_y,
#        seed_prec,
#        max_iter,
#        M,
#        kind=SHARK_FIN
#    )

def perturbation_nonholomorphic_skew_estimate(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        long max_iter,
        double M,
        const int kind
):
    cdef:
        unsigned long int ui_one = 1
        long i = 0
        iter_func_t iter_func
        iter_hessian_t iter_hessian

        mpfr_t xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t
        mpfr_t dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t
        mpfr_t abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy
        mpfr_t _tmp

    iter_func = select_func(kind)
    iter_hessian = select_hessian(kind)

    # initialisation
    mpfr_inits2(seed_prec, xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t, NULL)
    mpfr_inits2(seed_prec, dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t, NULL)
    mpfr_inits2(seed_prec, abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, NULL)
    mpfr_init2(_tmp, seed_prec)

    # set value of a + i b = c
    mpfr_set_str(a_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(b_t, seed_y, 10, MPFR_RNDN)

    # x0 = y0 = J0 = 0
    mpfr_set_si(xn_t, 0, MPFR_RNDN)
    mpfr_set_si(yn_t, 0, MPFR_RNDN)
    mpfr_set_si(dxnda_t, 0, MPFR_RNDN)
    mpfr_set_si(dxndb_t, 0, MPFR_RNDN)
    mpfr_set_si(dynda_t, 0, MPFR_RNDN)
    mpfr_set_si(dyndb_t, 0, MPFR_RNDN)

    for i in range(max_iter):
        iter_hessian(
            0,
            xn_t, yn_t, dxnda_t, dxndb_t, dynda_t, dyndb_t,
            abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, _tmp
        )
        iter_func(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t)        
        
        # Store in orbit as double
        x = mpfr_get_d(xn_t, MPFR_RNDN)
        y = mpfr_get_d(yn_t, MPFR_RNDN)
        abs_i = hypot(x, y)

        if abs_i > M: # escaping
            break

    print("escaped at iteration:", i)

    det(delta_t, dxnda_t, dxndb_t, dynda_t, dyndb_t, _tmp)

    # size = 1 / sqrt(abs(det(S)))
    mpfr_abs(delta_t, delta_t, MPFR_RNDN)
    mpfr_rec_sqrt(delta_t, delta_t, MPFR_RNDN)

    mpfr_mul(dxnda_t, dxnda_t, delta_t, MPFR_RNDN)
    mpfr_mul(dxndb_t, dxndb_t, delta_t, MPFR_RNDN)
    mpfr_mul(dynda_t, dynda_t, delta_t, MPFR_RNDN)
    mpfr_mul(dyndb_t, dyndb_t, delta_t, MPFR_RNDN)
    skew = np.array((
            (mpfr_get_d(dyndb_t, MPFR_RNDN), -mpfr_get_d(dxndb_t, MPFR_RNDN)),
            (-mpfr_get_d(dynda_t, MPFR_RNDN), mpfr_get_d(dxnda_t, MPFR_RNDN))
        ),
        dtype=np.float64
    )

    mpfr_clears(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t, NULL)
    mpfr_clears(dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t, NULL)
    mpfr_clears(abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, NULL)
    mpfr_clear(_tmp)

    return skew


def perturbation_BS_ball_method(
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
    C-translation of the following python code for std Mandelbrot

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
    return perturbation_nonholomorphic_ball_method(
        seed_x,
        seed_y,
        seed_prec,
        seed_px,
        maxiter,
        M_divergence,
        kind=BURNING_SHIP
    )

def perturbation_perpendicular_BS_ball_method(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        char * seed_px,
        long maxiter,
        double M_divergence
):
    return perturbation_nonholomorphic_ball_method(
        seed_x,
        seed_y,
        seed_prec,
        seed_px,
        maxiter,
        M_divergence,
        kind=PERPENDICULAR_BURNING_SHIP
    )

def perturbation_shark_fin_ball_method(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        char * seed_px,
        long maxiter,
        double M_divergence
):
    return perturbation_nonholomorphic_ball_method(
        seed_x,
        seed_y,
        seed_prec,
        seed_px,
        maxiter,
        M_divergence,
        kind=SHARK_FIN
    )

def perturbation_nonholomorphic_ball_method(
        char * seed_x,
        char * seed_y,
        long seed_prec,
        char * seed_px,
        long maxiter,
        double M_divergence,
        const int kind
):
    cdef:
        int cmp = 0
        unsigned long int ui_one = 1
        long ret = -1
        long i = 0
        double x, y, rx, ry
        iter_func_t iter_func
        iter_hessian_t iter_hessian

        mpfr_t xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t
        mpfr_t dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t
        mpfr_t abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy
        mpfr_t a, b, c, d
        mpfr_t rx_t, ry_t, pix_t, inv_pix_t
        mpfr_t _tmp

    iter_func = select_func(kind)
    iter_hessian = select_hessian(kind)

    # initialisation
    mpfr_inits2(seed_prec, xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t, NULL)
    mpfr_inits2(seed_prec, dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t, NULL)
    mpfr_inits2(seed_prec, abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, NULL)
    # mpfr_inits2(seed_prec, a, b, c, d, NULL)
    mpfr_inits2(seed_prec, rx_t, ry_t, pix_t, inv_pix_t, NULL)
    mpfr_init2(_tmp, seed_prec)

    # set value of a + i b = c
    mpfr_set_str(a_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(b_t, seed_y, 10, MPFR_RNDN)

    # set z = 0
    mpfr_set_si(xn_t, 0, MPFR_RNDN)
    mpfr_set_si(yn_t, 0, MPFR_RNDN)

    # L0 = 0
    mpfr_set_si(dxnda_t, 0, MPFR_RNDN)
    mpfr_set_si(dxndb_t, 0, MPFR_RNDN)
    mpfr_set_si(dynda_t, 0, MPFR_RNDN)
    mpfr_set_si(dyndb_t, 0, MPFR_RNDN)

    mpfr_set_str(pix_t, seed_px, 10, MPFR_RNDN)
    mpfr_ui_div(inv_pix_t, ui_one, pix_t, MPFR_RNDN)

    
    for i in range(1, maxiter + 1):

        iter_hessian(
            0,
            xn_t, yn_t, dxnda_t, dxndb_t, dynda_t, dyndb_t,
            abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, _tmp
        )
        iter_func(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t)
        

        # [rX] = J^-1 x [X] 
        # [rY]          [Y]
        matsolve(
            rx_t, ry_t,
            dxnda_t, dxndb_t, dynda_t, dyndb_t,
            xn_t, yn_t,
            delta_t, _tmp
        )
        mpfr_mul(rx_t, rx_t, inv_pix_t, MPFR_RNDN)
        mpfr_mul(ry_t, ry_t, inv_pix_t, MPFR_RNDN)

        # if |xn + i yn| > M_divergence:
        x = mpfr_get_d(xn_t, MPFR_RNDN)
        y = mpfr_get_d(yn_t, MPFR_RNDN)
        # print("x y", x, y)
        if hypot(x, y) > M_divergence: # escaping
            ret = -1
            break

        # Or did we find a cycle |(rx, ry)| < 1
        mpfr_abs(_tmp, rx_t, MPFR_RNDN)
        cmpx = mpfr_cmp_d(_tmp, 1.)
        mpfr_abs(_tmp, ry_t, MPFR_RNDN)
        cmpy = mpfr_cmp_d(_tmp, 1.)
        # Return a positive value if op1 > op2, zero if op1 = op2, and a
        # negative value if op1 < op2.
        if (cmpx < 0) and (cmpy < 0):
            rx = mpfr_get_d(rx_t, MPFR_RNDN)
            ry = mpfr_get_d(ry_t, MPFR_RNDN)
            if hypot(rx, ry) < 1.:
                ret = i
                break

    mpfr_clears(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t, NULL)
    mpfr_clears(dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t, NULL)
    mpfr_clears(abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, NULL)
    # mpfr_clears(a, b, c, d, NULL)
    mpfr_clears(rx_t, ry_t, pix_t, inv_pix_t, NULL)
    mpfr_clear(_tmp)

    return ret

def perturbation_BS_find_any_nucleus(
    char *seed_x,
    char *seed_y,
    long seed_prec,
    long order,
    long max_newton,
    char *seed_eps_cv,
    char *seed_eps_valid
):
    """
    Run Newton search to find z0 so that f^n(z0) == 0 (n being the order input)

    This implementation does not includes a "divide by undesired roots"
    technique, so divisors of n are considered.

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
    C-translation of the following python code - for standard Mandelbrot
    For burning ship we use a full jacobian matrix instead of complex
    derivative

        def find_nucleus(c, order, max_newton=None, eps_cv=None):
    
            c_loop = c
    
            for i_newton in range(max_newton): 
                zr = mpmath.mp.zero
                dzrdc = mpmath.mp.zero
                for i in range(1, order + 1):
                    dzrdc = 2. * dzrdc * zr + 1.
                    zr = zr * zr + c_loop
    
                cc = c_loop - zr / dzrdc
                newton_cv = mpmath.almosteq(cc, c_loop) 
                c_loop = cc
                if newton_cv and (i_newton > 0):
                    break

            return newton_cv, c_loop
        
    See also:
    ---------
    https://mathr.co.uk/blog/2018-11-17_newtons_method_for_periodic_points.html
    """
    return perturbation_nonholomorphic_find_any_nucleus(
        seed_x,
        seed_y,
        seed_prec,
        order,
        max_newton,
        seed_eps_cv,
        seed_eps_valid,
        kind=BURNING_SHIP
    )

def perturbation_perpendicular_BS_find_any_nucleus(
    char *seed_x,
    char *seed_y,
    long seed_prec,
    long order,
    long max_newton,
    char *seed_eps_cv,
    char *seed_eps_valid
):
    return perturbation_nonholomorphic_find_any_nucleus(
        seed_x,
        seed_y,
        seed_prec,
        order,
        max_newton,
        seed_eps_cv,
        seed_eps_valid,
        kind=PERPENDICULAR_BURNING_SHIP
    )

def perturbation_shark_fin_find_any_nucleus(
    char *seed_x,
    char *seed_y,
    long seed_prec,
    long order,
    long max_newton,
    char *seed_eps_cv,
    char *seed_eps_valid
):
    return perturbation_nonholomorphic_find_any_nucleus(
        seed_x,
        seed_y,
        seed_prec,
        order,
        max_newton,
        seed_eps_cv,
        seed_eps_valid,
        kind=SHARK_FIN
    )

def perturbation_nonholomorphic_find_any_nucleus(
    char *seed_x,
    char *seed_y,
    long seed_prec,
    long order,
    long max_newton,
    char *seed_eps_cv,
    char *seed_eps_valid,
    const int kind
):
    cdef:
        # unsigned long int ui_one = 1
        bint newton_cv = False
        int cmp = 0
        long fail = -1
        long i_newton = 0
        long i = 0
        object gmpy_mpc
        iter_func_t iter_func
        iter_hessian_t iter_hessian

        mpfr_t xn_t, yn_t, a_t, b_t, da_t, db_t, xsq_t, ysq_t, xy_t
        mpfr_t dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t
        mpfr_t abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy
        mpfr_t eps_t, rx_t, ry_t, abs_diff
        mpfr_t _tmp

        mpc_t c_t

    iter_func = select_func(kind)
    iter_hessian = select_hessian(kind)

    # initialisation
    mpfr_inits2(seed_prec, xn_t, yn_t, a_t, b_t, da_t, db_t, xsq_t, ysq_t, xy_t, NULL)
    mpfr_inits2(seed_prec, dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t, NULL)
    mpfr_inits2(seed_prec, abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, NULL)
    mpfr_inits2(seed_prec, eps_t, rx_t, ry_t, abs_diff, NULL)
    mpfr_init2(_tmp, seed_prec)

    mpc_init2(c_t, seed_prec)
    
    # from char: set value of c - and of r0 = r = px
    mpfr_set_str(a_t, seed_x, 10, MPFR_RNDN)
    mpfr_set_str(b_t, seed_y, 10, MPFR_RNDN)
    mpfr_set_str(eps_t, seed_eps_cv, 10, MPFR_RNDN)

    # Value of epsilon
    # 64 = 2**4 * 4  - see the notes below on mpmath.almosteq
    mpfr_mul_si(eps_t, eps_t, 64, MPFR_RNDN) 
    
    # print("max_newton", max_newton, "order", order, mpfr_get_d(eps_t, MPFR_RNDN))

    for i_newton in range(max_newton):
        # zr = dzrdc = dh = 0 /  h = 1
        mpfr_set_si(xn_t, 0, MPFR_RNDN)
        mpfr_set_si(yn_t, 0, MPFR_RNDN)
        mpfr_set_si(dxnda_t, 0, MPFR_RNDN)
        mpfr_set_si(dxndb_t, 0, MPFR_RNDN)
        mpfr_set_si(dynda_t, 0, MPFR_RNDN)
        mpfr_set_si(dyndb_t, 0, MPFR_RNDN)

        # Newton descent
        for i in range(1, order + 1):
            iter_hessian(
                0,
                xn_t, yn_t, dxnda_t, dxndb_t, dynda_t, dyndb_t,
                abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, _tmp
            )
            iter_func(xn_t, yn_t, a_t, b_t, xsq_t, ysq_t, xy_t)

        # [da] = -J^-1 x [xn]
        # [db]           [yn]
        matsolve(
            da_t, db_t,
            dxnda_t, dxndb_t, dynda_t, dyndb_t,
            xn_t, yn_t,
            delta_t, _tmp
        )
        mpfr_sub(a_t, a_t, da_t, MPFR_RNDN)
        mpfr_sub(b_t, b_t, db_t, MPFR_RNDN)

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

        mpfr_hypot(abs_diff, da_t, db_t, MPFR_RNDN)
        cmp = mpfr_greaterequal_p(eps_t, abs_diff)
        # print(i, mpfr_get_d(abs_diff, MPFR_RNDN))
        if cmp != 0:
            # Sanity check that c_t is not a 'red herring' - zr_t shall be null
            # with 'enough' precision - depend on local pixel size
            mpfr_hypot(abs_diff, xn_t, yn_t, MPFR_RNDN)
            mpfr_set_str(eps_t, seed_eps_valid, 10, MPFR_RNDN)
            newton_cv  = (mpfr_greaterequal_p(eps_t, abs_diff) != 0)
            print("Newton cv ?", i, newton_cv,
                  mpfr_get_d(eps_t, MPFR_RNDN), mpfr_get_d(abs_diff, MPFR_RNDN)
            )
            break

    mpc_set_fr_fr(c_t, a_t, b_t, MPC_RNDNN)
    gmpy_mpc = GMPy_MPC_From_mpc(c_t)

    mpfr_clears(xn_t, yn_t, a_t, b_t, da_t, db_t, xsq_t, ysq_t, xy_t, NULL)
    mpfr_clears(dxnda_t, dxndb_t, dynda_t, dyndb_t, delta_t, NULL)
    mpfr_clears(abs_xn, abs_yn, tmp_xx, tmp_xy, tmp_yx, tmp_yy, NULL)
    mpfr_clears(eps_t, rx_t, ry_t, abs_diff, NULL)
    mpfr_clear(_tmp)
    
    mpc_clear(c_t)

    if newton_cv:
        return newton_cv, gmpy_to_mpmath_mpc(gmpy_mpc, seed_prec)
    return False, mpmath.mpc("nan", "nan")

#==============================================================================
#==============================================================================

def gmpy_to_mpmath_mpf(mpfr_gmpy, prec):
    """ Conversion beween a mpfr from GMPY2 and a mpmath mpf """
    cdef:
        object man, exp

    man, exp = mpfr_gmpy.as_mantissa_exp()
    with mpmath.workprec(prec):
        return mpmath.mp.make_mpf(
            gmpy2._mpmath_create(man, int(exp))
        )

def gmpy_to_mpmath_mpc(mpc_gmpy, prec):
    """ Conversion beween a mpc from GMPY2 and a mpmath mpc """
    cdef:
        object man_real, man_imag, exp_real, exp_imag

    man_real, exp_real = mpc_gmpy.real.as_mantissa_exp()
    man_imag, exp_imag = mpc_gmpy.imag.as_mantissa_exp()
    with mpmath.workprec(prec):
        return mpmath.mpc(
            gmpy2._mpmath_create(man_real, int(exp_real)),
            gmpy2._mpmath_create(man_imag, int(exp_imag)),
        )


def _test_mpfr_to_python(
        char *seed_x,
        long seed_prec
):
    """ Debug / test function: mpfr_t to mpmath.mpf
    """
    cdef:
        mpfr_t x_t
        object ret
        object exp_pyint
    
    mpfr_init2(x_t, seed_prec)

    # from char *: set value of x
    mpfr_set_str(x_t, seed_x, 10, MPFR_RNDN)
    mpfr_mul_si(x_t, x_t, 2, MPFR_RNDN)

    mpfr_gmpy = GMPy_MPFR_From_mpfr(x_t)  
    mpfr_clear(x_t)
    return gmpy_to_mpmath_mpf(mpfr_gmpy, seed_prec)

def _test_mpc_to_python(
        char *seed_x,
        char *seed_y,
        long seed_prec
):
    """ Debug / test function: mpc_t to mpmath.mpc
    """
    cdef:
        mpfr_t x_t, y_t
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

    gmpy_mpc = GMPy_MPC_From_mpc(c_t)
    mpfr_clear(x_t)
    mpfr_clear(y_t)
    mpc_clear(c_t)
    return gmpy_to_mpmath_mpc(gmpy_mpc, seed_prec)

