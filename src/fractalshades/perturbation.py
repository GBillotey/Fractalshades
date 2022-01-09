# -*- coding: utf-8 -*-

import os
import pickle
# import copy
import concurrent
#import random

import mpmath
import numpy as np
import numba
#import random

import fractalshades as fs
import fractalshades.numpy_utils.xrange as fsx
import fractalshades.numpy_utils.numba_xr as fsxn
# import fractalshades.bivar_series

#from numba.pycc import CC
#cc = CC("fractalshades")

#import fractalshades.numpy_utils.numba_xr # as fsxn
#import fractalshades.utils as fsutils
#import fractalshades.postproc as fspp
#force_recompute_SA = True
# from fractalshades.mprocessing import Multiprocess_filler

class PerturbationFractal(fs.Fractal):

    def __init__(self, directory):
        """
The base class for escape-time fractals calculations implementing the 
perturbation technique.

Derived class should implement the actual calculation methods used in the
innner loop, but also reference loop calculation, and optionnaly,
Series approximation, glitch detection.
This class provides the outer looping (calculation is run on 
successive tiles), enables multiprocessing, and manage raw-result storing 
and retrieving.

Parameters
----------
directory : str
    Path for the working base directory
        """
        super().__init__(directory)
        

    @fs.utils.zoom_options
    def zoom(self, *,
             precision: int,
             x: mpmath.mpf,
             y: mpmath.mpf,
             dx: mpmath.mpf,
             nx: int,
             xy_ratio: float,
             theta_deg: float,
             projection: str="cartesian",
             antialiasing: bool=False):
        """
        Define and stores as class-attributes the zoom parameters for the next
        calculation.
        
        Parameters
        ----------
        precision : int
            number of significant base-10 digits to use for full precision
            calculations.
        x : str or mpmath.mpf
            x-coordinate of the central point
        y : str or mpmath.mpf
            y-coordinate of the central point
        dx : str or mpmath.mpf
            span of the view rectangle along the x-axis
        nx : int
            number of pixels of the image along the x-axis
        xy_ratio: float
            ratio of dx / dy and nx / ny
        theta_deg : float
            Pre-rotation of the calculation domain, in degree
        projection : "cartesian"
            Kind of projection used (only "cartesian" supported)
        antialiasing : bool
            If True, some degree of randomization is applied
        """
        mpmath.mp.dps = precision # in base 10 digit 
        
        
        # In case the user inputs were strings, we override with mpmath scalars
        self.x = mpmath.mpf(x)
        self.y = mpmath.mpf(y)
        self.dx = mpmath.mpf(dx)


    @property
    def xr_detect_activated(self):
        """ Triggers use of special dataype to avoid underflow in double """
        return (self.dx < fs.settings.xrange_zoom_level)


    def postproc_chunck(self, postproc_keys, chunk_slice, calc_name):
        """
        Postproc a stored array of data
        reshape as a sub-image crop to feed fractalImage

        called by :
        if reshape
            post_array[ix:ixx, iy:iyy, :]
        else
            post_array[:, npts], chunk_mask

        """
        params, codes, raw_data = self.reload_data_chunk(chunk_slice,
                                                         calc_name)
        chunk_mask, Z, U, stop_reason, stop_iter = raw_data
        complex_dic, int_dic, termination_dic = self.codes_mapping(*codes)

        full_Z = Z.copy()
        for key, val in complex_dic.items():
            # If this field is found in a full precision array, we add it :
            if key == self.FP_code:
                Z_path = self.Z_path
                full_Z[val, :] += Z_path[U[0, :]]  # U[0] is ref_cycle_iter

        full_raw_data = (chunk_mask, full_Z, U, stop_reason, stop_iter)

        post_array, chunk_mask = self.postproc(postproc_keys, codes,
            full_raw_data, chunk_slice)
        return self.reshape2d(post_array, chunk_mask, chunk_slice)


    def ref_point_file(self): # , iref, calc_name):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        reference orbit
        """
        return os.path.join(self.directory, "data", "ref_pt.dat")


    def ref_point_kc(self):
        """
        Return a scaling coefficient used as a convergence radius for serie 
        approximation, or as a reference scale for derivatives.

        Returns:
        --------
        kc: full precision, scaling coefficient
        """
        c0 = self.x + 1j * self.y
        corner_a = c0 + 0.5 * (self.dx + 1j * self.dy)
        corner_b = c0 + 0.5 * (- self.dx + 1j * self.dy)
        corner_c = c0 + 0.5 * (- self.dx - 1j * self.dy)
        corner_d = c0 + 0.5 * (self.dx - 1j * self.dy)

        ref = self.FP_params["ref_point"]

        # Let take some margin
        kc = max(abs(ref - corner_a), abs(ref - corner_b),
                 abs(ref - corner_c), abs(ref - corner_d)) * 2.0

        return fsx.mpf_to_Xrange(kc, dtype=np.float64)


    def ref_point_matching(self):
        """
        Test if the ref point can be used for this calculation ie 
           - same or more max_iter
           - not too far
           - and with a suitable dps
        """
        try:
            ref_point = self.FP_params["ref_point"]
            max_iter_ref = self.FP_params["max_iter"]
        except FileNotFoundError:
            return False


        # Parameters 'max_iter' borrowed from last "@fsutils.calc_options" call
        calc_options = self.calc_options
        max_iter = calc_options["max_iter"]

        drift_xr = fsx.mpc_to_Xrange((self.x + 1j * self.y) - ref_point)
        dx_xr = fsx.mpf_to_Xrange(self.dx)

        matching = (
            (mpmath.mp.dps <= self.FP_params["dps"] + 3)
            and ((drift_xr / dx_xr).abs2() < 1.e6)
            and (max_iter_ref >= max_iter)
        )
        print("ref point matching", matching)
        print("dps -->", (mpmath.mp.dps <= self.FP_params["dps"] + 3))
        print("position -->", ((drift_xr / dx_xr).abs2() < 1.e6))
        print("max_iter -->", (max_iter_ref >= max_iter))
        return matching


    def save_ref_point(self, FP_params, Z_path):
        """
        Write to a data file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.ref_point_file()
        fs.utils.mkdir_p(os.path.dirname(save_path))
        self._FP_params = FP_params
        self._Z_path = Z_path
        with open(save_path, 'wb+') as tmpfile:
            print("Path computed, saving", save_path)
            pickle.dump(FP_params, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(Z_path, tmpfile, pickle.HIGHEST_PROTOCOL)


    def reload_ref_point(self, scan_only=False):
        """
        Reload arrays from a data file
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.ref_point_file()
        with open(save_path, 'rb') as tmpfile:
            FP_params = pickle.load(tmpfile)
            if scan_only:
                return FP_params
            Z_path = pickle.load(tmpfile)
        return FP_params, Z_path

    @property
    def FP_params(self):
        """
        Return the FP_params attribute, if not available try to reload it
        from file
        """
        if hasattr(self, "_FP_params"):
            return self._FP_params
        FP_params = self.reload_ref_point(scan_only=True)
        self._FP_params = FP_params
        return FP_params
        
    @property
    def Z_path(self):
        """
        Return the Z_path attribute, if not available try to reload it
        from file
        """
        if hasattr(self, "_Z_path"):
            return self._Z_path
        FP_params, Z_path = self.reload_ref_point()
        self._Z_path = Z_path
        return Z_path


    def get_Ref_path(self):
        """ Builds a Ref_path object from FP_params and ref_path
        This object will be used in numba jitted functions
        """
        FP_params = self.FP_params
        Z_path = self.Z_path
        
        ref_xr_python = FP_params["xr"]
        ref_div_iter = FP_params["div_iter"]
        dx_xr = fsx.mpf_to_Xrange(self.dx, dtype=self.float_type)

        # Build 2 arrays to avoid using a dict in numba
        ref_index_xr = np.empty([len(ref_xr_python)], dtype=np.int32)
        # /!\ ref_xr at least len 1 to ensure typing as complex
        ref_xr = fsx.Xrange_array([0j] * max(len(ref_xr_python), 1))
        for i, xr_index in enumerate(ref_xr_python.keys()):
            ref_index_xr[i] = xr_index
            ref_xr[i] = ref_xr_python[xr_index][0]

        # Complex distance between image center and ref point 
        drift_xr = fsx.mpc_to_Xrange(
            (self.x + 1j * self.y) - FP_params["ref_point"],
            dtype=self.complex_type
        )
        Ref_path = fsxn.Ref_path(
            Z_path, ref_index_xr, ref_xr, ref_div_iter,
            drift_xr, dx_xr
        )
        return Ref_path

#==============================================================================
# Printing functions

    @staticmethod
    def print_FP(FP_params, ref_path):
        """
        Just a pretty-print of the reference path
        """
        print("--------------------------------------------------------------")
        print("Full precision orbit loaded, FP_params:")
        for k, v in FP_params.items():
            try:
                for kv, vv in v.items():
                    print(k, f"({kv}) --> ", str(vv))
            except AttributeError:
                print(k, " --> ", v)
        print("ref_path, shape: ", ref_path.shape, ref_path.dtype) 
        print(ref_path)
        print("--------------------------------------------------------------")

#==============================================================================
# Calculation functions
    def run(self):
        """
        Launch a full perturbation calculation with glitch correction.

        The parameters from the last 
        @ `fractalshades.zoom_options`\-tagged method call and last
        @ `fractalshades.calc_options`\-tagged method call will be used.

        If calculation results are already there, the parameters will be
        compared and if identical, the calculation will be skipped. This is
        done for each tile and each glitch correction iteration, so i enables
        calculation to restart from an unfinished status.
        """

        if not(self.res_available()):
            # We write the param file and initialize the
            # memmaps for progress reports and calc arrays
            # /!\ It is not process safe, do it before multi-processing
            # loop
            fs.utils.mkdir_p(os.path.join(self.directory, "data"))
            self.open_report_mmap()
            self.open_data_mmaps()
            self.save_params()

        # Initialise the Ref_path object
        self.get_FP_orbit()
        Ref_path = self.get_Ref_path()

        # Initialise the Bivar_interpolator object
        kc = self.ref_point_kc()
        SA_params = self.SA_params
        SA_loop = self.SA_loop()

        if SA_params is None:
            Bivar_interpolator = None
        elif SA_params["kind"] == "bivar":
            Bivar_interpolator = fsxn.make_Bivar_interpolator(
                Ref_path, SA_loop, kc, SA_params)
            print("Bivar_interpolator")
            print("bivar_SA_sto:\n", Bivar_interpolator.bivar_SA_sto)

        # Jitted function used in numba inner-loop
        self._initialize = self.initialize()
        self._iterate = self.iterate()        

        # Launch parallel computing of the inner-loop (Multi-threading with GIL
        # released)
        print("Launching parallel loop")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3 #os.cpu_count()
                    ) as threadpool:
            futures = (
                threadpool.submit(
                    self.cycles,
                    chunk_slice, Ref_path, Bivar_interpolator
                )
                for chunk_slice in self.chunk_slices()
            )
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        # Export to human-readable format
        if fs.settings.inspect_calc:
            self.inspect_calc()


    def cycles(self, chunk_slice, Ref_path, Bivar_interpolator):
        """
        Fast-looping for Julia and Mandelbrot sets computation.

        Parameters
        *initialize*  function(Z, U, c) modify in place Z, U (return None)
        *iterate*   function(Z, U, c, n) modify place Z, U (return None)

        *subset*   bool arrays, iteration is restricted to the subset of current
                   chunk defined by this array. In the returned arrays the size
                    of axis ":" is np.sum(subset[ix:ixx, iy:iyy]) - see below
        *codes*  = complex_codes, int_codes, termination_codes
        *calc_name* prefix identifig the data files
        *chunk_slice_c* None - provided by the looping wrapper
        
        *iref* *ref_path* : defining the reference path, for iterations with
            perturbation method. if iref > 0 : means glitch correction loop.
        

        *gliched* boolean Fractal_Data_array of pixels that should be updated
                  with a new ref point
        *irefs*   integer Fractal_Data_array of pixels current ref points
        
        
        Returns 
        None - save to a file. 
        *raw_data* = (chunk_mask, Z, U, stop_reason, stop_iter) where
            *chunk_mask*    1d mask
            *Z*             Final values of iterated complex fields shape [ncomplex, :]
            *U*             Final values of int fields [nint, :]       np.int32
            *stop_reason*   Byte codes -> reasons for termination [:]  np.int8
            *stop_iter*     Numbers of iterations when stopped [:]     np.int32
        """
        if self.is_interrupted():
            return
        if self.res_available(chunk_slice):
            return

        (c_pix, Z, U, stop_reason, stop_iter
         ) = self.init_cycling_arrays(chunk_slice)

        initialize = self._initialize
        iterate = self._iterate
        
        print("c_pix", c_pix.shape)
        print("Z", Z.shape)
        print("U", U.shape)

        numba_cycles_perturb(c_pix, Z, U, stop_reason, stop_iter,
                initialize, iterate, Ref_path, Bivar_interpolator)

        # Saving the results after cycling
        self.update_report_mmap(chunk_slice, stop_reason)
        self.update_data_mmaps(chunk_slice, Z, U, stop_reason, stop_iter)


    def param_matching(self, dparams):
        """
        If not matching shall trigger recomputing
        dparams is the stored computation
        """
        print("**CALLING param_matching +++", self.params)
        # TODO : note: when comparing iref should be disregarded ? 
        # or subclass specific implementation
        UNTRACKED = [
            "SA_params",
            "datetime",
            "debug",
        ]
        SPECIAL_CASE = ["prec"] # TODO should it be calc-param_prec
        for key, val in self.params.items():
            if (key in UNTRACKED):
                continue
            elif (key in SPECIAL_CASE):
                if key == "prec":
                    if dparams[key] < val:
                        print("KO, higher precision requested",
                              dparams[key], "-->",  val)
                        return False
                    else:
                        print("ok", key, val, dparams[key])
            else:
                if dparams[key] != val:
                    print("KO,", key, val, "-->", dparams[key])
                    return False
                print("ok", key, val, dparams[key])
        print("** TOTAL: param_matching")
        return True


    def init_cycling_arrays(self, chunk_slice):
        print("init_cycling_arrays", chunk_slice)
        c_pix = np.ravel(self.chunk_pixel_pos(chunk_slice))
        print("c_pix", c_pix, c_pix.shape)
        if self.subset is not None:
            chunk_mask = self.subset[chunk_slice]
            c_pix = c_pix[chunk_mask]
        
        # Initialise the result arrays
        (n_pts,) = c_pix.shape

        n_Z, n_U, n_stop = (len(code) for code in self.codes)

        Z = np.zeros([n_Z, n_pts], dtype=self.complex_type)
        U = np.zeros([n_U, n_pts], dtype=self.int_type)
        stop_reason = - np.ones([1, n_pts], dtype=self.termination_type)
        stop_iter = np.zeros([1, n_pts], dtype=self.int_type)

        return (c_pix, Z, U, stop_reason, stop_iter)



    def get_FP_orbit(self, c0=None, newton="cv", order=None,
                         randomize=False):
        """
        # Check if we have a reference point stored for iref, 
          - otherwise computes and stores it in a file
        
        newton: ["cv", "step", None]
        """
        # Early escape if file exists
        if self.ref_point_matching():
            print("Ref point already stored")
            return

        # no newton if zoom level is low. TODO: early escape possible
        if self.dx > fs.settings.newton_zoom_level:
            c0 = self.critical_pt
            newton = None

        # skip Newton if settings impose it
        if fs.settings.no_newton:
            newton = None
        
        if c0 is None:
            c0 = self.x + 1j * self.y

        if randomize:
            data_type = self.base_float_type
            rg = np.random.default_rng(0)
            diff = rg.random([2], dtype=data_type) * randomize
            print("RANDOMIZE, diff", diff)
            c0 = (c0 + self.dx * (diff[0] - 0.5)
                + self.dy * (diff[1] - 0.5) * 1j)

        pt = c0
        print("Proposed ref point:\n", c0)

        # If we plan a newton iteration, we launch the process
        # ball method to find the order, then Newton
        if (newton is not None) and (newton != "None"):
            if order is None:
                k_ball = 0.01
                order = self.ball_method(c0, self.dx * k_ball)
                if order is None: # ball method escaped... we try to recover
                    if randomize < 5:
                        randomize += 1
                        print("BALL METHOD RANDOM ", randomize)
                        self.get_ref_point(c0=None, newton=newton, order=None,
                                 randomize=randomize)
                    else:
                        raise ValueError("Ball method failed")

            max_newton = 1 if (newton == "step") else None
            print("newton ", newton, " with order: ", order)
            print("max newton iter ", max_newton)

            newton_cv, nucleus = self.find_nucleus(
                    c0, order, max_newton=max_newton)

            if not(newton_cv) and (newton != "step"):
                max_attempt = 2
                attempt = 0
                while not(newton_cv) and attempt < max_attempt:
                    attempt += 1
                    old_dps = mpmath.mp.dps
                    mpmath.mp.dps = int(1.25 * old_dps)
                    print("Newton not cv, dps boost to: ", mpmath.mp.dps)
                    if attempt < max_attempt:
                        newton_cv, nucleus = self.find_nucleus(
                            c0, order, max_newton=max_newton)
                    else:
                        newton_cv, nucleus = self.find_any_nucleus(
                            c0, order, max_newton=max_newton)
                        
#                    newton_cv, nucleus = self.find_any_nucleus(
#                        c0, order, max_newton=max_newton)

#                if not(newton_cv) and (newton != "step"):
#                    newton_cv, nucleus = self.find_any_attracting(
#                        c0, order, max_newton=max_newton)

            if newton_cv or (newton == "step"):
                shift = nucleus - (self.x + self.y * 1j)
                print("Reference nucleus at:\n", nucleus, order)
                print("With shift % from image center:\n",
                      shift.real / self.dx, shift.imag / self.dy)
                shift = nucleus - pt
                print("With shift % from proposed coords:\n",
                      shift.real / self.dx, shift.imag / self.dy)
            else:
                print("Newton failed with order", order)
                nucleus = pt

            pt = nucleus

        print("compute ref_point")
        
        self.compute_FP_orbit(pt, order)


    def compute_FP_orbit(self, ref_point, order=None):
        """
        Computes full precision orbit, and stores path in normal precision
        FP_params keys:
            ref_point : starting point for the FP orbit
            order : ref cycle order or None
            max_iter : max iteration possible
            FP_codes : orbit sored fields
            div_iter : First invalid iteration (either diverging or not stored)
            partials : dictionary iteration -> partial value
            xr : dictionary iteration -> xr_value
        """
        FP_code = self.FP_code
        # Parameters 'max_iter' borrowed from last "@fsutils.calc_options" call
        max_iter = self.max_iter
        
        FP_params = {
             "ref_point": ref_point,
             "dps": mpmath.mp.dps,
             "order": order,
             "max_iter": max_iter,
             "FP_code": FP_code
        }

        Z_path = np.empty([max_iter + 1], dtype=np.complex128)

        print("Computing full precision path")

        i, partial_dict, xr_dict = self.FP_loop(Z_path, ref_point)
        FP_params["partials"] = partial_dict
        FP_params["xr"] = xr_dict
        FP_params["div_iter"] = i

        self.save_ref_point(FP_params, Z_path)
        self._FP_params = FP_params
        self._Z_path = Z_path


    def ball_method(self, c, px, order=1, M_divergence=1.e5):
        """
        Use a ball centered on c = x + i y to find the first period (up to 
        maxiter) of nucleus
        """
        max_iter = self.max_iter
        print("ball method", c, px)
        if order == 1:
            return self._ball_method1(c, px, max_iter, M_divergence)
        elif order == 2:
            return self._ball_method2(c, px, max_iter, M_divergence)


# Numba JIT functions =========================================================
Xr_template = fsx.Xrange_array.zeros([1], dtype=np.complex128)

@numba.njit(nogil=True) #, cache=True)
#@cc.export(
#    'numba_cycles_perturb',
#    (
#        numba.complex128[:],
#        numba.complex128[:],
#        numba.int32[:],
#        numba.int8[:],
#        numba.int32[:],
#        numba.int32[:],
#        numba.int32[:],
#        fractalshades.bivar_series.Ref_path_Type(),
#        fractalshades.bivar_series.Bivar_interpolator_Type()
#    )
#)
def numba_cycles_perturb(c_pix, Z, U, stop_reason, stop_iter,
                initialize, iterate, Ref_path, Bivar_interpolator):
    """
    Run the perturbation cycles

    Parameters:
    -----------
    Z, U, c, stop_reason, stop_iter
        result arrays
    iterate :
        numba jitted function
    Ref_path:
        Ref_path numba object
    Bivar_interpolator:
        Bivar_interpolator numba object
    """
    print("in numba cycle")

    nz, npts = Z.shape
    Z_xr = Xr_template.repeat(nz)
    Z_xr_trigger = np.zeros((nz,), dtype=np.bool_)

    # npts = c.size
    
    for ipt in range(npts): # npts): # DEBUG  #npts):
        # skip this ipt if pixel not active 
        n_iter = 0
        Z_xr_trigger = np.zeros((nz,), dtype=np.bool_)
        refpath_ptr = np.zeros((2,), dtype=np.int32)

        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        cpt, c_xr = Ref_path.c_from_pix(c_pix[ipt])
        stop_pt = stop_reason[:, ipt]

        initialize(Zpt, Upt, cpt)

        cycling = True
        while cycling:
#            if n_iter%1000 == 0:
#                print("n_iter", n_iter)
            # n_iter += 1 need to have n_iter inside...
            n_iter = iterate(
                cpt, Zpt, Upt, stop_pt, n_iter,
                Ref_path, Bivar_interpolator,
                Z_xr_trigger, Z_xr, c_xr, refpath_ptr
            )
            cycling = (stop_pt[0] == -1)  # & n_iter > 200 # debug
            if not(cycling):
                stop_iter[0, ipt] = n_iter
                stop_reason[0, ipt] = stop_pt[0]
