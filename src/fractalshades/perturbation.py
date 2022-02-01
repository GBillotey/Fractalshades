# -*- coding: utf-8 -*-
import os
import pickle
import mpmath
import numpy as np
import numba

import fractalshades as fs
import fractalshades.numpy_utils.xrange as fsx
import fractalshades.numpy_utils.numba_xr as fsxn
from fractalshades.mthreading import Multithreading_iterator


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
        self.iref = 0 # Only one reference point


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


    def ref_point_file(self):
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
            print("no ref pt file found FP_params")
            return False

        # Parameters 'max_iter' borrowed from last "@fsutils.calc_options" call
        calc_options = self.calc_options
        max_iter = calc_options["max_iter"]
        
        print("## in ref_point_matching", ref_point, type(ref_point))

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
        print("saved ref point")
        self._FP_params = FP_params
        self._Z_path = Z_path
        save_path = self.ref_point_file()
        fs.utils.mkdir_p(os.path.dirname(save_path))
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
        print("in FP_params", hasattr(self, "_FP_params"))
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
        print("in Z_path", hasattr(self, "_Z_path"))
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
        dx_xr = fsx.mpf_to_Xrange(self.dx, dtype=self.float_type).ravel()

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
        ).ravel()

        has_xr = (len(ref_xr_python) > 0)

        return Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, drift_xr, dx_xr


    def SA_file(self): # , iref, calc_name):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        Series Approximation
        """
        return os.path.join(self.directory, "data", "SA.dat")


    def save_SA(self, FP_params, SA_params, SA_kc, P, n_iter, P_err):
        """
        Reload arrays from a data file
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.SA_file()
        fs.utils.mkdir_p(os.path.dirname(save_path))

        with open(save_path, 'wb+') as tmpfile:
            print("SA computed, saving", save_path)
            for item in (FP_params, SA_params, SA_kc, P, n_iter, P_err):
                pickle.dump(item, tmpfile, pickle.HIGHEST_PROTOCOL)


    def reload_SA(self, scan_only=False):
        """
        Reload arrays from a data file
           - FP_params = main parameters used for ref pt calc
           - SA_params = main parameters used for SA calc
            P, n_iter, P_err : The SA results

        """
        save_path = self.SA_file()
        with open(save_path, 'rb') as tmpfile:
            FP_params = pickle.load(tmpfile)
            SA_params = pickle.load(tmpfile)
            SA_kc = pickle.load(tmpfile)
            if scan_only:
                return FP_params, SA_params, SA_kc
            P = pickle.load(tmpfile)
            n_iter = pickle.load(tmpfile)
            P_err = pickle.load(tmpfile)
        return FP_params, SA_params, SA_kc, P, n_iter, P_err

    def get_SA_data(self):
        """ return attribute or try to reload from file """
        if hasattr(self, "_SA_data"):
            return self._SA_data
        else:
            _, _, _, P, n_iter, P_err = self.reload_SA()
            return P, n_iter, P_err

    def SA_matching(self):
        """
        Test if the SA stored can be used for this calculation ie 
           - same ref point
           - same SA parameters
        """
        try:
            (stored_FP_params, stored_SA_params, stored_kc
             ) = self.reload_SA(scan_only=True)
        except FileNotFoundError:
            return False

        valid_FP_params = (stored_FP_params == self.FP_params)
        valid_SA_params = (stored_SA_params == self.SA_params)
        valid_kc = (stored_kc == self.kc)
        print("validate stored SA", valid_FP_params, valid_SA_params, valid_kc)

        return (valid_FP_params and valid_SA_params and valid_kc)

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
        
        # Lazzy compilation of subset boolean array chunk-span
        self._mask_beg_end = None

        # Initialise the reference path
        self.get_FP_orbit()
        (Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, drift_xr, dx_xr
         ) = self.get_Ref_path()


        # Initialise SA interpolation
        print("Initialise SA interpolation")
        self.kc = kc = self.ref_point_kc().ravel()  # Make it 1d for numba use
        if kc == 0.:
            raise RuntimeError(
                "Resolution is too low for this zoom depth. Try to increase"
                "the reference calculation precicion."
            )

        if self.SA_params is None:
            n_iter = 0
            P = None
            P_err = None
        else:
            self.get_SA(Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter)
            P, n_iter, P_err = self.get_SA_data()

        # Jitted function used in numba inner-loop
        self._initialize = self.initialize()
        self._iterate = self.iterate()        

        # Launch parallel computing of the inner-loop (Multi-threading with GIL
        # released)
        self.cycles(
            Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, drift_xr, dx_xr, 
            P, kc, n_iter,
            chunk_slice=None
        )

        # Export to human-readable format
        if fs.settings.inspect_calc:
            self.inspect_calc()

    @Multithreading_iterator(
        iterable_attr="chunk_slices", iter_kwargs="chunk_slice")
    def cycles(
        self, 
        Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, drift_xr, dx_xr, 
        P, kc, n_iter,
        chunk_slice
    ):
        """
        Fast-looping for Julia and Mandelbrot sets computation.

        Parameters:
        -----------
        chunk_slice : 4-tuple int
            The calculation tile
        Z_path : complex128[]
            The array for the reference point orbit, in low precision
        has_xr : boolean
            True if the Z_path needs extended range values - when it gets close
            to 0.
        ref_index_xr : int[:]
            indices for the extended range values
        ref_xr : X_range[:]
            The extended range values
        ref_div_iter : int
            The first invalid iteration for the reference orbit (either
            missing or diverging)
        drift_xr : X_range
            vector between image center and reference point : center - ref_pt
        dx_xr :
            width of the image (self.dx)
        P : Xrange_polynomial
            Polynomial from the Series approximation
        kc :
            Scaling coefficient for the Series Approximation
        n_iter :
            iteration to which "jumps" the SA approximation

        Returns:
        --------
        None
        
        Save to a file : 
        *raw_data* = (chunk_mask, Z, U, stop_reason, stop_iter) where
            *chunk_mask*    1d mask
            *Z*             Final values of iterated fields [ncomplex, :]
            *U*             Final values of int fields [nint, :]       np.int32
            *stop_reason*   Byte codes -> reasons for termination [:]  np.int8
            *stop_iter*     Numbers of iterations when stopped [:]     np.int32
        """
        if self.is_interrupted():
            print("interrupted", chunk_slice)
            return

        if self.res_available(chunk_slice):
            print("res available: ", chunk_slice)
            return

        (c_pix, Z, U, stop_reason, stop_iter
         ) = self.init_cycling_arrays(chunk_slice)

        initialize = self._initialize
        iterate = self._iterate

        ret_code = numba_cycles_perturb(
            c_pix, Z, U, stop_reason, stop_iter,
            initialize, iterate,
            Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, drift_xr, dx_xr,
            P, kc, n_iter,
            self._interrupted
        )
        if ret_code == self.USER_INTERRUPTED:
            print("Interruption signal received")
            return

        # Saving the results after cycling
        self.update_report_mmap(chunk_slice, stop_reason)
        self.update_data_mmaps(chunk_slice, Z, U, stop_reason, stop_iter)


    def param_matching(self, dparams):
        """
        If not matching shall trigger recomputing
        dparams is the stored computation
        """
#        print("**CALLING param_matching +++", self.params)

        UNTRACKED = [
            # "SA_params",
            "datetime",
            "debug",
        ]
        SPECIAL_CASE = ["prec"] # TODO should it be calc-param_prec ?
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
                        pass
#                        print("ok", key, val, dparams[key])
            else:
                if dparams[key] != val:
                    print("KO,", key, dparams[key], "-->", val)
                    return False
#                print("ok", key, val, dparams[key])
#        print("** TOTAL: param_matching")
        return True


    def get_SA(self, Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter):
        """
        Check if we have a suitable SA approximation stored for iref, 
          - otherwise computes and stores it in a file
        """
        if self.SA_matching():
            print("SA already stored")
            return

        else:            
            SA_params = self.SA_params
            FP_params = self.FP_params
            kc = self.kc

            SA_loop = self.SA_loop()
            SA_cutdeg = SA_params["cutdeg"]
            SA_err_sq = SA_params["err"] ** 2
            SA_stop = SA_params.get("stop", -1)

            P, n_iter, P_err = numba_SA_run(
                SA_loop, 
                Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter,
                kc, SA_cutdeg, SA_err_sq, SA_stop
            )
            self._SA_data = (P, n_iter, P_err)
            self.save_SA(FP_params, SA_params, kc, P, n_iter, P_err)
            
            


    def get_FP_orbit(self, c0=None, newton="cv", order=None,
                     max_newton=None):
        """
        # Check if we have a reference point stored for iref, 
          - otherwise computes and stores it in a file

        newton: ["cv", "step", None]
        """
        if newton == "step":
            raise NotImplementedError("step option not Implemented (yet)")
        
        # Early escape if file exists
        if self.ref_point_matching():
            print("Ref point already stored")
            return

        # no newton if zoom level is low. TODO: early escape possible
        if self.dx > fs.settings.newton_zoom_level:
            c0 = self.critical_pt
            self.compute_critical_orbit(c0)
            return

        if c0 is None:
            c0 = self.x + 1j * self.y

        # skip Newton if settings impose it
        if fs.settings.no_newton or (newton is None) or (newton == "None"):
            self.compute_FP_orbit(c0, None)
            return
        
        # Here we will compute a Newton iteration. First, guess a cycle order
        print("Launching Newton iteration process with ref point:\n", c0)

        # Try Ball method to find the order, then Newton
        if order is None:
            k_ball = 1.0
            order = self.ball_method(c0, self.dx * k_ball)
            if order is None: 
                print("##### Ball method failed #####")
                print("##### Default to image center or c0 #####")
                self.compute_FP_orbit(c0, None)
                return

        if order is None:
            raise ValueError("Order must be specified for Newton"
                             "iteration")
        if (newton == "step"):
            max_newton = 1

        print("newton ", newton, " with order: ", order)
        print("max newton iter ", max_newton)

        newton_cv, nucleus = self.find_nucleus(
            c0, order, max_newton=max_newton
        )

        # If Newton did not CV, we try to boost the dps / precision
        if not(newton_cv) and (newton != "step"):
            max_attempt = 2
            attempt = 0
            while not(newton_cv) and attempt < max_attempt:
                attempt += 1
                old_dps = mpmath.mp.dps
                mpmath.mp.dps = int(1.25 * old_dps)
                print(f"Newton failed, dps incr. {old_dps} -> {mpmath.mp.dps}")

                newton_cv, nucleus = self.find_nucleus(
                    c0, order, max_newton=max_newton
                )

                if not(newton_cv) and (attempt == max_attempt):
                    # Last try, we just release constraint on the cycle order
                    # and consider also divisors.
                    newton_cv, nucleus = self.find_any_nucleus(
                        c0, order, max_newton=max_newton
                    )

        # Still not CV ? we default to the center of the image
        if not(newton_cv) and (newton != "step"):
            nucleus = c0

        shift = nucleus - (self.x + self.y * 1j)
        shift_x = shift.real / self.dx
        shift_y = shift.imag / self.dx
        print("Reference nucleus found at shift (expressed in dx units):\n", 
              f"({shift_x}, {shift_y})",
              f"with order {order}"
        )
        self.compute_FP_orbit(nucleus, order)

    def compute_critical_orbit(self, crit):
        """
        Basically nothing to 'compute' here, just short-cutting
        """
        print("Short cutting ref pt orbit calc for low res")
        FP_code = self.FP_code
        # Parameters 'max_iter' borrowed from last "@fsutils.calc_options" call
        max_iter = self.max_iter

        FP_params = {
             "ref_point": crit,
             "dps": mpmath.mp.dps,
             "order": 0,
             "max_iter": max_iter,
             "FP_code": FP_code
        }
        # Given the "single reference" implementation the loop will wrap when
        # we reach div_iter - this is probably better not to do it too often
        # Let's just pick a reasonnable figure
        div_iter = min(100, self.max_iter)
        FP_params["partials"] = {}
        FP_params["xr"] = {}
        FP_params["div_iter"] = div_iter
        
        Z_path = np.empty([div_iter + 1], dtype=np.complex128)
        Z_path[:] = crit

        self.save_ref_point(FP_params, Z_path)


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


    def ball_method(self, c, px, kind=1, M_divergence=1.e5):
        """
        Use a ball centered on c = x + i y to find the first period (up to 
        maxiter) of nucleus
        """
        
        max_iter = self.max_iter
        print("ball method", c, type(c), px, type(px))
        
        if kind == 1:
            return self._ball_method(c, px, max_iter, M_divergence)
        elif kind == 2:
            return self._ball_method2(c, px, max_iter, M_divergence)


# Numba JIT functions =========================================================
Xr_template = fsx.Xrange_array.zeros([1], dtype=np.complex128)
Xr_float_template = fsx.Xrange_array.zeros([1], dtype=np.float64)
USER_INTERRUPTED = 1

@numba.njit(nogil=True)
def numba_cycles_perturb(
    c_pix, Z, U, stop_reason, stop_iter,
    initialize, iterate, 
    Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, drift_xr, dx_xr,
    P, kc, n_iter,
    _interrupted
):
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
    n_iter:
        current iteration
    """
    n_iter_init = n_iter

    nz, npts = Z.shape
    Z_xr = Xr_template.repeat(nz)
    Z_xr_trigger = np.zeros((nz,), dtype=np.bool_)
    
    for ipt in range(npts):
        n_iter = n_iter_init

        Z_xr_trigger = np.zeros((nz,), dtype=np.bool_)
        refpath_ptr = np.zeros((2,), dtype=np.int32)
        ref_is_xr = np.zeros((1,), dtype=numba.bool_)
        ref_zn_xr = Xr_template.repeat(1)

        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        cpt, c_xr = ref_path_c_from_pix(c_pix[ipt], dx_xr, drift_xr)
        stop_pt = stop_reason[:, ipt]

        initialize(Zpt, Upt, c_xr, Z_xr_trigger, Z_xr, P, kc, dx_xr, n_iter)
        
        n_iter = iterate(
            cpt, Zpt, Upt, stop_pt, n_iter,
            Z_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, drift_xr, dx_xr,
            Z_xr_trigger, Z_xr, c_xr, refpath_ptr, ref_is_xr, ref_zn_xr
        )
        stop_iter[0, ipt] = n_iter
        stop_reason[0, ipt] = stop_pt[0]

        if _interrupted[0]:
            return USER_INTERRUPTED
    return 0


@numba.njit
def numba_SA_run(
        SA_loop, 
        ref_path, has_xr, ref_index_xr, ref_xr, ref_div_iter,
        kc, SA_cutdeg, SA_err_sq, SA_stop
):
    """
    SA_loop function with signature (P, n_iter, ref_zn_xr, kcX)
    Ref_path : Ref_path object
    kc = Xrange float
    SA_err_sq : float
    SA_stop : int or -1
    SA_cutdeg :  int
    
    SA_stop : user-provided max SA iter. If -1, will default to ref_path length
    ref_div_iter : point where Reference point DV
    """
    if SA_stop == -1:
        SA_stop = ref_div_iter
    else:
        SA_stop = min(ref_div_iter, SA_stop)

    print_freq = max(5, int(SA_stop / 100000.))
    print_freq *= 1000
#    print("numba_SA_cycles - output every", print_freq)

    SA_valid = True
    n_iter = 0

    P0_arr = Xr_template.repeat(1)
    P0_err = Xr_float_template.repeat(1)
    P = fsx.Xrange_SA(P0_arr, cutdeg=SA_cutdeg, err=P0_err) # P0

    kcX_arr = Xr_template.repeat(2)
    kcX_arr[1] = kc[0]
    kcX_err = Xr_float_template.repeat(1)
    kcX = fsx.Xrange_SA(kcX_arr, cutdeg=SA_cutdeg, err=kcX_err)
    
    # refpath_ptr = [prev_idx, curr_xr]
    refpath_ptr = np.zeros((2,), dtype=numba.int32)
    out_is_xr = np.zeros((2,), dtype=numba.bool_)
    out_xr = Xr_template.repeat(2)

    while SA_valid:
        n_iter +=1
        # keep a copy in case this iter is invalidated
        P_old = P.coeffs.copy()

        # Load reference point value
        # refpath_ptr = [prev_idx, curr_xr]
        ref_zn = ref_path_get(
            ref_path, n_iter - 1,
            has_xr, ref_index_xr, ref_xr, refpath_ptr,
            out_is_xr, out_xr, 0
        )

        ref_zn_xr = ensure_xr(ref_zn, out_xr[0], out_is_xr[0])
        P = SA_loop(P, n_iter, ref_zn_xr, kcX)

        coeffs_sum = fsxn.Xrange_scalar(0., numba.int32(0))
        for i in range(len(P.coeffs)):
            coeffs_sum = coeffs_sum + fsxn.extended_abs2(P.coeffs[i])
        err_abs2 = P.err[0] * P.err[0]

        SA_valid = (
            (err_abs2  <= SA_err_sq * coeffs_sum) # relative err
            and (coeffs_sum <= 1.e6) # 1e6 to allow 'low zoom'
            and (n_iter < SA_stop)
        )
        if not(SA_valid):
            P_ret = fsx.Xrange_polynomial(P_old, P.cutdeg)
            n_iter -= 1

        if n_iter % print_freq == 0 and SA_valid:
            ssum = np.sqrt(coeffs_sum)
            print(
                "SA running", n_iter,
                "err: ", fsxn.to_Xrange_scalar(P.err[0]),
                "<< ", ssum
            )

    return P_ret, n_iter, P.err


@numba.njit
def need_xr(x_std):
    """
    True if norm L-inf of std is lower than xrange_zoom_level
    """
    return (
        (abs(x_std).real < fs.settings.xrange_zoom_level)
         and (abs(x_std.imag) < fs.settings.xrange_zoom_level)
    )

@numba.njit
def ensure_xr(val_std, val_xr, is_xr):
    """
    Return a valid Xrange. if not(Z_xr_trigger) we return x_std
    converted
    
    val_xr : complex128_Xrange_scalar or float64_Xrange_scalar
    """
    if is_xr:
        return fsxn.to_Xrange_scalar(val_xr)
    else:
        return fsxn.to_Xrange_scalar(val_std)


@numba.njit
def ref_path_c_from_pix(pix, dx, drift):
    """
    Returns the true c (coords from ref point) from the pixel coords
    
    Parameters
    ----------
    pix :  complex
        pixel location in farction of dx
        
    Returns
    -------
    c, c_xr : c value as complex and as Xrange
    """
    c_xr = (pix * dx[0]) + drift[0]
    return fsxn.to_standard(c_xr), c_xr


@numba.njit
def ref_path_get(ref_path, idx, has_xr, ref_index_xr, ref_xr, refpath_ptr,
                 out_is_xr, out_xr, out_index):
    """
    Alternative to getitem which also takes as input prev_idx, curr_xr :
    allows to optimize the look-up of Xrange values in case of successive calls
    with strictly increasing idx.

    idx :
        index requested
    (prev_idx, curr_xr) :
        couple returned from last call, last index requested + next xr target
        Contract : curr_xr the smallest integer that verify :
            prev_idx <= ref_index_xr[curr_xr]
            or curr_xr = ref_index_xr.size (No more xr)
    Returns
    -------
    (val, xr_val, is_xr, prev_idx, curr_xr)
        val : np.complex128
        xr_val : complex128_Xrange_scalar
        is_xr : bool
        prev_idx == refpath_ptr[0] : int
        curr_xr == refpath_ptr[1] : int (index in path.ref_xr)
    """
    if not(has_xr):
        return ref_path[idx]

    # Not an increasing sequence, reset to restart a new sequence
    if idx < refpath_ptr[0]:
        # Rewind to 0
        refpath_ptr[0] = 0 # prev_idx = 0
        refpath_ptr[1] = 0 # curr_xr = 0

    # In increasing sequence (idx >= prev_idx)
    if (
        (refpath_ptr[1] >= ref_index_xr.size)
        or (idx < ref_index_xr[refpath_ptr[1]])
    ):
        refpath_ptr[0] = idx
        out_is_xr[out_index] = False
        return ref_path[idx]

    elif idx == ref_index_xr[refpath_ptr[1]]:
        refpath_ptr[0] = idx
        out_is_xr[out_index] = True
        out_xr[out_index] = ref_xr[refpath_ptr[1]]
        return ref_path[idx]

    else:
        # Here we have idx > ref_index_xr[curr_xr]:
        while (
            (idx > ref_index_xr[refpath_ptr[1]])
            and (refpath_ptr[1] < ref_index_xr.size)
        ):
            refpath_ptr[1] += 1
        if (
            (refpath_ptr[1] == ref_index_xr.size)
            or (idx < ref_index_xr[refpath_ptr[1]])
        ):
            refpath_ptr[0] = idx
            out_is_xr[out_index] = False
            return ref_path[idx]
        # Here idx == ref_index_xr[refpath_ptr[1]]
        refpath_ptr[0] = idx
        out_is_xr[out_index] = True
        out_xr[out_index] = ref_xr[refpath_ptr[1]]

        return ref_path[idx]
