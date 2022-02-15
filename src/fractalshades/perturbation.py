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

    def new_status(self, wget):
        """ Return a dictionnary that can hold the current progress status """
        base_status = super().new_status(wget)
        status = {
            "Reference": {
                "str_val": "--"
            },
            "Bilin. approx": {
                "str_val": "--"
            },
        }
        status.update(base_status)
        return status

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
                Zn_path = self.Zn_path
                full_Z[val, :] += Zn_path[U[0, :]]  # U[0] is ref_cycle_iter

        full_raw_data = (chunk_mask, full_Z, U, stop_reason, stop_iter)

        post_array, chunk_mask = self.postproc(postproc_keys, codes,
            full_raw_data, chunk_slice)
        return self.reshape2d(post_array, chunk_mask, chunk_slice)


    # file, matching,save, reload, Params, data

    def ref_point_file(self):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        reference orbit
        """
        return os.path.join(self.directory, "data", "ref_pt.dat")


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
        print("max_iter -->", (max_iter_ref >= max_iter), max_iter_ref, max_iter)
        return matching


    def save_ref_point(self, FP_params, Zn_path):
        """
        Write to a data file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        print("saved ref point")
        self._FP_params = FP_params
        self._Zn_path = Zn_path
        save_path = self.ref_point_file()
        fs.utils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            print("Path computed, saving", save_path)
            pickle.dump(FP_params, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(Zn_path, tmpfile, pickle.HIGHEST_PROTOCOL)


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
            Zn_path = pickle.load(tmpfile)
        return FP_params, Zn_path

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
    def Zn_path(self):
        """
        Return the Zn_path attribute, if not available try to reload it
        from file
        """
        print("in Zn_path", hasattr(self, "_Zn_path"))
        if hasattr(self, "_Zn_path"):
            return self._Zn_path
        _, Zn_path = self.reload_ref_point()
        self._Zn_path = Zn_path
        return Zn_path


    def get_ref_path_data(self):
        """ Builds a Ref_path tuple from FP_params and ref_path
        This tuple will be used in numba jitted functions
        """
        FP_params = self.FP_params
        Zn_path = self.Zn_path
        
        ref_xr_python = FP_params["xr"]
        ref_order = FP_params["order"]
        ref_div_iter = FP_params["div_iter"] # The first invalid iter
        # ref_div_iter should be ref div iter of the FP, only if it is div.
        # Otherwise, the max_iter from calc param
        if ref_order is not None: # The reference orbit is a cycle
            ref_div_iter = self.max_iter + 1

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


        print("ref_div_iter", ref_div_iter, np.shape(Zn_path)[0])
        print("has_xr", has_xr, (len(ref_index_xr) > 0))
#        assert(ref_div_iter == np.shape(Zn_path)[0])
        assert(has_xr == (len(ref_index_xr) > 0))

        return (Zn_path, ref_order, ref_index_xr, ref_xr, drift_xr, dx_xr)


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
                 abs(ref - corner_c), abs(ref - corner_d)) * 1.1

        return fsx.mpf_to_Xrange(kc, dtype=np.float64)


    def BLA_file(self): # , iref, calc_name):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        Series Approximation
        """
        return os.path.join(self.directory, "data", "BLA.dat")

    def BLA_param_matching(self, BLA_full_params):
        """
        Test if the BLA can be used for this calculation ie 
           - same kc
           - same FP_params
        """
        try:
            stored_BLA_full_params = self.BLA_full_params
        except FileNotFoundError:
            print("no ref pt file found BLA_params")
            return False
        
        kc_stored = stored_BLA_full_params["kc"]
        kc_expected = BLA_full_params["kc"]
        ref_stored = stored_BLA_full_params["FP_params"]["ref_point"]
        ref_expected = BLA_full_params["FP_params"]["ref_point"]
        eps_stored = stored_BLA_full_params["eps"]
        eps_expected = BLA_full_params["eps"]
        
        matching = (
            (kc_stored == kc_expected)
            and (ref_stored == ref_expected)
            and (eps_stored == eps_expected)
        )
        print("Result of BLA matching:", matching)
        return matching



    def save_BLA(self, BLA_full_params, A, B, r):
        """
        Save BLA parameters and data tree arrays
        """
        print("SAVE BLA")
        print("BLA_full_params: \n", BLA_full_params)
        print("A:\n", A)
        print("B:\n", B)
        print("r:\n", r)
        self._BLA_full_params = BLA_full_params
        self._BLA_arrays = (A, B, r)

        save_path = self.BLA_file()
        fs.utils.mkdir_p(os.path.dirname(save_path))

        with open(save_path, 'wb+') as tmpfile:
            print("BLA computed, saving", save_path)
            for item in (BLA_full_params, A, B, r):
                pickle.dump(item, tmpfile, pickle.HIGHEST_PROTOCOL)


    def reload_BLA(self, scan_only=False):
        """
        Reload arrays from a data file
           - FP_params = parameters used for ref pt calc
           - BLA_params = parameters used for BLA calc
            A, B, r : The BLA data arrays

        """
        save_path = self.BLA_file()
        with open(save_path, 'rb') as tmpfile:
            BLA_full_params = pickle.load(tmpfile)
            if scan_only:
                return BLA_full_params
            A = pickle.load(tmpfile)
            B = pickle.load(tmpfile)
            r = pickle.load(tmpfile)
        return BLA_full_params, A, B, r

    @property
    def BLA_full_params(self):
        """
        Return the BLA_full_params attribute, if not available try to reload it
        from file
        """
        if hasattr(self, "_BLA_full_params"):
            return self._BLA_full_params
        BLA_full_params = self.reload_BLA(scan_only=True)
        self._BLA_full_params = BLA_full_params
        return BLA_full_params

    @property
    def BLA_arrays(self):
        """ return attribute or try to reload from file """
        if hasattr(self, "_BLA_arrays"):
            return self._BLA_arrays
        else:
            _, A, B, r = self.reload_BLA()
            return (A, B, r)

    def get_BLA_data(self):
        """ Builds a Ref_path tuple from FP_params and ref_path
        This tuple will be used in numba jitted functions
        """
        BLA_full_params = self.BLA_full_params
        (A, B, r) = self.BLA_arrays
        return BLA_full_params, A, B, r
#    def BLA_matching(self):
#        """
#        Test if the BLA stored can be used for this calculation ie 
#           - same ref point
#           - same BLA parameters
#        """
#        try:
#            (stored_FP_params, stored_BLA_params
#             ) = self.reload_BLA(scan_only=True)
#        except FileNotFoundError:
#            return False
#
#        valid_FP_params = (stored_FP_params == self.FP_params)
#        valid_BLA_params = (stored_BLA_params == self.BLA_params)
#        # valid_kc = (stored_kc == self.kc)
#        print("validate stored FP / BLA", valid_FP_params, valid_BLA_params)
#
#        return (valid_FP_params and valid_BLA_params)

#==============================================================================
# Printing functions
    @staticmethod
    def print_FP(FP_params, Zn_path):
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
        print("Zn_path, shape: ", Zn_path.shape, Zn_path.dtype) 
        print(Zn_path)
        print("--------------------------------------------------------------")

#==============================================================================
# Calculation functions
    def run(self):
        """
        Launch a full perturbation calculation with bilinear approximation.

        The parameters from the last 
        @ `fractalshades.zoom_options`\-tagged method call and last
        @ `fractalshades.calc_options`\-tagged method call will be used.

        If calculation results are already there, the parameters will be
        compared and if identical, the calculation will be skipped. This is
        done for each tile so allows calculation to restart from an unfinished
        state.
        """
        has_status_bar = hasattr(self, "_status_wget")

        if not(self.res_available()):
            # We write the param file and initialize the
            # memmaps for progress reports and calc arrays
            # /!\ It is not thread safe, do it before multithreading
            # loop
            fs.utils.mkdir_p(os.path.join(self.directory, "data"))
            self.open_report_mmap()
            self.open_data_mmaps()
            self.save_params()

        # Lazzy compilation of subset boolean array chunk-span
        self._mask_beg_end = None

        # Initialise the reference path
        if has_status_bar:
            self.set_status("Reference", "running")
        self.get_FP_orbit()
        (Zn_path, ref_order, ref_index_xr, ref_xr, drift_xr, dx_xr
         ) = self.get_ref_path_data()
        # Deduce the derivatives at normal precision
#        dZndz_iter = self.dZndz_iter()
#        dZndc_iter = self.dZndc_iter()
        dZndz_path, dZndc_path = ref_path_derivatives(
            Zn_path, self.dZndz_iter(), self.dZndc_iter()
        )
        if has_status_bar:
            self.set_status("Reference", "completed")

        # Initialize the BLA 
        self.kc = kc = self.ref_point_kc().ravel()  # Make it 1d for numba use
        if kc == 0.: # Precision is way too small, throws an error
            raise RuntimeError(
                "Resolution is too low for this zoom depth. Increase"
                "the reference calculation precicion."
            )
        if self.BLA_params is not None:
            print("Initialise BLA interpolation")
            if has_status_bar:
                self.set_status("Bilin. approx", "running")
            print("** kc", kc)
            self.get_BLA_tree(Zn_path, kc)
            BLA_full_params, A_bla, B_bla, r_bla = self.get_BLA_data()


        # Jitted function used in numba inner-loop
        self._initialize = self.initialize()
        self._iterate = self.iterate()        

        # Launch parallel computing of the inner-loop (Multi-threading with GIL
        # released)
        self.cycles(
            Zn_path, ref_order, ref_index_xr, ref_xr, drift_xr, dx_xr, 
            A_bla, B_bla, r_bla, dZndz_path, dZndc_path,
            chunk_slice=None
        )
        if has_status_bar:
            self.set_status("Tiles", "completed")

        # Export to human-readable format
        if fs.settings.inspect_calc:
            self.inspect_calc()

    @Multithreading_iterator(
        iterable_attr="chunk_slices", iter_kwargs="chunk_slice")
    def cycles(
        self, 
        Zn_path, ref_order, ref_index_xr, ref_xr, drift_xr, dx_xr, 
        A_bla, B_bla, r_bla, dZndz_path, dZndc_path,
        chunk_slice
    ):
        """
        Fast-looping for Julia and Mandelbrot sets computation.

        Parameters:
        -----------
        chunk_slice : 4-tuple int
            The calculation tile
        Zn_path : complex128[]
            The array for the reference point orbit, in low precision
        has_xr : boolean
            True if the Zn_path needs extended range values - when it gets close
            to 0.
        ref_index_xr : int[:]
            indices for the extended range values
        ref_xr : X_range[:]
            The extended range values
        drift_xr : X_range
            vector between image center and reference point : center - ref_pt
        dx_xr :
            width of the image (self.dx)
        P : Xrange_polynomial
            Polynomial from the Series approximation
        kc :
            Scaling c coefficient for the BLA

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

        if ref_order is None:
            ref_order = 2**62 # a quite large int64

        ret_code = numba_cycles_perturb(
            c_pix, Z, U, stop_reason, stop_iter,
            initialize, iterate,
            Zn_path, ref_order, ref_index_xr, ref_xr, drift_xr, dx_xr,
            A_bla, B_bla, r_bla, dZndz_path, dZndc_path,
            self._interrupted
        )

        if ret_code == self.USER_INTERRUPTED:
            print("Interruption signal received")
            return

        if hasattr(self, "_status_wget"):
            self.incr_tiles_status()

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


    def get_BLA_tree(self, Zn_path, kc):
        """
        Check if we already have a BLA stored otherwise compute and stores it in
        a file
        """
        BLA_full_params = self.BLA_params.copy()
        BLA_full_params["FP_params"] = self.FP_params
        BLA_full_params["kc"] = kc
    
        if self.BLA_param_matching(BLA_full_params):
            print("BLA already stored")
            return

        dfdz = self.dfdz()
        eps = self.BLA_params["eps"]
        A_bla, B_bla, r_bla, stages = numba_make_BLA(Zn_path, dfdz, kc, eps)
        BLA_full_params["stages"] = stages
        self.save_BLA(BLA_full_params, A_bla, B_bla, r_bla)
    

    def get_FP_orbit(self, c0=None, newton="cv", order=None,
                     max_newton=None):
        """
        # Check if we have a reference point stored for iref, 
          - otherwise computes and stores it in a file

        newton: ["cv", "None", None]
        """
        if newton == "step":
            raise NotImplementedError("step option not Implemented")

        # Early escape if file exists
        if self.ref_point_matching():
            print("Ref point already stored")
            return

        # no newton if zoom level is low, just pick the critical point.
        if self.dx > fs.settings.newton_zoom_level:
            c0 = self.critical_pt
            self.make_constant_orbit(c0)
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

        print("Launch Newton", newton, " with order: ", order)
        print("max newton iter ", max_newton)
        eps_pixel = self.dx * (1. / self.nx)
        newton_cv, nucleus = self.find_nucleus(
            c0, order, eps_pixel, max_newton=max_newton
        )
        print("first", newton_cv, nucleus)

        # If Newton did not CV, we try to boost the dps / precision
        if not(newton_cv):
            max_attempt = 2
            attempt = 0
            old_dps = mpmath.mp.dps

            while not(newton_cv) and attempt < max_attempt:
                attempt += 1
                mpmath.mp.dps = int(1.25 * mpmath.mp.dps)
                
                print(f"Newton failed, dps incr. {old_dps} -> {mpmath.mp.dps}")
                eps_pixel = self.dx * (1. / self.nx)

                newton_cv, nucleus = self.find_nucleus(
                    c0, order, eps_pixel, max_newton=max_newton
                )
                print("incr", newton_cv, nucleus)

                if not(newton_cv) and (attempt == max_attempt):
                    # Last try, we just release constraint on the cycle order
                    # and consider also divisors.
                    newton_cv, nucleus = self.find_any_nucleus(
                        c0, order, eps_pixel, max_newton=max_newton
                    )
                    print("any", newton_cv, nucleus)

        # Still not CV ? we default to the center of the image
        if not(newton_cv):
            order = None # We cannot wrap the ref point here...
            nucleus = c0

        shift = nucleus - (self.x + self.y * 1j)
        shift_x = shift.real / self.dx
        shift_y = shift.imag / self.dx
        print("Reference nucleus found at shift (expressed in dx units):\n", 
              f"({shift_x}, {shift_y})",
              f"with order {order}"
        )
        self.compute_FP_orbit(nucleus, order)

    def make_constant_orbit(self, crit):
        """
        Basically nothing to 'compute' here, just short-cutting
        """
        print("Short cutting ref pt orbit calc for low res")
        FP_code = self.FP_code
        # Parameters 'max_iter' borrowed from last "@fsutils.calc_options" call
        # max_iter = self.max_iter

        FP_params = {
             "ref_point": crit,
             "dps": mpmath.mp.dps,
             "order": 100,
             "max_iter": self.max_iter,
             "FP_code": FP_code
        }
        # Given the "single reference" implementation the loop will wrap when
        # we reach div_iter - this is probably better not to do it too often
        # Let's just pick a reasonnable figure
        div_iter = 100 # min(100, self.max_iter)
        FP_params["partials"] = {}
        FP_params["xr"] = {}
        FP_params["div_iter"] = div_iter

        Zn_path = np.empty([div_iter + 1], dtype=np.complex128)
        Zn_path[:] = crit

        self.save_ref_point(FP_params, Zn_path)


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

        # If order is defined, we wrap
        ref_orbit_len = max_iter + 1
        if order is not None:
            ref_orbit_len = min(order, ref_orbit_len) # at order + 1, we wrap 
        FP_params["ref_orbit_len"] = ref_orbit_len
        Zn_path = np.empty([ref_orbit_len], dtype=np.complex128)

        print("Computing full precision path with max_iter", max_iter)
        if order is not None:
            print("order known, wraping at", ref_orbit_len)

        i, partial_dict, xr_dict = self.FP_loop(Zn_path, ref_point)
        FP_params["partials"] = partial_dict
        FP_params["xr"] = xr_dict
        FP_params["div_iter"] = i

        self.save_ref_point(FP_params, Zn_path)


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


#==============================================================================
# Numba JIT functions =========================================================
#==============================================================================
Xr_template = fsx.Xrange_array.zeros([1], dtype=np.complex128)
Xr_float_template = fsx.Xrange_array.zeros([1], dtype=np.float64)
USER_INTERRUPTED = 1

@numba.njit(nogil=True)
def numba_cycles_perturb(
    c_pix, Z, U, stop_reason, stop_iter,
    initialize, iterate, 
    Zn_path, ref_order, ref_index_xr, ref_xr, drift_xr, dx_xr,
    A_bla, B_bla, r_bla, dZndz_path, dZndc_path,
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
    _, npts = Z.shape
    
    for ipt in range(npts):
        cpt, c_xr = ref_path_c_from_pix(c_pix[ipt], dx_xr, drift_xr)
        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        stop_pt = stop_reason[:, ipt]

        initialize(Zpt, Upt, c_xr)

        n_iter = iterate(
            cpt, c_xr, Zpt, Upt, stop_pt,
            Zn_path, ref_order, ref_index_xr, ref_xr,
            A_bla, B_bla, r_bla, dZndz_path, dZndc_path
        )
        stop_iter[0, ipt] = n_iter
        stop_reason[0, ipt] = stop_pt[0]

        if _interrupted[0]:
            return USER_INTERRUPTED
    return 0

@numba.njit
def numba_make_BLA(ref_path, dfdz, kc, eps):
    """
    Generates a BVA tree with
    - bilinear approximation coefficients A and B
    - validaty radius
        z_n+2**stg = f**stg(z_n, c) with |c| < r_stg_n is approximated by 
        z_n+2**stg = A_stg_n * z_n + B_stg_n * c
    """
    # number of needed "stages" (ref_orbit_len).bit_length()
    kc_std = fsxn.to_standard(kc[0])
    ref_orbit_len = ref_path.shape[0]
    print("ref_orbit_len", ref_orbit_len)

    A = np.zeros((2 * ref_orbit_len,), dtype=np.complex128)
    B = np.zeros((2 * ref_orbit_len,), dtype=np.complex128)
    r = np.zeros((2 * ref_orbit_len,), dtype=np.float64)
    stages = init_BLA(A, B, r, ref_path, dfdz, kc_std, eps)
    
    return A, B, r, stages

@numba.njit
def init_BLA(A, B, r, Zn_path, dfdz, kc_std, eps):
    """
    Initialize BLA tree at stg 0
    """
    ref_orbit_len = Zn_path.shape[0] #min(ref_order, ref_div_iter) # at order + 1, we wrap
     
    for i in range(ref_orbit_len):
        i_0 = 2 * i # BLA index for (i, 0)
        A[i_0] = dfdz(Zn_path[i])
        B[i_0] = 1.
        # We use the following criteria :
        # |Z + z| shall stay *far* from O or discontinuity of F', for each c
        # For std Mandelbrot it means from z = 0
        # |zn| << |Zn|
        # For Burning ship x = 0 or y = 0
        # |zn| << |Xn|,  |zn| << |Yn|
        # We could additionnally consider a criterian based on hessian
        # |z| < A e / h where h Hessian - not useful (redundant)
        # for Mandelbrot & al.
        mZ = np.abs(Zn_path[i]) # for Burning ship & al use rather:
                                # mZ = min(Zn_path[i].real, Zn_path[i].imag);
        r[i_0] =  mZ * eps # max(0., (mZ - mB * kc_std) / (mA + 1.) * eps) # Note that
        # we need to consider due to orbit wrapping

    # Now the combine step
    # number of needed "stages" (ref_orbit_len).bit_length()
    stages = stages_bla(ref_orbit_len)
    for stg in range(1, stages):
        combine_BLA(A, B, r, kc_std, stg, ref_orbit_len, eps)
    return stages

@numba.njit
def stages_bla(ref_orbit_len):
    """
    number of needed "stages" (ref_orbit_len).bit_length()
    """
    return int(np.ceil(np.log2(ref_orbit_len)))

@numba.njit
def combine_BLA(A, B, r, kc_std, stg, ref_orbit_len, eps):
    """ Populate successive stages of a BLA tree
    A_bla, B_bla, r_bla : data of the BLA tree
    kc : majorant of |c|
    stg : stage of the tree that is populated by merging (stg - 1) items
    ref_orbit_len : the len for the reference orbit
    """
    # Combine all BVA at stage stg-1 to make stage stg with stg > 0
    step = (1 << stg)

    for i in range(0, ref_orbit_len - step, step):
        ii = i + (step // 2)
        # If ref_orbit_len is not a power of 2, we might get outside the array
        if ii >= ref_orbit_len:
            break
        index1 = BLA_index(i, stg - 1)
        index2 = BLA_index(ii, stg - 1)
        index_res = BLA_index(i, stg)

        # Combines linear approximations
        A[index_res] = A[index1] * A[index2]
        B[index_res] = (B[index1] * A[index2] + B[index2])

        # Combines the validity radii
        r1 = r[index1]
        r2 = r[index2]
        # r1 is a direct criteria however for r2 we need to go 'backw the flow'
        # z0 -> z1 -> z2 with z1 = A1 z0 + B1 c, |z1| < r2
        mA1 = np.abs(A[index1])
        mB1 = np.abs(B[index1])
        r2_backw = max(0., (r2 - mB1 * kc_std) / (mA1 + 1.))
        r[index_res] = min(r1, r2_backw)

@numba.njit
def BLA_index(i, stg):
    """
    Return the indices in BVA table for this iteration and stage
    this is the jump from i to j = i + (1 << stg)
    """
    # ex : 16 nb, st max = 4 (st in [0, 4])
    # 0  ->  0,  1,  3,  7, 15   [31 first missing]
    # 1  ->  2
    # 2  ->  4,  5
    # 3  ->  6
    # 4  ->  8,  9, 11
    # 5  -> 10
    # 6  -> 12, 13
    # 7  -> 14
    # 8  -> 16, 17, 19, 23
    # 9  -> 18
    # 10 -> 20, 21
    # 11 -> 22
    # 12 -> 24, 25, 27 
    # 13 -> 26
    # 14 -> 28, 29
    # 15 -> 30
    # index = (2 * i) + ((1 << stg) - 1) = i + j - 1
    return (2 * i) + ((1 << stg) - 1)


@numba.njit
def ref_BLA_get(A_bla, B_bla, r_bla, stages_bla, zn, n_iter):
    """
    Paramters:
    ----------
    A_bla, B_bla, r_bla: arrays
        Bilinear approx tree
    zn :
        The current value of dz
    n_iter :
        The current iteration for ref pt

    Returns:
    --------
    Ai, Bi
        Ai and Bi the applicable 
     step
         The interation "jump" provided by this linear interpolation
    """
    NULL = A_bla[0], B_bla[0], 0

    _iter = n_iter
    for stages in range(stages_bla):
        if _iter & 1:
            break
        _iter = _iter >> 1
    assert (n_iter % (1 << stages)) == 0

    invalid_step = (len(r_bla) // 2) - n_iter # The first invalid step

    # numba version of reversed(range(stages_bla)):
    for stg in range(stages, -1, -1):
        step = (1 << stg)
        if step >= invalid_step:
            continue
        index_bla = BLA_index(n_iter, stg)
        assert(index_bla < len(r_bla))
        r = r_bla[index_bla]
        # Important to use strict comparisons here: to rule out underflow
        if ((abs(zn.real) < r) and (abs(zn.imag) < r)):
            return A_bla[index_bla], B_bla[index_bla], step
    return NULL # No BLA applicable


@numba.njit
def need_xr(x_std):
    """
    True if norm L-inf of std is lower than xrange_zoom_level
    """
    return (
        (abs(x_std.real) < fs.settings.xrange_zoom_level)
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
    This is ONLY a final stage translation + homothetie, to avoid use of 
    Xrange when storing c value (as they are pre-scaled by dx)

    Parameters
    ----------
    pix :  complex
        pixel location in fraction of dx
        
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
    with increasing idx.

    idx :
        index requested
    (prev_idx, curr_xr) :
        couple returned from last call, last index requested + next xr target
        Contract : curr_xr the smallest integer that verify :
            prev_idx <= ref_index_xr[curr_xr]
            or curr_xr = ref_index_xr.size (No more xr)
            
    out_is_xr, out_xr
        in-place modification at out_index (out_index usually 0)

    Returns
    -------
    val
        standard value: ref_path[idx]
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

@numba.njit
def ref_path_derivatives(Zn_path, dZndz_iter, dZndc_iter):
    """
    Computes the needed derivatives from reference path
    Empty array returned if not needed
    """
    orbit_len = Zn_path.shape[0]
    if dZndz_iter is not None:
        dZndz_path = np.empty((orbit_len,), dtype=np.complex128)
        dZndz_path[0] = 0.
        for i in range(orbit_len - 1):
            dZndz_path[i + 1] = dZndz_iter(Zn_path[i], dZndz_path[i])
    else:
        print("** empty dZndz_path")
        dZndz_path = np.empty((0,), dtype=np.complex128)

    if dZndc_iter is not None:
        dZndc_path = np.empty((orbit_len,), dtype=np.complex128)
        dZndc_path[0] = 0.
        for i in range(orbit_len - 1):
            dZndc_path[i + 1] = dZndc_iter(Zn_path[i], dZndc_path[i])
    else:
        print("** empty dZndc_path")
        dZndc_path = np.empty((0,), dtype=np.complex128)
        
    print("*** dZndc_path", dZndc_path)

    return dZndz_path, dZndc_path
        