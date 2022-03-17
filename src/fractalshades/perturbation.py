# -*- coding: utf-8 -*-
import os
import pickle
import warnings

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
Base class for escape-time fractals calculations implementing the 
perturbation technique, with an iteration matching:

z_(n+1) = f(z_n) + c, critial point at 0

Derived classes should implement the actual function f

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
             antialiasing: bool=False,
             has_skew: bool=False,
             skew_00: float=1.,
             skew_01: float=0.,
             skew_10: float=0.,
             skew_11: float=1.
        ):
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
        has_skew : bool
            If True, unskew the view base on skew coefficients skew_ii
        """
        mpmath.mp.dps = precision # in base 10 digit 
        
        # In case the user inputs were strings, we override with mpmath scalars
        self.x = mpmath.mpf(x)
        self.y = mpmath.mpf(y)
        self.dx = mpmath.mpf(dx)

        # Stores the skew matrix
        self._skew = None
        if has_skew:
            self._skew = np.array(
                ((skew_00, skew_01), (skew_10, skew_11)), dtype=np.float64
            )

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
            if self.holomorphic:
                if (key == self.FP_code):
                    Zn_path = self.Zn_path
                    full_Z[val, :] += Zn_path[U[0, :]]  # U[0] is ref_cycle_iter
            else:
                if key in self.FP_code:
                    XnYn_path = self.Zn_path.view(dtype=np.float64)
                    key_index = (self.FP_code).index(key)
                    full_Z[val, :] += XnYn_path[U[0, :] * 2 + key_index]  # U[0] is ref_cycle_iter

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
                 abs(ref - corner_c), abs(ref - corner_d)) * 1.1

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
        FP_params, Zn_path = self.reload_ref_point()
        self._Zn_path = Zn_path
        return Zn_path


    def get_path_data(self):
        """ Builds a Zn_path data tuple from FP_params and Zn_path
        This object will be used in numba jitted functions
        """
        FP_params = self.FP_params
        Zn_path = self.Zn_path
        
        ref_xr_python = FP_params["xr"]
        has_xr = (len(ref_xr_python) > 0)
        ref_order = FP_params["order"]
        ref_div_iter = FP_params["div_iter"] # The first invalid iter
        # ref_div_iter should be ref div iter of the FP, only if it is div.
        # Otherwise, the max_iter from calc param
        if ref_order is not None: # The reference orbit is a cycle
            ref_div_iter = self.max_iter + 1

        dx_xr = fsx.mpf_to_Xrange(self.dx, dtype=self.float_type).ravel()
        ref_index_xr = np.empty([len(ref_xr_python)], dtype=np.int32)

        if self.holomorphic:
            # Complex distance between image center and ref point 
            drift_xr = fsx.mpc_to_Xrange(
                (self.x + 1j * self.y) - FP_params["ref_point"],
                dtype=np.complex128
            ).ravel()
    
            # Build 2 arrays to avoid using a dict in numba
            # /!\ ref_xr at least len 1 to ensure typing as complex
            ref_xr = fsx.Xrange_array([0j] * max(len(ref_xr_python), 1))
            for i, xr_index in enumerate(ref_xr_python.keys()):
                ref_index_xr[i] = xr_index
                ref_xr[i] = ref_xr_python[xr_index][0]
            return (Zn_path, has_xr, ref_index_xr, ref_xr,
                    ref_div_iter, ref_order, drift_xr, dx_xr)

        else:
            # Complex distance between image center and ref point 
            refp_ptx = (FP_params["ref_point"]).real
            refp_pty = (FP_params["ref_point"]).imag
            driftx_xr = fsx.mpf_to_Xrange(self.x - refp_ptx, dtype=np.float64
                ).ravel()
            drifty_xr = fsx.mpf_to_Xrange(self.y - refp_pty, dtype=np.float64
                ).ravel()

            # Build 2 arrays to avoid using a dict in numba
            # /!\ ref_xr at least len 1 to ensure typing as complex
            refx_xr = fsx.Xrange_array([0.] * max(len(ref_xr_python), 1))
            refy_xr = fsx.Xrange_array([0.] * max(len(ref_xr_python), 1))
            print("init refx_xr", refx_xr.shape, refx_xr.dtype)
            for i, xr_index in enumerate(ref_xr_python.keys()):
                ref_index_xr[i] = xr_index
                tmpx, tmpy = ref_xr_python[xr_index]
                refx_xr[i] = tmpx[0]
                refy_xr[i] = tmpy[0]
            return (Zn_path, has_xr, ref_index_xr, refx_xr, refy_xr,
                    ref_div_iter, ref_order, driftx_xr, drifty_xr,
                    dx_xr)
            



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
        print("ref_path, shape: ", Zn_path.shape, Zn_path.dtype) 
        print(Zn_path)
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
        ref_xr = None
        drift_xr = None
        refx_xr = None
        refy_xr = None
        driftx_xr = None
        drifty_xr = None
        if self.holomorphic:
            (Zn_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
             drift_xr, dx_xr) = self.get_path_data()
        else:
            (Zn_path, has_xr, ref_index_xr, refx_xr, refy_xr,
             ref_div_iter, ref_order, driftx_xr, drifty_xr,
             dx_xr) = self.get_path_data()

        if has_status_bar:
            self.set_status("Reference", "completed")
            
        # Initialize the derived ref path - with a scale coefficient self.dx
        # Note: only useful if the analytical derivatives are computed
        calc_deriv = None
        dZndc_path = None
        dXnda_path = None
        dXndb_path = None
        dYnda_path = None
        dYndb_path = None

        if self.holomorphic:
            calc_deriv = self.calc_dzndc
        else:
            calc_deriv = self.calc_hessian

        if calc_deriv is not None:
            dx_xr = fsx.mpf_to_Xrange(self.dx, dtype=self.float_type).ravel()
            casted_ref_order = ref_order # 
            if ref_order is None:
                casted_ref_order = (1 << 62) # a quite large int64
            xr_detect_activated = self.xr_detect_activated

            if self.holomorphic:
                dZndc_path, dZndc_xr_path = numba_dZndc_path(
                    Zn_path, has_xr, ref_index_xr, ref_xr,
                    ref_div_iter, casted_ref_order,
                    self.dfdz, dx_xr, xr_detect_activated
                )
                if xr_detect_activated:
                    dZndc_path = dZndc_xr_path

            else:
                (
                    dXnda_path, dXndb_path, dYnda_path, dYndb_path,
                    dXnda_xr_path, dXndb_xr_path, dYnda_xr_path, dYndb_xr_path
                ) = numba_dZndc_path_BS(
                    Zn_path, has_xr, ref_index_xr, refx_xr, refy_xr,
                    ref_div_iter, casted_ref_order,
                    self.dfxdx, self.dfxdy, self.dfydx, self.dfydy,
                    dx_xr, xr_detect_activated
                )
                if xr_detect_activated:
                    dXnda_path = dXnda_xr_path
                    dXndb_path = dXndb_xr_path
                    dYnda_path = dYnda_xr_path
                    dYndb_path = dYndb_xr_path

        self.kc = kc = self.ref_point_kc().ravel()  # Make it 1d for numba use
        if kc == 0.:
            raise RuntimeError(
                "Resolution is too low for this zoom depth. Try to increase"
                "the reference calculation precicion."
            )

        # Initialise SA interpolation
        if not(hasattr(self, "SA_params")) or (self.SA_params is None):
            n_iter = 0
            P = None
            P_err = None
        else:
            warnings.warn('SA is obsolete, use BLA instead',
                          DeprecationWarning)
            self.get_SA(Zn_path, has_xr, ref_index_xr, ref_xr, ref_div_iter,
                        ref_order)
            P, n_iter, P_err = self.get_SA_data()

        # Initialize BLA interpolation
        if self.BLA_params is None:
            M_bla = None
            r_bla = None
            bla_len = None
            stages_bla = None
        else:
            print("Initialise BLA interpolation")
            if has_status_bar:
                self.set_status("Bilin. approx", "running")
            eps = self.BLA_params["eps"]
            print("** eps", eps)
            M_bla, r_bla, bla_len, stages_bla = self.get_BLA_tree(
                    Zn_path, eps)
            if has_status_bar:
                self.set_status("Bilin. approx", "completed")


        # Jitted function used in numba inner-loop
        self._initialize = self.initialize()
        self._iterate = self.iterate()        

        # Launch parallel computing of the inner-loop (Multi-threading with GIL
        # released)
        self.cycles(
            Zn_path, dZndc_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
            has_xr, ref_index_xr, ref_xr, refx_xr, refy_xr,
            ref_div_iter, ref_order,
            drift_xr, driftx_xr, drifty_xr, dx_xr, 
            P, kc, n_iter, M_bla, r_bla, bla_len, stages_bla,
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
        Zn_path, dZndc_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
        has_xr, ref_index_xr, ref_xr, refx_xr, refy_xr,
        ref_div_iter, ref_order,
        drift_xr, driftx_xr, drifty_xr, dx_xr, 
        P, kc, n_iter, M_bla, r_bla, bla_len, stages_bla,
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
            if hasattr(self, "_status_wget"):
                self.incr_tiles_status()
            return

        (c_pix, Z, U, stop_reason, stop_iter
         ) = self.init_cycling_arrays(chunk_slice)

        initialize = self._initialize
        iterate = self._iterate

        if ref_order is None:
            ref_order = (1 << 62) # a quite large int64

        if self.holomorphic:
            ret_code = numba_cycles_perturb(
                c_pix, Z, U, stop_reason, stop_iter,
                initialize, iterate,
                Zn_path, dZndc_path,
                has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
                drift_xr, dx_xr,
                P, kc, n_iter, M_bla, r_bla, bla_len, stages_bla,
                self._interrupted
            )
        else:
            ret_code = numba_cycles_perturb_BS(
                c_pix, Z, U, stop_reason, stop_iter,
                initialize, iterate,
                Zn_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
                has_xr, ref_index_xr, refx_xr, refy_xr, ref_div_iter, ref_order,
                driftx_xr, drifty_xr, dx_xr,
                P, kc, n_iter, M_bla, r_bla, bla_len, stages_bla,
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


    def get_SA(self, Zn_path, has_xr, ref_index_xr, ref_xr, ref_div_iter,
               ref_order):
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

            if ref_order is None:
                ref_order = 2**62 # a quite large int64

            P, n_iter, P_err = numba_SA_run(
                SA_loop, 
                Zn_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
                kc, SA_cutdeg, SA_err_sq, SA_stop
            )
            self._SA_data = (P, n_iter, P_err)
            self.save_SA(FP_params, SA_params, kc, P, n_iter, P_err)

    def get_BLA_tree(self, Zn_path, eps):
        """
        Return
        ------
        M_bla, r_bla, stages_bla
        """
        kc = self.kc
        print("mabe BLA tree")

        if self.holomorphic:
            dfdz = self.dfdz
            return numba_make_BLA(Zn_path, dfdz, kc, eps)
        else:
            dfxdx = self.dfxdx
            dfxdy = self.dfxdy
            dfydx = self.dfydx
            dfydy = self.dfydy
            return numba_make_BLA_BS(
                Zn_path, dfxdx, dfxdy, dfydx, dfydy, kc, eps
            )

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

        print("Launch Newton", newton, " with order: ", order)
        print("max newton iter ", max_newton)
        eps_pixel = self.dx * (1. / self.nx)
        try:
            newton_cv, nucleus = self.find_nucleus(
                c0, order, eps_pixel, max_newton=max_newton
            )
        except NotImplementedError:
            print("*** Default to find any")
            newton_cv, nucleus = self.find_any_nucleus(
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

                try:
                    no_div = True
                    newton_cv, nucleus = self.find_nucleus(
                        c0, order, eps_pixel, max_newton=max_newton
                    )
                except NotImplementedError:
                    no_div = False
                    print("*** Default to find any")
                    newton_cv, nucleus = self.find_any_nucleus(
                        c0, order, eps_pixel, max_newton=max_newton
                    )
                     
                    
                print("incr", newton_cv, nucleus)

                if no_div and not(newton_cv) and (attempt == max_attempt):
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

    def compute_critical_orbit(self, crit):
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
        print()

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
# -0.67487014+1.15006259j
#==============================================================================
# GUI : "interactive options"
#==============================================================================
    def coords(self, x, y, pix, dps):
        """ x, y : coordinates of the event """
        x_str = str(x)
        y_str = str(y)
        res_str = f"""
coords = {{
    "x": "{x_str}"
    "y": "{y_str}"
}}
"""
        return res_str

    def ball_method_order(self, x, y, pix, dps,
                          maxiter: int=100000,
                          radius_pixels: int=25):
        """ x, y : coordinates of the event """
        c = x + 1j * y
        radius = pix * radius_pixels
        M_divergence = 1.e3
        order = self._ball_method(c, radius, maxiter, M_divergence)

        x_str = str(x)
        y_str = str(y)
        radius_str = str(radius)
        res_str = f"""
ball_order = {{
    "x": "{x_str}",
    "y": "{y_str}",
    "maxiter": {maxiter},
    "radius_pixels": {radius_pixels},
    "radius": "{radius_str}",
    "M_divergence": {M_divergence},
    "order": {order}
}}
"""
        return res_str


    def _newton_search(self, x, y, pix, dps,
                          maxiter: int=100000,
                          radius_pixels: int=3):
        """ x, y : coordinates of the event """
        c = x + 1j * y

        radius = pix * radius_pixels
        radius_str = str(radius)
        M_divergence = 1.e3
        order = self._ball_method(c, radius, maxiter, M_divergence)

        newton_cv = False
        max_attempt = 2
        attempt = 0
        while not(newton_cv) and attempt < max_attempt:
            if order is None:
                break
            attempt += 1
            dps = int(1.5 * dps)
            print("Newton, dps boost to: ", dps)
            with mpmath.workdps(dps):
                try:
                    newton_cv, c_newton = self.find_nucleus(
                        c, order, pix, max_newton=None, eps_cv=None
                    )
                except NotImplementedError:
                    newton_cv, c_newton = self.find_any_nucleus(
                        c, order, pix, max_newton=None, eps_cv=None
                    )
                if newton_cv:
                    xn_str = str(c_newton.real)
                    yn_str = str(c_newton.imag)

        if newton_cv:
            size_estimates = self._nucleus_size_estimate(
                c_newton, order
            )
        else:
            size_estimates = None
            xn_str = ""
            yn_str = ""

        x_str = str(x)
        y_str = str(y)
        
        return (
            x_str, y_str, maxiter, radius_pixels, radius_str, dps, order,
            xn_str, yn_str, size_estimates
        )


    def newton_search(self, x, y, pix, dps, maxiter: int=100000,
                      radius_pixels: int=3):
        """ x, y : coordinates of the event """
        (
            x_str, y_str, maxiter, radius_pixels, radius_str, dps, order,
            xn_str, yn_str, size_estimates
        ) = self._newton_search(
            x, y, pix, dps, maxiter, radius_pixels
        )
        if size_estimates is not None:
            (nucleus_size, julia_size) = size_estimates
        else:
            nucleus_size = None
            julia_size = None

        res_str = f"""
newton_search = {{
    "x_start": "{x_str}",
    "y_start": "{y_str}",
    "maxiter": {maxiter},
    "radius_pixels": {radius_pixels},
    "radius": "{radius_str}",
    "calculation dps": {dps}
    "order": {order}
    "x_nucleus": "{xn_str}",
    "y_nucleus": "{yn_str}",
    "nucleus_size": "{nucleus_size}",
    "julia_size": "{julia_size}",
}}
"""
        return res_str

#==============================================================================
# Numba JIT functions
#==============================================================================
Xr_template = fsx.Xrange_array.zeros([1], dtype=np.complex128)
Xr_float_template = fsx.Xrange_array.zeros([1], dtype=np.float64)
USER_INTERRUPTED = 1
STG_COMPRESSED = fs.settings.BLA_compression
STG_SKIP_MASK = (1 << STG_COMPRESSED) - 1

@numba.njit(nogil=True)
def numba_cycles_perturb(
    c_pix, Z, U, stop_reason, stop_iter,
    initialize, iterate,
    Zn_path, dZndc_path,
    has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
    drift_xr, dx_xr,
    P, kc, n_iter_init, M_bla, r_bla, bla_len, stages_bla,
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

    nz, npts = Z.shape
    Z_xr = Xr_template.repeat(nz)
    Z_xr_trigger = np.ones((nz,), dtype=np.bool_)

#    ref_orbit_len = Zn_path.shape[0]
#    first_invalid_index = min(ref_orbit_len, ref_div_iter, ref_order)
#    print("****first invalid", first_invalid_index, "min of", ref_orbit_len, ref_div_iter, ref_order)


    for ipt in range(npts):

        refpath_ptr = np.zeros((2,), dtype=np.int32)
        out_is_xr = np.zeros((2,), dtype=numba.bool_)
        out_xr = Xr_template.repeat(2)

        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        cpt, c_xr = ref_path_c_from_pix(c_pix[ipt], dx_xr, drift_xr)
        stop_pt = stop_reason[:, ipt]

        initialize(c_xr, Zpt, Z_xr, Z_xr_trigger, Upt, P, kc, dx_xr,
                   n_iter_init)

        n_iter = iterate(
            cpt, c_xr, Zpt, Z_xr, Z_xr_trigger, Upt, stop_pt, n_iter_init,
            Zn_path, dZndc_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
            refpath_ptr, out_is_xr, out_xr, M_bla, r_bla, bla_len, stages_bla
        )
        stop_iter[0, ipt] = n_iter
        stop_reason[0, ipt] = stop_pt[0]

        if _interrupted[0]:
            return USER_INTERRUPTED
        
#        print("###############################################################")
        
    return 0


def numba_initialize(zn, dzndz, dzndc):
    @numba.njit
    def numba_init_impl(c_xr, Z, Z_xr, Z_xr_trigger, U, P, kc, dx_xr,
                        n_iter_init):
        """
        ... mostly the SA
        Initialize 'in place' at n_iter_init :
            Z[zn], Z[dzndz], Z[dzndc]
        """
        if P is None:
            Z_xr[zn] = fsxn.to_Xrange_scalar(Z[zn])
            if dzndc != -1:
                Z_xr[dzndc] = fsxn.to_Xrange_scalar(Z[dzndc])
        else:
            # Apply the  Series approximation step
            U[0] = n_iter_init
            c_scaled = c_xr / kc[0]
            Z_xr[zn] = P.__call__(c_scaled)
            Z[zn] = fsxn.to_standard(Z_xr[zn])

            if dzndc != -1:
                P_deriv = P.deriv()
                deriv_scale =  dx_xr[0] / kc[0]
                Z[dzndc] = fsxn.to_standard(
                    P_deriv.__call__(c_scaled) * deriv_scale
                )

        if (dzndz != -1):
            Z[dzndz] = 1.
    return numba_init_impl


# Defines iterate via a function factory - jitted implementation
def numba_iterate(
        M_divergence_sq, max_iter, reason_max_iter, reason_M_divergence,
        epsilon_stationnary_sq, interior_detect_activated, reason_stationnary,
        SA_activated, xr_detect_activated, BLA_activated,
        calc_dzndc,
        zn, dzndz, dzndc,
        p_iter_zn, p_iter_dzndz, p_iter_dzndc
):

    @numba.njit
    def numba_impl(
        c, c_xr, Z, Z_xr, Z_xr_trigger, U, stop, n_iter,
        Zn_path, dZndc_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
        refpath_ptr, out_is_xr, out_xr, M_bla, r_bla, bla_len, stages_bla
    ):
        """
        Parameters
        ----------
        c, c_xr: c and it "Xrange" counterparts
        Z, Z_xr: idem for result vector Z
        Z_xr_trigger : bolean, activated when Z_xr need to be used
        """
        # SA skipped - wrapped iteration if we reach the cycle order 
        w_iter = n_iter
        if w_iter >= ref_order:
            w_iter = w_iter % ref_order
        # We know that :
        # ref_orbit_len = max_iter + 1 >= ref_div_iter
        # if order is not None:
        #    ref_orbit_len = min(order, ref_orbit_len)
        ref_orbit_len = Zn_path.shape[0]
        first_invalid_index = min(ref_orbit_len, ref_div_iter, ref_order)
        M_out = np.empty((2,), dtype=np.complex128)

        while True:
            #==========================================================
            # Try a BLA_step
            if BLA_activated and (w_iter & STG_SKIP_MASK) == 0:
                #       [    A    0     0]
                #  M =  [    0    A     B] 
                #       [    0    0     1]
                #
                #       [dzndc]
                #  Zn = [   zn]
                #       [    c]
                #
                #  Z_(n+1) = M * Zn
                #
                step = ref_BLA_get(
                    M_bla, r_bla, bla_len, stages_bla, Z[zn], w_iter,
                    first_invalid_index, M_out, True
                )
                if step != 0:
                    n_iter += step
                    w_iter = (w_iter + step) % ref_order
                    if xr_detect_activated:
                        Z_xr[zn] = M_out[0] * Z_xr[zn] + M_out[1] * c_xr
                        # /!\ keep this, needed for next BLA step
                        Z[zn] = fsxn.to_standard(Z_xr[zn])
                        if calc_dzndc:
                            Z_xr[dzndc] = M_out[0] * Z_xr[dzndc]
                    else:
                        # just the usual BLA step
                        Z[zn] = M_out[0] * Z[zn] + M_out[1] * c
                        if calc_dzndc:
                            Z[dzndc] = M_out[0] * Z[dzndc]
                    continue

            #==================================================================
            # BLA failed, launching a full perturbation iteration
            n_iter += 1 # the indice we are going to compute now
            # Load reference point value @ w_iter
            # refpath_ptr = [prev_idx, curr_xr]
            if xr_detect_activated:
                ref_zn = ref_path_get(
                    Zn_path, w_iter,
                    has_xr, ref_index_xr, ref_xr, refpath_ptr,
                    out_is_xr, out_xr, 0
                )
                ref_zn_xr = ensure_xr(ref_zn, out_xr[0], out_is_xr[0])
            else:
                ref_zn = Zn_path[w_iter]

            #==================================================================
            # Pertubation iter block
            #------------------------------------------------------------------
            # dzndc subblock
            if calc_dzndc:
                ref_dzndc = dZndc_path[w_iter] # This may be Xrange
                if xr_detect_activated:
                    p_iter_dzndc(Z_xr, ref_zn_xr, ref_dzndc)
                else:
                    p_iter_dzndc(Z, ref_zn, ref_dzndc)

            #------------------------------------------------------------------
            # Interior detection - Used only at low zoom level
            if interior_detect_activated and (n_iter > 1):
                p_iter_dzndz(Z) # 2. * (Z[zn] * Z[dzndz])

            #------------------------------------------------------------------
            # zn subblok
            if xr_detect_activated:
                p_iter_zn(Z_xr, ref_zn_xr, c_xr)# in place mod
                # std is used for div condition 
                Z[zn] = fsxn.to_standard(Z_xr[zn])
            else:
                p_iter_zn(Z, ref_zn, c)

            #==================================================================
            # Stopping condition: maximum iter reached
            if n_iter >= max_iter:
                stop[0] = reason_max_iter
                break

            #==================================================================
            # Stopping condition: Interior points detection
            if interior_detect_activated:
                bool_stationnary = (
                    Z[dzndz].real ** 2 + Z[dzndz].imag ** 2
                        < epsilon_stationnary_sq)
                if bool_stationnary:
                    stop[0] = reason_stationnary
                    break

            #==================================================================
            # Stopping condition: divergence
            # ZZ = "Total" z + dz
            w_iter += 1
            if w_iter >= ref_order:
                w_iter = w_iter % ref_order

            if xr_detect_activated:
                ref_zn_next = fs.perturbation.ref_path_get(
                    Zn_path, w_iter,
                    has_xr, ref_index_xr, ref_xr, refpath_ptr,
                    out_is_xr, out_xr, 0
                )
            else:
                ref_zn_next = Zn_path[w_iter]

            # div condition computation with std only
            ZZ = Z[zn] + ref_zn_next
            full_sq_norm = ZZ.real ** 2 + ZZ.imag ** 2

            # Flagged as 'diverging'
            bool_infty = (full_sq_norm > M_divergence_sq)
            if bool_infty:
                stop[0] = reason_M_divergence
                break

            #==================================================================
            # Glitch correction - reference point diverging
            if (w_iter >= ref_div_iter - 1):
                # Rebasing - we are already big no underflow risk
                Z[zn] = ZZ
                if xr_detect_activated:
                    Z_xr[zn] = fsxn.to_Xrange_scalar(ZZ)

#                if calc_dzndc:
#                    Z[dzndc] = Z[dzndc] + dZndc_path[w_iter]
                w_iter = 0
                continue

            #==================================================================
            # Glitch correction - "dynamic glitch"
            bool_dyn_rebase = (
                (abs(ZZ.real) <= abs(Z[zn].real))
                and (abs(ZZ.imag) <= abs(Z[zn].imag))
            )
            if bool_dyn_rebase:
                if xr_detect_activated:
                    # Can we *really* rebase ??
                    # Note: if Z[zn] underflows we might miss a rebase
                    # So we cast everything to xr
                    Z_xrn = Z_xr[zn]
                    if out_is_xr[0]:
                        # Reference underflows, use available xr ref
                        ZZ_xr = Z_xrn + out_xr[0]
                    else:
                        ZZ_xr = Z_xrn + ref_zn_next

                    bool_dyn_rebase_xr = (
                        fsxn.extended_abs2(ZZ_xr)
                        <= fsxn.extended_abs2(Z_xrn)   
                    )
                    if bool_dyn_rebase_xr:
                        
                        Z_xr[zn] = ZZ_xr
                        # /!\ keep this, needed for next BLA step - TODO: for BS
                        Z[zn] = fsxn.to_standard(ZZ_xr)
                        if calc_dzndc:
                            Z_xr[dzndc] = Z_xr[dzndc] + dZndc_path[w_iter]
                        w_iter = 0
                        continue
                else:
                    # No risk of underflow - safe to rebase
                    Z[zn] = ZZ
                    if calc_dzndc:
                        Z[dzndc] = Z[dzndc] + dZndc_path[w_iter]
                    w_iter = 0
                    continue

        # End of iterations for this point
        U[0] = w_iter
        if xr_detect_activated:
            Z[zn] = fsxn.to_standard(Z_xr[zn])
            Z[dzndc] = fsxn.to_standard(Z_xr[dzndc])
#        print(n_iter, w_iter, "--> exit with zn dzndc", Z[zn], Z[dzndc])
        return n_iter

    return numba_impl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Non-holomorphic perturbation iterations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@numba.njit(nogil=True)
def numba_cycles_perturb_BS(
    c_pix, Z, U, stop_reason, stop_iter,
    initialize, iterate,
    Zn_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
    has_xr, ref_index_xr, refx_xr, refy_xr, ref_div_iter, ref_order,
    driftx_xr, drifty_xr, dx_xr,
    P, kc, n_iter_init, M_bla, r_bla, bla_len, stages_bla,
    _interrupted
):
    print("in numba_cycles_perturb_BS")


    nz, npts = Z.shape
    Z_xr = Xr_float_template.repeat(nz)
    Z_xr_trigger = np.ones((nz,), dtype=np.bool_)

    for ipt in range(npts):

        refpath_ptr = np.zeros((2,), dtype=np.int32)
        out_is_xr = np.zeros((2,), dtype=numba.bool_)
        out_xr = Xr_float_template.repeat(4)

        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        apt, bpt, a_xr, b_xr = ref_path_c_from_pix_BS(
            c_pix[ipt], dx_xr, driftx_xr, drifty_xr
        )
        stop_pt = stop_reason[:, ipt]

        initialize(Zpt, Z_xr)

        n_iter = iterate(
            apt, bpt, a_xr, b_xr, Zpt, Z_xr, Z_xr_trigger,
            Upt, stop_pt, n_iter_init,
            Zn_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
            has_xr, ref_index_xr, refx_xr, refy_xr, ref_div_iter, ref_order,
            refpath_ptr, out_is_xr, out_xr, M_bla, r_bla, bla_len, stages_bla
        )
        # print('n_iter', n_iter)
        stop_iter[0, ipt] = n_iter#n_iterv- debug
        stop_reason[0, ipt] = stop_pt[0]
        
        # print("after iterate", ipt, npts)

        if _interrupted[0]:
            return USER_INTERRUPTED
    return 0


def numba_initialize_BS(xn, yn, dxnda, dxndb, dynda, dyndb):
    @numba.njit
    def numba_init_impl(Z, Z_xr):
        """
        Only : initialize the Xrange (no SA here)
        """
        for key in (xn, yn, dxnda, dxndb, dynda, dyndb):
            if key!= -1:
                Z_xr[key] = fsxn.to_Xrange_scalar(Z[key])
    return numba_init_impl


# Defines iterate for non-holomorphic function via a function factory
# jitted implementation
def numba_iterate_BS(
    M_divergence_sq, max_iter, reason_max_iter, reason_M_divergence,
    xr_detect_activated, BLA_activated,
    calc_hessian,
    xn, yn, dxnda, dxndb, dynda, dyndb,
    p_iter_zn, p_iter_hessian
):

    @numba.njit
    def numba_impl(
        a, b, a_xr, b_xr, Z, Z_xr, Z_xr_trigger,
        U, stop, n_iter,
        Zn_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
        has_xr, ref_index_xr, refx_xr, refy_xr, ref_div_iter, ref_order,
        refpath_ptr, out_is_xr, out_xr, M_bla, r_bla, bla_len, stages_bla
    ):
        """
        Parameters
        ----------
        c, c_xr: c and it "Xrange" counterparts
        Z, Z_xr: idem for result vector Z
        Z_xr_trigger : bolean, activated when Z_xr need to be used
        """
        # print("in numba impl")
        # SA skipped - wrapped iteration if we reach the cycle order 
        w_iter = n_iter
        if w_iter >= ref_order:
            w_iter = w_iter % ref_order
        # We know that :
        # ref_orbit_len = max_iter + 1 >= ref_div_iter
        # if order is not None:
        #    ref_orbit_len = min(order, ref_orbit_len)
        ref_orbit_len = Zn_path.shape[0]
        first_invalid_index = min(ref_orbit_len, ref_div_iter, ref_order)
        M_out = np.empty((8,), dtype=np.float64)

        while True:
            #==========================================================
            # Try a BLA_step
            if BLA_activated and (w_iter & STG_SKIP_MASK) == 0: # and False:
                Zn = Z[xn] + 1j * Z[yn]
                step = ref_BLA_get(
                    M_bla, r_bla, bla_len, stages_bla, Zn, w_iter,
                    first_invalid_index, M_out, False
                )
                
                if step != 0:
                    n_iter += step
                    w_iter = (w_iter + step) % ref_order
                    if xr_detect_activated:
                        apply_BLA_BS(M_out, Z, a_xr, b_xr, xn, yn)
                        # /!\ keep this, needed for next BLA step
                        Z[xn] = fsxn.to_standard(Z_xr[xn])
                        Z[yn] = fsxn.to_standard(Z_xr[yn])
                        if calc_hessian:
                            apply_BLA_deriv_BS(M_out, Z, a_xr, b_xr,
                                               dxnda, dxndb, dynda, dyndb)
                    else:
                        # just the usual BLA step
                        apply_BLA_BS(M_out, Z, a, b, xn, yn)
                        if calc_hessian:
                            apply_BLA_deriv_BS(M_out, Z, a, b,
                                               dxnda, dxndb, dynda, dyndb)
                    continue

            #==================================================================
            # BLA failed, launching a full perturbation iteration
            n_iter += 1 # the indice we are going to compute now
            # Load reference point value @ w_iter
            # refpath_ptr = [prev_idx, curr_xr]
            if xr_detect_activated:
                ref_zn = ref_path_get_BS(
                    Zn_path, w_iter,
                    has_xr, ref_index_xr, refx_xr, refy_xr, refpath_ptr,
                    out_is_xr, out_xr, 0, 2
                )
                ref_xn_xr, ref_yn_xr = ensure_xr_BS(
                    ref_zn, out_xr[0], out_xr[2], out_is_xr[0]
                )
            else:
                ref_zn = Zn_path[w_iter]
                ref_xn = ref_zn.real
                ref_yn = ref_zn.imag

            #==================================================================
            # Pertubation iter block
            #------------------------------------------------------------------
            # dzndc subblock
            if calc_hessian:
                ref_dxnda = dXnda_path[w_iter] # Note this may be Xrange
                ref_dxndb = dXndb_path[w_iter]
                ref_dynda = dYnda_path[w_iter]
                ref_dyndb = dYndb_path[w_iter]
                if xr_detect_activated:
                    p_iter_hessian(
                        Z_xr, ref_xn_xr, ref_yn_xr,
                        ref_dxnda, ref_dxndb, ref_dynda, ref_dyndb
                    )
                else:
                    p_iter_hessian(
                        Z, ref_xn, ref_yn,
                        ref_dxnda, ref_dxndb, ref_dynda, ref_dyndb
                    )

            #------------------------------------------------------------------
            # zn subblok
            if xr_detect_activated:
                # Z_xr[zn] = p_iter_zn(Z_xr, ref_zn_xr, c_xr)# in place mod
                p_iter_zn(Z_xr, ref_xn_xr, ref_yn_xr, a_xr, b_xr)# in place mod
                # std is used for div condition 
                Z[xn] = fsxn.to_standard(Z_xr[xn])
                Z[yn] = fsxn.to_standard(Z_xr[yn])
            else:
                # Z[zn] = p_iter_zn(Z, ref_zn, c)
                p_iter_zn(Z, ref_xn, ref_yn, a, b)

            #==================================================================
            # Stopping condition: maximum iter reached
            if n_iter >= max_iter:
                stop[0] = reason_max_iter
                break

            #==================================================================
            # Stopping condition: divergence
            # ZZ = "Total" z + dz
            w_iter += 1
            # print("incr w_iter", w_iter, n_iter, ref_order)
            if w_iter >= ref_order:
                w_iter = w_iter % ref_order

            if xr_detect_activated:
                ref_zn_next = ref_path_get_BS(
                    Zn_path, w_iter,
                    has_xr, ref_index_xr, refx_xr, refy_xr, refpath_ptr,
                    out_is_xr, out_xr, 1, 3
                )
            else:
                ref_zn_next = Zn_path[w_iter]

            # div condition computation with std only
            XX = Z[xn] + ref_zn_next.real
            YY = Z[yn] + ref_zn_next.imag
            full_sq_norm = XX ** 2 + YY ** 2

            # Flagged as 'diverging'
            bool_infty = (full_sq_norm > M_divergence_sq)
            if bool_infty:
                stop[0] = reason_M_divergence
                break

            #==================================================================
            # Glitch correction - reference point diverging
            if (w_iter >= ref_div_iter - 1):
                # print("reference point diverging rebase")
                # Rebasing - we are already big no underflow risk
                Z[xn] = XX
                Z[yn] = YY
                if xr_detect_activated:
                    Z_xr[xn] = fsxn.to_Xrange_scalar(XX)
                    Z_xr[yn] = fsxn.to_Xrange_scalar(YY)
#                if calc_dzndc:
#                    Z[dzndc] = Z[dzndc] + dZndc_path[w_iter]
                w_iter = 0
                continue

            #==================================================================
            # Glitch correction - "dynamic glitch"
            bool_dyn_rebase = (
                (abs(XX) <= abs(Z[xn])) and (abs(YY) <= abs(Z[yn]))
            )
            if bool_dyn_rebase:
                # print("bool_dyn_rebase")
                if xr_detect_activated:
                    # Can we *really* rebase ??
                    # Note: if Z[zn] underflows we might miss a rebase
                    # So we cast everything to xr
                    X_xrn = Z_xr[xn]
                    Y_xrn = Z_xr[yn]
                    if out_is_xr[1]:
                        # Reference underflows, use available xr ref
                        XX_xr = X_xrn + out_xr[1]
                        YY_xr = Y_xrn + out_xr[3]
                    else:
                        XX_xr = X_xrn + ref_zn_next.real
                        YY_xr = Y_xrn + ref_zn_next.imag

                    bool_dyn_rebase_xr = (
                        (XX_xr ** 2 + YY_xr ** 2) <= (X_xrn ** 2 + Y_xrn ** 2)
                    )
                    if bool_dyn_rebase_xr:
                        Z_xr[xn] = XX_xr
                        Z_xr[yn] = YY_xr
                        # Z[zn] = fsxn.to_standard(ZZ_xr)
                        if calc_hessian:
                            Z_xr[dxnda] = Z_xr[dxnda] + dXnda_path[w_iter]
                            Z_xr[dxndb] = Z_xr[dxndb] + dXndb_path[w_iter]
                            Z_xr[dynda] = Z_xr[dynda] + dYnda_path[w_iter]
                            Z_xr[dyndb] = Z_xr[dyndb] + dYndb_path[w_iter]
                        w_iter = 0
                        continue
                else:
                    # No risk of underflow - safe to rebase
                    Z[xn] = XX
                    Z[yn] = YY
                    if calc_hessian:
                        Z[dxnda] = Z[dxnda] + dXnda_path[w_iter]
                        Z[dxndb] = Z[dxndb] + dXndb_path[w_iter]
                        Z[dynda] = Z[dynda] + dYnda_path[w_iter]
                        Z[dyndb] = Z[dyndb] + dYndb_path[w_iter]
                    w_iter = 0
                    continue

        # End of iterations for this point
        U[0] = w_iter
        if xr_detect_activated:
            Z[xn] = fsxn.to_standard(Z_xr[xn])
            Z[yn] = fsxn.to_standard(Z_xr[yn])
            for key in (dxnda, dxndb, dynda, dyndb):
                Z[key] = fsxn.to_standard(Z_xr[key])
        # print(n_iter, w_iter, "--> exit with zn dzndc", Z[xn], Z[yn])
        return n_iter

    return numba_impl

@numba.njit
def apply_BLA_BS(M, Z, a, b, xn, yn):
    Z_xn = M[0] * Z[xn] + M[1] * Z[yn] + M[4] * a + M[5] * b
    Z_yn = M[2] * Z[xn] + M[3] * Z[yn] + M[6] * a + M[7] * b
    Z[xn] = Z_xn
    Z[yn] = Z_yn

@numba.njit
def apply_BLA_deriv_BS(M, Z, a, b, dxnda, dxndb, dynda, dyndb):
#    assert dxnda >= 0
#    assert dxndb < len(Z)
    Z_dxnda = M[0] * Z[dxnda] + M[1] * Z[dynda]
    Z_dxndb = M[0] * Z[dxndb] + M[1] * Z[dyndb]
    Z_dynda = M[2] * Z[dxnda] + M[3] * Z[dynda]
    Z_dyndb = M[2] * Z[dxndb] + M[3] * Z[dyndb]
    Z[dxnda] = Z_dxnda
    Z[dxndb] = Z_dxndb
    Z[dynda] = Z_dynda
    Z[dyndb] = Z_dyndb

#------------------------------------------------------------------------------
# Series approximations
@numba.njit
def numba_SA_run(
        SA_loop, 
        Zn_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
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
    # Note : SA 23064 23063 for order 23063
    # ref_path[ref_order-1] ** 2 == -c OK
    print("SA", ref_div_iter, ref_order)#, ref_path[ref_order-1])
    if SA_stop == -1:
        SA_stop = ref_div_iter
    else:
        SA_stop = min(ref_div_iter, SA_stop)

    print_freq = max(5, int(SA_stop / 100000.))
    print_freq *= 1000
#    print("numba_SA_cycles - output every", print_freq)

    SA_valid = True
    n_real_iter = 0
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
    out_is_xr = np.zeros((1,), dtype=numba.bool_)
    out_xr = Xr_template.repeat(1)

    while SA_valid:

        # keep a copy in case this iter is invalidated
        P_old = P.coeffs.copy()

        # Load reference point value
        # refpath_ptr = [prev_idx, curr_xr]
        ref_zn = ref_path_get(
            Zn_path, n_iter,
            has_xr, ref_index_xr, ref_xr, refpath_ptr,
            out_is_xr, out_xr, 0
        )

        # incr iter
        n_real_iter +=1
        n_iter += 1
        # wraps to 0 when reaching cycle order
        if n_iter >= ref_order:
            n_iter -= ref_order

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
            n_real_iter -= 1

        if n_iter % print_freq == 0 and SA_valid:
            ssum = np.sqrt(coeffs_sum)
            print(
                "SA running", n_real_iter,
                "err: ", fsxn.to_Xrange_scalar(P.err[0]),
                "<< ", ssum
            )

    return P_ret, n_real_iter, P.err


#------------------------------------------------------------------------------
# Bilinear approximation
# Note: the bilinear arrays being cheap, they  will not be stored but
# re-computed if needed
@numba.njit
def numba_make_BLA(Zn_path, dfdz, kc, eps):
    """
    Generates a BVA tree with
    - bilinear approximation coefficients A and B
    - validaty radius
        z_n+2**stg = f**stg(z_n, c) with |c| < r_stg_n is approximated by 
        z_n+2**stg = A_stg_n * z_n + B_stg_n * c
    """
    # number of needed "stages" is (ref_orbit_len).bit_length()
    kc_std = fsxn.to_standard(kc[0])
    ref_orbit_len = Zn_path.shape[0]
    print("ref_orbit_len", ref_orbit_len)
    bla_dim = 2
    M_bla = np.zeros((2 * ref_orbit_len, bla_dim), dtype=numba.complex128)
    r_bla = np.zeros((2 * ref_orbit_len,), dtype=numba.float64)
    M_bla_new, r_bla_new, bla_len, stages = init_BLA(
        M_bla, r_bla, Zn_path, dfdz, kc_std, eps
    )
    return M_bla_new, r_bla_new, bla_len, stages

@numba.njit
def numba_make_BLA_BS(Zn_path, dfxdx, dfxdy, dfydx, dfydy, kc, eps):
    """
    Generates a BVA tree for non-holomorphic functions with
    - bilinear approximation coefficients A and B
    - validaty radius
        z_n+2**stg = f**stg(z_n, c) with |c| < r_stg_n is approximated by 
        z_n+2**stg = A_stg_n * z_n + B_stg_n * c
    """
    # number of needed "stages" is (ref_orbit_len).bit_length()
    kc_std = fsxn.to_standard(kc[0])
    ref_orbit_len = Zn_path.shape[0]
    print("ref_orbit_len in BLA", ref_orbit_len)
    bla_dim = 8
    M_bla = np.zeros((2 * ref_orbit_len, bla_dim), dtype=numba.float64)
    r_bla = np.zeros((2 * ref_orbit_len,), dtype=numba.float64)
    M_bla_new, r_bla_new, bla_len, stages = init_BLA_BS(
        M_bla, r_bla, Zn_path, dfxdx, dfxdy, dfydx, dfydy, kc_std, eps
    )
    return M_bla_new, r_bla_new, bla_len, stages


@numba.njit
def init_BLA(M_bla, r_bla, Zn_path, dfdz, kc_std, eps):
    """
    Initialize BLA tree at stg 0
    """
    ref_orbit_len = Zn_path.shape[0]  # at order + 1, we wrap

    for i in range(ref_orbit_len):
        i_0 = 2 * i # BLA index for (i, 0)

        # Define a BLA_step by:
        #       [ M[0]    0     0]
        #  M =  [    0 M[0]  M[1]] 
        #       [    0    0     1]
        #
        #       [dzndc]
        #  Zn = [   zn]
        #       [    c]
        #
        #  Z_(n+1) = M * Zn

        Zn_i = Zn_path[i]
        M_bla[i_0, 0] = dfdz(Zn_i)
        M_bla[i_0, 1] = 1.

        # We use the following criteria :
        # |Z + z| shall stay *far* from O or discontinuity of F', for each c
        # For std Mandelbrot it means from z = 0
        # |zn| << |Zn|
        # For Burning ship x = 0 or y = 0
        # |zn| << |Xn|,  |zn| << |Yn|
        # We could additionnally consider a criterian based on hessian
        # |z| < A e / h where h Hessian - not useful (redundant)
        # for Mandelbrot & al.
        mZ = np.abs(Zn_path[i])
        ii = (i + 1) % ref_orbit_len
        mZZ = np.abs(Zn_path[ii])
        # mA = np.abs(M_bla[i_0, 0])

        r_bla[i_0] = max(
            0.,
            min(
                # error term is negligible
                mZ * eps,  
                # Avoid dyn glitch at next step
                mZZ * eps
                # ((0.5 * mZZ) - kc_std) / (1. + mA)
            )
        )

    # Now the combine step
    # number of needed "stages" (ref_orbit_len).bit_length()
    stages = _stages_bla(ref_orbit_len)
    for stg in range(1, stages):
        combine_BLA(M_bla, r_bla, kc_std, stg, ref_orbit_len, eps)
    M_bla_new, r_bla_new, bla_len = compress_BLA(M_bla, r_bla, stages)
    return M_bla_new, r_bla_new, bla_len, stages

@numba.njit
def init_BLA_BS(M_bla, r_bla, Zn_path, dfxdx, dfxdy, dfydx, dfydy,
                kc_std, eps):
    """
    Initialize BLA tree at stg 0
    """
    ref_orbit_len = Zn_path.shape[0]  # at order + 1, we wrap

    for i in range(ref_orbit_len):
        i_0 = 2 * i # BLA index for (i, 0)

        # Define a BLA_step by:
        #  Z_(n+1) = M * Zn where

        #       [dxnda]
        #       [dxndb]
        #       [dynda]            [  M_0    0     0]
        #  Zn = [dyndb]       M =  [    0  M_1   M_2] 
        #       [   xn]            [    0    0     I]
        #       [   yn]
        #       [    a]
        #       [    b]

        #        [M[0] 0    M[1] 0   ]
        #  M_0 = [0    M[0] 0    M[1]]
        #        [M[2] 0    M[3] 0   ]
        #        [0    M[2] 0    M[3]]
        #
        #  M_1 = [M[0] M[1]]   M_1_init = [dfxdx dfxdy]
        #        [M[2] M[3]]              [dfydx dfydy]
        #
        #  M_2 = [M[4] M[5]]   M_2_init = [1  0]
        #        [M[6] M[7]]              [0 -1]
        #

        Zn_i = Zn_path[i]
        Xn_i = Zn_i.real
        Yn_i = Zn_i.imag

        M_bla[i_0, 0] = dfxdx(Xn_i, Yn_i)
        M_bla[i_0, 1] = dfxdy(Xn_i, Yn_i)
        M_bla[i_0, 2] = dfydx(Xn_i, Yn_i)
        M_bla[i_0, 3] = dfydy(Xn_i, Yn_i)

        M_bla[i_0, 4] = 1.
        M_bla[i_0, 5] = 0.
        M_bla[i_0, 6] = 0.
        M_bla[i_0, 7] = -1.

        mZ = min(abs(Xn_i), abs(Yn_i))
        # ii = (i + 1) % ref_orbit_len
        
        
        r_bla[i_0] =  mZ * eps

    # Now the combine step
    # number of needed "stages" i.e. (ref_orbit_len).bit_length()
    stages = _stages_bla(ref_orbit_len)
    for stg in range(1, stages):
        combine_BLA_BS(M_bla, r_bla, kc_std, stg, ref_orbit_len, eps)
    M_bla_new, r_bla_new, bla_len = compress_BLA(M_bla, r_bla, stages)
    return M_bla_new, r_bla_new, bla_len, stages


@numba.njit
def _stages_bla(ref_orbit_len):
    """
    number of needed "stages" (ref_orbit_len).bit_length()
    """
    return int(np.ceil(np.log2(ref_orbit_len)))

@numba.njit
def combine_BLA(M, r, kc_std, stg, ref_orbit_len, eps):
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
        #  M_res =  [ M2[0]  M2[1]] * [ M1[0]  M1[1]] 
        #           [     0      1]   [     0      1]
        M[index_res, 0] = M[index2, 0] * M[index1, 0]
        M[index_res, 1] = M[index2, 0] * M[index1, 1] + M[index2, 1]

        # Combines the validity radii
        r1 = r[index1]
        r2 = r[index2]
        # r1 is a direct criteria however for r2 we need to go 'backw the flow'
        # z0 -> z1 -> z2 with z1 = A1 z0 + B1 c, |z1| < r2
        mA1 = np.abs(M[index1, 0])
        mB1 = np.abs(M[index1, 1])
        r2_backw = max(0., (r2 - mB1 * kc_std) / (mA1 + 1.)) # might use eps ?
        r[index_res] = min(r1, r2_backw)

@numba.njit
def combine_BLA_BS(M, r, kc_std, stg, ref_orbit_len, eps):
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

        #  M =  [M2_1   M2_2] x  [M1_1   M1_2]
        #       [   0      I]    [   0      I]
        #   Mx_1 = [M[0] M[1]]   Mx_2 = [M[4] M[5]]
        #          [M[2] M[3]]          [M[6] M[7]] 
        #  Mres_1 = M2_1 * M1_1
        #  Mres_2 = M2_1 * M1_1 + M2_2

        # Mres_1 = M2_1 * M1_1 :
        M[index_res, 0] = (
            M[index2, 0] * M[index1, 0] + M[index2, 1] * M[index1, 2]
        )
        M[index_res, 1] = (
            M[index2, 0] * M[index1, 1] + M[index2, 1] * M[index1, 3]
        )
        M[index_res, 2] = (
            M[index2, 2] * M[index1, 0] + M[index2, 3] * M[index1, 2]
        )
        M[index_res, 3] = (
            M[index2, 2] * M[index1, 1] + M[index2, 3] * M[index1, 3]
        )
        #  Mres_2 = M2_1 * M1_1 + M2_2
        M[index_res, 4] = (
            M[index2, 0] * M[index1, 4] + M[index2, 1] * M[index1, 6]
            + M[index2, 4]
        )
        M[index_res, 5] = (
            M[index2, 0] * M[index1, 5] + M[index2, 1] * M[index1, 7]
            + M[index2, 5]
        )
        M[index_res, 6] = (
            M[index2, 2] * M[index1, 4] + M[index2, 3] * M[index1, 6]
            + M[index2, 6]
        )
        M[index_res, 7] = (
            M[index2, 2] * M[index1, 5] + M[index2, 3] * M[index1, 7]
            + M[index2, 7]
        )

        # Combines the validity radii
        r1 = r[index1]
        r2 = r[index2]
        # r1 is a direct criteria however for r2 we need to go 'backw the flow'
        # z0 -> z1 -> z2 with z1 = A1 z0 + B1 c, |z1| < r2
        mA1 = max(
            np.abs(M[index1, 0]), 
            np.abs(M[index1, 1]),
            np.abs(M[index1, 2]), 
            np.abs(M[index1, 3]),
        )
        mB1 = max(
            np.abs(M[index1, 4]), 
            np.abs(M[index1, 5]),
            np.abs(M[index1, 6]), 
            np.abs(M[index1, 7]),
        )
        r2_backw = max(0., (r2 - mB1 * kc_std) / (mA1 + 1.)) # might use eps ?
        r[index_res] = min(r1, r2_backw)


@numba.njit
def compress_BLA(M_bla, r_bla, stages):
    """
    We build 'compressed' arrays which only feature multiples of 
    2 ** STG_COMPRESSED
    """
    k_comp = 1 << STG_COMPRESSED
    ref_orbit_len = M_bla.shape[0] // 2
    new_len = M_bla.shape[0] // k_comp
    bla_dim = M_bla.shape[1]

    M_bla_new = np.zeros((new_len * bla_dim,), dtype=M_bla.dtype)
    r_bla_new = np.zeros((new_len,), dtype=numba.float64)
    
    for stg in range(STG_COMPRESSED, stages):
        step = (1 << stg)
        for i in range(0, ref_orbit_len - step, step):
            index = BLA_index(i, stg)
            new_index = BLA_index(i // k_comp, stg - STG_COMPRESSED)
            for d in range(bla_dim):
                M_bla_new[new_index + d * new_len] = M_bla[index, d]

            r_bla_new[new_index] = r_bla[index]
    print("BLA tree compressed with coeff:", k_comp)
    return M_bla_new, r_bla_new, new_len

@numba.njit
def BLA_index(i, stg):
    """
    Return the indices in BVA table for this iteration and stage
    this is the jump from i to j = i + (1 << stg)
    """
    return (2 * i) + ((1 << stg) - 1)

@numba.njit
def ref_BLA_get(M_bla, r_bla, bla_len, stages_bla, zn, n_iter,
                first_invalid_index, M_out, holomorphic):
    """
    Paramters:
    ----------
    A_bla, B_bla, r_bla: arrays
        Bilinear approx tree
    zn :
        The current value of dz
    n_iter :
        The current iteration for ref pt
    M_out :
        Container for the Bla coefficient
    holomorphic: boolean
        True if the base function is holomorphic

    Returns:
    --------
     step
         The interation "jump" provided by this linear interpolation
    """
    k_comp = (1 << STG_COMPRESSED)
    _iter = (n_iter >> STG_COMPRESSED)
    for stages in range(STG_COMPRESSED, stages_bla):
        if _iter & 1:
            break
        _iter = _iter >> 1

    # The first invalid step /!\
    invalid_step = first_invalid_index - n_iter 

    # numba version of reversed(range(stages_bla)):
    for stg in range(stages, STG_COMPRESSED - 1, -1):
        step = (1 << stg)
        if step >= invalid_step:
            continue
        index_bla = BLA_index(n_iter // k_comp, stg - STG_COMPRESSED)

        r = r_bla[index_bla]
        # /!\ Use strict comparisons here: to rule out underflow
        if (abs(zn) < r):
            if holomorphic:
                M_out[0] = M_bla[index_bla]
                M_out[1] = M_bla[index_bla + bla_len]
                return step
            else:
                for i in range(8):
                    M_out[i] = M_bla[index_bla + i * bla_len]
                return step
    return 0 # No BLA applicable

@numba.njit
def need_xr(x_std):
    """
    True if norm L-inf of std is lower than xrange_zoom_level
    """
    return (
        (abs(np.real(x_std)) < fs.settings.xrange_zoom_level)
         and (abs(np.imag(x_std)) < fs.settings.xrange_zoom_level)
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
def ensure_xr_BS(val_std, valx_xr, valy_xr, is_xr):
    """
    Return a valid Xrange. if not(Z_xr_trigger) we return x_std
    converted
    
    val_xr : complex128_Xrange_scalar or float64_Xrange_scalar
    """
    if is_xr:
        return (
            fsxn.to_Xrange_scalar(valx_xr),
            fsxn.to_Xrange_scalar(valy_xr),
        )
    else:
        return (
            fsxn.to_Xrange_scalar(np.real(val_std)),
            fsxn.to_Xrange_scalar(np.imag(val_std))
        )

@numba.njit
def ref_path_c_from_pix(pix, dx, drift):
    """
    Returns the true c (coords from ref point) from the pixel coords
    
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
def ref_path_c_from_pix_BS(pix, dx, driftx_xr, drifty_xr):
    """
    Returns the true a + i b (coords from ref point) from the pixel coords

    Parameters
    ----------
    pix :  complex
        pixel location in fraction of dx
    dx : Xrange float
        width of the image

    Returns
    -------
    a, b, a_xr, b_xr : c value as complex and as Xrange
    """
    a_xr = (pix.real * dx[0]) + driftx_xr[0]
    b_xr = (pix.imag * dx[0]) + drifty_xr[0]
    return fsxn.to_standard(a_xr), fsxn.to_standard(b_xr), a_xr, b_xr


@numba.njit
def numba_dZndc_path(Zn_path, has_xr, ref_index_xr, ref_xr,
                    ref_div_iter, ref_order, dfdz, dx_xr, xr_detect_activated):
    """
    Compute dZndc in Xr, or std precision, depending on xr_detect_activated
    """
    ref_orbit_len = Zn_path.shape[0]
    valid_pts = min(ref_orbit_len, ref_div_iter)

    xr_act = xr_detect_activated
    dx = fsxn.to_standard(dx_xr[0])

    if xr_act:
        dZndc_path = np.zeros((1,), dtype=numba.complex128) # dummy
        dZndc_xr_path = Xr_template.repeat(ref_orbit_len) 

        refpath_ptr = np.zeros((2,), dtype=numba.int32)
        out_is_xr = np.zeros((1,), dtype=numba.bool_)
        out_xr = Xr_template.repeat(1)

        for i in range(1, valid_pts):
            ref_zn = ref_path_get(
                Zn_path, i - 1,
                has_xr, ref_index_xr, ref_xr, refpath_ptr,
                out_is_xr, out_xr, 0
            )
            ref_zn_xr = ensure_xr(ref_zn, out_xr[0], out_is_xr[0])
            dZndc_xr_path[i] = dfdz(ref_zn_xr) * dZndc_xr_path[i - 1] + dx_xr[0]

        if (i == ref_order - 1):
            # /!\ We have a cycle, use the "wrapped" value at 0
            # Note that this value will be used... a lot !
            ref_zn = ref_path_get(
                Zn_path, i,
                has_xr, ref_index_xr, ref_xr, refpath_ptr,
                out_is_xr, out_xr, 0
            )
            ref_zn_xr = ensure_xr(ref_zn, out_xr[0], out_is_xr[0])
            dZndc_xr_path[0] = dfdz(Zn_path[i]) * dZndc_xr_path[i] + dx_xr[0]

    else:
        dZndc_path = np.zeros((ref_orbit_len,), dtype=numba.complex128)
        dZndc_xr_path = Xr_template.repeat(1) # dummy
        for i in range(1, valid_pts):
            dZndc_path[i] = dfdz(Zn_path[i - 1]) * dZndc_path[i - 1] + dx
        if (i == ref_order - 1):
            # /!\ We have a cycle, use the "wrapped" value at 0
            # Note that this value will be used... a lot !
            dZndc_path[0] = dfdz(Zn_path[i]) * dZndc_path[i] + dx

    return dZndc_path, dZndc_xr_path


@numba.njit
def numba_dZndc_path_BS(Zn_path, has_xr, ref_index_xr, refx_xr, refy_xr,
                    ref_div_iter, ref_order, dfxdx, dfxdy, dfydx, dfydy,
                    dx_xr, xr_detect_activated):
    """
    Compute dXndb, dXnda, dYnda , dYndb in Xr, or std precision, depending on
    xr_detect_activated
    """
    ref_orbit_len = Zn_path.shape[0]
    valid_pts = min(ref_orbit_len, ref_div_iter)
    dx = fsxn.to_standard(dx_xr[0])

    if xr_detect_activated:
        dXnda_path = np.zeros((1,), dtype=numba.float64) # dummy
        dXndb_path = np.zeros((1,), dtype=numba.float64) # dummy
        dYnda_path = np.zeros((1,), dtype=numba.float64) # dummy
        dYndb_path = np.zeros((1,), dtype=numba.float64) # dummy
        dXnda_xr_path = Xr_float_template.repeat(ref_orbit_len) 
        dXndb_xr_path = Xr_float_template.repeat(ref_orbit_len) 
        dYnda_xr_path = Xr_float_template.repeat(ref_orbit_len) 
        dYndb_xr_path = Xr_float_template.repeat(ref_orbit_len) 

        refpath_ptr = np.zeros((2,), dtype=numba.int32)
        out_is_xr = np.zeros((1,), dtype=numba.bool_)
        out_xr = Xr_float_template.repeat(2) # coord X, coord Y

        for i in range(1, valid_pts):
            from_i = i - 1
            to_i = i
            ref_zn = ref_path_get_BS(
                Zn_path,from_i,
                has_xr, ref_index_xr, refx_xr, refy_xr, refpath_ptr,
                out_is_xr, out_xr, 0, 1
            )
            ref_xn_xr, ref_yn_xr = ensure_xr_BS(
                ref_zn, out_xr[0], out_xr[1], out_is_xr[0]
            )
            incr_deriv_ref_BS(
                dXnda_xr_path, dXndb_xr_path, dYnda_xr_path, dYndb_xr_path,
                from_i, to_i, dx_xr[0],
                ref_xn_xr, ref_yn_xr, dfxdx, dfxdy, dfydx, dfydy
            )

        if (i == ref_order - 1):
            # /!\ We have a cycle, use the "wrapped" value at 0
            # Note that this value will be used... a lot !
            from_i = i
            to_i = 0
            ref_zn = ref_path_get_BS(
                Zn_path, from_i,
                has_xr, ref_index_xr, refx_xr, refy_xr, refpath_ptr,
                out_is_xr, out_xr, 0, 1
            )
            ref_xn_xr, ref_yn_xr = ensure_xr_BS(
                ref_zn, out_xr[0], out_xr[1], out_is_xr[0]
            )
            incr_deriv_ref_BS(
                dXnda_xr_path, dXndb_xr_path, dYnda_xr_path, dYndb_xr_path,
                from_i, to_i, dx_xr[0],
                ref_xn_xr, ref_yn_xr, dfxdx, dfxdy, dfydx, dfydy
            )

    else:
        dXnda_path = np.zeros((ref_orbit_len,), dtype=numba.float64)
        dXndb_path = np.zeros((ref_orbit_len,), dtype=numba.float64)
        dYnda_path = np.zeros((ref_orbit_len,), dtype=numba.float64)
        dYndb_path = np.zeros((ref_orbit_len,), dtype=numba.float64)
        dXnda_xr_path = Xr_float_template.repeat(1) # dummy
        dXndb_xr_path = Xr_float_template.repeat(1) # dummy
        dYnda_xr_path = Xr_float_template.repeat(1) # dummy
        dYndb_xr_path = Xr_float_template.repeat(1) # dummy

        for i in range(1, valid_pts):
            from_i = i - 1
            to_i = i
            Xn = np.real(Zn_path[from_i]) #.real
            Yn = np.imag(Zn_path[from_i]) #.imag
            incr_deriv_ref_BS(
                dXnda_path, dXndb_path, dYnda_path, dYndb_path,
                from_i, to_i, dx,
                Xn, Yn, dfxdx, dfxdy, dfydx, dfydy
            )

        if (i == ref_order - 1):
            # /!\ We have a cycle, use the "wrapped" value at 0
            # Note that this value will be used... a lot !
            from_i = i
            to_i = 0
            Xn = np.real(Zn_path[from_i]) #.real
            Yn = np.imag(Zn_path[from_i]) #.imag
            incr_deriv_ref_BS(
                dXnda_path, dXndb_path, dYnda_path, dYndb_path,
                from_i, to_i, dx,
                Xn, Yn, dfxdx, dfxdy, dfydx, dfydy
            )

    return (
        dXnda_path, dXndb_path, dYnda_path, dYndb_path,
        dXnda_xr_path, dXndb_xr_path, dYnda_xr_path, dYndb_xr_path
    )

@numba.njit
def incr_deriv_ref_BS(
    dXnda_path, dXndb_path, dYnda_path, dYndb_path,
    from_i, to_i, dx,
    Xn, Yn, dfxdx, dfxdy, dfydx, dfydy
):
    """
    H = [dfxdx dfxdy]    [dfx] = H x [dx]
        [dfydx dfydy]    [dfy]       [dy]
    """
    dfxdx = dfxdx(Xn, Yn)
    dfxdy = dfxdy(Xn, Yn)
    dfydx = dfydx(Xn, Yn)
    dfydy = dfydy(Xn, Yn)

    dXnda = dXnda_path[from_i]
    dXndb = dXndb_path[from_i]
    dYnda = dYnda_path[from_i]
    dYndb = dYndb_path[from_i]

    dXnda_path[to_i] = dfxdx * dXnda + dfxdy * dYnda + dx
    dXndb_path[to_i] = dfxdx * dXndb + dfxdy * dYndb
    dYnda_path[to_i] = dfydx * dXnda + dfydy * dYnda
    dYndb_path[to_i] = dfydx * dXndb + dfydy * dYndb - dx


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
    Returns
    -------
    (val, xr_val, is_xr, prev_idx, curr_xr)
        val : np.complex128
    
    Modify in place:
        xr_val : complex128_Xrange_scalar -> pushed to out_xr[out_index]
        is_xr : bool -> pushed to out_is_xr[out_index]
        prev_idx == refpath_ptr[0] : int
        curr_xr == refpath_ptr[1] : int (index in path ref_xr)
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
def ref_path_get_BS(ref_path, idx, has_xr, ref_index_xr, refx_xr, refy_xr, refpath_ptr,
                    out_is_xr, out_xr, outx_index, outy_index):
    """
    Alternative to getitem which also takes as input prev_idx, curr_xr :
    allows to optimize the look-up of Xrange values in case of successive calls
    with increasing idx.

    idx :
        index requested
    refpath_ptr = (prev_idx, curr_xr) :
        couple returned from last call, last index requested + next xr target
        Contract : curr_xr the smallest integer that verify :
            prev_idx <= ref_index_xr[curr_xr]
            or curr_xr = ref_index_xr.size (No more xr)
    Returns
    -------
    val np.complex128
        satndard value
    
    Modify in place:
        xr_val : complex128_Xrange_scalar -> pushed to out_xr[out_index]
        is_xr : bool -> pushed to out_is_xr[out_index]
        prev_idx == refpath_ptr[0] : int
        curr_xr == refpath_ptr[1] : int (index in path ref_xr)
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
        out_is_xr[outx_index] = False
        return ref_path[idx]

    elif idx == ref_index_xr[refpath_ptr[1]]:
        refpath_ptr[0] = idx
        out_is_xr[outx_index] = True
        out_xr[outx_index] = refx_xr[refpath_ptr[1]]
        out_xr[outy_index] = refy_xr[refpath_ptr[1]]
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
            out_is_xr[outx_index] = False
            return ref_path[idx]
        # Here idx == ref_index_xr[refpath_ptr[1]]
        refpath_ptr[0] = idx
        out_is_xr[outx_index] = True


        out_xr[outx_index] = refx_xr[refpath_ptr[1]]
        out_xr[outy_index] = refy_xr[refpath_ptr[1]]

        return ref_path[idx]