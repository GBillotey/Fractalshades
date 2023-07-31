# -*- coding: utf-8 -*-
import os
#import typing
import pickle
#import warnings
import logging
import textwrap
import copy

import mpmath
import numpy as np
import numba

import fractalshades as fs
import fractalshades.numpy_utils.xrange as fsx
import fractalshades.numpy_utils.numba_xr as fsxn


logger = logging.getLogger(__name__)


class PerturbationFractal(fs.Fractal):

    def __init__(self, directory):
        """Base class for escape-time fractals implementing the 
perturbation technique, with an iteration matching:

.. math::

    z_0 &= 0 \\\\
    z_{n+1} &= f ( z_{n} ) + c

Where :math:`f` has a critical point at 0.
Derived classes should provide the implementation for the actual function
:math:`f`.

Parameters
----------
directory : str
    Path for the working base directory"""
        super().__init__(directory)

    @fs.utils.zoom_options
    def zoom(
             self, *,
             precision: int,
             x: mpmath.mpf,
             y: mpmath.mpf,
             dx: mpmath.mpf,
             nx: int,
             xy_ratio: float,
             theta_deg: float,
             projection: fs.projection.Projection=fs.projection.Cartesian(),
             has_skew: bool=False,
             skew_00: float=1.,
             skew_01: float=0.,
             skew_10: float=0.,
             skew_11: float=1.
    ):
        """
        Define the zoom parameters for the next calculation.

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
        projection : `fractalshades.projection.Projection`
            Kind of projection used (default to
            `fractalshades.projection.Cartesian`)
        has_skew : bool
            If True, unskew the view base on skew coefficients skew_ij
        skew_ij : float
            Components of the local skew matrix, with ij = 00, 01, 10, 11
        """
        mpmath.mp.dps = precision # in base 10 digit 

        # In case the user inputs were strings, we override with mpmath scalars
        self.x = mpmath.mpf(x)
        self.y = mpmath.mpf(y)
        self.dx = dx = mpmath.mpf(dx)

        # Backward compatibility: also accept projection = "cartesian"
        if isinstance(projection, str):
            logger.warning(textwrap.dedent(
                """\
                Use of str for projection is deprecated, and might be
                removed. Please use `fractalshades.projection.Projection`
                subclasses instead. Defaulting to
                `fractalshades.projection.Cartesian()`"""
            ))
            self.projection = projection = fs.projection.Cartesian()

        # Stores the skew matrix
        self._skew = None
        if has_skew:
            self._skew = np.array(
                ((skew_00, skew_01), (skew_10, skew_11)), dtype=np.float64
            )

        self.dx_std = dx_std = float(dx)
        self.dx_xr = dx_xr = fsx.mpf_to_Xrange(dx, dtype=np.float64).ravel()
        # Here we store the linscale as a 1d 1 element array
        self.lin_scale_xr = np.asanyarray(dx_xr).reshape(-1)
        self.lin_scale_std = np.asanyarray(dx_std).reshape(-1)
        self.lin_mat = self.get_lin_mat()

        projection.adjust_to_zoom(self)
        self.proj_impl = projection.f

        # check the dps...
        pix = projection.min_local_scale * self.dx / self.nx
        # pix = projection.min_local_scale * projection.scale * self.dx / self.nx
        with mpmath.workdps(6):
            # Sets the working dps to 10e-1 x pixel size
            required_dps = int(-mpmath.log10(pix / nx) + 1)
        if required_dps > precision:
            raise ValueError(
                "Precision is too low for min. pixel size and shall be "
                f"increased to {required_dps} (current setting: {precision})."
                "This might be a consequence of Exponential mapping or "
                "other non-cartesian projections."
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


    def ref_point_file(self):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        reference orbit
        """
        return os.path.join(self.directory, "data", "ref_pt.dat")


    def ref_point_kc(self):
        """
        Return a bound for dc in image pixels, used as a criteria for BLA
        validity

        Returns:
        --------
        kc: full precision, scaling coefficient
        """
        c0 = self.x + 1j * self.y

        w, h = self.projection.bounding_box(self.xy_ratio)
        dx = self.dx # * self.projection.scale
        mat = self.lin_mat

        corner_a = mpc_lin_proj_impl_noscale(mat, 0.5 * w, 0.5 * h) * dx
        corner_b = mpc_lin_proj_impl_noscale(mat, -0.5 * w , 0.5 * h) * dx
        corner_c = mpc_lin_proj_impl_noscale(mat, -0.5 * w, -0.5 * h) * dx
        corner_d = mpc_lin_proj_impl_noscale(mat, 0.5 * w, -0.5 * h) * dx

        c0 = self.x + 1j * self.y
        ref = self.FP_params["ref_point"]
        shift = ref - c0

        # Let take the largest distance + some margin
        kc = max(abs(shift - corner_a), abs(shift - corner_b),
                 abs(shift - corner_c), abs(shift - corner_d)) * 1.1

        return fsx.mpf_to_Xrange(kc, dtype=np.float64).ravel() # 1d for numba


    def get_std_cpt(self, c_pix):
        """ Return the c complex value from c_pix - standard precision """
        n_pts, = c_pix.shape  # Z of shape [n_Z, n_pts]
        # Explicit casting to complex / float
        center = complex(self.x + 1j * self.y)
        cpt = np.empty((n_pts,), dtype=c_pix.dtype)

        fill1d_std_C_from_pix(
            c_pix, self.proj_impl, self.lin_mat, self.lin_scale_std,
            center, cpt
        )
        return cpt


    def ref_point_matching(self):
        """
        Test if the ref point can be used for this calculation ie 
           - same or more max_iter
           - not too far
           - and with a suitable dps
           - Same fractal __init__ params 
        """
        init_kwargs = self.init_kwargs
        del init_kwargs["directory"]

        try:
            ref_point = self.FP_params["ref_point"]
            max_iter_ref = self.FP_params["max_iter"]
            init_kwargs_ref = self.FP_params["init_kwargs"]
        except FileNotFoundError:
            logger.debug("No data file found for ref point")
            return False

        # Parameters 'max_iter' borrowed from last "@fsutils.calc_options" call
        # calc_options = self.calc_options
        max_iter = self.max_iter #calc_options["max_iter"]

        drift_xr = fsx.mpc_to_Xrange((self.x + 1j * self.y) - ref_point)
        dx_xr = fsx.mpf_to_Xrange(self.dx)
        
        matching_init_kwargs = True
        for k, v in init_kwargs_ref.items():
            if init_kwargs[k] != v:
                matching_init_kwargs = False

        dic_match = {
           "dps": mpmath.mp.dps <= self.FP_params["dps"] + 3,
           "location": (drift_xr / dx_xr).abs2() < 1.e6,
           "max_iter": max_iter_ref >= max_iter,
           "init_kwargs": matching_init_kwargs
        }
        for item, match in dic_match.items():
            if not match:
                logger.info(f"Updating ref point: {item} not matching")
                break

        return all(match for match in dic_match.values())


    def save_ref_point(self, FP_params, Zn_path):
        """
        Write to a data file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        self._FP_params = FP_params
        self._Zn_path = Zn_path
        save_path = self.ref_point_file()
        fs.utils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            logger.info(textwrap.dedent(f"""\
                    Full precision path computed, saving to:
                      {save_path}"""
            ))

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
        # Note: uses ravel to change the shape from (,) - numpy scalar - to
        # (1,) - numpy array - for numba compatibility
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

            for i, xr_index in enumerate(ref_xr_python.keys()):
                ref_index_xr[i] = xr_index
                tmpx, tmpy = ref_xr_python[xr_index]
                refx_xr[i] = tmpx[0]
                refy_xr[i] = tmpy[0]
            return (Zn_path, has_xr, ref_index_xr, refx_xr, refy_xr,
                    ref_div_iter, ref_order, driftx_xr, drifty_xr,
                    dx_xr)


#==============================================================================
# Printing - export functions
    @staticmethod
    def print_FP(FP_params, Zn_path):
        """
        Just a pretty-print of the reference path, for debugging purposes
        """
        
        pp = "-------------------------------------------------------------"
 
        pp += "  Full precision orbit loaded, FP_params:"
        for k, v in FP_params.items():
            try:
                for kv, vv in v.items():
                    pp +=  f"  {k}, ({kv}) --> {vv}"
            except AttributeError:
                pp +=  f"  {k} --> {v}"
        pp += f"ref_path, shape: {Zn_path.shape}, {Zn_path.dtype}"
        pp += str(Zn_path)
        pp += "  -------------------------------------------------------------"
        logger.info(pp)


    @staticmethod
    def write_FP(path, out_file):
        """
        Export to csv format a complex path, for debugging purposes
        """
        path_real = np.empty((path.shape[0], 2), dtype = np.float64)
        path_real[:, 0] = path.real
        path_real[:, 1] = path.imag
        np.savetxt(out_file, path_real, delimiter=",") 

#==============================================================================
# Calculation functions

    @staticmethod
    def numba_cycle_call(cycle_dep_args, cycle_indep_args):
        """ Here we customize for perturbation iterations """
        # We still have a case switch on indep parameter "holomorphic"
        holomorphic = cycle_indep_args[0]
        cycle_indep_args = cycle_indep_args[1:]
        
        if holomorphic:
            return numba_cycles_perturb(
                *cycle_dep_args, *cycle_indep_args
            )
        else:
            return numba_cycles_perturb_BS(
                *cycle_dep_args, *cycle_indep_args
            )


    def get_cycle_indep_args(self, initialize, iterate):
        """
        Parameters independant of the cycle
        This is where the hard work is done
        """
        # ====================================================================
        # CUSTOM class impl
        # Initialise the reference path
        holomorphic = self.holomorphic
        calc_deriv_c = self.calc_dZndc if holomorphic else self.calc_hessian
        calc_dZndz = self.calc_dZndz if holomorphic else False

        # 1) compute or retrieve the reference orbit
        self.set_status("Reference", "running")
        self.get_FP_orbit()

        if holomorphic:
            (Zn_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
             drift_xr, dx_xr) = self.get_path_data()
        else: 
            (Zn_path, has_xr, ref_index_xr, refx_xr, refy_xr,
             ref_div_iter, ref_order, driftx_xr, drifty_xr,
             dx_xr) = self.get_path_data()

        if ref_order is None:
            ref_order = (1 << 62) # a quite large int64
        self.set_status("Reference", "completed")

        # 2) compute the orbit derivatives if needed
        # Note: this is where the "scale" of the derivative is chosen
        scale_deriv_xr = dx_xr
        if self.projection.scale != 1.:
            # If there is a projection-induced scale, we need to use it here.
            scale_xr = fsx.mpf_to_Xrange(
                self.projection.scale, dtype=self.float_type
            ).ravel()
            scale_deriv_xr = dx_xr * scale_xr
            logger.debug(f"Derivative ref scale set at: {dx_xr}")
            

        dZndc_path = None
        (dXnda_path, dXndb_path, dYnda_path, dYndb_path) = (None,) * 4

        if calc_deriv_c:
            xr_detect_activated = self.xr_detect_activated

            if holomorphic:
                dZndc_path, dZndc_xr_path = numba_dZndc_path(
                    Zn_path, has_xr, ref_index_xr, ref_xr,
                    ref_div_iter, ref_order,
                    self.dfdz, scale_deriv_xr, xr_detect_activated
                )
                if xr_detect_activated:
                    dZndc_path = dZndc_xr_path

            else:
                (dXnda_path, dXndb_path, dYnda_path, dYndb_path,
                 dXnda_xr_path, dXndb_xr_path, dYnda_xr_path, dYndb_xr_path
                ) = numba_dZndc_path_BS(
                    Zn_path, has_xr, ref_index_xr, refx_xr, refy_xr,
                    ref_div_iter, ref_order,
                    self.dfxdx, self.dfxdy, self.dfydx, self.dfydy,
                    scale_deriv_xr, xr_detect_activated #, self.reverse_y
                )
                if xr_detect_activated:
                    dXnda_path = dXnda_xr_path
                    dXndb_path = dXndb_xr_path
                    dYnda_path = dYnda_xr_path
                    dYndb_path = dYndb_xr_path

        # 2') compute the orbit derivatives wrt z if needed
        # (interior detection - only for holomorphic case)
        dZndz_path = None
        if calc_dZndz:
            xr_detect_activated = self.xr_detect_activated

            dZndz_path, dZndz_xr_path = numba_dZndz_path(
                Zn_path, has_xr, ref_index_xr, ref_xr,
                ref_div_iter, ref_order,
                self.dfdz, xr_detect_activated
            )
            if xr_detect_activated:
                dZndz_path = dZndz_xr_path


        self.kc = kc = self.ref_point_kc()
        if kc == 0.:
            raise RuntimeError(
                "Resolution is too low for this zoom depth. Try to increase"
                "the reference calculation precicion."
        )
        logger.debug(f"Bounding box defined, with kc: {self.kc }")

        # Initialize BLA interpolation
        if self.BLA_eps is None:
            M_bla = None
            r_bla = None
            bla_len = None
            stages_bla = None
            self.set_status("Bilin. approx", "N.A.")
        else:
            self.set_status("Bilin. approx", "running")
            eps = self.BLA_eps
            M_bla, r_bla, bla_len, stages_bla = self.get_BLA_tree(Zn_path, eps)
            self.set_status("Bilin. approx", "completed")   

        proj_dzndc_modifier = getattr(self.projection, "dzndc_modifier", None)

        if holomorphic:
            cycle_indep_args = (
                holomorphic,
                initialize, iterate,
                Zn_path, dZndc_path, dZndz_path,
                has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
                drift_xr, dx_xr, self.proj_impl, self.lin_mat, self.lin_scale_xr,
                kc, M_bla, r_bla, bla_len, stages_bla,
                proj_dzndc_modifier,
                self._interrupted
            )
        else:
            cycle_indep_args = (
                holomorphic,
                initialize, iterate,
                Zn_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
                has_xr, ref_index_xr, refx_xr, refy_xr, ref_div_iter, ref_order,
                driftx_xr, drifty_xr, dx_xr, self.proj_impl, self.lin_mat, self.lin_scale_xr,
                kc, M_bla, r_bla, bla_len, stages_bla,
                proj_dzndc_modifier,
                self._interrupted
            )

        return cycle_indep_args


    def reset_bla_tree(self, cycle_indep_args):
        """ 
        Returns a new cycle_indep_args with modification of:
            - BLA validaty radius
            - derivative scaling (dZndc_path)
        for deep expmap zooms

        cycle_indep_args : the data to modify in place
        Note: the fractal projection shall have been modified through
        its ``set_exp_zoom_step`` method
        """
        # self.kc = self.ref_point_kc()
        logger.debug(f"Bounding box reset with kc: {self.kc }")
        initialize = cycle_indep_args[1]
        iterate = cycle_indep_args[2]
        return self.get_cycle_indep_args(initialize, iterate)



    def fingerprint_matching(self, calc_name, test_fingerprint, log=False):
        """
        Test if the stored parameters match those of new calculation
        /!\ modified in subclass
        """
        flatten_fp = fs.utils.dic_flatten(test_fingerprint)

        state = self._calc_data[calc_name]["state"]
        expected_fp = fs.utils.dic_flatten(state.fingerprint)
        
        if log:
            logger.debug(f"flatten_fp:\n {flatten_fp}")
            logger.debug(f"expected_fp:\n {expected_fp}")

        precision_key = f"zoom_kwargs@precision"
        SPECIAL = [precision_key,]

        for key, val in expected_fp.items():
            if (key in SPECIAL):
                if key == precision_key:
                    # We allow precision to *decrease* without rerunning
                    # the whole calc
                    if flatten_fp[key] < val:
                        if log:
                            logger.debug(textwrap.dedent(f"""\
                        Higher precision needed: will trigger a recalculation
                          {key}, {flatten_fp[key]} --> {val}"""
                            ))
                        return False
                    elif flatten_fp[key] > val:
                        if log:
                            logger.debug(textwrap.dedent(f"""\
                        Lower precision requested: no need for recalculation
                          {key}, {flatten_fp[key]} --> {val}"""
                            ))
                continue
            else:
                if flatten_fp[key] != val:
                    if log:
                        logger.debug(textwrap.dedent(f"""\
                            Parameter mismatch ; will trigger a recalculation
                              {key}, {flatten_fp[key]} --> {val}"""
                        ))
                    return False
        return True


    def get_BLA_tree(self, Zn_path, eps):
        """
        Return
        ------
        M_bla, r_bla, stages_bla
        """
        kc = self.kc

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
        # Check if we have a reference point stored, 
          - otherwise computes and stores it in a file

        newton: ["cv", "step", None]
        """
        
        if newton == "step":
            raise NotImplementedError("step option not Implemented (yet)")

        # Early escape if file exists
        if self.ref_point_matching():
            logger.info("Reference point already stored, skipping calc")
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
            logger.info(textwrap.dedent(f"""\
                Skipping all Newton calculation according to calc options
                    fs.settings.no_newton: {fs.settings.no_newton}
                    newton arg: {newton}"""
            ))
            self.compute_FP_orbit(c0, None)
            return
        
        # Here we will compute a Newton iteration.
        # First, try Ball method to guess the order
        if order is None:
            k_ball = 1.0
            order = self.ball_method(c0, self.dx * k_ball)
            if order is None:
                logger.warning(
                    "Ball method failed - Default to image center"
                )
                self.compute_FP_orbit(c0, None)
                return

        if order is None:
            raise ValueError("Order must be specified for Newton iteration")

        # Now launch Newton descent
        max_attempt = 2
        logger.info(textwrap.dedent(f"""\
            "Launch Newton descent with parameters:
                order: {order}, max newton attempts: {max_attempt + 1}"""
        ))

        eps_pixel = self.dx * (1. / self.nx)
        try:
            newton_cv, nucleus = self.find_nucleus(
                c0, order, eps_pixel, max_newton=max_newton
            )
        except NotImplementedError:
            newton_cv, nucleus = self.find_any_nucleus(
                c0, order, eps_pixel, max_newton=max_newton
            )

        # If Newton did not CV, we try to boost the dps / precision
        attempt = 1
        if not(newton_cv):
            old_dps = mpmath.mp.dps

            while not(newton_cv) and attempt <= max_attempt:
                attempt += 1
                mpmath.mp.dps = int(1.25 * mpmath.mp.dps)
                logger.info(textwrap.dedent(f"""\
                    Newton attempt {attempt - 1} failed
                      Increasing dps for next : {old_dps} -> {mpmath.mp.dps}"""
                ))
                eps_pixel = self.dx * (1. / self.nx)

                try:
                    no_div = True
                    newton_cv, nucleus = self.find_nucleus(
                        c0, order, eps_pixel, max_newton=max_newton
                    )
                except NotImplementedError:
                    no_div = False
                    newton_cv, nucleus = self.find_any_nucleus(
                        c0, order, eps_pixel, max_newton=max_newton
                    )

                if no_div and not(newton_cv) and (attempt == max_attempt):
                    # Last try, we just release constraint on the cycle order
                    # and consider also divisors.
                    newton_cv, nucleus = self.find_any_nucleus(
                        c0, order, eps_pixel, max_newton=max_newton
                    )

        # Still not CV ? we default to the center of the image
        if newton_cv:
            logger.info(f"Newton descent converged at attempt {attempt}.")
        else:
            order = None # We cannot wrap the ref point here...
            nucleus = c0
            logger.warning("Newton descent failed - Default to image center")


        shift = nucleus - (self.x + self.y * 1j)
        shift_x = shift.real / self.dx
        shift_y = shift.imag / self.dx
        logger.info(textwrap.dedent(f"""\
            Reference nucleus found at shift from center:
              ({shift_x}, {shift_y}) in dx units
              with order {order}"""
            ))
        self.compute_FP_orbit(nucleus, order)

    def compute_critical_orbit(self, crit):
        """
        Basically nothing to 'compute' here, just short-cutting
        """
        logger.debug("Skipping ref pt orbit calc for low res")
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

        # Also, store the *init_kwargs* because if the Fractal is regenerated
        # from new initial inputs, we shall obvs invalidate the orbit
        init_kwargs = self.init_kwargs
        del init_kwargs["directory"]
        FP_params["init_kwargs"] = init_kwargs

        Zn_path = np.zeros([div_iter + 1], dtype=np.complex128)
        Zn_path[:] = crit
        
        logger.debug(f"Storing critical orbit {crit}:\n  {Zn_path}")

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
        # Parameter 'max_iter' from last "@fsutils.calc_options" call
        max_iter = self.max_iter
        logger.info(f"Computing full precision path with max_iter {max_iter}")
        FP_code = self.FP_code
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

        if order is not None:
            logger.info(f"Order is known, wraping at {ref_orbit_len}")

        i, partial_dict, xr_dict = self.FP_loop(Zn_path, ref_point)
        FP_params["partials"] = partial_dict
        FP_params["xr"] = xr_dict
        FP_params["div_iter"] = i

        # Also, store the *init_kwargs* because if the Fractal is regenerated
        # from new initial inputs, we shall obvs invalidate the orbit
        init_kwargs = self.init_kwargs
        del init_kwargs["directory"]
        FP_params["init_kwargs"] = init_kwargs

        logger.info(f"Storing  orbit for pt: \n  {ref_point}\n  {Zn_path}")
        self.save_ref_point(FP_params, Zn_path)


    def ball_method(self, c, px, kind=1, M_divergence=1.e5):
        """
        Use a ball centered on c = x + i y to find the first period (up to 
        maxiter) of nucleus
        """
        max_iter = self.max_iter

        if kind == 1:
            return self._ball_method(c, px, max_iter, M_divergence)
        elif kind == 2:
            return self._ball_method2(c, px, max_iter, M_divergence)


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
            logger.info(f"Newton, decimal precision (dps) boost to: {dps}")
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

@numba.njit(nogil=True, fastmath=True, error_model="numpy")
def numba_cycles_perturb(
    c_pix, Z, U, stop_reason, stop_iter,
    initialize, iterate,
    Zn_path, dZndc_path, dZndz_path,
    has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
    drift_xr, dx_xr, proj_impl, lin_mat, lin_scale_xr,
    kc, M_bla, r_bla, bla_len, stages_bla,
    proj_dzndc_modifier,
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

    for ipt in range(npts):

        refpath_ptr = np.zeros((2,), dtype=np.int32)
        out_is_xr = np.zeros((2,), dtype=numba.bool_)
        out_xr = Xr_template.repeat(2)

        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        cpt, c_xr = ref_path_c_from_pix(
            c_pix[ipt], proj_impl, lin_mat, lin_scale_xr, drift_xr
        )
        stop_pt = stop_reason[:, ipt]

        initialize(c_xr, Zpt, Z_xr, Z_xr_trigger, Upt, kc, dx_xr)

        n_iter = iterate(
            cpt, c_xr, Zpt, Z_xr, Z_xr_trigger, Upt, stop_pt, # n_iter_init,
            Zn_path, dZndc_path, dZndz_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
            refpath_ptr, out_is_xr, out_xr, M_bla, r_bla, bla_len, stages_bla,
            proj_dzndc_modifier, c_pix[ipt]
        )
        stop_iter[0, ipt] = n_iter
        stop_reason[0, ipt] = stop_pt[0]

        if _interrupted[0]:
            return USER_INTERRUPTED

    return 0


def numba_initialize(zn, dzndc, dzndz):
    @numba.njit(fastmath=True, error_model="numpy")
    def numba_init_impl(c_xr, Z, Z_xr, Z_xr_trigger, U, kc, dx_xr):
        """
        Initialize 'in place'  :
            Z[zn], Z[dzndz], Z[dzndc]
        """
        Z_xr[zn] = fsxn.to_Xrange_scalar(Z[zn])
        if dzndc != -1:
            Z_xr[dzndc] = fsxn.to_Xrange_scalar(Z[dzndc])

        if (dzndz != -1):
            Z[dzndz] = 0.
    return numba_init_impl


# Defines iterate via a function factory - jitted implementation
def numba_iterate(
        max_iter, M_divergence_sq, epsilon_stationnary_sq,
        reason_max_iter, reason_M_divergence, reason_stationnary,
        xr_detect_activated, BLA_activated,
        zn, dzndc, dzndz,
        p_iter_zn, p_iter_dzndz, p_iter_dzndc,
        calc_dzndc, calc_dzndz,
        calc_orbit, i_znorbit, backshift, zn_iterate # Added args
):

    @numba.njit(fastmath=True, error_model="numpy")
    def numba_impl(
        c, c_xr, Z, Z_xr, Z_xr_trigger, U, stop,
        Zn_path, dZndc_path, dZndz_path, has_xr, ref_index_xr, ref_xr, ref_div_iter, ref_order,
        refpath_ptr, out_is_xr, out_xr, M_bla, r_bla, bla_len, stages_bla,
        proj_dzndc_modifier, c_pix
    ):
        """
        Parameters
        ----------
        c, c_xr: c and it "Xrange" counterparts
        Z, Z_xr: idem for result vector Z
        Z_xr_trigger : bolean, activated when Z_xr need to be used
        """
        record_zero = Xr_template.repeat(1)[0]
        
        w_iter = 0
        n_iter = 0
        if w_iter >= ref_order:
            w_iter = w_iter % ref_order

        if calc_orbit:
            div_shift = 0
            orbit_zn1 = Z[zn]
            orbit_zn2 = Z[zn]
            orbit_i1 = 0
            orbit_i2 = 0
            
        if calc_dzndz:
            nullify_dZndz = False
            w_wraped = len(dZndz_path) - 1

        # We know that :
        # ref_orbit_len = max_iter + 1 >= ref_div_iter
        # if order is not None:
        #    ref_orbit_len = min(order, ref_orbit_len)
        ref_orbit_len = Zn_path.shape[0]
        first_invalid_index = min(ref_orbit_len, ref_div_iter, ref_order)
        M_out = np.empty((2,), dtype=np.complex128)

        # Index to keep track if we use the wraped index of dzndc
        bool_dyn_rebase = True

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
                        if calc_dzndz:
                            Z_xr[dzndz] = M_out[0] * Z_xr[dzndz]
                    else:
                        # just the usual BLA step
                        Z[zn] = M_out[0] * Z[zn] + M_out[1] * c
                        if calc_dzndc:
                            Z[dzndc] = M_out[0] * Z[dzndc]
                        if calc_dzndz:
                            Z[dzndz] = M_out[0] * Z[dzndz]
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
                ref_dzndc = dZndc_path[w_iter]
                if bool_dyn_rebase: # Avoid wraped value at 0
                    if xr_detect_activated:
                        ref_dzndc = record_zero # Record type
                    else:
                        ref_dzndc = 0.
                if xr_detect_activated:
                    p_iter_dzndc(Z_xr, ref_zn_xr, ref_dzndc)
                else:
                    p_iter_dzndc(Z, ref_zn, ref_dzndc)

            #------------------------------------------------------------------
            # Interior detection
            if calc_dzndz:
                if nullify_dZndz:
                    ref_dzndz = dZndz_path[0]  # This may be Xrange
                else:
                    # Note: no need of special case for n_iter == order here
                    # It is managed before during glitch correction step
                    ref_dzndz = dZndz_path[w_iter]

                if xr_detect_activated:
                    p_iter_dzndz(Z_xr, ref_zn_xr, ref_dzndz)
                else:
                    p_iter_dzndz(Z, ref_zn, ref_dzndz)

            #------------------------------------------------------------------
            # zn subblok
            if xr_detect_activated:
                p_iter_zn(Z_xr, ref_zn_xr, c_xr)# in place mod
                # std is used for div condition 
                Z[zn] = fsxn.to_standard(Z_xr[zn])
            else:
                p_iter_zn(Z, ref_zn, c)

            # Increment w_iter just before the stopping conditions
            w_iter += 1
            if w_iter >= ref_order:
                w_iter = w_iter % ref_order

            #==================================================================
            # Stopping condition: maximum iter reached
            if n_iter >= max_iter:
                stop[0] = reason_max_iter
                break

            #==================================================================
            # Stopping condition: Interior points detection (dzndz)
            if calc_dzndz:
                if nullify_dZndz:
                    ref_dzndz_next = dZndz_path[0]
                else:
                    ref_dzndz_next = dZndz_path[w_iter]
                    if n_iter == ref_order:
                        ref_dzndz_next = dZndz_path[w_wraped]

                if xr_detect_activated:
                    ZdZ = Z_xr[dzndz] + ref_dzndz_next
                    bool_stationnary = (
                        fsxn.extended_abs2(ZdZ)
                        < epsilon_stationnary_sq
                    )
                else:
                    ZdZ = Z[dzndz] + ref_dzndz_next
                    bool_stationnary = (
                        ZdZ.real ** 2 + ZdZ.imag ** 2
                        < epsilon_stationnary_sq
                    )
                if bool_stationnary:
                    stop[0] = reason_stationnary
                    break

            #==================================================================
            # Stopping condition: divergence
            # ZZ = "Total" z + dz

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
            
            # Storing the orbit for future use
            if calc_orbit:
                div = n_iter // backshift
                if div > div_shift:
                    div_shift = div
                    orbit_i2 = orbit_i1
                    orbit_zn2 = orbit_zn1
                    orbit_i1 = n_iter
                    orbit_zn1 = ZZ

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
                    if calc_dzndc:
                        Z_xr[dzndc] = Z_xr[dzndc] + dZndc_path[w_iter]
                    if calc_dzndz:
                        if not(nullify_dZndz):
                            added_index = w_iter
                            if n_iter == ref_order:
                                added_index = w_wraped
                            Z_xr[dzndz] = (
                                Z_xr[dzndz] + dZndz_path[added_index]
                            )
                        nullify_dZndz = True

                else:
                    if calc_dzndc:
                        # not a cycle, dZndc_path[0] == 0
                        Z[dzndc] += dZndc_path[w_iter]
                    if calc_dzndz:
                        if not(nullify_dZndz):
                            if n_iter == ref_order:
                                Z[dzndz] += dZndz_path[w_wraped]
                            else:
                                Z[dzndz] += dZndz_path[w_iter]
                        nullify_dZndz = True

                w_iter = 0
                continue

            #==================================================================
            # Glitch correction - "Dynamic glitch"
            bool_dyn_rebase = (
                (abs(ZZ.real) <= abs(Z[zn].real))
                and (abs(ZZ.imag) <= abs(Z[zn].imag))
            )
            if bool_dyn_rebase:# and False:
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
                        # /!\ keep this, needed for next BLA step
                        Z[zn] = fsxn.to_standard(ZZ_xr)
                        if calc_dzndc:
                            Z_xr[dzndc] = (
                                Z_xr[dzndc] + dZndc_path[w_iter]
                            )
                        if calc_dzndz:
                            if not(nullify_dZndz):
                                added_index = w_iter
                                if n_iter == ref_order:
                                    added_index = w_wraped
                                Z_xr[dzndz] = (
                                    Z_xr[dzndz] + dZndz_path[added_index]
                                )
                            nullify_dZndz = True

                        w_iter = 0
                        continue
                else:
                    # No risk of underflow - safe to rebase
                    Z[zn] = ZZ
                    if calc_dzndc:
                        Z[dzndc] += dZndc_path[w_iter]
                    if calc_dzndz:
                        if not(nullify_dZndz):
                            if n_iter == ref_order:
                                Z[dzndz] += dZndz_path[w_wraped]
                            else:
                                Z[dzndz] += dZndz_path[w_iter]
                        nullify_dZndz = True

                    w_iter = 0
                    continue

        # End of iterations for this point
        U[0] = w_iter

        if xr_detect_activated:
            Z[zn] = fsxn.to_standard(Z_xr[zn]) + Zn_path[w_iter]
            if calc_dzndc: # and n_iter > 1:
                Z[dzndc] = fsxn.to_standard(Z_xr[dzndc] + dZndc_path[w_iter])
        else:
            Z[zn] += Zn_path[w_iter]
            if calc_dzndc:  # and n_iter > 1:
                # if n_iter != 1:
                Z[dzndc] += dZndc_path[w_iter]

        if proj_dzndc_modifier is not None:
            Z[dzndc] *= proj_dzndc_modifier(c_pix)
        
        if calc_orbit: # Finalizing the orbit
            zn_orbit = orbit_zn2
            CC = c + Zn_path[1]
            while orbit_i2 < n_iter - backshift:
                zn_orbit = zn_iterate(zn_orbit, CC)
                orbit_i2 += 1
            Z[i_znorbit] = zn_orbit

        return n_iter

    return numba_impl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Non-holomorphic perturbation iterations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@numba.njit(nogil=True, fastmath=True, error_model="numpy")
def numba_cycles_perturb_BS(
    c_pix, Z, U, stop_reason, stop_iter,
    initialize, iterate,
    Zn_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
    has_xr, ref_index_xr, refx_xr, refy_xr, ref_div_iter, ref_order,
    driftx_xr, drifty_xr, dx_xr, proj_impl, lin_mat, lin_scale_xr,
    kc, M_bla, r_bla, bla_len, stages_bla, # suppressed P, n_iter_init
    proj_dzndc_modifier,
    _interrupted
):

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
            c_pix[ipt], proj_impl, lin_mat, lin_scale_xr, driftx_xr, drifty_xr
        )
        stop_pt = stop_reason[:, ipt]

        initialize(Zpt, Z_xr)

        n_iter = iterate(
            apt, bpt, a_xr, b_xr, Zpt, Z_xr, Z_xr_trigger,
            Upt, stop_pt,
            Zn_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
            has_xr, ref_index_xr, refx_xr, refy_xr, ref_div_iter, ref_order,
            refpath_ptr, out_is_xr, out_xr, M_bla, r_bla, bla_len, stages_bla,
            proj_dzndc_modifier, c_pix[ipt]
        )
        stop_iter[0, ipt] = n_iter
        stop_reason[0, ipt] = stop_pt[0]

        if _interrupted[0]:
            return USER_INTERRUPTED
    return 0


def numba_initialize_BS(xn, yn, dxnda, dxndb, dynda, dyndb):
    @numba.njit
    def numba_init_impl(Z, Z_xr):
        """
        Only : initialize the Xrange
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
    p_iter_zn, p_iter_hessian,
    calc_orbit, i_xnorbit, i_ynorbit, backshift, xnyn_iterate # new args
):

    @numba.njit
    def numba_impl(
        a, b, a_xr, b_xr, Z, Z_xr, Z_xr_trigger,
        U, stop,
        Zn_path, dXnda_path, dXndb_path, dYnda_path, dYndb_path,
        has_xr, ref_index_xr, refx_xr, refy_xr, ref_div_iter, ref_order,
        refpath_ptr, out_is_xr, out_xr, M_bla, r_bla, bla_len, stages_bla,
        proj_dzndc_modifier, c_pix
    ):
        """
        Parameters
        ----------
        c, c_xr: c and it "Xrange" counterparts
        Z, Z_xr: idem for result vector Z
        Z_xr_trigger : bolean, activated when Z_xr need to be used
        """
        record_zero = Xr_float_template.repeat(1)[0]

        # Wrapped iteration if we reach the cycle order 
        w_iter = 0
        n_iter = 0
        if w_iter >= ref_order:
            w_iter = w_iter % ref_order
        
        if calc_orbit:
            div_shift = 0
            orbit_xn1 = Z[xn]
            orbit_xn2 = Z[xn]
            orbit_yn1 = Z[yn]
            orbit_yn2 = Z[yn]
            orbit_i1 = 0
            orbit_i2 = 0
        
        # We know that :
        # ref_orbit_len = max_iter + 1 >= ref_div_iter
        # if order is not None:
        #    ref_orbit_len = min(order, ref_orbit_len)
        ref_orbit_len = Zn_path.shape[0]
        first_invalid_index = min(ref_orbit_len, ref_div_iter, ref_order)
        M_out = np.empty((8,), dtype=np.float64)
        
        # Index to keep track if we use the wraped index of dzndc
        bool_dyn_rebase = True

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
                        apply_BLA_BS(M_out, Z_xr, a_xr, b_xr, xn, yn)
                        # /!\ keep this, needed for next BLA step
                        Z[xn] = fsxn.to_standard(Z_xr[xn])
                        Z[yn] = fsxn.to_standard(Z_xr[yn])
                        if calc_hessian:
                            apply_BLA_deriv_BS(M_out, Z_xr, a_xr, b_xr,
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
                    out_is_xr, out_xr, 0, 1
                )
                ref_xn_xr, ref_yn_xr = ensure_xr_BS(
                    ref_zn, out_xr[0], out_xr[1], out_is_xr[0]
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

                if bool_dyn_rebase: # Avoid the wraped value at 0
                    if xr_detect_activated:
                        ref_dxnda = record_zero # Record type
                        ref_dxndb = record_zero # Record type
                        ref_dynda = record_zero # Record type
                        ref_dyndb = record_zero # Record type
                    else:
                        ref_dxnda = 0.
                        ref_dxndb = 0.
                        ref_dynda = 0.
                        ref_dyndb = 0.

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
            if w_iter >= ref_order:
                w_iter = w_iter % ref_order

            if xr_detect_activated:
                ref_zn_next = ref_path_get_BS(
                    Zn_path, w_iter,
                    has_xr, ref_index_xr, refx_xr, refy_xr, refpath_ptr,
                    out_is_xr, out_xr, 0, 1
                )
            else:
                ref_zn_next = Zn_path[w_iter]

            # div condition computation with std only
            XX = Z[xn] + ref_zn_next.real
            YY = Z[yn] + ref_zn_next.imag
            full_sq_norm = XX ** 2 + YY ** 2
            
            # Storing the orbit for future use
            if calc_orbit:
                div = n_iter // backshift
                if div > div_shift:
                    div_shift = div
                    orbit_i2 = orbit_i1
                    orbit_xn2 = orbit_xn1
                    orbit_yn2 = orbit_yn1
                    orbit_i1 = n_iter
                    orbit_xn1 = XX
                    orbit_yn1 = YY

            # Flagged as 'diverging'
            bool_infty = (full_sq_norm > M_divergence_sq)
            if bool_infty:
                stop[0] = reason_M_divergence
                break

            #==================================================================
            # Glitch correction - reference point diverging
            
            if (w_iter >= ref_div_iter - 1):
                # Rebasing - we are already big no underflow risk
                Z[xn] = XX
                Z[yn] = YY

                if xr_detect_activated:
                    Z_xr[xn] = fsxn.to_Xrange_scalar(XX)
                    Z_xr[yn] = fsxn.to_Xrange_scalar(YY)
                    # not a cycle, dZndc_path[0] == 0
                    # assert (dXnda_path[w_iter] == 0.)
                    if calc_hessian:
                        Z_xr[dxnda] = Z_xr[dxnda] + dXnda_path[w_iter]
                        Z_xr[dxndb] = Z_xr[dxndb] + dXndb_path[w_iter]
                        Z_xr[dynda] = Z_xr[dynda] + dYnda_path[w_iter]
                        Z_xr[dyndb] = Z_xr[dyndb] + dYndb_path[w_iter]
                else:
                    if calc_hessian:
                        # not a cycle, dZndc_path[0] == 0
                        Z[dxnda] += dXnda_path[w_iter]
                        Z[dxndb] += dXndb_path[w_iter]
                        Z[dynda] += dYnda_path[w_iter]
                        Z[dyndb] += dYndb_path[w_iter]

                w_iter = 0
                continue

            #==================================================================
            # Glitch correction - "dynamic glitch"
            bool_dyn_rebase = (
                (abs(XX) <= abs(Z[xn])) and (abs(YY) <= abs(Z[yn]))
            )
            if bool_dyn_rebase:
                if xr_detect_activated:
                    # Can we *really* rebase ??
                    # Note: if Z[zn] underflows we might miss a rebase
                    # So we cast everything to xr
                    X_xrn = Z_xr[xn]
                    Y_xrn = Z_xr[yn]
                    if out_is_xr[0]:
                        # Reference underflows, use available xr ref
                        XX_xr = X_xrn + out_xr[0]
                        YY_xr = Y_xrn + out_xr[1]
                    else:
                        XX_xr = X_xrn + ref_zn_next.real
                        YY_xr = Y_xrn + ref_zn_next.imag

                    bool_dyn_rebase_xr = (
                        (XX_xr * XX_xr + YY_xr * YY_xr)
                        <= (X_xrn * X_xrn + Y_xrn * Y_xrn)
                    )
                    if bool_dyn_rebase_xr:
                        Z_xr[xn] = XX_xr
                        Z_xr[yn] = YY_xr
                        # /!\ keep this, needed for next BLA step
                        Z[xn] = fsxn.to_standard(XX_xr)
                        Z[yn] = fsxn.to_standard(YY_xr)

                        if calc_hessian:
                            Z_xr[dxnda] = (
                                Z_xr[dxnda] + dXnda_path[w_iter]
                            )
                            Z_xr[dxndb] = (
                                Z_xr[dxndb] + dXndb_path[w_iter]
                            )
                            Z_xr[dynda] = (
                                Z_xr[dynda] + dYnda_path[w_iter]
                            )
                            Z_xr[dyndb] = (
                                Z_xr[dyndb] + dYndb_path[w_iter]
                            )
                        w_iter = 0
                        continue
                else:
                    # No risk of underflow - safe to rebase
                    Z[xn] = XX
                    Z[yn] = YY
                    if calc_hessian:
                        Z[dxnda] += dXnda_path[w_iter]
                        Z[dxndb] += dXndb_path[w_iter]
                        Z[dynda] += dYnda_path[w_iter]
                        Z[dyndb] += dYndb_path[w_iter]
                    w_iter = 0
                    continue

        # End of iterations for this point
        U[0] = w_iter
        
        # Total zn = Zn + zn
        ref_zn = Zn_path[w_iter]
        if xr_detect_activated:
            Z[xn] = fsxn.to_standard(Z_xr[xn] + ref_zn.real)
            Z[yn] = fsxn.to_standard(Z_xr[yn] + ref_zn.imag)
            if calc_hessian:
                Z[dxnda] = fsxn.to_standard(
                        Z_xr[dxnda] + dXnda_path[w_iter])
                Z[dxndb] = fsxn.to_standard(
                        Z_xr[dxndb] + dXndb_path[w_iter])
                Z[dynda] = fsxn.to_standard(
                        Z_xr[dynda] + dYnda_path[w_iter])
                Z[dyndb] = fsxn.to_standard(
                        Z_xr[dyndb] + dYndb_path[w_iter])
        else:
            Z[xn] += ref_zn.real
            Z[yn] += ref_zn.imag
            if calc_hessian:
                Z[dxnda] += dXnda_path[w_iter]
                Z[dxndb] += dXndb_path[w_iter]
                Z[dynda] += dYnda_path[w_iter]
                Z[dyndb] += dYndb_path[w_iter]

        if proj_dzndc_modifier is not None:
            Z[dxnda] *= proj_dzndc_modifier(c_pix)
            Z[dxndb] *= proj_dzndc_modifier(c_pix)
            Z[dynda] *= proj_dzndc_modifier(c_pix)
            Z[dyndb] *= proj_dzndc_modifier(c_pix)

        if calc_orbit: # Finalizing the orbit
            xn_orbit = orbit_xn2
            yn_orbit = orbit_yn2
            AA = a + np.real(Zn_path[1])
            BB = b + np.imag(Zn_path[1])
            while orbit_i2 < n_iter - backshift:
                xn_orbit, yn_orbit = xnyn_iterate(xn_orbit, yn_orbit, AA, BB)
                orbit_i2 += 1
            Z[i_xnorbit] = xn_orbit
            Z[i_ynorbit] = yn_orbit

        return n_iter

    return numba_impl

@numba.njit(nogil=True, cache=True)
def apply_BLA_BS(M, Z, a, b, xn, yn):
    Z_xn = M[0] * Z[xn] + M[1] * Z[yn] + M[4] * a + M[5] * b
    Z_yn = M[2] * Z[xn] + M[3] * Z[yn] + M[6] * a + M[7] * b
    Z[xn] = Z_xn
    Z[yn] = Z_yn

@numba.njit(nogil=True, cache=True)
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
# Bilinear approximation
# Note: the bilinear arrays being cheap, they  will not be stored but
# re-computed if needed

@numba.njit(nogil=True, cache=True, fastmath=True, error_model="numpy")
def numba_make_BLA(Zn_path, dfdz, kc, eps):
    """
    Generates a compressed BLA tree with
    - bilinear approximation coefficients A and B
    - validaty radius
        z_n+2**stg = f**stg(z_n, c) with |c| < r_stg_n is approximated by 
        z_n+2**stg = A_stg_n * z_n + B_stg_n * c
    """
    # number of needed "stages" is (ref_orbit_len).bit_length()
    # number of needed "stages" is (ref_orbit_len).bit_length()
    kc_std = fsxn.to_standard(kc[0])
    ref_orbit_len = Zn_path.shape[0]

    # Compressed quantities
    k_comp = 1 << STG_COMPRESSED
    comp_bla_len = ref_orbit_len // k_comp    

    bla_dim = 2
    M_bla = np.zeros((2 * comp_bla_len, bla_dim), dtype=numba.complex128)
    r_bla = np.zeros((2 * comp_bla_len,), dtype=numba.float64)

    # The temporary tree structure to get to the stored index
    temp_M_bla = np.zeros((2 * k_comp, bla_dim), dtype=numba.complex128)
    temp_r_bla = np.zeros((2 * k_comp,), dtype=numba.float64)
    temp_M_bla_template = np.copy(temp_M_bla)
    temp_r_bla_template = np.copy(temp_r_bla)

    for i in range(0, comp_bla_len):
        i_0 = 2 * i # BLA index for (i, 0)
        temp_M_bla = np.copy(temp_M_bla_template)
        temp_r_bla = np.copy(temp_r_bla_template)

        for j in range(0, k_comp):
            j_0 = 2 * j # BLA index for (j, 0)

            # Define a BLA_step by:
            #       [ M[0]    0     0]          [dzndc]
            #  M =  [    0 M[0]  M[1]]     Zn = [   zn]
            #       [    0    0     1]          [    c]
            #
            #  Z_(n+1) = M * Zn

            Zn_i = Zn_path[i * k_comp + j]
            temp_M_bla[j_0, 0] = dfdz(Zn_i)
            temp_M_bla[j_0, 1] = 1.
            temp_r_bla[j_0] = eps * abs(temp_M_bla[j_0, 0])

        # Combine step for temporary tree 'compression'
        for stg in range(1, STG_COMPRESSED + 1):
            combine_BLA(temp_M_bla, temp_r_bla, kc_std, stg, k_comp, eps)
        
        stored_index = k_comp - 1
        for d in range(bla_dim):
            M_bla[i_0, d] = temp_M_bla[stored_index, d]
        r_bla[i_0] = temp_r_bla[stored_index]

    # Combine step for stored data
    stages = _stages_bla(ref_orbit_len) 
    for stg in range(1, stages - STG_COMPRESSED):
        combine_BLA(M_bla, r_bla, kc_std, stg, comp_bla_len, eps)

    return M_bla.reshape(-1), r_bla, 2 * comp_bla_len, stages


@numba.njit(nogil=True, cache=True, fastmath=True, error_model="numpy")
def numba_make_BLA_BS(Zn_path, dfxdx, dfxdy, dfydx, dfydy, kc, eps):
    """
    Generates a compressed BLA tree with
    - bilinear approximation coefficients A and B
    - validaty radius
        z_n+2**stg = f**stg(z_n, c) with |c| < r_stg_n is approximated by 
        z_n+2**stg = A_stg_n * z_n + B_stg_n * c
    """
    # number of needed "stages" is (ref_orbit_len).bit_length()
    # number of needed "stages" is (ref_orbit_len).bit_length()
    kc_std = fsxn.to_standard(kc[0])
    ref_orbit_len = Zn_path.shape[0]

    # Compressed quantities
    k_comp = 1 << STG_COMPRESSED
    comp_bla_len = ref_orbit_len // k_comp    

    bla_dim = 8
    M_bla = np.zeros((2 * comp_bla_len, bla_dim), dtype=numba.float64)
    r_bla = np.zeros((2 * comp_bla_len,), dtype=numba.float64)

    # The temporary tree structure to get to the stored index
    temp_M_bla = np.zeros((2 * k_comp, bla_dim), dtype=numba.float64)
    temp_r_bla = np.zeros((2 * k_comp,), dtype=numba.float64)
    temp_M_bla_template = np.copy(temp_M_bla)
    temp_r_bla_template = np.copy(temp_r_bla)

    for i in range(0, comp_bla_len):
        i_0 = 2 * i # BLA index for (i, 0)
        temp_M_bla = np.copy(temp_M_bla_template)
        temp_r_bla = np.copy(temp_r_bla_template)

        for j in range(0, k_comp):
            j_0 = 2 * j # BLA index for (j, 0)
            
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
    
            Zn_i = Zn_path[i * k_comp + j]
            Xn_i = Zn_i.real
            Yn_i = Zn_i.imag
    
            temp_M_bla[j_0, 0] = dfxdx(Xn_i, Yn_i)
            temp_M_bla[j_0, 1] = dfxdy(Xn_i, Yn_i)
            temp_M_bla[j_0, 2] = dfydx(Xn_i, Yn_i)
            temp_M_bla[j_0, 3] = dfydy(Xn_i, Yn_i)
    
            temp_M_bla[j_0, 4] = 1.
            temp_M_bla[j_0, 5] = 0.
            temp_M_bla[j_0, 6] = 0.
            temp_M_bla[j_0, 7] = -1.

            temp_r_bla[j_0] =  eps * min(abs(Xn_i), abs(Yn_i))  

        # Combine step for temporary tree 'compression'
        for stg in range(1, STG_COMPRESSED + 1):
            combine_BLA_BS(temp_M_bla, temp_r_bla, kc_std, stg, k_comp, eps)

        stored_index = k_comp - 1
        for d in range(bla_dim):
            M_bla[i_0, d] = temp_M_bla[stored_index, d]
        r_bla[i_0] = temp_r_bla[stored_index]

    # Combine step for stored data
    stages = _stages_bla(ref_orbit_len) 
    for stg in range(1, stages - STG_COMPRESSED):
        combine_BLA_BS(M_bla, r_bla, kc_std, stg, comp_bla_len, eps)

    return M_bla.reshape(-1), r_bla, 2 * comp_bla_len, stages


@numba.njit(nogil=True, cache=True)
def _stages_bla(ref_orbit_len):
    """
    number of needed "stages" (ref_orbit_len).bit_length()
    """
    return int(np.ceil(np.log2(ref_orbit_len)))

@numba.njit(nogil=True, cache=True, fastmath=True, error_model="numpy")
def combine_BLA(M, r, kc_std, stg, ref_orbit_len, eps):
    """ Populate successive stages of a BLA tree
    A_bla, B_bla, r_bla : data of the BLA tree
    kc : majorant of |c|
    stg : stage of the tree that is populated by merging (stg - 1) items
    ref_orbit_len : the len for the reference orbit
    """
    # Combine all BVA at stage stg-1 to make stage stg with stg > 0
    step = (1 << stg)

    for i in range(0, ref_orbit_len - step + 1, step): # +1 for power of 2 case
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
        # Valid if:
        # |A1 z0| + |B1 c| < r2
        # |z0| < |r2 - |B1 c|| / |A1|

        # Note: "Strict inequality factor" of 0.95 is mandatory to pass the
        # glitch test

        mA1 = np.abs(M[index1, 0])
        mB1 = np.abs(M[index1, 1])
        r2_backw = 0.95 * max(0., (r2 - mB1 * kc_std) / max(mA1, eps))
        r[index_res] = min(r1, r2_backw)


@numba.njit(nogil=True, cache=True)
def combine_BLA_BS(M, r, kc_std, stg, ref_orbit_len, eps):
    """ Populate successive stages of a BLA tree
    A_bla, B_bla, r_bla : data of the BLA tree
    kc : majorant of |c|
    stg : stage of the tree that is populated by merging (stg - 1) items
    ref_orbit_len : the len for the reference orbit
    """
    # Combine all BVA at stage stg-1 to make stage stg with stg > 0
    step = (1 << stg)

    for i in range(0, ref_orbit_len - step + 1, step):
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
        r2_backw = 0.95 * max(0., (r2 - mB1 * kc_std) / max(mA1, eps))
        r[index_res] = min(r1, r2_backw)


@numba.njit(nogil=True, cache=True)
def BLA_index(i, stg):
    """
    Return the indices in BLA table for this iteration and stage
    this is the jump from i to j = i + (1 << stg)
    """
    return (2 * i) + ((1 << stg) - 1)

@numba.njit(nogil=True, cache=True)
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
                loc = 2 * index_bla
                M_out[0] = M_bla[loc]
                M_out[1] = M_bla[loc + 1]

                return step
            else:
                for i in range(8):
                    loc = 8 * index_bla
                    M_out[i] = M_bla[loc + i]

                return step
    return 0 # No BLA applicable

@numba.njit(nogil=True, cache=True)
def need_xr(x_std):
    """
    True if norm L-inf of std is lower than xrange_zoom_level
    """
    return (
        (abs(np.real(x_std)) < fs.settings.xrange_zoom_level)
         and (abs(np.imag(x_std)) < fs.settings.xrange_zoom_level)
    )

@numba.njit(nogil=True, cache=True)
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

@numba.njit(nogil=True, cache=True)
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

@numba.njit(nogil=True, cache=True)
def ref_path_c_from_pix(pix, proj_impl, lin_mat, lin_scale_xr, drift_xr):
    """
    Returns the true c (coords from ref point) from the pixel coords
    ie c = C - refp
    
    Parameters
    ----------
    pix :  complex
        pixel location in fraction of dx
        
    Returns
    -------
    c, c_xr : c value as complex and as Xrange
    """
    c_xr = lin_proj_impl(lin_mat, lin_scale_xr, proj_impl(pix)) + drift_xr[0]
    return fsxn.to_standard(c_xr), c_xr


@numba.njit(nogil=True, cache=True)
def std_C_from_pix(pix, proj_impl, lin_mat, lin_scale_std, center):
    """
    Returns the true C (C = cref + dc) from the pixel coords

    Parameters
    ----------
    pix :  complex
        pixel location in fraction of dx

    Returns
    -------
    C : Full C value, as complex
    """
    return center + lin_proj_impl(lin_mat, lin_scale_std, proj_impl(pix))


@numba.njit(nogil=True, cache=True)
def fill1d_std_C_from_pix(
        c_pix, proj_impl, lin_mat, lin_scale_std, center, c_out):
    """ same as std_C_from_pix but fills in-place a 1d vec """
    nx = c_pix.shape[0]
    for i in range(nx):
        c_out[i] = std_C_from_pix(
            c_pix[i], proj_impl, lin_mat, lin_scale_std, center
        )

@numba.njit(nogil=True, cache=True)
def ref_path_c_from_pix_BS(
        pix, proj_impl, lin_mat, lin_scale_xr, driftx_xr, drifty_xr):
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
    c_xr = lin_proj_impl(lin_mat, lin_scale_xr, proj_impl(pix))
    a_xr = np.real(c_xr) + driftx_xr[0]
    b_xr = np.imag(c_xr) + drifty_xr[0]
    return fsxn.to_standard(a_xr), fsxn.to_standard(b_xr), a_xr, b_xr

@numba.njit(nogil=True, cache=True)
def numba_dZndc_path(Zn_path, has_xr, ref_index_xr, ref_xr,
                    ref_div_iter, ref_order, dfdz,
                    scale_deriv_xr, xr_detect_activated):
    """
    Compute dZndc in Xr, or std precision, depending on xr_detect_activated
    """
    # Development note: the dx_xr which is imposed at each step
    # imposes the deivative scaling
    ref_orbit_len = Zn_path.shape[0]
    valid_pts = min(ref_orbit_len, ref_div_iter)

    xr_act = xr_detect_activated
    # dx = fsxn.to_standard(dx_xr[0])
    scale_deriv = fsxn.to_standard(scale_deriv_xr[0])

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
            dZndc_xr_path[i] = dfdz(ref_zn_xr) * dZndc_xr_path[i - 1] + scale_deriv_xr[0] 

        if (i == ref_order - 1):
            # /!\ We have a cycle, use the "wrapped" value at 0
            # Note that this value will be used... a lot !
            ref_zn = ref_path_get(
                Zn_path, i,
                has_xr, ref_index_xr, ref_xr, refpath_ptr,
                out_is_xr, out_xr, 0
            )
            ref_zn_xr = ensure_xr(ref_zn, out_xr[0], out_is_xr[0])
            dZndc_xr_path[0] = dfdz(Zn_path[i]) * dZndc_xr_path[i] + scale_deriv_xr[0] # Note: the wraped is [0] !!

    else:
        dZndc_path = np.zeros((ref_orbit_len,), dtype=numba.complex128)
        dZndc_xr_path = Xr_template.repeat(1) # dummy
        for i in range(1, valid_pts):
            dZndc_path[i] = dfdz(Zn_path[i - 1]) * dZndc_path[i - 1] + scale_deriv
        if (i == ref_order - 1):
            # /!\ We have a cycle, use the "wrapped" value at 0
            # Note that this value will be used... a lot !
            dZndc_path[0] = dfdz(Zn_path[i]) * dZndc_path[i] + scale_deriv # Note: the wraped is [0] !!

    return dZndc_path, dZndc_xr_path


@numba.njit(nogil=True, cache=True)
def numba_dZndc_path_BS(Zn_path, has_xr, ref_index_xr, refx_xr, refy_xr,
                    ref_div_iter, ref_order, dfxdx, dfxdy, dfydx, dfydy,
                    scale_deriv_xr, xr_detect_activated):
    """
    Compute dXndb, dXnda, dYnda , dYndb in Xr, or std precision, depending on
    xr_detect_activated
    """
    ref_orbit_len = Zn_path.shape[0]
    valid_pts = min(ref_orbit_len, ref_div_iter)
    scale_deriv = fsxn.to_standard(scale_deriv_xr[0])

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
                from_i, to_i, scale_deriv_xr[0],
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
                from_i, to_i, scale_deriv_xr[0],
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
                from_i, to_i, scale_deriv,
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
                from_i, to_i, scale_deriv,
                Xn, Yn, dfxdx, dfxdy, dfydx, dfydy
            )

    return (
        dXnda_path, dXndb_path, dYnda_path, dYndb_path,
        dXnda_xr_path, dXndb_xr_path, dYnda_xr_path, dYndb_xr_path
    )

@numba.njit(nogil=True, cache=True)
def incr_deriv_ref_BS(
    dXnda_path, dXndb_path, dYnda_path, dYndb_path,
    from_i, to_i, dx,
    Xn, Yn, dfxdx, dfxdy, dfydx, dfydy,
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


@numba.njit(nogil=True, cache=True)
def numba_dZndz_path(Zn_path, has_xr, ref_index_xr, ref_xr,
                    ref_div_iter, ref_order, dfdz, xr_detect_activated):
    """
    Compute dZndz in Xr, or std precision, depending on xr_detect_activated
    Same as dZndc except the +dx is missing
    """
    ref_orbit_len = Zn_path.shape[0]
    valid_pts = min(ref_orbit_len, ref_div_iter)

    xr_act = xr_detect_activated

    if xr_act:
        dZndz_path = np.zeros((1,), dtype=numba.complex128) # dummy
        dZndz_xr_path = Xr_template.repeat(ref_orbit_len + 1)
        dZndz_xr_path[1] = fsxn.one()

        refpath_ptr = np.zeros((2,), dtype=numba.int32)
        out_is_xr = np.zeros((1,), dtype=numba.bool_)
        out_xr = Xr_template.repeat(1)

        for i in range(2, valid_pts):
            ref_zn = ref_path_get(
                Zn_path, i - 1,
                has_xr, ref_index_xr, ref_xr, refpath_ptr,
                out_is_xr, out_xr, 0
            )
            ref_zn_xr = ensure_xr(ref_zn, out_xr[0], out_is_xr[0])
            dZndz_xr_path[i] = dfdz(ref_zn_xr) * dZndz_xr_path[i - 1]

        # /!\ Store the "wrapped" value at last pos
        ref_zn = ref_path_get(
            Zn_path, i,
            has_xr, ref_index_xr, ref_xr, refpath_ptr,
            out_is_xr, out_xr, 0
        )
        ref_zn_xr = ensure_xr(ref_zn, out_xr[0], out_is_xr[0])
        dZndz_xr_path[ref_orbit_len] = dfdz(ref_zn_xr) * dZndz_xr_path[i]

    else:
        dZndz_path = np.zeros((ref_orbit_len + 1,), dtype=numba.complex128)
        dZndz_xr_path = Xr_template.repeat(1) # dummy
        dZndz_path[1] = 1.

        for i in range(2, valid_pts):
            dZndz_path[i] = dfdz(Zn_path[i - 1]) * dZndz_path[i - 1]

        # /!\ Store the "wrapped" value at last pos
        dZndz_path[ref_orbit_len] = dfdz(Zn_path[i]) * dZndz_path[i]

    return dZndz_path, dZndz_xr_path


@numba.njit(nogil=True, cache=True)
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


@numba.njit(nogil=True, cache=True)
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

# Linear projection routines
@numba.njit(cache=True, fastmath=True, nogil=True)
def lin_proj_impl(lin_mat, linscale, z):
    x = z.real
    y = z.imag
    x1 = lin_mat[0, 0] * x + lin_mat[0, 1] * y
    y1 = lin_mat[1, 0] * x + lin_mat[1, 1] * y
    return  linscale[0] * complex(x1, y1)

#@numba.njit(cache=True, fastmath=True, nogil=True)
#def lin_proj_impl_noscale(lin_mat, z):
#    x = z.real
#    y = z.imag
#    x1 = lin_mat[0, 0] * x + lin_mat[0, 1] * y
#    y1 = lin_mat[1, 0] * x + lin_mat[1, 1] * y
#    return complex(x1, y1)

def mpc_lin_proj_impl_noscale(lin_mat, x, y):
    x1 = lin_mat[0, 0] * x + lin_mat[0, 1] * y
    y1 = lin_mat[1, 0] * x + lin_mat[1, 1] * y
    return mpmath.mpc(x1, y1)