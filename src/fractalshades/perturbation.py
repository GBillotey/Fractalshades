# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import copy
import mpmath
import random

import fractalshades as fs
import fractalshades.numpy_utils.xrange as fsx
import fractalshades.numpy_utils.numba_xr # as fsxn
import fractalshades.settings as fssettings
import fractalshades.utils as fsutils
import fractalshades.postproc as fspp
#force_recompute_SA = True

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
        

    @fsutils.zoom_options
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
            number of significant digits to use for reference point calculation
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
        mpmath.mp.dps = precision
        # In case the user inputs were strings, we override with mpmath scalars
        self.x = mpmath.mpf(x)
        self.y = mpmath.mpf(y)
        self.dx = mpmath.mpf(dx)
        # Lazzy dictionary of reference point pathes
        self._ref_array = {}

    @property
    def Xrange_Z_path(self):
        """ Return whether the full precision orbit shall be exported to 
        Xrange datatype """
        if self.base_complex_type == np.complex128:
            return (self.dx < 1.e-300 or self.Xrange_complex_type)
        else:
            raise NotImplementedError(self.base_complex_type)


    def diff_c_chunk(self, chunk_slice, iref, calc_name,
                     ensure_Xr=False):
        """
        Returns a 2d chunk of c_vec for the calculation
        Parameters
         - chunk_span
         - data_type: expected one of np.float64, np.longdouble
        
        Returns: 
        c_vec : [chunk_size x chunk_size] 1d-vec of type datatype
        
                as a "delta" wrt reference pt iref
        """
        offset_x, offset_y = self.chunk_offset(chunk_slice, ensure_Xr)
        FP_params = self.reload_ref_point(iref, calc_name, scan_only=True)
        drift_rp = (self.x + 1j * self.y) - FP_params["ref_point"] #.imag
        print("DIFF c_chunk Shifting c with respect to iref", iref, drift_rp)
        print("Shift in pc", drift_rp.real / self.dx, drift_rp.imag / self.dy)
        
        if self.Xrange_complex_type or ensure_Xr:
            drift_rp = fsx.mpc_to_Xrange(drift_rp, dtype=self.base_complex_type)
            diff = fsx.Xrange_array._build_complex(offset_x, offset_y)
        else:
            drift_rp = complex(drift_rp)
            diff = np.empty(offset_x.shape, dtype=self.base_complex_type)
            diff.real = offset_x
            diff.imag = offset_y

        return diff + drift_rp

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
        
        FP_codes, ref_array = self.get_ref_array(calc_name)#[:, :, :]
        full_Z = Z.copy()
        for key, val in complex_dic.items():
            # If this field in a full precision array, we add it :
            if key in FP_codes:
                FP_val = FP_codes.index(key)
                full_Z[val, :] += np.ravel(ref_array[U[int_dic["iref"], :],
                                            stop_iter[0, :], FP_val])
        print("U[int_dic[\"iref\"]]", U[int_dic["iref"]])
        print("stop_iter", stop_iter[0, :])
        full_raw_data = (chunk_mask, full_Z, U, stop_reason, stop_iter)

        post_array, chunk_mask = self.postproc(postproc_keys, codes,
            full_raw_data, chunk_slice)
        return self.reshape2d(post_array, chunk_mask, chunk_slice)


    def ref_point_count(self, calc_name):
        iref = 0
        while os.path.exists(self.ref_point_file(iref, calc_name)):
            iref += 1
        return iref

    def ref_point_file(self, iref, calc_name):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        data chunk
        """
        return os.path.join(self.directory, "data",
                calc_name + "_pt{0:d}.ref".format(iref))

    def ref_point_scaling(self, iref, calc_name):
        """
        Return a scaling coefficient used as a convergence radius for serie 
        approximation, or as a reference scale for derivatives.
        
        kc: full precision, scaling coefficient
        """
        c0 = self.x + 1j * self.y
        corner_a = c0 + 0.5 * (self.dx + 1j * self.dy)
        corner_b = c0 + 0.5 * (- self.dx + 1j * self.dy)
        corner_c = c0 + 0.5 * (- self.dx - 1j * self.dy)
        corner_d = c0 + 0.5 * (self.dx - 1j * self.dy)
        ref = self.reload_ref_point(
                iref, calc_name, scan_only=True)["ref_point"]
        print("ref point for SA", ref)
        # Let take some margin
        kc = max(abs(ref - corner_a), abs(ref - corner_b),
                 abs(ref - corner_c), abs(ref - corner_d)) * 2.0
        return kc

    def save_ref_point(self, FP_params, Z_path, iref, calc_name):
        """
        Write to a dat file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.ref_point_file(iref, calc_name)
        fsutils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            print("Path computed, saving", save_path)
            pickle.dump(FP_params, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(Z_path, tmpfile, pickle.HIGHEST_PROTOCOL)

    def reload_ref_point(self, iref, calc_name, scan_only=False):
        """
        Reload arrays from a data file
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.ref_point_file(iref, calc_name)
        with open(save_path, 'rb') as tmpfile:
            FP_params = pickle.load(tmpfile)
            if scan_only:
                return FP_params
            Z_path = pickle.load(tmpfile)
        return FP_params, Z_path

    def get_ref_array(self, calc_name):
        """
        Lazzy evaluation of array compiling all ref points 'pathes'
        """
        nref = self.ref_point_count(calc_name)
        FP_params = self.reload_ref_point(0, calc_name, scan_only=True)
        max_iter = FP_params["max_iter"]
        FP_codes = FP_params["FP_codes"]
        print("FP_codes",  FP_codes)
        
        try:
            ret = self._ref_array[calc_name]
            if ret.shape[0] == nref:
                return FP_codes, ret
            else:
                print("ref_array shape mismatch", ret.shape[0], nref)
                pass
        except KeyError:
            print("ref_array not found")
            pass

        # Never computed or need update: it in this case we compute
        if self.Xrange_complex_type:
            ref_array = fsx.Xrange_array.empty(
                    [nref, max_iter + 1, len(FP_codes)],
                    dtype=self.base_complex_type)
        else:
            ref_array = np.empty([nref, max_iter + 1, len(FP_codes)],
                                  dtype=self.complex_type)

        for iref in range(nref):
            FP_params, Z_path = self.reload_ref_point(iref, calc_name)
            ref_array[iref, : , :] = Z_path[:, :]
        self._ref_array[calc_name] = ref_array
        return FP_codes, ref_array

    def SA_file(self, iref, calc_name):
        """
        Returns the file path to store or retrieve params associated with a
        series approximation
        """
        return os.path.join(self.directory, "data", calc_name +
                            "_pt{0:d}.sa".format(iref) )

    def save_SA(self, SA_params, iref, calc_name):
        """
        Write to a dat file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.SA_file(iref, calc_name)
        fsutils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            print("SA computed, saving", save_path)
            pickle.dump(SA_params, tmpfile, pickle.HIGHEST_PROTOCOL)

    def reload_SA(self, iref, calc_name):
        """
        """
        save_path = self.SA_file(iref, calc_name)
        with open(save_path, 'rb') as tmpfile:
            SA_params = pickle.load(tmpfile)
        return SA_params

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
        #  glitch_stop_index : All points with a stop_reason >= glitch_stop_index
        #          are considered glitched (early ref exit glitch, or dynamic
        #          glitch). Subset of pixel for which 
        #          stop_reason == glitch_stop_index will be sorted according to 
        #          minimum value of field `glitch_sort_key` the minimal  point
        #          used as new reference.
        #          if None, no glitch correction.
        #                
        #  glitch_sort_key : the complex (Z array) field where is stored our
        #          'priority value' to select the next reference pixel.

        if not(self.res_available()):
            # We write the param file and initialize the
            # memmaps for progress reports and calc arrays
            # It is not process safe so we dot it before entering multi-processing
            # loop
            fsutils.mkdir_p(os.path.join(self.directory, "data"))
            self.open_report_mmap()
            self.open_data_mmaps()
            self.save_params()

        calc_name = self.calc_name
        max_iter = self.max_iter
        SA_params = copy.deepcopy(self.SA_params)

        self.iref = 0

        FP_params, ref_path = self.ensure_ref_point(self.FP_loop(),
                self.max_iter, calc_name, iref=self.iref, c0=None)
        FP_params0 = self.reload_ref_point(0, calc_name, scan_only=True)

        # If SA is activated :
        # Reload SA coefficient, otherwise compute them.
        cutdeg_glitch = None
        if SA_params is not None:
            use_Taylor_shift = SA_params.pop("use_Taylor_shift", True)
            cutdeg_glitch = SA_params.pop("cutdeg_glitch", None)
            try:
                SA_params = self.reload_SA(self.iref, calc_name)
            except FileNotFoundError:
                # Initialise the path and ref point
                ref_div_iter = FP_params0.get("div_iter",
                                              FP_params0["max_iter"] + 1) # TODO  +1 ??
                SA_params = self.series_approx(self.SA_init(), self.SA_loop(),
                    SA_params, self.iref, ref_div_iter, calc_name)
                self.save_SA(SA_params, self.iref, calc_name)

        # First a standard "perturbation" cycle, no glitch correction
        self._iterate = self.iterate()
        self.cycles(chunk_slice=None, SA_params=SA_params)

        # Exit if glitch correction inactive
        if self.glitch_stop_index is None or fssettings.skip_calc:
            return

        # Glitch correction loops
        # Lopping cycles until no more glitched areas remain
        # First inspecting already computed data, completing partial loop
        # (if any). Note that super.cycles will escape pixels irefs >= iref
        glitch_sort_key = self.glitch_sort_key
        glitch_stop_index = self.glitch_stop_index

        dyn_glitched = fspp.Fractal_array(
                self, calc_name, "stop_reason",
                func=lambda x: x == glitch_stop_index)

        # We store this one as an attribute, as this is the pixels which will
        # need an update "escaped_or_glitched"
        self.all_glitched = fspp.Fractal_array(
                self, calc_name, "stop_reason",
                func=lambda x: x >= glitch_stop_index)

        glitch_sorts = fspp.Fractal_array(
                self, calc_name, glitch_sort_key, func=None)
        
        stop_iters = fspp.Fractal_array(
                self, calc_name, "stop_iter", func=None)
        
        
        header, report = self.reload_report(None, calc_name)
        print("*** REPORT")
        print(header)
        print(report)
        all_glitch_count = np.sum(report[:, header.index("total-glitched")])
        dyn_glitch_count = np.sum(report[:, header.index("dyn-glitched")])
        prev_all_glitch_count = 0 # Track the progress
        
        print("ANY glitched ? dyn / total", dyn_glitch_count, all_glitch_count)

        while ((self.iref < self.glitch_max_attempt)
               and (all_glitch_count > 0)
               and not(self.is_interrupted())
            ):
            self.iref += 1
            print("Launching glitch correction cycle, iref = ", self.iref)
            print("with glitch and escaped pixel combined count (dyn / total)",
                  dyn_glitch_count, all_glitch_count)

            # We need to define the new c0. 
            # - if no dyn glitched, use some random 
            # - If we are stuck (no progress), we use some random
            if (dyn_glitch_count > 0 and
                prev_all_glitch_count != all_glitch_count):
                glitched = dyn_glitched
                flag = 'dyn'
            else:
                glitched = self.all_glitched
                flag = 'escape'
            prev_all_glitch_count = all_glitch_count

            # Minimize sort criteria over the selected glitch
            if flag == 'dyn':
                glitch_iter = self.largest_glitch_iter(stop_iters, glitched)
                (min_glitch_chunk, min_glitch_arg
                ) = self.min_glitch_pt(
                         glitch_sorts, stop_iters, glitched, glitch_iter,
                         header, report)
            elif flag == 'escape':
                 print("ESCAPE mode, count", all_glitch_count)
                 (min_glitch_chunk, min_glitch_arg
                 ) = self.random_glitch_pt(
                     all_glitch_count, self.all_glitched, header, report)
            print("candidate",flag, min_glitch_chunk, min_glitch_arg)

            # min_glitch_arg in 1d need to translate in 2d ...
            chunk_mask = None
            if self.subset is not None:
                chunk_mask = self.subset[min_glitch_chunk]

            min_glitch_arg = self.index2d(min_glitch_arg, chunk_mask,
                                          min_glitch_chunk)
            # Now lets define ci. offset from image center is known
            offset_x, offset_y = self.chunk_offset(min_glitch_chunk)

            if self.Xrange_complex_type:
                ci = (self.x + fsx.Xrange_to_mpfc(offset_x[min_glitch_arg]) + 1j
                      * (self.y + fsx.Xrange_to_mpfc(offset_y[min_glitch_arg])))
                print("With shift from center coords:\n",
                      fsx.Xrange_to_mpfc(offset_x[min_glitch_arg]) / self.dx,
                      fsx.Xrange_to_mpfc(offset_y[min_glitch_arg]) / self.dy)
            else:
                ci = (self.x + offset_x[min_glitch_arg] + 1j
                      * (self.y + offset_y[min_glitch_arg]))
                print("With shift from center coords:\n",
                      offset_x[min_glitch_arg] / self.dx,
                      offset_y[min_glitch_arg] / self.dy)
            
            if flag == 'dyn':
                # Overall slightly counter-productive to add a Newton step here
                # (less robust), we keep the raw selected point
                FP_params, Z_path = self.ensure_ref_point(self.FP_loop(), max_iter,
                    calc_name, iref=self.iref, c0=ci, newton="None", order=-1)
            elif flag == 'escape':
                # just keep the raw point
                FP_params, Z_path = self.ensure_ref_point(self.FP_loop(), max_iter,
                    calc_name, iref=self.iref, c0=ci, newton="None", order=-1)

            if (SA_params is not None):
                print("SA_params cutdeg / iref:",
                      SA_params["cutdeg"], SA_params["iref"])
                try:
                    SA_params = self.reload_SA(self.iref, calc_name)
                except FileNotFoundError:
                    if use_Taylor_shift:
                        # We will shift the SA params coefficient from the first
                        # reference point
                        print("Shifting SA from pt0 to new reference point")
                        SA_params0 = self.reload_SA(0, calc_name)
                        P0 = SA_params0["P"]
                        dc_ref = (fsx.mpc_to_Xrange(FP_params["ref_point"]
                            - FP_params0["ref_point"], self.base_complex_type)
                            / SA_params0["kc"])
                        print("##### dc_ref", dc_ref)
                        P_shifted = []
                        for P0i in P0:
                            P_shifted += [P0i.taylor_shift(dc_ref)]
                        # For fields known at FP, we have to correct the 1st 
                        # coeff which  is by definition 0 (due to the FP ref
                        # point shift)
                        print("P_shifted[0].coeffs[0]", P_shifted[0].coeffs[0])
                        P_shifted[0].coeffs[0] = 0.
                        SA_params = {"cutdeg": SA_params0["cutdeg"],
                                     "iref": self.iref,
                                     "kc": SA_params0["kc"],
                                     "n_iter": SA_params0["n_iter"],
                                     "P": P_shifted}
                        self.save_SA(SA_params, self.iref, calc_name)
                    else:
                        # Shift of SA approx is vetoed (no 'Taylor shift')
                        # -> Fall back to full SA recompute
                        if cutdeg_glitch is not None:
                            SA_params["cutdeg"] = cutdeg_glitch
                        
                        ref_div_iter = FP_params.get("div_iter",
                                                     FP_params["max_iter"] + 1) 
                        SA_params = self.series_approx(self.SA_init(),
                            self.SA_loop(), SA_params, self.iref, ref_div_iter,
                            calc_name)
                        self.save_SA(SA_params, self.iref, calc_name)



            self.cycles(chunk_slice=None, SA_params=SA_params)

            # Recomputing the exit condition
            header, report = self.reload_report(None, calc_name)
            all_glitch_count = np.sum(report[:,
                                      header.index("total-glitched")])
            dyn_glitch_count = np.sum(report[:, header.index("dyn-glitched")])

            print("ANY glitched ? ", all_glitch_count)
            
        # Export to human-readable format
        if fs.settings.inspect_calc:
            self.inspect_calc()


    def largest_glitch_iter(self, stop_iters, glitched):
        """
        Return the stop iteration with the largest number of pixel
        stop_iters: *Fractal_Data_array* wrapping the pixel stop iteration
        glitched: *Fractal_Data_array* wrapping the dyn glitched pixel bool
        """
        max_iter = self.max_iter
        glitch_bincount = np.zeros([max_iter], dtype=np.int32)
        for chunk_slice in self.chunk_slices():
            stop_iter = stop_iters[chunk_slice]
            chunck_glitched = glitched[chunk_slice]
            glitch_bincount += np.bincount(stop_iter[chunck_glitched],
                                           minlength=max_iter)

        return np.argmax(glitch_bincount) # glitch_iter


    def min_glitch_pt(self, glitch_sorts, stop_iters, glitched, glitch_iter,
                      header, report):
        """
        Return localisation of minimal pixel in a dyn glitch
        glitch_sorts: *Fractal_Data_array* wrapping the array used to sort
        stop_iters: *Fractal_Data_array* wrapping the pixel stop iteration
        glitched: *Fractal_Data_array* wrapping the dyn glitched pixel bool
        glitch_iter: The largest dyn glitch happens at this iter
        """
        min_glitch = np.inf
        min_glitch_chunk = None
        min_glitch_arg = None

        for i, chunk_slice in enumerate(self.chunk_slices()):
            # early exit this iteration if no glitched pixel
            glitched_count = report[i, header.index("dyn-glitched")]
            if glitched_count == 0:
                continue

            glitch_sort = (glitch_sorts[chunk_slice]).real
            indices = np.arange(len(glitch_sort), dtype=np.int32)
            # keep only the glitched + good iter part
            keep = (glitched[chunk_slice]
                    & (stop_iters[chunk_slice] == glitch_iter))

            glitch_sort = glitch_sort[keep]
            if len(glitch_sort) == 0: # if has glitched pix but none from the largest
                continue
            if self.Xrange_complex_type:
                glitch_sort = glitch_sort.view(fsx.Xrange_array).to_standard()
            chunk_min = np.nanmin(glitch_sort)
            indices = indices[keep]

            if chunk_min < min_glitch:
                min_glitch_arg = indices[np.nanargmin(glitch_sort)]
                min_glitch_chunk = chunk_slice
                min_glitch = chunk_min # updates the min

        print("Minimal criteria reached at", min_glitch_chunk, min_glitch_arg)
        return min_glitch_chunk, min_glitch_arg

    def random_glitch_pt(self, glitch_count, glitched, header, report):
        """
        Return localisation of a random pixel in a glitch
        glitch_count: int, the number of glitched pixels
        glitched: *Fractal_Data_array* wrapping the dyn glitched pixel bool
        """
        rd_int = random.randrange(0, glitch_count)
        for i, chunk_slice in enumerate(self.chunk_slices()):
            glitched_count = report[i, header.index("total-glitched")]
            if rd_int < glitched_count:
                chunck_glitched = glitched[chunk_slice]
                
                (nz0,) = np.nonzero(chunck_glitched) # indices of non-zero pts
                rd_index = nz0[rd_int]
                return(chunk_slice, rd_index)
            else:
                rd_int -= glitched_count
        # If we are here, raise 
        raise RuntimeError("glitch_count does not match glitched")

    def param_matching(self, dparams):
        """
        If not matching shall trigger recomputing
        dparams is the stored computation
        """
        print("**CALLING param_matching +++", self.params)
        # TODO : note: when comparing iref should be disregarded ? 
        # or subclass specific implementation
        UNTRACKED = ["SA_params", "datetime", "debug"]
        SPECIAL_CASE = ["prec", "glitch_max_attempt"] # TODO increased precision should be accepted
        for key, val in self.params.items():
            if (key in UNTRACKED):
                continue
            elif (key in SPECIAL_CASE):
                if key == "prec":
                    if dparams[key] < val:
                        print("Higher precision requested",
                              dparams[key], "-->",  val)
                        return False
                elif key == "glitch_max_attempt":
                    if dparams[key] < val:
                        print("Higher glitch max attempt requested",
                              dparams[key], "-->",  val)
                        return False
            else: 
                if dparams[key] != val:
                    print("Unmatching", key, val, "-->", dparams[key])
                    return False
            print("its a match", key, val, dparams[key] )
        print("** all good")
        return True


    def init_cycling_arrays(self, chunk_slice, SA_params):
        
#        subset = self.subset
#        codes = self.codes
        iref = self.iref
        calc_name = self.calc_name
#        SA_params = self.SA_params

        # Creating c and Xrc arrays
        
#        chunk_mask = self.chunk_mask(chunk_slice)
        c = np.ravel(self.diff_c_chunk(chunk_slice, iref, calc_name))
        if self.subset is not None:
            chunk_mask = self.chunk_mask[chunk_slice]
            c = c[chunk_mask]
        Xrc_needed = (SA_params is not None) and not(self.Xrange_complex_type)
        if Xrc_needed:
            Xrc = np.ravel(self.diff_c_chunk(chunk_slice, iref, calc_name,
                                             ensure_Xr=True))
            if self.subset is not None:
                Xrc = Xrc[chunk_mask]
        
        # Instanciate arrays
        (n_pts,) = c.shape
        n_Z, n_U, n_stop = (len(code) for code in self.codes)
        if self.Xrange_complex_type:
            Z = fsx.Xrange_array.zeros([n_Z, n_pts],
                                       dtype=self.base_complex_type)
        else:
            Z = np.zeros([n_Z, n_pts], dtype=self.complex_type)
        U = np.zeros([n_U, n_pts], dtype=self.int_type)
        stop_reason = -np.ones([1, n_pts], dtype=self.termination_type)
        stop_iter = np.zeros([1, n_pts], dtype=self.int_type)
        
        
        # Now, which are the indices active ?...  
        index_active = np.arange(c.size, dtype=self.int_type)
        if self.iref == 0:
            # First loop, all indices are active
            bool_active = np.ones(c.size, dtype=np.bool)
        else:
            # We are in a glitch correction loop, only glitched index are 
            # active. Or rather "escaped_or_glitched" (including glitches due
            # to reference point prematurate exit).
            glitched_chunk = np.ravel(self.all_glitched[chunk_slice])
            if self.subset is not None:
                glitched_chunk = glitched_chunk[chunk_mask]
            bool_active = glitched_chunk
            # We also need to keep previous value for pixels which are not
            # glitched 

            # Still needed even with use of memmap : fancy indexing only used 
            # when we push back the data
            params, codes = self.reload_params(calc_name)
            (k_chunk_mask, k_Z, k_U, k_stop_reason, k_stop_iter
                )= self.reload_data(chunk_slice, calc_name)
            keep = ~glitched_chunk

            Z[:, keep] = k_Z[:, keep]
            U[:, keep] = k_U[:, keep]
            stop_reason[:, keep] =  k_stop_reason[:, keep]
            stop_iter[:, keep] = k_stop_iter[:, keep]
            
            
        # We now initialize the active part
        c_act = c[bool_active]
        if Xrc_needed:
            Xrc_act = Xrc[bool_active]
        Z_act = Z[:, bool_active].copy()
        U_act = U[:, bool_active].copy()
        self.initialize()(Z_act, U_act, c_act, chunk_slice, iref)
        U[:, bool_active] = U_act
        
        
        if SA_params is None:
            n_iter = SA_iter = 0
#            c[bool_active] = c_act * self.dx
#            Z_act = c_act * self.dx
        else:
            n_iter = SA_iter = SA_params["n_iter"]
            SA_shift = (SA_params["iref"] != self.iref)
            if SA_shift:
                raise RuntimeError("SA should be shifted 'before' cycling:" + 
                                   "use taylor_shift")
            # n_iter = SA_params["n_iter"]
            kc = SA_params["kc"]
            P = SA_params["P"]
            for i_z in range(n_Z):
                if self.Xrange_complex_type:
                    Z_act[i_z, :] = (P[i_z])(c_act / kc)
                else:
                    Z_act[i_z, :] = ((P[i_z])(Xrc_act / kc)).to_standard()
        Z[:, bool_active] = Z_act
#        c[bool_active] = c_act * self.dx

        # Initialise the path and ref point
        FP_params, ref_path = self.reload_ref_point(iref, calc_name)
        ref_div_iter = FP_params.get("div_iter", FP_params["max_iter"] + 1) #2**63 - 1) # max int64
        if (self.Xrange_Z_path) and not(self.Xrange_complex_type):
            ref_path = ref_path.to_standard()
        
        print("###### datatype", c.dtype, Z.dtype)

        return (c, Z, U, stop_reason, stop_iter, n_stop, bool_active,
             index_active, n_iter, SA_iter, ref_div_iter, ref_path)



    def ensure_ref_point(self, FP_loop, max_iter, calc_name,
                         iref=0, c0=None, newton="cv", order=None,
                         randomize=False):
        """
        # Check if we have a reference point stored for iref, 
          - otherwise computes and stores it in a file
        
        newton: ["cv", "step", None]
        """
        
        # Early escape if file exists
        if self.ref_point_count(calc_name) > iref:
            FP_params, Z_path = self.reload_ref_point(iref, calc_name)
            pt = FP_params["ref_point"]
            print("reloading ref point", iref, pt, "center", self.x + 1j * self.y)
            return FP_params, Z_path

        # Early escape if zoom level is low
        if self.dx > fssettings.newton_zoom_level:
            c0 = self.critical_pt
        

#        if self.ref_point_count(calc_name) <= iref:
        if c0 is None:
            c0 = self.x + 1j * self.y

        # skip Newton if settings impose it
        if fssettings.no_newton:
            newton = None

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
                order = self.ball_method(c0, self.dx * k_ball, max_iter)
                if order is None: # ball method escaped... we try to recover
                    if randomize < 5:
                        randomize += 1
                        print("BALL METHOD RANDOM ", randomize)
                        self.ensure_ref_point(FP_loop, max_iter, calc_name,
                                 iref=0, c0=None, newton=newton, order=None,
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

        print("compute ref_point", iref, pt, "\ncenter:\n",
              self.x + 1j * self.y)
        FP_params, Z_path = self.compute_ref_point(
                FP_loop, pt, max_iter, iref, calc_name, order)

        return FP_params,  Z_path

    def compute_ref_point(self, FP_loop, c, max_iter,
                          iref, calc_name, order=None):
        """
        Computes full precision, and stores path in normal precision
        Note:
        - Extended range considered only if self.Xrange_complex_type
        """
        FP_params = {"ref_point": c,
                     "order": order,
                     "max_iter": max_iter,
                     "FP_codes": self.FP_codes}
#        if "div_iter" in  FP_params.keys():
#            del FP_params["div_iter"]
        FP_codes = FP_params["FP_codes"]
        FP_array = self.FP_init()()# copy.copy(FP_params["init_array"])


        if not(self.Xrange_Z_path):
            Z_path = np.empty(
                [max_iter + 1, len(FP_codes)],
                dtype=self.base_complex_type)
            Z_path[0, :] = np.array(FP_array) # mpc to float
        else:
            xr_dtype = fsx.get_xr_dtype(self.base_complex_type)
            Z_path = np.empty([max_iter + 1, len(FP_codes)], dtype=xr_dtype
                ).view(fsx.Xrange_array)
            for i, _ in enumerate(FP_codes):
                Z_path[0, i] = fsx.mpc_to_Xrange(FP_array[i])

        # Now looping and storing ...
#        FP_params.pop('div_iter', None) # delete key div_iter / not needed !
        print("Computing full precision path starting at: \n", c)
        for n_iter in range(1, max_iter + 1):
            if n_iter % 5000 == 0:
                print("Full precision iteration: ", n_iter)
            div_iter = FP_loop(FP_array, c, n_iter)
            if div_iter is not None:
                FP_params["div_iter"] = div_iter
                print("##### Full precision loop diverging at iter", div_iter)
                break

            if not(self.Xrange_Z_path): 
                Z_path[n_iter, :] = np.array(FP_array)
            else:
                for i, _ in enumerate(FP_codes):
                    Z_path[n_iter, i] = fsx.mpc_to_Xrange(FP_array[i])


        self.save_ref_point(FP_params, Z_path, iref, calc_name)
        return FP_params, Z_path


    def ball_method(self, c, px, maxiter, order=1, M_divergence=1.e5):
        """
        Use a ball centered on c = x + i y to find the first period (up to 
        maxiter) of nucleus
        """
        print("ball method", c, px)
        if order == 1:
            return self._ball_method1(c, px, maxiter, M_divergence)
        elif order == 2:
            return self._ball_method2(c, px, maxiter, M_divergence)
