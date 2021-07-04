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

#force_recompute_SA = True

class PerturbationFractal(fs.Fractal):

    def __init__(self, directory):
        """
        Sames args and kwargs as Fractal except :
            - expect a "precision" integer : number of significant digits
            - x and y input as strings
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
        mpmath.mp.dps = precision
        # In case the user inputs were strings, we override with mpmath scalars
        self.x = mpmath.mpf(x)
        self.y = mpmath.mpf(y)
        self.dx = mpmath.mpf(dx)
        # Lazzy dictionary of reference point pathes
        self._ref_array = {}

    def diff_c_chunk(self, chunk_slice, iref, file_prefix,
                     ensure_Xr=False):
        """
        Returns a chunk of c_vec for the calculation
        Parameters
         - chunk_span
         - data_type: expected one of np.float64, np.longdouble
        
        Returns: 
        c_vec : [chunk_size x chunk_size] 1d-vec of type datatype
        
                as a "delta" wrt reference pt iref
        """
        offset_x, offset_y = self.offset_chunk(chunk_slice, ensure_Xr)
        FP_params = self.reload_ref_point(iref, file_prefix, scan_only=True)
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

    def postproc_chunck(self, postproc_keys, chunk_slice, file_prefix):
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
                                                         file_prefix)
        chunk_mask, Z, U, stop_reason, stop_iter = raw_data
        complex_dic, int_dic, termination_dic = self.codes_mapping(*codes)
        
        FP_codes, ref_array = self.get_ref_array(file_prefix)#[:, :, :]
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


    def ref_point_count(self, file_prefix):
        iref = 0
        while os.path.exists(self.ref_point_file(iref, file_prefix)):
            iref += 1
        return iref

    def ref_point_file(self, iref, file_prefix):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        data chunk
        """
        return os.path.join(self.directory, "data",
                file_prefix + "_pt{0:d}.ref".format(iref))

    def ref_point_scaling(self, iref, file_prefix):
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
                iref, file_prefix,scan_only=True)["ref_point"]
        print("ref point for SA", ref)
        # Let take some margin
        kc = max(abs(ref - corner_a), abs(ref - corner_b),
                 abs(ref - corner_c), abs(ref - corner_d)) * 2.0
        return kc

    def save_ref_point(self, FP_params, Z_path, iref, file_prefix):
        """
        Write to a dat file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.ref_point_file(iref, file_prefix)
        fsutils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            print("Path computed, saving", save_path)
            pickle.dump(FP_params, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(Z_path, tmpfile, pickle.HIGHEST_PROTOCOL)

    def reload_ref_point(self, iref, file_prefix, scan_only=False):
        """
        Reload arrays from a data file
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.ref_point_file(iref, file_prefix)
        with open(save_path, 'rb') as tmpfile:
            FP_params = pickle.load(tmpfile)
            if scan_only:
                return FP_params
            Z_path = pickle.load(tmpfile)
        return FP_params, Z_path

    def get_ref_array(self, file_prefix):
        """
        Lazzy evaluation of array compiling all ref points 'pathes'
        """
        nref = self.ref_point_count(file_prefix)
        FP_params = self.reload_ref_point(0, file_prefix, scan_only=True)
        max_iter = FP_params["max_iter"]
        FP_codes = FP_params["FP_codes"]
        print("FP_codes",  FP_codes)
        
        try:
            ret = self._ref_array[file_prefix]
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
            FP_params, Z_path = self.reload_ref_point(iref, file_prefix)
            ref_array[iref, : , :] = Z_path[:, :]
        self._ref_array[file_prefix] = ref_array
        return FP_codes, ref_array

    def SA_file(self, iref, file_prefix):
        """
        Returns the file path to store or retrieve params associated with a
        series approximation
        """
        return os.path.join(self.directory, "data", file_prefix +
                            "_pt{0:d}.sa".format(iref) )

    def save_SA(self, SA_params, iref, file_prefix):
        """
        Write to a dat file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.SA_file(iref, file_prefix)
        fsutils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            print("SA computed, saving", save_path)
            pickle.dump(SA_params, tmpfile, pickle.HIGHEST_PROTOCOL)

    def reload_SA(self, iref, file_prefix):
        """
        """
        save_path = self.SA_file(iref, file_prefix)
        with open(save_path, 'rb') as tmpfile:
            SA_params = pickle.load(tmpfile)
        return SA_params

    def run(self):#, FP_loop, FP_params, SA_init, SA_loop, SA_params, max_iter,
               # initialize, iterate, subset, codes,
               # file_prefix, pc_threshold, iref=0, glitch_stop_index=None,
               # glitch_sort_key=None, glitch_max_attempt=0):
        """
        glitch_stop_index : All points with a stop_reason >= glitch_stop_index
                are considered glitched (early ref exit glitch, or dynamic
                glitch). Subset of pixel for which 
                stop_reason == glitch_stop_index will be sorted according to 
                minimum value of field `glitch_sort_key` the minimal  point
                used as new reference.
                
        glitch_sort_key : the complex (Z array) field where is stored our
            'priority value' to select the next reference pixel.
        """
        file_prefix = self.file_prefix
        max_iter = self.max_iter
        SA_params = copy.deepcopy(self.SA_params)

        self.iref = 0

        FP_params, ref_path = self.ensure_ref_point(self.FP_loop(),
                self.max_iter, file_prefix, iref=self.iref, c0=None)
        FP_params0 = self.reload_ref_point(0, file_prefix, scan_only=True)

        # If SA is activated :
        # Reload SA coefficient, otherwise compute them.
        cutdeg_glitch = None
        if SA_params is not None:
            use_Taylor_shift = SA_params.pop("use_Taylor_shift", True)
            cutdeg_glitch = SA_params.pop("cutdeg_glitch", None)
            try:
                SA_params = self.reload_SA(self.iref, file_prefix)
            except FileNotFoundError:
                SA_params = self.series_approx(self.SA_init(), self.SA_loop(),
                    SA_params, self.iref, file_prefix)
                self.save_SA(SA_params, self.iref, file_prefix)

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

        dyn_glitched = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                postproc_keys=('stop_reason',
                lambda x: x == self.glitch_stop_index), mode="r+raw")
        # We store this one as an attribute, as this is the pixels which will
        # need an update "escaped_or_glitched"
        self.all_glitched = fs.Fractal_Data_array(self,
                file_prefix=file_prefix, postproc_keys=('stop_reason',
                    lambda x: x >= glitch_stop_index), mode="r+raw")
        glitch_sorts = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=(glitch_sort_key, None), mode="r+raw")
        stop_iters = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=("stop_iter", None), mode="r+raw")
        

#        _gcc = 0
#        _gcc_max = self.glitch_max_attempt
        
        # This looping is for dynamic glitches, we loop the largest glitches
        # first, a glitch being defined as 'same stop iteration'

        all_glitch_count = self.all_glitched.nansum()
        dyn_glitch_count = dyn_glitched.nansum()
        print("ANY glitched ? ", dyn_glitch_count, all_glitch_count)

        while ((self.iref < self.glitch_max_attempt)
               and (all_glitch_count > 0)
            ):#glitched.nanmax():
#            _gcc += 1

            self.iref += 1
            print("Launching glitch correction cycle, iref = ", self.iref)

            # We need to define c0
#            glitched = fs.Fractal_Data_array(self, file_prefix=file_prefix,
#                    postproc_keys=('stop_reason',
#                    lambda x: x == glitch_stop_index), mode="r+raw")


            print("with glitch pixel total count",
                  dyn_glitch_count, glitch_sort_key)
            print("with glitch and escaped pixel combined count",
                  all_glitch_count)

            if dyn_glitch_count > 0:
                glitched = dyn_glitched
                flag = 'dyn'
            else:
                glitched = self.all_glitched
                flag = 'escape'
                print("ESCAPE GLITCH !!!")
                

            # Minimize sort criteria over the selected glitch
            if flag == 'dyn':
                glitch_iter = self.largest_glitch_iter(stop_iters, glitched)
                (min_glitch_chunk, min_glitch_arg
                ) = self.min_glitch_pt(
                         glitch_sorts, stop_iters, glitched, glitch_iter)
            elif flag == 'escape':
                 print("ESCAPE mode, count", all_glitch_count)
                 (min_glitch_chunk, min_glitch_arg
                 ) = self.random_glitch_pt(all_glitch_count,
                                            self.all_glitched)

            print("candidate",flag, min_glitch_chunk, min_glitch_arg)
            # Now lets define ci. offset from image center is known
            offset_x, offset_y = self.offset_chunk(min_glitch_chunk)
#            offset_x = np.ravel(offset_x)
#            offset_y = np.ravel(offset_y)
#            if self.subset is not None:
#                chunk_mask = np.ravel(self.subset[min_glitch_chunk])
#                offset_x = offset_x[chunk_mask]
#                offset_y = offset_y[chunk_mask]

            if self.Xrange_complex_type:
                print("With shift from center coords:\n",
                      fsx.Xrange_to_mpfc(offset_x[min_glitch_arg]) / self.dx,
                      fsx.Xrange_to_mpfc(offset_y[min_glitch_arg]) / self.dy)
            else:
                print("With shift from center coords:\n",
                      offset_x[min_glitch_arg] / self.dx,
                      offset_y[min_glitch_arg] / self.dy)

            if self.Xrange_complex_type:
                ci = (self.x + fsx.Xrange_to_mpfc(offset_x[min_glitch_arg]) + 1j
                      * (self.y + fsx.Xrange_to_mpfc(offset_y[min_glitch_arg])))
            else:
                ci = (self.x + offset_x[min_glitch_arg] + 1j
                      * (self.y + offset_y[min_glitch_arg]))
            
            if flag == 'dyn':
                # Overall slightly counter-productive to add a Newton step here
                # (less robust), we keep the raw selected point
                FP_params, Z_path = self.ensure_ref_point(self.FP_loop(), max_iter,
                    file_prefix, iref=self.iref, c0=ci, newton="None", order=glitch_iter)
            elif flag == 'escape':
                # just keep the raw point
                FP_params, Z_path = self.ensure_ref_point(self.FP_loop(), max_iter,
                    file_prefix, iref=self.iref, c0=ci, newton="None", order=-1)

            if (SA_params is not None):
                print("SA_params cutdeg / iref:",
                      SA_params["cutdeg"], SA_params["iref"])
                try:
                    SA_params = self.reload_SA(self.iref, file_prefix)
                except FileNotFoundError:
                    if use_Taylor_shift:
                        # We will shift the SA params coefficient from the first
                        # reference point
                        print("Shifting SA from pt0 to new reference point")
                        SA_params0 = self.reload_SA(0, file_prefix)
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
                        self.save_SA(SA_params, self.iref, file_prefix)
                    else:
                        # Shift of SA approx is vetoed (no 'Taylor shift')
                        # -> Fall back to full SA recompute
                        if cutdeg_glitch is not None:
                            SA_params["cutdeg"] = cutdeg_glitch
                        SA_params = self.series_approx(self.SA_init(),
                            self.SA_loop(), SA_params, self.iref, file_prefix)
                        self.save_SA(SA_params, self.iref, file_prefix)



            self.cycles(chunk_slice=None, SA_params=SA_params)
            
#            glitched = fs.Fractal_Data_array(self, file_prefix=file_prefix,
#                    postproc_keys=('stop_reason',
#                    lambda x: x == glitch_stop_index), mode="r+raw")

            # Recomputing the exit condition
            all_glitch_count = self.all_glitched.nansum()
            dyn_glitch_count = dyn_glitched.nansum()
            print("ANY glitched ? ", all_glitch_count)


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

        glitch_size = np.max(glitch_bincount)
        glitch_iter = np.argmax(glitch_bincount)

        debug = False
        if debug:
            print("check sum", np.sum(glitch_bincount))
            indices = np.arange(max_iter)
            print('non null glitch iter',
                  indices[glitch_bincount != 0])
            print('non null glitch counts',
                  glitch_bincount[glitch_bincount != 0])
        print("Largest glitch (pts: {}, iter: {})".format(
                glitch_size, glitch_iter))

        return glitch_iter

    def min_glitch_pt(self, glitch_sorts, stop_iters, glitched, glitch_iter):
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
        for chunk_slice in self.chunk_slices():
            glitch_sort = (glitch_sorts[chunk_slice]).real
#            print("debug, glitch_sort", glitch_sort.shape, glitch_sorts[chunk_slice].shape)

            if self.Xrange_complex_type:
                glitch_sort = glitch_sort.view(fsx.Xrange_array
                                               ).to_standard()

#            chunk_min = np.nanmin(glitch_sort) # debug
            # Non-glitched pixel disregarded
            chunck_glitched = glitched[chunk_slice]
            glitch_sort[~chunck_glitched] = np.inf

            # Glitched with non-target iter disregarded
            stop_iter = stop_iters[chunk_slice]
            is_glitch_iter = (stop_iter == glitch_iter)
            glitch_sort[~is_glitch_iter] = np.inf
#            print("debug2, glitch_sort", glitch_sort.shape)

            chunk_min = np.nanmin(glitch_sort)
            if chunk_min < min_glitch:
#                print("debug3, glitch_sort", glitch_sort.shape)
                chunk_argmin = np.nanargmin(glitch_sort)
#                print("debug3, chunk_argmin", chunk_argmin, np.unravel_index(chunk_argmin, glitch_sort.shape))
                min_glitch_chunk = chunk_slice
                min_glitch_arg = np.unravel_index(chunk_argmin,
                                                  glitch_sort.shape)

        print("Minimal criteria reached at", min_glitch_chunk, min_glitch_arg)
        return min_glitch_chunk, min_glitch_arg

    def random_glitch_pt(self, glitch_count, glitched):
        """
        Return localisation of a random pixel in a glitch
        glitch_count: int, the number of glitched pixels
        glitched: *Fractal_Data_array* wrapping the dyn glitched pixel bool
        """
        rd_int = random.randrange(0, glitch_count)
        for chunk_slice in self.chunk_slices():
            chunck_glitched = glitched[chunk_slice]
            glitched_count = np.sum(chunck_glitched)
#            print("counting", rd_int, glitched_count)
            if rd_int < glitched_count:
                nz0, nz1 = np.nonzero(chunck_glitched)
                rd_index = (nz0[rd_int], nz1[rd_int])
#                print("found index", chunck_glitched.shape, rd_index)
                return(chunk_slice, rd_index)
            else:
                rd_int -= glitched_count
        # If we are hre, raise 
        raise RuntimeError("glitch_count does not match glitched")


#    def random_glitch_pt(self, glitch_count, glitched):
#        """
#        Return localisation of a random pixel in a glitch
#        glitch_count: int, the number of glitched pixels
#        glitched: *Fractal_Data_array* wrapping the dyn glitched pixel bool
#        """
#        rd_int = random.randrange(0, glitch_count)
#        for chunk_slice in self.chunk_slices():
#            chunck_glitched = glitched[chunk_slice]
#            glitched_count = np.sum(chunck_glitched)
#            print("counting", rd_int, glitched_count)
#            if rd_int < glitched_count:
#                rd_index = np.nonzero(chunck_glitched)[rd_int]
#                print("foud index", glitched.shape, rd_index)
#                return(chunk_slice, rd_index)
#            else:
#                rd_int -= glitched_count

        

    def param_matching(self, dparams):
        """
        If we want to do some clever sanity test (not implemented)
        If not matching shall raise a ValueError
        """
        print("**CALLING param_matching +++", self.params)
        # TODO : note: when comparing iref should be disregarded ? 
        # or subclass specific implementation
        UNTRACKED = ["SA_params", "datetime", "debug"]
        for key, val in self.params.items():
            if not(key in UNTRACKED) and dparams[key] != val:
                print("Unmatching", key, val, "-->", dparams[key])
                return False
            print("its a match", key, val, dparams[key] )
        print("** all good")
        return True



    def res_available(self, chunk_slice=None):
        """  Returns True if chunkslice is already computed with current
        parameters
        (Otherwise False)
        """
        print("**CALLING res_available +++")
        if chunk_slice is None:
            try:
                (dparams, dcodes) = self.reload_data_param(self.file_prefix)
            except IOError:
                return False
            return self.param_matching(dparams)
        else:
            try:
                (dparams, dcodes) = self.reload_data_chunk(chunk_slice,
                    self.file_prefix, scan_only=True)
            except IOError:
                return False

        if dparams["iref"] >= self.iref:
            return self.param_matching(dparams)
            # return self.dic_matching(dparams, self.calc_params)
        
        # We are in the case where a file exists but not updated to the irefs
        # do we need to actually do something ?
        glitched_chunk = np.ravel(self.all_glitched[chunk_slice])
        if np.any(glitched_chunk):
            print("Glitch correction triggered, computing")
            return False
        else:
            print("No glitched pixels remaining: ")
            return self.param_matching(dparams)
            # return self.dic_matching(dparams, self.calc_params)
        
#        subset = self.subset
#        iref = self.iref
#        codes = self.codes
#        file_prefix = self.file_prefix
#                # First tries to reload the data :
#                
#        if subset is not None:
#            chunk_mask = np.ravel(subset[chunk_slice])
#        else:
#            chunk_mask = None
#                
#        try:
#            (dparams, dcodes) = self.reload_data_chunk(chunk_slice,
#                                                   file_prefix, scan_only=True)
#            self.dic_matching(dparams, self.params)
#            self.dic_matching(dcodes, codes)
#
#            if (iref is None) or iref == 0:
#                print("Data found, skipping calc: ", chunk_slice)
#                return
#            # Now if this is a glitch correction iteration... we should check
#            # if any pixel in this chunk is glitched with iref_pix < iref
#            else:
#                glitch_loop = True
#                glitched_chunk = np.ravel(glitched[chunk_slice])
#                print("glitched count", np.count_nonzero(glitched[chunk_slice]))
#                # glitched_chunk[...] = True #debug
#                irefs_chunk = np.ravel(irefs[chunk_slice])
#                if subset is not None:
#                    glitched_chunk = glitched_chunk[chunk_mask]
#                    irefs_chunk = irefs_chunk[chunk_mask]
#                if (np.any(glitched_chunk) and
#                    (np.min(irefs_chunk[glitched_chunk]) < iref)):
#                    print("Glitch correction triggered, computing: ", iref)
#                else:
#                    print("Data found in glitch correction loop, "
#                          "skipping calc: ", chunk_slice)        
#                    return
#        except IOError:
#            glitch_loop = False # File not found, nothing to 'unglitch'
#            print("Unable to find data_file, computing")


    def init_cycling_arrays(self, chunk_slice, SA_params):
        
#        subset = self.subset
#        codes = self.codes
        iref = self.iref
        file_prefix = self.file_prefix
#        SA_params = self.SA_params

        # Creating c and Xrc arrays
        chunk_mask = self.chunk_mask(chunk_slice)
        c = np.ravel(self.diff_c_chunk(chunk_slice, iref, file_prefix))
        if self.subset is not None:
            c = c[self.chunk_mask(chunk_slice)]
        Xrc_needed = (SA_params is not None) and not(self.Xrange_complex_type)
        if Xrc_needed:
            Xrc = np.ravel(self.diff_c_chunk(chunk_slice, iref, file_prefix,
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
            (dparams, dcodes, raw_data) = self.reload_data_chunk(chunk_slice,
                                       file_prefix, scan_only=False)
            k_chunk_mask, k_Z, k_U, k_stop_reason, k_stop_iter = raw_data
            keep = ~glitched_chunk
#            if k_chunk_mask is not None: # Safeguard
#                np.testing.assert_equal(k_chunk_mask, chunk_mask)
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
        FP_params, ref_path = self.reload_ref_point(iref, file_prefix)
        ref_div_iter = FP_params.get("div_iter", 2**63 - 1) # max int64

        return (c, Z, U, stop_reason, stop_iter, n_stop, bool_active,
             index_active, n_iter, SA_iter, ref_div_iter, ref_path)


    def save_data_chunk(self, save_path, params, codes, raw_data):
        """
        Write to a dat file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        params = copy.deepcopy(params)
        params["iref"] = self.iref
        return super().save_data_chunk(save_path, params, codes, raw_data)


    def ensure_ref_point(self, FP_loop, max_iter, file_prefix,
                         iref=0, c0=None, newton="cv", order=None,
                         randomize=False):
        """
        # Check if we have a reference point stored for iref, 
          - otherwise computes and stores it in a file
        
        newton: ["cv", "step", None]
        """
        
        # Early escape if file exists
        if self.ref_point_count(file_prefix) > iref:
            FP_params, Z_path = self.reload_ref_point(iref, file_prefix)
            pt = FP_params["ref_point"]
            print("reloading ref point", iref, pt, "center", self.x + 1j * self.y)
            return FP_params, Z_path

#        if self.ref_point_count(file_prefix) <= iref:
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

        # If we plan a newton iteration, we lauche the process
        # ball method to find the order, than Newton
        if (newton is not None) and (newton != "None"):
            if order is None:
                k_ball = 0.1
                order = self.ball_method(
                    c0, max(self.dx, self.dy) * k_ball, max_iter)
                if order is None: # ball method escaped... we try to recover
                    if randomize < 5:
                        randomize += 1
                        print("BALL METHOD RANDOM ", randomize)
                        self.ensure_ref_point(FP_loop, max_iter, file_prefix,
                                 iref=0, c0=None, newton="cv", order=None,
                                 randomize=randomize)
                    else:
                        raise ValueError("Ball method failed")

            max_newton = 1 if (newton == "step") else None
            print("newton ", newton, " with order: ", order)
            print("max newton iter ", max_newton)

            newton_cv, nucleus = self.find_nucleus(
                    c0, order, max_newton=max_newton)

            if not(newton_cv) and (newton != "step"):
                attempt = 0
                while not(newton_cv) and attempt < 2:
                    attempt += 1
                    old_dps = mpmath.mp.dps
                    mpmath.mp.dps = int(1.25 * old_dps)
                    print("Newton not cv, dps boost to: ", mpmath.mp.dps)
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
                raise ValueError("Newton failed with order", order)

            pt = nucleus

        print("compute ref_point", iref, pt, "\ncenter:\n",
              self.x + 1j * self.y)
        FP_params, Z_path = self.compute_ref_point(
                FP_loop, pt, max_iter, iref, file_prefix, order)

        # TODO : known failed
#            div_ref_pt = (FP_params.get("div_iter", None) is not None)
#            if div_ref_pt and (newton != "step"):
#                print("div_ref_pt", randomize)
#                if not(randomize):
#                    randomize = 1.
#                else:
#                    randomize *= 2.
#                print("div_ref_pt with randomize", randomize)
#                div_file = self.ref_point_file(iref, file_prefix)
#                os.unlink(div_file)
#                self.ensure_ref_point(FP_loop, max_iter, file_prefix,
#                         iref=0, c0=None, newton="cv", order=None,
#                         randomize=randomize)
                


        return FP_params,  Z_path

    def compute_ref_point(self, FP_loop, c, max_iter,
                          iref, file_prefix, order=None):
        """
        Computes full precision, and stores path in normal precision
        Note: Extended range not considered as orbit remind bounded.
        """
        FP_params = {"ref_point": c,
                     "order": order,
                     "max_iter": max_iter,
                     "FP_codes": self.FP_codes}
#        if "div_iter" in  FP_params.keys():
#            del FP_params["div_iter"]
        FP_codes = FP_params["FP_codes"]
        FP_array = self.FP_init()()# copy.copy(FP_params["init_array"])

        Z_path = np.empty([max_iter + 1, len(FP_codes)],
                          dtype=self.base_complex_type)
#        Z_path = np.zeros([max_iter + 1, len(FP_codes)],
#                          dtype=self.base_complex_type)
        Z_path[0, :] = np.array(FP_array)

        # Now looping and storing ...
#        FP_params.pop('div_iter', None) # delete key div_iter / not needed !
        print("Computing full precision path starting at: \n", c)
        for n_iter in range(1, max_iter + 1):
            if n_iter % 5000 == 0:
                print("Full precision iteration: ", n_iter)
            div_iter = FP_loop(FP_array, c, n_iter)
            if div_iter is not None:
                FP_params["div_iter"] = div_iter
                print("Full precision loop diverging at iter", div_iter)
                break
            Z_path[n_iter, :] = np.array(FP_array)
        self.save_ref_point(FP_params, Z_path, iref, file_prefix)
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
