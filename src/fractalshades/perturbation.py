# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import copy
import mpmath

import fractalshades as fs
import fractalshades.numpy_utils.xrange as fsx

force_recompute_SA = False

class PerturbationFractal(fs.Fractal):

    def __init__(self, directory, x, y, dx, nx, xy_ratio, theta_deg, **kwargs):
        """
        Sames args and kwargs as Fractal except :
            - expect a "precision" integer : number of significant digits
            - x and y input as strings
        """
        mpmath.mp.dps = kwargs.pop("precision")
        x = mpmath.mpf(x)
        y = mpmath.mpf(y)
        dx = mpmath.mpf(dx)
        self.ref_array = {}
        super().__init__(directory, x, y, dx, nx, xy_ratio, theta_deg,
             **kwargs)

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
        drift_rp = (self.x + 1.j * self.y) - FP_params["ref_point"] #.imag
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
        c0 = self.x + 1.j * self.y
        corner_a = c0 + 0.5 * (self.dx + 1.j * self.dy)
        corner_b = c0 + 0.5 * (- self.dx + 1.j * self.dy)
        corner_c = c0 + 0.5 * (- self.dx - 1.j * self.dy)
        corner_d = c0 + 0.5 * (self.dx - 1.j * self.dy)
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
        fs.mkdir_p(os.path.dirname(save_path))
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
            ret = self.ref_array[file_prefix]
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
        self.ref_array[file_prefix] = ref_array
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
        fs.mkdir_p(os.path.dirname(save_path))
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

    def cycles(self, FP_loop, FP_params, SA_init, SA_loop, SA_params, max_iter,
               initialize, iterate, subset, codes,
               file_prefix, pc_threshold, iref=0, glitch_stop_index=None,
               glitch_sort_key=None, glitch_max_attempt=0):
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
        if iref != 0:
            raise ValueError()
            
        FP_params0 = self.reload_ref_point(0, file_prefix, scan_only=True)

        FP_params, ref_path = self.ensure_ref_point(FP_loop, FP_params,
                max_iter, file_prefix, iref=iref, c0=None)

        # If SA is activated :
        # Reload SA coefficient, otherwise compute them.
        cutdeg_glitch = None
        if SA_params is not None:
            cutdeg_glitch = SA_params.pop("cutdeg_glitch", None)
            try:
                SA_params = self.reload_SA(iref, file_prefix)
            except FileNotFoundError:
                SA_params = self.series_approx(SA_init, SA_loop, SA_params, iref, file_prefix)
                self.save_SA(SA_params, iref, file_prefix)

        # First "perturbation" cycle, no glitch correction
        super().cycles(initialize, iterate, subset, codes,
            file_prefix, chunk_slice=None, pc_threshold=pc_threshold,
            iref=iref, ref_path=ref_path, SA_params=SA_params)
        
        # Exit if glitch correction inactive
        if glitch_stop_index is None:
            return
        
        # Glitch correction loops
        # Lopping cycles until no more glitched areas remain
        # First inspecting already computed data, completing partial loop
        # (if any). Note that super.cycles will escape pixels irefs >= iref
        glitched = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                postproc_keys=('stop_reason',
                lambda x: x == glitch_stop_index), mode="r+raw")

        _gcc = 0
        _gcc_max = glitch_max_attempt
        
        # This looping is for dynamic glitches, we loop the largest glitches
        # first, a glitch being defined as 'same stop iteration'
        print("ANY glitched ? ", glitched.nanmax())
        while (_gcc < _gcc_max) and glitched.nanmax():
            _gcc += 1
            irefs = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=('iref', None), mode="r+raw")
            iref += 1
            print("Launching glitch correction cycle, iref = ", iref)

            # We need to define c0
#            glitched = fs.Fractal_Data_array(self, file_prefix=file_prefix,
#                    postproc_keys=('stop_reason',
#                    lambda x: x == glitch_stop_index), mode="r+raw")
            escaped_or_glitched = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=('stop_reason',
                    lambda x: x >= glitch_stop_index), mode="r+raw")
            glitch_sorts = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=(glitch_sort_key, None), mode="r+raw")
            stop_iters = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=("stop_iter", None), mode="r+raw")

            print("with glitch pixel total count",
                  glitched.nansum(), glitch_sort_key)
            print("with glitch and escaped pixel combined count",
                  escaped_or_glitched.nansum())

            # Need to find the minimum ie highest 'priority'
            glitch_bincount = np.zeros([max_iter], dtype=np.int32)
            for chunk_slice in self.chunk_slices():
                stop_iter = stop_iters[chunk_slice]
                chunck_glitched = glitched[chunk_slice]
                glitch_bincount += np.bincount(stop_iter[chunck_glitched],
                                               minlength=max_iter)

            glitch_size = np.max(glitch_bincount)
            glitch_iter = np.argmax(glitch_bincount)
            print("check sum", np.sum(glitch_bincount))
            
            indices = np.arange(max_iter)
            print('non null glitch iter', indices[glitch_bincount != 0])
            print('non null glitch counts', glitch_bincount[glitch_bincount != 0])
            first_glitch_iter = (indices[glitch_bincount != 0])[0]
            first_glitch_size = glitch_bincount[first_glitch_iter]
            
            del glitch_bincount
            print("Largest glitch (pts: {}, iter: {})".format(
                    glitch_size, glitch_iter))
            print("First glitch (pts: {}, iter: {})".format(
                    first_glitch_size, first_glitch_iter))

            # Minimize sort criteria over the selected glitch
            min_glitch = np.inf
            min_glitch_chunk = None
            min_glitch_arg = None
            for chunk_slice in self.chunk_slices():
                glitch_sort = (glitch_sorts[chunk_slice]).real
                if self.Xrange_complex_type:
                    glitch_sort = glitch_sort.view(fsx.Xrange_array
                                                   ).to_standard()

                # Non-glitched pixel disregarded
                chunck_glitched = glitched[chunk_slice]
                glitch_sort[~chunck_glitched] = np.inf

                # Glitched with non-target iter disregarded
                stop_iter = stop_iters[chunk_slice]
                is_glitch_iter = (stop_iter == glitch_iter)
                glitch_sort[~is_glitch_iter] = np.inf

                chunk_min = np.nanmin(glitch_sort)
                if chunk_min < min_glitch:
                    chunk_argmin = np.nanargmin(glitch_sort)
                    min_glitch_chunk = chunk_slice
                    min_glitch_arg = chunk_argmin
            print("Minimal criteria reached at", min_glitch_chunk, min_glitch_arg)

            # Now lets define ci. offset from image center is known
            offset_x, offset_y = self.offset_chunk(min_glitch_chunk)
            offset_x = np.ravel(offset_x)
            offset_y = np.ravel(offset_y)
            if subset is not None:
                chunk_mask = np.ravel(subset[chunk_slice])
                offset_x = offset_x[chunk_mask]
                offset_y = offset_y[chunk_mask]
            
            if self.Xrange_complex_type:
                print("With shift from center coords:\n",
                      fsx.Xrange_to_mpfc(offset_x[min_glitch_arg]) / self.dx,
                      fsx.Xrange_to_mpfc(offset_y[min_glitch_arg]) / self.dy)
            else:
                print("With shift from center coords:\n",
                      offset_x[min_glitch_arg] / self.dx,
                      offset_y[min_glitch_arg] / self.dy)

            if self.Xrange_complex_type:
                ci = (self.x + fsx.Xrange_to_mpfc(offset_x[min_glitch_arg]) + 1.j
                      * (self.y + fsx.Xrange_to_mpfc(offset_y[min_glitch_arg])))
            else:
                ci = (self.x + offset_x[min_glitch_arg] + 1.j
                      * (self.y + offset_y[min_glitch_arg]))
            # Overall slightly counter-productive to add a Newton step here
            # (less robust), we keep the raw selected point
            FP_params, Z_path = self.ensure_ref_point(FP_loop, FP_params, max_iter,
                file_prefix, iref=iref, c0=ci, newton="step", order=glitch_iter)

            if SA_params is not None:
                print("SA_params cutdeg / iref:",
                      SA_params["cutdeg"], SA_params["iref"])

            if cutdeg_glitch is not None:
                SA_params["cutdeg"] = cutdeg_glitch

            if (SA_params is not None):
                try:
                    SA_params = self.reload_SA(iref, file_prefix)
                except FileNotFoundError:
                    if force_recompute_SA:
                        SA_params = self.series_approx(SA_init, SA_loop,
                                SA_params, iref, file_prefix)
                        self.save_SA(SA_params, iref, file_prefix)
                    else:
                        # We will shift the SA params coefficient from the first
                        # reference point
                        print("Shifting SA wrt new reference point")
                        SA_params0 = self.reload_SA(0, file_prefix)
                        P0 = SA_params0["P"]
                        dc_ref = (fsx.mpc_to_Xrange(FP_params["ref_point"]
                            - FP_params0["ref_point"], self.base_complex_type)
                            / SA_params0["kc"])
#                        print("P0", P0)
#                        print("dc_ref", dc_ref)
                        P_shifted = []
                        for P0i in P0:
                            P_shifted += [P0i.taylor_shift(dc_ref)]
#                            print("P0i shifted", P0i.taylor_shift(dc_ref))
                        # For fields known at FP, we have to correct the 1st 
                        # coeff which  is by definition 0 (due to the FP ref
                        # point shift.
                        P_shifted[0].coeffs[0] = 0.
                        SA_params = {"cutdeg": SA_params0["cutdeg"],
                                     "iref": iref,
                                     "kc": SA_params0["kc"],
                                     "n_iter": SA_params0["n_iter"],
                                     "P": P_shifted}
                        self.save_SA(SA_params, iref, file_prefix)
#                        raise ValueError()

            super().cycles(initialize, iterate, subset, codes,
                file_prefix, chunk_slice=None, pc_threshold=pc_threshold,
                iref=iref, ref_path=Z_path, SA_params=SA_params,
                glitched=escaped_or_glitched, irefs=irefs)
            
            glitched = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=('stop_reason',
                    lambda x: x == glitch_stop_index), mode="r+raw")
            # Recomputing the exit condition
            print("ANY glitched ? ", glitched.nanmax())


    def ensure_ref_point(self, FP_loop, FP_params, max_iter, file_prefix,
                         iref=0, c0=None, newton="cv", order=None):
        """
        # Check if we have at least one reference point stored, otherwise 
        # computes and stores it
        
        newton: ["cv", "step", None]
        """
        if self.ref_point_count(file_prefix) <= iref:
            if c0 is None:
                c0 = self.x + 1.j * self.y
            pt = c0
            print("Proposed ref point:\n", c0)
                
            if (newton is not None) and (newton != "None"):
                if order is None:
                    k_ball = 0.5
                    order = self.ball_method(c0,
                            max(self.dx, self.dy) * k_ball, max_iter)
                    if order is None: # ball method escaped...
                        order = 1
                max_newton = 1 if (newton == "step") else None
                print("newton ", newton, " with order: ", order)
                print("max newton iter ", max_newton)

                newton_cv, nucleus = self.find_nucleus(
                        c0, order, max_newton=max_newton)

                if newton_cv or (newton == "step"):
                    shift = nucleus - (self.x + self.y * 1.j)
                    print("Reference nucleus at:\n", nucleus, order)
                    print("With shift % from image center:\n",
                          shift.real / self.dx, shift.imag / self.dy)
                    shift = nucleus - pt
                    print("With shift % from proposed coords:\n",
                          shift.real / self.dx, shift.imag / self.dy)
                else:
                    data_type = self.base_float_type
                    rg = np.random.default_rng(0)
                    diff = rg.random([2], dtype=data_type)
                    c_shifted = (c0 + self.dx * (diff[0] - 0.5) + 
                                      self.dy * (diff[1] - 0.5) * 1.j)
                    print("*** Newton failed,")
                    print("*** Relauch with shifted ref point, ", diff)
                    return self.ensure_ref_point(FP_loop, FP_params, max_iter, file_prefix,
                                          iref, c0=c_shifted)
                pt = nucleus

            print("compute ref_point", iref, pt, "\ncenter:\n",
                  self.x + 1j * self.y)
            FP_params, Z_path = self.compute_ref_point(
                    FP_loop, FP_params, pt, max_iter, iref, file_prefix, order)
        else:
            FP_params, Z_path = self.reload_ref_point(iref, file_prefix)
            pt = FP_params["ref_point"]
            print("reloading ref point", iref, pt, "center", self.x + 1j * self.y)

        return FP_params,  Z_path

    def compute_ref_point(self, FP_loop, FP_params, c, max_iter,
                          iref, file_prefix, order=None):
        """
        Computes full precision, and stores path in normal precision
        Note: Extended range not considered as orbit remind bounded.
        """
        FP_params["ref_point"] = c
        FP_params["order"] = order
        FP_params["max_iter"] = max_iter
        if "div_iter" in  FP_params.keys():
            del FP_params["div_iter"]
        FP_codes = FP_params["FP_codes"]
        FP_array = copy.copy(FP_params["init_array"])

        Z_path = np.empty([max_iter + 1, len(FP_codes)],
                          dtype=self.base_complex_type)
        Z_path[0, :] = np.array(FP_array)

        # Now looping and storing ...
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
