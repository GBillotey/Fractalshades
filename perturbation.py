# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from fractal import (Fractal, mkdir_p, Fractal_plotter, Fractal_colormap,
                     Fractal_Data_array)

import pickle
import copy
import mpmath

from numpy_utils.xrange import (Xrange_array, Xrange_polynomial, Xrange_SA,
                                Xrange_to_mpfc, mpc_to_Xrange, mpf_to_Xrange)
import os
        

class PerturbationFractal(Fractal):
    #    https://mrob.com/pub/muency/reversebifurcation.html
#       http://www.fractalforums.com/announcements-and-news/pertubation-theory-glitches-improvement/
  
    # -1.74928893611435556407228 + 0 i 
    # 
    def __init__(self, directory, x, y, dx, nx, xy_ratio, theta_deg, **kwargs):
        """
        Sames args and kwargs as Fractal except :
            - expect a "precision" integer : number of significant digits
            - x and y input as strings
        """
        mpmath.mp.dps = kwargs.pop("precision")
        x = mpmath.mpf(x) # x
        y = mpmath.mpf(y) # y
        dx = mpmath.mpf(dx)
        self.ref_array = {}
        super().__init__(directory, x, y, dx, nx, xy_ratio, theta_deg,
             **kwargs)

            

    def diff_c_chunk(self, chunk_slice, iref, file_prefix,
                     ensure_Xr=False):#, data_type=None):
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
            drift_rp = mpc_to_Xrange(drift_rp, dtype=self.base_complex_type)
#            if self.Xrange_complex_type:
            diff = Xrange_array._build_complex(offset_x, offset_y)
#            else:
#                diff = Xrange_array._build_complex(
#                        Xrange_array(offset_x), Xrange_array(offset_y))
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

    
#    def init_ref_points(self, file_prefix):
#        self.directory
#        prefixes = []
#        pattern = "*_*.path"
#        data_dir = os.path.join(self.directory, "data")
#        if not os.path.isdir(data_dir):
#            return prefixes
#        with os.scandir(data_dir) as it:
#            for entry in it:
#                if (fnmatch.fnmatch(entry.name, pattern) and
#                    os.path.relpath(os.path.dirname(entry.path), data_dir) == "."):
#                    frags = entry.name.split("_")
#                    candidate = ""
#                    for f in frags[:-2]:
#                        candidate = candidate + f + "_"
#                    candidate = candidate[:-1]
#                    if candidate not in prefixes:
#                        prefixes += [candidate]
#        return prefixes

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
        corner_a = c0 + self.dx + 1.j * self.dy
        corner_b = c0 - self.dx + 1.j * self.dy
        corner_c = c0 - self.dx - 1.j * self.dy
        corner_d = c0 + self.dx - 1.j * self.dy
        ref = self.reload_ref_point(
                iref, file_prefix,scan_only=True)["ref_point"]
        print("ref point for SA", ref)
        kc = max(abs(ref - corner_a), abs(ref - corner_b),
                 abs(ref - corner_c), abs(ref - corner_d)) # * 1.5
        return kc

    def save_ref_point(self, FP_params, Z_path, iref, file_prefix):
        """
        Write to a dat file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.ref_point_file(iref, file_prefix)
        mkdir_p(os.path.dirname(save_path))
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

        # Never computed or need update it in this case we compute

        if not(self.Xrange_complex_type):
            ref_array = np.empty([nref, max_iter + 1, len(FP_codes)],
                                  dtype=self.complex_type)
        else: 
            ref_array = Xrange_array.empty(
                    [nref, max_iter + 1, len(FP_codes)],
                    dtype=self.base_complex_type)

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
        mkdir_p(os.path.dirname(save_path))
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


class Perturbation_mandelbrot(PerturbationFractal):
    
#    @classmethod
#    gui_default_data_dic = super().gui_default_data_dic
    
    
#    directory, x, y, dx, nx, xy_ratio, theta_deg, projection,
#    complex_type, M_divergence, epsilon_stationnary, pc_threshold
    
    def explore(self, x, y, dx, max_iter, M_divergence, epsilon_stationnary,
                pc_threshold):
        """
        Produces "explore.png" image for GUI navigation
        """
        self.x = x
        self.y = y
        self.dx = dx
        self.xy_ratio = 1.#xy_ratio
        self.theta_deg = 0.# theta_deg

        # clean-up the *.tmp files to force recalculation
        self.clean_up("explore")

        self.explore_loop(
            file_prefix="explore",
            subset=None,
            max_iter=max_iter,
            M_divergence =M_divergence,
            epsilon_stationnary=epsilon_stationnary,
            pc_threshold=pc_threshold)

        plotter = Fractal_plotter(
            fractal=self,
            base_data_key=("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3}),
            base_data_prefix="explore",
            base_data_function=lambda x: x,
            colormap=-Fractal_colormap((0.1, 1.0, 200), plt.get_cmap("magma")),
            probes_val=[0., 1.],
            probes_kind="qt",
            mask=None)

        plotter.plot("explore")
    
#    @property
#    def tag_dict(self):
#        return {"Software": "py-fractal",
#                "fractal": type(self).__name__,
#                "projection": self.fractal.projection,
#                "x": repr(self.x),
#                "y": repr(self.y),
#                "dx": repr(self.dx),
#                "nx": repr(self.dx),
#                "ny": repr(self.dx),
#                "theta_deg": repr(self.theta_deg)
#                "max_iter": repr(self.theta_deg)
#                "M_divergence": repr(self.theta_deg)
#                "epsilon_stationnary": repr(self.fractal.theta_deg)
#                }
    
    def cycles(self, FP_loop, FP_params, SA_init, SA_loop, SA_params, max_iter,
               initialize, iterate, subset, codes,
               file_prefix, pc_threshold, iref=0, glitch_stop_index=None,
               glitch_sort_key=None):
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
        cutdeg_glitch = SA_params.pop("cutdeg_glitch", None)

        _, ref_path = self.ensure_ref_point(FP_loop, FP_params, max_iter,
                file_prefix, iref=iref, c0=None)

        # If SA is activated :
        # Reload SA coefficient, otherwise compute them.
        if SA_params is not None:
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
        glitched = Fractal_Data_array(self, file_prefix=file_prefix,
                postproc_keys=('stop_reason',
                lambda x: x == glitch_stop_index), mode="r+raw")

        _gcc = 0
        _gcc_max = 15
        
        # This looping is for dynamic glitches, we loop the largest glitches
        # first, a glitch being defined as 'same stop iteration'
        while (_gcc < _gcc_max) and glitched.nanmax(): # At least one chunk is glitched
            _gcc += 1
            irefs = Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=('iref', None), mode="r+raw")
            iref += 1
            print("Launching glitch correction cycle, iref = ", iref)

            # We need to define c0 !!!
# In mandelbrot-perturbator[1] I use a tree structure for recursively solving
# glitches.  At the iteration P when a glitch occurs, I find the pixel with the
# minimum |Z+deltaZ| value, whose C is near the minibrot of period P whose
# influence is causing the glitch.  At the minibrot's nucleus the |Z+deltaZ|
# would be 0, because of the periodicity, so using the derivative calculated
# anyway for DE, you can do one step of Newton's method very cheaply (no need
# to do any more iterations) to get a better new reference (said minibrot of
# period P).  Then rebase all the pixels that glitched (at the same iteration
# number with the same parent reference) to the new reference, no need to
# restart from the beginning, because of periodicity:  the new deltaZ is just
# Z+deltaZ, the new deltaC is a translation by the difference of the old and
# new reference (be sure to use the crrect sign).
            glitched = Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=('stop_reason',
                    lambda x: x == glitch_stop_index), mode="r+raw")
            escaped_or_glitched = Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=('stop_reason',
                    lambda x: x >= glitch_stop_index), mode="r+raw")
            glitch_sorts = Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=(glitch_sort_key, None), mode="r+raw")
            stop_iters = Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=("stop_iter", None), mode="r+raw")

            print("with glitch pixel total count", glitched.nansum(), glitch_sort_key)
            print("with glitch and escaped pixel combined count", escaped_or_glitched.nansum())
#            glitch_iter_min = glitch_iters.nanmin()
#            print("glitch_iter_min: ", glitch_iter_min)
            
            # Need to find the minimum ie highest 'priority'
#            glitch_min_iter = 2**31 - 1
            glitch_bincount = np.zeros([max_iter], dtype=np.int32)
            for chunk_slice in self.chunk_slices():
                stop_iter = stop_iters[chunk_slice]
                chunck_glitched = glitched[chunk_slice]
                glitch_bincount += np.bincount(stop_iter[chunck_glitched],
                                               minlength=max_iter)
#                chunk_glitch_min_iter = np.min(stop_iter[chunck_glitched])
#                glitch_min_iter = min(chunk_glitch_min_iter, glitch_min_iter)
            glitch_size = np.max(glitch_bincount)
            glitch_iter = np.argmax(glitch_bincount)
            print("check sum", np.sum(glitch_bincount))
            
            indices = np.arange(max_iter)
            print('non null glitch iter', indices[glitch_bincount != 0])
            print('non null glitch counts', glitch_bincount[glitch_bincount != 0])
            
            del glitch_bincount
            print("Largest glitch (pts: {}, iter: {})".format(
                    glitch_size, glitch_iter))

            # Minimize sort criteria over the selected glitch
            min_glitch = np.inf
            min_glitch_chunk = None
            min_glitch_arg = None
            for chunk_slice in self.chunk_slices():
                glitch_sort = (glitch_sorts[chunk_slice]).real
#                print("glitch_sort", type(glitch_sort), glitch_sort.dtype, glitch_sort.shape)
                if self.Xrange_complex_type:
                    glitch_sort = glitch_sort.view(Xrange_array).to_standard()

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
            offset_x, offset_y = self.offset_chunk(min_glitch_chunk) # wrt center... # however, 
            offset_x = np.ravel(offset_x)
            offset_y = np.ravel(offset_y)
            if subset is not None:
                chunk_mask = np.ravel(subset[chunk_slice])
                offset_x = offset_x[chunk_mask]
                offset_y = offset_y[chunk_mask]
            
            if self.Xrange_complex_type:
                print("With shift from center coords:\n",
                      Xrange_to_mpfc(offset_x[min_glitch_arg]) / self.dx,
                      Xrange_to_mpfc(offset_y[min_glitch_arg]) / self.dy)
            else:
                print("With shift from center coords:\n",
                      offset_x[min_glitch_arg] / self.dx,
                      offset_y[min_glitch_arg] / self.dy)
            

            if self.Xrange_complex_type:
                ci = (self.x + Xrange_to_mpfc(offset_x[min_glitch_arg]) + 1.j
                      * (self.y + Xrange_to_mpfc(offset_y[min_glitch_arg])))
            else:
                ci = (self.x + offset_x[min_glitch_arg] + 1.j
                      * (self.y + offset_y[min_glitch_arg]))
            # TODO: To refine, we *could* make use of 1 step of newton
            # only if ref point comes from a 'dynamic glitch'
            # F^N(ci) known (Zn), should target 0.
            # d(F^N(ci)) = dZndC known also.
            # ci_newton = ci - (Zn / dZndC)
            FP_params, Z_path = self.ensure_ref_point(FP_loop, FP_params, max_iter,
                file_prefix, iref=iref, c0=ci, newton="step", order=glitch_iter)

            # Debug
            #SA_params = None
            print("SA_params cutdeg / iref:", SA_params["cutdeg"], SA_params["iref"])
            if cutdeg_glitch is not None:
                SA_params["cutdeg"] = cutdeg_glitch

            recompute_SA = True
            if (SA_params is not None) and recompute_SA:
                try:
                    SA_params = self.reload_SA(iref, file_prefix)
                except FileNotFoundError:
                    SA_params = self.series_approx(SA_init, SA_loop, SA_params, iref, file_prefix)
                    self.save_SA(SA_params, iref, file_prefix)
            # Debug
#            else:
#                SA_params = None
                

            super().cycles(initialize, iterate, subset, codes,
                file_prefix, chunk_slice=None, pc_threshold=pc_threshold,
                iref=iref, ref_path=Z_path, SA_params=SA_params,
                glitched=escaped_or_glitched, irefs=irefs)




#    def glitch_correction_cycle_ref_point(self, glitched, irefs):
#        """
#        Return the max reference pt with at least one glitched pixel 
#        Note: if no more glitched pixel, return the 
#        
#        # at each loop the glitch correction index is raised for each chunk
#        # So if there remain one glitched (in the wavefront) means the loop with index 'N+1' is not finished...
#        # shall return  chunk_slice, iref
#        """
#        min_ref = 0
#        for chunk_slice in self.chunk_slices():
#            chunk_glitched = glitched[chunk_slice]
#            chunk_irefs = irefs[chunk_slice]
#            if np.any(chunk_glitched):
#                min_ref = np.min()
#
#    def glitch_correction_cycles(self, initialize, iterate, terminate, subset, codes,
#            file_prefix, chunk_slice=None, pc_threshold=0.2):
#        pass
#    # https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/cluster/_optics.py#L24


    def ensure_ref_point(self, FP_loop, FP_params, max_iter, file_prefix,
                         iref=0, c0=None, newton="cv", order=None):
        """
        # Check if we have at least one reference point stored, otherwise 
        # computes and stores it
        
        newton: "cv" or "step" or None
        """
        if self.ref_point_count(file_prefix) <= iref:
            if c0 is None:
                c0 = self.x + 1.j * self.y
            pt = c0
            print("Proposed ref point:\n", c0)
                
            if newton is not None:# and (iref == 0):
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
                    print("*** Relauch with shifted ref point, ", diff)
                    return self.ensure_ref_point(FP_loop, FP_params, max_iter, file_prefix,
                                          iref, c0=c_shifted)
        
                pt = nucleus

            print("compute ref_point", iref, pt, "\ncenter:\n", self.x + 1j * self.y)#, pt - (self.x + 1j * self.y))
            FP_params, Z_path = self.compute_ref_point(
                    FP_loop, FP_params, pt, max_iter, iref, file_prefix)
        else:
            FP_params, Z_path = self.reload_ref_point(iref, file_prefix)
            pt = FP_params["ref_point"]
            print("reloading ref point", iref, pt, "center", self.x + 1j * self.y)
        #print("Zpath", Z_path)
        return FP_params,  Z_path

    def compute_ref_point(self, FP_loop, FP_params, c, max_iter,
                          iref, file_prefix):
        """
        Computes full precision, and stores path in normal precision
        (optionaly with extended range)
        """
        FP_params["ref_point"] = c
        FP_params["max_iter"] = max_iter
        FP_params["Xrange_complex_type"] = self.Xrange_complex_type

        FP_codes = FP_params["FP_codes"]
        FP_array = copy.copy(FP_params["init_array"])

        if not(self.Xrange_complex_type):
            Z_path = np.empty([max_iter + 1, len(FP_codes)],
                           dtype=self.complex_type)
            Z_path[0, :] = np.array(FP_array, dtype=self.complex_type)
        else:
            Z_path = Xrange_array.empty([max_iter + 1, len(FP_codes)],
                                        dtype=self.base_complex_type)
            m, exp = zip(*[mpmath.frexp(item) for item in FP_array])
            Z_path[0, :] = Xrange_array(
                    np.array(m, dtype=self.base_complex_type), exp)

        # Now looping and storing ...
        for n_iter in range(1, max_iter + 1):
            if n_iter%1000 == 0:
                print("Full precision iteration: ", n_iter)
            
            div_iter = FP_loop(FP_array, c, n_iter)
            if div_iter is not None:
                FP_params["div_iter"] = div_iter
                print("### Full precision loop diverging at iter", div_iter)
                break

                
            if not(self.Xrange_complex_type):
                Z_path[n_iter, :] = np.array(FP_array, dtype=self.complex_type)
                # print("Z_path", Z_path[n_iter, :])
            else:
                # warning, this time we do have complex...
                m_re, exp_re = zip(*[mpmath.frexp(item.real) for item in FP_array])
                m_im, exp_im = zip(*[mpmath.frexp(item.imag) for item in FP_array])

                Z_path[n_iter, :] = Xrange_array._build_complex(
                        Xrange_array(np.array(m_re, dtype=self.base_float_type),
                                     np.array(exp_re, dtype=np.int32)),
                        Xrange_array(np.array(m_im, dtype=self.base_float_type),
                                     np.array(exp_im, dtype=np.int32)))
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
        
    @staticmethod
    def _ball_method1(c, px, maxiter, M_divergence):#, M_divergence):
        #c = x + 1j * y
        z = 0.
        r0 = px      # first radius
        r = r0 * 1.
        az = abs(z)
        dzdc = 0.

        for i in range(1, maxiter + 1):
            r = (az  + r)**2 - az**2 + r0
            z = z**2 + c
            dzdc =  2. * z * dzdc + 1.
            az = abs(z)
            if az > M_divergence:
                return None
            if (r > az):
                print("Ball method 1 found period:", i)
                return i #, z, dzdc


    @staticmethod
    def find_nucleus(c, order, max_newton=None, eps_cv=None):
        """
        https://en.wikibooks.org/wiki/Fractals/Mathematics/Newton_method#center
        https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html
        https://mathr.co.uk/blog/2018-11-17_newtons_method_for_periodic_points.html
        https://code.mathr.co.uk/mandelbrot-numerics/blob/HEAD:/c/lib/m_d_nucleus.c
        Run Newton search to find z0 so that f^n(z0) == 0
        """
        if order is None:
            return False, c
        if max_newton is None:
            max_newton = mpmath.mp.prec
        if eps_cv is None:
            eps_cv = mpmath.mpf(2.)**(-mpmath.mp.prec)
        c_loop = c

        for i_newton in range(max_newton): 
            print("newton", i_newton)
            zr = mpmath.mp.zero
            dzrdc = mpmath.mp.zero
            h = mpmath.mp.one
            dh = mpmath.mp.zero
            for i in range(1, order + 1):
                dzrdc = mpmath.mpf("2.") * dzrdc * zr + 1.
                zr = zr * zr + c_loop
                # divide by unwanted periods
                if i < order and order % i == 0:
                    h *= zr
                    dh += dzrdc / zr
            f = zr / h
            df = (dzrdc * h - zr * dh) / (h * h)
            cc = c_loop - f / df
            newton_cv = abs(cc - c_loop ) <= eps_cv
            c_loop = cc
            if newton_cv:
                print("cv@", i_newton)#, zr)#, dzrdc)
                break
        return newton_cv, c_loop

#    @staticmethod
#    def find_any_attracting(c, order, max_newton=None, eps_cv=None):
#        """
#        https://en.wikibooks.org/wiki/Fractals/Mathematics/Newton_method#center
#        https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html
#        https://mathr.co.uk/blog/2018-11-17_newtons_method_for_periodic_points.html
#        https://code.mathr.co.uk/mandelbrot-numerics/blob/HEAD:/c/lib/m_d_nucleus.c
#        Run Newton search to find z0 so that f^n(z0) == 0
#        """
#        if max_newton is None:
#            max_newton = mpmath.mp.prec
#        if eps_cv is None:
#            eps_cv = mpmath.mpf(2.)**(-mpmath.mp.prec)
#            print("eps_cv", eps_cv)
#        c_loop = c
#
#        for i_newton in range(max_newton):     
#            if (i_newton % 5 ==0):
#                print("i_newton", i_newton, " / ", max_newton)
#            zr = c_loop #mpmath.mp.zero
#            dzrdc = mpmath.mp.one# mpmath.mp.zero
#            # d2zrdzdc = 0.   # == 1. hence : constant wrt c...
#            for i in range(1, order + 1):
#                # d2zrdzdc = 2 * (d2zrdzdc * zr + dzrdz * dzrdc)
#                dzrdc = mpmath.mpf("2.") * dzrdc * zr + mpmath.mp.one
#                zr = zr * zr + c_loop
##                    zr = zr / 
##                if abs(zr) < mpmath.mpf("0.0001"):
##                    print("zr", zr, i)
#            cc = c_loop - (zr - c_loop) / (dzrdc - mpmath.mp.one)
#
#            newton_cv = abs(cc - c_loop ) <= eps_cv
#            c_loop = cc
#            if newton_cv:
#                break
#            
#        return newton_cv, c_loop

    def series_approx(self, SA_init, SA_loop, SA_params, iref, file_prefix):
        """
        Zk, zk
        C, c
        
        SA_params keys:
            init_P
            kc
            iref
            n_iter
            P
        """
        P = SA_init(SA_params["cutdeg"])
        SA_params["iref"] = iref
        
        # Ensure corner points strictly included in convergence disk :
        kc = self.ref_point_scaling(iref, file_prefix)

        # Convert kc into a Xrange array (in all cases the SA iterations use
        # extended range)
        kc = mpf_to_Xrange(kc, dtype=self.base_float_type)
        SA_params["kc"] = kc
        print('SA_params["kc"]', kc)
        kcX = np.insert(kc, 0, 0.)

        SA_valid = True
        _, ref_path = self.reload_ref_point(iref, file_prefix)
        n_iter = 0
        while SA_valid:
            n_iter +=1
            # keep a copy
            
            P_old0 = P[0].coeffs.copy()
            P_old1 = P[1].coeffs.copy()
            SA_loop(P, n_iter, ref_path[n_iter - 1, :], kcX)
            coeffs_sum = np.sum(P[0].coeffs.abs2())
            SA_valid = ((P[0].err.abs2()  <= 1.e-6 * coeffs_sum) # 1e-6
                        & (coeffs_sum <= 1.e6))
            if not(SA_valid):
                P[0].coeffs = P_old0
                P[1].coeffs = P_old1
                n_iter -=1
                print("SA stop", n_iter, P[0].err, np.sqrt(coeffs_sum))
                deriv_scale = mpc_to_Xrange(self.dx) / SA_params["kc"]
                if not(self.Xrange_complex_type):
                    deriv_scale = deriv_scale.to_standard()
                # We derive the polynomial wrt to c. However the constant coeff
                # Shall be set to null as we already know it with FP...
                # (or to the contrary should we use it to evaluate the FP iter
                # ????)
                P_deriv = P[0].deriv() * deriv_scale # (1. / SA_params["kc"])
                # P_deriv.coeffs[0] = 0.
                P_deriv2 = P_deriv.deriv() * deriv_scale # (1. / SA_params["kc"])
                # P_deriv2.coeffs[0] = 0.
                P += [P_deriv, P_deriv2]
            if n_iter%500 == 0:
                print("SA running", n_iter, "err: ", P[0].err, "<<", np.sqrt(np.sum(P[0].coeffs.abs2())))
        SA_params["n_iter"] = n_iter
        SA_params["P"] = P

        return SA_params


    def full_loop(self, file_prefix, subset, max_iter, M_divergence,
                   epsilon_stationnary, pc_threshold=1.0, 
                   SA_params=None, glitch_eps=1.e-3, interior_detect=False):
        """
        Computes the full data and derivatives
        - "zn"
        - "dzndz" (only if *interior_detect* is True)
        - "dzndc"
        - "d2zndc2"
        
        Note: if *interior_detect* is false we still allocate the arrays for
        *dzndz* but we do nnot iterate.
        """ 
        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2"]
        int_codes = ["iref"]
        stop_codes = ["max_iter", "divergence", "stationnary",
                      "dyn_glitch", "divref_glitch"]
        codes = (complex_codes, int_codes, stop_codes)
        
        FP_fields = [0, 1] # , 2, 3]
        FP_params = {"FP_codes": [complex_codes[f] for f in FP_fields],
                     "init_array": [mpmath.mp.zero, mpmath.mp.zero]}

#        if SA_params is not None:
##            cutdeg = SA_params["cutdeg"]
#            #SA_params["iref"] = 0 #iref
#            # Initialize with SA(zn), SA(dzndc) (dzndz & dzndz2 deduced 
#            # by polynomial derivation in method `series_approx`)
#            def init_P(cutdeg):
#                return [Xrange_SA([0.], cutdeg=cutdeg),
#                        Xrange_polynomial([0.], cutdeg=cutdeg)]
#            SA_params["init_P"] = init_P

        def FP_loop(FP_array, c0, n_iter):
            """ Full precision loop
            derivatives corrected by lenght kc
            """
            # skipping these iters as already computed at SA stage
#            FP_array[3] = 2. * (FP_array[3] * FP_array[0] + FP_array[2]**2)
#            FP_array[2] = 2. * FP_array[2] * FP_array[0] + 1.
            if interior_detect:
                FP_array[1] = 2. * FP_array[1] * FP_array[0]
                if n_iter == 1:
                    FP_array[1] = 1.
            FP_array[0] = FP_array[0]**2 + c0
            # If FP iteration is divergent do not raise an error,
            # return the flag n_iter (to allow subsequent glitch correction)
            if (((FP_array[0]).conjugate() * FP_array[0]).real
                > M_divergence**2):
                print("Reference point FP iterations escaping at", n_iter)
                return n_iter

        def SA_init(cutdeg):
            return [Xrange_SA([0.], cutdeg=cutdeg),
                    Xrange_polynomial([0.], cutdeg=cutdeg)]#,
#                    Xrange_polynomial([0.], cutdeg=cutdeg),
#                    Xrange_polynomial([0.], cutdeg=cutdeg)]

        def SA_loop(P, n_iter, ref_path, kcX):
            """ Series Approximation loop
            Note that derivatives w.r.t dc will be computed directly from the
            S.A polynomial. To the contrary, derivatives w.r.t dz need to be
            evaluated
            """
#            print("SA loop", n_iter, ref_path, kcX)
#            print("P[0] in", P[0])
            
#            P[3] = 2. * (ref_path[3] * P[0] + 2. * ref_path[2] * P[2] +
#                         ref_path[0] * P[3] + P[0] * P[3] + P[2] * P[2])
#            P[2] = 2. * (ref_path[2] * P[0] +
#                        ref_path[0] * P[2] + P[0] * P[2])
            
            if interior_detect:
                P[1] = 2. * (P[0] * ref_path[1]
                             + P[1] * ref_path[0] + P[0] * P[1])
#                if n_iter <= 1:
#                    P[1] = 1.

            P[0] = P[0] * (P[0] + 2. * ref_path[0]) + kcX
#            print("P[0] out", P[0])

        def initialize(Z, U, c, chunk_slice, iref):
            Z[3, :] = 0.
            Z[2, :] = 0.
            Z[1, :] = 0.
            Z[0, :] = 0.
            U[0, :] = iref

        def iterate(Z, U, c, stop_reason, n_iter, SA_iter,
                    ref_div_iter, ref_path, ref_path_next):
            """
            dz(n+1)dz   <- 2. * dzndz * zn
            z(n+1)      <- zn**2 + c 
            
            Termination codes
            0 -> max_iter reached ('interior')
            1 -> M_divergence reached by np.abs(zn)
            2 -> dzn stationnary ('interior early detection')
            3 -> glitched (Dynamic glitches...)
            4 -> glitched (Ref point diverging  ...)
            """
#            print("iterate", n_iter)
#            print("ref_path", ref_path[:])
            
            Z[3, :] = 2. * (# ref_path[3] * Z[0, :] +
                            # 2. * ref_path[2] * Z[2, :] +
                            ref_path[0] * Z[3, :] +
                            Z[0, :] * Z[3, :] + Z[2, :]**2)
            Z[2, :] = 2. * (#ref_path[2] * Z[0, :] +
                            ref_path[0] * Z[2, :] + Z[0, :] * Z[2, :])

            if interior_detect:
                Z[1, :] = 2. * (ref_path[1] * Z[0, :] +
                                ref_path[0] * Z[1, :] + Z[0, :] * Z[1, :])

            Z[0, :] = Z[0, :] * (Z[0, :] + 2. * ref_path[0]) + c
            
#            print("z3", Z[3, :])
#            print("z2", Z[2, :])
#            print("z0", Z[0, :])
#            print("c", c)

            # Tests for cycling termination
            #print("iter", n_iter, max_iter, ref_div_iter)
            if n_iter >= max_iter:
                stop_reason[0, :] = 0
                return

            # ref_div_iter dernier coeff calculé. ref_path_next not 
            # initialized, but we exit here 
            if n_iter >= ref_div_iter:
                # flagged as 'ref point diverging' glitch
                stop_reason[0, :] = 4 
                # We generate a random glitch_sort_key (save as d2zndc2 field),
                # However < 0 to get priority over any "dynamic glitches".
#                rg = np.random.default_rng(1)
#                Z[3, :] = -rg.random([Z.shape[1]], dtype=self.base_float_type)
                return
            
#            if n_iter <= (SA_iter + 2):
#                Z[1, :] = 1. - ref_path[1]


            if self.Xrange_complex_type:
                full_sq_norm = (Z[0, :] + ref_path_next[0]).abs2()
            else:
                full_sq_norm = ((Z[0, :].real + ref_path_next[0].real)**2 + 
                                (Z[0, :].imag + ref_path_next[0].imag)**2)

            ref_sq_norm = ref_path_next[0].real**2 + ref_path_next[0].imag**2


            bool_infty = (full_sq_norm > M_divergence**2)
            stop_reason[0, bool_infty] = 1
            
            # Interior points detection
            if interior_detect and (n_iter > 1):
                bool_stationnary = ((Z[1, :].real + ref_path[1].real)**2 + 
                        (Z[1, :].imag + ref_path[1].imag)**2 <
                         epsilon_stationnary**2)
                stop_reason[0, bool_stationnary] = 2
                
            # Glitched points detection "dynamic glitches"
            # We generate a glitch_sort_key based on 'close to secondary
            # nucleus' criteria (use d2zndc2 field to save it).
            bool_glitched = (full_sq_norm  < (ref_sq_norm * glitch_eps**2))
            # debug
            if np.any(bool_glitched):
                print("some are glitched at iter", n_iter)
                print("init:\n", Z[3, bool_glitched])
                print("new:\n", full_sq_norm[bool_glitched])
                
            stop_reason[0, bool_glitched] = 3
            Z[3, bool_glitched] = full_sq_norm[bool_glitched]

        # Lauching cycles, with glitch correction 'ON'
        self.cycles(FP_loop, FP_params, SA_init, SA_loop, SA_params, max_iter,
               initialize, iterate, subset, codes,
               file_prefix, pc_threshold, glitch_stop_index=3,
               glitch_sort_key="d2zndc2")



    # NOT UPDATED !!!
#    def explore_loop(self, file_prefix, subset, max_iter, M_divergence,
#                   epsilon_stationnary, pc_threshold=1.0, iref=0, 
#                   SA_params=None, glitch_eps=1.e-3):
#        """
#        Only computes the necessary to draw level set of real iteration number
#        + interior point detection for speed-up
#        """ 
#        raise NotImplementedError()
#        complex_codes = ["zn"]
#        int_codes = ["iref"]
#        stop_codes = ["max_iter", "divergence", "stationnary", "glitched"]
#        codes = (complex_codes, int_codes, stop_codes)
#        
#        FP_params = {"FP_codes": complex_codes,
#                     "init_array": [mpmath.mp.zero]}
#
#        if SA_params is not None:
#            cutdeg = SA_params["cutdeg"]
#            SA_params["iref"] = iref
#            SA_params["init_P"] = [
#                    Xrange_SA([0.], cutdeg=cutdeg)]#,
##                    Xrange_polynomial([0.], cutdeg=cutdeg),
##                    Xrange_polynomial([0.], cutdeg=cutdeg)]
#
#        def FP_loop(FP_array, c0, n_iter):
#            """ Full precision loop """
##            FP_array[3] = 2. * (FP_array[3] * FP_array[0] + FP_array[2]**2)
##            FP_array[2] = 2. * FP_array[2] * FP_array[0] + 1.
##            FP_array[1] = 2. * FP_array[1] * FP_array[0]
#            FP_array[0] = FP_array[0]**2 + c0
#            if abs(FP_array[0]) > M_divergence:
#                print("Reference point FP iterations escaping at", n_iter)
#                return n_iter
#
#        def SA_loop(P, n_iter, ref_path, kcX):
#            """ Series Approximation loop """
##            P[3] = 2. * (ref_path[3] * P[0] + 2. * ref_path[2] * P[2] +
##                         ref_path[0] * P[3] + P[0] * P[3] + P[2] * P[2])
##            P[2] = 2. * (ref_path[2] * P[0] +
##                        ref_path[0] * P[2] + P[0] * P[2])
##            P[1] = 2. * (P[0] * ref_path[1] + P[1] * ref_path[0] + P[0] * P[1])
#            #print("kc", kc, type(kc))
#            P[0] = P[0] * (P[0] + 2. * ref_path[0]) + kcX#[0., kc]
##
##            if n_iter == 1:
#                
#
#        def initialize(Z, U, c, chunk_slice, iref):
##            Z[3, :] = 0.
##            Z[2, :] = 0.
##            Z[1, :] = 0.
#            Z[0, :] = 0.
#            U[0, :] = iref
#
#        def iterate(Z, U, c, n_iter, ref_path):
#            """
#            dz(n+1)dz   <- 2. * dzndz * zn
#            z(n+1)      <- zn**2 + c 
#            """
##            print("#interate")
##            print("ref_path", type(ref_path), ref_path.dtype)
##            print("Z", type(Z), Z.dtype)
##            two = self.two
##            Z[3, :] = 2. * (ref_path[3] * Z[0, :] +
##                            2. * ref_path[2] * Z[2, :] +
##                            ref_path[0] * Z[3, :] +
##                            Z[0, :] * Z[3, :] + Z[2, :]**2)
##            Z[2, :] = 2. * (ref_path[2] * Z[0, :] +
##                            ref_path[0] * Z[2, :] + Z[0, :] * Z[2, :])
##            Z[1, :] = 2. * (ref_path[1] * Z[0, :] +
##                            ref_path[0] * Z[1, :] + Z[0, :] * Z[1, :])
#            Z[0, :] = Z[0, :] * (Z[0, :] + 2. * ref_path[0]) + c
#
##            if n_iter == 1:
###                if np.max(Z[1, :]) > epsilon_stationnary:
###                    raise ValueError("mandatory to start at a critical point")
###                else:
###                    print("Initialisation at critical point OK")
##                Z[1, :] = 1.
#
#        def terminate(stop_reason, Z, U, c, n_iter, ref_path):
#            """
#            Tests for a cycle termination
#            Shall modify in place stop_reason, if != -1 loop will stop for this
#            pixel.
#             0 -> max_iter reached
#             1 -> M_divergence reached by np.abs(zn)
#             2 -> dzn stationnary
#            """
#            if n_iter >= max_iter:
##                print("max iter")
#                stop_reason[0, :] = 0
#                
##            print("Z[0, :].real", Z[0, :].real)
##            print("M_divergence", M_divergence)
#            if self.Xrange_complex_type:
#                full_sq_norm = (Z[0, :] + ref_path[0]).abs2()
#            else:
#                full_sq_norm = ((Z[0, :].real + ref_path[0].real)**2 + 
#                                (Z[0, :].imag + ref_path[0].imag)**2)
#            ref_sq_norm = ref_path[0].real**2 + ref_path[0].imag**2
#
#            bool_infty = (full_sq_norm > M_divergence**2)
#            #bool_infty = bool_infty | ~np.isfinite(Z[0, :])            
#            stop_reason[0, bool_infty] = 1
#            
#
##            bool_stationnary = ((Z[1, :].real + ref_path[1].real)**2 + 
##                    (Z[1, :].imag + ref_path[1].imag)**2 <
##                     epsilon_stationnary**2)
##            stop_reason[0, bool_stationnary] = 2
#
#            bool_glitched = full_sq_norm  < (ref_sq_norm * glitch_eps**2)
#            stop_reason[0, bool_glitched] = 3
#
#        # Now we can launch the loop over all pixels
#
#        self.cycles(FP_loop, FP_params, SA_loop, SA_params, max_iter,
#               initialize, iterate, terminate, subset, codes,
#               file_prefix, pc_threshold, iref, glitched=None)


#==============================================================================
#    def first_loop(self, file_prefix, subset, max_iter, M_divergence,
#                   epsilon_stationnary, pc_threshold=0.2):
#        """
#        Applies the "usual" cycles to subset with full result arrays
#        """
#        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2"]
#        int_codes = []
#        stop_codes = ["max_iter", "divergence", "stationnary"]
#        codes = (complex_codes, int_codes, stop_codes)
#
#        def initialize(Z, U, c, chunk_slice):
#            Z[0, :] = 0.
#            Z[1, :] = 0.
#            Z[2, :] = 0.
#            Z[3, :] = 0.
#
#        def iterate(Z, U, c, n_iter):
#            """
#            d2z(n+1)dc2 <- 2. * (d2zndc2 * zn + dzndc**2)
#            dz(n+1)dc   <- 2. * dzndc * zn + 1.
#            dz(n+1)dz   <- 2. * dzndz * zn
#            z(n+1)      <- zn**2 + c 
#            """
#            Z[3, :] = 2. * (Z[3, :] * Z[0, :] + Z[2, :]**2)
#            Z[2, :] = 2. * Z[2, :] * Z[0, :] + 1.
#            Z[1, :] = 2. * Z[1, :] * Z[0, :]
#            Z[0, :] = Z[0, :]**2 + c
#
#            if n_iter == 1:
#                if np.max(Z[1, :]) > epsilon_stationnary:
#                    raise ValueError("mandatory to start at a critical point")
#                else:
#                    print("Initialisation at critical point OK")
#                Z[1, :] = 1.
#
#        def terminate(stop_reason, Z, U, c, n_iter):
#            """
#            Tests for a cycle termination
#            Shall modify in place stop_reason, if != -1 loop will stop for this
#            pixel.
#             0 -> max_iter reached
#             1 -> M_divergence reached by np.abs(zn)
#             2 -> dzn stationnary
#            """
#            if n_iter > max_iter:
#                stop_reason[0, :] = 0
#                
##            print("Z[0, :]", Z[0, :])
##            print("Z[0, :].real", Z[0, :].real)
##            print("Z[0, :].real", (Z[0, :]).re())
#
#            bool_infty = (Z[0, :].real**2 + Z[0, :].imag**2 >
#                          M_divergence**2)
##            bool_infty = (np.abs(Z[0, :]) > M_divergence)
#            
#            bool_infty = bool_infty | ~np.isfinite(Z[0, :])
#            stop_reason[0, bool_infty] = 1
#
##            bool_stationnary = (np.abs(Z[1, :]) < epsilon_stationnary)
#            bool_stationnary = (Z[1, :].real**2 + Z[1, :].imag**2 <
#                                epsilon_stationnary**2)
#            stop_reason[0, bool_stationnary] = 2
#
#        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
#                    pc_threshold=pc_threshold)



if __name__ == "__main__":
    pass
    #Perturbation_mandelbrot.series_approx(None, 0, 64)