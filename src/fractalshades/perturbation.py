# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import copy
import mpmath

import fractalshades.core as fs
import fractalshades.numpy_utils.xrange as fsx

#from fsxrange import (Xrange_array, Xrange_polynomial, Xrange_SA,
#                      Xrange_to_mpfc, mpc_to_Xrange, mpf_to_Xrange)


class PerturbationFractal(fs.Fractal):
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
        _, ref_path = self.ensure_ref_point(FP_loop, FP_params, max_iter,
                file_prefix, iref=iref, c0=None)

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
        while (_gcc < _gcc_max) and glitched.nanmax():
            _gcc += 1
            irefs = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=('iref', None), mode="r+raw")
            iref += 1
            print("Launching glitch correction cycle, iref = ", iref)

            # We need to define c0
            glitched = fs.Fractal_Data_array(self, file_prefix=file_prefix,
                    postproc_keys=('stop_reason',
                    lambda x: x == glitch_stop_index), mode="r+raw")
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
            
            del glitch_bincount
            print("Largest glitch (pts: {}, iter: {})".format(
                    glitch_size, glitch_iter))

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
            offset_x, offset_y = self.offset_chunk(min_glitch_chunk) # wrt center... # however, 
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
            FP_params, Z_path = self.ensure_ref_point(FP_loop, FP_params, max_iter,
                file_prefix, iref=iref, c0=ci, newton="step", order=glitch_iter)

            if SA_params is not None:
                print("SA_params cutdeg / iref:",
                      SA_params["cutdeg"], SA_params["iref"])

            if cutdeg_glitch is not None:
                SA_params["cutdeg"] = cutdeg_glitch

            recompute_SA = True
            if (SA_params is not None) and recompute_SA:
                try:
                    SA_params = self.reload_SA(iref, file_prefix)
                except FileNotFoundError:
                    SA_params = self.series_approx(SA_init, SA_loop, SA_params, iref, file_prefix)
                    self.save_SA(SA_params, iref, file_prefix)

            super().cycles(initialize, iterate, subset, codes,
                file_prefix, chunk_slice=None, pc_threshold=pc_threshold,
                iref=iref, ref_path=Z_path, SA_params=SA_params,
                glitched=escaped_or_glitched, irefs=irefs)



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

        plotter = fs.Fractal_plotter(
            fractal=self,
            base_data_key=("potential", 
                     {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3}),
            base_data_prefix="explore",
            base_data_function=lambda x: x,
            colormap=-fs.Fractal_colormap((0.1, 1.0, 200), plt.get_cmap("magma")),
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
            print("Newton iteration", i_newton)
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
                print("Newton iteration cv @ ", i_newton)
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
        kc = fsx.mpf_to_Xrange(kc, dtype=self.base_float_type)
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
            # P_old1 = P[1].coeffs.copy()
            SA_loop(P, n_iter, ref_path[n_iter - 1, :], kcX)
            coeffs_sum = np.sum(P[0].coeffs.abs2())
            SA_valid = ((P[0].err.abs2()  <= 1.e-6 * coeffs_sum) # 1e-6
                        & (coeffs_sum <= 1.e6))
            if not(SA_valid):
                P[0].coeffs = P_old0
                # P[1].coeffs = P_old1
                n_iter -=1
                print("SA stop", n_iter, P[0].err, np.sqrt(coeffs_sum))
                deriv_scale = fsx.mpc_to_Xrange(self.dx) / SA_params["kc"]
                if not(self.Xrange_complex_type):
                    deriv_scale = deriv_scale.to_standard()
                P1 = fsx.Xrange_polynomial([1.], cutdeg=SA_params["cutdeg"])
                # We derive the polynomial wrt to c. However the constant coeff
                # Shall be set to null as we already know it with FP...
                # (or to the contrary should we use it to evaluate the FP iter
                # ????)
                P_deriv = P[0].deriv() * deriv_scale # (1. / SA_params["kc"])
                # P_deriv.coeffs[0] = 0.
                P_deriv2 = P_deriv.deriv() * deriv_scale # (1. / SA_params["kc"])
                # P_deriv2.coeffs[0] = 0.
                P += [P1, P_deriv, P_deriv2]
            if n_iter%500 == 0:
                print("SA running", n_iter, "err: ", P[0].err, "<<", np.sqrt(np.sum(P[0].coeffs.abs2())))
        SA_params["n_iter"] = n_iter
        SA_params["P"] = P

        return SA_params


    def full_loop(self, file_prefix, subset, max_iter, M_divergence,
                   epsilon_stationnary, pc_threshold=1.0, 
                   SA_params=None, glitch_eps=1.e-3, interior_detect=False,
                   glitch_max_attempt=0):
        """
        Computes the full data and derivatives
        - "zn"
        - "dzndz" (only if *interior_detect* is True)
        - "dzndc"
        
        Note: if *interior_detect* is false we still allocate the arrays for
        *dzndz* but we do not iterate.
        """ 
        M_divergence_sq = M_divergence**2
        complex_codes = ["zn", "dzndz", "dzndc"]
        int_codes = ["iref"]
        stop_codes = ["max_iter", "divergence", "stationnary",
                      "dyn_glitch", "divref_glitch"]
        codes = (complex_codes, int_codes, stop_codes)
        
        if SA_params is None:
            FP_fields = [0, 1, 2]
            FP_params = {"FP_codes": [complex_codes[f] for f in FP_fields],
                         "init_array": [mpmath.mp.zero, mpmath.mp.zero,
                                        mpmath.mp.zero]}
        else:
            # If SA activated, derivatives will be deduced.
            FP_fields = [0]
            FP_params = {"FP_codes": [complex_codes[f] for f in FP_fields],
                         "init_array": [mpmath.mp.zero]}#, mpmath.mp.zero]}

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
            if SA_params is None:
                    FP_array[2] = 2. * FP_array[2] * FP_array[0] + 1.
            FP_array[0] = FP_array[0]**2 + c0

            # If FP iteration is divergent, raise the semaphore n_iter
            # We use the sqare not the disc for obvious saving
            if ((abs((FP_array[0]).real) > M_divergence)
                or (abs((FP_array[0]).imag) > M_divergence)):
                print("Reference point FP iterations escaping at", n_iter)
                return n_iter

        def SA_init(cutdeg):
            return [fsx.Xrange_SA([0.], cutdeg=cutdeg)]

        def SA_loop(P, n_iter, ref_path, kcX):
            """ Series Approximation loop
            Note that derivatives w.r.t dc will be deduced directly from the
            S.A polynomial.
            """
            P[0] = P[0] * (P[0] + 2. * ref_path[0]) + kcX

        def initialize(Z, U, c, chunk_slice, iref):
            Z[2, :] = 0.
            Z[1, :] = 1.
            Z[0, :] = 0.
            U[0, :] = iref

        def iterate(Z, U, c, stop_reason, n_iter, SA_iter,
                    ref_div_iter, ref_path, ref_path_next):
            """
            dz(n+1)dc   <- 2. * dzndc * zn + 1.
            dz(n+1)dz   <- 2. * dzndz * zn
            z(n+1)      <- zn**2 + c 
            
            Termination codes
            0 -> max_iter reached ('interior')
            1 -> M_divergence reached by np.abs(zn)
            2 -> dzn stationnary ('interior early detection')
            3 -> glitched (Dynamic glitches...)
            4 -> glitched (Ref point diverging  ...)
            """
            if SA_params is None:
                Z[2, :] = 2. * (ref_path[2] * Z[0, :] + ref_path[0] * Z[2, :]
                                + Z[0, :] * Z[2, :])
            else:
                Z[2, :] = 2. * (ref_path[0] * Z[2, :] + Z[0, :] * Z[2, :])

            if interior_detect and (n_iter > SA_iter + 1):
                Z[1, :] = 2. * (ref_path[0] * Z[1, :] + Z[0, :] * Z[1, :])

            Z[0, :] = Z[0, :] * (Z[0, :] + 2. * ref_path[0]) + c

            if n_iter >= max_iter:
                # Flagged as 'non-diverging point'
                stop_reason[0, :] = 0
                return

            if n_iter >= ref_div_iter:
                # Flagged as 'ref point diverging' glitch
                # This cannot occur during "final glitch correction loop"
                # as ref 0 point is non-escaping.
                stop_reason[0, :] = 4 
                return

            # Interior points detection
            if interior_detect and (n_iter > SA_iter):
                bool_stationnary = ((Z[1, :].real)**2 +  # + ref_path_next[1].real
                        (Z[1, :].imag)**2 < # + ref_path_next[1].imag
                         epsilon_stationnary**2)
                stop_reason[0, bool_stationnary] = 2

            

#            if self.Xrange_complex_type:
#                full_sq_norm = (Z[0, :] + ref_path_next[0]).abs2()
#            else:
#                full_sq_norm = ((Z[0, :].real + ref_path_next[0].real)**2 + 
#                                (Z[0, :].imag + ref_path_next[0].imag)**2)
            # We save one square norm calculation.
            ZZ = Z[0, :] + ref_path_next[0]
            # full_sq_norm = np.maximum(np.abs(ZZ.real), np.abs(ZZ.imag))
            if self.Xrange_complex_type:
                full_sq_norm = ZZ.abs2()
            else:
                full_sq_norm = ZZ.real**2 + ZZ.imag**2

            # Flagged as 'diverging'
            bool_infty = (full_sq_norm > M_divergence_sq)
            stop_reason[0, bool_infty] = 1

            # Glitch detection
            if glitch_max_attempt == 0:
                return
            ref_sq_norm = ref_path_next[0].real**2 + ref_path_next[0].imag**2
#            ref_sq_norm = np.maximum(np.abs(ref_path_next[0].real),
#                                     np.abs(ref_path_next[0].imag))

            # Flagged as "dynamic glitch"
            bool_glitched = (full_sq_norm  < (ref_sq_norm * glitch_eps))
            stop_reason[0, bool_glitched] = 3
            # We generate a glitch_sort_key based on 'close to secondary
            # nucleus' criteria
            # We use d2zndc2 field to save it, as specified by glitch_sort_key
            # parameter of self.cycles call.
            Z[1, bool_glitched] = full_sq_norm[bool_glitched]

        # Lauching cycles, with glitch correction 'ON'
        self.cycles(FP_loop, FP_params, SA_init, SA_loop, SA_params, max_iter,
               initialize, iterate, subset, codes,
               file_prefix, pc_threshold,
               glitch_stop_index=3,
               glitch_sort_key="dzndz",
               glitch_max_attempt=glitch_max_attempt)



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
