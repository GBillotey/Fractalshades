# -*- coding: utf-8 -*-
import numpy as np
from fractal import Fractal, Color_tools, mkdir_p

import pickle
import copy
import mpmath

from numpy_utils.xrange import Xrange_array, Xrange_polynomial
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
        self.ref_array = {}
        super().__init__(directory, x, y, dx, nx, xy_ratio, theta_deg,
             **kwargs)

    def diff_c_chunk(self, chunk_slice, iref, file_prefix):#, data_type=None):
        """
        Returns a chunk of c_vec for the calculation
        Parameters
         - chunk_span
         - data_type: expected one of np.float64, np.longdouble
        
        Returns: 
        c_vec : [chunk_size x chunk_size] 1d-vec of type datatype
        
                as a "delta" wrt reference
        """
        offset = self.offset_chunk(chunk_slice)
#        if iref is not None:
        print("Shifting c with respect to iref", iref)
        FP_params = self.reload_ref_point(iref, file_prefix, scan_only=True)
        offset[0] += float(self.x - FP_params["ref_point"].real)
        offset[1] += float(self.y - FP_params["ref_point"].imag)
        
        print("diff_c_chunk x", float(self.x - FP_params["ref_point"].real))
        print("diff_c_chunk y", float(self.y - FP_params["ref_point"].imag))

        return offset[0] + offset[1] * 1j
    
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

    def ref_point_file(self, i_ref, file_prefix):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        data chunk
        """
        print("file_prefix", file_prefix)
        return os.path.join(self.directory, "data",
                file_prefix + "_pt{0:d}.ref".format(i_ref))

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
        try:
            return self.ref_array["file_prefix"]
        except KeyError:
            # We need to build it... 
            nref = self.ref_point_count(file_prefix)
            FP_params = self.reload_ref_point(0, file_prefix, scan_only=True)
            max_iter = FP_params["max_iter"]
            FP_codes = FP_params["FP_codes"]
            ref_array = np.empty([nref, max_iter + 1, len(FP_codes)],
                                  dtype=self.complex_type)
            for iref in range(nref):
                FP_params, Z_path = self.reload_ref_point(iref, file_prefix)
                #print("Z_path")
                ref_array[iref, :, :] = Z_path
            self.ref_array["file_prefix"] = ref_array
            return ref_array

    def SA_file(self, file_prefix):
        """
        Returns the file path to store or retrieve params associated with a
        series approximation
        """
        return os.path.join(self.directory, "data", file_prefix + ".SA")

    def save_SA(self, SA_params, file_prefix):
        """
        Write to a dat file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.SA_file(file_prefix)
        mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            print("SA computed, saving", save_path)
            pickle.dump(SA_params, tmpfile, pickle.HIGHEST_PROTOCOL)

    def reload_SA(self, file_prefix):
        """
        """
        save_path = self.SA_file(file_prefix)
        with open(save_path, 'rb') as tmpfile:
            SA_params = pickle.load(tmpfile)
        return SA_params


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
        
        ref_array = self.get_ref_array(file_prefix)[:, :, :]
        full_Z = Z.copy()
        for key, val in complex_dic.items():
#            t = ref_array[U[int_dic["ref_point"], :],
#                                        stop_iter, val]
            full_Z[val, :] += np.ravel(ref_array[U[int_dic["ref_point"], :],
                                        stop_iter, val])
        full_raw_data = (chunk_mask, full_Z, U, stop_reason, stop_iter)

        post_array, chunk_mask = self.postproc(postproc_keys, codes,
            full_raw_data, chunk_slice)
        return self.reshape2d(post_array, chunk_mask, chunk_slice)


class Perturbation_mandelbrot(PerturbationFractal):
    
    def cycles(self, FP_loop, FP_params, SA_loop, SA_params, max_iter,
               initialize, iterate, terminate, subset, codes,
               file_prefix, pc_threshold, iref):

        ref_path = self.ensure_ref_point(FP_loop, FP_params, max_iter,
                file_prefix, iref=iref, c0=None)
        
        if SA_params is not None:
            try:
                SA_params = self.reload_SA(file_prefix)
            except FileNotFoundError:
                self.series_approx(file_prefix, SA_loop, SA_params)
                self.save_SA(SA_params, file_prefix)

        super().cycles(initialize, iterate, terminate, subset, codes,
            file_prefix, chunk_slice=None, pc_threshold=pc_threshold,
            iref=iref, ref_path=ref_path, SA_params=SA_params)

    def glitch_correction_cycles(self, initialize, iterate, terminate, subset, codes,
            file_prefix, chunk_slice=None, pc_threshold=0.2):
        pass
    # https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/cluster/_optics.py#L24

    def ensure_ref_point(self, FP_loop, FP_params, max_iter, file_prefix,
                         iref=0, c0=None):
        """
        # Check if we have at least one reference point stored, otherwise 
        # computes and stores it
        """
        if self.ref_point_count(file_prefix) <= iref:
            if c0 is None:
                c0 = self.x + 1j * self.y

            order = self.ball_method(c0, self.dx / self.nx * 10., max_iter)
            newton_cv, nucleus = self.find_nucleus(c0, order)

            in_image = False
            if newton_cv:
                shift = c0 - nucleus
                in_image = ((abs(shift.real) < self.dx) and
                            (abs(shift.imag) < self.dy))

            if not(in_image):
                select = {np.complex256: np.float128,
                          np.complex128: np.float64}
                data_type = select[self.complex_type]
                rg = np.random.default_rng(0)
                diff = rg.random([2], dtype=data_type)
                c_shifted = (c0 + self.dx * (diff[0] - 0.5) + 
                                  self.dy * (diff[1] - 0.5) * 1j)
                print("*** Relauch with shifted ref point, ", diff)
                self.ensure_ref_point(FP_loop, FP_params, max_iter, file_prefix,
                                      iref, c0=c_shifted)

            print("Reference nucleus found at:\n", nucleus, order)
            print("With img coords:\n", shift.real / self.dx,
                      shift.imag / self.dy)

            FP_params, Z_path = self.compute_ref_point(
                    FP_loop, FP_params, nucleus, max_iter, iref, file_prefix)
        else:
            print("reloading ref point", iref)
            FP_params, Z_path = self.reload_ref_point(iref, file_prefix)
        print("Zpath", Z_path)
        return Z_path

    def compute_ref_point(self, FP_loop, FP_params, c, max_iter,
                          iref, file_prefix):
        """
        Computes full precision, and stores path in normal precision
        """
        FP_params["ref_point"] = c
#        print('FP_params["ref_point"]' , c)
        FP_params["max_iter"] = max_iter
        FP_codes = FP_params["FP_codes"]
        FP_array = copy.copy(FP_params["init_array"])
        print("init_array", FP_params["init_array"])
        Z_path = np.empty([max_iter + 1, len(FP_codes)],
                           dtype=self.complex_type)
        Z_path[0, :] = np.array(FP_array, dtype=self.complex_type)

        for n_iter in range(1, max_iter + 1):
            FP_loop(FP_array, c, n_iter)
            Z_path[n_iter, :] = np.array(FP_array, dtype=self.complex_type)
        
        self.save_ref_point(FP_params, Z_path, iref, file_prefix)

        return FP_params, Z_path
        
            

    def ball_method(self, c, px, maxiter, order=1, M_divergence=1e5):
        """
        Use a ball centered on c = x + i y to find the first period (up to 
        maxiter) of nucleus
        """
        if order == 1:
            return self._ball_method1(c, px, maxiter)
        elif order == 2:
            return self._ball_method2(c, px, maxiter, M_divergence)
        
    @staticmethod
    def _ball_method1(c, px, maxiter):#, M_divergence):
        #c = x + 1j * y
        z = 0.
        r0 = px      # first radius
        r = r0 * 1.
        az = abs(z)
        dzdc = 0.

        for i in range(1, maxiter + 1):
            r = (az  + r)**2 - az**2 + r0
            z = z**2 + c
            dzdc =  2 * z * dzdc + 1.
            az = abs(z)
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
                print("cv@", i_newton, zr, dzrdc)
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

    def series_approx(self, file_prefix, SA_loop, SA_params):
        """
        Zk, zk
        C, c
        """
        P = SA_params["init_P"]
        cutdeg = SA_params["cutdeg"]
        kc = SA_params["kc"]
        SA_valid = True
        _, ref_path = self.reload_ref_point(SA_params["iref"], file_prefix)
        n_iter = 0
        while SA_valid:
            n_iter +=1
            SA_loop(P, n_iter, ref_path[n_iter - 1, :], kc)
            SA_valid = (P[0].tail(kc, cutdeg // 4) < 1.e-15)#n_iter < 10.#np.all(P.tail_bound(radius= , ntail= ) <= eps)
            SA_valid = (SA_valid and (np.all((P[0].coeffs).abs2() < 4.)))
#            print(P[0])
            if not(SA_valid):
                print("SA stop", 
                      n_iter, P[0].tail(1., cutdeg // 4), P[0].tail(kc, cutdeg))
            if n_iter%500 == 0:
                print("SA running", n_iter)
        SA_params["n_iter"] = n_iter
        SA_params["P"] = P


    def dev_loop(self, file_prefix, subset, max_iter, M_divergence,
                   epsilon_stationnary, pc_threshold=1.0, iref=0, 
                   SA_params=None, glitch_eps=1.e-3):
        """
        Only computes the necessary to draw level set of real iteration number
        + interior point detection for speed-up
        """ 
        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2"]
        int_codes = ["ref_point"]
        stop_codes = ["max_iter", "divergence", "stationnary", "glitched"]
        codes = (complex_codes, int_codes, stop_codes)
        
        FP_params = {"FP_codes": complex_codes,
                     "init_array": [mpmath.mp.zero, mpmath.mp.zero,
                                    mpmath.mp.zero, mpmath.mp.zero]}

        if SA_params is not None:
            cutdeg=SA_params["cutdeg"]
            SA_params["iref"] = iref
            SA_params["kc"] = np.sqrt(self.dx**2 +  self.dy**2)
            SA_params["init_P"] = [
                    Xrange_polynomial([0.], cutdeg=cutdeg),
                    Xrange_polynomial([0.], cutdeg=cutdeg),
                    Xrange_polynomial([0.], cutdeg=cutdeg),
                    Xrange_polynomial([0.], cutdeg=cutdeg)]

        def FP_loop(FP_array, c0, n_iter):
            FP_array[3] = 2. * (FP_array[3] * FP_array[0] + FP_array[2]**2)
            FP_array[2] = 2. * FP_array[2] * FP_array[0] + 1.
            FP_array[1] = 2. * FP_array[1] * FP_array[0]
            FP_array[0] = FP_array[0]**2 + c0
            if n_iter == 1:
                FP_array[1] = 1.
            if abs(FP_array[0]) > M_divergence:
                raise ValueError("Expecting a non-escaping reference point.",
                                 n_iter)

        def SA_loop(P, n_iter, ref_path, kc):
            """ SA iterations
            """
            P[3] = 2. * (ref_path[3] * P[0] + 2 * ref_path[2] * P[2] +
                         ref_path[0] * P[3] + P[0] * P[3] + P[2] * P[2])
            P[2] = 2 * (ref_path[2] * P[0] +
                        ref_path[0] * P[2] + P[0] * P[2])
            P[1] = 2 * (ref_path[1] * P[0] +
                        ref_path[0] * P[1] + P[0] * P[1])
            #print("kc", kc, type(kc))
            P[0] = P[0] * (P[0] + 2. * ref_path[0]) + [0., kc]
#
#            if n_iter == 1:
                

        def initialize(Z, U, c, chunk_slice, iref):
            Z[3, :] = 0.
            Z[2, :] = 0.
            Z[1, :] = 0.
            Z[0, :] = 0.
            U[0, :] = iref

        def iterate(Z, U, c, n_iter, ref_path):
            """
            dz(n+1)dz   <- 2. * dzndz * zn
            z(n+1)      <- zn**2 + c 
            """
            Z[3, :] = 2. * (ref_path[3] * Z[0, :] +
                            2 * ref_path[2] * Z[2, :] +
                            ref_path[0] * Z[3, :] +
                            Z[0, :] * Z[3, :] + Z[2, :]**2)
            Z[2, :] = 2. * (ref_path[2] * Z[0, :] +
                            ref_path[0] * Z[2, :] + Z[0, :] * Z[2, :])
            Z[1, :] = 2. * (ref_path[1] * Z[0, :] +
                            ref_path[0] * Z[1, :] + Z[0, :] * Z[1, :])
            Z[0, :] = Z[0, :] * (Z[0, :] + 2. * ref_path[0]) + c

#            if n_iter == 1:
##                if np.max(Z[1, :]) > epsilon_stationnary:
##                    raise ValueError("mandatory to start at a critical point")
##                else:
##                    print("Initialisation at critical point OK")
#                Z[1, :] = 1.

        def terminate(stop_reason, Z, U, c, n_iter, ref_path):
            """
            Tests for a cycle termination
            Shall modify in place stop_reason, if != -1 loop will stop for this
            pixel.
             0 -> max_iter reached
             1 -> M_divergence reached by np.abs(zn)
             2 -> dzn stationnary
            """
            if n_iter >= max_iter:
                print("max iter")
                stop_reason[0, :] = 0
                
#            print("Z[0, :].real", Z[0, :].real)
#            print("M_divergence", M_divergence)
            full_sq_norm = ((Z[0, :].real + ref_path[0].real)**2 + 
                            (Z[0, :].imag + ref_path[0].imag)**2)
            ref_sq_norm = ref_path[0].real**2 + ref_path[0].imag**2

            bool_infty = (full_sq_norm > M_divergence**2)
            bool_infty = bool_infty | ~np.isfinite(Z[0, :])            
            stop_reason[0, bool_infty] = 1
            
#            print("ref_path[1]", ref_path[1], Z[1, :].real)
            # if n_iter > 5:
            bool_stationnary = ((Z[1, :].real + ref_path[1].real)**2 + 
                    (Z[1, :].imag + ref_path[1].imag)**2 <
                     epsilon_stationnary**2)
            stop_reason[0, bool_stationnary] = 2

#            bool_glitched = full_sq_norm  < (ref_sq_norm * glitch_eps**2)
#            stop_reason[0, bool_glitched] = 3

        # Now we can launch the loop over all pixels
        self.cycles(FP_loop, FP_params, SA_loop, SA_params, max_iter,
               initialize, iterate, terminate, subset, codes,
               file_prefix, pc_threshold, iref)


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