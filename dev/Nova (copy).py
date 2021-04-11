# -*- coding: utf-8 -*-
import numpy as np
from fractal import Fractal


class Nova(Fractal):
    """
    Nova fractal class for :
            f(z) = z**p - 1.
            z(n+1) <- zn - R * f / dfdz + c
    """
    def __init__(self, *args, **kwargs):
        self.R = kwargs.pop("R")
        self.p = kwargs.pop("p")
        self.z0 = self.critical_point # Initialized at critical point
        super().__init__(*args, **kwargs)

    @property
    def critical_point(self):
        R, p = self.R, self.p
        return np.exp(np.log((p - 1.) * R / (p - R)) * (1. / p))


    def d1_iteration(self, zn, dzndc, c):
        R, p = self.R, self.p
        zp1 = zn**(p-1)
        zp = zp1 * zn
        f = zp - 1.
        dfdz = p * zp1
        RfQdfdz = R * f / dfdz
        d2fdz2Qdfdz = (p-1.) / zn
        iter_dzndc = dzndc * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz) + 1.
        iter_zn = zn - RfQdfdz + c
        return (iter_zn, iter_dzndc)

#    @staticmethod
#    def interior_distance(z_star, R=None, p=None):
#        
#        zp1 = z_star**(p-1)
#        zp = zp1 * z_star
#        f = zp - 1.
#        dfdz = p * zp1
#        RfQdfdz = R * f / dfdz
#        d2fdz2Qdfdz = (p-1.) / z_star
#        d3fdz3Qdfdz = (p-2) * d2fdz2Qdfdz / z_star
#        
##        d2z0dcdz = 0.
##        dz0dc = 1.
#        d2zsdz2 = (RfQdfdz * d3fdz3Qdfdz - 2. * RfQdfdz * d2fdz2Qdfdz**2 +
#                    R * d2fdz2Qdfdz)
#        dzsdz = ((1. - R)  + RfQdfdz * d2fdz2Qdfdz)
#        dist = (1. - np.abs(dzsdz)**2) / np.abs(d2zsdz2 / (1. - dzsdz))
#        return dist

#==============================================================================
    def first_loop(self, file_prefix, subset, max_iter, epsilon_cv,
                   epsilon_stationnary, pc_threshold=0.2,
                   zr_file_prefix=None):
        """
        Applies the "usual" cycles to subset with full result arrays
        """
        R, p, z0 = self.R, self.p, self.z0
        
        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2"]
        if zr_file_prefix is not None:
            complex_codes += ["zr", "alpha", "dzrdc", "dattrdc", "d2zrdzdc", "dzrdz"]
        int_codes = []
        stop_codes = ["max_iter", "convergence", "stationnary"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            if z0 == "c":
                Z[0, :] = c
            else:
                Z[0, :] = z0
            Z[1, :] = 1.
            Z[2, :] = 0.

            if zr_file_prefix is not None:
                chunk_mask = np.ravel(subset[chunk_slice])
                zr = self.get_raw_data(code="zr", kind="complex",
                        file_prefix=zr_file_prefix, chunk_slice=chunk_slice)

                Z[4, :] =  np.ravel(zr)[chunk_mask]
                attractivity = self.get_raw_data(code="attractivity",
                        kind="complex",  file_prefix=zr_file_prefix,
                        chunk_slice=chunk_slice)
                Z[5, :] = 1. / np.ravel(attractivity)[chunk_mask]
                dzrdc = self.get_raw_data(code="dzrdc",
                        kind="complex",  file_prefix=zr_file_prefix,
                        chunk_slice=chunk_slice)
                Z[6, :] = np.ravel(dzrdc)[chunk_mask]
                dattrdc = self.get_raw_data(code="dattrdc",
                        kind="complex",  file_prefix=zr_file_prefix,
                        chunk_slice=chunk_slice)
                Z[7, :] = 1. / np.ravel(dattrdc)[chunk_mask]
                d2zrdc2 = self.get_raw_data(code="d2zrdzdc",
                        kind="complex",  file_prefix=zr_file_prefix,
                        chunk_slice=chunk_slice)
                Z[8, :] = 1. / np.ravel(d2zrdc2)[chunk_mask]
                d2zrdc2 = self.get_raw_data(code="dzrdz",
                        kind="complex",  file_prefix=zr_file_prefix,
                        chunk_slice=chunk_slice)
                Z[9, :] = 1. / np.ravel(d2zrdc2)[chunk_mask]


        def iterate(Z, U, c, n_iter):
            """
            f(z) = z**p - 1.
            z(n+1) <- zn - R * f / dfdz + c
            """
            zpm1 = Z[0, :]**(p-1)
            zp = zpm1 * Z[0, :]
            zpp1 = zp * Z[0, :]
            
            f = (1. - R/p) * Z[0, :] + (R/p) / zpm1
            df = (1. - R/p)  + (R/p - R) / zp
            d2f = R * (p-1) / zpp1
#            d3f = R * (1 - p*p) / zpp1 / Z[0, :]

            Z[3, :] = Z[3, :] * df + Z[2, :] * d2f
            Z[2, :] = Z[2, :] * df + 1.
            Z[1, :] = Z[1, :] * df
            Z[0, :] = f + c
            
#            f = zp - 1.
#            dfdz = p * zp1
#            RfQdfdz = R * f / dfdz
#            d2fdz2Qdfdz = (p-1) / Z[0, :]
#
#            Z[2, :] = Z[2, :] * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz) + 1.
#            Z[1, :] = Z[1, :] * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz)
#            Z[0, :] = Z[0, :] - RfQdfdz + c

            if n_iter == 1:
                if np.max(np.abs(Z[1, :])) > 1.e-8:
                    print("Not a critical point, f'(z0) max value {}".format(
                          np.max(np.abs(Z[1, :]))))
                else:
                    print("Initialisation at critical point OK")
                Z[1, :] = 1.


        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for a cycle termination
            Shall modify in place stop_reason, if != -1 loop will stop for this
            pixel.
             0 -> max_iter reached
             1 -> convergence reached to provided zr
             2 -> dzn stationnary
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0
                
            if zr_file_prefix is not None:
                # We know the limit cycle we iterate until convergence
                bool_cv = (np.abs((Z[0, :] - Z[4, :])) < epsilon_cv)
                stop_reason[0, bool_cv] = 1
            else:
                # We do not know the limit cycle we iterate rough stabilization
                # ! do not stop on this criteria if zr_file_prefix is given...
                bool_stationnary = (np.abs(Z[1, :]) < epsilon_stationnary)
                stop_reason[0, bool_stationnary] = 2

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)


#==============================================================================
    def r_candidate_cycles(self, file_prefix, subset, max_iter,
                           eps_pixel, start_file, k_power=1.02,
                           pc_threshold=0.2):
        """
        This cycling shall be restricted to the non-divergent points.
        Follow a modified  Brent's algorithm to find a cycle (real cycle order
        will be a multiple of r)
        Ref :
        https://en.wikipedia.org/wiki/Cycle_detection

        Here we can have a long way before reaching convergence ; we do not
        want the cycle order to "explose" so instead of setting *power* to be
        successive powers of 2 following Brent, we use :
        power = power * k_power + 1  (for detection of maxi 20000-cycles with
        1e6 iterations -> k_power = 1.02)
        """
        R, p = self.R, self.p

        complex_codes = ["_zn_tortoise", "_zn_hare"]
        int_codes = ["_power", "r_candidate"]
        stop_codes = ["max_iter", "cycle"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            """
            
            """
            # teleportation at n
            start_z = self.get_raw_data(code="zn", kind="complex",
                file_prefix=start_file, chunk_slice=chunk_slice)
            
            start_z = np.ravel(start_z)
            if subset is not None:
                chunk_mask = np.ravel(subset[chunk_slice])
                start_z = start_z[chunk_mask]
            Z[0, :] = start_z # tortoise

            zpm1 = Z[0, :]**(p-1)
            f = (1. - R/p) * Z[0, :] + (R/p) / zpm1
            Z[1, :] = f + c # one iteration

            U[0, :] = 1 # power
            U[1, :] = 1 # lambda = r_candidate

        def iterate(Z, U, c, n):
            """ """
            new_pw2 = (U[0, :] == U[1, :])
            Z[0, new_pw2] = Z[1, new_pw2]               # tortle teleportation
            U[0, new_pw2] = U[0, new_pw2] * k_power + 1 # Modified vs Brent
            U[1, new_pw2] = 0  # lambda
            
            zpm1 = Z[1, :]**(p-1)
            f = (1. - R/p) * Z[1, :] + (R/p) / zpm1
            Z[1, :] = f + c   # iterate hare
            U[1, :] += 1

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for a new minimum reached
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_cycle = (np.abs(Z[0, :] - Z[1, :]) < eps_pixel)
            stop_reason[0, bool_cycle] = 1

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)


#==============================================================================
    def teleportation(self, file_prefix, subset, max_iter,
                      eps_pixel, start_file, r_candidate_file,
                      pc_threshold=0.2):
        """
        Cycles until condition is met with condition :
        np.abs(zn - zn + r) < eps_pixel, r given
        """
        R, p = self.R, self.p

        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2", "_zn_hare"]
        int_codes = ["r_candidate", "n_iter"]
        stop_codes = ["max_iter", "cycle"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            """
            
            """
            temp = []
            chunk_mask = np.ravel(subset[chunk_slice])
            # teleportation at indx n of the tortoise
            for i_field, code_field in enumerate(["zn", "dzndz", "dzndc", 
                                                  "d2zndc2"]):
                temp = self.get_raw_data(code=code_field, kind="complex",
                file_prefix=start_file, chunk_slice=chunk_slice)
                Z[i_field, :] = np.ravel(temp)[chunk_mask] # tortoise
            Z[4, :] = np.copy(Z[0, :]) # hare = tortoise

            # Known r candidate for cycle         
            temp = self.get_raw_data(code="r_candidate", kind="int",
                file_prefix=r_candidate_file, chunk_slice=chunk_slice)
            U[0, :] = np.ravel(temp)[chunk_mask] # pre computed r_candidate

            temp = self.get_raw_data(code="stop_iter", kind="stop_iter",
                file_prefix=r_candidate_file, chunk_slice=chunk_slice)
            U[1, :] = np.ravel(temp)[chunk_mask]

            if Z.shape[1] != 0:
                print("U range_field, code_field", np.nanmin(U), np.nanmax(U))

        def iterate(Z, U, c, n):
            """
            """
            hare_running = (U[0, :] != 0)

            # first we make the hare run to position
            if np.any(hare_running):
                U[0, hare_running] -= 1

                zpm1 = Z[4, hare_running]**(p-1)
                f = (1. - R/p) * Z[4, hare_running] + (R/p) / zpm1
                Z[4, hare_running] = f + c[hare_running]   # iterate hare

#                zp1 = Z[3, hare_running]**(p-1)
#                dfdz = p * zp1
#                f = zp1 * Z[3, hare_running] - 1.
#                dfdz = p * zp1
#                Z[3, hare_running] = Z[3, hare_running] - R * f / dfdz + c[
#                                                                  hare_running]
                return None
            print("Hare running finished")
            # If the hare is in position, back to 'normal' cycles 
            # Iterates the tortle
            zpm1 = Z[0, :]**(p-1)
            zp = zpm1 * Z[0, :]
            zpp1 = zp * Z[0, :]
            
            f = (1. - R/p) * Z[0, :] + (R/p) / zpm1
            df = (1. - R/p)  + (R/p - R) / zp
            d2f = R * (p-1) / zpp1

            Z[3, :] = Z[3, :] * df + Z[2, :] * d2f
            Z[2, :] = Z[2, :] * df + 1.
            Z[1, :] = Z[1, :] * df
            Z[0, :] = f + c
            
            
#            zp1 = Z[0, :]**(p-1)
#            zp = zp1 * Z[0, :]
#            f = zp - 1.
#            dfdz = p * zp1
#            RfQdfdz = R * f / dfdz
#            d2fdz2Qdfdz = (p-1) / Z[0, :]
#            Z[2, :] = Z[2, :] * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz) + 1.
#            Z[1, :] = Z[1, :] * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz)
#            Z[0, :] = Z[0, :] - RfQdfdz + c

            # iterates the hare
            zpm1 = Z[4, :]**(p-1)
            f = (1. - R/p) * Z[4, :] + (R/p) / zpm1
            Z[4, :] = f + c
            
#            zp1 = Z[3, :]**(p-1)
#            zp = zp1 * Z[3, :]
#            f = zp - 1.
#            dfdz = p * zp1
#            Z[3, :] = Z[3, :] - R * f / dfdz + c
            
            
            
            U[1, :] += 1

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for the first cycle reached
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_cycle = (np.abs(Z[0, :] - Z[4, :]) < eps_pixel)
            stop_reason[0, bool_cycle] = 1

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)                   


#==============================================================================
    def newton_cycles(self, file_prefix, subset, max_newton, max_cycle,
                      eps_cv, start_file, r_candidate_file,
                      known_order=None, pc_threshold=0.2):
        """
        Code inspired by :
https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html
https://mathr.co.uk/blog/2014-11-02_practical_interior_distance_rendering.html
https://en.wikibooks.org/wiki/Fractals/Iterations_in_the_complex_plane/Mandelbrot_set_interior
https://fractalforums.org/fractal-mathematics-and-new-theories/28/interior-colorings/3352
        """
        R, p = self.R, self.p

        complex_codes = ["zr",  "dzrdz", "dzrdc", "d2zrdzdc", "attractivity", "dattrdc"]
        int_codes = ["r_candidate", "r"]
        stop_codes = ["max_cycle", "cycle_confirmed"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            """
            """
            chunk_mask = np.ravel(subset[chunk_slice])

            temp = self.get_raw_data(code="zn", kind="complex",
                file_prefix=start_file, chunk_slice=chunk_slice)
            Z[0, :] = np.ravel(temp)[chunk_mask]

            temp = self.get_raw_data(code="dzndz", kind="complex",
                file_prefix=start_file, chunk_slice=chunk_slice)
            Z[1, :] = np.ravel(temp)[chunk_mask]
            Z[1, :] = 0

            temp = self.get_raw_data(code="dzndc", kind="complex",
                file_prefix=start_file, chunk_slice=chunk_slice)
            Z[2, :] = np.ravel(temp)[chunk_mask]
            temp = self.get_raw_data(code="d2zndc2", kind="complex",
                file_prefix=start_file, chunk_slice=chunk_slice)
            Z[3, :] = np.ravel(temp)[chunk_mask]

            Z[4, :] = 0
            Z[5, :] = 0

            temp = self.get_raw_data(code="r_candidate", kind="int",
                file_prefix=r_candidate_file, chunk_slice=chunk_slice)
            U[0, :] = np.ravel(temp)[chunk_mask]
            U[1, :] = 0


        def iterate(Z, U, c, n):
            """
            U[0, :] is the cycle detected
            n is the candidate cycle order
            """
            if known_order is None:
                is_divisor = (U[0, :] % n == 0)
            else:
                # May always have a 1-cycle
                is_divisor = ((U[0, :] % n == 0) & (
                               (n == 1) or (n % known_order == 0)))

            if np.any(is_divisor): 
                n_loop = np.count_nonzero(is_divisor)
                print("in newton loop", n, "entering num: ", n_loop)
                newton_cv = np.zeros([n_loop], dtype=np.bool)
                z0 = Z[0, is_divisor]     # zn
                dz0dz = z0 * 0. + 1.      # in fact here: dz0dz0
                dz0dc = Z[2, is_divisor]  # dzndc
#                d2z0dc2 = Z[3, is_divisor]  # d2zndc2
                d2z0dzdc = z0 * 0.

                z0_loop = np.copy(z0)
#                dz0dz_loop = np.copy(dz0dz)
                dz0dc_loop = np.copy(dz0dc)
                dz0dz_loop = np.copy(dz0dz)
                crit = np.copy(dz0dz)
#                d2z0dc2_loop = np.copy(d2z0dc2)
                d2z0dzdc_loop = np.copy(d2z0dzdc)

                c_loop = c[is_divisor]
                indices_loop = np.arange(n_loop, dtype=np.int32)
                track_newton_cv = np.zeros(n_loop, dtype=np.bool)

                for i_newton in range(max_newton):     
                    if (i_newton % 5 ==0) and (n_loop > 100):
                        print("active", np.count_nonzero(~newton_cv),
                              " / ", n_loop)
                        print("i_newton", i_newton, " / ", max_newton)
                    zr = np.copy(z0_loop)
                    dzrdz =  0. * zr + 1.
                    d2zrdzdc = 0 * zr   # == 1. hence : constant wrt c...
                    d2zrdz2 = 0 * zr
#                    d3zrdzdc2 = 0 * zr
                    d3zrdz2dc = 0 * zr

                    dzrdc = np.copy(dz0dc_loop)
                    #d2zrdc2 = np.copy(d2z0dc2_loop)
                    #d2zrdzdc = np.copy(d2z0dzdc_loop)
                    
                    for i in range(n):
                        zpm1 = zr**(p-1)
                        zp = zpm1 * zr
                        zpp1 = zp * zr
                        
                        f = (1. - R/p) * zr + (R/p) / zpm1
                        df = (1. - R/p)  + (R/p - R) / zp
                        d2f = R * (p-1) / zpp1
                        d3f = R * (1 - p*p) / zpp1 / zr
            
#                        d3zrdzdc2 = ((d3zrdzdc2 * df + d2zrdc2 * dzrdz * d2f) +
#                            (2 * d2zrdzdc * dzrdc * d2f + dzrdc * dzrdc * dzrdz * d3f))
                        d3zrdz2dc = ((d3zrdz2dc * df + d2zrdz2 * dzrdc * d2f) + 
                            (2 * d2zrdzdc * dzrdz * d2f +  dzrdz * dzrdz * dzrdc* d3f))

                        d2zrdz2 = d2zrdz2 * df + dzrdz * dzrdz * d2f
#                        print([np.shape(tab) for tab in  (d2zrdc2,  df, dzrdc, d2f)])
#                        d2zrdc2 = d2zrdc2 * df + dzrdc * dzrdc * d2f
                        d2zrdzdc = d2zrdzdc * df + dzrdc * dzrdz * d2f

                        dzrdc = dzrdc * df + 1.
                        dzrdz = dzrdz * df
                        zr = f + c_loop
                        
                        
#                        zp1 = zr**(p-1)
#                        zp = zp1 * zr
#                        f = zp - 1.
#                        dfdz = p * zp1
#                        RfQdfdz = R * f / dfdz
#                        d2fdz2Qdfdz = (p-1) / zr
#                        d3fdz3Qdfdz = (p-2) * d2fdz2Qdfdz / zr
#
#                        d2zrdzdc = (d2zrdzdc * ((1. - R)  + 
#                                                RfQdfdz * d2fdz2Qdfdz) + 
#                                    dzrdz * dzrdc * (
#                                                RfQdfdz * d3fdz3Qdfdz -
#                                                2. * RfQdfdz * d2fdz2Qdfdz**2 +
#                                                R * d2fdz2Qdfdz))
#                        dzrdz = dzrdz * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz)
#                        dzrdc = dzrdc * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz
#                                         ) + 1.
#                        zr = zr - RfQdfdz + c_loop
                        
                        
                        

                    zz = z0_loop - (zr - z0_loop) / (dzrdz - 1.)
                    newton_cv = np.abs(z0_loop - zz) < eps_cv

#                    d2z0dc2_loop = d2z0dc2_loop - ((
#                                 (d2zrdc2 - d2z0dc2_loop) / (dzrdz - 1.) -  
#                                 (dzrdc - dz0dc_loop) * d2zrdzdc / (dzrdz - 1.)**2) -
#                                 (((dzrdc - dz0dc_loop) * d2zrdzdc + (zr - z0_loop) * d3zrdzdc2) / (dzrdz - 1.)**2 -
#                                 (zr - z0_loop) * d2zrdzdc * 2. * d2zrdzdc / (dzrdz - 1.)**3))

                    d2z0dzdc_loop = d2z0dzdc_loop - ((
                                 (d2zrdzdc - d2z0dzdc_loop) / (dzrdz - 1.) -  
                                 (dzrdc - dz0dc_loop) * d2zrdz2 / (dzrdz - 1.)**2) -
                                 (((dzrdz - dz0dz_loop) * d2zrdzdc + (zr - z0_loop) * d3zrdz2dc) / (dzrdz - 1.)**2 -
                                 (zr - z0_loop) * d2zrdzdc * 2. * d2zrdz2 / (dzrdz - 1.)**3))
                        
                    dz0dz_loop = dz0dz_loop - (
                                 (dzrdz - dz0dz_loop) / (dzrdz - 1.) -
                                 (zr - z0_loop) * d2zrdz2 / (dzrdz - 1.)**2)

                    dz0dc_loop = dz0dc_loop - (
                                 (dzrdc - dz0dc_loop) / (dzrdz - 1.) -
                                 (zr - z0_loop) * d2zrdzdc / (dzrdz - 1.)**2)

                    z0_loop = zz

                    if np.any(newton_cv):
                        index_stopped = indices_loop[newton_cv]
                        z0[index_stopped] = z0_loop[newton_cv]
                        dz0dc[index_stopped] = dz0dc_loop[newton_cv]
                        dz0dz[index_stopped] = dz0dz_loop[newton_cv]
                        d2z0dzdc[index_stopped] = d2z0dzdc_loop[newton_cv]

                        crit[index_stopped] = dzrdz[newton_cv]
                        track_newton_cv[index_stopped] = True
                        
                        z0_loop = z0_loop[~newton_cv]
                        dz0dc_loop = dz0dc_loop[~newton_cv]
                        dz0dz_loop = dz0dz_loop[~newton_cv]
#                        d2z0dc2_loop = d2z0dc2_loop[~newton_cv]
                        d2z0dzdc_loop = d2z0dzdc_loop[~newton_cv]
                        c_loop = c_loop[~newton_cv]
                        indices_loop = indices_loop[~newton_cv]

                    if np.all(newton_cv):
                        break           

                # We have a candidate but is it the good one ?
                is_confirmed = ((np.abs(crit) <= 1.) & track_newton_cv)
                r_confirmed = self.subsubset(is_divisor, is_confirmed)

                if np.any(is_confirmed):
                    print("CONFIRMED", np.count_nonzero(is_confirmed),
                          "With cycle order:", n)
                    zi = z0[is_confirmed]
                    dzidc = dz0dc[is_confirmed]
                    attr = z0[is_confirmed] * 0. + 1.   # f'(zi)
                    dattrdc = dz0dc[is_confirmed] * 0.  # constant wrt c

                    for i in range(n):
                        zp1 = zi**(p-1)
                        zp = zp1 * zi
                        f = zp - 1.
                        dfdz = p * zp1
                        RfQdfdz = R * f / dfdz
                        d2fdz2Qdfdz = (p-1) / zi
                        d3fdz3Qdfdz = (p-2) * d2fdz2Qdfdz / zi

                        dattrdc = (dattrdc * ((1. - R)  + 
                                                RfQdfdz * d2fdz2Qdfdz) + 
                                    attr * dzidc * (
                                                RfQdfdz * d3fdz3Qdfdz - 
                                                2. * RfQdfdz * d2fdz2Qdfdz**2 +
                                                R * d2fdz2Qdfdz))
                        attr = attr * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz)
                        dzidc = dzidc * ((1. - R)  + RfQdfdz * d2fdz2Qdfdz
                                         ) + 1.
                        zi = zi - RfQdfdz + c[r_confirmed]

                    Z[0, r_confirmed] = z0[is_confirmed]
                    Z[1, r_confirmed] = dz0dz[is_confirmed]
                    Z[2, r_confirmed] = dz0dc[is_confirmed]
                    Z[3, r_confirmed] = d2z0dzdc[is_confirmed]
                    Z[4, r_confirmed] = attr
                    Z[5, r_confirmed] = dattrdc

                    U[0, r_confirmed] = n
                    U[1, r_confirmed] = n

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            exit the loop if:
            iter > max r_canditates
            """
            if n_iter > max_cycle:
                stop_reason[0, :] = 0

            cv = (U[1, :] != 0)
            stop_reason[0, cv] = 1

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)