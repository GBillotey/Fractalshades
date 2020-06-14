# -*- coding: utf-8 -*-
import numpy as np
from fractal import Fractal


class Nova(Fractal):
    """
    Nova fractal class for :
            f(z) = z**p - 1.
            z(n+1) <- zn - R * f / dfdz + c
    """

#==============================================================================
    def first_loop(self, file_prefix, subset, max_iter, epsilon_cv,
                   epsilon_stationnary, pc_threshold=0.2, R=1.0, p=3, z0=1.,
                   zr_file_prefix=None):
        """
        Applies the "usual" cycles to subset with full result arrays
        """
        complex_codes = ["zn", "dzndz", "dzndc"]
        if zr_file_prefix is not None:
            complex_codes += ["zr", "alpha", "dzrdc"]
        int_codes = []
        stop_codes = ["max_iter", "convergence", "stationnary"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            Z[0, :] = z0
            Z[1, :] = 1.
            Z[2, :] = 0.

            if zr_file_prefix is not None:
                (ix, ixx, iy, iyy) = chunk_slice
                chunk_mask = np.ravel(subset[ix:ixx, iy:iyy])
                zr = []
                self.fill_raw_data(zr, code="zr", kind="complex",
                            file_prefix=zr_file_prefix,
                            chunk_slice=chunk_slice, c=c,
                            return_as_list=True)
                Z[3, :] =  np.ravel(zr)[chunk_mask]
                attractivity = []
                self.fill_raw_data(attractivity, code="attractivity",
                            kind="complex", file_prefix=zr_file_prefix,
                            chunk_slice=chunk_slice, c=c,
                            return_as_list=True)
                Z[4, :] = 1. / np.ravel(attractivity)[chunk_mask]
                dzrdc = []
                self.fill_raw_data(dzrdc, code="dzrdc", kind="complex",
                            file_prefix=zr_file_prefix,
                            chunk_slice=chunk_slice, c=c,
                            return_as_list=True)
                Z[5, :] = np.ravel(dzrdc)[chunk_mask]

        def iterate(Z, U, c, n_iter):
            """
            f(z) = z**p - 1.
            z(n+1) <- zn - R * f / dfdz + c
            """
            zp2 = Z[0, :]**(p-2)
            zp1 = zp2 * Z[0, :]
            zp = zp1 * Z[0, :]
            f = zp - 1.
            dfdz = p * zp1
            d2fdz2 = p * (p-1) * zp2

            Z[2, :] = Z[2, :] * ((1. - R)  + R * f * d2fdz2 / dfdz**2) + 1.
            Z[1, :] = Z[1, :] * ((1. - R)  + R * f * d2fdz2 / dfdz**2)
            Z[0, :] = Z[0, :] - R * f / dfdz + c


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
                bool_cv = ((Z[0, :] - Z[3, :]).real**2 +
                           (Z[0, :] - Z[3, :]).imag**2 <
                              epsilon_cv**2)
                stop_reason[0, bool_cv] = 1
            else:
                # We do not know the limit cycle we iterate rough stabilization
                # ! do not stop on this criteria if zr_file_prefix is given...
                bool_stationnary = (Z[1, :].real**2 + Z[1, :].imag**2 <
                                    epsilon_stationnary**2)
                stop_reason[0, bool_stationnary] = 2

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)


#==============================================================================
    def r_candidate_cycles(self, file_prefix, subset, max_iter,
                           eps_pixel, start_file, k_power=1.02,
                           pc_threshold=0.2, R=1.0, p=3, z0=1.):
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
        complex_codes = ["_zn_tortoise", "_zn_hare"]
        int_codes = ["_power", "r_candidate"]
        stop_codes = ["max_iter", "cycle"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            """
            
            """
            (ix, ixx, iy, iyy) = chunk_slice
            start_z = []
            # teleportation at n
            self.fill_raw_data(start_z, code="zn", kind="complex",
                        file_prefix=start_file, chunk_slice=chunk_slice, c=c,
                        return_as_list=True)
            start_z = np.ravel(start_z)
            chunk_mask = np.ravel(subset[ix:ixx, iy:iyy])
            Z[0, :] = start_z[chunk_mask] # tortoise

            zp1 = Z[0, :]**(p-1)
            dfdz = p * zp1
            f = zp1 * Z[0, :] - 1.
            Z[1, :] = Z[0, :] - R * f / dfdz + c # one iteration

            U[0, :] = 1 # power
            U[1, :] = 1 # lambda = r_candidate

        def iterate(Z, U, c, n):
            """ """
            new_pw2 = (U[0, :] == U[1, :])
            Z[0, new_pw2] = Z[1, new_pw2]               # tortle teleportation
            U[0, new_pw2] = U[0, new_pw2] * k_power + 1 # Modified vs Brent
            U[1, new_pw2] = 0  # lambda


            zp1 = Z[1, :]**(p-1)
            dfdz = p * zp1
            f = zp1 * Z[1, :] - 1.
            Z[1, :] = Z[1, :] - R * f / dfdz + c   # iterate hare
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
                      pc_threshold=0.2, R=1.0, p=3, z0=1.):
        """
        Cycles until condition is met with condition :
        np.abs(zn - zn + r) < eps_pixel, r given
        """
        complex_codes = ["zn", "dzndz", "dzndc", "_zn_hare"]
        int_codes = ["r_candidate", "n_iter"]
        stop_codes = ["max_iter", "cycle"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            """
            
            """
            (ix, ixx, iy, iyy) = chunk_slice
            temp = []
            chunk_mask = np.ravel(subset[ix:ixx, iy:iyy])
            # teleportation at indx n of the tortoise
            for i_field, code_field in enumerate(["zn", "dzndz", "dzndc"]):
                temp = []
                self.fill_raw_data(temp, code=code_field, kind="complex",
                        file_prefix=start_file, chunk_slice=chunk_slice, c=c,
                        return_as_list=True)
                Z[i_field, :] = np.ravel(temp)[chunk_mask] # tortoise
            Z[3, :] = np.copy(Z[0, :]) # hare = tortoise

            # Known r candidate for cycle         
            temp = []
            self.fill_raw_data(temp, code="r_candidate", kind="int",
                        file_prefix=r_candidate_file,
                        chunk_slice=chunk_slice, c=c, return_as_list=True)
            U[0, :] = np.ravel(temp)[chunk_mask] # pre computed r_candidate

            # The teleportation iteration number         
            temp = []
            self.fill_raw_data(temp, code="stop_iter", kind="stop_iter",
                        file_prefix=start_file,
                        chunk_slice=chunk_slice, c=c, return_as_list=True)
            U[1, :] = np.ravel(temp)[chunk_mask]

            if Z.shape[1] != 0:
                print "U range_field, code_field", np.nanmin(U), np.nanmax(U)

        def iterate(Z, U, c, n):
            """
            """
            hare_running = (U[0, :] != 0)

            # first we make the hare run to position
            if np.any(hare_running):
                U[0, hare_running] -= 1

                zp1 = Z[3, hare_running]**(p-1)
                dfdz = p * zp1
                f = zp1 * Z[3, hare_running] - 1.
                dfdz = p * zp1
                Z[3, hare_running] = Z[3, hare_running] - R * f / dfdz + c[
                                                                  hare_running]
                return None

            # If the hare is in position, back to 'normal' cycles 
            # Iterates the tortle
            zp2 = Z[0, :]**(p-2)
            zp1 = zp2 * Z[0, :]
            zp = zp1 * Z[0, :]
            f = zp - 1.
            dfdz = p * zp1
            d2fdz2 = p * (p-1) * zp2
            Z[2, :] = Z[2, :] * ((1. - R)  + R * f * d2fdz2 / dfdz**2) + 1.
            Z[1, :] = Z[1, :] * ((1. - R)  + R * f * d2fdz2 / dfdz**2)
            Z[0, :] = Z[0, :] - R * f / dfdz + c
            
            # iterates the hare
            zp1 = Z[3, :]**(p-1)
            zp = zp1 * Z[3, :]
            f = zp - 1.
            dfdz = p * zp1
            Z[3, :] = Z[3, :] - R * f / dfdz + c
            U[1, :] += 1


        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for the first cycle reached
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_cycle = (np.abs(Z[0, :] - Z[3, :]) < eps_pixel)
            stop_reason[0, bool_cycle] = 1

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)                   


#==============================================================================
    def newton_cycles(self, file_prefix, subset, max_newton, max_cycle,
                      eps_cv, start_file, r_candidate_file,
                      known_order=None, pc_threshold=0.2, R=1.0, p=3, z0=1.):
        """
        Code inspired by :
https://mathr.co.uk/blog/2013-04-01_interior_coordinates_in_the_mandelbrot_set.html
https://mathr.co.uk/blog/2014-11-02_practical_interior_distance_rendering.html
https://en.wikibooks.org/wiki/Fractals/Iterations_in_the_complex_plane/Mandelbrot_set_interior
https://fractalforums.org/fractal-mathematics-and-new-theories/28/interior-colorings/3352
        """

        complex_codes = ["zr", "attractivity", "dzrdc", "dattrdc"]
        int_codes = ["r_candidate", "r"]
        stop_codes = ["max_cycle", "cycle_confirmed"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            """
            """
            (ix, ixx, iy, iyy) = chunk_slice
            chunk_mask = np.ravel(subset[ix:ixx, iy:iyy])
            temp = []
            self.fill_raw_data(temp, code="zn", kind="complex",
                         file_prefix=start_file, chunk_slice=chunk_slice, c=c,
                         return_as_list=True)
            Z[0, :] = np.ravel(temp)[chunk_mask]
            Z[1, :] = 0

            temp = []
            self.fill_raw_data(temp, code="dzndc", kind="complex",
                         file_prefix=start_file, chunk_slice=chunk_slice, c=c,
                         return_as_list=True)
            Z[2, :] = np.ravel(temp)[chunk_mask]
            Z[3, :] = 0

            temp = []
            self.fill_raw_data(temp, code="r_candidate", kind="int",
                         file_prefix=r_candidate_file,
                         chunk_slice=chunk_slice, c=c, return_as_list=True)
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
                print "in newton loop", n, "entering num: ", n_loop
                newton_cv = np.zeros([n_loop], dtype=np.bool)
                z0 = Z[0, is_divisor]     # zn
                dz0dz = z0 * 0. + 1.      # in fact here: dz0dz0
                dz0dc = Z[2, is_divisor]  # dzndc

                z0_loop = np.copy(z0)
                dz0dc_loop = np.copy(dz0dc)
                c_loop = c[is_divisor]
                indices_loop = np.arange(n_loop, dtype=np.int32)
                track_newton_cv = np.zeros(n_loop, dtype=np.bool)

                for i_newton in range(max_newton):     
                    if (i_newton % 5 ==0) and (n_loop > 100):
                        print "active", np.count_nonzero(~newton_cv),
                        print " / ", n_loop
                        print "i_newton", i_newton, " / ", max_newton
                    zr = np.copy(z0_loop)
                    dzrdz = np.zeros_like(zr) + 1.
                    d2zrdzdc = 0 * dzrdz   # == 1. hence : constant wrt c...
                    dzrdc = np.copy(dz0dc_loop)
                    
                    for i in range(n):
                        zp3 = zr**(p-3)
                        zp2 = zp3 * zr
                        zp1 = zp2 * zr
                        zp = zp1 * zr
                        f = zp - 1.
                        dfdz = p * zp1
                        d2fdz2 = p * (p-1) * zp2
                        d3fdz3 = p * (p-1) * (p-2) * zp3

                        d2zrdzdc = (d2zrdzdc * ((1. - R)  + 
                                                R * f * d2fdz2 / dfdz**2) + 
                                    dzrdz * dzrdc * R * (
                                                (f * d3fdz3) / dfdz**2 - 
                                                2. * f * d2fdz2**2 / dfdz**3 + 
                                                d2fdz2 / dfdz))
                        dzrdz = dzrdz * ((1. - R)  + R * f * d2fdz2 / dfdz**2)
                        dzrdc = dzrdc * ((1. - R)  + R * f * d2fdz2 / dfdz**2
                                         ) + 1.
                        zr = zr - R * f / dfdz + c_loop

                    zz = z0_loop - (zr - z0_loop) / (dzrdz - 1.)
                    newton_cv = np.abs(z0_loop - zz) < eps_cv
                    dz0dc_loop = dz0dc_loop - (
                                 (dzrdc - dz0dc_loop) / (dzrdz - 1.) -
                                 (zr - z0_loop) * d2zrdzdc / (dzrdz - 1.)**2)
                    z0_loop = zz

                    if np.any(newton_cv):
                        index_stopped = indices_loop[newton_cv]
                        z0[index_stopped] = z0_loop[newton_cv]
                        dz0dc[index_stopped] = dz0dc_loop[newton_cv]
                        dz0dz[index_stopped] = dzrdz[newton_cv]
                        track_newton_cv[index_stopped] = True
                        
                        z0_loop = z0_loop[~newton_cv]
                        dz0dc_loop = dz0dc_loop[~newton_cv]
                        c_loop = c_loop[~newton_cv]
                        indices_loop = indices_loop[~newton_cv]

                    if np.all(newton_cv):
                        break           

                # We have a candidate but is it the good one ?
                is_confirmed = ((np.abs(dz0dz) <= 1.) & track_newton_cv)
                r_confirmed = self.subsubset(is_divisor, is_confirmed)

                if np.any(is_confirmed):
                    print "CONFIRMED", np.count_nonzero(is_confirmed),
                    print "With cycle order:", n
                    zi = z0[is_confirmed]
                    dzidc = dz0dc[is_confirmed]
                    attr = z0[is_confirmed] * 0. + 1.   # f'(zi)
                    dattrdc = dz0dc[is_confirmed] * 0.  # constant wrt c
                    
                    for i in range(n):
                        zp3 = zi**(p-3)
                        zp2 = zp3 * zi
                        zp1 = zp2 * zi
                        zp = zp1 * zi
                        f = zp - 1.
                        dfdz = p * zp1
                        d2fdz2 = p * (p-1) * zp2
                        d3fdz3 = p * (p-1) * (p-2) * zp3

                        dattrdc = (dattrdc * ((1. - R)  + 
                                              R * f * d2fdz2 / dfdz**2) + 
                                   attr * dzidc * R * (
                                              (f * d3fdz3) / dfdz**2 -
                                               2. * f * d2fdz2**2 / dfdz**3 +
                                               d2fdz2 / dfdz))
                        attr = attr * ((1. - R)  + R * f * d2fdz2 / dfdz**2)
                        dzidc = dzidc * ((1. - R)  + R * f * d2fdz2 / dfdz**2
                                          ) + 1.
                        zi = zi - R * f / dfdz + c[r_confirmed]

                    Z[0, r_confirmed] = z0[is_confirmed]
                    Z[1, r_confirmed] = attr
                    Z[2, r_confirmed] = dz0dc[is_confirmed]
                    Z[3, r_confirmed] = dattrdc
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