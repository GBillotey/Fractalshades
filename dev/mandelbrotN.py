# -*- coding: utf-8 -*-
import numpy as np
from fractal import Fractal


class mandelbrotN(Fractal):
    
    @staticmethod
    def pN(base, N):
        """
        Fast integer exponentiation
        """
        _base = np.copy(base)
        exponent = N
        result = np.ones_like(base)
        while exponent > 0:
            if (exponent % 2 == 1):
                result = result * _base
            exponent = exponent >> 1
            if exponent > 0:
                _base *= _base
        return result


    def draft_loop(self, file_prefix, subset, max_iter, M_divergence,
                   epsilon_stationnary, pc_threshold=1.0, d=6):
        """
        Only computes the necessary to draw level set of real iteration number
        + interior point detection for speed-up
        """ 
        complex_codes = ["zn", "dzndz"]
        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            Z[0, :] = 0.
            Z[1, :] = 0.

        def iterate(Z, U, c, n_iter):
            """
            dz(n+1)dz   <- N * dzndz * zn**N-1
            z(n+1)      <- zn**N + c 
            """
            ZpN1 = self.pN(Z[0, :], d-1)
            Z[1, :] = d * ZpN1 * Z[0, :]
            Z[0, :] = Z[0, :] * ZpN1 + c

            if n_iter == 1:
                if np.max(Z[1, :]) > epsilon_stationnary:
                    raise ValueError("mandatory to start at a critical point")
                else:
                    print "Initialisation at critical point OK"
                Z[1, :] = 1.

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for a cycle termination
            Shall modify in place stop_reason, if != -1 loop will stop for this
            pixel.
             0 -> max_iter reached
             1 -> M_divergence reached by np.abs(zn)
             2 -> dzn stationnary
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_infty = (Z[0, :].real**2 + Z[0, :].imag**2 >
                          M_divergence**2)
            stop_reason[0, bool_infty] = 1

            bool_stationnary = (Z[1, :].real**2 + Z[1, :].imag**2 <
                                epsilon_stationnary**2)
            stop_reason[0, bool_stationnary] = 2

        # Now we can launch the loop over all pixels
        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)


#==============================================================================
    def first_loop(self, file_prefix, subset, max_iter, M_divergence,
                   epsilon_stationnary, pc_threshold=0.2, d=6):
        """
        Applies the "usual" cycles to subset with full result arrays
        """
        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2"]
        int_codes = []
        stop_codes = ["max_iter", "divergence", "stationnary"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            Z[0, :] = 0.
            Z[1, :] = 0.
            Z[2, :] = 0.
            Z[3, :] = 0.

        def iterate(Z, U, c, n_iter):
            """
            d2z(n+1)dc2 <- N * (d2zndc2 * zn**(N-1) + dzndc**2 * (N-1) * zn**(N-2))
            dz(n+1)dc   <- N * dzndc * zn**(N-1) + 1.
            dz(n+1)dz   <- N * dzndz * zn**(N-1)
            z(n+1)      <- zn**N + c 
            """
            ZpN2 = self.pN(Z[0, :], d-2)
            ZpN1 = Z[0, :] * ZpN2
            Z[3, :] = d * (Z[3, :] * ZpN1 + Z[2, :]**2 * (d-1) * ZpN2)
            Z[2, :] = d * Z[2, :] * ZpN1 + 1.
            Z[1, :] = d * Z[1, :] * ZpN1
            Z[0, :] = Z[0, :] * ZpN1 + c

            if n_iter == 1:
                if np.max(Z[1, :]) > epsilon_stationnary:
                    raise ValueError("mandatory to start at a critical point")
                else:
                    print "Initialisation at critical point OK"
                Z[1, :] = 1.

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for a cycle termination
            Shall modify in place stop_reason, if != -1 loop will stop for this
            pixel.
             0 -> max_iter reached
             1 -> M_divergence reached by np.abs(zn)
             2 -> dzn stationnary
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_infty = (Z[0, :].real**2 + Z[0, :].imag**2 >
                          M_divergence**2)
            stop_reason[0, bool_infty] = 1

            bool_stationnary = (Z[1, :].real**2 + Z[1, :].imag**2 <
                                epsilon_stationnary**2)
            stop_reason[0, bool_stationnary] = 2

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)


#==============================================================================
    def r_candidate_cycles(self, file_prefix, subset, max_iter, M_divergence,
                           eps_pixel, start_file, k_power=1.02,
                           pc_threshold=0.2, d=6):
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
        stop_codes = ["max_iter", "divergence", "cycle"]
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
            Z[1, :] = self.pN(Z[0, :], d) + c      #hare

            U[0, :] = 1 # power
            U[1, :] = 1 # lambda = r_candidate

        def iterate(Z, U, c, n):
            
            new_pw2 = (U[0, :] == U[1, :])
            Z[0, new_pw2] = Z[1, new_pw2]               # tortle teleportation
            U[0, new_pw2] = U[0, new_pw2] * k_power + 1 # Modified vs Brent
            U[1, new_pw2] = 0  # lambda

            Z[1, :] = self.pN(Z[1, :], d) + c  # iterate z
            U[1, :] += 1

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for a new minimum reached
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_infty = (Z[0, :].real**2 + Z[0, :].imag**2 >
                          M_divergence**2)
            stop_reason[0, bool_infty] = 1

            bool_cycle = (np.abs(Z[0, :] - Z[1, :]) < eps_pixel)
            stop_reason[0, bool_cycle] = 2

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)
        

#==============================================================================
    def teleportation(self, file_prefix, subset, max_iter, M_divergence,
                      eps_pixel, start_file, r_candidate_file,
                      pc_threshold=0.2, d=6):
        """
        Cycles until condition is met with condition :
        np.abs(zn - zn + r) < eps_pixel, r given
        """
        complex_codes = ["zn", "dzndz", "dzndc", "d2zndc2", "_zn_hare"]
        int_codes = ["r_candidate", "n_iter"]
        stop_codes = ["max_iter", "divergence", "cycle"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            """
            
            """
            (ix, ixx, iy, iyy) = chunk_slice
            temp = []
            chunk_mask = np.ravel(subset[ix:ixx, iy:iyy])
            # teleportation at indx n of the tortoise
            for i_field, code_field in enumerate(["zn", "dzndz", "dzndc",
                                                  "d2zndc2"]):
                temp = []
                self.fill_raw_data(temp, code=code_field, kind="complex",
                        file_prefix=start_file, chunk_slice=chunk_slice, c=c,
                        return_as_list=True)
                Z[i_field, :] = np.ravel(temp)[chunk_mask] # tortoise
            Z[4, :] = np.copy(Z[0, :]) # hare = tortoise

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
            U[1, :] = np.ravel(temp)[chunk_mask] # pre computed r_candidate

            if Z.shape[1] != 0:
                print "U range_field, code_field", np.nanmin(U), np.nanmax(U)

        def iterate(Z, U, c, n):
            """
            """
            hare_running = (U[0, :] != 0)

            # first we make the hare run to position
            if np.any(hare_running):
                U[0, hare_running] -= 1
                Z[4, hare_running] = (self.pN(Z[4, hare_running], d) + 
                                      c[hare_running])
                return None

            # If the hare is in position, back to 'normal' cycles
            # iterate tortle
            ZpN2 = self.pN(Z[0, :], d-2)
            ZpN1 = Z[0, :] * ZpN2
            Z[3, :] = d * (Z[3, :] * ZpN1 + Z[2, :]**2 * (d-1) * ZpN2)
            Z[2, :] = d * Z[2, :] * ZpN1 + 1.
            Z[1, :] = d * Z[1, :] * ZpN1
            Z[0, :] = Z[0, :] * ZpN1 + c

            # iterate hare
            Z[4, :] = self.pN(Z[4, :], d) + c
            U[1, :] += 1

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for the first cycle reached
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_infty = (Z[0, :].real**2 + Z[0, :].imag**2 >
                          M_divergence**2)
            stop_reason[0, bool_infty] = 1

            bool_cycle = (np.abs(Z[0, :] - Z[4, :]) < eps_pixel)
            stop_reason[0, bool_cycle] = 2

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)                   


#==============================================================================
    def newton_cycles(self, file_prefix, subset, max_newton, max_cycle,
                      eps_cv, start_file, r_candidate_file,
                      known_order=None, pc_threshold=0.2, d=6,
                      min_order=0):
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
        self.known_order = known_order
        self.local_order = (known_order is None) # defined locally in each 'chunk'

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
            known_order = self.known_order

            if n == 1 and self.local_order:
                self.known_order = None

            if n < min_order:
                return

            if known_order is None:
                is_divisor = (U[0, :] % n == 0)
            else:
                is_divisor = ((U[0, :] % n == 0) & (
                               n % known_order == 0))

            if np.any(is_divisor): 
                n_loop = np.count_nonzero(is_divisor)
                print "in newton loop", n, "entering num: ", n_loop
                stopped = np.zeros([n_loop], dtype=np.bool)
                newton_cv = np.zeros([n_loop], dtype=np.bool)
                newton_dv = np.zeros([n_loop], dtype=np.bool)
                z0 = Z[0, is_divisor]     # zn
                dz0dz = z0 * 0. + 1.      # in fact here: dz0dz0
                dz0dc = Z[2, is_divisor]  # dzndc

                z0_loop = np.copy(z0)
                dz0dc_loop = np.copy(dz0dc)
                c_loop = c[is_divisor]
                indices_loop = np.arange(n_loop, dtype=np.int32)
                track_newton_cv = np.zeros(n_loop, dtype=np.bool)

                for i_newton in range(max_newton):     
                    if (i_newton % 1 == 0) and (n_loop > 100):
                        print "active", np.count_nonzero(~stopped), " / ", n_loop
                        print "i_newton", i_newton, " / ", max_newton
                        print "cv / dv", np.count_nonzero(newton_cv), np.count_nonzero(newton_dv)
                    zr = np.copy(z0_loop)
                    dzrdz = np.zeros_like(zr) + 1.
                    d2zrdzdc = 0 * dzrdz   # == 1. hence : constant wrt c...
                    dzrdc = np.copy(dz0dc_loop)
                    for i in range(n):
                        zrN2 = self.pN(zr, d-2)
                        zrN1 = zr * zrN2
                        d2zrdzdc = d * (d2zrdzdc * zrN1 +
                                        dzrdz * dzrdc * (d-1) * zrN2)
                        dzrdz = d * dzrdz * zrN1
                        dzrdc = d * dzrdc * zrN1 + 1.
                        zr = zr * zrN1 + c_loop
                        
                        
                    zz = z0_loop - (zr - z0_loop) / (dzrdz - 1.)
                    newton_cv = np.abs(z0_loop - zz) < eps_cv
                    dz0dc_loop = dz0dc_loop - (
                                 (dzrdc - dz0dc_loop) / (dzrdz - 1.) -
                                 (zr - z0_loop) * d2zrdzdc / (dzrdz - 1.)**2)
                    z0_loop = zz
                    newton_dv = ~np.isfinite(zz)
                    stopped = newton_cv | newton_dv

                    if np.any(stopped):
                        index_cv= indices_loop[newton_cv]
                        index_stopped = indices_loop[stopped]

                        z0[index_stopped] = z0_loop[stopped]
                        dz0dc[index_stopped] = dz0dc_loop[stopped]
                        dz0dz[index_stopped] = dzrdz[stopped]
                        
                        track_newton_cv[index_cv] = True
                        
                        z0_loop = z0_loop[~stopped]
                        dz0dc_loop = dz0dc_loop[~stopped]
                        c_loop = c_loop[~stopped]
                        indices_loop = indices_loop[~stopped]

                    if np.all(newton_cv):
                        break           

                # We have a candidate but is it the good one ?
                is_confirmed = ((np.abs(dz0dz) <= 1.00) & track_newton_cv)
                #is_confirmed = track_newton_cv
                r_confirmed = self.subsubset(is_divisor, is_confirmed)

                if np.any(is_confirmed):
                    print "CONFIRMED", np.count_nonzero(is_confirmed),
                    print "With cycle order:", n
#                    if (np.count_nonzero(is_confirmed) > 2 and 
#                                                     self.known_order is None):
#                        self.known_order = n
#                        print "Resetting known_order to:", n
                    
                    zi = z0[is_confirmed]
                    dzidc = dz0dc[is_confirmed]
                    attr = z0[is_confirmed] * 0. + 1.   # f'(zi)
                    dattrdc = dz0dc[is_confirmed] * 0.  # constant wrt c
                    for i in range(n):
                        ziN2 = self.pN(zi, d-2)
                        ziN1 = zi * ziN2
                        
                        dattrdc = d * (dattrdc * ziN1 + attr * dzidc * (d-1) * ziN2)
                        attr = d * attr * ziN1       # attr = product of dzidz
                        dzidc = d * dzidc * ziN1 + 1.
                        zi = zi * ziN1 + c[r_confirmed]

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