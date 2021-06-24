# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mpmath
import numba

import fractalshades as fs
import fractalshades.numpy_utils.xrange as fsx
import fractalshades.utils as fsutils









class Classical_mandelbrot(fs.Fractal):
    def __init__(self, *args, **kwargs):
        # default values used for postprocessing (potential)
        self.potential_kind = "infinity"
        self.potential_d = 2
        self.potential_a_d = 1.
        super().__init__(*args, **kwargs)

    def explore(self, x, y, dx, max_iter, M_divergence, epsilon_stationnary,
                pc_threshold, px_ball):
        """
        Produces "explore.png" image for GUI navigation
        """
        self.x = x
        self.y = y
        self.dx = dx

        # clean-up the *.tmp files to force recalculation
        self.clean_up("explore")

        self.explore_loop(
            file_prefix="explore",
            subset=None,
            max_iter=max_iter,
            M_divergence =M_divergence,
            epsilon_stationnary=epsilon_stationnary,
            pc_threshold=pc_threshold,
            px_ball=px_ball)

        plotter = fs.Fractal_plotter(
            fractal=self,
            base_data_key=("raw", {"code": "ball_period"}),
            base_data_prefix="explore",
            base_data_function=lambda x: np.log(1. + x),
            colormap=-fs.Fractal_colormap((0.1, 1.0, 200), plt.get_cmap("magma")),
            probes_val=[0., 1.],
            probes_kind="qt",
            mask=None)

        layer_key = ("DEM_explore",
            {"px_snap": 0.5, "potential_dic": {"kind": "infinity"}})
        plotter.add_grey_layer(
            postproc_key=layer_key,
            intensity=0.5, 
            normalized=False,
            skewness=0.0, 
            shade_type={"Lch": 2., "overlay": 1., "pegtop": 2.})

        plotter.plot("explore")


    def explore_loop(self, file_prefix, subset, max_iter, M_divergence,
                   epsilon_stationnary, pc_threshold=1., px_ball=3.):
        """
        Only computes the necessary to draw level set of real iteration number
        + interior point detection for speed-up
        + ball method implementation
        """ 
        complex_codes = ["zn", "dzndz", "dzndc", "_rn", "_rc"]
        int_codes = ["ball_period"]
        stop_codes = ["max_iter", "divergence", "stationnary"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            Z[0, :] = 0.
            Z[1, :] = 0.
            Z[2, :] = 0.

            # Ball arithmetic
            px = np.ravel(self.px_chunk(chunk_slice))
            if subset is not None:
                chunk_mask = np.ravel(subset[chunk_slice])
                px = px[chunk_mask]
            ball_diam = px_ball * px
            Z[3, :] = ball_diam
            Z[4, :] = ball_diam
            U[0, :] = 0.

        def iterate(Z, U, c, n_iter):
            """
            dz(n+1)dc   <- 2. * dzndc * zn + 1.
            dz(n+1)dz   <- 2. * dzndz * zn
            z(n+1)      <- zn**2 + c
            """
            Z[2, :] = 2. * Z[2, :] * Z[0, :] + 1.
            Z[1, :] = 2. * Z[1, :] * Z[0, :]
            Z[0, :] = Z[0, :]**2 + c

            ball_rolling = (U[0, :] == 0)
            if np.any(ball_rolling):
                rn = np.real(Z[3, :])
                rc = np.real(Z[4, :])
                abs_zn = np.empty_like(Z[0, :])
                abs_zn[ball_rolling] = np.abs(Z[0, ball_rolling])
                U[0, ball_rolling & (rn > abs_zn)] = n_iter
                np.putmask(Z[3, :], ball_rolling, 
                           rn**2 + 2. * rn * abs_zn + rc)

            if n_iter == 1:
                if np.max(Z[1, :]) > epsilon_stationnary:
                    raise ValueError("mandatory to start at a critical point")
                else:
                    print("Initialisation at critical point OK")
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
            bool_infty = bool_infty | ~np.isfinite(Z[0, :])
            stop_reason[0, bool_infty] = 1

            bool_stationnary = (Z[1, :].real**2 + Z[1, :].imag**2 <
                                epsilon_stationnary**2)
            stop_reason[0, bool_stationnary] = 2

        # Now we can launch the loop over all pixels
        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)


#==============================================================================
    def first_loop(self, file_prefix, subset, max_iter, M_divergence,
                   epsilon_stationnary, pc_threshold=0.2):
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
            d2z(n+1)dc2 <- 2. * (d2zndc2 * zn + dzndc**2)
            dz(n+1)dc   <- 2. * dzndc * zn + 1.
            dz(n+1)dz   <- 2. * dzndz * zn
            z(n+1)      <- zn**2 + c 
            """
            
                
                
#                x = Z[0, ball_rolling].real
#                y = Z[0, ball_rolling].imag
#                p = np.sqrt((x - 0.25)**2 + y**2)
#                inside_cardiod = (x < (p - 2. * p**2 + 0.25))
#                r_candidate = self.subsubset(ball_rolling, inside_cardiod)
#                U[0, r_candidate] = (n_iter - 1)# - U[1, inside_cardiod]
#                #U[1, inside_cardiod] = (n_iter - 1)
            
            
            Z[3, :] = 2. * (Z[3, :] * Z[0, :] + Z[2, :]**2)
            Z[2, :] = 2. * Z[2, :] * Z[0, :] + 1.
            Z[1, :] = 2. * Z[1, :] * Z[0, :]
            Z[0, :] = Z[0, :]**2 + c


#            ball_rolling = (U[0, :] == 0)
#            if np.any(ball_rolling):
#                rn = Z[4, ball_rolling]
#                rc = Z[5, ball_rolling]
#                abs_zn = np.abs(Z[0, ball_rolling])
#                rcMabs_dzndc = rc * np.abs(Z[2, ball_rolling])
#
##                contains_origin = (rn + rcMabs_dzndc > abs_zn)
#                contains_origin = (rn.real > abs_zn)
#                r_candidate = self.subsubset(ball_rolling, contains_origin)
#                if np.sum(r_candidate) > 10:
#                    print("found", np.sum(r_candidate), "order", n_iter)
#                U[0, r_candidate] = n_iter
##
##                Z[4, ball_rolling] = (rn**2 + 2. * rn * (
##                        abs_zn + rcMabs_dzndc) + rcMabs_dzndc**2)
#                Z[4, ball_rolling] = (rn**2 + 2. * rn * abs_zn + rc)
            
            

            if n_iter == 1:
                if np.max(Z[1, :]) > epsilon_stationnary:
                    raise ValueError("mandatory to start at a critical point")
                else:
                    print("Initialisation at critical point OK")
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
            
            bool_infty = bool_infty | ~np.isfinite(Z[0, :])
            stop_reason[0, bool_infty] = 1

            # more complex ...
            bool_stationnary = (Z[1, :].real**2 + Z[1, :].imag**2 <
                                epsilon_stationnary)
            #ball_rolling = (U[0, :] == 0)
            stop_reason[0, bool_stationnary] = 2

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)


#==============================================================================
    def r_candidate_cycles(self, file_prefix, subset, max_iter, M_divergence,
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
#            get_raw_data(self, code, kind, file_prefix,
#                      chunk_slice, return_as_list=False)

            start_z = self.get_raw_data(code="zn", kind="complex",
                        file_prefix=start_file, chunk_slice=chunk_slice)
            start_z = np.ravel(start_z)
            chunk_mask = np.ravel(subset[chunk_slice])#ix:ixx, iy:iyy])

            Z[0, :] = start_z[chunk_mask] # tortoise
            Z[1, :] = Z[0, :]**2 + c      #hare

            U[0, :] = 1 # power
            U[1, :] = 1 # lambda = r_candidate

        def iterate(Z, U, c, n):
            
            new_pw2 = (U[0, :] == U[1, :])
            Z[0, new_pw2] = Z[1, new_pw2]               # tortle teleportation
            U[0, new_pw2] = U[0, new_pw2] * k_power + 1 # Modified vs Brent
            U[1, new_pw2] = 0  # lambda

            Z[1, :] = Z[1, :]**2 + c  # iterate z
            U[1, :] += 1

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for a new minimum reached
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_infty = (Z[0, :].real**2 + Z[0, :].imag**2 >
                          M_divergence**2)
            bool_infty = bool_infty | ~np.isfinite(Z[0, :])
            stop_reason[0, bool_infty] = 1

            bool_cycle = (np.abs(Z[0, :] - Z[1, :]) < eps_pixel)
            stop_reason[0, bool_cycle] = 2

        self.cycles(initialize, iterate, terminate, subset, codes, file_prefix,
                    pc_threshold=pc_threshold)
        

#==============================================================================
    def teleportation(self, file_prefix, subset, max_iter, M_divergence,
                      eps_pixel, start_file, r_candidate_file,
                      pc_threshold=0.2):
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
            chunk_mask = np.ravel(subset[chunk_slice])#ix:ixx, iy:iyy])
            # teleportation at indx n of the tortoise
            for i_field, code_field in enumerate(["zn", "dzndz", "dzndc",
                                                  "d2zndc2"]):
                temp = self.get_raw_data(code=code_field, kind="complex",
                        file_prefix=start_file, chunk_slice=chunk_slice)
                Z[i_field, :] = np.ravel(temp)[chunk_mask] # tortoise
            Z[4, :] = np.copy(Z[0, :]) # hare = tortoise

            # Known r candidate for cycle         
#            temp = []
            temp = self.get_raw_data(code="r_candidate", kind="int",
                        file_prefix=r_candidate_file, chunk_slice=chunk_slice)
            U[0, :] = np.ravel(temp)[chunk_mask] # pre computed r_candidate

            # The teleportation iteration number         
            temp = self.get_raw_data(code="stop_iter", kind="stop_iter",
                        file_prefix=start_file, chunk_slice=chunk_slice)
            U[1, :] = np.ravel(temp)[chunk_mask] # pre computed r_candidate

            if Z.shape[1] != 0:
                print("U range_field, code_field", np.nanmin(U), np.nanmax(U))

        def iterate(Z, U, c, n):
            """
            """
            hare_running = (U[0, :] != 0)

            # first we make the hare run to position
            if np.any(hare_running):
                U[0, hare_running] -= 1
                Z[4, hare_running] = Z[4, hare_running]**2 + c[hare_running]
                return None

            # If the hare is in position, back to 'normal' cycles
            # iterate tortle
            Z[3, :] = 2. * (Z[3, :] * Z[0, :] + Z[2, :]**2)
            Z[2, :] = 2. * Z[2, :] * Z[0, :] + 1.
            Z[1, :] = 2. * Z[1, :] * Z[0, :]
            Z[0, :] = Z[0, :]**2 + c
            # iterate hare
            Z[4, :] = Z[4, :]**2 + c
            U[1, :] += 1

        def terminate(stop_reason, Z, U, c, n_iter):
            """
            Tests for the first cycle reached
            """
            if n_iter > max_iter:
                stop_reason[0, :] = 0

            bool_infty = (Z[0, :].real**2 + Z[0, :].imag**2 >
                          M_divergence**2)
            bool_infty = bool_infty | ~np.isfinite(Z[0, :])
            stop_reason[0, bool_infty] = 1

            bool_cycle = (np.abs(Z[0, :] - Z[4, :]) < eps_pixel)
            stop_reason[0, bool_cycle] = 2

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

        complex_codes = ["zr", "attractivity", "dzrdc", "dattrdc", "d2attrdc2"]
        int_codes = ["r_candidate", "r"]
        stop_codes = ["max_cycle", "cycle_confirmed"]
        codes = (complex_codes, int_codes, stop_codes)

        def initialize(Z, U, c, chunk_slice):
            """
            """
            (ix, ixx, iy, iyy) = chunk_slice
            chunk_mask = np.ravel(subset[chunk_slice])#ix:ixx, iy:iyy])
            temp = self.get_raw_data(code="zn", kind="complex",
                         file_prefix=start_file, chunk_slice=chunk_slice)
            Z[0, :] = np.ravel(temp)[chunk_mask]
            Z[1, :] = 0

            temp = self.get_raw_data(code="dzndc", kind="complex",
                         file_prefix=start_file, chunk_slice=chunk_slice)
            Z[2, :] = np.ravel(temp)[chunk_mask]
            Z[3, :] = 0

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
                is_divisor = ((U[0, :] % n == 0) & (
                               n % known_order == 0))

            if np.any(is_divisor): 
                n_loop = np.count_nonzero(is_divisor)
                print("in newton loop", n, "entering num: ", n_loop)
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
                        print("active", np.count_nonzero(~newton_cv), " / ", n_loop)
                        print("i_newton", i_newton, " / ", max_newton)
                    zr = np.copy(z0_loop)
                    dzrdz = np.zeros_like(zr) + 1.
                    d2zrdzdc = 0 * dzrdz   # == 1. hence : constant wrt c...
                    dzrdc = np.copy(dz0dc_loop)
                    for i in range(n):
                        d2zrdzdc = 2 * (d2zrdzdc * zr + dzrdz * dzrdc)
                        dzrdz = 2. * dzrdz * zr
                        dzrdc = 2 * dzrdc * zr + 1.
                        zr = zr**2 + c_loop
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
                    print("CONFIRMED", np.count_nonzero(is_confirmed),
                          "With cycle order:", n)
                    zi = z0[is_confirmed]
                    dzidc = dz0dc[is_confirmed]
                    attr = z0[is_confirmed] * 0. + 1.   # f'(zi)
                    dattrdc = dz0dc[is_confirmed] * 0.  # constant wrt c
                    for i in range(n):
                        dattrdc = 2 * (dattrdc * zi + attr * dzidc)
                        attr = 2. * attr * zi       # attr = product of dzidz
                        dzidc = 2 * dzidc * zi + 1.
                        zi = zi**2 + c[r_confirmed]

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
