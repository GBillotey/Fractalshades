# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib.format import open_memmap

import fractalshades as fs
#import fractalshades.numpy_utils.xrange as fsx
import fractalshades.numpy_utils.expr_parser as fs_parser



class Postproc_batch:
    def __init__(self, fractal, calc_name):
        """
        Container for several postprocessing objects (instances of `Postproc`)
        which share the same Fractal & calculation (*calc_name*)
        
        It is usefull to have such container to avoid re-calculation of data
        that can be shared between several post-proicessing objects.
        
        Parameters
        ==========
        fractal : fractalshades.Fractal instance
            The shared fractal object
        calc_name : str 
            The string identifier for the shared calculation
            
        
        """
        self.fractal = fractal
        self.calc_name = calc_name
        self.posts = dict()
        self.postnames_2d = []
        # Temporary data used when iterating
        self.clear_chunk_data()

    @property
    def postnames(self):
        return self.posts.keys()

    def add_postproc(self, postname, postproc):
        """
        Add successive postprocessing and ensure they have consistent
        fractal & calc_name
        
        Parameters
        ==========
        postname : str
            The string identifier that will be associated with this
            post-processing object
        postproc : `Postproc` instance
            The post-processing object to be added

        Notes
        =====

        .. warning::

            A Postproc can only be added to one batch.
        """
        if postproc.batch is not None:
            raise ValueError("Postproc can only be linked to one batch")


        postproc.link_batch(self) # Give access to batch members from postproc
        if postproc.field_count == 1:
            self.posts[postname] = postproc

        elif postproc.field_count == 2:
            # We need to add 2 separate fields
            x_field = postproc.x
            y_field = postproc.y
            x_field.link_sibling(y_field)
            y_field.link_sibling(x_field)
            x_field.link_batch(self)
            y_field.link_batch(self)
            self.posts[postname + "_x"] = x_field
            self.posts[postname + "_y"] = y_field
            self.postnames_2d += [postname]
        else:
            raise ValueError("postproc.field_count bigger than 2: {}".format(
                    postproc.field_count))

    def set_chunk_data(self, chunk_slice, chunk_mask, Z, U, stop_reason,
            stop_iter, complex_dic, int_dic, termination_dic):
        self.chunk_slice = chunk_slice
        self.raw_data = (chunk_mask, Z, U, stop_reason, stop_iter,
            complex_dic, int_dic, termination_dic)
        self.context = dict()

    def update_context(self, chunk_slice, context_update):
        if chunk_slice != self.chunk_slice:
            raise RuntimeError("Attempt to update context"
                               " from a different chunk")
        self.context.update(context_update)

    def clear_chunk_data(self):
        """ Temporary data shared by postproc items when iterating a given
        chunk """
        self.chunk_slice = None
        self.raw_data = None
        self.context = None


class Postproc:
    """
    Abstract base class for all postprocessing objects.
    """
    field_count = 1
    def __init__(self):
        """ Filter None values : removing them from post_dic.
        """
        self.batch = None
        self._holomorphic = None

    def link_batch(self, batch):
        """
        Link to Postproc group

        :meta private:
        """
        self.batch = batch

    @property
    def fractal(self):
        return self.batch.fractal

    @property
    def calc_name(self):
        return self.batch.calc_name

    @property
    def raw_data(self):
        return self.batch.raw_data

    @property
    def context(self):
        return self.batch.context

    @property
    def holomorphic(self):
        if self._holomorphic is None:
            complex_type = self.batch.fractal.complex_type
            select = {
                np.dtype(np.float64): False,
                np.dtype(np.complex128): True
            }
            self._holomorphic = select[np.dtype(complex_type)]
        return self._holomorphic

    def ensure_context(self, key):
        """
        Check that the provided context contains the expected data

        :meta private:
        """
        try:
            return self.context[key]
        except KeyError:
            msg = ("{} should be computed before {}. "
                   "Please add the relevant item to this post-processing "
                   "batch.")
            raise RuntimeError(msg.format(key, type(self).__name__))

    def __getitem__(self, chunk_slice):
        """ Subclasses should implement
        """
        raise NotImplementedError()
        
    def get_zn(self, Z, complex_dic):
        if self.holomorphic:
            return Z[complex_dic["zn"], :]
        else:
            return Z[complex_dic["xn"], :] + 1j * Z[complex_dic["yn"], :]
        
    def get_dzndc(self, Z, complex_dic):
        if self.holomorphic:
            return Z[complex_dic["dzndc"], :]
        else:
            # Note : if there is some skew, we will need to take it into account
            # rotation ???
#            X = Z[complex_dic["xn"], :] / abs_zn
#            Y = Z[complex_dic["yn"], :] / abs_zn
            dXdA = Z[complex_dic["dxnda"], :]
            dXdB = Z[complex_dic["dxndb"], :]
            dYdA = Z[complex_dic["dynda"], :]
            dYdB = Z[complex_dic["dyndb"], :]
#            U = X * dXdA + Y * dYdA
#            V = X * dXdB + Y * dYdB
            return dXdA, dXdB, dYdA, dYdB


class Raw_pp(Postproc):
    def __init__(self, key, func=None):
        """
        A raw postproc provide direct access to the res data stored (memmap)
        during calculation. (Similar to what does `Fractal_array`  but 
        here within a `Postproc_batch`)

        Parameters
        ==========
        key : 
            The raw data key. Should be one of the *complex_codes*,
            *int_codes*, *termination_codes* for this calculation.
        func: None | callable | a str of variable x (e.g. "np.sin(x)")
              will be applied as a pre-processing step to the raw data if not
              `None`
        """
        super().__init__()
        self.key = key
        self.func = func
        if isinstance(func, str):
            self.func = fs_parser.func_parser(["x"], func)

    def __getitem__(self, chunk_slice):
        """ Returns the raw array, with self.func applied if not None """
        fractal = self.fractal
        calc_name = self.calc_name
        func = self.func
        key = self.key
        (params, codes) = fractal.reload_params(calc_name)
        (chunk_mask, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data# [chunk_slice]
        
        kind = fractal.kind_from_code(self.key, codes)
        if kind == "complex":
            if func is None:
                raise ValueError("for batch post-processing of {}, complex "
                "vals should be mapped to float, provide `func` ({}).".format(
                        type(self).__name__, self.key))
            arr = Z[complex_dic[key], :]
        elif kind == "int":
            arr = U[int_dic[key], :]
        elif key == "stop_reason":
            arr = stop_reason[0, :]
        elif key == "stop_iter":
            arr = stop_iter[0, :]
        
        if func is None:
            return arr, {}
        return func(arr), {}


class Continuous_iter_pp(Postproc):
    def __init__(self, kind=None, d=None, a_d=None, M=None, epsilon_cv=None,
                 floor_iter=0):
        """
        Return a continuous iteration number: this is the iteration count at
        bailout, plus a fractionnal part to allow smooth coloring.
        Implementation based on potential, for details see [#f3]_.

        .. [#f3] *On Smooth Fractal Coloring Techniques*,
                  **Jussi Härkönenen**, Abo University, 2007

        Parameters
        ==========
        kind :  str "infinity" | "convergent" | "transcendent"
            the classification of potential function.
        d : None | int
            degree of polynome d >= 2 if kind = "infinity"
        a_d : None | float
            coeff of higher order monome if kind = "infinity"
        M : None | float
            High number corresponding to the criteria for stopping iteration
            if kind = "infinity"
        epsilon_cv : None | float
            Small number corresponding to the criteria for stopping iteration
            if kind = "convergent"
        floor_iter : int
            If not 0, a shift will be applied and floor_iter become the 0th 
            iteration.
            Used to avoids loss of precision (banding) for very large iteration
            numbers, usually above several milions. For 
            reference, the largest integer that cannot be accurately
            represented with a float32 is > 16 M), banding will start to be
            noticeable before.

        Notes
        =====

        .. note::

            Usually it is not recommended to pass any parameter ; 
            if a parameter <param>
            is not provided, it will defaut to the attribute of 
            the fractal with name "potential_<param>", which should be the 
            recommended value.
        """
        super().__init__()
        post_dic = {
            "kind": kind,
            "d": d,
            "a_d": a_d,
            "M": M,
            "epsilon_cv": epsilon_cv
        }
        self.post_dic = {k: v for k, v in post_dic.items() if v is not None}
        self._potential_dic = None
        self._floor_iter = floor_iter

    @property
    def potential_dic(self):
        """ 
        Returns the potential parameters properties
           priority order :
               1) user input to Continuous_iter_pp constructor
               2) fractal objet defaut potential attributes
                   ("potential_" + prop)
               3) default to None if 1) and 2) fail. """
        if self._potential_dic is not None:
            return self._potential_dic

        post_dic = self.post_dic
        _potential_dic = {}
        fractal = self.fractal

        for prop in ["kind", "d", "a_d", "M", "epsilon_cv"]:
            _potential_dic[prop] =  post_dic.get(
                prop, getattr(fractal, "potential_" + prop, None))
        self._potential_dic = _potential_dic
        return _potential_dic

    def __getitem__(self, chunk_slice):
        """  Returns the real Iteration number
        """
        (chunk_mask, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data# [chunk_slice]


        n = stop_iter[0, :]
        potential_dic = self.potential_dic
        zn = self.get_zn(Z, complex_dic) #["zn"], :]

        if potential_dic["kind"] == "infinity":
            d = potential_dic["d"]
            a_d = potential_dic["a_d"]
            M = potential_dic["M"]
            k = np.abs(a_d) ** (1. / (d - 1.))
            # k normaliszation corefficient, because the formula given
            # in https://en.wikipedia.org/wiki/Julia_set                    
            # suppose the highest order monome is normalized
            nu_frac = -(np.log(np.log(np.abs(zn * k)) / np.log(M * k))
                        / np.log(d))

        elif potential_dic["kind"] == "convergent":
            eps = potential_dic["epsilon_cv"]
            alpha = 1. / Z[complex_dic["attractivity"]]
            z_star = Z[complex_dic["zr"]]
            nu_frac = + np.log(eps / np.abs(zn - z_star)  # nu frac > 0
                               ) / np.log(np.abs(alpha))

        elif potential_dic["kind"] == "transcendent":
            # Not possible to define a proper potential for a 
            # transcendental fonction
            nu_frac = 0.

        else:
            raise NotImplementedError("Potential 'kind' unsupported")

        # We need to take care of special cases to ensure that
        # -1 < nu_frac <= 0. 
        # This happen e.g. when the pixel hasn't reach the limit circle
        # at current max_iter, so its status is undefined.
        nu_div, nu_mod = np.divmod(-nu_frac, 1.)
        nu_frac = - nu_mod
        n = n - nu_div.astype(n.dtype) # need explicit casting to int

        nu = (n - self._floor_iter) + nu_frac
        val = nu

        context_update = {
            "potential_dic": self.potential_dic,
            "nu_frac": nu_frac,
            "n": n
        }

        return val, context_update
    

class Fieldlines_pp(Postproc):
    def __init__(self, n_iter=5, swirl=0., damping_ratio=0.25):
        """
        Return a continuous orbit-averaged angular value, allowing to reveal
        fieldlines
        
        Fieldlines approximate the external rays which are very important
        in the study of the Mandelbrot set and more generally in
        holomorphic dynamics. Implementation based on [#f2]_.

        .. [#f2] *On Smooth Fractal Coloring Techniques*,
                  **Jussi Härkönenen**, Abo University, 2007

        Parameters
        ==========
        n_iter :  int
            the number of orbit points used for the average
        swirl : float between -1. and 1.
            adds a random dephasing effect between each successive orbit point
        damping_ratio : float
            a geometric damping is used, damping ratio represent the ratio
            between the scaling coefficient of the last orbit point and 
            the first orbit point. 
            For no damping (arithmetic mean) use 1.

        Notes
        =====

        .. note::

            The Continuous iteration number  shall have been calculated before
            in the same `Postproc_batch`
        """
        super().__init__()
        self.n_iter = n_iter
        self.swirl = swirl
        self.damping_ratio = damping_ratio

    def __getitem__(self, chunk_slice):
        """  Returns 
        """
        nu_frac = self.ensure_context("nu_frac")
        potential_dic = self.ensure_context("potential_dic")
        d = potential_dic["d"]
        a_d = potential_dic["a_d"]

        (chunk_mask, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
            termination_dic) = self.raw_data
        zn = Z[complex_dic["zn"], :]

        if potential_dic["kind"] == "infinity":
            # C1-smooth phase angle for z -> a_d * z**d
            # G. Billotey
            k_alpha = 1. / (1. - d)       # -1.0 if N = 2
            k_beta = - d * k_alpha        # 2.0 if N = 2
            z_norm = zn * a_d**(-k_alpha) # the normalization coeff

            n_iter_fl = self.n_iter
            swirl_fl = self.swirl
            t = [np.angle(z_norm)]
            val = np.zeros_like(t)
            # Geometric serie, last term being damping_ratio * first term
            damping = self.damping_ratio ** (1. / (n_iter_fl + 1))
            di = 1.
            rg = np.random.default_rng(0)
            dphi_arr = rg.random(n_iter_fl) * swirl_fl * np.pi
            for i in range(1, n_iter_fl + 1):
                t += [d * t[i - 1]] # chained list
                dphi = dphi_arr[i-1]
                angle = 1. + np.sin(t[i-1] + dphi) + (
                      k_alpha + k_beta * d**nu_frac) * (
                      np.sin(t[i] + dphi) - np.sin(t[i-1] + dphi))
                val += di * angle
                di *= damping
            del t

        elif potential_dic["kind"] == "convergent":
            alpha = 1. / Z[complex_dic["attractivity"]]
            z_star = Z[complex_dic["zr"]]
            beta = np.angle(alpha)
            val = np.angle(z_star - zn) + nu_frac * beta
            # We have an issue if z_star == zn...
            val = val * 2.
        else:
            raise ValueError(
                "Unsupported potential '{}' for field lines".format(
                        potential_dic["kind"]))

        return val, {}

#class TIA_pp(Postproc):
#    def __init__(self, n_iter=5):
#        """ Triangular average inequality - TODO """
#        super().__init__()
#        self.n_iter = n_iter
#
#    def __getitem__(self, chunk_slice):
#        """  Returns 
#        """
#        nu_frac = self.ensure_context("nu_frac") #- 2
#        potential_dic = self.ensure_context("potential_dic")
#        d = potential_dic["d"]
#        a_d = potential_dic["a_d"]
#
#        (chunk_mask, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
#            termination_dic) = self.raw_data# (chunk_slice)
#        zn = Z[complex_dic["zn"], :]
#        c = np.ravel(self.fractal.c_chunk(chunk_slice))
#
#        if potential_dic["kind"] == "infinity":
#            val = None
#            raise NotImplementedError("TODO LIST")
#
#        elif potential_dic["kind"] == "convergent":
#            raise NotImplementedError()
#        else:
#            raise ValueError(
#                "Unsupported potential '{}' for field lines".format(
#                        potential_dic["kind"]))
#
#        return val, {}

class DEM_normal_pp(Postproc):
    field_count = 2
    def __init__(self, kind="potential"):
        
        """ 
        Return the 2 components of the distance estimation method derivative
        (*normal map*).

        This postproc is normally used in combination with a
        `fractalshades.colors.layers.Normal_map_layer`

        Parameters
        ==========
        `kind`:  str "potential" | "Milnor" | "convergent"
            if "potential" (default) DEM is base on the potential.
            For Mandelbrot power 2, "Milnor" option is also available which 
            gives similar but esthetically slightly different results.
            (Use "convergent" for convergent fractals.)

        Notes
        =====

        .. note::

            The Continuous iteration number  shall have been calculated before
            in the same `Postproc_batch`

            Alternatively, if kind="Milnor", the following raw complex fields
            must be available from the calculation:

                - "zn", "dzndc", "d2zndc2"

        References
        ==========
        .. [1] <https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set>
        """
        super().__init__()
        self.kind = kind

    @property
    def x(self):
        """ Postproc for the real part """
        return XYCoord_wrapper_pp(self, "x")
    @property
    def y(self):
        """ Postproc for the imag part """
        return XYCoord_wrapper_pp(self, "y")

    def __getitem__(self, chunk_slice):
        """  Returns the normal as a complex (x, y, 1) is the normal vec
        """
        potential_dic = self.ensure_context("potential_dic")
        (chunk_mask, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
            termination_dic) = self.raw_data # (chunk_slice)
        zn = self.get_zn(Z, complex_dic) #["zn"], :]
        
        if self.kind == "Milnor":   # use only for z -> z**2 + c
# https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
            dzndc = Z[complex_dic["dzndc"], :]
            d2zndc2 = Z[complex_dic["d2zndc2"], :]
            abs_zn = np.abs(zn)
            lo = np.log(abs_zn)
            normal = zn * dzndc * ((1+lo)*np.conj(dzndc * dzndc)
                          -lo * np.conj(zn * d2zndc2))
            # normal = normal / np.abs(normal)

        elif self.kind == "potential":
            if potential_dic["kind"] == "infinity":
                # pulls back the normal direction from an approximation of 
                # potential: phi = log(|zn|) / 2**n
                if self.holomorphic:
                    dzndc = self.get_dzndc(Z, complex_dic) #, :]
                    normal = zn / dzndc
                else:
                    (dXdA, dXdB, dYdA, dYdB) = self.get_dzndc(Z, complex_dic)
                    # J = [dXdA dXdB]    J.T = [dXdA dYdA]   n = J.T * zn
                    #     [dYdA dYdB]          [dXdB dYdB]
                    normal = (
                        (dXdA * zn.real + dYdA * zn.imag)
                        + 1j * (dXdB * zn.real + dYdB * zn.imag)
                    )
                #normal = normal / np.abs(normal)

            elif potential_dic["kind"] == "convergent":
            # pulls back the normal direction from an approximation of
            # potential (/!\ need to derivate zr w.r.t. c...)
            # phi = 1. / (abs(zn - zr) * abs(alpha))
                dzndc = Z[complex_dic["dzndc"], :]
                zr = Z[complex_dic["zr"], :]
                dzrdc = Z[complex_dic["dzrdc"], :]
                normal = - (zr - zn) / (dzrdc - dzndc)
                #normal = normal / np.abs(normal)

        skew = self.fractal.skew
        if skew is not None:
            nx = normal.real
            ny = normal.imag
            # These are contravariant coordinates -> we unskew
            fs.core.apply_unskew_1d(skew, nx, ny)

        normal = normal / np.abs(normal)

        return normal, None


class XYCoord_wrapper_pp(Postproc):
    def __init__(self, pp_2d, coord):
        """ Helper class to split one complex Postproc into 2 real & imag parts
        """
        self.pp_2d = pp_2d
        self.coord = coord

    @property
    def context_getitem_key(self):
        return type(self.pp_2d).__name__ + ".__getitem__"

    def __getitem__(self, chunk_slice):
        if self.coord == "x":
            normal, _ = self.pp_2d[chunk_slice]
            # Save imaginary part in context dict
            context_update = {self.context_getitem_key: normal.imag}
            return normal.real, context_update
        elif self.coord == "y":
            # Retrieve imaginary part from context, 
            val = self.context[self.context_getitem_key]
            # Then delete it
            del self.context[self.context_getitem_key]
            return val, {}
        else:
            raise ValueError(self.coord)

    def link_sibling(self, sibling):  # Needed ??
        """ The other coord """
        self.sibling = sibling


class DEM_pp(Postproc):
    field_count = 1
    def __init__(self, px_snap=None):
        """ 
        Return a distance estimation of the pixel to the fractal set.
        
        Parameters
        ==========
        `px_snap`:  None | float
            if not None, pixels at a distance from the fractal lower
            than `px_snap` (expressed in image pixel) will be snapped to 0.

        Notes
        =====
    
        .. note::

            The Continuous iteration number  shall have been calculated before
            in the same `Postproc_batch`
        """
        self.px_snap = px_snap
        super().__init__()

    def __getitem__(self, chunk_slice):
        """  Returns the DEM - """
        potential_dic = self.ensure_context("potential_dic")
        (chunk_mask, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
            termination_dic) = self.raw_data

        zn = self.get_zn(Z, complex_dic) #Z[complex_dic["zn"], :]

        if potential_dic["kind"] == "infinity":
            abs_zn = np.abs(zn)
#            print("--> pp, DEM, inf")
            if self.holomorphic:
                abs_dzndc = np.abs(self.get_dzndc(Z, complex_dic)) #, :]
                val = abs_zn * np.log(abs_zn) / abs_dzndc
            else:
                (dXdA, dXdB, dYdA, dYdB) = self.get_dzndc(Z, complex_dic)
                # In which direction ? Lets take the mean
#                abs_dzndc = np.sqrt(
#                    dXdA ** 2 + dXdB ** 2 + dYdA ** 2 + dYdB ** 2
#                )
                # Lets take the maximal singular value
# https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
                # J = [dXdA dXdB] = [a b]
                #     [dYdA dYdB]   [c d]
                Q = np.hypot(dXdA + dYdB, dXdB - dYdA)
                R = np.hypot(dXdA - dYdB, dXdB + dYdA)
                abs_dzndc = 0.5 * (Q + R)
#                S1 = dXdA ** 2 + dXdB ** 2 + dYdA ** 2 + dYdB ** 2
#                S2 = np.sqrt(
#                    dXdA ** 2 + dXdB ** 2 + dYdA ** 2 + dYdB ** 2
#                    + 4 * (dXdA * + dYdB)
#                )
#                normal = (
#                    (dXdA * zn.real + dYdA * zn.imag)
#                    + 1j * (dXdB * zn.real + dYdB * zn.imag)
#                )
#                abs_dzndc = np.abs(normal)
#                normal = normal / np.abs(normal)
#                # abs_dzndc = dXdA
#                u = (
#                    (dXdA * normal.real + dXdB * normal.imag)
#                    + 1j * (dYdA * normal.real + dYdB * normal.imag)
#                )
            val = abs_zn * np.log(abs_zn) / abs_dzndc

        elif potential_dic["kind"] == "convergent":
            assert self.holomorphic
            dzndc = self.get_dzndc(Z, complex_dic)
            zr = Z[complex_dic["zr"]]
            dzrdc = Z[complex_dic["dzrdc"], :]
            val = np.abs((zr - zn) / (dzrdc - dzndc))

        else:
            raise ValueError(potential_dic["kind"])
            
#        print("val before\n", val)
#        px = self.fractal.dx / float(self.fractal.nx)
#        val = val / px
#        print("dx nx ps", self.fractal.dx, float(self.fractal.nx), px)
#        print("val after\n", val)
#        raise ValueError()

        px_snap = self.px_snap
        if px_snap is not None:
            val = np.where(val < px_snap, 0., val)

        return val, {}


class Attr_normal_pp(Postproc):
    field_count = 2
    def __init__(self):
        """ 
        Return the 2 components of the cycle attractivity derivative (*normal
        map*).

        This postproc is normally used in combination with a
        `fractalshades.colors.layers.Normal_map_layer`

        Notes
        =====
    
        .. note::

            The following complex fields must be available from a previously
            run calculation:

                - "attractivity"
                - "order" (optionnal)
        """
        super().__init__()

    @property
    def x(self):
        """ Postproc for the real part """
        return XYCoord_wrapper_pp(self, "x")
    @property
    def y(self):
        """ Postproc for the imag part """
        return XYCoord_wrapper_pp(self, "y")

    def __getitem__(self, chunk_slice):
        """  Returns the normal as a complex (x, y, 1) is the normal vec
        """
        (chunk_mask, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
            termination_dic) = self.raw_data
        attr = np.copy(Z[complex_dic["attractivity"], :])
        dattrdc = np.copy(Z[complex_dic["dattrdc"], :])

        invalid = (np.abs(attr) > 1.)
        np.putmask(attr, invalid, 1.)
        np.putmask(dattrdc, invalid, 1.)

        # val = np.sqrt(1. - attr * np.conj(attr)) / r
        # Now let's take the total differential of this
        # While not totally exact this gives good results :
        normal = attr * np.conj(dattrdc) / np.abs(dattrdc)
        
        return normal, None


class Attr_pp(Postproc):
    field_count = 1
    def __init__(self, scale_by_order=False):
        """ 
        Return the cycle attractivity
        
        Parameters
        ==========
        scale_by_order : bool
            If True return the attractivity divided by ccyle order (this
            allows more realistic 3d-view exports)

        Notes
        =====
    
        .. note::

            The following complex fields must be available from a previously
            run calculation:

                - "attractivity"
                - "order" (optionnal)
        """
        super().__init__()
        self.scale_by_order = scale_by_order

    def __getitem__(self, chunk_slice):
        """  Returns the normal as a complex (x, y, 1) is the normal vec
        """
        (chunk_mask, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
            termination_dic) = self.raw_data
        # Plotting the 'domed' height map for the cycle attractivity
        attr = Z[complex_dic["attractivity"], :]
        abs2_attr = np.real(attr)**2 + np.imag(attr)**2
        abs2_attr = np.clip(abs2_attr, None, 1.)
        # divide by the cycle order
        val = np.sqrt(1. - abs2_attr) # / r
        if self.scale_by_order:
            r = U[int_dic["order"], :]
            val *= (1. / r)
        return val, {}

class Fractal_array:
    def __init__(self, fractal, calc_name, key, func=None):
        """
        Class which provide a direct access to the res data stored (memory
        mapping)  during calculation.

        One application, it can be passed as a *subset* parameter to most
        calculation models, e.g. : `fractalshades.models.Mandelbrot.base_calc`

        Parameters
        ==========
        fractal : `fractalshades.Fractal` derived class
            The reference fracal object
        calc_name : str 
            The calculation name
        key : 
            The raw data key. Should be one of the *complex_codes*,
            *int_codes*, *termination_codes* for this calculation.
        func: None | callable | a str of variable x (e.g. "np.sin(x)")
              will be applied as a pre-processing step to the raw data if not
              `None`
        """
        self.fractal = fractal
        self.calc_name = calc_name
        self.key = key
        self.func = self.parse_func(func)
        # Manage ~ notation
        self.inv = False

    @staticmethod
    def parse_func(func):
        if isinstance(func, str):
            return fs_parser.func_parser(["x"], func)
        else:
            return func

    def __getitem__(self, chunk_slice):
        """ Returns the raw array, with self.func applied if not None """
        fractal = self.fractal
        calc_name = self.calc_name
        (params, codes) = fractal.reload_params(calc_name)
        kind = fractal.kind_from_code(self.key, codes)

        # localise the pts range for the chunk /!\ use the right calc_name
        if chunk_slice is not None:
            items = fractal.REPORT_ITEMS
            report_mmap = open_memmap(filename=fractal.report_path(calc_name),
                                      mode='r')
            rank = fractal.chunk_rank(chunk_slice)
            beg = report_mmap[rank, items.index("chunk1d_begin")]
            end = report_mmap[rank, items.index("chunk1d_end")]

        # load the data
        arr_from_kind = {"complex": "Z",
                         "int": "U",
                         "stop_reason": "stop_reason",
                         "stop_iter": "stop_iter"}
        arr_str = arr_from_kind[kind]
        data_path = fractal.data_path(calc_name)
        mmap = open_memmap(filename=data_path[arr_str], mode='r')
        
        # field from key :
        key = self.key
        complex_dic, int_dic, termination_dic = fractal.codes_mapping(*codes)
        if kind == "complex":
            field = complex_dic[key]
        elif kind == "int":
            field = int_dic[key]
        elif kind in ["stop_reason", "stop_iter"]:
            field = 0
        else:
            raise ValueError(kind)

        if chunk_slice is None:
            # We return the full pts range
            arr = mmap[field, :]
        else:
            arr = mmap[field, beg:end]

        # Apply last steps (func / inversion) and returns
        if self.func is not None:
            arr = self.func(arr)
        if self.inv:
            arr = ~arr
        return arr

    def __invert__(self):
        """ Allows use of the ~ notation """
        ret = Fractal_array(self.fractal, self.calc_name, self.key, self.func)
        ret.inv = True
        return ret
