# -*- coding: utf-8 -*-
import pickle
import inspect
import logging

import numpy as np
import numba
from numpy.lib.format import open_memmap

import fractalshades as fs
#import fractalshades.numpy_utils.xrange as fsx
import fractalshades.numpy_utils.expr_parser as fs_parser

logger = logging.getLogger(__name__)

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

        # Temporary data used when iterating chunk_slices
        self.raw_data = dict()
        self.context = dict()
        # self.clear_chunk_data()

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
            raise ValueError(
                f"postproc.field_count bigger than 2: {postproc.field_count}"
            )


    def set_chunk_data(self, chunk_slice, chunk_mask, c_pt, Z, U, stop_reason,
            stop_iter, complex_dic, int_dic, termination_dic):
        # self.chunk_slice = chunk_slice
        self.raw_data[chunk_slice] = (
            chunk_mask, c_pt, Z, U, stop_reason, stop_iter,
            complex_dic, int_dic, termination_dic
        )
        self.context[chunk_slice] = dict()


    def update_context(self, chunk_slice, context_update):
        self.context[chunk_slice].update(context_update)

    def clear_chunk_data(self, chunk_slice):
        """ Temporary data shared by postproc items when iterating a given
        chunk """
        del self.raw_data[chunk_slice]
        del self.context[chunk_slice]


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
    def state(self):
        # The fractal state object
        return self.fractal._calc_data[self.calc_name]["state"]

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

    def ensure_context(self, chunk_slice, key):
        """
        Check that the provided context contains the expected data
        :meta private:
        """
        try:
            return self.context[chunk_slice][key]
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
            dXdA = Z[complex_dic["dxnda"], :]
            dXdB = Z[complex_dic["dxndb"], :]
            dYdA = Z[complex_dic["dynda"], :]
            dYdB = Z[complex_dic["dyndb"], :]
            return dXdA, dXdB, dYdA, dYdB


class Raw_pp(Postproc):
    def __init__(self, key, func=None):
        """A raw postproc provide direct access to the res data stored (memmap)
        during calculation. (Similar to what does `Fractal_array`  but 
        here within a `Postproc_batch`)

        Parameters
        ==========
        key: str
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
        # (params, codes) = fractal.reload_params(calc_name)
        codes = fractal._calc_data[calc_name]["saved_codes"]
        (chunk_mask, c_pt, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data[chunk_slice]
        
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
        """Return a continuous iteration number: this is the iteration count at
        bailout, plus a fractionnal part to allow smooth coloring.
        Implementation based on potential, for details see:

            - *On Smooth Fractal Coloring Techniques*,
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
               2) fractal (or fractal 'state') defaut potential attributes
                   ("potential_" + prop)
               3) default to None if 1) and 2) fail. """
        if self._potential_dic is not None:
            return self._potential_dic

        post_dic = self.post_dic
        _potential_dic = {}


        fractal = self.fractal # The factal
        for prop in ["kind", "d", "a_d", "M_cutoff"]:
            _potential_dic[prop] =  post_dic.get(
                prop, getattr(fractal, "potential_" + prop, None)
        )

        state = self.state # The fractal state
        for prop in ["M", "epsilon_cv"]:
            _potential_dic[prop] =  post_dic.get(
                prop, getattr(state, "potential_" + prop, None)
        )

        self._potential_dic = _potential_dic
        return _potential_dic

    def __getitem__(self, chunk_slice):
        """  Returns the real Iteration number
        """
        (chunk_mask, c_pt, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data[chunk_slice]

        n = stop_iter[0, :]
        potential_dic = self.potential_dic
        
        zn = self.get_zn(Z, complex_dic) #["zn"], :]

        if potential_dic["kind"] == "infinity":
            d = potential_dic["d"]
            a_d = potential_dic["a_d"]
            M = potential_dic["M"]

            nu_frac = Continuous_iter_pp_infinity(zn, d, a_d, M)

        elif potential_dic["kind"] == "convergent":
            eps = potential_dic["epsilon_cv"]
            attractivity = Z[complex_dic["attractivity"]]
            z_star = Z[complex_dic["zr"]]
            nu_frac = Continuous_iter_pp_convergent(
                zn, eps, attractivity, z_star
            )

        elif potential_dic["kind"] == "transcendent":
            # Not possible to define a proper potential for a 
            # transcendental fonction
            nu_frac = 0.

        else:
            raise NotImplementedError(
                f"Potential kind: {potential_dic['kind']} unsupported"
        )

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
            "n": n, # + jump,
            "backshift": getattr(self.state, "backshift", None)
        }

        return val, context_update


class Fieldlines_pp(Postproc):
    def __init__(self, n_iter=5, swirl=0., endpoint_k=1.0):
        """
        Return a continuous orbit-averaged angular value, allowing to reveal
        fieldlines. Implementation based on [#f2]_. The averaging orbit starts
        at the computed `zn_orbit` field if it exists, otherwise at the exit 
        `zn` value, and is truncated after `n_iter` iterations.

        Fieldlines approximate the external rays which are very important
        in the study of the Mandelbrot set and more generally in
        holomorphic dynamics. 

        .. [#f2] *On Smooth Fractal Coloring Techniques*,
          **Jussi Härkönenen**, Abo University, 2007

        Parameters
        ==========
        n_iter :  int > 0
            the number of orbit points used for the averageing
        swirl : float between -1. and 1.
            adds a random dephasing effect between each successive orbit point
        endpoint_k : float > 0.
            A geometric serie is used for scaling individual orbit points,
            this parameters is the ratio between the last and the first scaling
            (For an arithmetic mean over the truncated orbit, use 1.)
        Notes
        =====
        .. note::
            The Continuous iteration number  shall have been calculated before
            in the same `Postproc_batch`
        """
        super().__init__()
        self.n_iter = n_iter
        self.swirl = swirl
        self.endpoint_k = endpoint_k

    def __getitem__(self, chunk_slice):
        """  Returns the post arr
        """
        nu_frac = self.ensure_context(chunk_slice, "nu_frac")
        potential_dic = self.ensure_context(chunk_slice, "potential_dic")

        (chunk_mask, c_pt, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data[chunk_slice]

        if potential_dic["kind"] == "infinity":
            # Following Jussi Härkönenen - 2007
            backshift = self.ensure_context(chunk_slice, "backshift")
            n = self.ensure_context(chunk_slice, "n")
            n_iter_fl = self.n_iter
            swirl_fl = self.swirl
            # Geometric serie, last term being damping_ratio * first term
            k_arr = np.geomspace(1., self.endpoint_k, num=n_iter_fl) 
            k_arr = k_arr / np.sum(k_arr)
    
            # randomizing the phse angle
            rg = np.random.default_rng(0)
            phi_arr = rg.random(n_iter_fl) * swirl_fl * np.pi

            if self.fractal.holomorphic:
                try:
                    zn = Z[complex_dic["zn_orbit"], :]
                    is_backward = True
                    assert backshift is not None
                except KeyError:
                    logger.warning(
                        "No stored backshift field found for key: \"zn_orbit\""
                        "\ndefaulting to key: \"zn\""
                    )
                    zn = Z[complex_dic["zn"], :]
                    is_backward = False
                    backshift = 0 # Need the right datatype for jit compiler
    
                zn_iterate = self.fractal.zn_iterate
    
                val = Fieldlines_pp_infinity(
                    c_pt, zn, nu_frac, k_arr, phi_arr, zn_iterate,
                    is_backward, backshift, n
                )
            else:
                # Non-holomorphic case
                try:
                    xn = Z[complex_dic["xn_orbit"], :]
                    yn = Z[complex_dic["yn_orbit"], :]
                    is_backward = True
                    assert backshift is not None
                except KeyError:
                    logger.warning(
                        "No stored backshift field found for key: \"zn_orbit\""
                        "\ndefaulting to key: \"zn\""
                    )
                    xn = Z[complex_dic["xn"], :]
                    yn = Z[complex_dic["yn"], :]
                    is_backward = False
                    backshift = 0 # Need the right datatype for jit compiler
                    
                xnyn_iterate = self.fractal.xnyn_iterate
                
                val = Fieldlines_pp_infinity_BS(
                    c_pt, xn, yn, nu_frac, k_arr, phi_arr, xnyn_iterate,
                    is_backward, backshift, n
                )


        elif potential_dic["kind"] == "convergent":
            zn = Z[complex_dic["zn"], :]
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
            gives slightly different results (sharper edges).
            Use "convergent" for convergent fractals...

        Notes
        =====
        .. note::
            The Continuous iteration number shall have been calculated before
            in the same `Postproc_batch`
            If kind="Milnor", the following raw complex fields
            must be available from the calculation:
            "zn", "dzndc", "d2zndc2"
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
        potential_dic = self.ensure_context(chunk_slice, "potential_dic")
        (chunk_mask, c_pt, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data[chunk_slice]
        zn = self.get_zn(Z, complex_dic) #["zn"], :]

        if self.kind == "Milnor":   # use only for z -> z**n + c
# https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
            dzndc = Z[complex_dic["dzndc"], :]
            d2zndc2 = Z[complex_dic["d2zndc2"], :]
            abs_zn = np.abs(zn)
            lo = np.log(abs_zn)
            normal = zn * dzndc * ((1+lo)*np.conj(dzndc * dzndc)
                          -lo * np.conj(zn * d2zndc2))

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

            elif potential_dic["kind"] == "convergent":
            # pulls back the normal direction from an approximation of
            # potential (/!\ need to derivate zr w.r.t. c...)
            # phi = 1. / (abs(zn - zr) * abs(alpha))
                dzndc = Z[complex_dic["dzndc"], :]
                zr = Z[complex_dic["zr"], :]
                dzrdc = Z[complex_dic["dzrdc"], :]
                normal = - (zr - zn) / (dzrdc - dzndc)

        skew = self.fractal.skew
        if skew is not None:
            nx = normal.real
            ny = normal.imag
            # These are contravariant coordinates -> we UNskew
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
            val = self.context[chunk_slice][self.context_getitem_key]
            # Then delete it
            del self.context[chunk_slice][self.context_getitem_key]
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
        potential_dic = self.ensure_context(chunk_slice,"potential_dic")
        (chunk_mask, c_pt, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data[chunk_slice]

        zn = self.get_zn(Z, complex_dic) #Z[complex_dic["zn"], :]

        if potential_dic["kind"] == "infinity":
            abs_zn = np.abs(zn)

            if self.holomorphic:
                abs_dzndc = np.abs(self.get_dzndc(Z, complex_dic)) #, :]
                # val = abs_zn * np.log(abs_zn) / abs_dzndc
            else:
                (dXdA, dXdB, dYdA, dYdB) = self.get_dzndc(Z, complex_dic)
                # In which direction for an abs value ? 
                # Lets take the maximal singular value
# https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
                # J = [dXdA dXdB] = [a b]
                #     [dYdA dYdB]   [c d]
                Q = np.hypot(dXdA + dYdB, dXdB - dYdA)
                R = np.hypot(dXdA - dYdB, dXdB + dYdA)
                abs_dzndc = 0.5 * (Q + R)

            val = abs_zn * np.log(abs_zn) / abs_dzndc

        elif potential_dic["kind"] == "convergent":
            assert self.holomorphic
            dzndc = self.get_dzndc(Z, complex_dic)
            zr = Z[complex_dic["zr"]]
            dzrdc = Z[complex_dic["dzrdc"], :]
            val = np.abs((zr - zn) / (dzrdc - dzndc))

        else:
            raise ValueError(potential_dic["kind"])

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
        (chunk_mask, c_pt, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data[chunk_slice]
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
        (chunk_mask, c_pt, Z, U, stop_reason, stop_iter, complex_dic, int_dic,
         termination_dic) = self.raw_data[chunk_slice]
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
    def __init__(self, fractal, calc_name, key, func=None, inv=False):
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

        try:
            pickle.dumps(func)
        except:
            # Note: possible improvement, see:
            # https://stackoverflow.com/questions/11878300/serializing-and-deserializing-lambdas
            # Seems over-the top here, just raising a detailed error
            source_code = inspect.getsource(func)
            raise ValueError(
                "func is unserializable:\n"
                + f"{source_code}\n"
                + "Consider passing func definition by string instead"
                + " (e.g. \"np.sin(x)\")"
            )
        self._func = func # stored for future serialization
        self.func = self.parse_func(func)

        # Manage ~ notation
        self.inv = inv

    @staticmethod
    def parse_func(func):
        if isinstance(func, str):
            return fs_parser.func_parser(["x"], func)
        else:
            return func

    def __reduce__(self):
        """ Enable standard Python serialization mechanism
        """
        args = (self.fractal, self.calc_name, self.key, self._func, self.inv)
        return (self.__class__, args, None, None, None, None)

    def __getitem__(self, chunk_slice):
        """ Returns the raw array, with self.func applied if not None """
        fractal = self.fractal

        if fractal is None:
            raise ValueError(
                "Fractal_array without valid Fractal, "
                "might be an unbound state from unppickling. Use "
                "self.bind_fractal to rebind"
            )

        f_class = fractal.__class__

        calc_name = self.calc_name
        codes = fractal._calc_data[calc_name]["saved_codes"]
        kind = f_class.kind_from_code(self.key, codes)

        # localise the pts range for the chunk /!\ use the right calc_name
        if chunk_slice is not None:
            items = fractal.REPORT_ITEMS
            report_mmap = open_memmap(filename=fractal.report_path(calc_name),
                                      mode='r')
            rank = fractal.chunk_rank(chunk_slice)
            beg = report_mmap[rank, items.index("chunk1d_begin")]
            end = report_mmap[rank, items.index("chunk1d_end")]

        # load the data
        arr_from_kind = {
            "complex": "Z",
             "int": "U",
             "stop_reason": "stop_reason",
             "stop_iter": "stop_iter"
        }
        arr_str = arr_from_kind[kind]
        data_path = fractal.data_path(calc_name)
        mmap = open_memmap(filename=data_path[arr_str], mode='r')

        # field from key :
        key = self.key
        (complex_dic, int_dic, termination_dic) = f_class.codes_mapping(*codes)
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
        """ Allows use of the ~ notation 'ala numpy' """
        ret = Fractal_array(
            self.fractal, self.calc_name, self.key, self.func, not(self.inv)
        )
        return ret

    def __eq__(self, other):
        """ Equality testing, useful when pickling / unpickling"""
        if isinstance(other, self.__class__):
            eq = (
                (self._func == other._func)
                and (self.calc_name == other.calc_name)
                and (self.key == other.key)
                and (self.inv == other.inv)
            )
            return eq
        else:
            return NotImplemented


#==============================================================================
# Numba Nogil implementations
        
@numba.njit(nogil=True, fastmath=True)
def Continuous_iter_pp_infinity(zn, d, a_d, M):
    k = np.abs(a_d) ** (1. / (d - 1.))
    # k normaliszation corefficient, because the formula given
    # in https://en.wikipedia.org/wiki/Julia_set                    
    # suppose the highest order monome is normalized
    nu_frac = -(np.log(np.log(np.abs(zn * k)) / np.log(M * k))
                / np.log(d))
    return nu_frac

@numba.njit(nogil=True, fastmath=True)
def Continuous_iter_pp_convergent(zn, eps, attractivity, z_star):
    alpha = 1. / attractivity
    nu_frac = + np.log(eps / np.abs(zn - z_star)  # nu frac > 0
                               ) / np.log(np.abs(alpha))
    return nu_frac


# Catmull-Rom polynomials - C1 Smoothing of an orbit value
@numba.njit(nogil=True, fastmath=True)
def cm_H0(x):
     return 0.5 * x * (-x + x ** 2)

@numba.njit(nogil=True, fastmath=True)
def cm_H1(x):
     return 0.5 * x * (1. + 4.* x - 3. * x ** 2)

@numba.njit(nogil=True, fastmath=True)
def cm_H2(x):
     return 1. + 0.5 * x * (-5. * x + 3. * x ** 2)

@numba.njit(nogil=True, fastmath=True)
def cm_H3(x):
     return 0.5 * x * (-1. + 2. * x - x ** 2)


@numba.njit(nogil=True, fastmath=True)
def Fieldlines_pp_infinity(
    c_pt, zn, nu_frac, k_arr, phi_arr, zn_iterate,
    is_backward, backshift, n
):
    M_cutoff = 100000.  # To be tuned...
    n_iter_fl = len(k_arr)
    assert len(phi_arr) == n_iter_fl

    nvec, = zn.shape
    orbit_z = np.empty((nvec, n_iter_fl + 3), dtype=zn.dtype)
    orbit_angle = np.empty((nvec, n_iter_fl + 3), dtype=np.float64) #zn.dtype)
    val = np.zeros((nvec,), dtype=np.float64)
    
    for j in range(nvec):
        z_loc = zn[j]
        orbit_z[j, 0] = z_loc
        orbit_angle[j, 0] = np.angle(z_loc)
    
    for i in range(n_iter_fl + 2):
        for j in range(nvec):
            z_loc = orbit_z[j, i]
            if abs(z_loc) > M_cutoff:
                z_loc = np.exp(orbit_angle[j, i] * 1j) * M_cutoff
                z_loc = zn_iterate(z_loc, complex(0.))
                orbit_z[j, i + 1] = z_loc # flag
                orbit_angle[j, i + 1] = np.angle(z_loc)
            else:
                z_loc = zn_iterate(orbit_z[j, i], c_pt[j])
                orbit_z[j, i + 1] = z_loc
                orbit_angle[j, i + 1] = np.angle(z_loc)
    
    # Now lets do some smoothing with Catmull-Rom polynomials
    for j in range(nvec):
        d = -nu_frac[j]
        # Correction if applying backshift would yield a negative index
        # in this case we just keep a constant nu_frac to avoid banding
        if is_backward:
            if backshift > n[j]:
                d = 1.
        # Catmull-Rom spline weighting polynomials.
        a0 = cm_H0(d)
        a1 = cm_H1(d)
        a2 = cm_H2(d)
        a3 = cm_H3(d)
        for i in range(n_iter_fl):
            val_loc = k_arr[i] * (
                a0 * np.sin(orbit_angle[j, i] + phi_arr[i])
                + a1 * np.sin(orbit_angle[j, i + 1] + phi_arr[i])
                + a2 * np.sin(orbit_angle[j, i + 2] + phi_arr[i])
                + a3 * np.sin(orbit_angle[j, i + 3] + phi_arr[i])
            )
            val[j] += val_loc

    return val

#def Fieldlines_pp_infinity_BS(
#                    c_pt, xn, yn, nu_frac, k_arr, phi_arr, xnyn_iterate,
#                    is_backward, backshift, n
#                )

@numba.njit(nogil=True, fastmath=True)
def Fieldlines_pp_infinity_BS(
    c_pt, xn, yn, nu_frac, k_arr, phi_arr, xnyn_iterate,
    is_backward, backshift, n
):
    M_cutoff = 100000.  # To be tuned...
    n_iter_fl = len(k_arr)
    assert len(phi_arr) == n_iter_fl

    nvec, = xn.shape
    orbit_x = np.empty((nvec, n_iter_fl + 3), dtype=xn.dtype)
    orbit_y = np.empty((nvec, n_iter_fl + 3), dtype=yn.dtype)
    orbit_angle = np.empty((nvec, n_iter_fl + 3), dtype=np.float64)
    val = np.zeros((nvec,), dtype=np.float64)

    for j in range(nvec):
        x_loc = xn[j]
        y_loc = yn[j]
        orbit_x[j, 0] = x_loc
        orbit_y[j, 0] = y_loc
        orbit_angle[j, 0] = np.arctan2(y_loc, x_loc)
    
    for i in range(n_iter_fl + 2):
        for j in range(nvec):
            aj = c_pt[j].real
            bj = c_pt[j].imag
            x_loc = orbit_x[j, i]
            y_loc = orbit_y[j, i]
            if np.hypot(x_loc, y_loc) > M_cutoff:
                
                x_loc = np.cos(orbit_angle[j, i]) * M_cutoff
                y_loc = np.sin(orbit_angle[j, i]) * M_cutoff
                x_loc, y_loc = xnyn_iterate(x_loc, y_loc, 0., 0.)

                orbit_x[j, i + 1] = x_loc # flag
                orbit_y[j, i + 1] = y_loc # flag
                orbit_angle[j, i + 1] = np.arctan2(y_loc, x_loc)
            else:
                x_loc, y_loc = xnyn_iterate(
                        orbit_x[j, i], orbit_y[j, i], aj, bj
                )
                orbit_x[j, i + 1] = x_loc
                orbit_y[j, i + 1] = y_loc
                orbit_angle[j, i + 1] = np.arctan2(y_loc, x_loc)
    
    # Now lets do some smoothing with Catmull-Rom polynomials
    for j in range(nvec):
        d = -nu_frac[j]
        # Correction if applying backshift would yield a negative index
        # in this case we just keep a constant nu_frac to avoid banding
        if is_backward:
            if backshift > n[j]:
                d = 1.
        # Catmull-Rom spline weighting polynomials.
        a0 = cm_H0(d)
        a1 = cm_H1(d)
        a2 = cm_H2(d)
        a3 = cm_H3(d)
        for i in range(n_iter_fl):
            val_loc = k_arr[i] * (
                a0 * np.sin(orbit_angle[j, i] + phi_arr[i])
                + a1 * np.sin(orbit_angle[j, i + 1] + phi_arr[i])
                + a2 * np.sin(orbit_angle[j, i + 2] + phi_arr[i])
                + a3 * np.sin(orbit_angle[j, i + 3] + phi_arr[i])
            )
            val[j] += val_loc

    return val