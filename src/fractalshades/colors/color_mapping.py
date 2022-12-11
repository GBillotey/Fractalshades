# -*- coding: utf-8 -*-
import os
import typing
import enum
import pprint
import pickle

import numpy as np
#import matplotlib.colors
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_agg import FigureCanvasAgg
import numba
import PIL
import PIL.ImageQt

import fractalshades.numpy_utils as np_utils



class Color_tools():
    """ A bunch of staticmethods
    Color conversions from http://www.easyrgb.com/en/math.php#text7
    All take as input and return arrays of size [n, 3]"""
    # CIE LAB constants Illuminant= D65
    # Simulates noon daylight with correlated color temperature of 6504 K. 
    Lab_ref_white = np.array([0.95047, 1., 1.08883])
    # CIE LAB constants Illuminant= D55
    # Simulates mid-morning or mid-afternoon daylight with correlated color
    # temperature of 5500 K. 
    D55_ref_white = np.array([0.9568, 1., 0.9214])
    # CIE LAB constants Illuminant= D50
    # Simulates warm daylight at sunrise or sunset with correlated color
    # temperature of 5003 K. Also known as horizon light.
    D50_ref_white = np.array([0.964212, 1., .825188])
    # CIE standard illuminant A, . Simulates typical, domestic,
    # tungsten-filament lighting with correlated color temperature of 2856 K. 
    A_ref_white = np.array([1.0985, 1.0000, 0.3558])

    @staticmethod
    def rgb_to_XYZ(rgb):
        arr = np.swapaxes(rgb, 0, 1)
        arr = np.where(arr > 0.04045, ((arr + 0.055) / 1.055) ** 2.4,
                       arr / 12.92)
        matrix = np.array([[0.4124, 0.3576, 0.1805],
                           [0.2126, 0.7152, 0.0722],
                           [0.0193, 0.1192, 0.9505]])
        return np.swapaxes(np.dot(matrix, arr), 0, 1)
    @staticmethod
    def XYZ_to_rgb(XYZ):
        arr = np.swapaxes(XYZ, 0, 1)
        matrix_inv = np.array([[ 3.24062548, -1.53720797, -0.4986286 ],
                               [-0.96893071,  1.87575606,  0.04151752],
                               [ 0.05571012, -0.20402105,  1.05699594]])
        arr = np.dot(matrix_inv, arr)
        arr[arr < 0.] = 0.
        arr = np.where(arr > 0.0031308,
                       1.055 * np.power(arr, 1. / 2.4) - 0.055, arr * 12.92)
        arr[arr > 1.] = 1.
        return np.swapaxes(arr, 0, 1)

    @staticmethod    
    def XYZ_to_CIELab(XYZ, ref_white=Lab_ref_white):
        arr = XYZ / ref_white
        arr = np.where(arr > 0.008856, arr ** (1. / 3.),
                       (7.787 * arr) + 16. / 116.)
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        L, a , b = (116. * y) - 16. , 500.0 * (x - y) , 200.0 * (y - z)
        return np.swapaxes(np.vstack([L, a, b]), 0, 1)
    @staticmethod 
    def CIELab_to_XYZ(Lab, ref_white=Lab_ref_white):
        L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
        y = (L + 16.) / 116.
        x = (a / 500.) + y
        z = y - (b / 200.)
        arr = np.vstack([x, y, z]) 
        arr = np.where(arr > 0.2068966, arr ** 3.,
                       (arr - 16. / 116.) / 7.787)
        return np.swapaxes(arr, 0, 1) * ref_white

    @staticmethod 
    def CIELab_to_CIELch(Lab):
        L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
        h = np.arctan2(b, a)
        h = np.where(h > 0, h / np.pi * 180.,
                     360. + h / np.pi * 180.)
        c = np.sqrt(a**2 + b**2)
        arr = np.vstack([L, c, h])
        return np.swapaxes(arr, 0, 1)
    @staticmethod 
    def CIELch_to_CIELab(Lch):
        L, c, h = Lch[:, 0], Lch[:, 1], Lch[:, 2]
        h_rad = h * np.pi / 180.
        a = c * np.cos(h_rad)
        b = c * np.sin(h_rad)
        arr = np.vstack([L, a, b])
        return np.swapaxes(arr, 0, 1)

    @staticmethod 
    def rgb_to_CIELab(rgb):
        return Color_tools.XYZ_to_CIELab(Color_tools.rgb_to_XYZ(rgb))
    @staticmethod 
    def rgb_to_CIELch(rgb):
        return Color_tools.CIELab_to_CIELch(
            Color_tools.XYZ_to_CIELab(Color_tools.rgb_to_XYZ(rgb)))
    @staticmethod
    def CIELab_to_rgb(Lab):
        return Color_tools.XYZ_to_rgb(Color_tools.CIELab_to_XYZ(Lab))
    @staticmethod
    def CIELch_to_rgb(Lch):
        return Color_tools.XYZ_to_rgb(
            Color_tools.CIELab_to_XYZ(Color_tools.CIELch_to_CIELab(Lch)))

    @staticmethod
    def Lab_gradient(color_1, color_2, n, f=lambda t: t):
        """
        Return a color gradient between color_1 and color_2, default linear in 
        Lab
        *color1* and *color_2* in rgb coordinate
        *n* integer number of color in gradient
        *f* function default to linear, general case c1 + f * (c2 - c1)
        """
        return Color_tools.customized_Lab_gradient(color_1, color_2, n, f, f)

    @staticmethod
    def desaturated(color):
        """
        Return the grey tone of identical Luminance
        """
        L = Color_tools.rgb_to_CIELab(color[np.newaxis, :])[0, 0]
        return Color_tools.CIELab_to_rgb(np.array([[L, 0., 0.]]))[0, :]

    @staticmethod
    def Lch_gradient(color_1, color_2, n, long_path=False, f=lambda t: t):
        """
        Return a color gradient between color_1 and color_2, default linear in
        Lch
        *color1* and *color_2* in rgb coordinate
        *n* integer number of color in gradient
        *long_path* boolean If True, select the path that is > 180° in hue
        *f* function default to linear, general case c1 + f * (c2 - c1)
        """
        return Color_tools.customized_Lch_gradient(color_1, color_2, n,
                                                   long_path, f, f, f)
    
    @staticmethod
    def color_gradient(kind, color_1, color_2, n, f=lambda t: t):
        """
        Return a color gradient between color_1 and color_2, default linear in
        Lch
        *kind* "Lab" | "Lch" 
        *color1* and *color_2* in rgb coordinate
        *n* integer number of color in gradient
        *f* function default to linear, general case c1 + f * (c2 - c1)
        """
        if kind == "Lab":
            return Color_tools.Lab_gradient(color_1, color_2, n, f=f)
        elif kind == "Lch":
            return Color_tools.Lch_gradient(color_1, color_2, n, f=f)
        else: 
            raise ValueError(
                    "Unsupported kind {}, expecting Lab or Lch".format(kind))

    @staticmethod
    def customized_Lab_gradient(color_1, color_2, n, 
        L=lambda t: t, ab=lambda t: t):
        """
        Idem Lab_gradient except we customize 2 functions L and a = b noted ab
        """
        Lab_1 = Color_tools.rgb_to_CIELab(color_1[np.newaxis, :])[0, :]
        Lab_2 = Color_tools.rgb_to_CIELab(color_2[np.newaxis, :])[0, :]
        arr = np.vstack([Lab_1[i] + 
                         f(np.linspace(0., 1., n)) * (Lab_2[i] - Lab_1[i]) 
                         for f, i in zip([L, ab, ab], [0, 1, 2])]).T
        return Color_tools.CIELab_to_rgb(arr)

    @staticmethod
    def customized_Lch_gradient(color_1, color_2, n, long_path=False,
        L=lambda t: t, c=lambda t: t, h=lambda t: t):
        """
        Idem Lch_gradient except we customize all 3 functions L, c, h
        """
        Lch_1 = Color_tools.rgb_to_CIELch(color_1[np.newaxis, :])[0, :]
        Lch_2 = Color_tools.rgb_to_CIELch(color_2[np.newaxis, :])[0, :]
        h1, h2 = Lch_1[2], Lch_2[2]
        if np.abs(h2- h1) > 180. and (not long_path):
            Lch_2[2] = h2 - np.sign(h2 - h1) * 360. # take the shortest path
        if np.abs(h2- h1) < 180. and long_path:
            Lch_2[2] = h2 - np.sign(h2 - h1) * 360. # take the longest path
        arr = np.vstack([Lch_1[i] + 
                         f(np.linspace(0., 1., n)) * (Lch_2[i] - Lch_1[i]) 
                         for f, i in zip([L, c, h], [0, 1, 2])]).T
        return Color_tools.CIELch_to_rgb(arr)


    @staticmethod
    def blend(rgb, shade, shade_type=None):
        """
        Provides several "shading" options based on shade_type dict
        *rgb* array of colors to 'shade', shape (nx, 3) or (nx, ny, 3)
        *shade* N&B array shape similar to rgb but last dim is 1
        *shade_type* {"Lch": x1, "overlay": x2, "pegtop": x3}
            x1, x2, x3 positive scalars, the proportion of each shading method
            in the final image.
        """
        if shade_type is None:
            shade_type = {"Lch": 4., "overlay": 4., "pegtop": 1.}        
        
        blend_T = float(sum((shade_type.get(key, 0.)
                        for key in["Lch", "overlay", "pegtop"])))

        is_image = (len(rgb.shape) == 3)
        if is_image:
            imx, imy, ichannel = rgb.shape
            if ichannel!= 3:
                raise ValueError("expectd rgb array")
            rgb = np.copy(rgb.reshape(imx * imy, 3))
            shade = np.copy(shade.reshape(imx * imy, 1))

        XYZ = Color_tools.rgb_to_XYZ(rgb[:, 0:3])

        XYZ_overlay = np.zeros([imx * imy, 3])
        XYZ_pegtop = np.zeros([imx * imy, 3])
        XYZ_Lch = np.zeros([imx * imy, 3])

        ref_white = Color_tools.D50_ref_white


        if shade_type.get("overlay", 0.) != 0:
            low = 2. * shade * XYZ
            high = ref_white * 100. - 2.  * (1. - shade) * (ref_white * 100. - XYZ)
            XYZ_overlay =  np.where(XYZ <= 0.5 * ref_white * 100., low, high)            

        if shade_type.get("pegtop", 0.) != 0:
            XYZ_pegtop = 2. * shade * XYZ + (1. - 2. * shade) * XYZ**2 / ref_white

        if shade_type.get("Lch", 0.) != 0:
            shade = 2. * shade - 1.
            Lab = Color_tools.XYZ_to_CIELab(XYZ)
            L = Lab[:, 0, np.newaxis]
            a = Lab[:, 1, np.newaxis]
            b = Lab[:, 2, np.newaxis]
            np.putmask(L, shade > 0, L  + shade * (100. - L))     # lighten
            np.putmask(L, shade < 0, L * (1. +  shade ))          # darken
            np.putmask(a, shade > 0, a  - shade**2 * a)           # lighten
            np.putmask(a, shade < 0, a * (1. -  shade**2 ))       # darken
            np.putmask(b, shade > 0, b  - shade**2 * b)           # lighten
            np.putmask(b, shade < 0, b * (1. -  shade**2 ))       # darken
            Lab[:, 0] = L[:, 0]
            Lab[:, 1] = a[:, 0]
            Lab[:, 2] = b[:, 0]
            XYZ_Lch = Color_tools.CIELab_to_XYZ(Lab)

        XYZ = (XYZ_overlay * shade_type["overlay"] +
               XYZ_pegtop * shade_type["pegtop"] +
               XYZ_Lch * shade_type["Lch"]) / blend_T

        # Convert modified hsv back to rgb.
        blend = Color_tools.XYZ_to_rgb(XYZ)
        if is_image:
            blend = blend.reshape([imx, imy, 3])
        return blend

    @staticmethod
    def shade_layer(normal, theta_LS, phi_LS, shininess=0., ratio_specular=0.,
                    **kwargs):
        """
        *normal* flat array of normal vect
        shade_dict:
            "theta_LS" angle of incoming light [0, 360]
            "phi_LS"   azimuth of incoming light [0, 90] 90 is vertical
            "shininess" material coefficient for specular
            "ratio_specular" ratio of specular to lambert
        Returns 
        *shade* array of light intensity, greyscale image (value btwn 0 and 1)
        https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
        """
        if "LS_coords" in kwargs.keys():
            # LS is localized somewhere in the image computing theta_LS
            # as a vector
            LSx, LSy = kwargs["LS_coords"]
            (ix, ixx, iy, iyy) = kwargs["chunk_slice"]
            chunk_mask = kwargs["chunk_mask"]
            nx = kwargs["nx"]
            ny = kwargs["ny"]
            nx_grid = (np.arange(ix, ixx, dtype=np.float32) / nx) - 0.5
            ny_grid = (np.arange(iy, iyy, dtype=np.float32) / ny) - 0.5
            ny_vec, nx_vec  = np.meshgrid(ny_grid, nx_grid)
            theta_LS = - np.ravel(np.arctan2(LSy - ny_vec, nx_vec - LSx)) + np.pi
            if chunk_mask is not None:
                theta_LS = theta_LS[chunk_mask]
        else:
            # Default case LS at infinity incoming angle provided
            theta_LS = theta_LS * np.pi / 180.
        phi_LS = phi_LS * np.pi / 180.
        
        if "exp_map" in kwargs.keys():
            raise ValueError() # debug
            # Normal angle correction in case of exponential map
            if kwargs["exp_map"]:
                (ix, ixx, iy, iyy) = kwargs["chunk_slice"]
                chunk_mask = kwargs["chunk_mask"]
                nx = kwargs["nx"]
                ny = kwargs["ny"]
                nx_grid = (np.arange(ix, ixx, dtype=np.float32) / nx) - 0.5
                ny_grid = (np.arange(iy, iyy, dtype=np.float32) / ny) - 0.5
                ny_vec, nx_vec  = np.meshgrid(ny_grid, nx_grid)
                expmap_angle = np.ravel(np.exp(-1j * (ny_vec) * np.pi * 2.))
                if chunk_mask is not None:
                    expmap_angle = expmap_angle[chunk_mask]
                normal = normal * expmap_angle

        # k_ambient = - 1. / (2. * ratio_specular + 1.)
        k_lambert = 1. #- 2. * k_ambient
        k_spec = ratio_specular * k_lambert

        # Light source coordinates
        LSx = np.cos(theta_LS) * np.cos(phi_LS)
        LSy = np.sin(theta_LS) * np.cos(phi_LS)
        LSz = np.sin(phi_LS) 

        # Normal vector coordinates - Lambert shading
        nx = normal.real
        ny = normal.imag
        nz = np.sqrt(1. - nx**2 - ny**2) 
        if "inverse_n" in kwargs.keys():
            if kwargs["inverse_n"]:
                nx = -nx
                ny = -ny
                
        lambert = LSx * nx + LSy * ny + LSz * nz
        np.putmask(lambert, lambert < 0., 0.)

        # half-way vector coordinates - Blinn Phong shading
        specular = np.zeros_like(lambert)
        if ratio_specular != 0.:
            phi_half = (np.pi * 0.5 + phi_LS) * 0.5
            half_x = np.cos(theta_LS) * np.sin(phi_half)
            half_y = np.sin(theta_LS) * np.sin(phi_half)
            half_z = np.cos(phi_half)
            spec_angle = half_x * nx + half_y * ny + half_z * nz
            np.putmask(spec_angle, spec_angle < 0., 0.)
            specular = np.power(spec_angle, shininess)
            
        res =  k_lambert * lambert + k_spec * specular # + k_ambient
        
        #res[normal == 0.] = 0.5 * (np.nanmin(res) + np.nanmax(res))
        try:
            np.putmask(res, normal == 0., np.nanmin(res) + 0.5 * (np.nanmax(res) - np.nanmin(res)))
        except ValueError:
            pass

        return res# k_ambient + k_lambert * lambert + k_spec * specular

class Color:
    def __init__(self, arr):
        """ An interface for QtGui.QColor """
        self._validate(arr)
        self.arr = arr

    def _validate(self, arr):
        if len(arr) not in (3, 4):
            raise ValueError(f"arr shape not as expected: {arr.shape}")
        


class Fractal_colormap:

    KIND_ENUM = enum.Enum(
        "kind",
        ("Lab", "Lch"),
        module=__name__
    )
    kind_type = typing.Literal[KIND_ENUM]

    EXTENT_ENUM = enum.Enum(
        "extent",
        ("clip", "mirror", "repeat"),
        module=__name__
    )
    extent_type = typing.Literal[EXTENT_ENUM]
    
    def __init__(
        self,
        colors: Color,
        kinds: kind_type,
        grad_npts: int=12,
        grad_funcs: np_utils.Numpy_expr="x",
        extent: extent_type="mirror"
    ):
        """
        Fractal_colormap, concatenates (n-1) color gradients (passing
        through n colors).

        Parameters
        ==========
        colors : rgb float np.array of shape [n, 3]
            The successives colors of the colormap
        kinds: arrays of string [n-1] , "Lab" or "Lch"
            The kind of gradient between colors n and n+1 : either linear in
            Lab space of Lch space
        grad_npts : int np.array of size [n-1]
            number of internal points stores, gradient between colors n and n+1
            a typical value is 32.
        grad_funcs : [n-1] array of callables mapping [0, 1.] to [0., 1]
            These are passed to each gradient. Callable passed as a string
            expression of the "x" variable
            (ie, the callable is the evaluation of "lambda x: " + expr).
            Default to identity ("x").
        extent : "clip" | "mirror" | "repeat"
            What to do with out-of-range values.

        Notes
        =====
        .. note::

            See also the colormap section of the gallery for the available
            templates.
        """
        self.colors = colors = np.asarray(colors)
        self.n_grads = n_grads = colors.shape[0] - 1
        self.n_probes = colors.shape[0]

        # Provides sensible defaults if a scalar is provided
        if isinstance(kinds, str):
            kinds = [kinds] * n_grads
        if isinstance(grad_npts, int):
            grad_npts = [grad_npts] * n_grads
        if isinstance(grad_funcs, str): #callable(grad_funcs):
            grad_funcs = [grad_funcs] * n_grads

        self.kinds = kinds # = np.asarray(kinds)
        self.grad_npts = grad_npts = np.asarray(grad_npts)
        self.grad_funcs = grad_funcs # = np.asarray(grad_funcs)
        self.extent = extent

        # Deprecated option (not compatible with GUI editor)
        if callable(grad_funcs):
            raise ValueError("Callable grad_funcs deprecated, use string")
        
        self._load_internal_arrays()


    def _load_internal_arrays(self):
        """ To be called when one of the base array is modified """
        n_grads = self.n_grads
        n_probes = self.n_probes

        colors = self.colors
        grad_npts = self.grad_npts
        grad_funcs = self.grad_funcs
        kinds = self.kinds

        # Should define 2 internal arrays
        self._n_interp_colors = sum(grad_npts) - n_grads + 1
        self._interp_colors = np.empty(
                [self._n_interp_colors, 3], dtype=np.float64
        )
        self._probes = np.empty([n_probes], dtype=np.float32)

        grad_func_evals = [
            np_utils.func_parser(["x"], grad_funcs[i_grad])
            for i_grad in range(n_grads)
        ]

        i_col = 0
        for i_grad in range(n_grads):
            grad = Color_tools.color_gradient(
                kinds[i_grad], colors[i_grad, :], colors[i_grad + 1, :],
                grad_npts[i_grad], grad_func_evals[i_grad])

            self._interp_colors[i_col: i_col + grad_npts[i_grad], :] = grad
            self._probes[i_grad] = i_col
            i_col += (grad_npts[i_grad] - 1)
        self._probes[n_grads] = i_col # last piquet

        # Is is it a known template ?
        self._template = None

    #~~~~~~~~~~~~~~ GUI interaction methods
    @property
    def n_rows(self):
        return len(self.colors)

    def modify_item(self, col_key, irow, value):
        """ In place modifcation of cmap """
#        print("in color.Colormap modify_item", col_key, irow, value, type(value))
#        print("In OBJECT modify_item", col_key, irow, value, type(value))
        getattr(self, col_key)[irow] = value
        self._load_internal_arrays()

    def col_data(self, col_key):
        """ Returns a column of the expected field """
        return getattr(self, col_key)

    @property
    def default_kind(self):
        return self.kinds[-1]

    @property
    def default_func(self):
        # Default value for next line when adjusting
        return self.grad_funcs[-1]

    @property
    def default_color(self):
        # Default value for next line when adjusting
        return self.colors[-1, :]

    @property
    def default_grad_npts(self):
        # Default value for next line when adjusting
        return self.grad_npts[-1]

    def adjust_size(self, n_probes):
        """ In place modifcation of array size"""
        n_probes_old = self.n_probes
        diff = n_probes - n_probes_old
        n_grads = self.n_grads + diff

        if diff < 0:
            self.colors = self.colors[:n_probes, :]
            self.kinds = self.kinds[:n_grads]
            self.grad_npts = self.grad_npts[:n_grads]
            self.grad_funcs = self.grad_funcs[:n_grads]

        elif diff > 0:
            self.kinds = self.kinds + [self.default_kind] * diff
            self.grad_funcs = self.grad_funcs + [self.default_func] * diff
            # numpy arrays:
            self.grad_npts =  np.concatenate([
                    self.grad_npts,
                    self.default_grad_npts * np.ones((diff,), dtype=np.int64)],
                axis=0
            )
            self.colors = np.concatenate([
                    self.colors,
                    np.vstack([self.default_color[np.newaxis, :]] * diff)
                ],
                axis=0
            )
        else:
            # Nothing has changed...
            return

        # Store the new values !
        self.n_probes = n_probes
        self.n_grads = n_grads
        self._load_internal_arrays()

    def save_as_pickle(self, save_path):
        """ Save as .cmap pickle file"""
        with open(save_path, 'wb+') as tmpfile:
            pickle.dump(self, tmpfile, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_as_pickle(cls, save_path):
        """ Load a .cmap pickle file and returns the result"""
        with open(save_path, "rb") as tmpfile:
            cmap_res = pickle.load(tmpfile)
        return cmap_res


    def script_repr(self, indent=0):
        """ Return a string that can be used to restore the colormap
        """
        shift = " " * (4 * (indent + 1))
        shift_arr = " " * (4 * (indent + 2))

        if hasattr(self, "_template") and self._template is not None:
            return f"fs.colors.cmap_register[\"{self._template}\"]"

        colors_str = np.array2string(self.colors, separator=', ')
        colors_str = colors_str.replace("\n", "\n" + shift_arr)

        kinds_str = pprint.pformat(self.kinds, compact=True) # repr(self.kinds)
        kinds_str = kinds_str.replace("\n", "\n" + shift_arr)

        grad_npts_str = np.array2string(self.grad_npts, separator=', ')
        grad_npts_str = grad_npts_str.replace("\n", "\n" + shift_arr)

        grad_funcs_str = pprint.pformat(self.grad_funcs, compact=True) # repr(self.grad_funcs)
        grad_funcs_str = grad_funcs_str.replace("\n", "\n" + shift_arr)

        extent_str = self.extent
        return (
            "fs.colors.Fractal_colormap(\n"
            f"{shift}colors={{}},\n"
            f"{shift}kinds={{}},\n"
            f"{shift}grad_npts={{}},\n"
            f"{shift}grad_funcs={{}},\n"
            f"{shift}extent=\'{{}}\'\n)"
        ).format(colors_str, kinds_str, grad_npts_str, grad_funcs_str,
                 extent_str)


    def _output(self, nx, ny):
        nx_fi = nx
        ny_fi = ny
        
        nx *= 2
        ny *= 2
        
        margin = 1
        nx_im = nx - 4 * margin
        ny_im = ny - 4 * margin
        arrow_size = 0.05
        
        img = np.repeat(
            np.linspace(-arrow_size, 1. + arrow_size, nx_im)[:, np.newaxis],
            ny_im, axis=1
        )
        i_arr = np.copy(img)
        j_arr = np.repeat(
            np.linspace(-arrow_size, arrow_size, ny_im)[np.newaxis, :],
            nx_im, axis=0
        )
        
        img = self.colorize(img, np.linspace(0., 1., len(self._probes)))
        img = np.uint8(img * 255.999)
        self._add_arrow(img, i_arr, j_arr, arrow_size)
        
        # Apply LANCZOS filter
        PIL_img = PIL.Image.fromarray(img)
        PIL_img = PIL_img.resize(
             size=((ny_fi - 2 * margin), (nx_fi - 2 * margin)),
             resample=PIL.Image.LANCZOS
        )
        img = np.array(PIL_img)
        
        B = np.ones([nx_fi, ny_fi, 3], dtype=np.uint8) * 255
        B[margin:nx_fi - margin, margin:ny_fi - margin, :] = img
        return np.swapaxes(B, 0, 1)
    
    @staticmethod
    def _add_arrow(img, i_arr, j_arr, arrow_size):
        """ Draws the shape of the extra arrow in white """
        arr1 = (i_arr + j_arr < -arrow_size)
        arr2 = (i_arr - j_arr < -arrow_size)
        arr3 = (i_arr + j_arr > 1. + arrow_size)
        arr4 = (i_arr - j_arr > 1. + arrow_size)
        for arrloc in (arr1, arr2, arr3, arr4):
            np.putmask(img[:, :, 0], arrloc, (255,))
            np.putmask(img[:, :, 1], arrloc, (255,))
            np.putmask(img[:, :, 2], arrloc, (255,))

        

    def output_png(self, dir_path, file_name, nx, ny):
        """
        Outputs the colorbar to a .png files:
        dir_path/file_name.cbar.png
        """
        B = self._output(nx, ny)
        PIL.Image.fromarray(B).save(
            os.path.join(dir_path, file_name + ".cbar.png"))

    def output_ImageQt(self, nx, ny):
#https://stackoverflow.com/questions/34697559/pil-image-to-qpixmap-conversion-issue
        B = self._output(nx, ny)
        return PIL.ImageQt.ImageQt(PIL.Image.fromarray(B))

    def colorize(self, z, probes_z):
        """
        Returns a color array bazed on z values
        """
        nx, ny = z.shape
        z = np.ravel(z)
        z = self.normalize(z, probes_z)
        # linear interpolation in sorted color array
        indices = np.searchsorted(np.arange(self._n_interp_colors), z)
        alpha = indices - z
        search_colors = np.vstack([
            self._interp_colors[0, :],
            self._interp_colors,
            self._interp_colors[-1, :]
        ])
        z_colors = (alpha[:, np.newaxis] * search_colors[indices, :] + 
             (1.-alpha[:, np.newaxis]) * search_colors[indices + 1, :])
        return np.reshape(z_colors, [nx, ny, 3])  

    def greyscale(self, z, probes_z):
        """
        Returns a color array bazed on z values
        """
        z = self.normalize(z, probes_z)
        # linear interpolation in sorted greyscale array
        indices = np.searchsorted(np.arange(self.n_interp_colors), z)
        alpha = indices - z
        search_colors = np.concatenate([
            [0.], np.linspace(0., 1., self.n_interp_colors),  [1.]])
        z_greys = (alpha * search_colors[indices] +
                   (1. - alpha) * search_colors[indices + 1])
        return z_greys

    def normalize(self, z, probes_z):
        """
        Normalise z , z -> z* so that z*min = 0 / z*max = self.n_interp_colors
        *z*  array to normalise
        *probes_z* array of dim self.n_probes: values of z at probes
        """
        if probes_z.shape != self._probes.shape:
            if probes_z.shape == (2,):
                probes_z = np.linspace(probes_z[0], probes_z[1],
                    num=self._probes.shape[0], dtype=probes_z.dtype)
                assert probes_z.shape == self._probes.shape
            else:
                raise ValueError("Expected *probes_values* of shape {0}, "
                    "given {1}".format(self._probes.shape, probes_z.shape))        

        # on over / under flow : clip or mirror
        ext_min = np.min(probes_z)
        ext_max = np.max(probes_z)
        if self.extent == "clip":
            z = self.clip(z, ext_min, ext_max)
        elif self.extent == "mirror":
            z = self.mirror(z, ext_min, ext_max)
        elif self.extent == "repeat":
            z = self.repeat(z, ext_min, ext_max)
        else:
            raise ValueError("Unexpected extent property {}".format(
                    self.extent))

        return np.interp(z, probes_z, self._probes)
    
    @staticmethod
    def clip(x, ext_min, ext_max):
        np.putmask(x, x < ext_min, ext_min)
        np.putmask(x, x > ext_max, ext_max)
        return x

    @staticmethod
    def mirror(x, ext_min, ext_max):
        """ Mirroring of x on ext_min & ext_max
Formula: https://en.wikipedia.org/wiki/Triangle_wave
4/p * (t - p/2 * floor(2t/p + 1/2)) * (-1)**floor(2t/p + 1/2) where p = 4
        """
        t = (x - ext_min) / (ext_max - ext_min)  # btw. 0. and 1
        e = np.floor((t + 1) / 2)
        t = np.abs((t - 2. * np.floor(e)) * (-1)**e)
        return ext_min + (t * (ext_max - ext_min))
    
    @staticmethod
    def repeat(x, ext_min, ext_max):
        """
        Repeating x based on ext_min & ext_max (Sawtooth)
        """
        return ext_min + ((x - ext_min) % (ext_max - ext_min))


class Image_interpolator:
    def __init__(self, image, wrap=False, screen_coord=True):
        """
        Interpolation of points in an image
        image: Pillow image
        wrap : if True, margins are wraped
        screen_coord : if True pixels to interpolate are in screen coords
                       (y downwards)
        """
        self._im = image
        self._wrap = wrap
        self._screen_coord = screen_coord

    def interpolate(self, x, y):
        mode = self._im.mode 
        if mode == "RGB":
            return self.interpolate_rgb(x, y)
        else:
            raise ValueError(f"Unsupported {mode}")
    
    def interpolate_rgb(self, x, y):
        """
        x arrays of floats between [0, nx]
        y arrays of floats between [0, ny]
        
        Return an RGB array of shape shape-x) + (3,), dtype np.uint8)
        """
        shape = x.shape
        res = np.empty(shape + (3,), dtype=np.uint8)
        wrap = self._wrap
        screen_coord = self._screen_coord
        if y.shape != shape:
            raise ValueError("x and y shall have same shape")
        xr= np.ravel(x)
        yr = np.ravel(y)
        print("channels", self._im)
        # channels <PIL.PngImagePlugin.PngImageFile image mode=RGB size=3600x3600 at 0x7FF2C0CA20A0>
        im_arr = np.array(self._im)
        for ic in range(3):
            print("channel", ic)
            channel = im_arr[:, :, ic]#self._im.getchannel(ic)
            out = np.empty(len(xr))
            im_interpolate(channel, xr, yr, wrap, screen_coord, out)
            res[..., ic] = out #/ 255.
        return res
        

@numba.njit
def im_interpolate(channel, x, y, wrap, screen_coord, out):
    """
    Numba jitted function to interpolate in an image channel

    channel: The 2d image channel to interpolate 
    x: The 1d x-axis pts for interpolation
    y: The 1d y-axis pts for interpolation
    wrap: if True, wraps, otherwise, clip
    out: the res array

    # note :  channel.shape: height x width (column-major)
    """
    (ny, nx) =  channel.shape
    npts, = x.shape
    for ipt in range(npts):
        # The x coords
        xloc = x[ipt] - 0.5
        divx, modx = np.divmod(xloc, 1.)
        divx = int(divx)
        if 0 <= divx <= nx - 2:
            ix1 = divx
            ix2 = ix1 + 1
        elif divx < 0: # divx = -1
            ix2 = 0
            if wrap:
                ix1 = nx - 1
            else:
                ix1 = 0
        else: # divx = nx - 1
            ix1 = nx - 1
            if wrap:
                ix1 = 0
            else:
                ix1 = nx - 1
        # The y coords
        if False: #screen_coord:
            yloc = y[ipt] - ny + 0.5
        else:
            yloc = y[ipt] - 0.5
        divy, mody = np.divmod(yloc, 1.)
        divy = int(divy)
        if 0 <= divy <= ny - 2:
            iy1 = divy
            iy2 = iy1 + 1
        elif divy < 0: # divy = -1
            iy2 = 0
            if wrap:
                iy1 = ny - 1
            else:
                iy1 = 0
        else: # divx = ny - 1
            iy1 = ny - 1
            if wrap:
                iy1 = 0
            else:
                iy1 = ny - 1
        # The interpolation as barycentric coordinates
        out[ipt] = (
            (1. - modx) * (1. - mody) * channel[iy1, ix1]
            + (1. - modx) * mody * channel[iy2, ix1]
            + modx * (1. - mody) * channel[iy1, ix2]
            + modx * mody * channel[iy2, ix2]
        )
            







class Curve:
    def __init__(self, fn="x", brightness=None, hardness=None,
                 blur_ranges=None):
        """
        TODO - work in progress
        A transfert curve from [0, 1] to [0, 1]
        ("Grey levels map")
        brightness float, O. leaves untouched, 0..+1 brighten
                                              -1..0 darken
        hardness float, 1. leaves untouched, 1..inf more contrast 
                                              0..1 les contrast
        blur_ranges: danping base on layer "damping_pp"               
        """
        # TODO: refactor as an editor with control points ?
        self.fn = fn
        self.brightness = brightness
        self.hardness = hardness
        self.blur_ranges = blur_ranges
    
    def set_layer(self, layer):
        """  register the curve as linked to a layer
        """
        self.layer = layer

    @property
    def blur_base(self):
        return self.layer.damping_pp

    def __call__(self, data):      
        brightness = self.brightness
        hardness = self.hardness
        blur_ranges = self.blur_ranges

        # renormalise between -1 and 1
        data = 2. * data - 1.

        if brightness is not None:
            data = self.brighten(brightness, data)
        if hardness is not None:
            data = self.harden(hardness, data)
        if blur_ranges is not None:
            data = self.blur(blur_ranges, data, self.blur_base)
        data = 0.5 * (data + 1.)
        return data


    @staticmethod
    def brighten(brightness, data):
        """
        Input : data between [-1, 1]
        output: data linearly rescaled between [min, max] so that :
            - min = -1 or max = +1
            - skewness = 0.5 * (min+max)
        """
        if abs(brightness) >= 1.:
            raise ValueError("Expected skew strictly between -1. and 1.")
        if brightness >= 0.:
            return 1. - (1. - data) * (1. - brightness)
        else:
            return -1. + (1. + data) * (1. + brightness)

    @staticmethod
    def harden(hardness, data):
        """
        Adjust contrast for a np.array between -1 and 1
        Parameter:
        *hardness*   if  1. data unchanged
                     if  0. < hardness < 1., soften the contrast
                     if  hardness > 1., accentuate the contrast
        """
        if hardness < 0.:
            raise ValueError(hardness)
        return np.sign(data) * np.abs(data) ** (1. / hardness)


    @staticmethod
    def bluring_coeff(blur1, blur2, blur3, x):
        """
        Define bluring coefficents for *x*, parameter blur1, blur2, blur3
        Paramters
        *blur1*, *blur2*, *blur3*    Monotonic, real inputs
        *x*                          Array
        
        Returns
        *y*   Arrays of bluring coefficient at x location, as below if blur 
              param are increasing :
                 1 below blur1.    
                 smooth from blur1 to blur 2 1. -> 0.
                 stay at 0 between blur2 and blur 3
                 back to 1. after blur3
            If param blur1 to blur3 are decreasing inverse the logic (mirror)
        """
        if np.sign(blur2 - blur1) * np.sign(blur3 - blur2) == -1:
            raise ValueError("Expected monotonic blur1, blur2, blur3")
        sign = np.sign(blur3 - blur1)
        if sign == 0:
            raise ValueError("Unable to define blur direction, as blur1=blur3")
        elif sign == -1:
            return Curve.bluring_coeff(-blur1, -blur2, -blur3, -x)
        else:
            y = np.zeros_like(x)
            y[x <= blur1] = 1.
            y[x > blur3] = 1. 
            y[(x > blur2) & (x <= blur3)] = 0.
            mid = (blur1 + blur2) * 0.5
            l = mid - blur1
            mask = (x > blur1) & (x <= mid)
            y[mask] = 1. - ((x[mask] - blur1) / l)**3 * 0.5 
            mask = (x > mid) & (x <= blur2)
            y[mask] = ((blur2 - x[mask]) / l)**3 * 0.5 
            return y

    @staticmethod
    def blur(blur_ranges, data, blur_base):
        """
        Selectively "blurs" ie contracts to 0. the data inside "blur ranges"
        defined as quantiles of the base_layer_data.
        Parameters :
        blur_ranges [[blur1, blur2, blur3],
                     [blur2, blur2, blur3], ...]  
                     bluri expressed as quantiles of base_layer_data (in 0 - 1)

        qt_base     distribution function applied to base data values 
                    (= the quantiles)
        """
        for blur_range in blur_ranges:
            blur1, blur2, blur3 = blur_range
            data = data * Curve.bluring_coeff(blur1, blur2, blur3, blur_base)
        return data
