# -*- coding: utf-8 -*-
import numpy as np
import pickle

import matplotlib.colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import os, errno, sys
import functools
import copy
import PIL


def mkdir_p(path):
    """ Creates directory ; if exists does nothing """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc

class Color_tools():
    """ A bunch of staticmethods
    Color conversions from http://www.easyrgb.com/en/math.php#text7
    All take as input and return arrays of size [n, 3]"""
    # CIE LAB constants Illuminant= D65
    Lab_ref_white = np.array([0.95047, 1., 1.08883])
    # CIE LAB constants Illuminant= D50
    D50_ref_white = np.array([0.964212, 1., .825188])

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
    def customized_Lab_gradient(color_1, color_2, n, 
        L = lambda t: t, ab = lambda t: t):
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
        L = lambda t: t, c = lambda t: t, h = lambda t: t):
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

        ref_white = Color_tools.Lab_ref_white

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
            np.putmask(L, shade > 0, L  + shade * (100. - L))  # lighten
            np.putmask(L, shade < 0, L * (1. +  shade ))       # darken
            np.putmask(a, shade > 0, a  - shade**2 * a)        # lighten
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
    def shade_layer(normal, theta_LS, phi_LS, shininess=0., ratio_specular=0.):
        """
        *normal* flat array of normal vect
        shade_dict:
            "theta_LS" angle of incoming light [0, 360]
            "phi_LS"   azimuth of incoming light [0, 90] 90 is vertical
            "shininess" material coefficient for specular
            "ratio_specular" ratio of specular to lambert
        Returns 
        *shade* array of light intensity, n&B image (value btwn 0 and 1)
        https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
        """
        theta_LS = theta_LS * np.pi / 180.
        phi_LS = phi_LS * np.pi / 180.

        k_ambient = - 1. / (2. * ratio_specular + 1.)
        k_lambert = - 2. * k_ambient
        k_spec = ratio_specular * k_lambert

        # Light source coordinates
        LSx = np.cos(theta_LS) * np.cos(phi_LS)
        LSy = np.sin(theta_LS) * np.cos(phi_LS)
        LSz = np.sin(phi_LS) 

        # Normal vector coordinates - Lambert shading
        nx = normal.real
        ny = normal.imag
        nz = np.sqrt(1. - nx**2 - ny**2) 
        lambert = LSx * nx + LSy * ny + LSz * nz
        np.putmask(lambert, lambert < 0., 0.)

        # half-way vector coordinates - Blinn Phong shading
        specular = np.zeros_like(lambert)
        if ratio_specular != 0.:
            phi_half = (np.pi * 0.5 + phi_LS) * 0.5
            half_x = np.cos(theta_LS) * np.cos(phi_half)
            half_y = np.sin(theta_LS) * np.cos(phi_half)
            half_z = np.sin(phi_half)
            spec_angle = half_x * nx + half_y * ny + half_z * nz
            np.putmask(spec_angle, spec_angle < 0., 0.)
            specular = np.power(spec_angle, shininess)

        return  k_ambient + k_lambert * lambert + k_spec * specular


class Fractal_colormap():
    def __init__(self, color_gradient, colormap=None):
        """
        Creates from : 
            - Directly a color gradient array (as output by Color_tools
              gradient functions, array of shape (n_colors, 3))
            - A matplotlib colormap ; in this case color_gradient shall be a
              list of the form (start, stop, n _colors)
        *colormap* 
        """
        if colormap is None:
            self._colors = color_gradient
            n_colors, _ = color_gradient.shape
        elif isinstance(colormap, matplotlib.colors.Colormap):
            x1, x2, n_colors = color_gradient
            self._colors = colormap(np.linspace(x1, x2, n_colors))[:, :3]
        else:
            raise ValueError("Invalid *colormap* argument, expected a "
                             "mpl.colors.Colormap or None")
        self._probes = np.array([0, n_colors-1])
        self.quantiles_ref = None


    @property
    def n_colors(self):
        """
        Total number of colors in the colormap.
        """
        return self._colors.shape[0]

    @property
    def probes(self):
        """
        Position of the "probes" ie transitions between the different parts of
        the colormap. Read-only.
        """
        return np.copy(self._probes)

    def __neg__(self):
        """
        Returns a reversed colormap
        """
        other = Fractal_colormap(self._colors[::-1, :])
        other._probes = self._probes[-1] - self._probes[::-1]
        return other
    
    def __add__(self, other):
        """
        Concatenates 2 Colormaps
        """
        fcm = Fractal_colormap(np.vstack([self._colors, other._colors]))
        fcm._probes = np.concatenate([self._probes,
            (other._probes + self._probes[-1] + 1)[1:]])
        return fcm

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def output_png(self, dir_path, file_name):
        """
        Outputs the colorbar to a .png files:
        dir_path/file_name.cbar.png
        """
        img = np.repeat(self._colors[: , np.newaxis, :],
                        30 , axis=1)
        img = np.uint8(img * 255.99)
        ix, iy, _ = np.shape(img)
        margin = 10
        B = np.ones([ix + 2 * margin, iy + 2 * margin, 3], 
                    dtype=np.uint8) * 255
        B[margin:margin+ix, margin:margin+iy, :] = img
        PIL.Image.fromarray(np.swapaxes(B, 0, 1)).save(
            os.path.join(dir_path, file_name + ".cbar.png"))

    def colorize(self, z, probe_values):
        """
        Returns a color array bazed on z values
        """
        z = self.normalize(z, probe_values)
        # on over / under flow, default to max / min color
        np.putmask(z, z < 0., 0.)
        np.putmask(z, z > self.n_colors - 1., self.n_colors - 1.)
        # linear interpolation in sorted color array
        indices = np.searchsorted(np.arange(self.n_colors), z)
        alpha = indices - z
        search_colors = np.vstack([self._colors[0, :],
                                   self._colors,
                                   self._colors[-1, :]])
        z_colors = (alpha[:, np.newaxis] * search_colors[indices, :] + 
             (1.-alpha[:, np.newaxis]) * search_colors[indices + 1, :])
        return z_colors
    
    def colorize2d(self, z, probe_values):
        nx, ny = z.shape
        z = np.ravel(z)
        z_colors = self.colorize(z, probe_values)
        return np.reshape(z_colors, [nx, ny, 3])  
    
    
    
    def normalize(self, z, probe_values, percentiles_ref=None):
        """
        Normalise z , z -> z* so that
            - z* distribution function F* goes through the points
              (probe, probe_values) ie F*^-1(probe_values) = probe
            - z* is monotonic and C1-smooth
        *z*  array to normalise
        *probes_values* array of dim self.n_probes: stricly increasing
                        fractiles between 0. and 1.
        """
        probe_values = np.asanyarray(probe_values)
        if np.any(probe_values.shape != self._probes.shape):
            raise ValueError("Expected *probes_values* of shape {0}, "
                             "found {1}".format(self._probes.shape,
                                                 probe_values.shape))
        if np.any(np.diff(probe_values) <= 0):
            raise ValueError("Expected strictly increasing probe_values")
        n_probes, = probe_values.shape

        if self.quantiles_ref is not None:
            qt = self.quantiles_ref(probe_values)
        else:
            data_min, data_max = np.nanmin(z),  np.nanmax(z)
            hist, bins = np.histogram(z, bins=100, range=(data_min, data_max))
            qt_ref = np.cumsum(hist) / float(np.count_nonzero(~np.isnan(z)))
            qt_ref = np.insert(qt_ref, 0, 0.)
            qt = lambda x: np.interp(x, qt_ref, bins)            

        return np.interp(z, qt, self._probes)



class Fractal_plotter(object):
    def  __init__(self, fractal, base_data_key, base_data_prefix, 
                  base_data_function, colormap, probe_values, calc_layers=[],
                  mask=None):
        """
        
        *base_data_key* key wich identifies a prosprocessed field from Fractal
                        will be used as the "base source" of color.
                          (post_name, post_dic) for just 1 key here.
        *base_data_prefix* The prefix for the files for accessing the base data
                           arrays.
        """
        self.fractal = fractal
        self.base_data_key = base_data_key
        self.file_prefix = base_data_prefix
        self.base_data_function = base_data_function
        self.colormap = colormap
        self.probe_values = probe_values
        self.mask = mask
        self.calc_layers = calc_layers
        self.hist, self.bins = self.define_hist()
        self.NB_layer = []

    @property
    def plot_dir(self):
        return self.fractal.directory

    @staticmethod
    def loop_crops(func):
        """
        We provide looping over the image pixels for the keyword-arguments:
            - chunk_slice
        The other arguments are simply passed through.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            args = args[1:]
            for chunk_slice in self.fractal.chunk_slices(
                                                      self.fractal.chunk_size):
                kwargs["chunk_slice"] = chunk_slice
                func(self, *args, **kwargs)
        return wrapper


    def define_hist(self, debug_mode=True):
        """
        First pass only to compute the histogramm of "base data"
        """
        nx, ny = self.fractal.nx, self.fractal.ny
        base_data = np.empty([nx, ny], dtype=self.fractal.float_postproc_type)
        self.fill_base_data(base_data)
        base_data = np.ravel(base_data)
        # Need to provide min & max for an histogramm with potential nan values
        data_min, data_max = np.nanmin(base_data),  np.nanmax(base_data)
        print "computing histo", data_min, data_max, base_data.shape
        if np.isnan(data_min) or np.isnan(data_max):
            data_min, data_max = 0., 1.
        hist, bins = np.histogram(base_data, bins=100,
                                  range=(data_min, data_max))
   
        qt_ref = np.cumsum(hist) / float(np.count_nonzero(~np.isnan(base_data)))
        qt_ref = np.insert(qt_ref, 0, 0.)  # pos, val                    

        self.colormap.quantiles_ref = lambda x: np.interp(x, qt_ref, bins)
        self.base_distrib = lambda x: np.interp(x, bins, qt_ref)

        if debug_mode:
             # plot the histogramm
             fig = Figure(figsize=(6., 6.))
             FigureCanvasAgg(fig);
             ax = fig.add_subplot(111)
             width = 1.0 * (bins[1] - bins[0])
             center = (bins[:-1] + bins[1:]) / 2
             ax.bar(center, hist, align='edge', width=width)  
             ax2 = ax.twinx()
             ax2.plot(bins, qt_ref, color="red", linewidth=2)
             fig.savefig(os.path.join(
                 self.plot_dir, self.file_prefix + "_hist.png"))
        print "end computing histo"
        return hist, bins

    def apply_mask(self, array, chunk_slice):
        """
        array shape [:, :] or [n, :, :]
        """
        (ix, ixx, iy, iyy) = chunk_slice
        sh = array.shape
        if self.mask is not None:
            if len(sh) == 2:
                return np.where(self.mask[ix:ixx, iy:iyy], np.nan, array)
            elif len(sh) == 3:
                mask = self.mask[ix:ixx, iy:iyy][np.newaxis, :, :]
                return np.where(mask, np.nan, array)
        return array


    @loop_crops.__func__
    def fill_base_data(self, base_data, chunk_slice=None):
        """"
        fill successive chunks of base_data array
        """
        keys_with_prerequisites = list(self.calc_layers) + [self.base_data_key]

        i_base = len(self.calc_layers)
        chunk_2d = self.fractal.postproc_chunck(keys_with_prerequisites,
             chunk_slice, self.file_prefix)
        chunk_2d = self.apply_mask(chunk_2d, chunk_slice)

        (ix, ixx, iy, iyy) = chunk_slice
        chunk_2d = self.base_data_function(chunk_2d[i_base, :, :])
        base_data[ix:ixx, iy:iyy] = chunk_2d

    def add_calculation_layer(self, postproc_key):
        """
        Adds an intermediate calculation step, which might be needed for 
        postprocessing
        """
        self.calc_layers += [{
            "postproc_key": postproc_key, # field as postprocessed by Fractal
            }]


    def add_NB_layer(self, postproc_key, output=True, normalized=False,
                     Fourrier=None, skewness=None, hardness=None,
                     intensity=None, blur_ranges=None, shade_type=None):
        """
        postproc_key : the key to retrieve a prosprocessed field from Fractal
                          (post_name, post_dic) for just 1 key here.
        output:       Boolen, if true the layer will be output to a N&B image
        Fourrier, hardness, intensity, blur_ranges : arguments passed to 
            further image post-procesing functions, see details in each
            function descriptions
        """

        def func(data, data_min, data_max, qt_base):
            """
            The function that will be applied on successive crops.
            data_min, data_max cannot be evaluated locally and need to be passed
            qt_base is the distribution function applied to the base levels
                    (ie quantiles)
            """
            # First if data is given as a phase, generate the 'physical' data
            if Fourrier is not None:
                data = self.data_from_phase(Fourrier, data)
            # renormalise between -1 and 1
            if not(normalized):
                data = (2 * data - (data_max + data_min)) / (
                        data_max - data_min)

            if skewness is not None:
                data = self.skew(skewness, data)
            if hardness is not None:
                data = self.harden(hardness, data)
            if intensity is not None:
                data = self.intensify(intensity, data)
            if blur_ranges is not None:
                data = self.blur(blur_ranges, data, qt_base)
            data = 0.5 * (data + 1.)
            return data

        i_layer = len(self.NB_layer) + 1
        self.NB_layer += [{
            "postproc_key": postproc_key, # field as postprocessed by Fractal
            "layer_func": func,   # will be applied to the postprocessed field
            "output": output,
            "Fourrier": Fourrier,
            "shade_type": shade_type
            }]
        return i_layer


    def plot(self, file_name, transparency=False, mask_color=(0., 0., 0.)):
        """
        file_name
        
        Note: Pillow image modes:
        P (8-bit pixels, mapped to any other mode using a color palette)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        """
        self.colormap.output_png(self.plot_dir, file_name)

        mode = "RGB"
        if (len(mask_color) > 3) and (self.mask is not None):
            mode = "RGBA"

        nx, ny = self.fractal.nx, self.fractal.ny
        base_img = PIL.Image.new(mode=mode, size=(nx, ny), color=0)

        postproc_keys = []
        NB_img = []
        postproc_keys = list(self.calc_layers)
        postproc_keys += [self.base_data_key]

        for i_layer, layer_options in enumerate(self.NB_layer):
            postproc_keys += [layer_options["postproc_key"]]
            if layer_options["output"]:
                NB_img += [PIL.Image.new(mode="P", size=(nx, ny), color=0)]

        layers_minmax = [[np.inf, -np.inf] for layer in self.NB_layer]
        self.compute_layers_minmax(layers_minmax, postproc_keys)   
        
        self.plot_cropped(base_img, NB_img, postproc_keys, 
                          mask_color, layers_minmax)

        base_img.save(os.path.join(self.plot_dir, file_name + ".png"))
        i_NB_img = 0
        for i_layer, layer_options in enumerate(self.NB_layer):
            if layer_options["output"]:
                NB_img[i_NB_img].save(os.path.join(self.plot_dir,
                    file_name + "_" + layer_options["postproc_key"][0] + 
                    ".layer" + str(i_NB_img) + ".png"))
                i_NB_img += 1

    @loop_crops.__func__
    def compute_layers_minmax(self, layers_minmax, postproc_keys,
                              chunk_slice=None):
        chunk_2d = self.fractal.postproc_chunck(postproc_keys, chunk_slice,
                                                self.file_prefix)
        chunk_2d = self.apply_mask(chunk_2d, chunk_slice)

        i_base = len(self.calc_layers)
        for i_layer, layer_options in enumerate(self.NB_layer):
            data_layer = chunk_2d[i_layer + i_base + 1, :, :]
            if layer_options["Fourrier"] is not None:
                data_layer = self.data_from_phase(layer_options["Fourrier"],
                                                  data_layer)
            lmin, lmax = np.nanmin(data_layer), np.nanmax(data_layer)
            cmin, cmax = layers_minmax[i_layer]
            layers_minmax[i_layer] = [np.nanmin([lmin, cmin]),
                                      np.nanmax([lmax, cmax])]

    @staticmethod
    def np2PIL(arr):
        """
        Unfortunately this is a mess between numpy and pillow
        """
        sh = arr.shape
        if len(sh) == 2:
            nx, ny = arr.shape
            return np.swapaxes(arr, 0 , 1 )[::-1, :]
        nx, ny, _ = arr.shape
        return np.swapaxes(arr, 0 , 1 )[::-1, :, :]

    @loop_crops.__func__
    def plot_cropped(self, base_img, NB_img, postproc_keys, 
                     mask_color, layers_minmax, chunk_slice=None):
        """
        base_img: a reference to the "base" image
        NB_img :  a dict containing references to the individual grayscale 
                  "secondary layers" images, if layer *output* is true
        chunk_slice: None, provided by the wrapping looping function
        """
        print "plotting", chunk_slice
        (ix, ixx, iy, iyy) = chunk_slice
        nx = self.fractal.nx
        ny = self.fractal.ny
        # box – The crop rectangle, as a (left, upper, right, lower)-tuple.
        crop_slice = (iy, nx-ixx, iyy, nx-ix)
        crop_slice = (ix, ny-iyy, ixx, ny-iy)
        chunk_2d = self.fractal.postproc_chunck(postproc_keys, chunk_slice,
                                                self.file_prefix)
        chunk_2d = self.apply_mask(chunk_2d, chunk_slice)

        # Now we render in color - remember to apply base data function
        i_base = len(self.calc_layers) # pure "calculation" layers
        rgb = self.colormap.colorize2d(self.base_data_function(
                                    chunk_2d[i_base, :, :]), self.probe_values)

        # now we render the layers
        i_NB_img = 0  # here the first...
        for i_layer, layer_options in enumerate(self.NB_layer):

            shade = chunk_2d[i_layer + 1 + i_base, :, :]
            shade_function = layer_options["layer_func"]
            data_min, data_max = layers_minmax[i_layer]
            qt_base = self.base_distrib(self.base_data_function(
                                    chunk_2d[i_base, :, :])) # distribution function
            shade = shade_function(shade, data_min, data_max, qt_base)
            shade = np.where(np.isnan(shade), 0.50, shade)

            if layer_options["output"]:
                paste_layer = PIL.Image.fromarray(self.np2PIL(
                                                  np.uint8(255 * shade)))
                NB_img[i_NB_img].paste(paste_layer, box=crop_slice)
                i_NB_img += 1
            shade = np.expand_dims(shade, axis=2)
            shade_type = layer_options.get("shade_type",
                              {"Lch": 4., "overlay": 4., "pegtop": 1.})
            rgb = Color_tools.blend(rgb, shade, shade_type)

        rgb = np.uint8(rgb * 255)
        paste = PIL.Image.fromarray(self.np2PIL(rgb))

        if self.mask is not None:
            # define if we need transparency...
            mask_channel_count = len(mask_color)
            has_transparency = (mask_channel_count > 3)
            if has_transparency:
                paste.putalpha(255)

            lx, ly = ixx-ix, iyy-iy
            crop_mask = PIL.Image.fromarray(self.np2PIL(
                                    np.uint8(255 * self.mask[ix:ixx, iy:iyy])))
            mask_colors = np.tile(np.array(mask_color), (lx, ly)).reshape(
                                                  [lx, ly, mask_channel_count])
            mask_colors = PIL.Image.fromarray(self.np2PIL(
                                                  np.uint8(255 * mask_colors)))
            paste.paste(mask_colors, (0, 0), crop_mask)

        base_img.paste(paste, box=crop_slice)


    @staticmethod
    def data_from_phase(Fourrier, phase):
        """
        Return physical data from phase
        Parameter:
        *Fourrier* (sin_coeff, cos_coeff) where
                    sin_coeff = [a1, a2, a3, ...]
                    cos_coeff = [b1, b2, b3, ...]
                    are the Fourrier coefficients
        """
        (sin_coeff, cos_coeff) = Fourrier
        ret = np.zeros(phase.shape, phase.dtype)
        for k, ak in enumerate(sin_coeff):
            ret += ak * np.sin((k + 1.) * phase)
        for k, bk in enumerate(cos_coeff):
            ret += bk * np.cos((k + 1.) * phase)
        return ret

    @staticmethod
    def skew(skewness, data):
        """
        Input : data between [-1, 1]
        output: data linearly rescaled between [min, max] so that :
            - min = -1 of max = +1
            - skewness = 0.5 * (min+max)
        """
        if abs(skewness) >= 1.:
            raise ValueError("Expected skew strictly between -1. and 1.")
        if skewness >= 0.:
            return 1. - (1. - data) * (1. - skewness)
        else:
            return -1. + (1. + data) * (1. + skewness)

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
    def intensify(intensity, data):
        """
        Adjust intensity for a np.array between -1 and 1
        Parameter:
        *intensity*  -1. < intensity < 1., scaling coefficient 
        """
        if np.abs(intensity) > 1.:
            raise ValueError(intensity)
        return data * intensity

    @staticmethod
    def bluring_coeff(blur1, blur2, blur3, x):
        """
        Define bluring coefficents for *x*, parameter blur1, blur2, blur3
        Paramters
        *blur1*, *blur2*, *blur3*    Monotonic, real inputs
        *x*                          Array
        
        Returns
        *y*   Arays of bluring coefficient at x location, as below if blur param
              are increasing :
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
            return Fractal_plotter.bluring_coeff(-blur1, -blur2, -blur3, -x)
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
    def blur(blur_ranges, data, qt_base):
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
            data = data * Fractal_plotter.bluring_coeff(
                                                  blur1, blur2, blur3, qt_base)
        return data

    def blend_plots(self, im_file1, im_file2, output_mode='RGB'):
        """
        Programatically merge 2 images taking into account transparency
        Image.alpha_composite(im, dest=(0, 0), source=(0, 0))
        """
        print "blending"
        im_path1 = os.path.join(self.plot_dir, im_file1)
        im_path2 = os.path.join(self.plot_dir, im_file2)
        im_path = os.path.join(self.plot_dir, "composite" + ".png")

        print "blend source 1", im_path1
        print "blend source 2", im_path2
        print "blend dest", im_path

        im1 = PIL.Image.open(im_path1).convert('RGBA')
        im2 = PIL.Image.open(im_path2).convert('RGBA')

        im = PIL.Image.alpha_composite(im1, im2).convert(output_mode)
        im.save(im_path)


class Fractal(object):
    """
    Abstract base class
    Derived classes are reponsible for computing, storing and reloading the raw
    data composing a fractal image.
    """

    def __init__(self, directory, x, y, dx, nx, xy_ratio, theta_deg,
                 chunk_size=200, complex_type=np.complex128):
        """
        Defines 
           - the pixels composing the image
           - The working directory
        """
        self.x = x
        self.y = y
        self.nx = nx
        self.dx = dx
        self.xy_ratio = xy_ratio
        self.theta_deg = theta_deg
        self.directory = directory
        self.chunk_size = chunk_size
        self.complex_type = complex_type
        self.complex_store_type = complex_type
        self.float_postproc_type = np.float32
        self.termination_type = np.int8
        self.int_type = np.int32

    @property
    def ny(self):
        return int(self.nx * self.xy_ratio + 0.5)

    @property
    def dy(self):
        return self.dx * self.xy_ratio

    @property    
    def params(self):
        return {"x": self.x,
                "y": self.y,
                "nx": self.nx,
                "ny": self.ny,
                "dx": self.dx,
                "dy": self.dy,
                "theta_deg": self.theta_deg }

    def data_file(self, chunk_slice, file_prefix):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        data chunk
        """
        return os.path.join(self.directory, "data",
                file_prefix + "_{0:d}-{1:d}_{2:d}-{3:d}.tmp".format(
                *chunk_slice))

    def save_data_chunk(self, chunk_slice, file_prefix,
                        params, codes, raw_data):
        """
        Write to a dat file the following data:
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.data_file(chunk_slice, file_prefix)
        mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            print "Data computed, saving", save_path
            pickle.dump(params, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(codes, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(raw_data, tmpfile, pickle.HIGHEST_PROTOCOL)

    def reload_data_chunk(self, chunk_slice, file_prefix, scan_only=False):
        """
        Reload arrays from a data file
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.data_file(chunk_slice, file_prefix)
        with open(save_path, 'rb') as tmpfile:
            params = pickle.load(tmpfile)
            codes = pickle.load(tmpfile)
            if scan_only:
                return params, codes
            raw_data = pickle.load(tmpfile)
        return params, codes, raw_data

    @staticmethod
    def loop_pixels(func):
        """
        func shall have *chunk_slice*, *c* as keyword arguments.
        If *chunk_slice* or *c* are None (or not provided):
        We loop over the image pixels with these 2 keyword-arguments:
            - chunk_slice
            - c
            (The other args and kwargs are passed untouched)
        If *chunk_slice* AND *c* are provided, simply the usual call to f.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            args = args[1:]
            if (kwargs.get("chunk_slice", None) is None
                ) and (kwargs.get("c", None) is None):
                for chunk_slice in self.chunk_slices():
                    c = self.c_chunk(chunk_slice)
                    kwargs["chunk_slice"] = chunk_slice
                    kwargs["c"] = c
                    func(self, *args, **kwargs)
            else:
                func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def dic_matching(dic1, dic2):
        """
        If we want to do some clever sanity test
        If not matching shall raise a ValueError
        """
        return

    def chunk_slices(self, chunk_size=None):
        """
        Genrator function
        Yields the chunks spans (ix, ixx, iy, iyy)
        with each chunk of size chunk_size x chunk_size
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        for ix in range(0, self.nx, chunk_size):
            ixx = min(ix + chunk_size, self.nx)
            for iy in range(0, self.ny, chunk_size):
                iyy = min(iy + chunk_size, self.ny)
                yield  (ix, ixx, iy, iyy)

    def c_chunk(self, chunk_slice, data_type=None):
        """
        Returns a chunk of c_vec for the calculation
        Parameters
         - chunk_span
         - data_type: expected one of np.float64, np.longdouble
        
        Returns: 
        span :  (ix, ixx, iy, iyy) The slice in the x and y  direction
        c_vec : [chunk_size x chunk_size] 1d-vec of type datatype
        
        """
        if data_type is None:
            select = {np.complex256: np.float128,
                      np.complex128: np.float64}
            data_type = select[self.complex_type]
        (x, y, nx, ny, dx, dy, theta_deg) = (self.x, self.y, self.nx, self.ny,
            self.dx, self.dy, self.theta_deg)
        (ix, ixx, iy, iyy) = chunk_slice

        theta = theta_deg / 180. * np.pi
        dx_grid = np.linspace(- dx * 0.5, dx * 0.5, num=nx, dtype=data_type)
        dy_grid = np.linspace(- dy * 0.5, dy * 0.5, num=ny, dtype=data_type) 

        dy_vec, dx_vec  = np.meshgrid(dy_grid[iy:iyy], dx_grid[ix:ixx])
        x_vec = x + (dx_vec * np.cos(theta)) - (dy_vec * np.sin(theta))
        y_vec = y + (dx_vec * np.sin(theta)) + (dy_vec * np.cos(theta))

        return x_vec + y_vec * 1j


    @loop_pixels.__func__
    def cycles(self, initialize, iterate, terminate, subset, codes,
               file_prefix, chunk_slice=None, c=None, pc_threshold=0.2):
        """
        Fast-looping for Julia and Mandelbrot sets computation.

        Parameters
        *initialize*  function(Z, U, c) modify in place Z, U (return None)
        *iter_func*   function(Z, U, c, n) modify place Z, U (return None)
        *terminate*   function (stop_reason, Z, U, c, n) modifiy in place
                      stop_reason (return None); if stop reason in range(nstop)
                      -> stopping iteration of this point.
        *subset*   bool arrays, iteration is restricted to the subset of current
                   chunk defined by this array. In the returned arrays the size
                    of axis ":" is np.sum(subset[ix:ixx, iy:iyy]) - see below
        *codes*  = complex_codes, int_codes, termination_codes
        *file_prefix* prefix identifig the data files
        *chunk_slice* None - provided by the looping wrapper
        *c*           None - provided by the looping wrapper

        Returns 
        None - save to a file
        
        - *raw_data* = (Z, U, stop_reason, stop_iter) where
            *chunk_mask*    1d mask
            *Z*             Final values of iterated complex fields shape [ncomplex, :]
            *U*             Final values of int fields [nint, :]       np.int32
            *stop_reason*   Byte codes -> reasons for termination [:]  np.int8
            *stop_iter*     Numbers of iterations when stopped [:]     np.int32
        """
        
        # First Tries to reload the data :
        try:
            (dparams, dcodes) = self.reload_data_chunk(chunk_slice,
                                                   file_prefix, scan_only=True)
            self.dic_matching(dparams, self.params)
            self.dic_matching(dcodes, codes)
            print "Data found, skipping calc: ", chunk_slice
            return
        except IOError:
            print "Unable to find data_file, computing"
        except:
            print sys.exc_info()[0], "\nUnmaching params, computing"
            
        # We did't find we need to compute
        (ix, ixx, iy, iyy) = chunk_slice
        c = np.ravel(c)
        if subset is not None:
            chunk_mask = np.ravel(subset[ix:ixx, iy:iyy])
            c = c[chunk_mask]
        else:
            chunk_mask = None

        (n_pts,) = c.shape
        
        n_Z, n_U, n_stop = (len(code) for code in codes)

        Z = np.zeros([n_Z, n_pts], dtype=c.dtype)
        U = np.zeros([n_U, n_pts], dtype=self.int_type)
        stop_reason = -np.ones([1, n_pts], dtype=self.termination_type) # np.int8 -> -128 to 127
        stop_iter = np.zeros([1, n_pts], dtype=self.int_type) 
        initialize(Z, U, c, chunk_slice)
        
        #  temporary array for looping
        index_active = np.arange(c.size, dtype=self.int_type)
        bool_active = np.ones(c.size, dtype=np.bool)
        Z_act = np.copy(Z)
        U_act = np.copy(U)
        stop_reason_act = np.copy(stop_reason)
        c_act = np.copy(c)
        
        looping = True
        n_iter = 0
        pc_trigger = [False, 0, 0]   #  flag, counter, count_trigger
        while looping:
            n_iter += 1

            # 1) Reduce of all arrays to the active part at n_iter
            if not(np.all(bool_active)):
                #print "select"
                c_act = c_act[bool_active]
                Z_act = Z_act[:, bool_active]
                U_act = U_act[:, bool_active]
                stop_reason_act = stop_reason_act[:, bool_active]
                index_active = index_active[bool_active]

            # 2) Iterate all active parts vec, delegate to *iterate*
            iterate(Z_act, U_act, c_act, n_iter)

            # 3) Partial loop ends, delagate to *terminate*
            terminate(stop_reason_act, Z_act, U_act, c_act, n_iter)
            for stop in range(n_stop):
                stopped = (stop_reason_act[0, :] == stop)
                index_stopped = index_active[stopped]
                stop_iter[0, index_stopped] = n_iter
                stop_reason[0, index_stopped] = stop
                Z[:, index_stopped] = Z_act[:, stopped]
                U[:, index_stopped] = U_act[:, stopped]
                if stop == 0:
                    bool_active = ~stopped
                else:
                    bool_active[stopped] = False

            count_active = np.count_nonzero(bool_active)
            pc = 0 if(count_active == 0) else count_active / (c.size *.01)

            if pc < pc_threshold and n_iter > 5000:
                if pc_trigger[0]:
                    pc_trigger[1] += 1   # increment counter
                else:
                    pc_trigger[0] = True # start couting
                    pc_trigger[2] = int(n_iter * 0.25) # set_trigger to 125%
                    print "Trigger hit at", n_iter
                    print "will exit at most at", pc_trigger[2] + n_iter

            looping = (count_active != 0 and
                       (pc_trigger[1] <= pc_trigger[2]))
            if n_iter%5000 == 0:
                reasons_pc = np.zeros([n_stop], dtype=np.float32)
                for stop in range(n_stop):
                    reasons_pc[stop] = np.count_nonzero(
                        stop_reason[0, :] == stop) /(c.size *.01)
                status = ("iter{0} : active: {1} - {2}% " +
                          "- by reasons (%): {3}").format(
                          n_iter, count_active, pc, reasons_pc)
                print status
        
        # Don't save temporary codes - i.e. those which startwith "_"
        # inside Z and U arrays
        select = {0: Z, 1: U}
        target = [None, None]
        save_codes = copy.deepcopy(codes)
        for icode, code in enumerate(save_codes):
            if icode == 2:
                break
            to_del = [ifield for ifield, field in enumerate(code)
                      if field[0] == "_"]
            target[icode] = np.delete(select[icode], to_del ,0)
            for index in to_del[::-1]:
                del code[index]
        save_Z, save_U = target

        raw_data = [chunk_mask, save_Z.astype(self.complex_store_type),
                    save_U, stop_reason, stop_iter]
        self.save_data_chunk(chunk_slice, file_prefix, self.params,
                             save_codes, raw_data)

    def raw_data(self, code, file_prefix):
        """
        Return an array of stored raw data from its code
        *code* one item of complex_codes, int_codes, termination_codes
        *file_prefix* prefix identifig the data files
        *chunk_slice* None - provided by the looping wrapper
        *c*           None - provided by the looping wrapper
        """
        for chunk_slice in self.chunk_slices():
            break
        (params, codes) = self.reload_data_chunk(chunk_slice,
                              file_prefix, scan_only=True)
        
        # Select the case
        dtype, kind = self.type_kind_from_code(code, codes)

        data = np.empty([self.nx, self.ny], dtype=dtype)
        self.fill_raw_data(data, code, kind, file_prefix)
        return data

    def type_kind_from_code(self, code, codes):
        """
        codes as returned by 
        (params, codes) = self.reload_data_chunk(chunk_slice,
                                                   file_prefix, scan_only=True)
        """
        complex_codes, int_codes, _ = codes
        if code in complex_codes:
            dtype, kind = self.complex_store_type, "complex"
        elif code in int_codes:
            dtype, kind = self.int_type, "int"
        elif code == "stop_reason":
            dtype , kind = self.termination_type, code
        elif code == "stop_iter":
            dtype , kind = self.int_type, code
        else:
            raise KeyError("raw data code unknow" + code)
        return dtype , kind


    @loop_pixels.__func__
    def fill_raw_data(self, data, code, kind, file_prefix,
                      chunk_slice=None, c=None, return_as_list=False):
        """
        *chunk_slice* None - provided by the looping wrapper
        *c*           None - provided by the looping wrapper
        """
        (ix, ixx, iy, iyy) = chunk_slice
        params, codes, raw_data = self.reload_data_chunk(chunk_slice,
                                                         file_prefix)
        complex_dic, int_dic, termination_dic = self.codes_mapping(*codes)
        chunk_mask, Z, U, stop_reason, stop_iter = raw_data
        select = {"complex": complex_dic, "int": int_dic}

        if kind == "complex":
            filler = Z[select[kind][code], :]
        elif kind == "int":
            filler = U[select[kind][code], :]
        elif code == "stop_reason":
            filler = stop_reason[0, :]
        elif code == "stop_iter":
            filler = stop_iter[0, :]
        else:
            raise KeyError("raw data code unknow")
        filler = filler[np.newaxis, :]
        filler = self.reshape2d(filler, chunk_mask, chunk_slice)
        if return_as_list:
            data += [filler] # only makes sense if not looping...
            return
        data[ix:ixx, iy:iyy] = filler[0, :, :]
        

    def reshape2d(self, chunk_array, chunk_mask, chunk_slice):
        """
        Returns 2d versions of the 1d stored vecs
               chunk_array of size (n_post, n_pts)
        
        # note : to get a 2-dimensionnal vec do:
                 if bool_mask is not None:
                 we need to inverse
                     chunk_mask = np.ravel(subset[ix:ixx, iy:iyy])
                     c = c[chunk_mask]
        """
        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy

        n_post, n_pts = chunk_array.shape
        if chunk_mask is None:
            chunk_2d = np.copy(chunk_array)
        else:
            indices = np.arange(nx * ny)[chunk_mask]
            chunk_2d = np.empty([n_post, nx * ny], dtype=chunk_array.dtype)
            chunk_2d[:] = np.nan
            chunk_2d[:, indices] = chunk_array

        return np.reshape(chunk_2d, [n_post, nx, ny])


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
        post_array, chunk_mask = self.postproc(postproc_keys, codes, raw_data,
                                               chunk_slice)
        return self.reshape2d(post_array, chunk_mask, chunk_slice)


    def postproc(self, postproc_keys, codes, raw_data, chunk_slice,
                 dtype=None):
        """
        Computes secondary fields from raw data "catalogue of preferred postproc"
        Parameters
        *postproc_keys*    identifiers of the data expected as output
                           (post_name, post_dic) for each key 
        *codes*        identifiers of the raw_res given as input
        *raw_res*      the raw data in input identified by codes
        Returns
        *post_array*      np.float32 [n_keys, n_pts]
        
        called by :
        post_array[ix:ixx, iy:iyy, :] = self.postproc(postproc_keys, codes, 
                                                      raw_data)
                            
        Available (post_name, post_dic) options :
        post_name         |   post_dic
        [potential]       | {"kind": "infinity",
                          | "d": degree of polynome d >= 2,
                          |  "a_d": coeff of higher order monome
                          |  "N": High number defining a boundary of \infty}
                          | {"kind": "convergent",
                          |  "epsilon_cv": Small number defining boundary of 0}
                          |  {"kind": "transcendent" in this cas we just return n}
                          |
        [DEM_shade]       |{"kind": ["Milnor" / "potential" / "convergent"], 
                          | "theta_LS":, "phi_LS":, "shininess":, 
                          | "ratio_specular":}
                          |
        ["attr_shade"]    |[{"theta_LS":, "phi_LS":, "shininess":, 
                          |"ratio_specular":} 
        [power_attr_shade]| Variation on ["attr_shade"] 
                          |
        [field_lines]     |{} # Calculation of a C1 phase angle (modulo 2 pi)
        [Lyapounov]       |{"n_iter": [field / None]}
        [raw]             | {"code": "raw_code"}
        [phase]           | {"source": "raw_code"}
        [minibrot_phase]  |{"source": "raw_code"}  same than *phase* but with
                          |                        smoothing around 0
        [abs]             | {"source": "raw_code"}
                            
        """
        chunk_mask, Z, U, stop_reason, stop_iter = raw_data
        complex_dic, int_dic, termination_dic = self.codes_mapping(*codes)

        n_pts = Z.shape[1]  # Z of shape [n_Z, n_pts]

        if dtype is None:
            dtype = self.float_postproc_type
        post_array = np.empty([len(postproc_keys), n_pts], dtype=dtype)

        # The potential & real iteration number is a prerequisite for some
        # other postproc...
        has_potential = False

        for i_key, postproc_key in enumerate(postproc_keys):
            post_name, post_dic = postproc_key

            if post_name == "potential":
                has_potential = True
                potential_dic = post_dic
                n = stop_iter
                zn = Z[complex_dic["zn"], :]

                if potential_dic["kind"] == "infinity":
                    d = potential_dic["d"]
                    a_d = potential_dic["a_d"]
                    N = potential_dic["N"]
                    k = np.abs(a_d) ** (1. / (d - 1.))
                    # k normaliszation corefficient, because the formula given
                    # in https://en.wikipedia.org/wiki/Julia_set                    
                    # suppose the highest order monome is normalized
                    nu_frac = -(np.log(np.log(np.abs(zn * k)) / np.log(N * k))
                                / np.log(d))


                elif potential_dic["kind"] == "convergent":
                    eps = potential_dic["epsilon_cv"]
                    alpha = Z[complex_dic["alpha"]]
                    z_star = Z[complex_dic["zr"]]
                    nu_frac = - np.log(eps / np.abs(zn - z_star)
                                       ) / np.log(np.abs(alpha))

                elif potential_dic["kind"] == "transcendent":
                    # Not possible to define a proper potential for a 
                    # transcencdent fonction
                    nu_frac = 0.

                else:
                    raise NotImplementedError("Potential 'kind' unsupported")
                nu = n + nu_frac
                val = nu


            elif post_name == "DEM_shade":
                if not has_potential:
                    raise ValueError("Potential shall be defined before shade")
                dzndc = Z[complex_dic["dzndc"], :]
                color_tool_dic = post_dic.copy()
                shade_kind = color_tool_dic.pop("kind")

                if shade_kind == "Milnor":   # use only for z -> z**2 + c
# https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                    d2zndc2 = Z[complex_dic["d2zndc2"], :]
                    lo = np.log(np.abs(zn))
                    normal = zn * dzndc * ((1+lo)*np.conj(dzndc * dzndc)
                                  -lo * np.conj(zn * d2zndc2))
                    normal = normal / np.abs(normal)

                elif shade_kind == "potential": # safer
                    if potential_dic["kind"] == "infinity":
                    # pulls back the normal direction from an approximation of 
                    # potential: phi = log(zn) / 2**n 
                        normal = zn / dzndc
                        normal = normal / np.abs(normal)

                    elif potential_dic["kind"] == "convergent":
                    # pulls back the normal direction from an approximation of
                    # potential (/!\ need to derivate zr w.r.t. c...)
                    # phi = 1. / (abs(zn - zr) * abs(alpha)) 
                        zr = Z[complex_dic["zr"]]
                        dzrdc = Z[complex_dic["dzrdc"]]
                        normal = - (zr - zn) / (dzrdc - dzndc)
                        normal = normal / np.abs(normal)

                val = Color_tools.shade_layer(
                        normal * np.cos(post_dic["phi_LS"] * np.pi / 180.),
                        **color_tool_dic)


            elif post_name == "field_lines":
                if not has_potential:
                    raise ValueError("Potential shall be defined before " +
                                     "stripes")
                if potential_dic["kind"] == "infinity":
                    # Hopefully C1-smooth phase angle for z -> a_d * z**d
                    # G. Billotey
                    k_alpha = 1. / (1. - d)       # -1.0 if N = 2
                    k_beta = - d * k_alpha        # 2.0 if N = 2
                    z_norm = zn * a_d ** (-k_alpha) # the normalization coeff

                    t = [np.angle(z_norm)]
                    val = np.zeros_like(t)
                    for i in range(1, 6):
                        t += [d * t[i - 1]]
                        dphi = - i * np.pi * 0.
                        val += (0.5)**i  * (np.sin(t[i-1] + dphi) + (
                                  k_alpha + k_beta * d**nu_frac) * (
                                  np.sin(t[i] + dphi) - np.sin(t[i-1] + dphi)))
                    val = np.arcsin(val)
                    del t

                elif potential_dic["kind"] == "convergent":
                    alpha = Z[complex_dic["alpha"]]
                    z_star = Z[complex_dic["zr"]]
                    beta = np.angle(alpha)
                    val = np.angle(z_star - zn) - nu_frac * beta


            elif post_name == "Lyapounov":
                dzndz = Z[complex_dic["dzndz"], :]
                if post_dic["n_iter"] is None:
                    n = stop_iter
                else:
                    code = post_dic["n_iter"]
                    n = U[int_dic[code], :]
                liap = np.log(np.abs(dzndz)) / (n)
                val = liap


            elif post_name == "attr_shade":
                # attr is normalized between 0. and 1.
                # So here we can pull back not only the radial direction but a
                # full "3d" normal vec.
                attr = Z[complex_dic["attractivity"], :]
                dattrdc = Z[complex_dic["dattrdc"], :]
                normal = attr * np.conj(dattrdc) / np.abs(dattrdc)

                color_tool_dic = post_dic.copy()
                val = Color_tools.shade_layer(
                        normal,
                        **color_tool_dic)

            elif post_name == "power_attr_shade":
                # This is a trick to get a finer 'petals' details for
                # transcendent function (like the power-tower)
                # We apply a mix between radial direction pull-back and full
                # normal
                attr = Z[complex_dic["attractivity"], :]
                invalid = (np.abs(attr) > 1.)
                attr = np.where(invalid, attr / np.abs(attr), attr)

                dattrdc = Z[complex_dic["dattrdc"], :]
                dattrdc = np.where(invalid, 0., dattrdc)
                normal = (.2 + .8 * attr) * np.conj(dattrdc) / np.abs(dattrdc)

                color_tool_dic = post_dic.copy()
                val = Color_tools.shade_layer(
                        normal * np.cos(post_dic["phi_LS"] * np.pi / 180.),
                        **color_tool_dic)     

            elif post_name == "phase":
                # Plotting raw complex data : plot the angle
                code = post_dic["source"]
                z = Z[complex_dic[code], :]
                val = np.angle(z)

            elif post_name == "abs":
                # Plotting raw complex data : plot the abs
                code = post_dic["source"]
                z = Z[complex_dic[code], :] 
                val = np.abs(z)         

            elif post_name == "minibrot_phase":
                # Pure graphical post-processing for minibrot 'attractivity'
                # The phase 'vanihes' at the center
                code = post_dic["source"]
                z = Z[complex_dic[code], :]
                val = np.cos(np.angle(z) * 24) * np.abs(z)**8

            elif post_name == "raw":
                code = post_dic["code"]
                _ , kind = self.type_kind_from_code(code, codes)
                if kind == "complex":
                    raise ValueError("Complex, use \"phase\" or \"abs\"")
                elif kind == "int":
                    val = U[int_dic[code], :]
                elif code == "stop_reason":
                    val = stop_reason[0, :]
                elif code == "stop_iter":
                    val = stop_iter[0, :]
                else:
                    raise KeyError("raw data code unknow")
            else:
                    raise ValueError("Unknown post_name", post_name)

            post_array[i_key, :] = val

        return post_array, chunk_mask


    @staticmethod
    def codes_mapping(complex_codes, int_codes, termination_codes):
        """
        Utility function, returns the inverse mapping code  -> int
        """
        complex_dic, int_dic, termination_dic = [dict(
            zip(tab, range(len(tab)))) for tab in [
            complex_codes, int_codes, termination_codes]]
        return complex_dic, int_dic, termination_dic

    @staticmethod
    def subsubset(bool_set, bool_subset_of_set):
        """
        Returns boolean array for a subset
        Parameters    
         - *bool_set* bool array of shape N, defines a set 
         - *bool_subset_of_set* bool array of shape Card(set)
        Returns
         - *bool_subset* bool array of shape N
        """
        set_count = np.sum(bool_set)
        Card_set, = np.shape(bool_subset_of_set)
        if Card_set != set_count:
            raise ValueError("Expected bool_subset_of_set of shape [Card(set)]"
                             )
        bool_subset = np.copy(bool_set)
        bool_subset[bool_set] = bool_subset_of_set
        return bool_subset