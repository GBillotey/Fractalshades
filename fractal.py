# -*- coding: utf-8 -*-
import numpy as np
import mpmath
from numpy_utils.xrange import Xrange_array, mpc_to_Xrange, mpf_to_Xrange
import pickle

import matplotlib.colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import os, errno, sys
import fnmatch
import functools
import copy
import PIL
#from PIL.PngImagePlugin import PngInfo
#from perturbation import PerturbationFractal

import multiprocessing
import tempfile

import datetime

enable_multiprocessing = True
no_compute = False

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


def process_init(job, redirect_path):
    """
    Save *job* as a global for the child-process ; redirects stdout and stderr.
    -> 'On Unix a child process can make use of a shared resource created in a
    parent process using a global resource. However, it is better to pass the
    object as an argument to the constructor for the child process.'
    """
    global process_job
    process_job = job
    mkdir_p(redirect_path)
    out_file = str(os.getpid())
    sys.stdout = open(os.path.join(redirect_path, out_file + ".out"), "a")
    sys.stderr = open(os.path.join(redirect_path, out_file + ".err"), "a")

def job_proxy(key):
    global process_job
    return process_job(key)

class Multiprocess_filler():
    def __init__(self, iterable_attr, res_attr=None, redirect_path_attr=None,
                 iter_kwargs="key", veto_multiprocess=False):
        """
        Decorator class for an instance-method *method*

        *iterable_attr* : string, getattr(instance, iterable_attr) is an
            iterable.
        *res_attr* : string or None, if not None (instance, res_attr) is a
            dict-like.
        *redirect_path_attr* : string or None. If veto_multiprocess is False,
            getattr(instance, redirect_path_attr) is the directory where
            processes redirects sys.stdout and sys.stderr.
        veto_multiprocess : if True, defaults to normal iteration without
            multiprocessing

        Usage :
        @Multiprocess_filler(iterable_attr, res_attr, redirect_pathattr,
                             iter_kwargs)
        def method(self, *args, iter_kwargs=None, **otherkwargs):
            (... CPU-intensive calculations ...)
            return val

        - iter_kwargs will be filled with successive values yield by *iterable*
            and successive outputs calculated by multiprocessing.cpu_count()
            child-processes,
        - these outputs will be pushed in place to res in parent process:
            res[key] = val
        - the child-processes stdout and stderr are redirected to os.getpid()
            files in subdir *redirect_path* (extensions .out, in)
        """
        self.iterable = iterable_attr
        self.res = res_attr
        self.redirect_path_attr = redirect_path_attr
        self.iter_kwargs = iter_kwargs
        self.veto_multiprocess = veto_multiprocess

    def __call__(self, method):
        @functools.wraps(method)
        def wrapper(instance, *args, **kwargs):
            iterable = getattr(instance, self.iterable)
            res = None
            if self.res is not None:
                res = getattr(instance, self.res)
            def job(key):
                kwargs[self.iter_kwargs] = key
                return method(instance, *args, **kwargs)

            if enable_multiprocessing and not(self.veto_multiprocess):
                print("Launch Multiprocess_filler of ", method.__name__)
                print("cpu count:", multiprocessing.cpu_count())
                redirect_path = getattr(instance, self.redirect_path_attr)
                with multiprocessing.Pool(initializer=process_init,
                        initargs=(job, redirect_path),
                        processes=multiprocessing.cpu_count()) as pool:
                    worker_res = {key: pool.apply_async(job_proxy, (key,))
                                  for key in iterable()}
                    for key, val in worker_res.items():
                        if res is not None:
                            res[key] = val.get()
                        else:
                            val.get()
                    pool.close()
                    pool.join()
            else:
                for key in iterable():
                    if res is not None:
                        res[key] = job(key)
                    else:
                        job(key)
        return wrapper


class Fractal_Data_array():
    def __init__(self, fractal, file_prefix=None, postproc_keys=None,
                 mode="r_raw"):
        """
        mode :
            r+raw          direct read of stored file (from file_prefix)
            r+postproc     postproc layer on stored file (from file_prefix)
            rw+temp        tempfile.SpooledTemporaryFile (file_prefix ignored)   

        postproc_keys:
            mode r_raw : tuple (code, None) or (code, function)
            mode r_postproc : as expeted by Fractal.postproc
            mode rw_temp : ignored
        """
        self.fractal = fractal
        self.file_prefix = file_prefix
        self.postproc_keys = postproc_keys
        self.mode = mode

        self._ref = None
        self._kind = None

    @property
    def kind(self):
        """ lazzy evaluation of the data type on first chunk, for r_raw mode
        """
        if self._kind is not None:
            return self._kind
        else:
            fractal = self.fractal
            chunk_slice = next(fractal.chunk_slices())
            (params, codes) = fractal.reload_data_chunk(
                    chunk_slice, self.file_prefix, scan_only=True)
            kind = fractal.kind_from_code(self.code, codes)
            self._kind = kind
            return kind

    def __getitem__(self, chunk_slice):
        """
        Returns chunk_item dentified by key "chunk_slice"
        Returns:
            np array of shape (nfields, chunk, chunk)
        """
        fractal = self.fractal
        mode = self.mode

        if mode == "r+raw":
            self.code, func = self.postproc_keys
            ret = fractal.get_raw_data(self.code, self.kind, self.file_prefix,
                chunk_slice)
            if func is None:
                return ret
            else:
                return func(ret)
        elif mode == "r+postproc":
            return fractal.postproc_chunck(self.postproc_keys,
                chunk_slice, self.file_prefix)
        elif mode == "rw+temp":
            subarray_file = self._ref[chunk_slice]
            _ = subarray_file.seek(0)
            # Note : no "extended_range" magic... to be handled downstream
            # if accessing raw Z values
            return np.load(subarray_file)
        else:
            raise ValueError("Unexpected read call for mode : " + 
                             "{}".format(mode))

    def __setitem__(self, chunk_slice, val):
        """
        Writes subarray *val* to temporary file.
        """
        if self.mode == "rw+temp":
            if self._ref is None:
                self._ref = dict()
            self._ref[chunk_slice] = tempfile.SpooledTemporaryFile(mode='w+b')
            to_disk = False
            if to_disk:
                self._ref[chunk_slice].fileno()
            np.save(self._ref[chunk_slice], val)
        else:
            raise ValueError("Unexpected write call for mode: " + 
                             "{}".format(self.mode))

    def __invert__(self):
        """ Allows use of the ~ notation """
        if self.mode != "r+raw":
            raise ValueError("~ operator not defined for this mode{}".format(
                             self.mode))
        code, func = self.postproc_keys
        if func is None:
            inv_func = lambda x: ~x
        else:
            inv_func = lambda x: ~func(x)
        return Fractal_Data_array(self.fractal, self.file_prefix,
                postproc_keys=(code, inv_func), mode=self.mode)

    def free(self):
        """ Closes the SpooledTemporaryFiles """
        if self.mode == "rw+temp":
            if self._ref is not None:
                for val in self._ref.values():
                    val.close()
                    
    def __iter__(self):
        """
        Iterate the Fractal_Data_array by chunks.
        Note: Should make min, max just work.
        """
        for chunk_slice in self.fractal.chunk_slices():
            yield self[chunk_slice]
            
    def nanmax(self):
        """ extension of np.nanmax """
        return max([np.nanmax(chunk) for chunk in self])
    
    def nanmin(self):
        """ extension of np.nanmin """
        return max([np.nanmin(chunk) for chunk in self])
    
    def nansum(self):
        """ extension of np.nansum """
        return sum([np.nansum(chunk) for chunk in self])



class Fractal_colormap():
    """
    Class responsible for mapping a real array to a colors array.
    Attributes :
        *_colors* Internal list of possible colors, [0 to self.n_colors]
        *_probes* list of indices in self._colors array, identifying the
                  transitions between differrent sections of the colormap. Each
                  of this probe is mapped to *_probe_value*, either given by the
                  user or computed at plot time.
    """
    def __init__(self, color_gradient, colormap=None, extent="clip"):
        """
        Creates from : 
            - Directly a color gradient array (as output by Color_tools
              gradient functions, array of shape (n_colors, 3))
            - A matplotlib colormap ; in this case color_gradient shall be a
              list of the form (start, stop, n _colors)
*color_gradient*  Either: a Color gradient array from COlortool class, or if a
maltplotlib colormap is provided at second input, a tuple (xmin, xmax,
n_colors) where xmi, xmax refer to this colormap (xmin xmax btw. 0 and 1).
*colormap*  None or a matplotlib colormap
*extent*    What to do with out of range Values. "clip" or "mirror"
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
        self.extent = extent

    @property
    def n_colors(self):
        """ Total number of colors in the colormap. """
        return self._colors.shape[0]

    @property
    def probes(self):
        """ Position of the "probes" ie transitions between the different parts
        of the colormap. Read-only. """
        return np.copy(self._probes)

    def __neg__(self):
        """
        Returns a reversed colormap
        """
        other = Fractal_colormap(self._colors[::-1, :])
        other._probes = self._probes[-1] - self._probes[::-1]
        return other
    
    def __add__(self, other):
        """ Concatenates 2 Colormaps """
        fcm = Fractal_colormap(np.vstack([self._colors, other._colors]))
        fcm._probes = np.concatenate([self._probes,
            (other._probes + self._probes[-1] + 1)[1:]])
        return fcm

    def __sub__(self, other):
        """ Sbstract a colormap ie adds its reversed version """
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

    def colorize(self, z, probes_z):
        """
        Returns a color array bazed on z values
        """
        nx, ny = z.shape
        z = np.ravel(z)
        z = self.normalize(z, probes_z)
        # linear interpolation in sorted color array
        indices = np.searchsorted(np.arange(self.n_colors), z)
        alpha = indices - z
        search_colors = np.vstack([self._colors[0, :],
                                   self._colors,
                                   self._colors[-1, :]])
        z_colors = (alpha[:, np.newaxis] * search_colors[indices, :] + 
             (1.-alpha[:, np.newaxis]) * search_colors[indices + 1, :])
        return np.reshape(z_colors, [nx, ny, 3])  
    
    def greyscale(self, z, probes_z):
        """
        Returns a color array bazed on z values
        """
        z = self.normalize(z, probes_z)
        # linear interpolation in sorted greyscale array
        indices = np.searchsorted(np.arange(self.n_colors), z)
        alpha = indices - z
        search_colors = np.concatenate([
            [0.], np.linspace(0., 1., self.n_colors),  [1.]])
        z_greys = (alpha * search_colors[indices] +
                   (1. - alpha) * search_colors[indices + 1])
        return z_greys

    def normalize(self, z, probes_z):
        """
        Normalise z , z -> z* so that z*min = 0 / z*max = self.n_colors
        *z*  array to normalise
        *probes_z* array of dim self.n_probes: values of z at probes
        """
        if np.any(probes_z.shape != self._probes.shape):
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


class Fractal_plotter():
    def  __init__(self, fractal, base_data_key, base_data_prefix, 
            base_data_function, colormap, probes_val, probes_kind="qt",
            mask=None):
        """
        Parameters
*fractal* The plotted fractal object

*base_data_key* key wich identifies a prosprocessed field from Fractal
                will be used as the "base source" of color.
                (post_name, post_dic) for just 1 key here.
*base_data_prefix* The prefix for the files were are stored the base data
                   arrays (on file per 'chunk' ie calculation tile).
*base_data_function* function which will be applied as a last postproc step to
the array returned by base_data_key
*colormap* a Fractal_colormap object used for the float to color mapping

*probes_qt*   Fractiles of the total image area, used to compute
    the mapping of the colormap probes ie *probe_values*.
    User can also impose directly the *probes_z* (e.g. movies or
    compositing of several images). If probe_fractiles, will default to
    probe_values (which cannot be None).

*probes_kind*  One of "qt" of "z"

*mask*  bool Fractal_Data_array, where True the point will be flagged
        as masked (a special color can be applied to masked points in
        subsequent plotting operations)
        """
        self.fractal = fractal
        self.base_data_key = base_data_key
        self.file_prefix = base_data_prefix
        self.base_data_function = base_data_function

        self.colormap = colormap
        
        self.probes_kind = probes_kind
        if probes_kind == "qt":
            self.probes_qt = np.asanyarray(probes_val)
        elif probes_kind == "z": 
            self.probes_z = np.asanyarray(probes_val)

        self.mask = mask
        self.calc_layers = []
        self.grey_layers = []
        self.normal_layers = []
        self.plot_mask = False
        self.plot_zoning = False

        # Attibutes used for multiprocessing loops
        self.plot_dir = self.fractal.directory
        self.multiprocess_dir = os.path.join(self.fractal.directory,
                                             "plot_multiproc")
        self.chunk_slices = self.fractal.chunk_slices

        # Calculated attributes used during plot
        self.base_min = np.inf
        self.base_max = -np.inf
        self.hist = None
        self.bins = None


    def add_calculation_layer(self, postproc_key):
        """
        Adds an intermediate calculation step, which might be needed for 
        postprocessing. No layer image is output.
        """
        self.calc_layers += [postproc_key] # field as postprocessed by Fractal

    def add_normal_map(self, postproc_key):
        """
        Adds a colored normal layer following OpenGL normal map format.
        """
        post_name, post_dic = postproc_key
        self.normal_layers += [
                {"postproc_keys":  [(post_name + "x", post_dic),
                                    (post_name + "y", post_dic)]}]

    def add_grey_layer(self, postproc_key, veto_blend=False, normalized=False,
                     Fourrier=None, skewness=None, hardness=None,
                     intensity=None, blur_ranges=None, shade_type=None,
                     disp_layer=False, layer_mask_color=(0,),
                     user_func=None, clip_min=None, clip_max=None):
        """
        Adds a greyscale layer.

        postproc_key : the key to retrieve a prosprocessed field from Fractal
                          (post_name, post_dic) for just 1 key here.
        veto_blend:  Boolean, if true the layer will be output to a greyscale
            image but NOT blended with the rgb base image to give the final
            image. Note that irrelative to veto_blend, a displacement layer
            (see doc for disp_layer param) is intended for further
            post-processing and never blended to the final image.
        Fourrier, skewness, hardness, intensity, blur_ranges : arguments passed
            to further image post-procesing functions, see details in each
            function descriptions.
        shade_type: If None, default to {"Lch": 4., "overlay": 4., "pegtop": 1.})
            Passed to Color_tools.blend when blending to the color image.
        disp_layer: If True will use 32-bit signed integer pixels for the image
            output (instead of 8 bits) to allow further 3d postprocessing.
            Displacement layers are NOT blended with the final rgb image.
        layer_mask_color: Only applied in case of "disp" layer, otherwise no
            real interest as will be anyhow mixed with color rgb mask.
        user_func: function which will be applied as a last postproc step to
            the array returned by postproc_key
        clip_min, clip_max: Imposed cliping range. If None, the full range will
            be considered.
        """
        def func(data, data_min, data_max, blur_base):
            """
            The function that will be applied on successive crops.
            data_min, data_max cannot be evaluated locally and need to be passed
            qt_base is the distribution function applied to the base levels
                    (ie quantiles)
            """
            # First if data is given as a phase, generate the 'physical' data
            if Fourrier is not None:
                data = self.data_from_phase(Fourrier, data)

            if user_func is not None:
                data = user_func(data)
            
            # renormalise between -1 and 1
            if not(normalized):
                data = (2 * data - (data_max + data_min)) / (
                        data_max - data_min)
                # Clipping in case of imposed min max range
                data = np.where(data < -1., -1., data)
                data = np.where(data > 1., 1., data)


            if skewness is not None:
                data = self.skew(skewness, data)
            if hardness is not None:
                data = self.harden(hardness, data)
            if intensity is not None:
                data = self.intensify(intensity, data)
            if blur_ranges is not None:
                data = self.blur(blur_ranges, data, blur_base)
            data = 0.5 * (data + 1.)
            return data

        self.grey_layers += [{
            "postproc_key": postproc_key, # field as postprocessed by Fractal
            "layer_func": func,   # will be applied to the postprocessed field
            "veto_blend": veto_blend,
            "Fourrier": Fourrier,
            "user_func": user_func,
            "shade_type": shade_type,
            "disp_layer": disp_layer,
            "layer_mask_color": layer_mask_color,
            "clip_min": clip_min,
            "clip_max": clip_max
            }]

    def add_mask_layer(self):
        """
        Adds a black and white boolean layer based on mask.
        """
        self.plot_mask = True

    def add_zoning_layer(self):
        """ Adds a greyscale layers based on colormap mapping """
        self.plot_zoning = True

    def apply_mask(self, array, chunk_slice):
        """
        Apply the 2d self.mask to the array
        array shape expected [:, :] or [n, :, :]
        """
        #(ix, ixx, iy, iyy) = chunk_slice
        sh = array.shape
        if self.mask is not None:
            if len(sh) == 2:
                return np.where(self.mask[chunk_slice], np.nan, array)
            elif len(sh) == 3:
                mask = self.mask[chunk_slice][np.newaxis, :, :]
                return np.where(mask, np.nan, array)
        return array

    def precompute_plot(self):
        """
        Precompute the data-fields needed for the plot, use a data-wrapper of
        type Fractal_Data_array
        """
        postproc_keys = list(self.calc_layers)
        postproc_keys += [self.base_data_key]
        for i_layer, layer_options in enumerate(self.grey_layers):
            postproc_keys += [layer_options["postproc_key"]]
        for i_layer, layer_options in enumerate(self.normal_layers):
            postproc_keys += layer_options["postproc_keys"]
        
        fractal_data = Fractal_Data_array(self.fractal, mode="r+postproc",
                    file_prefix=self.file_prefix, postproc_keys=postproc_keys)
        self._data = Fractal_Data_array(self.fractal, mode="rw+temp")
        self.precompute_chunk(fractal_data)

    @Multiprocess_filler(iterable_attr="chunk_slices", res_attr="_data",
        redirect_path_attr="multiprocess_dir", iter_kwargs="chunk_slice")
    def precompute_chunk(self, fractal_data, chunk_slice=None):
        return fractal_data[chunk_slice]


    def plot(self, file_name, transparency=False, mask_color=(0., 0., 0.)):
        """
        file_name
        transparency: used for 
        
        Note: Pillow image modes:
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        P (8-bit pixels, mapped to any other mode using a color palette)
        I (32-bit signed integer pixels)
        """
        self.precompute_plot()

        # Compute base layer histogram
        self.compute_baselayer_minmax()
        self.compute_baselayer_hist()
        
        self.colormap.output_png(self.plot_dir, file_name)

        mode = "RGB"
        if (len(mask_color) > 3) and (self.mask is not None):
            mode = "RGBA"

        nx, ny = self.fractal.nx, self.fractal.ny
        base_img = PIL.Image.new(mode=mode, size=(nx, ny), color=0)

        postproc_keys = []
        grey_img = []
        postproc_keys = list(self.calc_layers)
        postproc_keys += [self.base_data_key]

        for i_layer, layer_options in enumerate(self.grey_layers):
            postproc_keys += [layer_options["postproc_key"]]
#            if layer_options["output"]:
            layer_mode = {True: "I",
                          False: "P"}[layer_options["disp_layer"]]
            grey_img += [PIL.Image.new(mode=layer_mode,
                                     size=(nx, ny), color=0)]

        layers_minmax = [[np.inf, -np.inf] for layer in self.grey_layers]
        self.compute_greylayers_minmax(layers_minmax)#, postproc_keys)

        # Magic trick if user imposed min and max
        layers_computed_minmax = copy.copy(layers_minmax)
        layers_imposed_minmax = [[layer_options.get("clip_min", None),
                                  layer_options.get("clip_max", None)
                                  ] for layer_options in self.grey_layers]
        for i_layer, _ in enumerate(self.grey_layers):
            layers_minmax[i_layer] = [
                    layers_computed_minmax[i_layer][0] if 
                      layers_imposed_minmax[i_layer][0] is None else
                      layers_imposed_minmax[i_layer][0],
                    layers_computed_minmax[i_layer][1] if 
                      layers_imposed_minmax[i_layer][1] is None else
                      layers_imposed_minmax[i_layer][1]] 

        # for the 'blender normal' type, no need of layer min max
        # only outputing to separeted image
        normal_img = []
        for i_layer, layer_options in enumerate(self.normal_layers):
            postproc_keys += layer_options["postproc_keys"]
            normal_img += [PIL.Image.new(mode="RGB", size=(nx, ny))]
            
        # Otional mask layer
        mask_img = None
        if self.plot_mask:
            mask_mode = "1"
            mask_img = PIL.Image.new(mode=mask_mode, size=(nx, ny), color=0)
            
        # Optional "zoning" layer
        zoning_img = None
        if self.plot_zoning:
            zoning_mode = "P"
            zoning_img = PIL.Image.new(mode=zoning_mode, size=(nx, ny), color=0)

        self.plot_cropped(base_img, grey_img, normal_img, mask_img, zoning_img,
                          postproc_keys, mask_color, layers_minmax)

        base_img_path = os.path.join(self.plot_dir, file_name + ".png")
#        tag_dicts = 
        self.save_tagged(base_img, base_img_path,
                          {"Software": "py-fractal",
                           "fractal": type(self.fractal).__name__,
                           "projection": self.fractal.projection,
                           "x": repr(self.fractal.x),
                           "y": repr(self.fractal.y),
                           "dx": repr(self.fractal.dx),
                           "theta_deg": repr(self.fractal.theta_deg)
                           })

        if self.plot_mask:
            mask_img.save(os.path.join(self.plot_dir, file_name + ".mask.png"))
        if self.plot_zoning:
            zoning_img.save(os.path.join(self.plot_dir, file_name +
                                         ".zoning.png"))

        i_grey_img = 0
        for i_layer, layer_options in enumerate(self.grey_layers):
            #if layer_options["output"]:
            file_img = (file_name + "_" + 
                        layer_options["postproc_key"][0] + 
                        ".layer" + str(i_grey_img) + ".png")
            file_minmax = (file_name + "_" + 
                        layer_options["postproc_key"][0] + 
                        ".layer" + str(i_grey_img) + ".minmax")
            grey_img[i_grey_img].save(os.path.join(self.plot_dir, file_img))
            i_grey_img += 1
            with open(os.path.join(self.plot_dir, file_minmax), 'w'
                      ) as minmax_file:
                str_format = "{} / {}\n"
                minmax_file.write("imposed min max:\n")
                minmax_file.write(str_format.format(
                        *layers_imposed_minmax[i_layer]))
                minmax_file.write("computed min max:\n")
                minmax_file.write(str_format.format(
                        *layers_computed_minmax[i_layer]))
                minmax_file.write("final min max:\n")
                minmax_file.write(str_format.format(
                        *layers_minmax[i_layer]))

        i_normal_img = 0
        for i_layer, layer_options in enumerate(self.normal_layers):
            key = layer_options["postproc_keys"][0][0][:-1]
            normal_img[i_normal_img].save(os.path.join(self.plot_dir,
                    file_name + "_" + key + 
                    ".nmap" + str(i_grey_img) + ".png"))
            i_normal_img += 1

        # let's free some RAM
        self._data.free()

    def save_tagged(self, img, img_path, tag_dict):
        """
        Saves *img* to png format at *path*, tagging with *tag_dict*.
        https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_PNG_files
        """
        pnginfo = PIL.PngImagePlugin.PngInfo()
        for k, v in tag_dict.items():
            pnginfo.add_text(k, v)
        img.save(img_path, pnginfo=pnginfo)


    @Multiprocess_filler(iterable_attr="chunk_slices",
            iter_kwargs="chunk_slice", veto_multiprocess=True)
    def compute_greylayers_minmax(self, layers_minmax, chunk_slice=None):
        """
        Computes the min & max for the grey layers
        """
        if chunk_slice is None:
            return

        chunk_2d = self._data[chunk_slice]
        chunk_2d = self.apply_mask(chunk_2d, chunk_slice)

        i_base = len(self.calc_layers)
        for i_layer, layer_options in enumerate(self.grey_layers):
            data_layer = chunk_2d[i_layer + i_base + 1, :, :]
            if layer_options["Fourrier"] is not None:
                data_layer = self.data_from_phase(layer_options["Fourrier"],
                                                  data_layer)
            if layer_options["user_func"] is not None:
                data_layer = layer_options["user_func"](data_layer)
            lmin, lmax = np.nanmin(data_layer), np.nanmax(data_layer)
            cmin, cmax = layers_minmax[i_layer]
            layers_minmax[i_layer] = [np.nanmin([lmin, cmin]),
                                      np.nanmax([lmax, cmax])]


    @Multiprocess_filler(iterable_attr="chunk_slices",
            iter_kwargs="chunk_slice", veto_multiprocess=True)
    def compute_baselayer_minmax(self, chunk_slice=None, debug_mode=True):
        """ Compute the min & max of "base data" """
        chunk_2d = self._data[chunk_slice]
        chunk_2d = self.apply_mask(chunk_2d, chunk_slice)
        base_data = self.base_data_function(
                chunk_2d[len(self.calc_layers), :, :])
        lmin, lmax = np.nanmin(base_data), np.nanmax(base_data)

        self.base_min = np.nanmin([lmin, self.base_min])
        self.base_max = np.nanmax([lmax, self.base_max])


    def compute_baselayer_hist(self):
        """ Compute the histogramm of "base data"
        """
        print("computing histo", self.base_min, self.base_max)
        count_nonzero = [0]
        self.chunk_baselayer_hist(count_nonzero)
        bins_qt = np.cumsum(self.hist) / float(count_nonzero[0])
        bins_qt = np.insert(bins_qt, 0, 0.) 
        self.qt_to_z = lambda x: np.interp(x, bins_qt, self.bins)
        self.z_to_qt = lambda x: np.interp(x, self.bins, bins_qt)
        print("end computing histo")

        if self.probes_kind == "qt":
            # Define mappings :  quantiles -> z and z -> quantiles
            if np.any(np.diff(self.probes_qt) <= 0):
                raise ValueError("Expected strictly increasing probes_qt")
            self.probes_z = self.qt_to_z(self.probes_qt)
        elif self.probes_kind == "z":
            pass
        else:
            raise ValueError("Unexpected probes_kind {}".format(
                    self.probes_kind))

        # plot the histogramm
        bins = self.bins
        fig = Figure(figsize=(6., 6.))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        width = 1.0 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, self.hist, align='edge', width=width)  
        ax2 = ax.twinx()
        ax2.plot(bins, bins_qt, color="red", linewidth=2)
        fig.savefig(os.path.join(
             self.plot_dir, self.file_prefix + "_hist.png"))
        # saving to a human-readable format
        pc = np.linspace(0., 1., 101)
        np.savetxt(os.path.join(
            self.plot_dir, self.file_prefix + ".hist_bins.csv"),
            np.vstack([self.bins, bins_qt, self.qt_to_z(pc), pc]).T,
                      delimiter="\t", header="bins\tbins_qt\tpc_z\tpc")


    @Multiprocess_filler(iterable_attr="chunk_slices",
        iter_kwargs="chunk_slice", veto_multiprocess=True)
    def chunk_baselayer_hist(self, count_nonzero, chunk_slice=None):
        """
        Looping over chunks to compute self.hist, self.bins
        """
        chunk_2d = self._data[chunk_slice]
        chunk_2d = self.apply_mask(chunk_2d, chunk_slice)
        base_data = self.base_data_function(
                chunk_2d[len(self.calc_layers), :, :])

        loc_count_nonzero = np.count_nonzero(~np.isnan(base_data))
        if loc_count_nonzero == 0:
            return

        count_nonzero[0] += loc_count_nonzero
        loc_hist, loc_bins = np.histogram(base_data, bins=100,
                                  range=(self.base_min, self.base_max))
        if self.bins is None:
            self.bins = loc_bins
        if self.hist is None:
            self.hist = loc_hist
        else:
            self.hist += loc_hist


    @staticmethod
    def np2PIL(arr):
        """ Unfortunately this is a mess between numpy and pillow """
        sh = arr.shape
        if len(sh) == 2:
            return np.swapaxes(arr, 0 , 1 )[::-1, :]
        elif len(sh) == 3:
            return np.swapaxes(arr, 0 , 1 )[::-1, :, :]
        else:
            raise ValueError("Expected 2 or 3 dim array, got: {}".format(
                             len(sh)))

    @Multiprocess_filler(iterable_attr="chunk_slices",
        iter_kwargs="chunk_slice", veto_multiprocess=True)
    def plot_cropped(self, base_img, grey_img, normal_img, mask_img,
            zoning_img, postproc_keys, mask_color, layers_minmax,
            chunk_slice=None):
        """
        base_img: a reference to the "base" image
        grey_img :  a dict containing references to the individual grayscale 
                  "secondary layers" images, if layer *output* is true
        chunk_slice: None, provided by the wrapping looping function
        """
        print("plotting", chunk_slice)
        (ix, ixx, iy, iyy) = chunk_slice
        ny = self.fractal.ny

        # box – The crop rectangle, as a (left, upper, right, lower)-tuple.
        crop_slice = (ix, ny-iyy, ixx, ny-iy)
        chunk_2d = self._data[chunk_slice]
        chunk_2d = self.apply_mask(chunk_2d, chunk_slice)

        # Now we render in color - remember to apply base data function
        i_base = len(self.calc_layers) # pure "calculation" layers
        rgb = self.colormap.colorize(self.base_data_function(
            chunk_2d[i_base, :, :]), self.probes_z)

        # now we render the layers
        i_grey_img = 0  # here the first...
        for i_layer, layer_options in enumerate(self.grey_layers):
            shade = chunk_2d[i_layer + 1 + i_base, :, :]
            shade_function = layer_options["layer_func"]
            data_min, data_max = layers_minmax[i_layer]
            if self.probes_kind == "qt":
                blur_base = self.z_to_qt(self.base_data_function(
                                         chunk_2d[i_base, :, :]))
            elif self.probes_kind == "z":
                blur_base = self.base_data_function(chunk_2d[i_base, :, :])
            # Shading, only bluring is based on qt values or z values
            shade = shade_function(shade, data_min, data_max, blur_base)
            shade = np.where(np.isnan(shade), 0.50, shade)

            if layer_options["disp_layer"]:
                shade = np.where(np.isnan(shade), 0.0, shade)

#            if layer_options["output"]:
#                if layer_options["disp_layer"]:
                layer_mask_color = layer_options["layer_mask_color"]
                paste_layer = PIL.Image.fromarray(self.np2PIL(
                        np.int32(shade * 65535))) # 2**16-1
                if self.mask is not None:
                    self.apply_color_to_mask_cropped(paste_layer,
                        chunk_slice, layer_mask_color, mode="I")
            else:
                paste_layer = PIL.Image.fromarray(self.np2PIL(
                        np.uint8(255 * shade)))
            grey_img[i_grey_img].paste(paste_layer, box=crop_slice)
            i_grey_img += 1
            shade = np.expand_dims(shade, axis=2)
            shade_type = layer_options.get("shade_type",
                              {"Lch": 4., "overlay": 4., "pegtop": 1.})
            if not (layer_options["disp_layer"] or
                    layer_options["veto_blend"]):
                rgb = Color_tools.blend(rgb, shade, shade_type)

        rgb = np.uint8(rgb * 255)
        paste = PIL.Image.fromarray(self.np2PIL(rgb))

        # Here we render the normal maps for further post-processing
        i_normal_img = 0
        for i_layer, layer_options in enumerate(self.normal_layers):
            i_dzdx = i_base + len(self.grey_layers) + 2*i_normal_img + 1
            i_dzdy = i_dzdx + 1
            rgb =  np.zeros([ixx - ix, iyy - iy, 3], dtype=np.float32)
            rgb[:, :, 0] = chunk_2d[i_dzdx, :, :]
            rgb[:, :, 1] = chunk_2d[i_dzdy, :, :]
            rgb[:, :, 2] = -1.
            k = np.sqrt(rgb[:, :, 0]**2 + rgb[:, :, 1]**2 + rgb[:, :, 2]**2)
            for ik in range(3):
                rgb[:, :, ik] = rgb[:, :, ik] / k
            rgb = 0.5 * (-rgb + 1.)
            paste_layer = PIL.Image.fromarray(self.np2PIL(
                                              np.uint8(255 * rgb)))
            if self.mask is not None:
                self.apply_color_to_mask_cropped(paste_layer, chunk_slice,
                                                 mask_color=(0.5, 0.5, 1))
            normal_img[i_normal_img].paste(paste_layer, box=crop_slice)
            i_normal_img += 1

        # The optionnal mask layer
        if self.plot_mask:
            paste_layer = PIL.Image.fromarray(self.np2PIL(
                    self.mask[chunk_slice]))
            mask_img.paste(paste_layer, box=crop_slice)

        # The optionnal zoning layer
        if self.plot_zoning:
            greys = self.colormap.greyscale(self.base_data_function(
                chunk_2d[i_base, :, :]), self.probes_z)
            paste_layer = PIL.Image.fromarray(self.np2PIL(
                    np.uint8(255 * greys)))
            zoning_img.paste(paste_layer, box=crop_slice)

        if self.mask is not None:
            # define if we need transparency...
            if len(mask_color) > 3:
                paste.putalpha(255)
                
            self.apply_color_to_mask_cropped(paste, chunk_slice, mask_color)

        base_img.paste(paste, box=crop_slice)


    def apply_color_to_mask_cropped(self, cropped_img, chunk_slice,
                                    mask_color, mode=None):
        """
        Take the image
        Uniformely applies "mask_color" to the masked pixel
        """        
        mask_channel_count = len(mask_color)
        (ix, ixx, iy, iyy) = chunk_slice
        lx, ly = ixx-ix, iyy-iy

        crop_mask = PIL.Image.fromarray(self.np2PIL(
            np.uint8(255 * self.mask[chunk_slice])))
        mask_colors = np.tile(np.array(mask_color), (lx, ly)).reshape(
                                              [lx, ly, mask_channel_count])
        if mode == "I":
            mask_colors = self.np2PIL(np.int32((2**16 -1) * mask_colors))
        else:
            mask_colors = self.np2PIL(np.uint8(255 * mask_colors))

        if mask_channel_count == 1:
            mask_colors = mask_colors[:, :, 0]
            if mode is None:
                mode = "L"

        mask_colors = PIL.Image.fromarray(mask_colors, mode=mode)
        cropped_img.paste(mask_colors, (0, 0), crop_mask)


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
            data = data * Fractal_plotter.bluring_coeff(
                                                  blur1, blur2, blur3, blur_base)
        return data

    def blend_plots(self, im_file1, im_file2, im_mask=None, output_mode='RGB',
                    output_file="composite.png"):
        """ Programatically blend 2 images :
        - if *im_mask* is None, taking into account transparency of im2
        - else taking into account *im_mask* to select between image 1 and 2.
        """
        im_path1 = os.path.join(self.plot_dir, im_file1)
        im_path2 = os.path.join(self.plot_dir, im_file2)
        im_path = os.path.join(self.plot_dir, output_file)
        mask_path = None
        if im_mask is not None: 
            mask_path = os.path.join(self.plot_dir, im_mask)

        print("Blending :")
        print("blend source 1", im_path1)
        print("blend source 2", im_path2)
        print("blend mask", mask_path)
        print("blend dest", im_path)

        if im_mask is None:
            im1 = PIL.Image.open(im_path1).convert('RGBA')
            im2 = PIL.Image.open(im_path2).convert('RGBA')
            im = PIL.Image.alpha_composite(im1, im2).convert(output_mode)
        else:
            im1 = PIL.Image.open(im_path1)
            im2 = PIL.Image.open(im_path2)
            mask = PIL.Image.open(mask_path)
            im = PIL.Image.composite(im1, im2, mask).convert(output_mode)

        im.save(im_path)
        
#    def to_cartesian(self, exp_img_file, zooms):
#        """
#        """
#        im_path_source = os.path.join(self.plot_dir, exp_img_file)
#        exp_img = PIL.Image.open(im_path_source)
#        for zoom in zooms:
#            map_nx, map_ny = exp_img.size
#            h_max =  2 * np.pi * map_nx / map_ny
#            res_nx = map_ny / np.pi
#            res_ny = res_nx
#            # define y_loc, xloc in 0..1 x 0..1
#            x_loc, y_loc = np.meshgrid(np.linspace(-1., 1., res_nx),
#                                       np.linspace(-1., 1., res_ny))
#            rho = np.sqrt(y_loc, x_loc) * zoom
#            phi = np.arctan2(y_loc, x_loc)
#            xbar = np.where(rho < 1., rho, np.log(rho) + 1.)
#            ybar = phi * (np.arctan(xbar * np.pi / 2.) / np.pi * 2.)
#            rgb_zoom = np.array([res_nx, res_ny, 3], dtype= np.float32)
#            
#            h_chunk = (100. / map_nx) * h_max
#            xbar_min = np.min(xbar)
#            xbar_max = np.max(xbar)
#
#            # loop by steps of 1h
#            for ih in range(int((xbar_max - xbar_min) / h_chunk + 1)):
#                crop_h = (xbar_max + ih * h_chunk, 
#                          xbar_max + (ih + 1) * h_chunk )
#                self.to_cartesian_cropped(self, crop_h, xbar, ybar, rgb_zoom)
#            # now look in the original pic in terms of xbar ybar
#            
#        
##    @Multiprocess_filler(iterable_attr="chunk_slices",
##        iter_kwargs="zoom_slice", veto_multiprocess=True)
#    def to_cartesian_cropped(self, zoom_img, crop_h, xbar, ybar, rgb_zoom):
#        in_crop = (xbar >= crop_h) & (xbar < crop_h + 1.)
#        h_loc = np.log(zoom)
#        numpy.array(zoom_img.)
    
#        elif self.projection == "exp_map":
#            h_max = 2. * np.pi / self.xy_ratio # max h reached on the picture
#            xbar = (dx_vec / dx + 0.5) * h_max  # 0 .. hmax
#            ybar = dy_vec / dy * 2. * np.pi     # -pi .. +pi
#
#            phi = ybar / (np.arctan(xbar * np.pi / 2.) / np.pi * 2.)
#            phi = np.where(np.abs(phi) > np.pi, np.nan, phi) # outside
#            rho = np.where(xbar < 1., xbar, np.exp(xbar - 1.))
#            rho = rho * dx
#
#            x_vec = x + rho * np.cos(phi)
#            y_vec = y + rho * np.sin(phi) 
#    
    

class Fractal():
    """
    Abstract base class
    Derived classes are reponsible for computing, storing and reloading the raw
    data composing a fractal image.
    
    ref:
https://en.wikibooks.org/wiki/Pictures_of_Julia_and_Mandelbrot_Sets/The_Mandelbrot_set
    """

    def __init__(self, directory, x, y, dx, nx, xy_ratio, theta_deg,
                 chunk_size=200, complex_type=np.complex128,
                 projection="cartesian"):
        """
        Parameters
    *directory*   The working base directory
    *x* *y*  coordinates of the central point
    *dx*     reference data range (on x axis). Definition depends of the
        projection selected, for 'cartesian' (default) this is the total span.
    *nx*     number of pixels of the image along x axis
    *xy_ratio*  ratio of dy / dx and ny / nx
    *theta_deg*    Pre-rotation of the calculation domain
    *chunk_size*   size of the basic calculation 'tile' is chunk_size x chunk_size
    *complex_type*  numpy type or ("Xrange", numpy type)
    *projection*   "cartesian" "spherical" "exp_map"
        """
        self.x = x
        self.y = y
        self.nx = nx
        self.dx = dx
        self.xy_ratio = xy_ratio
        self.theta_deg = theta_deg
        self.directory = directory
        self.chunk_size = chunk_size
        self.projection = projection
        self.multiprocess_dir = os.path.join(self.directory, "multiproc_calc")

        # Data types
        self.complex_type = complex_type
        self.Xrange_complex_type = False
        self.base_complex_type = complex_type
        if type(self.complex_type) is tuple:
            type_modifier, base_complex_type = self.complex_type
            if type_modifier != "Xrange":
                raise ValueError(type_modifier)
            self.Xrange_complex_type = True
            self.base_complex_type = base_complex_type

        select = {np.complex64: np.float32,
                  np.complex128: np.float64}
        self.base_float_type = select[self.base_complex_type]

        # self.complex_store_type = self.base_complex_type # unused
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
    def px(self):
        if not(hasattr(self, "_px")): # is None:
            self._px = self.dx / self.nx
        return self._px

    @property    
    def params(self):
        """ Used for tagging a data chunk
        """
        return {"Software": "py-fractal",
                "fractal": type(self).__name__,
                "datetime": datetime.datetime.today().strftime(
                        '%Y-%m-%d_%H:%M:%S'),
                "x": self.x,
                "y": self.y,
                "nx": self.nx,
                "ny": self.ny,
                "dx": self.dx,
                "dy": self.dy,
                "theta_deg": self.theta_deg,
                "projection": self.projection}
#                "max_iter": self.max_iter,
#                "M_divergence": self.M_divergence,
#                "epsilon_stationnary": self.epsilon_stationnary}

    def data_file(self, chunk_slice, file_prefix):
        """
        Returns the file path to store or retrieve data arrays associated to a 
        data chunk
        """
        return os.path.join(self.directory, "data",
                file_prefix + "_{0:d}-{1:d}_{2:d}-{3:d}.tmp".format(
                *chunk_slice))
        
    def clean_up(self, file_prefix):
        """ Deletes all data files associated with a given file_prefix
        """
        pattern = file_prefix + "_*-*_*-*.tmp"
        data_dir = os.path.join(self.directory, "data")
        if not os.path.isdir(data_dir):
            return
        with os.scandir(data_dir) as it:
            for entry in it:
                if (fnmatch.fnmatch(entry.name, pattern)):
                    os.unlink(entry.path)

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
            print("Data computed, saving", save_path)
            pickle.dump(params, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(codes, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(raw_data, tmpfile, pickle.HIGHEST_PROTOCOL)
            
        # debug
        chunk_mask, Z, U, stop_reason, stop_iter = raw_data
                # debug

        print("debug save", chunk_slice)
        glitched = (stop_reason[0, :] == 3)
        print("**abs raw of ", chunk_slice)
        print("Z0 glitched shape", ((Z[3, :])[glitched]).shape)
        print("Z0 glitched", (Z[3, :])[glitched][:100])
            #raise ValueError

    def reload_data_chunk(self, chunk_slice, file_prefix, scan_only=False):
        """
        Reload arrays from a data file
           - params = main parameters used for the calculation
           - codes = complex_codes, int_codes, termination_codes
           - arrays : [Z, U, stop_reason, stop_iter]
        """
        save_path = self.data_file(chunk_slice, file_prefix)

        try:
            with open(save_path, 'rb') as tmpfile:
                params = pickle.load(tmpfile)
                codes = pickle.load(tmpfile)
                if scan_only:
                    return params, codes
                raw_data = pickle.load(tmpfile)
            
            return params, codes, raw_data
        
        # If no_compute ...
        except FileNotFoundError:
            if not(no_compute):
                raise
            
            # Else: no_compute selected, return empty fields
            # Except: first chunk will be computed (maybe more if
            # multiprocessing)
            for ref_chunk_slice in self.chunk_slices():
                ref_path = self.data_file(ref_chunk_slice, file_prefix)
                with open(ref_path, 'rb') as tmpfile:
                    params = pickle.load(tmpfile)
                    codes = pickle.load(tmpfile)
                    if scan_only:
                        return params, codes
                    raw_data = pickle.load(tmpfile)

                    n_Z, n_U, n_stop = (len(code) for code in codes)
                    (ix, ixx, iy, iyy) = chunk_slice
                    nxy = (ixx - ix) * (iyy - iy)
                    chunk_mask = np.zeros([nxy], dtype=np.bool)
                    if not(self.Xrange_complex_type):
                        Z = np.zeros([n_Z, 0], dtype=self.base_complex_type)
                    else:
                        Z = Xrange_array.zeros([n_Z, 0],
                                               dtype=self.base_complex_type)
                    U = np.zeros([n_U, 0], dtype=self.int_type)
                    stop_reason = -np.ones([1, 0], dtype=self.termination_type)
                    stop_iter = np.zeros([1, 0], dtype=self.int_type)

                    raw_data = (chunk_mask, Z, U, stop_reason, stop_iter)
                return params, codes, raw_data
            


    @staticmethod
    def dic_matching(dic1, dic2):
        """
        If we want to do some clever sanity test (not implemented)
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


    def c_chunk(self, chunk_slice):#, data_type=None):
        """
        Returns a chunk of c_vec for the calculation
        Parameters
         - chunk_span
         - data_type: expected one of np.float64, np.longdouble
        
        Returns: 
        c_vec : [chunk_size x chunk_size] 2d-vec of type datatype

        Projection availables cases :
            - cartesian : standard cartesisan
            - spherical : uses a spherical projection
            - exp_map : uses an exponential map projection
            - mixed_exp_map :  a mix of cartesian, exponential

        Note : return type is always standard prec - standard range
        """
        
        (x, y)  = (self.x, self.y)

        offset = self.offset_chunk(chunk_slice)
        return (x + offset[0]) + (y + offset[1]) * 1j


    def offset_chunk(self, chunk_slice, ensure_Xr=False):
        """
        Only computes the delta around ref central point for different projections
        Note : return type is always standard prec - standard or extended range
        """
#        select = {np.complex256: np.float128,
#                  np.complex128: np.float64}
        data_type = self.base_float_type # select[self.base_complex_type]

        (xy_ratio, theta_deg)  = (self.xy_ratio, self.theta_deg)
        (nx, ny, dx, dy) = (self.nx, self.ny, self.dx, self.dy)

        if self.Xrange_complex_type or ensure_Xr:
            dx_m, dx_exp = mpmath.frexp(dx)
            dx_m = np.array(dx_m, data_type)
            dx = Xrange_array(dx_m, dx_exp)
            dy_m, dy_exp = mpmath.frexp(dy)
            dy_m = np.array(dy_m, data_type)
            dy = Xrange_array(dy_m, dy_exp)
        else:
            dx = float(dx)
            dy = float(dy)
        
        (ix, ixx, iy, iyy) = chunk_slice
        theta = theta_deg / 180. * np.pi

        dx_grid = dx * np.linspace(-0.5, 0.5, num=nx, dtype=data_type)
        dy_grid = dy * np.linspace(-0.5, 0.5, num=ny, dtype=data_type)
        dy_vec, dx_vec  = np.meshgrid(dy_grid[iy:iyy], dx_grid[ix:ixx])

        if self.projection == "cartesian":
            offset = [(dx_vec * np.cos(theta)) - (dy_vec * np.sin(theta)),
                      (dx_vec * np.sin(theta)) + (dy_vec * np.cos(theta))]

        elif self.projection == "spherical":
            dr_sc = np.sqrt(dx_vec**2 + dy_vec**2) / max(dx, dy) * np.pi
            k = np.where(dr_sc >= np.pi * 0.5, np.nan,  # outside circle
                         np.where(dr_sc < 1.e-12, 1., np.tan(dr_sc) / dr_sc))
            dx_vec *= k
            dy_vec *= k
            offset = [(dx_vec * np.cos(theta)) - (dy_vec * np.sin(theta)),
                      (dx_vec * np.sin(theta)) + (dy_vec * np.cos(theta))]

        elif self.projection == "mixed_exp_map":
            # square + exp. map
            h_max = 2. * np.pi / xy_ratio # max h reached on the picture
            xbar = (dx_vec + 0.5 * dx - dy) / dx * h_max # 0 .. hmax
            ybar = dy_vec / dy * 2. * np.pi              # -pi .. +pi
            rho = dx * 0.5 * np.where(xbar > 0., np.exp(xbar), 0.)
            phi = ybar + theta
            dx_vec = (dx_vec + 0.5 * dx - 0.5 * dy) / xy_ratio
            dy_vec = dy_vec / xy_ratio
            offset = [np.where(xbar <= 0.,
                          (dx_vec * np.cos(theta)) - (dy_vec * np.sin(theta)),
                          rho * np.cos(phi)),
                      np.where(xbar <= 0.,
                          (dx_vec * np.sin(theta)) + (dy_vec * np.cos(theta)),
                          rho * np.sin(phi))]

        elif self.projection == "exp_map":
            # only exp. map
            h_max = 2. * np.pi / xy_ratio # max h reached on the picture
            xbar = (dx_vec + 0.5 * dx - dy) / dx * h_max # 0 .. hmax
            ybar = dy_vec / dy * 2. * np.pi              # -pi .. +pi
            rho = dx * 0.5 * np.exp(xbar)
            phi = ybar + theta
            offset = [rho * np.cos(phi), rho * np.sin(phi)]

        else:
            raise ValueError("Projection not implemented: {}".format(
                              self.projection))
        return offset


    def px_chunk(self, chunk_slice):
        """
        Local size of pixel for different projections
        """
#        select = {np.complex64: np.float32,
#                  np.complex128: np.float64}
        data_type = self.base_float_type# select[self.base_complex_type]

        xy_ratio  = self.xy_ratio
        (nx, ny, dx, dy) = (self.nx, self.ny, self.dx, self.dy)
        (ix, ixx, iy, iyy) = chunk_slice
        
        if not(self.Xrange_complex_type):
            dx = float(dx)
            dy = float(dy)
        else:
            dx_m, dx_exp = mpmath.frexp(dx)
            dx_m = np.array(dx_m, data_type)
            dx = Xrange_array(dx_m, dx_exp)
            dy_m, dy_exp = mpmath.frexp(dy)
            dy_m = np.array(dy_m, data_type)
            dy = Xrange_array(dy_m, dy_exp)

        dx_grid = dx * np.linspace(-0.5, 0.5, num=nx, dtype=data_type)
        dy_grid = dy * np.linspace(-0.5, 0.5, num=ny, dtype=data_type)
        dy_vec, dx_vec  = np.meshgrid(dy_grid[iy:iyy], dx_grid[ix:ixx])

#        dy_vec, dx_vec  = np.meshgrid(dy_grid[iy:iyy], dx_grid[ix:ixx])
        if self.projection == "cartesian":
            px = (dx / (nx - 1.)) #* np.ones_like(dx_vec)

        elif self.projection == "spherical":
            raise NotImplementedError()

        elif self.projection == "mixed_exp_map":
            raise NotImplementedError()

        elif self.projection == "exp_map":
            h_max = 2. * np.pi / xy_ratio
            xbar = (dx_vec + 0.5 * dx - dy) / dx * h_max
            px = (dx / (nx - 1.)) * 0.5 * h_max * np.exp(xbar)

        else:
            raise ValueError("Projection not implemented: {}".format(
                              self.projection))
        return px


    @Multiprocess_filler(iterable_attr="chunk_slices",
                         redirect_path_attr="multiprocess_dir",
                         iter_kwargs="chunk_slice")
    def cycles(self, initialize, iterate, subset, codes, file_prefix,
               chunk_slice=None, pc_threshold=0.2,
               iref=None, ref_path=None, SA_params=None,
               glitched=None, irefs=None):
        """
        Fast-looping for Julia and Mandelbrot sets computation.

        Parameters
        *initialize*  function(Z, U, c) modify in place Z, U (return None)
        *iterate*   function(Z, U, c, n) modify place Z, U (return None)

        *subset*   bool arrays, iteration is restricted to the subset of current
                   chunk defined by this array. In the returned arrays the size
                    of axis ":" is np.sum(subset[ix:ixx, iy:iyy]) - see below
        *codes*  = complex_codes, int_codes, termination_codes
        *file_prefix* prefix identifig the data files
        *chunk_slice_c* None - provided by the looping wrapper
        
        *iref* *ref_path* : defining the reference path, for iterations with
            perturbation method. if iref > 0 : means glitch correction loop.
        

        
        *gliched* boolean Fractal_Data_array of pixels that should be updated
                  with a new ref point
        *irefs*   integer Fractal_Data_array of pixels current ref points
        
        
        Returns 
        None - save to a file. 
        *raw_data* = (chunk_mask, Z, U, stop_reason, stop_iter) where
            *chunk_mask*    1d mask
            *Z*             Final values of iterated complex fields shape [ncomplex, :]
            *U*             Final values of int fields [nint, :]       np.int32
            *stop_reason*   Byte codes -> reasons for termination [:]  np.int8
            *stop_iter*     Numbers of iterations when stopped [:]     np.int32
        """
        # LEGACY dev
#      *terminate*   function (stop_reason, Z, U, c, n) modifiy in place
#      stop_reason (return None); if stop reason in range(nstop)
#     -> stopping iteration of this point.
        print("**CALLING cycles ")
        print("iref", iref)
        print("SA_params cutdeg", SA_params["cutdeg"])
        print("SA_params iref", SA_params["iref"])
        print("SA_params kc", SA_params["kc"])
        print("**/CALLING cycles ")
        
        
        if subset is not None:
            chunk_mask = np.ravel(subset[chunk_slice])
        else:
            chunk_mask = None
        
        # First tries to reload the data :
        try:
            (dparams, dcodes) = self.reload_data_chunk(chunk_slice,
                                                   file_prefix, scan_only=True)
            self.dic_matching(dparams, self.params)
            self.dic_matching(dcodes, codes)

            if (iref is None) or iref == 0:
                print("Data found, skipping calc: ", chunk_slice)
                return
            # Now if this is a glitch correction iteration... we should check
            # if any pixel in this chunk is glitched with iref_pix < iref
            else:
                glitch_loop =True
                glitched_chunk = np.ravel(glitched[chunk_slice])  # TODO: mask ??? Non because all gliched necessarly unmasked
                #glitched_chunk[...] = True #debug
                irefs_chunk = np.ravel(irefs[chunk_slice])
                if subset is not None:
                    glitched_chunk = glitched_chunk[chunk_mask]
                    irefs_chunk = irefs_chunk[chunk_mask]
                if (np.any(glitched_chunk) and
                    (np.min(irefs_chunk[glitched_chunk]) < iref)):
                    print("Glitch correction triggered, computing: ", iref)
                else:
                    print("Data found in glitch correction loop, "
                          "skipping calc: ", chunk_slice)        
                    return
        except IOError:
            glitch_loop = False # File not found, nothing to 'unglitch'
            print("Unable to find data_file, computing")



        # We didn't find, we need to compute
        if iref is not None: # Using perturbation
            c = self.diff_c_chunk(chunk_slice, iref, file_prefix)
            Xrc_needed = False
            if not(self.Xrange_complex_type) and (SA_params is not None):
                # We still need a Xrange version, to evaluate...
                Xrc_needed = True
                Xrc = self.diff_c_chunk(chunk_slice, iref, file_prefix,
                                        ensure_Xr=True)
        else:
            c = self.c_chunk(chunk_slice)
        (ix, ixx, iy, iyy) = chunk_slice
        print("c_chunk", chunk_slice, ":\n", c)

        c = np.ravel(c)
        if subset is not None:
            c = c[chunk_mask]

        if Xrc_needed:
            Xrc = np.ravel(Xrc)
            if subset is not None:
                Xrc = Xrc[chunk_mask]

        (n_pts,) = c.shape
        n_Z, n_U, n_stop = (len(code) for code in codes)
        if self.Xrange_complex_type:
            Z = Xrange_array.zeros([n_Z, n_pts], dtype=self.base_complex_type)
        else:
            Z = np.zeros([n_Z, n_pts], dtype=self.complex_type)
        U = np.zeros([n_U, n_pts], dtype=self.int_type)
        stop_reason = -np.ones([1, n_pts], dtype=self.termination_type) # np.int8 -> -128 to 127
        stop_iter = np.zeros([1, n_pts], dtype=self.int_type)

        if iref is not None:
            initialize(Z, U, c, chunk_slice, iref)
            # check is ref point had a early exit ??
            FP_params = self.reload_ref_point(iref, file_prefix, scan_only=True)
            ref_div_iter = FP_params.get("div_iter", 2**63 - 1) # max int64
            print("in cycles, ref point maxiter", ref_div_iter,
                  "ref_point", FP_params["ref_point"])
        else:
            initialize(Z, U, c, chunk_slice)

        #  temporary array for looping
        index_active = np.arange(c.size, dtype=self.int_type)
        bool_active = np.ones(c.size, dtype=np.bool)
        looping = True

        if glitch_loop:
            # Glitch correction engaged - we will only loop the glitched pixel
            # and keep the other 'as is'. We reload "raw data"
            (dparams, dcodes, raw_data) = self.reload_data_chunk(chunk_slice,
                                       file_prefix, scan_only=False)
            k_chunk_mask, k_Z, k_U, k_stop_reason, k_stop_iter = raw_data

            keep = ~glitched_chunk
            if k_chunk_mask is not None: # Safeguard
                np.testing.assert_equal(k_chunk_mask, chunk_mask)

            Z[:, keep] = k_Z[:, keep]
            U[:, keep] = k_U[:, keep]
            stop_reason[:, keep] =  k_stop_reason[:, keep]
            stop_iter[:, keep] = k_stop_iter[:, keep]

            
            index_active = index_active[glitched_chunk]
            Z_act = (Z[:, glitched_chunk]).copy() # ok for ndarray subclass
            U_act = np.copy(U[:, glitched_chunk])
            stop_reason_act = np.copy(stop_reason[:, glitched_chunk])
            c_act = (c[glitched_chunk]).copy() # np.copy(c)
            if Xrc_needed:
                Xrc = Xrc[glitched_chunk]
            bool_active = glitched_chunk[glitched_chunk]

            
        else:
            # No glitch correction, every pixel active...
            Z_act = Z.copy() # ok for ndarray subclass
            U_act = np.copy(U)
            stop_reason_act = np.copy(stop_reason)
            c_act = c.copy() # np.copy(c)


        n_iter = 0
        SA_iter = 0
        # Skipping first iterations if Series approximation activated
        if SA_params is not None:# and False:
            # Check if SA given is from another ref point in this case it will
            # need to be translated
            bool_SA_shift = (SA_params["iref"] != iref)
            print("bool_SA_shift", bool_SA_shift)
            # Seems that even if SA_params["iref"] != iref we do not need
            # to shift ??? How comes ???
            
            n_iter = SA_params["n_iter"]
            SA_iter = n_iter
            kc = SA_params["kc"]
            P = SA_params["P"]
#            if bool_SA_shift:
#                raise ValueError()
#                # Need to correct Z etc. by ref point shift
#                ref_array = self.get_ref_array(file_prefix)[:, :, :]
#                FP_params0 = self.reload_ref_point(SA_params["iref"], file_prefix,
#                                                   scan_only=True)
#                FP_paramsi = self.reload_ref_point(iref, file_prefix,
#                                                   scan_only=True)
#                dc_ref = (FP_paramsi["ref_point"] - FP_params0["ref_point"])
#                
#                print("refSA", SA_params["iref"], FP_params0["ref_point"])
#                print("ref1", iref, FP_paramsi["ref_point"])
#                
#                if self.Xrange_complex_type:
#                    dc_ref = mpc_to_Xrange(dc_ref, self.base_complex_type)
#                else:
#                    dc_ref = complex(dc_ref)#, self.base_complex_type)
#                    
#            else:
#                dc_ref = 0.
            
            
            for i_z in range(n_Z):
                if self.Xrange_complex_type:
#                    if bool_SA_shift:
#                        print("c shifted\n:", c_act - dc_ref)  # OK with return diff - drift_rp
#                        Z_act[i_z, :] = ((P[i_z])((c_act + dc_ref) / kc) 
#                                - ref_array[iref, n_iter, i_z]
#                                + ref_array[SA_params["iref"], n_iter, i_z])
#                    else:
                    Z_act[i_z, :] = (P[i_z])(c_act / kc)
#                    if bool_SA_shift and False:
                else:
#                    if bool_SA_shift:
#                        Z_act[i_z, :] = ((P[i_z])((c_act + dc_ref) / kc)
#                                - ref_array[iref, n_iter, i_z]
#                                + ref_array[SA_params["iref"], n_iter, i_z]
#                                 ).to_standard()
#                    else:
                    Z_act[i_z, :] = ((P[i_z])(Xrc / kc)).to_standard()
#                    if bool_SA_shift:
#                        Z_act[i_z, :] += (ref_array[iref, n_iter, i_z] - 
#                                ref_array[SA_params["iref"], n_iter, i_z]
#                                ).to_standard()

        pc_trigger = [False, 0, 0]   #  flag, counter, count_trigger
        
        print("**/CALLING cycles looping,  n_stop", n_stop)
        while looping:
            n_iter += 1

            # 1) Reduce of all arrays to the active part at n_iter
            if not(np.all(bool_active)):
                #print "select"
                c_act = c_act[bool_active]
                Z_act = Z_act[:, bool_active]
#                print(type(Z_act), Z_act.dtype)
                U_act = U_act[:, bool_active]
                stop_reason_act = stop_reason_act[:, bool_active]
                index_active = index_active[bool_active]

            # 2) Iterate all active parts vec, delegate to *iterate*
            if iref is not None:
#                iterate(Z_act, U_act, c_act, n_iter, ref_path[n_iter - 1, :])
                iterate(Z_act, U_act, c_act, stop_reason_act, n_iter, SA_iter,
                        ref_div_iter, ref_path[n_iter - 1 , :], ref_path[n_iter, :])
            else:
                iterate(Z_act, U_act, c_act, stop_reason_act, n_iter)

            # 3) Partial loop ends, delegate to *terminate*
#            if iref is not None:
#                terminate(stop_reason_act, Z_act, U_act, c_act, n_iter,
#                          ref_path[n_iter, :])
#            else:
#                terminate(stop_reason_act, Z_act, U_act, c_act, n_iter)
            stopped = (stop_reason_act[0, :] >= 0)
            index_stopped = index_active[stopped]
            stop_iter[0, index_stopped] = n_iter
            stop_reason[0, index_stopped] = stop_reason_act[0, stopped]
            Z[:, index_stopped] = Z_act[:, stopped]
            U[:, index_stopped] = U_act[:, stopped]
            bool_active = ~stopped
            
            
#            for stop in range(n_stop):
#                stopped = (stop_reason_act[0, :] == stop)
#                index_stopped = index_active[stopped]
#                stop_iter[0, index_stopped] = n_iter
#                stop_reason[0, index_stopped] = stop
#                Z[:, index_stopped] = Z_act[:, stopped]
#                U[:, index_stopped] = U_act[:, stopped]
#                if stop == 0:
#                    bool_active = ~stopped
#                else:
#                    bool_active[stopped] = False
#                    # debug
#                if np.any(stopped) and stop == 3:
#                    print("dyn glitch µµµ")
#                    #print(Z_act[3, stopped])
#                    print(Z[3, index_stopped])
                    

            count_active = np.count_nonzero(bool_active)
            pc = 0 if(count_active == 0) else count_active / (c.size *.01)

            if (count_active < 3 or pc < pc_threshold) and n_iter > 5000:
                if pc_trigger[0]:
                    pc_trigger[1] += 1   # increment counter
                else:
                    pc_trigger[0] = True # start couting
                    pc_trigger[2] = int(n_iter * 0.25) # set_trigger to 125%
                    print("Trigger hit at", n_iter)
                    print("will exit at most at", pc_trigger[2] + n_iter)
                    sys.stdout.flush()
                    
                    reasons_pc = np.zeros([n_stop], dtype=np.float32)
                    for stop in range(n_stop):
                        reasons_pc[stop] = np.count_nonzero(
                            stop_reason[0, :] == stop) /(c.size *.01)
                    status = ("iter{0} : active: {1} - {2}% " +
                              "- by reasons (%): {3}").format(
                              n_iter, count_active, pc, reasons_pc)
                    print(status)

            looping = (count_active != 0 and
                       (pc_trigger[1] <= pc_trigger[2]))
            if n_iter%500 == 0: # 5000
                reasons_pc = np.zeros([n_stop], dtype=np.float32)
                for stop in range(n_stop):
                    reasons_pc[stop] = np.count_nonzero(
                        stop_reason[0, :] == stop) /(c.size *.01)
                status = ("iter{0} : active: {1} - {2}% " +
                          "- by reasons (%): {3}").format(
                          n_iter, count_active, pc, reasons_pc)
                print(status)
                sys.stdout.flush()



        # debug
        if chunk_slice == (0, 200, 0, 200):
            print("debug")
            glitched = (stop_reason[0, :] == 3)
            print("**abs raw of ", chunk_slice)
            print("Z0 glitched shape", ((Z[3, :])[glitched]).shape)
            print("Z0 glitched", (Z[3, :])[glitched][:100])
            #raise ValueError
            

        # Don't save temporary codes - i.e. those which startwith "_"
        # inside Z and U arrays
        select = {0: Z, 1: U}
        target = [None, None]
        save_codes = copy.copy(codes) # deepcopy
        for icode, code in enumerate(save_codes):
            if icode == 2:
                break
            to_del = [ifield for ifield, field in enumerate(code)
                      if field[0] == "_"]
            target[icode] = np.delete(select[icode], to_del ,0)
            for index in to_del[::-1]:
                del code[index]
        save_Z, save_U = target


        
#        params = self.params + 
#        if (iref is not None) and iref > 0:
#            # Glich correction engaged - shall only modify in place
#            # glitched pixels
#            for icode, code in enumerate(save_codes):
#                pass
#            raise NotImplementedError()
            
            
        raw_data = [chunk_mask, save_Z, save_U, stop_reason, stop_iter]
        
        
        # debug
        if chunk_slice == (0, 200, 0, 200):
            print("save_Z glitched", (save_Z[3, :])[glitched][:100])

        
        self.save_data_chunk(chunk_slice, file_prefix, self.params,
                             save_codes, raw_data)


    def kind_from_code(self, code, codes):
        """
        codes as returned by 
        (params, codes) = self.reload_data_chunk(chunk_slice,
                                                   file_prefix, scan_only=True)
        """
        complex_codes, int_codes, _ = codes
        if code in complex_codes:
            kind = "complex"
        elif code in int_codes:
            kind = "int"
        elif code == "stop_reason":
            kind = code
        elif code == "stop_iter":
            kind = code
        else:
            print("complex_codes: ", complex_codes)
            raise KeyError("raw data code unknow: " + code)
        return kind


    def get_raw_data(self, code, kind, file_prefix, chunk_slice):
        """
        Note: looping to be done outside of this func
        #*chunk_slice* None - provided by the looping wrapper
        #*c*           None - provided by the looping wrapper
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
        return filler[0, :, :]


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
                                               chunk_slice, file_prefix)
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
        ["attr_shade"dc]  |[{"theta_LS":, "phi_LS":, "shininess":, 
                          | "ratio_specular":, "LS_coords"} 
        [power_attr_shade]| Variation on ["attr_shade"] 
                          |
        [field_lines]     |{} # Calculation of a C1 phase angle (modulo 2 pi)
        [Lyapounov]       |{"n_iter": [field / None]}
        [raw]             | {"code": "raw_code"}
        [phase]           | {"source": "raw_code"}
        [minibrot_phase]  | {"source": "raw_code"}  same than *phase* but with
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
        
        def fill_color_tool_dic():
            color_tool_dic = post_dic.copy()
            color_tool_dic["chunk_slice"] = chunk_slice
            color_tool_dic["chunk_mask"] = chunk_mask
#            print("sending chunk_mask", chunk_mask)
            color_tool_dic["nx"] = self.nx
            color_tool_dic["ny"] = self.ny
            return color_tool_dic

        for i_key, postproc_key in enumerate(postproc_keys):
#            print("postproc_keys", postproc_keys)
#            print("i_key, postproc_key", i_key, postproc_key)
            post_name, post_dic = postproc_key

            if post_name == "potential": # In fact the 'real iteration number'
                has_potential = True
                potential_dic = post_dic
                n = stop_iter[0, :]
                zn = Z[complex_dic["zn"], :]
#                print("zn", type(zn), zn.dtype)

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
#                    print("nu_frac", type(nu_frac), nu_frac.dtype)

                elif potential_dic["kind"] == "convergent":
                    eps = potential_dic["epsilon_cv"]
                    alpha = 1. / Z[complex_dic["attractivity"]]
                    z_star = Z[complex_dic["zr"]]
                    nu_frac = + np.log(eps / np.abs(zn - z_star)  # nu frac > 0
                                       ) / np.log(np.abs(alpha))

                elif potential_dic["kind"] == "transcendent":
                    # Not possible to define a proper potential for a 
                    # transcencdent fonction
                    nu_frac = 0.

                else:
                    raise NotImplementedError("Potential 'kind' unsupported")
#                print("nu_frac", type(nu_frac), nu_frac.dtype, nu_frac.shape)
#                print("n", type(n), n.dtype, n.shape)
                #nu = n[].astype(self.base_complex_type) + nu_frac
                if type(nu_frac) == Xrange_array:
                    nu_frac = nu_frac.to_standard()
                nu = n + nu_frac
                val = nu


            elif post_name == "DEM_shade":
                if not has_potential:
                    raise ValueError("Potential shall be defined before shade")
                dzndc = Z[complex_dic["dzndc"], :]
                color_tool_dic = fill_color_tool_dic() #post_dic.copy()
#                color_tool_dic["chunk_slice"] = chunk_slice
#                color_tool_dic["chunk_mask"] = chunk_mask
#                color_tool_dic["nx"] = self.nx
#                color_tool_dic["ny"] = self.ny
                shade_kind = color_tool_dic.pop("kind")

                if shade_kind == "Milnor":   # use only for z -> z**2 + c
# https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
                    d2zndc2 = Z[complex_dic["d2zndc2"], :]
                    abs_zn = np.abs(zn)
                    lo = np.log(abs_zn)
                    normal = zn * dzndc * ((1+lo)*np.conj(dzndc * dzndc)
                                  -lo * np.conj(zn * d2zndc2))
                    normal = normal / np.abs(normal)
                    
                    # Normal doesnt't mean anything when too close...
                    dist = abs_zn * lo / np.abs(dzndc)
                    px = self.px  #self.dx / float(self.nx)
                    if type(px) == mpmath.ctx_mp_python.mpf:
                        m, exp = mpmath.frexp(px)
                        px = Xrange_array(float(m), int(exp))
#                        print("px", px)
#                        print("dist", dist, type(dist), dist.dtype)
#                        print("ineq", dist < (px * 0.01))
                        if type(normal) == Xrange_array:
                            normal = normal.to_standard()
                        #normal[dist < (px * 0.25)] = 0.#, 0., 1.)

#                px_snap = post_dic.get("px_snap", 1.)
                    #val = np.where(val < px * px_snap, 0., val)

                elif shade_kind == "potential": # safer
                    if potential_dic["kind"] == "infinity":
                    # pulls back the normal direction from an approximation of 
                    # potential: phi = log(zn) / 2**n 
                        normal = zn / dzndc# zn / dzndc
                        normal = normal / np.abs(normal)
                        if type(normal) == Xrange_array:
                            normal = normal.to_standard()

                    elif potential_dic["kind"] == "convergent":
                    # pulls back the normal direction from an approximation of
                    # potential (/!\ need to derivate zr w.r.t. c...)
                    # phi = 1. / (abs(zn - zr) * abs(alpha)) 
                        zr = Z[complex_dic["zr"]]
                        dzrdc = Z[complex_dic["dzrdc"]]
                        normal = - (zr - zn) / (dzrdc - dzndc)
                        normal = normal / np.abs(normal)
                        
#                if self.Xrange_complex_type:
#                    normal = normal.to_standard()
                    

#                fade = ((dist > px * 0.5) & (dist < px))
#                print("normal", type(normal), normal.dtype)
#                print("dist", type(dist), dist.dtype)
#                print("fade", type(fade), fade.dtype)
#                print("px", type(px), px.dtype)
#                _ = dist[fade] / px
#                print("1 ok")
#                _ = ((dist[fade] / px) - 0.5)
#                print("2 ok")
#                _ = (((dist[fade] / px) - 0.5) * 2.)
#                print("3 ok")
#                test = (((dist[fade] / px) - 0.5) * 2.).to_standard()
#                print("4 ok", type(test), test.dtype, np.min(test), np.max(test))
#                normal[fade] = normal[fade] * (
#                        ((dist[fade] / px) - 0.5) * 2.).to_standard()
                
#                print("sending !", np.shape(zn), color_tool_dic)
#                print("sending normal !", normal.shape, type(normal), normal.dtype)
                val = Color_tools.shade_layer(
                        normal * np.cos(post_dic["phi_LS"] * np.pi / 180.),
                        **color_tool_dic)
                #val = np.where(val < px * 0.1, 0.5, val)


            elif post_name == "field_lines":
                if not has_potential:
                    raise ValueError("Potential shall be defined before " +
                                     "stripes")
                if potential_dic["kind"] == "infinity":
                    # Hopefully C1-smooth phase angle for z -> a_d * z**d
                    # G. Billotey
                    k_alpha = 1. / (1. - d)       # -1.0 if N = 2
                    k_beta = - d * k_alpha        # 2.0 if N = 2
                    z_norm = zn * a_d**(-k_alpha) # the normalization coeff

                    t = [np.angle(z_norm)]
#                    print("t", t[0].dtype)
                    val = np.zeros_like(t)
                    valk = 0.
#                    print("val", val.dtype)
                    for i in range(1, 6):
                        t += [d * t[i - 1]]
                        dphi =  i * np.pi * 0.1
#                        print("val2", type((0.5)**i)) #float
#                        print("val3", type(np.sin(t[i-1] + dphi)))
#                        print("val3", type(t[i-1]))
#                        print("val4", type(k_alpha + k_beta * d**nu_frac.to_standard()))
                        val += (0.9)**i  * (np.sin(t[i-1] + dphi) + (
                                  k_alpha + k_beta * d**nu_frac) * (
                                  np.sin(t[i] + dphi) - np.sin(t[i-1] + dphi)))
                        valk += (0.9)**i
                    val = np.arcsin(((((val / valk) + 1) * 0.5) % 1.) * 2. - 1.)
                    del t

                elif potential_dic["kind"] == "convergent":
                    alpha = 1. / Z[complex_dic["attractivity"]]
                    z_star = Z[complex_dic["zr"]]
                    beta = np.angle(alpha)
                    # val = np.angle((z_star - zn) / (-nu_frac * beta))
                    val = np.angle(z_star - zn) + nu_frac * beta
                    # We have an issue if z_star == zn...
                    val = val * 2.


            elif post_name == "Lyapounov":
                dzndz = Z[complex_dic["dzndz"], :]
                if post_dic["n_iter"] is None:
                    n = stop_iter[0, :]
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

                color_tool_dic = fill_color_tool_dic()
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

                color_tool_dic = fill_color_tool_dic()
                val = Color_tools.shade_layer(
                        normal * np.cos(post_dic["phi_LS"] * np.pi / 180.),
                        **color_tool_dic)     

            elif "attr_normal_n" in post_name:
                # Plotting the 2 main coords for a complex normal map
                coord = post_name[13]
                attr = Z[complex_dic["attractivity"], :]
                dattrdc = Z[complex_dic["dattrdc"], :]
                normal = attr * np.conj(dattrdc)  / np.abs(dattrdc)
                if coord == "x":
                    val = - np.real(normal)
                elif coord == "y":
                    val = - np.imag(normal)
                else:
                    raise ValueError(coord)

            elif "DEM_shade_normal_n" in post_name:
                # Plotting the 2 main coords for a complex normal map based on
                # Millnor distance estimator
                coord = post_name[18]
                zn = Z[complex_dic["zn"], :]
                dzndc = Z[complex_dic["dzndc"], :]
                d2zndc2 = Z[complex_dic["d2zndc2"], :]
                lo = np.log(np.abs(zn))
                normal = zn * dzndc * ((1.+lo)*np.conj(dzndc * dzndc)
                                  -lo * np.conj(zn * d2zndc2))
                normal = normal / np.abs(normal)

                if coord == "x":
                    val = - np.real(normal)
                elif coord == "y":
                    val = - np.imag(normal)
                else:
                    raise ValueError(coord)

            elif post_name == "attr_height_map" :
                # Plotting the 'domed' height map for the cycle attractivity
                attr = Z[complex_dic["attractivity"], :]
                invalid = (np.abs(attr) > 1.)
                attr = np.where(invalid, attr / np.abs(attr), attr)
                dattrdc = Z[complex_dic["dattrdc"], :]
                dattrdc = np.where(invalid, 1., dattrdc)                
                normal = attr * np.conj(dattrdc) / np.abs(dattrdc)

                r = U[int_dic["r"], :]
                val = np.sqrt(1. - attr * np.conj(attr)) / r


            elif post_name == "DEM_height_map" :
                # This height may have a noticeable banding (especially when
                # exported to 3d)
                # in this case the user can pass the function :
                # zn -> zn+1 and dzndc -> dzn+1dc 
                # (iter_zn, iter_dzndc) = iteration(zn, dzndc, c)
                # iteration
                if not has_potential:
                    raise ValueError("Potential shall be defined before " +
                                     "DEM_height_map")
                zn = Z[complex_dic["zn"], :]
                dzndc = Z[complex_dic["dzndc"], :]

                iteration = post_dic.get("iteration", None)
                if iteration is not None:
                    c = np.ravel(self.c_chunk(chunk_slice))
                    if chunk_mask is not None:
                        c = c[chunk_mask]
                    iter_zn, iter_dzndc = self.d1_iteration(zn, dzndc, c,
                                                            **iteration)
                    s_zn =  iter_zn + nu_frac * (zn - iter_zn)
                    s_dzndc = dzndc * (nu_frac) + (1. - nu_frac) * iter_dzndc
                    

                else:
                    s_zn, s_dzndc = zn, dzndc

                if potential_dic["kind"] == "infinity":
                    abs_zn = np.abs(s_zn)
                    val = abs_zn * np.log(abs_zn) / np.abs(s_dzndc)

                elif potential_dic["kind"] == "convergent":
                    zr = Z[complex_dic["zr"]]
                    dzrdc = Z[complex_dic["dzrdc"], :]
                    val = np.abs((zr - s_zn) / (dzrdc - s_dzndc))

                px = self.dx / float(self.nx)
                px_snap = post_dic.get("px_snap", 1.)
                val = np.where(val < px * px_snap, 0., val)

            elif post_name == "DEM_explore" :
                # This height may have a noticeable banding (especially when
                # exported to 3d)
                # in this case the user can pass the function :
                # zn -> zn+1 and dzndc -> dzn+1dc 
                # (iter_zn, iter_dzndc) = iteration(zn, dzndc, c)
                # iteration
                if not has_potential:
                    potential_dic = post_dic.get("potential_dic", None)
                    if potential_dic is None:
                        raise ValueError("Potential shall be defined before "
                            "DEM_height_map")

                zn = Z[complex_dic["zn"], :]
                dzndc = Z[complex_dic["dzndc"], :]

                if potential_dic["kind"] == "infinity":
                    abs_zn = np.abs(zn)
                    val = abs_zn * np.log(abs_zn) / np.abs(dzndc)

                elif potential_dic["kind"] == "convergent":
                    zr = Z[complex_dic["zr"]]
                    dzrdc = Z[complex_dic["dzrdc"], :]
                    val = np.abs((zr - zn) / (dzrdc - dzndc))

                px = self.dx / float(self.nx)
                px_snap = post_dic.get("px_snap", 1.)
                val = np.where(val < px * px_snap, 0., 1.)


            elif post_name == "potential_height_map" :
                if not has_potential:
                    raise ValueError("Potential shall be defined before " +
                                     "potential_height_map")
                # Plotting the height map for the Potential
                nu = stop_iter[0, :] + nu_frac
                val = np.log(nu)

            elif post_name == "power_attr_heightmap":
                r = U[int_dic["r"], :]
                val =  np.where(r == 0, 0., 1./np.log(1./ r))
                
            elif "power_attr__normal_n" in post_name:
                # Plotting the 2 main coords for a complex map based on Millnor
                # distance
                coord = post_name[20]
                attr = Z[complex_dic["attractivity"], :]
                invalid = (np.abs(attr) > 1.)
                attr = np.where(invalid, attr / np.abs(attr), attr)

                dattrdc = Z[complex_dic["dattrdc"], :]
                dattrdc = np.where(invalid, 0., dattrdc)
                normal = (.2 + .8 * attr) * np.conj(dattrdc) / np.abs(dattrdc)

                if coord == "x":
                    val = - np.real(normal)
                elif coord == "y":
                    val = - np.imag(normal)
                else:
                    raise ValueError(coord)

            elif post_name == "_special1":
                code = post_dic["r_candidate"]
                order = U[int_dic[code], :]
                val = order
#                has_potential = True
#                potential_dic = {"kind": "infinity", "d": 2, "a_d": 1., "N": 1e3}
#                dx = 2.375e-14 * 0.36 * 0.2 * 2.
#                n = stop_iter
#                zn = Z[complex_dic["zn"], :]
#                d = 2.
#                a_d = 1.
#                N = 1e3
#                k = np.abs(a_d) ** (1. / (d - 1.))
#                    # k normaliszation corefficient, because the formula given
#                    # in https://en.wikipedia.org/wiki/Julia_set                    
#                    # suppose the highest order monome is normalized
#                nu_frac = -(np.log(np.log(np.abs(zn * k)) / np.log(N * k))
#                                / np.log(d))
#                nu = n + nu_frac
#                val1 = np.log(nu)
#                val1 = np.arctan(val1 - 7.35906457901001)
#                min1 = -0.48491284251213074
#                max1 = 1.4317854642868042
#                val1 = (val1 - min1) / (max1 - min1)
#                val1 = np.where(val1 > max1, max1, val1)
#                val1 = np.where(val1 < min1, min1, val1)
#
#                dzndc = Z[complex_dic["dzndc"], :]
#                abs_zn = np.abs(zn)
#                val2 = abs_zn * np.log(abs_zn) / np.abs(dzndc)
#                val2 = np.where(val2 > dx, dx, val2)
#                px = dx / 4000.# float(self.nx)
#                px_snap = post_dic.get("px_snap", 1.)
#                val2 = np.where(val2 < px * px_snap, 0., val2)
#                min2 = 0.0
#                max2 = 3.4199999999999996e-15
#                val2 = (val2 - min2) / (max2 - min2)
#                val = np.where(val2 == 0., val1, -val2)

            elif post_name == "phase":
                # Plotting raw complex data : plot the angle
                code = post_dic["source"]
                z = Z[complex_dic[code], :]
                val = np.angle(z)

            elif post_name == "abs":
                # Plotting raw complex data : plot the abs
                code = post_dic["source"]
                multiplier = post_dic.get("multiplier", 1.)
                z = Z[complex_dic[code], :] 
                val = multiplier * np.abs(z)
                
                # debug
                debug = True
                if debug:
        #                if type(val) == Xrange_array:
        #                    val = val.to_standard()
                    #val[~np.isfinite(val)] = 0.
                    glitched = (stop_reason[0, :] == 3)
                    print("**abs raw of ", chunk_slice)
                    print("field ", code, complex_dic[code])
                    print("abs print", val[glitched][:100])
                    print("Z0 glitched shape", ((Z[3, :])[glitched]).shape)
                    print("Z0 glitched", (Z[3, :])[glitched][:100])
                    val = np.log(val)
                    #raise ValueError
    
            elif post_name == "minibrot_phase":
                # Pure graphical post-processing for minibrot 'attractivity'
                # The phase 'vanihes' at the center
                code = post_dic["source"]
                z = Z[complex_dic[code], :]
                val = np.cos(np.angle(z) * 24) * np.abs(z)**8

            elif post_name == "raw":
                code = post_dic["code"]
                kind = self.kind_from_code(code, codes)
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

            if type(val) == Xrange_array:
                val = val.to_standard()
            post_array[i_key, :] = val  #.to_standard()

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
            raise ValueError("Expected bool_subset_of_set of shape"
                             " [Card(set)]")
        bool_subset = np.copy(bool_set)
        bool_subset[bool_set] = bool_subset_of_set
        return bool_subset
