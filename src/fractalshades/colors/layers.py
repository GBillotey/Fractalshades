# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib.format import open_memmap
#import matplotlib.colors
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_agg import FigureCanvasAgg


import PIL
import PIL.ImageQt
import os

import fractalshades.numpy_utils.expr_parser as fs_parser
import fractalshades.colors as fscolors

#import fractalshades.settings as fssettings
#import fractalshades.utils as fsutils

#        Note: Pillow image modes:
#        RGB (3x8-bit pixels, true color)
#        RGBA (4x8-bit pixels, true color with transparency mask)
#        1 (1-bit pixels, black and white, stored with one pixel per byte)
#        L ((8-bit pixels, black and white)
#        I (32-bit signed integer pixels)
"""
fractal -> postproc -> Secondaryla -> layer -> combined layer (blend, shading)

"""

class Virtual_layer:

    N_CHANNEL_FROM_MODE = {
        "1": 1,
        "L": 1,
        "I": 1,
        "RGB": 3,
        "RGBA": 4
    }
    DTYPE_FROM_MODE = {
        "1": bool,
        "L": np.uint8,
        "I": np.int32,
        "RGB": np.uint8,
        "RGBA": np.uint8
    }
    default_mask_color = None

    def __init__(self, postname, func, output=True):
        """ 
    Base class for all layer objects.
    
    Parameters
    ----------
    postname : str
        Unique identifier key for this layer. When added to a 
        `fractalshades.Fractal_plotter` plotter (see 
        `fractalshades.Fractal_plotter.add_layer`) the layer can then be
        accessed through this key :
            
            layer = plotter["postname"]
        
        The `postname` must match a `fractalshades.postproc.Postproc`
        associated with this plotter.

    func : None | callable
        mapping applied during the post-processing initial step

    output : bool
        If True : an image will be saved when
        `fractalshades.Fractal_plotter.add_layer` directive issued.
    """
        self.postname = postname
        self.min = np.inf     # neutral value for np.nanmin
        self.max = -np.inf    # neutral value for np.nanmax
        self._scaling_defined = False # tracker
        self._func_arg = func # kept for the record
        self.func = self.parse_func(func)
        self.output = output
        self.mask = None

    @staticmethod
    def parse_func(func):
        if isinstance(func, str):
            return fs_parser.func_parser(["x"], func)
        else:
            return func

    def link_plotter(self, plotter):
        self.plotter = plotter

    @property
    def fractal(self, fractal):
        return self.plotter.fractal

    @property
    def n_channel(self):
        """ private - The number of channels for self.arr """
        return self.N_CHANNEL_FROM_MODE[self.mode]

    @property
    def dtype(self):
        """ private - The dtype for self.arr """
        return self.DTYPE_FROM_MODE[self.mode]

    def set_mask(self, layer, mask_color=None):
        """
        Masks the base-layer

        Parameters
        ----------
        layer : `Bool_layer` | `Grey_layer`
            The layer mask
        mask_color : None | 3-uplet or 4_uplet of floats belonging to [0., 1.]
            If None, a default value will be provided depending on the masked
            array subclass
            If a n-uplet is provided, it shall be with RGB or RGBA values for a
            colored layer
        """
        if mask_color is None: # default class-implementation
            mask_color = self.default_mask_color
        self.mask = (layer, mask_color)

    @property
    def mask_kind(self):
        """ private - 3 possiblilities : 
                - no mask (returns None)
                - boolean mask (return "bool")
                - greyscale i.e. float mask (return "float) """
        if self.mask is None:
            return None
        layer, mask_color = self.mask
        if isinstance(layer, Bool_layer):
            return "bool"
        elif isinstance(layer, Grey_layer):
            return "float"
        else:
            raise ValueError("Mask layer type not recognized"
                             + type(layer).__name__)

    def __getitem__(self, chunk_slice):
        """ read the base data array for this layer
        Returns a numpy array of 
        - shape (lx, ly) or (n_fields, lx, ly) if n_fields > 1
        - datatype self.dtype
        
        Note: this is the raw, unscaled values
        """
        (ix, ixx, iy, iyy) = chunk_slice
        plotter = self.plotter
        mmap = open_memmap(filename=plotter.temporary_mmap_path(), mode='r')
        postname = self.postname
        try:
            rank = list(plotter.postnames).index(postname)
            arr = mmap[rank, ix:ixx, iy:iyy]
        except ValueError:
            # Could happen that we need 2 fields (e.g., normal map...)
            if postname not in self.plotter.postnames_2d:
                raise ValueError("postname not found: {}".format(postname))
            rank = list(self.plotter.postnames).index(postname + "_x")
            rank2 = list(self.plotter.postnames).index(postname + "_y")
            if rank2 != rank + 1:
                raise ValueError("x y coords not contiguous for postname: "
                                 "{}".format(postname))
            arr = mmap[rank:rank+2, ix:ixx, iy:iyy]
        return arr

    def update_scaling(self, chunk_slice):
        """ Update the overall min - max according to what is found in this 
         chunk_slice  """
        arr = self[chunk_slice]
        # If a user-mapping is defined, apply it
        if self.func is not None:
            arr = self.func(arr)
        
        sh = arr.shape
        if len(sh) == 2:
            n_fields = 1
        elif len(sh) == 3:
            n_fields = arr.shape[0]
        
        if n_fields == 1: # standard case
            min_chunk = self.nanmin_with_mask(arr, chunk_slice)
            max_chunk = self.nanmax_with_mask(arr, chunk_slice)
            self.min = np.nanmin([self.min, min_chunk])
            self.max = np.nanmax([self.max, max_chunk])
        elif n_fields == 2: # case of normal maps
            # Normalizing by its module
            arr = arr[0, :, :]**2 + arr[1, :, :]**2
            max_chunk = self.nanmax_with_mask(arr, chunk_slice)
            max_chunk = np.sqrt(max_chunk)
            self.max = np.nanmax([self.max, max_chunk])
            self.min = - self.max
        else:
            raise ValueError(n_fields)
        self._scaling_defined = True

    def nanmin_with_mask(self, arr, chunk_slice):
        """ nanmin but disregarding the masked vals (if layer has a mask)"""
        mask_kind = self.mask_kind
        if mask_kind is None:
            return np.nanmin(arr)
        elif mask_kind in ["bool", "float"]:
            if mask_kind == "bool":
                keep_arr = ~self.mask[0][chunk_slice]
            elif mask_kind == "float":
                keep_arr = (self.mask[0][chunk_slice] != 1.)
            if np.any(keep_arr):
                return np.nanmin(arr[keep_arr])
            return np.inf
        else:
            raise ValueError(mask_kind)

    def nanmax_with_mask(self, arr, chunk_slice):
        """ nanmax but disregarding the masked vals (if layer has a mask)"""
        mask_kind = self.mask_kind
        if mask_kind is None:
            return np.nanmax(arr)
        elif mask_kind in ["bool", "float"]:
            if mask_kind == "bool":
                keep_arr = ~self.mask[0][chunk_slice]
            elif mask_kind == "float":
                keep_arr = (self.mask[0][chunk_slice] != 1.)
            if np.any(keep_arr):
                return np.nanmax(arr[keep_arr])
            return -np.inf
        else:
            raise ValueError(mask_kind) 

    def crop(chunk_slice):
        """ Return the image for this chunk
        Subclasses should implement"""
        raise NotImplementedError("Derivate classes should implement "
                                  "this method")

    @staticmethod
    def np2PIL(arr):
        """ Utility function
        Unfortunately we have a mess between numpy and pillow """
        sh = arr.shape
        if len(sh) == 2:
            return np.swapaxes(arr, 0 , 1 )[::-1, :]
        elif len(sh) == 3:
            return np.swapaxes(arr, 0 , 1 )[::-1, :, :]
        else:
            raise ValueError("Expected 2 or 3 dim array, got: {}".format(
                             len(sh)))

    @staticmethod
    def PIL2np(arr):
        """ Inverse of np2PIL """
        sh = arr.shape
        if len(sh) == 2:
            return np.swapaxes(arr, 0 , 1 )[:, ::-1]
        elif len(sh) == 3:
            return np.swapaxes(arr, 0 , 1 )[:, ::-1, :]
        else:
            raise ValueError("Expected 2 or 3 dim array, got: {}".format(
                             len(sh)))

class Color_layer(Virtual_layer):
    default_mask_color = (0., 0., 0.)

    def __init__(self, postname, func, colormap, probes_z=[0, 1],
                 probes_kind="relative", output=True):
        """
        A colored layer.
        
        Parameters
        ----------
        postname : str
            passed to `Virtual_layer` constructor
        func :
            passed to `Virtual_layer` constructor
        colormap : fractalshades.colors.Fractal_colormap
            a colormap
        probes_z : 2-floats list
            The preprocessing affine rescaling. If [0., 1.] and probes_kind is 
            `relative` (default) the minimum value of the field will be mapped
            to 0. and the maximum to 1.
        probes_kind : "relative" | "absolute"
            The key for  probes_z values. If relative, they are expressed
            relatively to min and max field values (0. <-> min, 1. <-> max) ;
            if absolute they are used directly
        output : bool
            passed to `Virtual_layer` constructor
        """
        super().__init__(postname, func, output)
        # Will only become RGBA if 
        # - bool masked & mask_color has transparency
        # - float masked
        self.colormap = colormap
        self.probe_z = np.asarray(probes_z)
        self.probes_kind = probes_kind
        # Init all modifiers to empty list
        self._modifiers = []
        self._twin_field = None

    @property
    def mode(self):
        if self.mask_kind is None:
            return "RGB"
        elif self.mask_kind == "bool":
            maskcolorlen = len(self.mask[1])
            mode_from_maskcolorlen = {
                3: "RGB",
                4: "RGBA"
            }
            return mode_from_maskcolorlen[maskcolorlen]
        elif self.mask_kind == "float":
            return "RGBA"
        else:
            raise ValueError(self.mask_kind)

    def overlay(self, layer, overlay_mode):
        """
        Combine the output of current layer with an overlayed layer *after*
        color-mapping.

        Parameters
        ----------
        layer : `Color_layer` | `Grey_layer`
            The layer to overlay. See `Overlay_mode` documentation for the
            options and restrictions.
        overlay_mode : `Overlay_mode`
            Object describing the kind of overlay used.
        """
        self._modifiers += [("overlay", layer, overlay_mode)]

    def shade(self, layer, lighting):
        """
        Adds scene lighting to the layer through a normal map effect.

        Parameters
        ----------
        layer : `Normal_map_layer`
            The layer holding the normal vector field.
        lighting : `Blinn_lighting`
            Object describing the scene lighting.
        """
        self._modifiers += [("shade", layer, lighting)]

    def set_twin_field(self, layer, scale):
        """ 
        Combine two postprocessing in one.

        Combine the current layer with another layer *before* color-mapping.
        (i.e. add directly the underlying float
        `fractalshades.postproc.Postproc` arrays), tuning the intensity of
        the effect with a scaling coefficient.

        Parameters
        ----------
        layer : `Virtual_Layer` instance
            The combined layer
        scale : float
            The scaling coefficient ; if 1. both layer will have the same
            min-max contribtion. If 0.1 the combined layer will only contribute
            to 10 % (This is based on min and max values reached for each
            layer)

        Notes
        -----

        .. note::
            Note : The mask of the current layer will be applied.
        """
        if layer is None:
            self._twin_field = None
        else:
            self._twin_field = (layer, scale)
            if self.mask is not None:
                layer.set_mask(self.mask[0])

    def crop(self, chunk_slice):
        """ private - Return the image for this chunk"""
        # 1) The "base" image
        arr = self[chunk_slice]
        # If a user-mapping is defined, apply it
        if self.func is not None:
            arr = self.func(arr)

        # is there a twin-field ? If yes we add it here, before colormaping
        if self._twin_field is not None:
            twin_layer, scale = self._twin_field
            if not twin_layer._scaling_defined:
                raise RuntimeError("Twin layer should be computed before")
            k = scale * (self.max - self.min
                         ) / (twin_layer.max - twin_layer.min)
            twin_func = twin_layer.func
            if twin_func is None:
                arr += k * twin_layer[chunk_slice]
            else:
                arr += k * twin_func(twin_layer[chunk_slice])

        # colorize from colormap 
        # taking into account probes for scaling
        if self.probes_kind == "relative":
            probes = (self.probe_z * self.max
                      + (1. - self.probe_z) * self.min)
        elif self.probes_kind == "absolute":
            probes = self.probe_z
        else:
            raise ValueError(self.probes_kind, "not in [relative, absolute]")
        rgb = self.colormap.colorize(arr, probes)

        # Apply the modifiers
        for (kind, layer, option) in self._modifiers:
            if kind == "shade":
                rgb = self.apply_shade(rgb, layer, option, chunk_slice)
            elif kind == "overlay":
                rgb = self.apply_overlay(rgb, layer, option, chunk_slice)
            else:
                raise ValueError("kind :", kind)

        rgb = np.uint8(rgb * 255)
        crop = PIL.Image.fromarray(self.np2PIL(rgb))
        if self.mask_kind is None:
            return crop

        # Here we have a mask, apply it
        (ix, ixx, iy, iyy) = chunk_slice
        crop_size = (ixx-ix, iyy-iy)
        mask_layer, mask_color = self.mask
        self.apply_mask(crop, crop_size,  mask_layer[chunk_slice], mask_color)
        return crop

    def apply_mask(self, crop, crop_size, mask_arr, mask_color):
        """ private - apply the mask to the image
        """
        mask_kind = self.mask_kind
        crop_mask = PIL.Image.fromarray(self.np2PIL(
                                        np.uint8(255 * mask_arr)))

        if mask_kind == "bool":
            lx, ly = crop_size
            mask_colors = np.tile(np.array(mask_color), crop_size
                                  ).reshape([lx, ly, len(mask_color)])
            mask_colors = self.np2PIL(np.uint8(255 * mask_colors))
            mask_colors = PIL.Image.fromarray(mask_colors, mode=self.mode)

            if self.mode == "RGBA":
                crop.putalpha(255)
            crop.paste(mask_colors, (0, 0), crop_mask)

        elif mask_kind == "float":
            # TODO should scale !!
            crop.putalpha(crop_mask)

        else:
            raise ValueError(mask_kind)

    @staticmethod
    def apply_shade(rgb, layer, lighting, chunk_slice):
        """ private - apply the shading to rgb color array
        """
        if not(isinstance(layer, Normal_map_layer)):
            raise ValueError("Can only shade with Normal_map_layer")
        normal = layer[chunk_slice]
        nf, nx, ny = normal.shape
        complex_n = np.empty(shape=(nx, ny), dtype=np.complex64)
        # max slope (from layer property) used for renormalisation
        coeff = np.sin(layer.max_slope) / (np.sqrt(2.) * layer.max)
        complex_n.real = normal[0, :, :] * coeff
        complex_n.imag = normal[1, :, :] * coeff
        return lighting.shade(rgb, complex_n)

    @staticmethod
    def apply_overlay(rgb, layer, overlay_mode, chunk_slice):
        """ private - apply the overlay to rgb color array
        """
        return overlay_mode.overlay(rgb, layer, chunk_slice)


class Grey_layer(Virtual_layer):
    default_mask_color = 0.
    mode = "L"

    def __init__(self, postname, func, curve=None, probes_z=[0., 1.],
                 probes_kind="relative", output=True):
        """
        A grey layer.
        
        Parameters
        ----------
        postname : str
            passed to `Virtual_layer` constructor
        func :
            passed to `Virtual_layer` constructor
        curve : None | callable
            A mapping from [0, 1] to [0, 1] applied *after* rescaling (this is 
            the equivalent of a colormap in greyscale)
        probes_z : 2-floats list
            The preprocessing affine rescaling. If [0., 1.] and probes_kind is 
            `relative` (default) the minimum value of the field will be mapped
            to 0. and the maximum to 1.
        probes_kind : "relative" | "absolute"
            The key for  probes_z values. If relative, they are expressed
            relatively to min and max field values (0. <-> min, 1. <-> max) ;
            if absolute they are used directly
        output : bool
            passed to `Virtual_layer` constructor
        """
        super().__init__(postname, func, output)
        # Will only become RGBA if masked & mask_color has transparency
        self.curve = curve
        self.probe_z = np.asarray(probes_z)
        self.probes_kind = probes_kind

    def set_mask(self, layer, mask_color=None):
        """
        Masks the base-layer

        Parameters
        ----------
        layer : Bool_layer | Grey_layer
            The layer mask
        mask_color : None | float belonging to [0., 1.]
            If None, a default value will be provided depending on the masked
            array subclass
            If a value is provided, it shall be with :
            a float belonging to [0. (black) ; 1. (white)] interval
        """
        if mask_color is None: # default class-implementation
            mask_color = self.default_mask_color
        self.mask = (layer, mask_color)

    @property
    def mode(self):
        return "L"

    def crop(self, chunk_slice):
        """ Return the image for this chunk"""
        # 1) The "base" image
        arr = self[chunk_slice]
        # If a user-mapping is defined, apply it
        if self.func is not None:
            arr = self.func(arr)

        # taking into account probes for scaling
        if self.probes_kind == "relative":
            probes = (self.probe_z * self.max
                      + (1. - self.probe_z) * self.min)
        elif self.probes_kind == "absolute":
            probes = self.probe_z
        else:
            raise ValueError(self.probes_kind, "not in [relative, absolute]")
            
        # clip values in excess, rescale to [0, 1]
        arr = np.clip(arr, probes[0], probes[1])
        arr = (arr - probes[0]) / (probes[1] - probes[0])

        # then apply the transfert curve if provided
        if self.curve is not None:
            arr = self.curve(arr)

        # export to image format
        grey = self.get_grey(arr)
        crop = PIL.Image.fromarray(self.np2PIL(grey))
        if self.mask_kind is None:
            return crop

        # Here we have a mask, apply it
        (ix, ixx, iy, iyy) = chunk_slice
        crop_size = (ixx-ix, iyy-iy)
        mask_layer, mask_color = self.mask
        self.apply_mask(crop, crop_size,  mask_layer[chunk_slice], mask_color)
        return crop

    def get_grey(self, arr):
        return np.uint8(arr * 255)
    
    @property
    def mask_kind(self):
        """ private - 3 possiblilities : 
                - no mask (returns None)
                - boolean mask (return "bool")
                - greyscale i.e. float mask (return "float) """
        if self.mask is None:
            return None
        layer, mask_color = self.mask
        if isinstance(layer, Bool_layer):
            return "bool"
        else:
            raise ValueError("Mask layer type not allowed here"
                             + type(layer).__name__)

    def apply_mask(self, crop, crop_size, mask_arr, mask_color):
        """ private - apply the mask to the image
        """

        if self.mask_kind == "bool":
            crop_mask = PIL.Image.fromarray(self.np2PIL(
                    self.get_grey(mask_arr)))
            lx, ly = crop_size
            mask_colors = np.tile(np.array(mask_color), crop_size
                                  ).reshape([lx, ly])
            mask_colors = self.np2PIL(np.uint8(255 * mask_colors))
            mask_colors = PIL.Image.fromarray(mask_colors, mode=self.mode)
            crop.paste(mask_colors, (0, 0), crop_mask)
        else:
            raise ValueError(self.mask_kind)

class Disp_Layer(Grey_layer):
    """
    Grey layer with 32-bits precision.
    
    Same functionnality as a `Grey_layer` but the image export is with 32-bits
    precision (useful when used as an height map for 3d processing like
    Blender)
    """
    mode = "I"
    def get_grey(self, arr):
        return np.int32(arr * 65535) # 2**16-1

class Normal_map_layer(Color_layer):
    default_mask_color = (0.5, 0.5, 1.)

    def __init__(self, postname, max_slope=70, output=True):
        """  Defines a normal map layer that can be exported to blender
        (OpenGL normal map format - normal = (2*color)-1 // on each component)
        color = 0.5 * (normal + 1)
        `max_slope`: maximal angle (in degree) of the normal map, will be used
                     for re-normalisation
        """
        super().__init__(postname, None, colormap=None, output=output)
        self.max_slope = max_slope * np.pi / 180
    
    def crop(self, chunk_slice):
        """ Return the image for this chunk"""
        # 1) The "base" image
        arr = self[chunk_slice]
        

#        print('crop bool', arr, arr.shape, arr.dtype, arr[0:100])
#        rgb = np.uint8(rgb * 255)
        (ix, ixx, iy, iyy) = chunk_slice
        rgb =  np.zeros([ixx - ix, iyy - iy, 3], dtype=np.float32)
        
        # max slope (from layer property) used for renormalisation
        coeff = np.sin(self.max_slope) / (np.sqrt(2.) * self.max)
        rgb[:, :, 0] = - arr[0, :, :] * coeff # nx component
        rgb[:, :, 1] = - arr[1, :, :] * coeff # ny component

        # The final normal field need to be of norm 1 
        k = np.sqrt(rgb[:, :, 0]**2 + rgb[:, :, 1]**2 + 1.)
        rgb[:, :, 0] = rgb[:, :, 0] / k
        rgb[:, :, 1] = rgb[:, :, 1] / k
        rgb[:, :, 2] = 1. / k        

        rgb = 0.5 * (rgb + 1.)
        crop = PIL.Image.fromarray(self.np2PIL(np.uint8(255 * rgb)))

        if self.mask_kind is None:
        # Note : mask_color=(0.5, 0.5, 1) for this type of layer
            return crop

        # We have a mask, apply it
        (ix, ixx, iy, iyy) = chunk_slice
        crop_size = (ixx-ix, iyy-iy)
        mask_layer, mask_color = self.mask
        mask_color
        self.apply_mask(crop, crop_size,
                        mask_layer[chunk_slice], mask_color) #, self.mask_kind)
        return crop


class Bool_layer(Virtual_layer):
    def __init__(self, postname, output=True):
        """ Defines a boolean mask array.
        Where is 1 or True, the attached masked layer values are considered
        masked.
        """
        super().__init__(postname, None, output)
        self.mode = "1"

    def __getitem__(self, chunk_slice):
        arr = super().__getitem__(chunk_slice)
        bool_arr = np.array(arr, dtype=bool)
        return bool_arr

    def crop(self, chunk_slice):
        """ Return the image for this bool layer chunk"""
        # 1) The "base" image
        arr = self[chunk_slice]
        crop_mask = PIL.Image.fromarray(self.np2PIL(arr))
        return crop_mask


def _2d_rgb_to_XYZ(rgb, nx, ny):
    res = fscolors.Color_tools.rgb_to_XYZ(rgb.reshape(nx * ny, 3))
    rgb.reshape(nx, ny, 3) # restaure original shape
    return res.reshape(nx, ny, 3)

def _2d_XYZ_to_rgb(XYZ, nx, ny):
    res = fscolors.Color_tools.XYZ_to_rgb(XYZ.reshape(nx * ny, 3))
    return res.reshape(nx, ny, 3)

def _2d_XYZ_to_CIELab(XYZ, nx, ny):
    res = fscolors.Color_tools.XYZ_to_CIELab(XYZ.reshape(nx * ny, 3))
    return res.reshape(nx, ny, 3)

def _2d_CIELab_to_XYZ(Lab, nx, ny):
    res = fscolors.Color_tools.CIELab_to_XYZ(Lab.reshape(nx * ny, 3))
    return res.reshape(nx, ny, 3)
    

class Blinn_lighting:
    def __init__(self, k_Ambient, color_ambient):
        """
        This class holds the properties for the scene lightsources. Its
        instances can be passed as a parameter to `Color_layer.shade`.
        
        Parameters
        ==========
        k_Ambient : 
            ambient lighting coefficient. A typical value is 0.2
        color_ambient: 3-uplet float
            a RGB 3-tuple, channels values in [0, 1] : the ambient lighting
            color. Usually white (1., 1., 1.)
        """
        # ref : https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
        # http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
        self.k_Ambient = k_Ambient
        self.color_ambient = np.asarray(color_ambient)
        self.light_sources = []

    def add_light_source(self, k_diffuse, k_specular, shininess,
                         angles, coords=None, color=np.array([1., 1., 1.]),
                         material_specular_color=None):
        """
        Adds a lightsource to the scene
        
        Parameters
        ==========
        k_diffuse : 3-uplet float
            diffuse lighting coefficients, as RGB
        k_specular : 3-uplet float
            diffuse lighting pourcentage, as RGB
        shininess : float
            Phong exponent for specular model
        angles : 2-uplet float
            angles (theta, phi) of light source at infinity, in degree
            theta = 0 : aligned to x
            phi = 0 : normal to z (= in xy plane) 
        coords : 2-uplet float
            position of lightsource from image center,
            in pourcentage of current image
            (if None (default), source is at infinity ; 
            if non-null, LS_angle has no effect)
        """
        self.light_sources += [{
            "k_diffuse": np.asarray(k_diffuse),
            "k_specular": np.asarray(k_specular),
            "shininess": shininess,
            "_angles": angles, 
            "angles": tuple(a * np.pi / 180. for a in angles),
            "coords": coords,
            "color": np.asarray(color),
            "material_specular_color": material_specular_color
        }]

    def shade(self, rgb, normal):
        nx, ny, _ = rgb.shape
        XYZ = _2d_rgb_to_XYZ(rgb, nx, ny)
        XYZ_shaded = XYZ * self.k_Ambient * self.color_ambient
        for ls in self.light_sources:
            XYZ_shaded += self.partial_shade(ls, XYZ, normal)
        return _2d_XYZ_to_rgb(XYZ_shaded, nx, ny)

    def partial_shade(self, ls, XYZ, normal):
        theta_LS, phi_LS = ls['angles']

        # Light source coordinates
        LSx = np.cos(theta_LS) * np.cos(phi_LS)
        LSy = np.sin(theta_LS) * np.cos(phi_LS)
        LSz = np.sin(phi_LS) 

        # Normal vector coordinates
        nx = normal.real
        ny = normal.imag
        nz = np.sqrt(1. - nx**2 - ny**2) # sin of max_slope
        lambert = LSx * nx + LSy * ny + LSz * nz
        np.putmask(lambert, lambert < 0., 0.)
        
        # half-way vector coordinates - Blinn Phong shading
        specular = np.zeros_like(lambert)
        if ls["k_specular"] != 0.:
            phi_half = (np.pi * 0.5 + phi_LS) * 0.5
            half_x = np.cos(theta_LS) * np.cos(phi_half)
            half_y = np.sin(theta_LS) * np.cos(phi_half)
            half_z = np.sin(phi_half)
            spec_angle = half_x * nx + half_y * ny + half_z * nz
            np.putmask(spec_angle, spec_angle < 0., 0.)
            specular = np.power(spec_angle, ls["shininess"])

        if ls["material_specular_color"] is None:
            res =  (ls["k_diffuse"] * lambert[:, :, np.newaxis] * XYZ
                    + ls["k_specular"] * specular[:, :, np.newaxis]  * XYZ)
        else:
            XYZ_sp = np.asarray(ls["material_specular_color"])
            res =  (ls["k_diffuse"] * lambert[:, :, np.newaxis] * XYZ
                    + ls["k_specular"] * specular[:, :, np.newaxis]  * XYZ_sp)
                   
        return res * ls["color"]

        
class Overlay_mode:
    def __init__(self, mode, **mode_options):
        """
        This class holds the properties for overlaying. Its instances can be
        passed as a parameter to `Color_layer.overlay` method.

        Parameters
        ----------
        mode:   "alpha_composite" | "tint_or_shade"
                the kind of overlay  applied
                
                - alpha_composite :
                    alpha-compositing of 2 color layers (its is
                    the mask of the upper layer which will be considered for
                    compositing)
                - tint_or_shade : 
                    the color of the lower layer will be adjusted
                    (tinted of shaded) depending of the upper layer value
                    (tinted above 0.5, shaded below 0.5). The upper layer in
                    this case shall be a `Grey_layer` instance.

        mode_options : a dict with the applicable option for each mode:

        Optional Parameters
        -------------------
        `shade_type`:  for alpha_composite 
        `ref_white`:   for tint_or_shade
        """
        self.mode = mode
        self.mode_options = mode_options

    def overlay(self, rgb, layer, chunk_slice):
        if self.mode == "alpha_composite":
            return self.alpha_composite(rgb, layer, chunk_slice)
        elif self.mode == "tint_or_shade":
            return self.tint_or_shade(rgb, layer, chunk_slice)

    def alpha_composite(self, rgb, overlay, chunk_slice):
        """ Paste a "masked layer" over a standard layer
        layer should be a colored layer with a mask 
        The base layer should not have a mask as otherwise it could be applied
        later in postprocessing
        """
        if not isinstance(overlay, Color_layer):
            raise ValueError("{} not allowed for alpha compositing".format(
                             type(overlay).__name__ ))
        # For alpha compositing, we obviously need an alpha channel
        if overlay.mask is None:
            raise ValueError("alpha_composite called with non-masked overlay "
                             "layer")
        # This is safer than using the A-channel from the image
        mask_arr = overlay.mask[0][chunk_slice][:, :, np.newaxis]

        nx, ny, _ = rgb.shape
        XYZ = _2d_rgb_to_XYZ(rgb, nx, ny)

        # overlay crop : first get the raw image than convert to numpy
        # dumping the alpha channel
        crop = np.array(overlay.crop(chunk_slice))
        rgb2 = Virtual_layer.PIL2np(crop)[:, :, :3] / 255.
        XYZ_2 = _2d_rgb_to_XYZ(rgb2, nx, ny)
        
        # Apply the alpha_composite
        res = mask_arr * XYZ + (1. - mask_arr) * XYZ_2
        # Second pass needed to handle Nan values
        res = np.where(mask_arr==0, XYZ_2, res)
        res = np.where(mask_arr==1, XYZ, res)
        alpha_composite = _2d_XYZ_to_rgb(res, nx, ny)

        return alpha_composite


    def tint_or_shade(self, rgb, overlay, chunk_slice):
        """ 
        """
        if not isinstance(overlay, Grey_layer):
            raise ValueError("{} not allowed for tint_and_shade "
                "overlay".format(type(overlay).__name__ ))

        opt = self.mode_options
        ref_white = opt.get("ref_white", fscolors.Color_tools.Lab_ref_white)
        k_pegtop = opt.get("pegtop", 0.)
        k_Lch = opt.get("Lch", 0.)
        blend_T = k_pegtop + k_Lch
        # Default if nothing is provided
        if blend_T == 0.:
            k_pegtop = 4.0
            k_Lch = 1.0
            blend_T = k_pegtop + k_Lch
        if (k_pegtop < 0) or (k_Lch < 0):
            raise ValueError(k_pegtop, k_Lch)

        blend_T = k_pegtop + k_Lch
        # This is safer than using the A-channel from the image
        mask_arr = None
        if overlay.mask is not None:
            mask_arr = overlay.mask[0][chunk_slice][:, :, np.newaxis]
            raise NotImplementedError("Still in TODO", mask_arr.shape)

        # overlay crop : first get the raw image than convert to numpy
        crop = np.array(overlay.crop(chunk_slice))
        shade = Virtual_layer.PIL2np(crop) / 255.
        shade = shade[:, :, np.newaxis]
        
        # shade = overlay[chunk_slice][:, :, np.newaxis]
#        if np.any(shade > 1.):
#            print(np.max(shade), np.min(shade))
#            raise ValueError()
        
        nx, ny, _ = rgb.shape
        XYZ = _2d_rgb_to_XYZ(rgb, nx, ny)
        
        if np.any(rgb > 1.):
            raise ValueError()
        XYZ_pegtop = np.zeros([nx, ny, 3])
        XYZ_Lch = np.zeros([nx, ny, 3])


        if k_pegtop != 0:
            XYZ_pegtop = (2. * shade * XYZ 
                          + (1. - 2. * shade) * XYZ**2 / ref_white)
        if k_Lch != 0:
            XYZ_Lch = self.shade_Lch(XYZ, shade)
        XYZ = (XYZ_pegtop * k_pegtop + XYZ_Lch * k_Lch) / blend_T

        # Convert modified hsv back to rgb.
        blend = _2d_XYZ_to_rgb(XYZ, nx, ny)
        if np.any(blend > 1.):
            raise ValueError()

        return blend

    @staticmethod
    def shade_Lch(XYZ, shade):
        # Only if Greyscale shade
        shade = shade[:, :, 0]
        shade = 2. * shade - 1.
        nx, ny, _ = XYZ.shape
        Lab = _2d_XYZ_to_CIELab(XYZ, nx, ny)
        L = Lab[:, :, 0]
        a = Lab[:, :, 1]
        b = Lab[:, :, 2]
        lighten = shade > 0

        Lab[:, :, 0] = np.where(lighten,
                                L  + shade * (100. - L),  L * (1. +  shade))
        Lab[:, :, 1] = np.where(lighten,
                                a  - shade**2 * a, a * (1. -  shade**2))
        Lab[:, :, 2] = np.where(lighten,
                                b  - shade**2 * b, b * (1. -  shade**2))

        return _2d_CIELab_to_XYZ(Lab, nx, ny)
        