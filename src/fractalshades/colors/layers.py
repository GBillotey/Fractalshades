# -*- coding: utf-8 -*-
import logging

import numpy as np
import numba
import PIL
import PIL.ImageQt

import fractalshades.numpy_utils.expr_parser as fs_parser
import fractalshades.colors as fscolors

#        Note: Pillow image modes:
#        RGB (3x8-bit pixels, true color)
#        RGBA (4x8-bit pixels, true color with transparency mask)
#        1 (1-bit pixels, black and white, stored with one pixel per byte)
#        L ((8-bit pixels, black and white)
#        I (32-bit signed integer pixels)
"""
fractal -> postproc -> Secondaryla -> layer -> combined layer (blend, shading)

"""

logger = logging.getLogger(__name__)

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


    def get_postproc_index(self):
        """ Refactoring code common to get_postproc_batch / getitem """
        plotter = self.plotter
        postname = self.postname
        try:
            field_count = 1
            post_index = list(plotter.postnames).index(self.postname)
            return (field_count, post_index)
        except ValueError:
            # Could happen that we need 2 fields (e.g., normal map...)
            if postname not in self.plotter.postnames_2d:
                raise ValueError("postname not found: {}".format(postname))
            post_index_x = list(self.plotter.postnames).index(postname + "_x")
            post_index_y = list(self.plotter.postnames).index(postname + "_y")
            if post_index_y != post_index_x + 1:
                raise ValueError(
                    "x y coords not contiguous for postname: {}".format(
                        postname)
                )
            field_count = 2
            return (field_count, (post_index_x, post_index_y))

    def postproc_batch(self):
        """ layer -> postproc -> batch """
        postname = self.postname
        for pbatch in self.plotter.postproc_batches:
            if postname in pbatch.postnames():
                return pbatch
            if postname in pbatch.postnames_2d:
                return pbatch
        raise ValueError(
            f"Postname not found in this plotter: {postname}"
        )

    def __getitem__(self, chunk_slice):
        """ read the base data array for this layer
        Returns a numpy array of 
        - shape (lx, ly) or (n_fields, lx, ly) if n_fields > 1
        - datatype self.dtype
        
        Note: this is the raw, unscaled values
        """
        plotter = self.plotter
        dtype = plotter.post_dtype
        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
        
        ssg = plotter.supersampling
        if ssg is not None:
            nx *= ssg
            ny *= ssg
        
        field_count, post_index = self.get_postproc_index()
        
        if field_count == 1:
            arr = np.empty((nx, ny), dtype)
            ret = plotter.get_2d_arr(post_index, chunk_slice)
            if ret is None:
                return None
            arr[:] = ret
    
        elif field_count == 2:
            (post_index_x, post_index_y) = post_index
            arr = np.empty((2, nx, ny), dtype)
            ret0 = plotter.get_2d_arr(post_index_x, chunk_slice)
            ret1 = plotter.get_2d_arr(post_index_y, chunk_slice)
            if (ret0 is None) or (ret1 is None):
                return None
            arr[0, :] = ret0
            arr[1, :] = ret1
            
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
        else:
            raise ValueError(f"arr shape: {sh}")
        
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

    def __init__(self, postname, func, colormap, probes_z=[0, 1], output=True):
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
            probes_z = (z_min, zmax) ; the z_min value of the field will be
            mapped to 0. and zmax to 1.
        output : bool
            passed to `Virtual_layer` constructor
        """
        super().__init__(postname, func, output)
        # Will only become RGBA if 
        # - bool masked & mask_color has transparency
        # - float masked
        self.colormap = colormap
        self.probe_z = np.asarray(probes_z)
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
            A scaling coefficient which will be applied to the twin layer field
            before adding it to the base field

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
            k = scale 
            twin_func = twin_layer.func
            if twin_func is None:
                arr += k * twin_layer[chunk_slice]
            else:
                arr += k * twin_func(twin_layer[chunk_slice])

        probes = self.probe_z
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
        nx, ny = ixx - ix, iyy - iy

        ssg = self.plotter.supersampling
        if ssg is not None:
            nx *= ssg
            ny *= ssg

        crop_size = (nx, ny)
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
            # TODO Not tested 
            crop.putalpha(crop_mask)

        else:
            raise ValueError(mask_kind)

    @staticmethod
    def apply_shade(rgb, layer, lighting, chunk_slice):
        """ private - apply the shading to rgb color array
        """
        if not(isinstance(layer, Normal_map_layer)):
            raise ValueError("Can only shade with a Normal_map_layer")
        normal = layer[chunk_slice]
        nf, nx, ny = normal.shape
        complex_n = np.empty(shape=(nx, ny), dtype=np.complex64)
        # max slope (from layer property) used for renormalisation
        # /!\ Responsability of the caller to ensure that nx**2 + ny**2 <= 1.
        coeff = np.sin(layer.max_slope)
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
    k_int = 255

    def __init__(self, postname, func, curve=None, probes_z=[0., 1.],
                 output=True):
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
            The preprocessing affine rescaling. If [min, max] the minimum value
            of the field will be mapped
            to 0. and the maximum to 1.
        
        output : bool
            passed to `Virtual_layer` constructor
        """
        super().__init__(postname, func, output)
        # Will only become RGBA if masked & mask_color has transparency
        self.curve = curve
        self.probe_z = np.asarray(probes_z)

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

    def crop(self, chunk_slice):
        """ Return the image for this chunk"""
        # 1) The "base" image
        arr = self[chunk_slice]

        # If a user-mapping is defined, apply it
        if self.func is not None:
            arr = self.func(arr)

        probes = self.probe_z

        # arr = np.clip(arr, probes[0], probes[1])
        # wraps values in excess, rescale to [0, 1]
        # Formula: https://en.wikipedia.org/wiki/Triangle_wave
        # 4/p * (t - p/2 * floor(2t/p + 1/2)) * (-1)**floor(2t/p + 1/2) where p = 4
        arr = (arr - probes[0]) / (probes[1] - probes[0])
        e = np.floor((arr + 1.) / 2.)
        arr = np.abs((arr - 2. * np.floor(e)) * (-1)**e)

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
        nx, ny = ixx - ix, iyy - iy

        ssg = self.plotter.supersampling
        if ssg is not None:
            nx *= ssg
            ny *= ssg

        crop_size = (nx, ny)

        mask_layer, mask_color = self.mask
        self.apply_mask(crop, crop_size,  mask_layer[chunk_slice], mask_color)
        return crop

    def get_grey(self, arr):
        # Fits to numerical range
        return self.dtype(arr * self.k_int)

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
            lx, ly = crop_size
#            mask_arr = np.tile(np.array(mask_arr), crop_size
#                ).reshape([lx, ly])
            crop_mask = PIL.Image.fromarray(
                self.np2PIL(np.uint8(mask_arr * 255)), mode="L"
            )
            mask_colors = np.tile(np.array(mask_color), crop_size
                ).reshape([lx, ly])
            mask_colors = PIL.Image.fromarray(
                    self.np2PIL(self.get_grey(mask_colors)),
                    mode=self.mode
            )
            crop.paste(mask_colors, (0, 0), crop_mask)
        else:
            raise ValueError(self.mask_kind)

class Disp_layer(Grey_layer):
    """
    Grey layer with 32-bits precision.
    
    Same functionnality as a `Grey_layer` but the image export is with 32-bits
    precision (useful when used as an height map for 3d processing like
    Blender)
    """
    mode = "I"
    # According to Pillow doc
    # "a 32-bit signed integer has a range of 0-65535". why? cf source code:
    # https://github.com/python-pillow/Pillow/blob/7bf5246b93cc89cfb9d6cca78c4719a943b10585/src/PIL/PngImagePlugin.py#L693-L708
    k_int = 65535


class Normal_map_layer(Color_layer):
    default_mask_color = (0.5, 0.5, 1.)

    def __init__(self, postname, max_slope=70, output=True):
        """  
        Defines a normal map layer
        
        This layer can be either:

        - plotted directly (OpenGL normal map format)
          and used in a post-processing Workflow - e.g. Blender 

        - or used to apply shading to a `Color_layer.shade`

        Parameters
        ----------
        postname : str
            passed to `Virtual_layer` constructor
        max_slope : float
            maximal angle (in degree) of the normal map, will be used
            for re-normalisation
        output : bool
            passed to `Virtual_layer` constructor
        """
        super().__init__(postname, None, colormap=None, output=output)
        self.max_slope = max_slope * np.pi / 180
    
    def crop(self, chunk_slice):
        """ Return the image for this chunk"""
        # 1) The "base" image
        arr = self[chunk_slice]

        # Note: rgb = np.uint8(rgb * 255)
        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy

        ssg = self.plotter.supersampling
        if ssg is not None:
            nx *= ssg
            ny *= ssg

        rgb =  np.zeros((nx, ny, 3), dtype=np.float32)
        
        # max slope (from layer property) used for renormalisation
        coeff = np.sin(self.max_slope) / np.sqrt(2.)
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
        nx, ny = ixx - ix, iyy - iy

        ssg = self.plotter.supersampling
        if ssg is not None:
            nx *= ssg
            ny *= ssg

        crop_size = (nx, ny)
        mask_layer, mask_color = self.mask
        self.apply_mask(crop, crop_size,
                        mask_layer[chunk_slice], mask_color)
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
    def __init__(self, k_ambient, color_ambient, **light_sources):
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
        self.k_ambient = k_ambient
        self.color_ambient = np.asarray(color_ambient)
        self.light_sources = []

        for ls in light_sources.values():
            self.add_light_source(**ls)


    def add_light_source(
        self, k_diffuse, k_specular, shininess, polar_angle, azimuth_angle,
        color=np.array([1., 1., 1.]), material_specular_color=None
    ):
        """
        Adds a lightsource to the scene
        
        Parameters
        ==========
        k_diffuse: 3-uplet float
            diffuse lighting coefficients, as RGB
        k_specular: 3-uplet float
            diffuse lighting pourcentage, as RGB
        shininess: float
            Phong exponent for specular model
        polar_angle: float
            angle (theta) of light source direction, in degree
            theta = 0 if incoming light is in xOz plane
        azimuth_angle: float
            angle (phi) of light source direction, in degree
            phi = 0 if incoming light is in xOy plane
        """
        angles = (polar_angle, azimuth_angle)
        self.light_sources += [{
            "k_diffuse": np.asarray(k_diffuse),
            "k_specular": np.asarray(k_specular),
            "shininess": shininess,
            "polar_angle": polar_angle,
            "azimuth_angle": azimuth_angle,
            "angles_radian": tuple(a * np.pi / 180. for a in angles),
            "color": np.asarray(color),
            "material_specular_color": material_specular_color
        }]

    def shade(self, rgb, normal):
        nx, ny, _ = rgb.shape
        XYZ = _2d_rgb_to_XYZ(rgb, nx, ny)
        XYZ_shaded = XYZ * self.k_ambient * self.color_ambient
        for ls in self.light_sources:
            XYZ_shaded += self.partial_shade(ls, XYZ, normal)
        return _2d_XYZ_to_rgb(XYZ_shaded, nx, ny)

    def partial_shade(self, ls, XYZ, normal):
        theta_LS, phi_LS = ls['angles_radian']

        # Light source coordinates
        LSx = np.cos(theta_LS) * np.cos(phi_LS)
        LSy = np.sin(theta_LS) * np.cos(phi_LS)
        LSz = np.sin(phi_LS) 

        # Normal vector coordinates
        nx = normal.real
        ny = normal.imag
        nz = np.sqrt(1. - nx**2 - ny**2) # cos of max_slope
        lambert = LSx * nx + LSy * ny + LSz * nz
        np.putmask(lambert, lambert < 0., 0.)

        # half-way vector coordinates - Blinn Phong shading
        specular = np.zeros_like(lambert)
        if ls["k_specular"] != 0.:
            # half azimuth angle vector between light and view
            phi_half = (np.pi * 0.5 + phi_LS) * 0.5
            half_x = np.cos(theta_LS) * np.cos(phi_half)
            half_y = np.sin(theta_LS) * np.cos(phi_half)
            half_z = np.sin(phi_half)

            specular_coeff = half_x * nx + half_y * ny + half_z * nz
            np.putmask(specular_coeff, specular_coeff < 0., 0.)
            specular = np.power(specular_coeff, ls["shininess"])

        if ls["material_specular_color"] is None:
            res =  (ls["k_diffuse"] * lambert[:, :, np.newaxis] * XYZ
                    + ls["k_specular"] * specular[:, :, np.newaxis]  * XYZ)
        else:
            XYZ_sp = np.asarray(ls["material_specular_color"])
            res =  (ls["k_diffuse"] * lambert[:, :, np.newaxis] * XYZ
                    + ls["k_specular"] * specular[:, :, np.newaxis]  * XYZ_sp)

        return res * ls["color"]
    
    def _output(self, nx, ny):
        """ Return a RGB uint8 array of shape (nx, ny, 3)
        """
        margin = 1
        nx_im = nx - 2 * margin
        ny_im = ny - 2 * margin

        normal = np.empty((nx_im, ny_im), dtype=np.complex128)
        rgb = 0.5 * np.ones((nx_im, ny_im, 3), dtype=np.float64)
        _Blinn_lighting_sphere_fill_normal_vec(0.75, normal)
        img = self.shade(rgb, normal)
        img = np.uint8(img * 255.999)

        B = np.ones([nx, ny, 3], dtype=np.uint8) * 255
        B[margin:nx - margin, margin:ny - margin, :] = img
        return np.flip(np.swapaxes(B, 0, 1), axis=0)


    def output_ImageQt(self, nx, ny):
#https://stackoverflow.com/questions/34697559/pil-image-to-qpixmap-conversion-issue
        B = self._output(nx, ny)
        return PIL.ImageQt.ImageQt(PIL.Image.fromarray(B))

    
    #----------- methods for GUI interaction
    @property
    def n_rows(self):
        return len(self.light_sources)

    def modify_item(self, col_key, irow, value):
        """ In place modification of lightsource irow """
        if col_key == "color":
            self.light_sources[irow][col_key] = np.asarray(value)
        # Keep aligned the angles in radian
        else:
            self.light_sources[irow][col_key] = float(value)
            if col_key == "polar_angle":
                self.light_sources[irow]["angles_radian"] = (
                    float(value) * np.pi / 180.,
                    self.light_sources[irow]["angles_radian"][1]
                )
            if col_key == "azimuth_angle":
                self.light_sources[irow]["angles_radian"] = (
                    self.light_sources[irow]["angles_radian"][0],
                    float(value) * np.pi / 180.,
                )


    def col_data(self, col_key):
        """ Returns a column of the expected field """
        ret = []
        for ls in self.light_sources:
            ret += [ls[col_key]]

        return ret

    @property
    def default_ls_kwargs(self):
        return {
            "k_diffuse": 1.8,
            "k_specular": 2.0,
            "shininess": 15.,
            "polar_angle": 50.,
            "azimuth_angle": 20.,
            "color": np.array([1.0, 1.0, 0.95])
        }

    def adjust_size(self, n_ls):
        """ Adjust the number of lightsources to n_ls"""
        n_ls_old = self.n_rows
        diff = n_ls - n_ls_old
        if diff < 0:
            del self.light_sources[diff:]
        else:
            for _ in range(diff):
                self.add_light_source(**self.default_ls_kwargs)

    def script_repr(self, indent=0):
        """ Return a string that can be used to restore the colormap
        """
        k_ambient_str = repr(self.k_ambient)
        color_ambient_str = np.array2string(self.color_ambient, separator=', ')
        lightings_str = ""
        for ils, ls in enumerate(self.light_sources):
            k_diffuse = ls["k_diffuse"]
            k_specular = ls["k_specular"]
            shininess = ls["shininess"]
            polar_angle = ls["polar_angle"]
            azimuth_angle = ls["azimuth_angle"]
            ls_color_str = np.array2string(ls["color"], separator=', ')
            matcolor = ls["material_specular_color"]
            ls_matcolor_str = (
                None if matcolor is None
                else np.array2string(matcolor, separator=', ')
            )

            lightings_str += (
                f"    ls{ils}={{\n        "
                f"'k_diffuse': {k_diffuse},\n        "
                f"'k_specular': {k_specular},\n        "
                f"'shininess': {shininess},\n        "
                f"'polar_angle': {polar_angle},\n        "
                f"'azimuth_angle': {azimuth_angle},\n        "
                f"'color': {ls_color_str},\n        "
                f"'material_specular_color': {ls_matcolor_str}\n"
                "    },\n"
            )
        ret =  (
            "fs.colors.layers.Blinn_lighting(\n"
            "    k_ambient={},\n"
            "    color_ambient={},\n"
            "{})"
        ).format(k_ambient_str, color_ambient_str, lightings_str)

        shift = " " * (4 * (indent + 1))
        ret.replace("\n", "\n" + shift)
        return ret



@numba.njit
def _Blinn_lighting_sphere_fill_normal_vec(kr, normal):
    # Fills in-place the Open-GL normal map field for a sphere
    # kr scalar, 0 < kr < 1 (usually ~ 0.75)
    # normal: 2d vec
    (nx_im, ny_im) = normal.shape
    center_x = nx_im / 2.
    center_y = ny_im / 2.
    r_sphe = min(center_x, center_y) * kr

    for i in range(nx_im):
        for j in range(ny_im):
            ipix = (i - center_x) / r_sphe
            jpix = (j - center_y) / r_sphe
            rloc = np.hypot(ipix, jpix)
            if rloc > 1.:
                # Flat surface
                normal[i, j] = 0.
            else:
                phi = np.arctan2(jpix, ipix)
                alpha = np.arcsin(rloc) # 0 where r=0, pi/2 where r=1
                normal[i, j] = np.sin(alpha) * np.exp(1j * phi)


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
                    alpha-compositing of 2 color layers (a Bool_layer may be
                    provided as optionnal parameter, otherwise its is
                    the mask of the upper layer which will be considered for
                    compositing)
                - tint_or_shade : 
                    the color of the lower layer will be adjusted
                    (tinted of shaded) depending of the upper layer value
                    (tinted above 0.5, shaded below 0.5). The upper layer in
                    this case shall be a `Grey_layer` instance.

        Other Parameters
        ----------------
        ref_white: 3-uplet float, Optionnal
            reference white light RGB coeff for ``mode`` "tint_or_shade"
             Default to CIE LAB constants Illuminant D65
        alpha_mask: `Bool_layer`, Optionnal
            Boolean array used for compositing, for alpha_composite mode
            Defaults to the mask of the upper layer
        inverse_mask: bool, Optionnal
            if True, inverse the mask
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
        layer should be a colored layer with a mask, or the mask provided
        separately
        Note: The base layer shall not have a mask: otherwise it will be
        applied later in postprocessing destroying the intended effect.
        """
        if not isinstance(overlay, Color_layer):
            raise ValueError("{} not allowed for alpha compositing".format(
                             type(overlay).__name__ ))

        opt = self.mode_options

        # For alpha compositing, we obviously need an alpha channel
        if "alpha_mask" in opt.keys():
            mask_arr = opt["alpha_mask"][chunk_slice][:, :, np.newaxis]
        else:
            if overlay.mask is None:
                raise ValueError(
                    "alpha_composite called with non-masked overlay "
                    "layer - Add a mask to overlay layer or provide the "
                    "mask separately through alpha_mask parameter."
            )
            # This is safer than using the A-channel from the image
            mask_arr = overlay.mask[0][chunk_slice][:, :, np.newaxis]

        inverse_mask = opt.get("inverse_mask", False)
        if inverse_mask:
            mask_arr = ~mask_arr 

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
        