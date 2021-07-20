# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.colors
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_agg import FigureCanvasAgg


import PIL
import PIL.ImageQt
import os



#import fractalshades.numpy_utils.xrange as fsx
#import fractalshades.settings as fssettings
#import fractalshades.utils as fsutils

#def mkdir_p(path):
#    """ Creates directory ; if exists does nothing """
#    try:
#        os.makedirs(path)
#    except OSError as exc:
#        if exc.errno == errno.EEXIST and os.path.isdir(path):
#            pass
#        else:
#            raise exc

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


class Fractal_colormap():
    def __new__(cls, colors, kinds, grad_npts, grad_funcs=None,
                extent="mirror"):
        """
        Fractal_colormap factory, concatenates (n-1) color gradients (passing
        through n colors).

        colors : rgb float np.array of shape [n, 3]
        kinds: arrays of string [n-1] , "Lab" or "Lch"
        grad_npts : int np.array of size [n-1] : number of internal points
            used, for each gradient
        grad_funcs : [n-1] array of callables mapping [0, 1.] to [0., 1] passed
            to each gradient. Default to identity.
        extent : scalar, ["clip", "mirror", "repeat"] What to do with out of
            range values.
        """      
        
        # Provides sensible defaults if a scalar is provided
        npts = colors.shape[0] - 1
        if isinstance(kinds, str):
            kinds = [kinds] * npts
        if isinstance(grad_npts, int):
            grad_npts = [grad_npts] * npts
        if grad_funcs is None:
            grad_funcs = lambda x: x
        if callable(grad_funcs):
            grad_funcs = [grad_funcs] * npts
        
        # If color 2 not provided... we roll and cut the last item
        # colors2 = np.roll(colors, -1, axis=0)
        colors1 = colors[:-1]
        colors2 = colors[1:]
#        kinds = kinds[:-1]
#        n = n[:-1]
#        funcs = funcs[:-1]
#        npts -= 1

        fc = sum(tuple(_Fractal_colormap(Color_tools.color_gradient(
                       kinds[ipt], colors1[ipt, :], colors2[ipt, :], grad_npts[ipt],
                       grad_funcs[ipt])) for ipt in range(1, npts)),
                 start=_Fractal_colormap(Color_tools.color_gradient(
                       kinds[0], colors1[0, :], colors2[0, :], grad_npts[0],
                       grad_funcs[0])))
        # Stores "constructor" vals
        fc.colors = np.asarray(colors)
        fc.kinds = np.asarray(kinds)
        fc.grad_npts = np.asarray(grad_npts)
        fc.grad_funcs = np.asarray(grad_funcs)
        fc.extent = extent

        return fc


#    def params(self):
#        return (self.kinds, self.colors1, self.colors2, self.funcs, 
#                self.extent)
    # Disables operations, as "constructor" vals will not follow
    def __neg__(self):
        raise NotImplementedError()
    def __add__(self, other):
        raise NotImplementedError()
    def __sub__(self, other):
        raise NotImplementedError()


class _Fractal_colormap():
    # TODO : should be refactored so that it takes an array for which each line
    # is a list of argument to Color gradient array from Colortool class,
    # Each of them added to make the colormap
    # Discontinue support for matplotlib colormaps
    """
    Class responsible for mapping a real array to a colors array.
    Attributes :
        *_colors* Internal list of possible colors, [0 to self.n_colors]
        *_probes* list of indices in self._colors array, identifying the
                  transitions between differrent sections of the colormap. Each
                  of this probe is mapped to *_probe_value*, either given by the
                  user or computed at plot time.
    """
    def __init__(self, color_gradient, extent="mirror"):
        """
Creates a colormap from a color gradient array (as output by Color_tools
gradient functions, array of shape (n_colors, 3))

*color_gradient*  a Color gradient array from Colortool class
*extent*  ["mirror", "repeat", "clip"] specifies what to do with out of range Values.
        """
        self._colors = color_gradient
        n_colors, _ = color_gradient.shape
        self._probes = np.array([0, n_colors-1], dtype=np.float32)
#        self.quantiles_ref = None
        self.extent = extent

    @property
    def n_colors(self):
        """ Total number of colors in the colormap. """
        return self._colors.shape[0]

    @property
    def npts(self):
        return self.colors.shape[0]

    @property
    def probes(self):
        """ Position of the "probes" ie transitions between the different parts
        of the colormap. Read-only. """
        return np.copy(self._probes)

    def __neg__(self):
        """
        Returns a reversed colormap
        """
        other = _Fractal_colormap(self._colors[::-1, :])
        other._probes = self._probes[-1] - self._probes[::-1]
        return other

    def __add__(self, other):
        """ Concatenates 2 Colormaps """
        fcm = _Fractal_colormap(np.vstack([self._colors, other._colors[1:]]))
#        print("self._probes", self._probes)
#        print("other._probes", other._probes)
#        print("other._probes + (self._probes[-1])", other._probes + (self._probes[-1]))
        #self._probes[-1] += 0.5
        fcm._probes = np.concatenate([
            self._probes,
            (other._probes + (self._probes[-1]))[1:]
            ])
        return fcm

    def __sub__(self, other):
        """ Sbstract a colormap ie adds its reversed version """
        return self.__add__(other.__neg__())

    def _output(self, nx, ny):
        margin = 1
        nx_im = nx - 2 * margin
        ny_im = ny - 2 * margin
        img = np.repeat(np.linspace(0., 1., nx_im)[:, np.newaxis],
                        ny_im, axis=1)
        img = self.colorize(img, np.linspace(0., 1., len(self._probes)))
        img = np.uint8(img * 255.999)
        B = np.ones([nx, ny, 3], dtype=np.uint8) * 255
        B[margin:nx - margin, margin:ny - margin, :] = img
        return np.swapaxes(B, 0, 1)

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

## Defines the neutral element for cmap addition
#EMPTY_CMAP = _Fractal_colormap(np.array([]).reshape([0, 3]))
#EMPTY_CMAP._probes = np.array([-1])
