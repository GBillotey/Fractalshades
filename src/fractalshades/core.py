# -*- coding: utf-8 -*-
import os
import sys
import fnmatch
import copy
import tempfile
import datetime
import pickle

import numpy as np
from numpy.lib.format import open_memmap
import mpmath
import PIL
import numba

import fractalshades as fs
import fractalshades.numpy_utils.xrange as fsx
import fractalshades.settings as fssettings
import fractalshades.utils as fsutils

from fractalshades.mprocessing import Multiprocess_filler



"""

"""

class Fractal_plotter():
    def  __init__(self, postproc_batch):
        """
        A Fractal plotter
        - point to a single Fractal
        - can hold several postproc_batches each one point to a single
          (fracal, calculation) couple
         - can hold several post-processing layers
        """
        # postproc_batchescan be single or an enumeration
        postproc_batches = postproc_batch
        if isinstance(postproc_batch, fs.postproc.Postproc_batch):
            postproc_batches = [postproc_batch]
        for i, postproc_batch in enumerate(postproc_batches):
            if i == 0:
                self.postproc_batches = [postproc_batch]
                self.posts = copy.copy(postproc_batch.posts)
                self.postnames_2d = postproc_batch.postnames_2d
                # iterator :
                self.fractal = postproc_batch.fractal
            else:
                self.add_postproc_batch(postproc_batch)

        self.chunk_slices = self.fractal.chunk_slices
        # layer data
        self.layers = []
        self.scalings = None # to be computed !
        # Plotting directory
        self.plot_dir = self.fractal.directory
    
    @property
    def postnames(self):
        return self.posts.keys()
    
    @property
    def size(self):
        f = self.fractal
        return (f.nx, f.ny)

    def add_postproc_batch(self, postproc_batch):
        """
        Adds another post-processing batch & register the associated postproc
        names. `postproc_batch` shall map to the same fractal.
        Note : several postproc_batches are needed whenever different 
        calculations need to be combined in an output plot, as a postproc
        batch can only map to a unique calc_name.
        """
        if postproc_batch.fractal != self.fractal:
            raise ValueError("Attempt to add a postproc_batch from a different"
                             "fractal: {} from {}".format(
                postproc_batch.fractal, postproc_batch.calc_name))
        self.postproc_batches += [postproc_batch]
        self.posts.update(postproc_batch.posts)
        self.postnames_2d += postproc_batch.postnames_2d

    def add_layer(self, layer):
        """
        Adds a layer field
        Layer postname shall have been already registered in one of the plotter
        batches.        
        When a layer is added, a link layer-> Fractal_plotter is 
        created ; a layer can only point to a single plotter.
        """
        postname = layer.postname
        if postname not in (list(self.postnames) + self.postnames_2d):
            raise ValueError("Layer `{}` shall be registered in "
                             "Fractal_plotter postproc_batches: {}".format(
            postname, list(self.postnames) + self.postnames_2d))
        self.layers += [layer]
        layer.link_plotter(self)

    def __getitem__(self, layer_name):
        """ Get the layer by its postname
        """
        for layer in self.layers:
            if layer.postname == layer_name:
#                print("checking", layer.postname, layer_name)
                return layer
        raise KeyError("Layer {} not in available layers: {}".format(
                layer_name, list(l.postname for l in self.layers)))

    def plot(self):
        self.store_postprocs()
        self.compute_scalings()
        self.write_postproc_report()
        self.open_images()
        self.push_layers_to_images()
        self.save_images()

    def store_postprocs(self):
        """ Computes and stores posprocessed data in a temporary mmap
        """
        self.open_temporary_mmap()
        inc_posproc_rank = 0
        self.postname_rank = dict()
        
        for batch in self.postproc_batches:
            self.store_temporary_mmap(
                    chunk_slice=None,
                    batch=batch,
                    inc_posproc_rank=inc_posproc_rank)
            for i, postname in enumerate(batch.postnames):
                self.postname_rank[postname] = inc_posproc_rank + i
            inc_posproc_rank += len(batch.posts)

    def temporary_mmap_path(self):
        """ Path to the temporary memmap used to stored plotting arrays"""
        # from tempfile import mkdtemp
        return os.path.join(
            self.fractal.directory, "data", "_plotter.tpm")

    def open_temporary_mmap(self):
        """
        Creates the memory mappings for postprocessed arrays
        Note: We expand to 2d format
        """
        f = self.fractal
        nx, ny = (f.nx, f.ny)
        n_pp = len(self.posts)
        mmap = open_memmap(
            filename=self.temporary_mmap_path(), 
            mode='w+',
            dtype=f.float_postproc_type,
            shape=(n_pp, nx, ny),
            fortran_order=False,
            version=None)
        del mmap

    @Multiprocess_filler(iterable_attr="chunk_slices",
        iter_kwargs="chunk_slice", veto_multiprocess=False)
    def store_temporary_mmap(self, chunk_slice, batch, inc_posproc_rank):
        """ Compute & store temporary arrays for this postproc batch
            Note : inc_posproc_rank rank shift to take into account potential
            other batches for this plotter.
        """
        f = self.fractal
        inc = inc_posproc_rank
        post_array, chunk_mask = self.fractal.postproc(batch, chunk_slice)
        arr_2d = f.reshape2d(post_array, chunk_mask, chunk_slice)
        n_posts, cx, cy = arr_2d.shape
        (ix, ixx, iy, iyy) = chunk_slice
        mmap = open_memmap(filename=self.temporary_mmap_path(), mode='r+')
        mmap[inc:inc+n_posts, ix:ixx, iy:iyy] = arr_2d

    # All methods needed for plotting
    def compute_scalings(self):
        """ Compute the scaling for all layer field
        (needed for mapping to color) """
        for layer in self.layers:
            self.compute_layer_scaling(chunk_slice=None, layer=layer)
    
    def write_postproc_report(self):
        report_path = os.path.join(
            self.fractal.directory, type(self).__name__ + ".txt")

        def write_layer_report(i, layer, report):
            report.write(" - Layer #{} :\n".format(i))
            postname = layer.postname
            report.write("post-processing: `{}`\n".format(postname))
            report.write("kind: {}\n".format(type(layer).__name__))
            report.write("func: {}\n".format(layer._func_arg))

            mask = layer.mask
            if mask is None:
                mask_str = "None"
            else:
                mask_str = "{} `{}` with mask color: {}".format(
                    type(mask[0]).__name__,
                    mask[0].postname,
                    mask[1])
            report.write("mask: {}\n".format(mask_str))

            report.write("output: {}\n".format(layer.output))
            report.write("min: {}\n".format(layer.min))
            report.write("max: {}\n\n".format(layer.max))
            
        with open(report_path, 'w', encoding='utf-8') as report:
            for i, layer in enumerate(self.layers):
                write_layer_report(i, layer, report)


    @Multiprocess_filler(iterable_attr="chunk_slices",
        iter_kwargs="chunk_slice", veto_multiprocess=True)
    def compute_layer_scaling(self, chunk_slice, layer):
        """ Compute the scaling for this layer """
        layer.update_scaling(chunk_slice)

    def open_images(self):
        self._im = []
        for layer in self.layers:
            if layer.output:
                self._im += [PIL.Image.new(mode=layer.mode, size=self.size)]
            else:
                self._im += [None]

    def save_images(self):
        for i, layer in enumerate(self.layers):
            if not(layer.output):
                continue
            file_name = "{}_{}".format(type(layer).__name__, layer.postname)
            base_img_path = os.path.join(self.plot_dir, file_name + ".png")
            self.save_tagged(self._im[i], base_img_path, self.fractal.params)

    def save_tagged(self, img, img_path, tag_dict):
        """
        Saves *img* to png format at *path*, tagging with *tag_dict*.
        https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_PNG_files
        """
        pnginfo = PIL.PngImagePlugin.PngInfo()
        for k, v in tag_dict.items():
            pnginfo.add_text(k, str(v))
        img.save(img_path, pnginfo=pnginfo)

    def push_layers_to_images(self):
        for i, layer in enumerate(self.layers):
            if not(layer.output):
                continue
            self.push_cropped(chunk_slice=None, layer=layer, im=self._im[i])

    @Multiprocess_filler(iterable_attr="chunk_slices",
        iter_kwargs="chunk_slice", veto_multiprocess=True)
    def push_cropped(self, chunk_slice, layer, im):
        """ push "cropped image" from layer for this chunk to the image"""
        (ix, ixx, iy, iyy) = chunk_slice
        ny = self.fractal.ny
        crop_slice = (ix, ny-iyy, ixx, ny-iy)
        paste_crop = layer.crop(chunk_slice)
        im.paste(paste_crop, box=crop_slice)

        



class Fractal():
    
    REPORT_ITEMS = [
        "chunk1d_begin",
        "chunk1d_end",
        "iref",
        "glitch_max_attempt",
        "chunk_pts",
        "total-glitched",
        "dyn-glitched"]

    # Note : chunk_mask is pre-computed and saved also but not at the same 
    # stage (at begining of calculation)
    SAVE_ARRS = [
        "Z",
        "U",
        "stop_reason",
        "stop_iter"]
    
    """
    Abstract base class
    Derived classes are reponsible for computing, storing and reloading the raw
    data composing a fractal image.
    
    ref:
https://en.wikibooks.org/wiki/Pictures_of_Julia_and_Mandelbrot_Sets/The_Mandelbrot_set
    """

    def __init__(self, directory):
        """
        Parameters
    *directory*   The working base directory
    *complex_type*  numpy type or ("Xrange", numpy type)
        """
        self.directory = directory
        self.iref = None # None when no reference point used / needed
        self.glitch_max_attempt = 0
        self.subset = None
        self.glitch_stop_index = None

    def init_data_types(self, complex_type):
        if type(complex_type) is tuple:
            type_modifier, _ = complex_type
            if type_modifier != "Xrange":
                raise ValueError(type_modifier)
        self.complex_type = complex_type
        self.float_postproc_type = np.float32
        self.termination_type = np.int8
        self.int_type = np.int32

    @fsutils.zoom_options
    def zoom(self, *,
             x: float,
             y: float,
             dx: float,
             nx: int,
             xy_ratio: float,
             theta_deg: float,
             projection: str="cartesian",
             antialiasing: bool=False):
        """
        Parameters
    *x* *y*  coordinates of the central point
    *dx*     reference data range (on x axis). Definition depends of the
        projection selected, for 'cartesian' (default) this is the total span.
    *nx*     number of pixels of the image along x axis
    *xy_ratio*  ratio of dx / dy and nx / ny
    *theta_deg*    Pre-rotation of the calculation domain
    *complex_type*  numpy type or ("Xrange", numpy type)
    *projection*   "cartesian" "spherical" "exp_map"
        """
        # We're all set, the job is done by `zoom_options` wrapper...

    def run(self):
        """
        Lauch a full calculation
        """
        
        if not(self.res_available()):
            # We write the param file and initialize the
            # memmaps for progress reports and calc arrays
            # It is not process safe so we dot it before entering multi-processing
            # loop
            fsutils.mkdir_p(os.path.join(self.directory, "data"))
            self.open_report_mmap()
            self.open_data_mmaps()
            self.save_params()
        
        # Lazzy compilation of subset boolean array chunk-span
        self._mask_beg_end = None

        # JIT-compiled function
        # self.jitted_numba_cycles = numba.njit(numba_cycles)
        self._iterate = self.iterate()
        self.cycles()


    @property
    def ny(self):
        return int(self.nx / self.xy_ratio + 0.5)

    @property
    def dy(self):
        return self.dx / self.xy_ratio

    @property
    def px(self):
        if not(hasattr(self, "_px")): # is None:
            self._px = self.dx / self.nx
        return self._px

    @property
    def multiprocess_dir(self):
        """ Directory used for multiprocess stdout stderr streams redirection
        """
        return os.path.join(self.directory, "multiproc_calc")

    @property
    def Xrange_complex_type(self):
        """ Return True if the data type is a xrange array
        """
        if type(self.complex_type) is tuple:
            type_modifier, _ = self.complex_type
            return type_modifier == "Xrange"
        return False

    @property
    def base_complex_type(self):
        complex_type = self.complex_type
        if type(complex_type) is tuple:
            _, complex_type = complex_type
        return complex_type

    @property
    def base_float_type(self):
        select = {np.dtype(np.complex64): np.dtype(np.float32),
                  np.dtype(np.complex128): np.dtype(np.float64)}
        return select[np.dtype(self.base_complex_type)]

    @property    
    def params(self):
        """ Used to tag an output image or check if data is already computed
        and stored
        """
        software_params = {
                "Software": "fractalshades " + fs.__version__,
                "fractal_type": type(self).__name__,
                # "debug": ("1234567890" * 10), # tested 10000 chars ok
                "datetime": datetime.datetime.today().strftime(
                        '%Y-%m-%d_%H:%M:%S')}
        zoom_params = self.zoom_options
        calc_function = self.calc_options_lastcall # TODO rename to calc_callable
        calc_params = self.calc_options

        res = dict(software_params)
        res.update(zoom_params)
        res["calc-function"] = calc_function
        res.update({"calc-param_" + k: v for (k, v) in calc_params.items()})

        return res


    def clean_up(self, calc_name):
        """ Deletes all data files associated with a given calc_name
        """
        for pattern in [
                calc_name + "_*.arr",
                calc_name + ".params",
                calc_name + ".report",
                calc_name + "_pt*.ref",
                calc_name + "_pt*.sa"
        ]:
            data_dir = os.path.join(self.directory, "data")
            if not os.path.isdir(data_dir):
                return
            with os.scandir(data_dir) as it:
                for entry in it:
                    if (fnmatch.fnmatch(entry.name, pattern)):
                        os.unlink(entry.path)
            
    @property
    def pts_count(self):
        """ Return the total number of points for the current calculation 
        taking into account the `subset` parameter """
        if self.subset is not None:
#            print("in pts_count", self.subset)
#            print("in pts_count", np.sum(self.subset[None]),
#                  np.count_nonzero(self.subset[None]))
            return np.count_nonzero(self.subset[None])
        else:
            return self.nx * self.ny


    # The various method associated with chunk mgmt ===========================
    def chunk_slices(self): #, chunk_size=None):
        """
        Generator function
        Yields the chunks spans (ix, ixx, iy, iyy)
        with each chunk of size chunk_size x chunk_size
        """
        # if chunk_size is None:
        chunk_size = fssettings.chunk_size

        for ix in range(0, self.nx, chunk_size):
            ixx = min(ix + chunk_size, self.nx)
            for iy in range(0, self.ny, chunk_size):
                iyy = min(iy + chunk_size, self.ny)
                yield  (ix, ixx, iy, iyy)

    @property
    def chunks_count(self):
        """
        Return the total number of chunks (tiles) for the current image
        """
        chunk_size = fssettings.chunk_size
        (cx, r) = divmod(self.nx, chunk_size)
        if r != 0:
            cx += 1
        (cy, r) = divmod(self.ny, chunk_size)
        if r != 0: cy += 1
        return cx * cy

    def chunk_rank(self, chunk_slice):
        """
        Return the generator yield index for chunk_slice
        """
        chunk_size = fssettings.chunk_size
        (ix, _, iy, _) = chunk_slice
        chunk_item_x = ix // chunk_size
        chunk_item_y = iy // chunk_size
        (cy, r) = divmod(self.ny, chunk_size)
        if r != 0: cy += 1
        return chunk_item_x * cy + chunk_item_y

    def chunk_from_rank(self, rank):
        """
        Return the chunk_slice from the generator yield index
        """
        chunk_size = fssettings.chunk_size
        (cy, r) = divmod(self.ny, chunk_size)
        if r != 0: cy += 1
        chunk_item_x, chunk_item_y = divmod(rank, cy)
        ix = chunk_item_x * chunk_size
        iy = chunk_item_y * chunk_size
        ixx = min(ix + chunk_size, self.nx)
        iyy = min(iy + chunk_size, self.ny)
        return (ix, ixx, iy, iyy)

    def mask_beg_end(self, rank):
        """ Return the span for the boolean mask index for the chunk index
        `rank` """
        if self._mask_beg_end is None:
            arr = np.empty((self.chunks_count + 1,), dtype=np.int32)
            arr[0] = 0
            for i, chunk_slice in enumerate(self.chunk_slices()):
                (ix, ixx, iy, iyy) = chunk_slice
                arr[i + 1] = arr[i] + (ixx - ix) * (iyy - iy)
            self._mask_beg_end = arr
        return self._mask_beg_end[rank: rank + 2]

    def chunk_pts(self, chunk_slice):
        """
        Return the number of compressed 1d points for this chunk_slice
        (taking into account 2d subset bool if available)
        """
        subset = self.subset
        (ix, ixx, iy, iyy) = chunk_slice
        if subset is not None:
            subset_pts = np.count_nonzero(subset[chunk_slice]) # TODO : test this
            return subset_pts
        else:
            return (ixx - ix) * (iyy - iy)

    @property
    def chunk_mask(self):#, chunk_slice):
        """ Legacy - simple alias """
        return self.subset
#        subset = self.subset
#        if subset is not None:
#            return ~subset # np.ravel(subset[chunk_slice])
#        else:
#            return None

    def c_chunk(self, chunk_slice):
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

        offset = self.chunk_offset(chunk_slice)
        return (x + offset[0]) + (y + offset[1]) * 1j # TODO test this


    def chunk_offset(self, chunk_slice, ensure_Xr=False):
        """
        Only computes the delta around ref central point for different projections
        Note : return type is always standard prec - standard or extended range
        
        ensure_Xr : enforce extended range if True
        """
#        select = {np.complex256: np.float128,
#                  np.complex128: np.float64}
        data_type = self.base_float_type # select[self.base_complex_type]

        (xy_ratio, theta_deg)  = (self.xy_ratio, self.theta_deg)
        (nx, ny, dx, dy) = (self.nx, self.ny, self.dx, self.dy)

        if self.Xrange_complex_type or ensure_Xr:
            dx_m, dx_exp = mpmath.frexp(dx)
            dx_m = np.array(dx_m, data_type)
            dx = fsx.Xrange_array(dx_m, dx_exp)
            dy_m, dy_exp = mpmath.frexp(dy)
            dy_m = np.array(dy_m, data_type)
            dy = fsx.Xrange_array(dy_m, dy_exp)
        else:
            dx = float(dx)
            dy = float(dy)
        
        (ix, ixx, iy, iyy) = chunk_slice
        theta = theta_deg / 180. * np.pi

        dx_grid = np.linspace(-0.5, 0.5, num=nx, dtype=data_type)
        dy_grid = np.linspace(-0.5, 0.5, num=ny, dtype=data_type)
        dy_vec, dx_vec  = np.meshgrid(dy_grid[iy:iyy], dx_grid[ix:ixx])

        if self.antialiasing:
            rg = np.random.default_rng(0)
            dx_vec += (0.5 - rg.random(dx_vec.shape, dtype=data_type)) * 0.5 / nx
            dy_vec += (0.5 - rg.random(dy_vec.shape, dtype=data_type)) * 0.5 / ny

        dx_vec = dx * dx_vec
        dy_vec = dy * dy_vec

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
            h_max = 2. * np.pi * xy_ratio # max h reached on the picture
            xbar = (dx_vec + 0.5 * dx - dy) / dx * h_max # 0 .. hmax
            ybar = dy_vec / dy * 2. * np.pi              # -pi .. +pi
            rho = dx * 0.5 * np.where(xbar > 0., np.exp(xbar), 0.)
            phi = ybar + theta
            dx_vec = (dx_vec + 0.5 * dx - 0.5 * dy) * xy_ratio
            dy_vec = dy_vec * xy_ratio
            offset = [np.where(xbar <= 0.,
                          (dx_vec * np.cos(theta)) - (dy_vec * np.sin(theta)),
                          rho * np.cos(phi)),
                      np.where(xbar <= 0.,
                          (dx_vec * np.sin(theta)) + (dy_vec * np.cos(theta)),
                          rho * np.sin(phi))]

        elif self.projection == "exp_map":
            # only exp. map
            h_max = 2. * np.pi * xy_ratio # max h reached on the picture
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
        data_type = self.base_float_type

        xy_ratio  = self.xy_ratio
        (nx, ny, dx, dy) = (self.nx, self.ny, self.dx, self.dy)
        (ix, ixx, iy, iyy) = chunk_slice
        
        if not(self.Xrange_complex_type):
            dx = float(dx)
            dy = float(dy)
        else:
            dx_m, dx_exp = mpmath.frexp(dx)
            dx_m = np.array(dx_m, data_type)
            dx = fsx.Xrange_array(dx_m, dx_exp)
            dy_m, dy_exp = mpmath.frexp(dy)
            dy_m = np.array(dy_m, data_type)
            dy = fsx.Xrange_array(dy_m, dy_exp)

        dx_grid = dx * np.linspace(-0.5, 0.5, num=nx, dtype=data_type)
        dy_grid = dy * np.linspace(-0.5, 0.5, num=ny, dtype=data_type)
        dy_vec, dx_vec  = np.meshgrid(dy_grid[iy:iyy], dx_grid[ix:ixx])

        if self.projection == "cartesian":
            px = (dx / (nx - 1.)) #* np.ones_like(dx_vec)

        elif self.projection == "spherical":
            raise NotImplementedError()

        elif self.projection == "mixed_exp_map":
            raise NotImplementedError()

        elif self.projection == "exp_map":
            h_max = 2. * np.pi * xy_ratio
            xbar = (dx_vec + 0.5 * dx - dy) / dx * h_max
            px = (dx / (nx - 1.)) * 0.5 * h_max * np.exp(xbar)

        else:
            raise ValueError("Projection not implemented: {}".format(
                              self.projection))
        return px



    def param_matching(self, dparams):
        """
        Test if the stored parameters match those of new calculation
        /!\ modified in subclass
        """
        print("**CALLING param_matching +++", self.params)
        # TODO : note: when comparing iref should be disregarded ? 
        # or subclass specific implementation
        UNTRACKED = ["datetime", "debug"] 
        for key, val in self.params.items():
            if not(key in UNTRACKED) and dparams[key] != val:
                print("Unmatching", key, val, "-->", dparams[key])
                return False
#            print("its a match", key, val, dparams[key] )
        print("** all good")
        return True

    def res_available(self, chunk_slice=None): #, calc_name=None):
        """  
        If chunk_slice is None, check that stored calculation parameters
        matches.
        If chunk_slice is provided, checks that calculation results are
        available up to current self.iref
        """
        try:
            params, codes = self.reload_params()
        except IOError:
            return False
        matching = self.param_matching(params) # TODO if cal_name...
        if not(matching):
            return False
        if chunk_slice is None:
            return matching # True

        try:
            report = self.reload_report(chunk_slice)
        except IOError:
            return False

        if self.iref is None:
            return report["iref"] >= -1 # -2 means not yet calculated
        else:
            completed = (report["iref"] >= self.iref)
            not_needed = (report["total-glitched"] == 0)
            return (not_needed or completed)


    @Multiprocess_filler(iterable_attr="chunk_slices",
                         redirect_path_attr="multiprocess_dir",
                         iter_kwargs="chunk_slice")
    def cycles(self, chunk_slice=None, SA_params=None):
        """
        Fast-looping for Julia and Mandelbrot sets computation.

        Parameters
        *initialize*  function(Z, U, c) modify in place Z, U (return None)
        *iterate*   function(Z, U, c, n) modify place Z, U (return None)

        *subset*   bool arrays, iteration is restricted to the subset of current
                   chunk defined by this array. In the returned arrays the size
                    of axis ":" is np.sum(subset[ix:ixx, iy:iyy]) - see below
        *codes*  = complex_codes, int_codes, termination_codes
        *calc_name* prefix identifig the data files
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

#        print("**CALLING cycles +++")
        if self.res_available(chunk_slice):
            return

#        print("init_cycling_arrays", type(self))
        if self.iref is None:
            (c, Z, U, stop_reason, stop_iter, n_stop, bool_active,
             index_active, n_iter) = self.init_cycling_arrays(chunk_slice)
            SA_iter = 0
        else:
            (c, Z, U, stop_reason, stop_iter, n_stop, bool_active,
             index_active, n_iter, SA_iter, ref_div_iter, ref_path
             ) = self.init_cycling_arrays(chunk_slice, SA_params)
        modified_in_cycle = np.copy(bool_active)

        iterate = self._iterate # Suppressed the ()
        iref = self.iref

#        print("**/CALLING cycles looping,  n_stop", n_stop)
#        print("**/active: ", np.count_nonzero(bool_active),
#              np.shape(bool_active))

        if iref is None:
            # Standard iterations
#            print("calling numba_cycles", numba_cycles, type(numba_cycles), type(self.jitted_numba_cycles))
            numba_cycles(Z, U, c, stop_reason, stop_iter, bool_active,
                         n_iter, iterate)
        else:
            # Perturbation iterations
            last_iref = (iref == self.glitch_max_attempt) 
            numba_cycles_perturb(Z, U, c, stop_reason, stop_iter, bool_active,
                iref, n_iter, SA_iter, ref_div_iter, ref_path, iterate,
                last_iref)

        # Saving the results after cycling
        self.update_report_mmap(chunk_slice, stop_reason)
        self.update_data_mmaps(chunk_slice, Z, U, stop_reason, stop_iter,
                               modified_in_cycle)


    def init_cycling_arrays(self, chunk_slice):
        """
        Prepared the chunk arrays for subsequent looping
        """
        c = np.ravel(self.c_chunk(chunk_slice))
        if self.subset is not None:
            c = c[self.chunk_mask[chunk_slice]]
#        c = self._2d_to_1d(self.c_chunk(chunk_slice), chunk_slice)

        (n_pts,) = c.shape
        n_Z, n_U, n_stop = (len(code) for code in self.codes)

        if self.Xrange_complex_type:
            Z = fsx.Xrange_array.zeros([n_Z, n_pts], # [n_Z, n_pts],
                                       dtype=self.base_complex_type)
        else:
            Z = np.zeros([n_Z, n_pts], dtype=self.complex_type)
        U = np.zeros([n_U, n_pts], dtype=self.int_type)
        stop_reason = -np.ones([1, n_pts], dtype=self.termination_type)
        stop_iter = np.zeros([1, n_pts], dtype=self.int_type)

        self.initialize()(Z, U, c, chunk_slice)

        # We start at 0 with all index active
        n_iter = 0
        index_active = np.arange(c.size, dtype=self.int_type)
        bool_active = np.ones(c.size, dtype=np.bool)

        return (c, Z, U, stop_reason, stop_iter, n_stop, bool_active,
                index_active, n_iter)


    # ======== The various storing files for a calculation ====================
    @staticmethod
    def filter_stored_codes(codes):
        """ Don't store temporary codes - i.e. those which starts with "_" """
        return list(filter(lambda x: not(x.startswith("_")), codes))

    def params_path(self, calc_name=None):
        if calc_name is None:
            calc_name = self.calc_name
        return os.path.join(
            self.directory, "data", calc_name + ".params")
    
    def serializable_params(self, params):
        """ Some params we do not want to save as-is but only keep partial
        information. Saving them would duplicate information + require coding
        ad-hoc __getstate__, __setstate__ methods.
        """
        unserializable = ("calc-param_subset",)
        ret = {}
        for k, v in params.items():
            if k in unserializable:
                if v is None:
                    ret[k] = v
                else:
                    ret[k] = repr(v)
            else:
                ret[k] = v
        # print("modified params ready to save", ret)
        return ret

    def save_params(self):
        """
        Save (pickle) current calculation parameters in data file,
        Don't save temporary codes - i.e. those which startwith "_"
        This should only be used to tag images and not to re-run a calculation.
        """
        (complex_codes, int_codes, stop_codes) = self.codes
        f_complex_codes = self.filter_stored_codes(complex_codes)
        f_int_codes = self.filter_stored_codes(int_codes)
        saved_codes = (f_complex_codes, f_int_codes, stop_codes)

        save_path = self.params_path()
        fsutils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as tmpfile:
            s_params = self.serializable_params(self.params)
            pickle.dump(s_params, tmpfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(saved_codes, tmpfile, pickle.HIGHEST_PROTOCOL)
#        print("Saved calc params", save_path)

    def reload_params(self, calc_name=None): # public
        save_path = self.params_path(calc_name)
        with open(save_path, 'rb') as tmpfile:
            params = pickle.load(tmpfile)
            codes = pickle.load(tmpfile)
            return (params, codes)

    def report_path(self, calc_name=None): # public
        if calc_name is None:
            calc_name = self.calc_name
        return os.path.join(
            self.directory, "data", calc_name + ".report")

    def open_report_mmap(self): # private
        """
        Create the memory mapping for calculation reports by chunks
        [chunk1d_begin, chunk1d_end,
                    iref, glitch_max_attempt, chunk_pts, chunk_glitched]

        Initialized as:
        [chunk1d_begin, chunk1d_end,
        -2, self.glitch_max_attempt, pts_total, -2]
        """
        items = self.REPORT_ITEMS
        chunks_count = self.chunks_count
        report_cols_count = len(items)

        mmap = open_memmap(
            filename=self.report_path(), 
            mode='w+',
            dtype=np.int32,
            shape=(chunks_count, report_cols_count),
            fortran_order=False,
            version=None)

        mmap[:, items.index("iref")] = -2 # -1 used if calculated
        mmap[:, items.index("total-glitched")] = -1
        mmap[:, items.index("dyn-glitched")] = -1
        mmap[:, items.index("glitch_max_attempt")] = self.glitch_max_attempt

        # Number of points per chunk
        chunk_pts = np.empty((chunks_count,), dtype=np.int32)
        for i, chunk_slice in enumerate(self.chunk_slices()):
            chunk_pts[i] = self.chunk_pts(chunk_slice)
        mmap[:, items.index("chunk_pts")] = chunk_pts

        # full_cumsum is np.cumsum(chunk_pts) with inserted 0
        full_cumsum = np.empty((chunks_count + 1,), dtype=np.int32)
        np.cumsum(chunk_pts, out=full_cumsum[1:])
        full_cumsum[0] = 0
        mmap[:, items.index("chunk1d_begin")] = full_cumsum[:-1]
        mmap[:, items.index("chunk1d_end")] = full_cumsum[1:]

        del mmap

    def update_report_mmap(self, chunk_slice, stop_reason): # private
        """
        """
        items = self.REPORT_ITEMS
        chunk_rank = self.chunk_rank(chunk_slice)
        glitch_stop_index = self.glitch_stop_index
        mmap = open_memmap(filename=self.report_path(), mode='r+')
        total_glitched = dyn_glitched = 0 # Default if no glitch correction 
        if glitch_stop_index is not None:
            total_glitched = np.count_nonzero(stop_reason >= glitch_stop_index)
            dyn_glitched = np.count_nonzero(stop_reason == glitch_stop_index)
        mmap[chunk_rank, items.index("total-glitched")] = total_glitched
        mmap[chunk_rank, items.index("dyn-glitched")] = dyn_glitched
        mmap[chunk_rank, items.index("iref")] = (
                self.iref if (self.iref is not None) else -1)
#        print("report updated", chunk_slice, "iref:", self.iref, "chunk_glitched:",  total_glitched)
        
        del mmap

    def reload_report(self, chunk_slice, calc_name=None): # public
        """ Return a report extract for the given chunk, as a dict
             If no chunk provided, return the full report (header, report)
#        """
        items = self.REPORT_ITEMS
        mmap = open_memmap(filename=self.report_path(calc_name), mode='r')
        if chunk_slice is None:
            report = np.empty(mmap.shape, mmap.dtype)
            #  print(mmap.shape, mmap.dtype)
            report[:, :] = mmap[:, :]
            return  self.REPORT_ITEMS, report
        rank = self.chunk_rank(chunk_slice)
        report = dict(zip(
            items,
            (mmap[rank, items.index(it)] for it in items)
        ))
        return report

    def data_path(self, calc_name=None):
        if calc_name is None:
            calc_name = self.calc_name
        keys = ["chunk_mask"] + self.SAVE_ARRS 
        def file_map(key):
            return os.path.join(self.directory, "data",
                                calc_name + "_" + key + ".arr")
        return dict(zip(keys, map(file_map, keys)))

    def open_data_mmaps(self):
        """
        Creates the memory mappings for calculated arrays
        [chunk_mask, Z, U, stop_reason, stop_iter]
        
        Note : chunk_mask can be initialized here
        """
#        items = self.REPORT_ITEMS
        keys = self.SAVE_ARRS #[""Z", "U", "stop_reason", "stop_iter"]
        data_type = {
            # "chunk_mask": np.bool,
            "Z": self.complex_type, # TODO Xrange array
            "U": self.int_type,
            "stop_reason": self.termination_type,
            "stop_iter": self.int_type,
        }
        data_path = self.data_path()

        pts_count = self.pts_count # the memmap 1st dim
        (complex_codes, int_codes, stop_codes) = self.codes
        # keep only the one which do not sart with "_"
        f_complex_codes = self.filter_stored_codes(complex_codes)
        f_int_codes = self.filter_stored_codes(int_codes)
        n_Z, n_U, n_stop = (len(codes) for codes in 
                            (f_complex_codes, f_int_codes, stop_codes))
        # Followin C row-major order --> arr[x, :] shall be fast
        data_dim = {
            # "chunk_mask": (pts_count,),
            "Z": (n_Z, pts_count),
            "U": (n_U, pts_count),
            "stop_reason": (1, pts_count),
            "stop_iter": (1, pts_count),
        }

        for key in keys:
            mmap = open_memmap(
                filename=data_path[key], 
                mode='w+',
                dtype=data_type[key],
                shape=data_dim[key],
                fortran_order=False,
                version=None)
            del mmap

        # Store the chunk_mask (if there is one) at this stage : it is already
        # known
        # /!\ the size of the chunk_mask is always the same, irrespective of
        # the number of items masked
        if self.subset is not None:
            
            mmap = open_memmap(
                filename=data_path["chunk_mask"], 
                mode='w+',
                dtype=np.bool,
                shape=(self.nx * self.ny,),
                fortran_order=False,
                version=None)
            for i, chunk_slice in enumerate(self.chunk_slices()):
                beg_end = self.mask_beg_end(i)
                mmap[beg_end[0]: beg_end[1]] = self.chunk_mask[chunk_slice]


    def update_data_mmaps(self, chunk_slice, Z, U, stop_reason, stop_iter,
                          modified_in_cycle):
        keys = self.SAVE_ARRS
        items = self.REPORT_ITEMS
        data_path = self.data_path()
        arr_map = {
            "Z": Z,
            "U": U,
            "stop_reason": stop_reason,
            "stop_iter": stop_iter,
        }
        report_mmap = open_memmap(filename=self.report_path(), mode='r')
        rank = self.chunk_rank(chunk_slice)
        beg = report_mmap[rank, items.index("chunk1d_begin")]
        end = report_mmap[rank, items.index("chunk1d_end")]

        # codes mapping - taking into account suppressed fields (starting with
        # "_")
        (complex_codes, int_codes, stop_codes) = self.codes
        # keep only the one which do not sart with "_"
        f_complex_codes = self.filter_stored_codes(complex_codes)
        f_int_codes = self.filter_stored_codes(int_codes)
        n_Z, n_U, n_stop = (len(codes) for codes in 
                            (f_complex_codes, f_int_codes, stop_codes))
        codes_index_map = {
            "Z": (range(n_Z), list(complex_codes.index(f_complex_codes[i]) 
                                   for i in range(n_Z))),
            "U": (range(n_U), list(int_codes.index(f_int_codes[i])
                                     for i in range(n_U))),
            "stop_reason": (range(1), range(1)),
            "stop_iter": (range(1), range(1))
        }

        for key in keys:
            mmap = open_memmap(filename=data_path[key], mode='r+')
            arr = arr_map[key]

            fancy_indexing = np.arange(beg, end, dtype=np.int32)
            fancy_indexing = fancy_indexing[modified_in_cycle]

            for (field, f_field) in zip(*codes_index_map[key]):
                mmap[field, fancy_indexing] = arr[f_field, modified_in_cycle]

    def reload_data(self, chunk_slice, calc_name=None): # public
        """ Reload all strored raw arrays for this chunk : 
        raw_data = chunk_mask, Z, U, stop_reason, stop_iter
        """
        keys = self.SAVE_ARRS
        items = self.REPORT_ITEMS
        # Retrieve 1d-coordinates for this chunck
        report_mmap = open_memmap(filename=self.report_path(calc_name),
                                  mode='r')
        rank = self.chunk_rank(chunk_slice)
        beg = report_mmap[rank, items.index("chunk1d_begin")]
        end = report_mmap[rank, items.index("chunk1d_end")]

        arr = dict()
        data_path = self.data_path(calc_name)
        for key in keys:
            mmap = open_memmap(filename=data_path[key], mode='r')
            arr[key] = mmap[:, beg:end]

        # Here we can t always rely on self.subset, it has to be consistent
        # with calc_name ie loaded from params options
        subset = self.subset
        if calc_name is not None:
            params, _ = self.reload_params(calc_name)
            subset = params["calc-param_subset"]
            
        if subset is not None:
            # /!\ fixed-size irrespective of the mask
            mmap = open_memmap(filename=data_path["chunk_mask"], mode='r')
            beg_end = self.mask_beg_end(rank)
            arr["chunk_mask"] = mmap[beg_end[0]: beg_end[1]]
        else:
            arr["chunk_mask"] = None
        
        return (arr["chunk_mask"], arr["Z"], arr["U"], arr["stop_reason"],
                arr["stop_iter"])


    @staticmethod
    def kind_from_code(code, codes):
        """
        codes as returned by 
        (params, codes) = self.reload_data_chunk(chunk_slice, calc_name)
        Used for "raw" post-processing
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
            raise KeyError("raw data code unknow: " + code, complex_codes,
                           int_codes, "stop_reason", "stop_iter")
        return kind

    @staticmethod
    def reshape2d(chunk_array, chunk_mask, chunk_slice):
        """
        Returns 2d versions of the 1d stored vecs
               chunk_array of size (n_post, n_pts)
        
        # note : to get a 2-dimensionnal vec do:
                 if bool_mask is not None:
                 we need to inverse
                     chunk_mask = np.ravel(subset[chunk_size])
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
    
    @staticmethod
    def index2d(index_1d, chunk_mask, chunk_slice):
        """ Return the 2d-indexing from 1d + mask 
        chunk_mask = None | self.subset[chunk_slice]
        """
        # chunk_size = fssettings.chunk_size
        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
        ix, iy = np.indices((nx, ny))
        ix = np.ravel(ix)
        iy = np.ravel(iy)
        if chunk_mask is not None:
            ix = ix[chunk_mask]
            iy = iy[chunk_mask]
        return ix[index_1d], iy[index_1d]

    @staticmethod
    def codes_mapping(complex_codes, int_codes, termination_codes):
        """
        Utility function, returns the inverse mapping code -> int
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

    def postproc(self, postproc_batch, chunk_slice):
        """ Computes the output of postproc_batch for chunk_slice
        Return
          post_array of shape(nposts, chunk_n_pts)
          chunk_mask
        """
        if postproc_batch.fractal is not self:
            raise ValueError("Postproc batch from a different factal provided")

        # Input data
        calc_name = postproc_batch.calc_name
        chunk_mask, Z, U, stop_reason, stop_iter = self.reload_data(
                chunk_slice, calc_name)
        params, codes = self.reload_params(calc_name)
        complex_dic, int_dic, termination_dic = self.codes_mapping(*codes)
        postproc_batch.set_chunk_data(chunk_slice, chunk_mask, Z, U,
            stop_reason, stop_iter, complex_dic, int_dic, termination_dic)

        # Output data
        n_pts = Z.shape[1]  # Z of shape [n_Z, n_pts]
        post_array = np.empty((len(postproc_batch.posts), n_pts),
                               dtype=self.float_postproc_type)
        

        for i, postproc in enumerate(postproc_batch.posts.values()):

            val, context_update = postproc[chunk_slice]
            # Debug
#            if np.iscomplexobj(val):
#                raise ValueError(val, "i", i, postproc.key)
            post_array[i, :]  = val
            postproc_batch.update_context(chunk_slice, context_update)

        postproc_batch.clear_chunk_data()

        return post_array, chunk_mask


# Numba JIT functions =========================================================
@numba.njit
def numba_cycles_perturb(Z, U, c, stop_reason, stop_iter, bool_active, iref,
                n_iter, SA_iter, ref_div_iter, ref_path, iterate,
                last_iref):
    """ Run the perturbation cycles
    """
    npts = c.size
    n_iter_init = n_iter
    for ipt in range(npts):
        # skip this ipt if pixel not active
        if not(bool_active[ipt]):
            continue
        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        cpt = c[ipt]
        stop_pt = stop_reason[:, ipt]
        n_iter = n_iter_init
        cycling = True
        while cycling:
            n_iter += 1 
            iterate(Zpt, Upt, cpt, stop_pt, n_iter, SA_iter,
                    ref_div_iter, ref_path[n_iter - 1 , :],
                    ref_path[n_iter, :], last_iref)
            cycling = (stop_pt[0] == -1)
            if not(cycling):
                stop_iter[0, ipt] = n_iter
                stop_reason[0, ipt] = stop_pt[0]

@numba.njit
def numba_cycles(Z, U, c, stop_reason, stop_iter, bool_active,
                 n_iter, iterate):
    """ Run the standard cycles
    """
    npts = c.size
    n_iter_init = n_iter
    for ipt in range(npts):
        # skip this ipt if pixel not active
        if not(bool_active[ipt]):
            continue
        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        cpt = c[ipt]
        stop_pt = stop_reason[:, ipt]
        n_iter = n_iter_init
        cycling = True
        while cycling:
            n_iter += 1 
            iterate(Zpt, Upt, cpt, stop_pt, n_iter)
            cycling = (stop_pt[0] == -1)
            if not(cycling):
                stop_iter[0, ipt] = n_iter
                stop_reason[0, ipt] = stop_pt[0]
