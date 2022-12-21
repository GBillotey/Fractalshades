# -*- coding: utf-8 -*-
import os
import fnmatch
import copy
import datetime
import pickle
import logging
import textwrap
import time
import types
import typing
import enum
import inspect

import numpy as np
from numpy.lib.format import open_memmap
import PIL
import PIL.PngImagePlugin
import numba

import fractalshades as fs
import fractalshades.settings
import fractalshades.utils
import fractalshades.postproc

from fractalshades.mthreading import Multithreading_iterator
import fractalshades.numpy_utils.expr_parser as fs_parser


logger = logging.getLogger(__name__)

class _Pillow_figure:
    def __init__(self, img, pnginfo):
        """
        This class is a wrapper that can be used to redirect a Fractal_plotter
        output, for instance when generating the documentation.
        """
        self.img = img
        self.pnginfo = pnginfo

    def save_png(self, im_path):
        """
        Saves as a png with Lanczos resizing filter if exceeds the max width
        """
        im = self.img
        width, height = im.size
        max_width = fs.settings.output_context["doc_max_width"]

        if width > max_width:
            ratio = float(width) / float(max_width)
            new_height = int(height / ratio)
            im = im.resize((max_width, new_height), PIL.Image.LANCZOS)
        im.save(im_path, format="png", pnginfo=self.pnginfo)


SUPERSAMPLING_DIC = {
    "2x2": 2,
    "3x3": 3,
    "4x4": 4,
    "5x5": 5,
    "6x6": 6,
    "7x7": 7,
    "None": None
}

SUPERSAMPLING_ENUM = enum.Enum(
    "SUPERSAMPLING_ENUM",
    list(SUPERSAMPLING_DIC.keys()),# ,
    module=__name__
)

supersampling_type = typing.Literal[SUPERSAMPLING_ENUM]


class Fractal_plotter:

    def  __init__(self,
        postproc_batch,
        final_render: bool = False,
        supersampling: supersampling_type = "None",
        jitter: bool = False,
        recovery_mode: bool = False,
        _delay_registering: bool = False # Private
    ):
        """
        The base plotting class.
        
        A Fractal plotter is a container for 
        `fractalshades.postproc.Postproc_batch` and fractal layers.
         
        Parameters
        ----------
        postproc_batch
            A single `fractalshades.postproc.Postproc_batch` or a list of 
            theses
        final_render: bool
            - If ``False``, this is an exploration rendering, the raw arrays
              will be stored to allow fast modifcation of the plotting
              parameters - without recomputing. High-quality rendering
              (supersampling, jitter) is disabled.
            - If ``True``, this is the final rendering, the image tiles will
              be directly computed by chunks on the fly to limit disk usage.
              High quality rendering options are available only in this case
              (antialising, jitter). Raw arrays are *not* stored in this case,
              any change in the plotting parametrers will need a new
              calculation.
        supersampling: None | "2x2" | "3x3" | ... | "7x7"
            Used only for the final render. if not None, the final image will
            leverage supersampling (from 4 to 49 pixels computed for 1 pixel in 
            the saved image)
        jitter: bool | float
            Used only for the final render. If not None, the final image will
            leverage jitter, default intensity is 1. This can help reduce moiré
            effect
        recovery_mode: bool
            Used only for the final render. If True, will attempt to reload
            the image tiles already computed. Allows to restart an interrupted
            calculation in final render mode (this will fail if plotting
            parameters have been modified).

        Notes
        -----

        .. warning::
            When passed a list of `fractalshades.postproc.Postproc_batch`
            objects, each postprocessing batch shall point to the same
            unique `Fractal` object.
        """
        if supersampling is None: # Nonetype not allowed in Enum...
            supersampling = "None"

        self.postproc_options = {
            "final_render": final_render,
            "supersampling": supersampling,
            "jitter": jitter,
            "recovery_mode": recovery_mode
        }

        # postproc_batches can be single or an enumeration
        # At this stage it is unfrozen (more can be added),
        # waiting for a register_postprocs call
        postproc_batches = postproc_batch
        if isinstance(postproc_batch, fs.postproc.Postproc_batch):
            postproc_batches = [postproc_batch]
        self._postproc_batches = postproc_batches # unfrozen 

        if not(_delay_registering):
            # freeze
            self.register_postprocs()


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
        :meta private:
        """
        if postproc_batch.fractal != self.fractal:
            raise ValueError("Attempt to add a postproc_batch from a different"
                             "fractal: {} from {}".format(
                postproc_batch.fractal, postproc_batch.calc_name))
        self.postproc_batches += [postproc_batch]
        self.posts.update(postproc_batch.posts)
        self.postnames_2d += postproc_batch.postnames_2d
    
    def register_postprocs(self):
        """
        Freezing of the postprocs - call might be delayed after a plotter
        instanciation but shall be before any coloring method.
        """
        postproc_batches = self._postproc_batches

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

        # Mem mapping dtype
        self.post_dtype = np.dtype(fs.settings.postproc_dtype)


    def add_layer(self, layer):
        """
        Adds a layer field to allow subsequent graphical operations or outputs.

        Parameters
        ----------
        layer : `fractalshades.colors.layers.Virtual_layer` or a derived class
            The layer to add

        Notes
        -----
        .. warning::
            Layer `postname` shall have been already registered in one of the
            plotter batches (see method
            `fractalshades.postproc.Postproc_batch.add_postproc`).

        .. warning::
            When a layer is added, a link layer -> Fractal_plotter is 
            created ; a layer can therefore only be added to a single
            `Fractal_plotter`.
        """
        postname = layer.postname
        if postname not in (list(self.postnames) + self.postnames_2d):
            raise ValueError(
                "Layer `{}` shall be registered in Fractal_plotter "
                "postproc_batches: {}".format(
                    postname, list(self.postnames) + self.postnames_2d
                )
            )
        self.layers += [layer]
        layer.link_plotter(self)

    def __getitem__(self, layer_name):
        """ Get the layer by its postname
        """
        for layer in self.layers:
            if layer.postname == layer_name:
                return layer
        raise KeyError("Layer {} not in available layers: {}".format(
                layer_name, list(l.postname for l in self.layers)))

    def plotter_info_str(self):
        str_info = "Plotting images: plotter options"
        for k, v in self.postproc_options.items():
            str_info += f"\n    {k}: {v}"
        str_info += ("\n  /!\\ supersampling and jitter only activated "
                     + "for final render")
        return str_info
    
    def zoom_info_str(self):
        str_info = "Plotting images: zoom kwargs"
        for k, v in self.fractal.zoom_kwargs.items():
            str_info += f"\n    {k}: {v}"
        return str_info
    
    def plot(self):
        """
        The base method to produce images.
        
        When called, it will got through all the instance-registered layers
        and plot each layer for which the `output` attribute is set to `True`.
        """
        logger.info(self.plotter_info_str())
        logger.info(self.zoom_info_str())

        # Open the image memory mappings ; open PIL images
        self.open_images()
        if self.final_render:
            for i, layer in enumerate(self.layers):
                if layer.output:
                    self.create_img_mmap(layer)
            self.fractal.delete_fingerprint()

        self._raw_arr = dict()
        self._current_tile = {
                "value": 0,
                "time": 0.  # time of last evt in seconds
        }

        # Here we follow a 2-step approach:
        # if "dev" (= not final) render, we first compute the 'raw' arrays.
        # This is done by pbatches, due to the univoque relationship
        # one postproc_batches <-> Fractal calculation
        # if "final" we do not store raw datas, but we still need to create a
        # mmap for the "subset" arrays
        f = self.fractal
        for pbatch in self.postproc_batches:
            calc_name = pbatch.calc_name
            f.clean_postproc_attr()
            if self.postproc_options["final_render"]:
                if (hasattr(f, "_subset_hook")
                        and calc_name in f._subset_hook.keys()):
                    logger.info(f"Adding subset output for {calc_name}")
                    f.init_subset_mmap(calc_name, self.supersampling)
            else:
                logger.info(f"Computing and storing raw data for {calc_name}")
                f.calc_raw(calc_name)

        # Now 2nd step approach:
        # if "dev" (= not final) render, we already know the 'raw' arrays, so
        # just a postprocessing step
        # if "final" render, we compute on the fly
        self.plot_tiles(chunk_slice=None)

        self.save_images()
        logger.info("Plotting images: done")

        # Output data file
        self.write_postproc_report()

        if fs.settings.inspect_calc:
            # Detailed debugging "inspect_calc" report
            for pbatch in self.postproc_batches:
                calc_name = pbatch.calc_name
                f.write_inspect_calc(
                    calc_name, final=self.final_render
                )

        # Clean-up
        for pbatch in self.postproc_batches:
            if self.postproc_options["final_render"]:
                if (hasattr(f, "_subset_hook")
                        and calc_name in f._subset_hook.keys()):
                    f.close_subset_mmap(calc_name, self.supersampling)


    @property
    def try_recover(self):
        """ Will we try to reopen saved image chunks ?"""
        return (
            self.postproc_options["recovery_mode"]
            and self.postproc_options["final_render"]
        )

    @property
    def final_render(self):
        """ Just an alias"""
        return self.postproc_options["final_render"]
    
    @property
    def supersampling(self):
        """ Will really implement antialiasing ? """
        if self.postproc_options["final_render"]:
            return SUPERSAMPLING_DIC[self.postproc_options["supersampling"]]
        # Note: implicitely None by default...

    @Multithreading_iterator(
        iterable_attr="chunk_slices", iter_kwargs="chunk_slice"
    )
    def plot_tiles(self, chunk_slice):
        """
        
        """
        # early exit if already computed
        if self.try_recover:
            f = self.fractal
            rank = f.chunk_rank(chunk_slice)
            _mmap_status = open_memmap(
                filename=self.mmap_status_path, mode="r+"
            )
            is_valid = (_mmap_status[rank] > 0)
            del _mmap_status
            if is_valid:
                for i, layer in enumerate(self.layers):
                    # We need the postproc
                    # TODO what about 'update scaling' ???
                    # It is invalid as we lost the data...
                    self.push_reloaded(
                        chunk_slice, layer=layer, im=self._im[i], ilayer=i
                    )
                f.incr_tiles_status(which="Plot tiles")
                self.incr_tiles_status(chunk_slice)
                return

        # 1) Compute the postprocs for this field
        n_pp = len(self.posts)
        (ix, ixx, iy, iyy) = chunk_slice
        npts = (ixx - ix) * (iyy - iy)

        if self.supersampling is not None:
            npts = npts * self.supersampling ** 2

        arr_shp = (n_pp, npts)
        raw_arr = np.empty(shape=arr_shp, dtype=self.post_dtype)

        batch_first_post = 0 # index of the first post for the current batch
        is_ok = 0
        for batch in self.postproc_batches:
            # Do the real job
            is_ok += self.fill_raw_arr(
                raw_arr, chunk_slice, batch, batch_first_post
            )
            # Just count index
            batch_first_post += len(batch.posts)
        if is_ok != 0:
            return

        self._raw_arr[chunk_slice] = raw_arr

        # 2) Push each layer crop to the relevant image 
        for i, layer in enumerate(self.layers):
            layer.update_scaling(chunk_slice)
            if layer.output:
                self.push_cropped(
                    chunk_slice, layer=layer, im=self._im[i], ilayer=i
                )

        # clean-up
        self.incr_tiles_status(chunk_slice) # mmm not really but...
        del self._raw_arr[chunk_slice]


    def fill_raw_arr(self, raw_arr, chunk_slice, batch, batch_first_post):
        """ Compute & store temporary arrays for this postproc batch
            Note : inc_postproc_rank rank shift to take into account potential
            other batches for this plotter.
            (memory mapping version)
        """
        f = self.fractal
        inc = batch_first_post
        ret = f.postproc(batch, chunk_slice, self.postproc_options)
        if ret is None:
            # Interrupted calculation
            return 1

        post_array, subset = ret
        arr_1d = f.reshape1d(
            post_array, subset, chunk_slice,self.supersampling
        )
        n_posts, _ = arr_1d.shape
        raw_arr[inc: (inc + n_posts), :] = arr_1d
        return 0 # OK


    def incr_tiles_status(self, chunk_slice):
        f = self.fractal
        curr_val = self._current_tile["value"] + 1
        self._current_tile["value"] = curr_val

        prev_log = self._current_tile["time"]
        curr_time = time.time()
        time_diff = curr_time - prev_log
        
        ntiles = f.chunks_count
        bool_log = ((time_diff > 1) or (curr_val == 1) or (curr_val == ntiles))
        
        # Marking this chunk as valid (in case of 'final render')
        if self.final_render:
            rank = f.chunk_rank(chunk_slice)
            _mmap_status = open_memmap(
                filename=self.mmap_status_path, mode="r+"
            )
            _mmap_status[rank] = 1
            del _mmap_status

        if bool_log:
            self._current_tile["time"] = curr_time
            str_val = str(curr_val)+ " / " + str(f.chunks_count)
            logger.info(f"Image output: {str_val}")


    def get_2d_arr(self, post_index, chunk_slice):
        """
        Returns a 2d view of a chunk for the given post-processed field
        """
        try:
            arr = self._raw_arr[chunk_slice][post_index, :]
        except KeyError:
            return None

        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
        
        ssg = self.supersampling
        if ssg is not None:
            nx *= ssg
            ny *= ssg

        return np.reshape(arr, (nx, ny))


    def write_postproc_report(self):
        txt_report_path = self.fractal.txt_report_path

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

        with open(txt_report_path, 'w', encoding='utf-8') as report:
            for i, layer in enumerate(self.layers):
                write_layer_report(i, layer, report)
        
        logger.info(textwrap.dedent(f"""\
            Plotting image - postprocessing fields info saved to:
              {txt_report_path}"""
        ))


    def image_name(self, layer):
        return "{}_{}".format(type(layer).__name__, layer.postname)

    def open_images(self):
        """ Open 
         - the image files
         - the associated memory mappings in case of "final render"
        ("""
        self._im = []
        if self.final_render:
            self.open_mmap_status()
        
        for layer in self.layers:
            if layer.output:
                self._im += [PIL.Image.new(mode=layer.mode, size=self.size)]

            else:
                self._im += [None]

        if self.try_recover:
            _mmap_status = open_memmap(
                filename=self.mmap_status_path, mode="r+"
            )
            valid_chunks = np.count_nonzero(_mmap_status)
            del _mmap_status
            n = self.fractal.chunks_count
            logger.info(
                "Attempt to restart interrupted calculation,\n"
                f"    Valid image tiles found: {valid_chunks} / {n}"
            )
        elif self.final_render:
            logger.info("Reloading option disabled, all image recomputed")


    def open_mmap_status(self):
        """ Small array to flag the validated image tiles
        Only for final render """
        n_chunk = self.fractal.chunks_count
        file_path = self.mmap_status_path

        try:
            # Does layer the mmap already exists, and does it seems to suit
            # our need ?
            if not(self.try_recover):
                raise ValueError("Invalidated mmap_status")
            _mmap_status = open_memmap(
                filename=file_path, mode="r+"
            )
            if (_mmap_status.shape != (n_chunk,)):
                raise ValueError("Incompatible shapes for plotter mmap_status")

        except (FileNotFoundError, ValueError):
            # Lets create it from scratch
            logger.debug(f"No valid plotter status file found - recompute img")
            _mmap_status = open_memmap(
                    filename=file_path, 
                    mode='w+',
                    dtype=np.int32,
                    shape=(n_chunk,),
                    fortran_order=False,
                    version=None
            )
            _mmap_status[:] = 0
            del _mmap_status


    @property
    def mmap_status_path(self):
        return os.path.join(self.plot_dir,"data", "final_render" + ".arr")


    def img_mmap(self, layer):
        """ The file path for the img memory mapping"""
        file_name = self.image_name(layer)
        file_path = os.path.join(self.plot_dir, "data", file_name + "._img")
        return file_path

    def create_img_mmap(self, layer):
        """ Just open - or reopen - the image memory mapping
        """
        mode = layer.mode
        dtype = fs.colors.layers.Virtual_layer.DTYPE_FROM_MODE[mode]
        channel = fs.colors.layers.Virtual_layer.N_CHANNEL_FROM_MODE[mode]
        nx, ny = self.size
        file_name = self.image_name(layer)
        file_path = self.img_mmap(layer)

        # Does the mmap already exists, and does it seems to suit our need ?
        try:
            # Does layer the mmap already exists, and does it seems to suit
            # our need ?
            if not(self.postproc_options["recovery_mode"]):
                raise ValueError("Invalidated mmap_status")
            mmap = open_memmap(
                filename=file_path, mode="r+"
            )
            if mmap.shape != (ny, nx, channel):
                del mmap
                raise ValueError("Incompatible shapes for mmap")
            if mmap.dtype != dtype:
                del mmap
                raise ValueError("Incompatible dtype for mmap")

        except (FileNotFoundError, ValueError):
            # Create a new one...
            mmap = open_memmap(
                filename=file_path, 
                mode='w+',
                dtype=np.dtype(dtype),
                shape=(ny, nx, channel),
                fortran_order=False,
                version=None
            )                
            # Here as we didnt find information for this layer, sadly the whole
            # memory mapping is invalidated
            logger.debug(f"No valid data found for layer {file_name}")
            
            _mmap_status = open_memmap(
                filename=self.mmap_status_path, mode="r+"
            )
            _mmap_status[:] = 0
            del _mmap_status

        del mmap
        

    def open_img_mmap(self, layer):
        """ mmap filled in // of the actual image - to allow restart
        Only for final render 
        """
        file_path = self.img_mmap(layer)
        mmap = open_memmap(filename=file_path, mode='r+')
        return mmap


    def save_images(self):
        """ Writes the images to disk """
        for i, layer in enumerate(self.layers):
            if not(layer.output):
                continue
            file_name = self.image_name(layer)
            base_img_path = os.path.join(self.plot_dir, file_name + ".png")
            self.save_tagged(self._im[i], base_img_path, self.fractal.tag)

    def save_tagged(self, img, img_path, tag_dict):
        """
        Saves *img* to png format at *path*, tagging with *tag_dict*.
        https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_PNG_files
        """
        pnginfo = PIL.PngImagePlugin.PngInfo()
        for k, v in tag_dict.items():
            pnginfo.add_text(k, str(v))
        if (fs.settings.output_context["doc"]
            and not(fs.settings.output_context["gui_iter"] > 0)):
            fs.settings.add_figure(_Pillow_figure(img, pnginfo))
        else:
            img.save(img_path, pnginfo=pnginfo)
            
            logger.info(textwrap.dedent(f"""\
                Image of shape {self.size} saved to:
                  {img_path}"""
            ))


    def push_cropped(self, chunk_slice, layer, im, ilayer):
        """ push "cropped image" from layer for this chunk to the image"""
        (ix, ixx, iy, iyy) = chunk_slice
        ny = self.fractal.ny
        crop_slice = (ix, ny-iyy, ixx, ny-iy)
        # Key: Calling get_2d_arr
        paste_crop = layer.crop(chunk_slice)
        if paste_crop is None:
            return
        
        if self.supersampling:
            # Here, we should apply a resizig filter
            # Image.resize(size, resample=None, box=None, reducing_gap=None)
            resample = PIL.Image.LANCZOS
            paste_crop = paste_crop.resize(
                size=(ixx - ix, iyy-iy),
                resample=resample,
                box=None,
                reducing_gap=None
            )

        im.paste(paste_crop, box=crop_slice)
        
        if self.final_render:
            # NOW let's also try to save this beast
            paste_crop_arr = np.asarray(paste_crop)
            layer_mmap = self.open_img_mmap(layer)

            if layer_mmap.shape[2] == 1:
                # For a 1-channel image, PIL will remove the last dim...
                layer_mmap[iy: iyy, ix: ixx, 0] = paste_crop_arr
            else:
                layer_mmap[iy: iyy, ix: ixx, :] = paste_crop_arr

            del layer_mmap
                


    def push_reloaded(self, chunk_slice, layer, im, ilayer):
        """ Just grap the already computed pixels and paste them"""
        if im is None:
            return
        (ix, ixx, iy, iyy) = chunk_slice
        ny = self.fractal.ny
        crop_slice = (ix, ny-iyy, ixx, ny-iy)
        layer_mmap = self.open_img_mmap(layer)
        
        paste_crop = PIL.Image.fromarray(layer_mmap[iy: iyy, ix: ixx, :])
        im.paste(paste_crop, box=crop_slice)

        del layer_mmap


class _Null_status_wget:
    def __init__(self, fractal):
        """ Internal class
        Emulates a status bar wget in case we do not use the GUI
        """
        status = {}
        status.update(fractal.new_status(self))
        self._status = status

    def update_status(self, key, str_val):
        """ Update the text status
        """
        self._status[key]["str_val"] = str_val


PROJECTION_ENUM = enum.Enum(
    "PROJECTION_ENUM",
    ("cartesian", "spherical", "expmap"),
    module=__name__
)
projection_type = typing.Literal[PROJECTION_ENUM]


class Fractal:

    REPORT_ITEMS = [
        "chunk1d_begin",
        "chunk1d_end",
        "chunk_pts",
        "done",
    ]

    SAVE_ARRS = [
        "Z",
        "U",
        "stop_reason",
        "stop_iter"
    ]

    USER_INTERRUPTED = 1 # return code for a user-interrupted computation

    def __init__(self, directory: str):
        """
The base class for all escape-time fractals calculations.

Derived class should implement the actual calculation methods used in the
innner loop. This class provides the outer looping (calculation is run on 
successive tiles), enables multiprocessing, and manage raw-result storing 
and retrieving.

Parameters
----------
directory : str
    Path for the working base directory


Notes
-----

These notes describe implementation details and should be useful mostly to
advanced users when subclassing.

.. note::

    **Special methods**
    
    this class and its subclasses may define several methods decorated with
    specific tags:
    
    `fractalshades.zoom_options`
        decorates the method used to define the zooming
    `fractalshades.calc_options`
        decorates the methods defining the calculation inner-loop
    `fractalshades.interactive_options` 
        decorates the methods that can be called
        interactively from the GUI (right-click then context menu selection).
        The coordinates of the click are passed to the called method.

.. note::
    
    **Calculation parameters**

    To launch a calculation, call `~fractalshades.Fractal.run`. The parameters
    from the last 
    @ `fractalshades.zoom_options` call and last 
    @ `fractalshades.calc_options` call will be used. 
    They are stored as class attributes, above a list of such attributes and
    their  meaning (non-exhaustive as derived class cmay define their own).
    Note that they should normally not be directly accessed but defined in
    derived class through zoom and calc methods.

.. note::

    **Saved data**
    
        The calculation raw results (raw output of the inner loop at exit) are
        saved to disk and internally accessed during plotting phase through
        memory-mapping. They are however not saved for a final render.
        These arrays are:

        subset    
            boolean - alias for `subset`
            Saved to disk as ``calc_name``\_Z.arr in ``data`` folder
        Z
            Complex fields, several fields can be defined and accessed through
            a field string identifier.
            Saved to disk as ``calc_name``\_Z.arr in ``data`` folder
        U
            Integer fields, several fields can be defined and accessed through
            a field string identifier.
            Saved to disk as ``calc_name``\_U.arr in ``data`` folder
        stop_reason
            Byte codes: the reasons for loop exit (max iteration reached ?
            overflow ? other ?) A string identifier
            Saved to disk as ``calc_name``\_stop_reason.arr in ``data`` folder
        stop_iter
            Integer: iterations count at loop exit
            Saved to disk as ``calc_name``\_stop_iter.arr in ``data`` folder

        The string identifiers are stored in ``codes`` attributes.
"""
        self.directory = directory
        self.subset = None
        self._interrupted = np.array([0], dtype=np.bool_)
        
        # datatypes used for raw data storing
        self.float_postproc_type = np.dtype(fs.settings.postproc_dtype)
        self.termination_type = np.int8
        self.int_type = np.int32

        # Default listener
        self._status_wget = _Null_status_wget(self)

    @property
    def init_kwargs(self):
        """ Return a dict of parameters used during __init__ call"""
        init_kwargs = {}
        for (
            p_name, param
        ) in inspect.signature(self.__init__).parameters.items():
            # By contract, __init__ params shall be stored as instance attrib.
            # ('Fractal' interface)
            init_kwargs[p_name] = getattr(self, p_name)
        return init_kwargs

    def init_signature(self):
        return inspect.signature(self.__init__)

    def __reduce__(self):
        """ Serialisation of a Fractal object. The fractal state (zoom, calc)
        is dropped and shall be re-instated externally if need be.
        """
        vals = tuple(self.init_kwargs.values())
        return (self.__class__, vals)

    def script_repr(self, indent):
        """String used to serialize this instance in GUI-generated scripts
        """
        # Mainly a call to __init__ with the directory tuned to variable
        # `plot_dir`
        fullname = fs.utils.Code_writer.fullname(self.__class__)
        kwargs = self.init_kwargs
        kwargs["directory"] = fs.utils.Rawcode("plot_dir") # get rid of quotes
        kwargs_code = fs.utils.Code_writer.func_args(kwargs, 1)
        str_call_init = f"{fullname}(\n{kwargs_code})"
        return str_call_init


    @fs.utils.zoom_options
    def zoom(self, *,
             x: float = 0.,
             y: float = 0.,
             dx: float = 8.,
             nx: int = 800,
             xy_ratio: float = 1.,
             theta_deg: float = 0.,
             projection: projection_type="cartesian",
             has_skew: bool = False,
             skew_00:float = 1.0,
             skew_01: float = 0.0,
             skew_10: float = 0.0,
             skew_11: float = 1.0
    ):
        """
        Define and stores as class-attributes the zoom parameters for the next
        calculation.
        
        Parameters
        ----------
        x : float
            x-coordinate of the central point
        y : float 
            y-coordinate of the central point
        dx : float
            span of the view rectangle along the x-axis
        nx : int
            number of pixels of the image along the x-axis
        xy_ratio: float
            ratio of dx / dy and nx / ny
        theta_deg : float
            Pre-rotation of the calculation domain, in degree
        projection : "cartesian" | "spherical" | "exp_map"
            Kind of projection used (default to cartesian)
        has_skew : bool
            If True, unskew the view base on skew coefficients skew_ij
        skew_ij : float
            Components of the local skw matrix, ij = 00, 01, 10, 11
        """
        # Safeguard in case the GUI inputs were strings
        if isinstance(x, str) or isinstance(y, str) or isinstance(dx, str):
            raise RuntimeError("Float expected for x, y, dx")

        # Stores the skew matrix
        self._skew = None
        if has_skew:
            self._skew = np.array(
                ((skew_00, skew_01), (skew_10, skew_11)), dtype=np.float64
            )

    def new_status(self, wget):
        """ Return a dictionnary that can hold the current progress status """
        self._status_wget = wget
        status = {
            "Calc tiles": {
                "val": 0,
                "str_val": "- / -",
                "last_log": 0.  # time of last logged change in seconds
            },
            "Plot tiles": {
                "val": 0,
                "str_val": "- / -",
                "last_log": 0.  # time of last logged change in seconds
            }
        }
        return status

    def set_status(self, key, str_val, bool_log=True):
        """ Just a simple text status """
        self._status_wget.update_status(key, str_val)
        if bool_log:
            logger.debug(f"Status update {key}: {str_val}")

    def incr_tiles_status(self, which="Calc tiles"):
        """ Dealing with more complex status : 
        Increase by 1 the number of computed tiles reported in status bar
        """
        dic = self._status_wget._status
        curr_val = dic[which]["val"] + 1
        dic[which]["val"] = curr_val

        prev_event = dic[which]["last_log"]
        curr_time = time.time()
        time_diff = curr_time - prev_event

        ntiles = self.chunks_count
        bool_log = ((time_diff > 1) or (curr_val == 1) or (curr_val == ntiles))
        if bool_log:
            dic[which]["last_log"] = curr_time

        str_val = str(dic[which]["val"]) + " / " + str(ntiles)
        self.set_status(which, str_val, bool_log)


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
    def skew(self):
        if not(hasattr(self, "_skew")): # is None:
            self._skew = None
        return self._skew

    @property
    def multiprocess_dir(self):
        """ Directory used for multiprocess stdout stderr streams redirection
        :meta private:
        """
        return os.path.join(self.directory, "multiproc_calc")


    @property
    def float_type(self):
        select = {
            np.dtype(np.float64): np.float64,
            np.dtype(np.complex128): np.float64
        }
        return select[np.dtype(self.complex_type)]


    def clean_up(self, calc_name=None):
        """
        Deletes all saved data files associated with a given ``calc_name``.

        Parameters
        ----------
        calc_name : str | None
            The string identifying the calculation run for which we want to
            delete the files.
            If None, delete all calculation files
        """
        if calc_name is None:
            calc_name = "*"

        patterns = (
             calc_name + "_*.arr",
             calc_name + ".report",
             calc_name + ".fingerprint",
             "*._img", # ._img files are tagged by the output layer name
             "ref_pt.dat",
             "SA.dat"
        )

        data_dir = os.path.join(self.directory, "data")
        if not os.path.isdir(data_dir):
            logger.warning(textwrap.dedent(f"""\
                Clean-up cancelled, directory not found:
                  {data_dir}"""
            ))
            return
        logger.info(textwrap.dedent(f"""\
            Cleaning data directory:
              {data_dir}"""
        ))

        for pattern in patterns:
            with os.scandir(data_dir) as it:
                for entry in it:
                    if (fnmatch.fnmatch(entry.name, pattern)):
                        os.unlink(entry.path)
                        logger.debug(f"File deleted: {entry.name}")

        # Delete also the temporay attributes
        temp_attrs = ("_FP_params", "_Z_path", "_SA_data")
        for temp_attr in temp_attrs:
            if hasattr(self, temp_attr):
                delattr(self, temp_attr)
    
    def delete_fingerprint(self):
        """ Deletes the figerprint files """
        patterns = (
             "*.fingerprint",
        )
        data_dir = os.path.join(self.directory, "data")
        if not os.path.isdir(data_dir):
            return
        for pattern in patterns:
            with os.scandir(data_dir) as it:
                for entry in it:
                    if (fnmatch.fnmatch(entry.name, pattern)):
                        os.unlink(entry.path)
                        logger.debug(f"File deleted: {entry.name}")


    def clean_postproc_attr(self):
        # Deletes the postproc-related temporary attributes, thus forcing
        # a recompute of invalidated data
        postproc_attrs = ("_uncompressed_beg_end",)
        for temp_attr in postproc_attrs:
            if hasattr(self, temp_attr):
                delattr(self, temp_attr)


    # The various method associated with chunk mgmt ===========================
    def chunk_slices(self): #, chunk_size=None):
        """
        Generator function
        Yields the chunks spans (ix, ixx, iy, iyy)
        with each chunk of size chunk_size x chunk_size
        """
        # if chunk_size is None:
        chunk_size = fs.settings.chunk_size

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
        chunk_size = fs.settings.chunk_size
        (cx, r) = divmod(self.nx, chunk_size)
        if r != 0:
            cx += 1
        (cy, r) = divmod(self.ny, chunk_size)
        if r != 0:
            cy += 1
        return cx * cy

    def chunk_rank(self, chunk_slice):
        """
        Return the generator yield index for chunk_slice
        """
        chunk_size = fs.settings.chunk_size
        (ix, _, iy, _) = chunk_slice
        chunk_item_x = ix // chunk_size
        chunk_item_y = iy // chunk_size
        (cy, r) = divmod(self.ny, chunk_size)
        if r != 0:
            cy += 1
        return chunk_item_x * cy + chunk_item_y

    def chunk_from_rank(self, rank):
        """
        Return the chunk_slice from the generator yield index
        """
        chunk_size = fs.settings.chunk_size
        (cy, r) = divmod(self.ny, chunk_size)
        if r != 0: cy += 1
        chunk_item_x, chunk_item_y = divmod(rank, cy)
        ix = chunk_item_x * chunk_size
        iy = chunk_item_y * chunk_size
        ixx = min(ix + chunk_size, self.nx)
        iyy = min(iy + chunk_size, self.ny)
        return (ix, ixx, iy, iyy)

    def uncompressed_beg_end(self, rank):
        """ Return the span for the boolean mask index for the chunk index
        `rank` """
        # Note: indep of the mask.
        valid = False
        if hasattr(self, "_uncompressed_beg_end"):
            arr, in_nx, in_ny, in_csize = self._uncompressed_beg_end
            valid = (
                self.nx == in_nx,
                self.ny == in_ny,
                fs.settings.chunk_size == in_csize
            )
        if not valid:
            arr = self.recompute_uncompressed_beg_end(rank)
        return arr[rank], arr[rank + 1]

    def recompute_uncompressed_beg_end(self, rank):
        """ A bit brutal but does the job
        """
        arr = np.empty((self.chunks_count + 1,), dtype=np.int32)
        arr[0] = 0
        for i, chunk_slice in enumerate(self.chunk_slices()):
            (ix, ixx, iy, iyy) = chunk_slice
            arr[i + 1] = arr[i] + (ixx - ix) * (iyy - iy)
        self._uncompressed_beg_end = (
                arr, self.nx, self.ny, fs.settings.chunk_size
        )
        return arr

    def compressed_beg_end(self, calc_name, rank):
        """ Return the span for a stored array
        """
        # Note : depend of the mask... hence the calc_name
        mmap = self.get_report_memmap(calc_name)

        items = self.REPORT_ITEMS
        beg = mmap[rank, items.index("chunk1d_begin")]
        end = mmap[rank, items.index("chunk1d_end")]
        del mmap

        return beg, end


    def pts_count(self, calc_name, chunk_slice=None):
        """ Return the number of compressed 1d points for the current
        calculation, and the chunk_slice
        (if chunk_slice is None: return the summ for all chunks)
        """
        state = self._calc_data[calc_name]["state"]
        subset = state.subset

        if chunk_slice is not None:
            (ix, ixx, iy, iyy) = chunk_slice

        if subset is not None:
            if chunk_slice is None:
                return np.count_nonzero(self.subset[None])
            else:
                subset_pts = np.count_nonzero(subset[chunk_slice])
                return subset_pts
        else:
            if chunk_slice is None:
                return self.nx * self.ny
            else:
                return (ixx - ix) * (iyy - iy)


    def chunk_pixel_pos(self, chunk_slice, jitter, supersampling):
        """
        Return the image pixels vector distance to center in fraction of image
        width, as a complex
           pix = center + (pix_frac_x * dx,  pix_frac_x * dy)
        """
        data_type = self.float_type

        theta = self.theta_deg / 180. * np.pi
        (nx, ny) = (self.nx, self.ny)
        (ix, ixx, iy, iyy) = chunk_slice

        kx = 0.5 / (nx - 1) # interval width
        ky = 0.5 / (ny - 1) # interval width

        if supersampling is None:
            x_1d = np.linspace(
                kx * (2 * ix - nx + 1),
                kx * (2 * ixx - nx - 1),
                num=(ixx - ix),
                dtype=data_type
            )
            y_1d = np.linspace(
                ky * (2 * iy - ny + 1),
                ky * (2 * iyy - ny - 1),
                num=(iyy - iy),
                dtype=data_type
            )

        else:
            ssg = supersampling
            ssg_gap = (ssg - 1.) / ssg
            x_1d = np.linspace(
                kx * (2 * ix - nx + 1 - ssg_gap),
                kx * (2 * ixx - nx - 1 + ssg_gap),
                num = (ixx - ix) * ssg,
                dtype=data_type
            )
            y_1d = np.linspace(
                ky * (2 * iy - ny + 1 - ssg_gap),
                ky * (2 * iyy - ny - 1 + ssg_gap),
                num = (iyy - iy) * ssg,
                dtype=data_type
            )

        dy_vec, dx_vec  = np.meshgrid(y_1d, x_1d)

        if jitter:
            rg = np.random.default_rng(0)
            rand_x = rg.random(dx_vec.shape, dtype=data_type)
            rand_y = rg.random(dy_vec.shape, dtype=data_type)
            k = 0.7071067811865476
            jitter_x = (0.5 - rand_x) * k / (nx - 1) * jitter
            jitter_y = (0.5 - rand_y) * k / (ny - 1) * jitter
            if supersampling is not None:
                jitter_x /= supersampling
                jitter_y /= supersampling
            dx_vec += jitter_x
            dy_vec += jitter_y

        dy_vec /= self.xy_ratio

        # Apply the Linear part of the tranformation
        if theta != 0 and self.projection != "expmap":
            apply_rot_2d(theta, dx_vec, dy_vec)

        if self.skew is not None:
            apply_skew_2d(self.skew, dx_vec, dy_vec)

        res = dx_vec + 1j * dy_vec

        return res


    @property    
    def tag(self):
        """ Used to tag an output image
        """
        tag = {
            "Software": "fractalshades " + fs.__version__,
            "datetime": datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        }
        # Adding detailed data per calculation
        for calc_name in self._calc_data.keys():
            state = self._calc_data[calc_name]["state"]
            tag[calc_name] = state.fingerprint
        # Adding the reference zoom parameters for GUI navigation
        tag.update(self.zoom_kwargs)

        return fs.utils.dic_flatten(tag)


    def fingerprint_matching(self, calc_name, test_fingerprint, log=False):
        """
        Test if the stored parameters match those of new calculation
        /!\ modified in subclass
        """
        flatten_fp = fs.utils.dic_flatten(test_fingerprint)

        state = self._calc_data[calc_name]["state"]
        expected_fp = fs.utils.dic_flatten(state.fingerprint)
        
        if log:
            logger.debug(f"fingerprint_matching flatten_fp:\n {flatten_fp}")
            logger.debug(f"fingerprint_matching expected_fp:\n {expected_fp}")

        # We currently do not have special keys to handle
        # (Note: The software version and calculation info are now only added
        # to the tagged image)
        SPECIAL = []

        for key, val in expected_fp.items():
            if (key in SPECIAL):
                continue
            else:
                if flatten_fp[key] != val:
                    if log:
                        logger.debug(textwrap.dedent(f"""\
                            Parameter mismatch ; will trigger a recalculation
                              {key}, {flatten_fp[key]} --> {val}"""
                        ))
                    return False
        return True

    def res_available(self, calc_name, chunk_slice=None):
        """  
        If chunk_slice is None, check that stored calculation fingerprint
        matches.
        If chunk_slice is provided, checks that calculation results are
        available
        """
        try:
            fingerprint = self.reload_fingerprint(calc_name)
        except IOError:
            logger.debug(textwrap.dedent(f"""\
                No fingerprint file found for {calc_name};
                  will trigger a recalculation"""
                ))
            return False

        log = (chunk_slice is None)
        matching = self.fingerprint_matching(calc_name, fingerprint, log)
        if log:
            logger.debug(
                f"Fingerprint match for {chunk_slice}: {matching}"
            )
        if not(matching):
            return False
        if chunk_slice is None:
            return matching # should be True

        try:
            report = self.reload_report(calc_name, chunk_slice)
        except IOError:
            return False

        return report["done"] > 0


    def calc_hook(self, calc_callable, calc_kwargs, return_dic):
        """
        Called by a calculation wrapper (ref. utils.py)
        Prepares & stores the data needed for future calculation of tiles

        /!\ If error here check that the Fractal calculation implementation
        returns the expected dict of functionss:
        {
            "set_state": set_state,
            "initialize": initialize,
            "iterate": iterate
        }
        """
        calc_name = calc_kwargs["calc_name"]

        # Storage for future calculation
        if not hasattr(self, "_calc_data"):
            self._calc_data = dict()

        # Setting internal state - needed to unwrap other data
        # /!\ using in also a separate namespace, thread-safe option
        state = types.SimpleNamespace()
        for k, v in calc_kwargs.items():
            setattr(state, k, v)
            setattr(self, k, v)
        set_state = return_dic["set_state"]()
        set_state(state)
        set_state(self)

        initialize = return_dic["initialize"]()
        iterate = return_dic["iterate"]()
        cycle_indep_args = self.get_cycle_indep_args(initialize, iterate)
        saved_codes = self.saved_codes(state.codes)

        # stores the data for later use
        self._calc_data[calc_name] = {
            "calc_class": type(self).__name__,
            "calc_callable": calc_callable,
            "calc_kwargs": calc_kwargs,
            "zoom_kwargs": self.zoom_kwargs,
            "state": state,
            "cycle_indep_args": cycle_indep_args,
            "saved_codes": saved_codes,
            "init_kwargs": self.init_kwargs
        }

        # Takes a 'fingerprint' of the calculation parameters
        fp_items = (
            "calc_class", "calc_callable", "calc_kwargs", "zoom_kwargs",
            "init_kwargs"
        )
        state.fingerprint = {
            k: self._calc_data[calc_name][k] for k in fp_items
        }

        # Adding a subset hook for future 'on the fly' computation
        if "subset" in calc_kwargs.keys():
            subset = calc_kwargs["subset"]
            if subset is not None:
                self.add_subset_hook(subset, calc_name)

        # We cannot open new memaps at this early stage : because
        # subset may still be unknown. But, we shall track it.
        if self.res_available(calc_name):
            # There *should* be mmaps available but lets double-check this
            try:
                mmap = self.get_report_memmap(calc_name, mode="r+")
                del mmap
                for key in self.SAVE_ARRS:
                    mmap = self.get_data_memmap(calc_name, key, mode="r+")
                    del mmap
                if subset is not None:
                    mmap = self.get_data_memmap(calc_name, "subset", mode="r+")
                    del mmap
                self._calc_data[calc_name]["need_new_mmap"] = False
                logger.info(
                    f"Found suitable raw results files set for {calc_name}\n"
                    "  -> only the missing tiles will be recomputed"
                )
            except FileNotFoundError:
                self._calc_data[calc_name]["need_new_mmap"] = True
                logger.info(
                    f"Raw results files set for {calc_name} is incomplete\n"
                    "  -> all tiles will be recomputed"
                )
        else:
            logger.info(
                f"Missing up-to-date result file available for {calc_name}\n"
                    "  -> all tiles will be recomputed"
            )
            self._calc_data[calc_name]["need_new_mmap"] = True
            self.save_fingerprint(calc_name, state.fingerprint)

    def raise_interruption(self):
        self._interrupted[0] = True

    def lower_interruption(self):
        self._interrupted[0] = False

    def is_interrupted(self):
        """ Either programmatically 'interrupted' (from the GUI) or by the user 
        in batch mode through fs.settings.skip_calc """
        return (self._interrupted[0] or fs.settings.skip_calc)

    @staticmethod
    def numba_cycle_call(cycle_dep_args, cycle_indep_args):
        # Just a thin wrapper to allow customization in derived classes
        return numba_cycles(*cycle_dep_args, *cycle_indep_args)

    def get_cycle_indep_args(self, initialize, iterate):
        """
        When not a perturbation rendering:
        This is just a diggest of the zoom and calculation parameters
        """
        dx = self.dx
        center = self.x + 1j * self.y
        xy_ratio = self.xy_ratio
        theta = self.theta_deg / 180. * np.pi # used for expmap
        projection = getattr(PROJECTION_ENUM, self.projection).value
        # was : self.PROJECTION_ENUM[self.projection]

        return (
            initialize, iterate,
            dx, center, xy_ratio, theta, projection,
            self._interrupted
        )
    

    def get_cycling_dep_args(
            self, calc_name, chunk_slice,
            final=False, jitter=False, supersampling=None):
        """
        The actual input / output arrays
        """
        c_pix = np.ravel(
            self.chunk_pixel_pos(chunk_slice, jitter, supersampling)
        )

        state = self._calc_data[calc_name]["state"]
        subset = state.subset
        if subset is not None:
            if final:
                # Here we need a substitution - as raw arrays not stored
                subset = self._subset_hook[calc_name]
            chunk_subset = subset[chunk_slice]
            c_pix = c_pix[chunk_subset]
        else:
            chunk_subset = None

        # Initialise the result arrays
        (n_pts,) = c_pix.shape

        n_Z, n_U, n_stop = (len(code) for code in self.codes)

        Z = np.zeros([n_Z, n_pts], dtype=self.complex_type)
        U = np.zeros([n_U, n_pts], dtype=self.int_type)
        stop_reason = - np.ones([1, n_pts], dtype=self.termination_type)
        stop_iter = np.zeros([1, n_pts], dtype=self.int_type)

        return (c_pix, Z, U, stop_reason, stop_iter), chunk_subset

#==============================================================================

    def add_subset_hook(self, f_array, calc_name_to):
        """ Create a namespace for a hook that shall be filled during
        on-the fly computation - used for final calculations with subset on. 
           Hook structure:  dict 
           hook[calcname_to] = [f_array_dat, supersampling_k, mmap]
        """
        logger.debug(f"Add Subset hook for {calc_name_to}")

        if not(hasattr(self, "_subset_hook")):
            self._subset_hook = dict()

        self._subset_hook[calc_name_to] = _Subset_temporary_array(
            self, f_array.calc_name, calc_name_to,
            f_array.key, f_array._func, f_array.inv
        )

    def init_subset_mmap(self, calc_name_to, supersampling): # private
        """ Create the memory mapping
        Format: 1-d UNcompressed per chunk, takes into account supersampling
        as needed. 
        """
        logger.debug(f"Opening subset mmap for: {calc_name_to}")
        self._subset_hook[calc_name_to].init_mmap(supersampling)

    def close_subset_mmap(self, calc_name_to, supersampling): # private
        """ Closed the memory mapping
        Format: 1-d UNcompressed per chunk, takes into account supersampling
        as needed. 
        """
        self._subset_hook[calc_name_to].close_mmap(supersampling)

    def save_subset_arrays(
            self, calc_name, chunk_slice, postproc_options, ret
    ):
        """ Checks wether saving is needed"""

        for calc_name_to, sta in self._subset_hook.items():
            if sta.calc_name_from != calc_name:
                # Calc data not needed for this fractal array
                return
            logger.debug("Saving subset data chunk for on the fly plot:\n" +
                         f"{calc_name} -> {sta.calc_name_to}")
            sta.save_array(chunk_slice, ret)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fingerprint_path(self, calc_name):
        return os.path.join(
            self.directory, "data", calc_name + ".fingerprint"
        )

    def save_fingerprint(self, calc_name, fingerprint):
        save_path = self.fingerprint_path(calc_name)
        fs.utils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as fp_file:
            pickle.dump(fingerprint, fp_file, pickle.HIGHEST_PROTOCOL)

    def reload_fingerprint(self, calc_name):
        """ Reloading the fingerprint from the saved files """
        save_path = self.fingerprint_path(calc_name)
        with open(save_path, 'rb') as tmpfile:
            fingerprint = pickle.load(tmpfile)
        # Here we could finalize unpickling of Fractal arrays by binding
        # again the Fractal object - it is however not needed (we only use
        # the unpickled object for equality testing)
        return fingerprint

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Codes filtering
    def saved_codes(self, codes):
        (complex_codes, int_codes, stop_codes) = codes
        f_complex_codes = self.filter_stored_codes(complex_codes)
        f_int_codes = self.filter_stored_codes(int_codes)
        return (f_complex_codes, f_int_codes, stop_codes)

    @staticmethod
    def filter_stored_codes(codes):
        """ Don't store temporary codes - i.e. those which starts with "_" """
        return list(filter(lambda x: not(x.startswith("_")), codes))

#==============================================================================
    @property
    def txt_report_path(self):
        """The main report is written by the Fractal_plotter, only the name is 
        defined here"""
        return  os.path.join(
            self.directory, self.__class__.__name__ + ".txt"
        )

    # Report path ".inspect" tracks the progress of the calculations
    def report_path(self, calc_name): # public
        return os.path.join(
            self.directory, "data", calc_name + ".report"
        )

    def init_report_mmap(self, calc_name): # private
        """
        Create the memory mapping for calculation reports by chunks
        [chunk1d_begin, chunk1d_end, chunk_pts]
        """
        items = self.REPORT_ITEMS
        chunks_count = self.chunks_count
        report_cols_count = len(items)

        logger.debug(
            "Create new report_mmap with chunks_count: "
            f"{self.chunks_count}"
        )

        save_path = self.report_path(calc_name)
        fs.utils.mkdir_p(os.path.dirname(save_path))

        mmap = open_memmap(
            filename=save_path, 
            mode='w+',
            dtype=np.dtype(np.int32),
            shape=(chunks_count, report_cols_count),
            fortran_order=False,
            version=None
        )
        mmap[:, items.index("done")] = 0 # Will be set to 1 when done

        # Number of points per chunk
        chunk_pts = np.empty((chunks_count,), dtype=np.int32)
        for i, chunk_slice in enumerate(self.chunk_slices()):
            chunk_pts[i] = self.pts_count(calc_name, chunk_slice) # (calc_name, chunk_slice)
        mmap[:, items.index("chunk_pts")] = chunk_pts

        # full_cumsum is np.cumsum(chunk_pts) with inserted 0
        full_cumsum = np.empty((chunks_count + 1,), dtype=np.int32)
        np.cumsum(chunk_pts, out=full_cumsum[1:])
        full_cumsum[0] = 0
        mmap[:, items.index("chunk1d_begin")] = full_cumsum[:-1]
        mmap[:, items.index("chunk1d_end")] = full_cumsum[1:]

        del mmap


    def get_report_memmap(self, calc_name, mode='r+'):
        # Development Note - Memory mapping 
        # From IEEE 1003.1:
        #    The mmap() function shall establish a mapping between a process'
        #    address space and a file, shared memory object, or [TYM] typed
        #    memory object.
        # 1)
        # Every instance of np.memmap creates its own mmap of the whole file
        # (even if it only creates an array from part of the
        # file). The implications of this are A) you can't use np.memmap's
        # offset parameter to get around file size limitations, and B) you
        # shouldn't create many numpy.memmaps of the same file. To work around
        # B, you should create a single memmap, and dole out views and slices.
        # 2)
        # Yes, it allocates room for the whole file in your process's LOGICAL
        # address space. However, it doesn't actually reserve any PHYSICAL
        # memory, or read in any data from the disk, until you've actually
        # access the data. And then it only reads small chunks in, not the
        # whole file.
        val = open_memmap(
            filename=self.report_path(calc_name), mode=mode
        )
        return val


    def report_memmap_attr(self, calc_name):
        return "__report_mmap_" + "_" + calc_name

    def update_report_mmap(self, calc_name, chunk_slice, stop_reason):
        """
        """
        mmap = self.get_report_memmap(calc_name, mode="r+")
        items = self.REPORT_ITEMS
        chunk_rank = self.chunk_rank(chunk_slice)
        mmap[chunk_rank, items.index("done")] = 1
        del mmap

    def reload_report(self, calc_name, chunk_slice): # public
        """ Return a report extract for the given chunk, as a dict
             If no chunk provided, return the full report (header, report)
        """
        mmap = self.get_report_memmap(calc_name)
        items = self.REPORT_ITEMS
        if chunk_slice is None:
            report = np.empty(mmap.shape, mmap.dtype)
            report[:, :] = mmap[:, :]
            return  self.REPORT_ITEMS, report

        rank = self.chunk_rank(chunk_slice)
        report = dict(zip(
            items,
            (mmap[rank, items.index(it)] for it in items)
        ))
        del mmap
        return report


    def write_inspect_calc(self, calc_name, final):
        """
        Outputs a report for the current calculation
        """
        outpath = os.path.join(self.directory, calc_name + ".inspect")

        if final:
            log_txt = ("Cannot write detailed output *.inspect_calc"
                       " for a final render")
            if os.path.exists(outpath):
                os.remove(outpath)
                log_txt += f", deleting obsolete file {outpath}"
            logger.info(log_txt)
            return

        REPORT_ITEMS, report = self.reload_report(calc_name, None)
        report_header = ("chnk_beg|chnk_end|chnk_pts|done|")

        # There are other interesting items to inspect
        chunks_count = self.chunks_count

        stop_ITEMS = ["min_stop_iter", "max_stop_iter", "mean_stop_iter"]
        stop_report = np.zeros([chunks_count, 3], dtype = np.int32)
        stop_header = "min_stop|max_stop|mean_stp|"

        reason_ITEMS = []
        reason_reports = []
        reason_header = ""
        reason_template = np.zeros([chunks_count, 1], dtype = np.int32)

        for i, chunk_slice in enumerate(self.chunk_slices()):
            subset, _, Z, U, stop_reason, stop_iter = self.reload_data(
                chunk_slice, calc_name)
            # Outputs a summary of the stop iter
            has_item = (stop_iter.size != 0)
            
            for j, it in enumerate(stop_ITEMS):
                if it == "min_stop_iter" and has_item:
                    stop_report[i, j] = np.min(stop_iter)
                elif it == "max_stop_iter" and has_item:
                    stop_report[i, j] = np.max(stop_iter)
                elif it == "mean_stop_iter" and has_item:
                    stop_report[i, j] = int(np.mean(stop_iter))
                else:
                    stop_report[i, j] = -1

            # Outputs a summary of the stop reason
            if (stop_reason.size == 0): # Nothing to report
                continue
            max_chunk_reason = np.max(stop_reason)
            for r in range(len(reason_ITEMS), max_chunk_reason + 1):
                reason_ITEMS += ["reason_" + str(r)]
                reason_reports += [reason_template.copy()]
                reason_header += ("reason_" + str(r) + "|")
            bc = np.bincount(np.ravel(stop_reason))
            for r, bc_r in enumerate(bc):
                reason_reports[r][i, 0] = bc_r

        # Stack the results
        header = REPORT_ITEMS + stop_ITEMS + reason_ITEMS
        n_header = len(header)

        full_report = np.empty((chunks_count, n_header), dtype = np.int32)
        l1 = len(REPORT_ITEMS)
        l2 = l1 + len(stop_ITEMS)
        full_report[:, :l1] = report
        full_report[:, l1:l2] = stop_report
        for i in range(l2, n_header):
            r = i - l2
            full_report[:, i] = reason_reports[r][:, 0]

        np.savetxt(
            outpath,
            full_report,
            fmt=('%8i|%8i|%8i|%4i|%8i|%8i|%8i|' + '%8i|' * len(reason_ITEMS)),
            header=(report_header + stop_header + reason_header),
            comments=''
        )


    def data_path(self, calc_name):
        keys = ["subset"] + self.SAVE_ARRS 
        def file_map(key):
            return os.path.join(self.directory, "data",
                                calc_name + "_" + key + ".arr")
        return dict(zip(keys, map(file_map, keys)))


    def init_data_mmaps(self, calc_name):
        """
        Creates the memory mappings for calculated arrays
        [subset, Z, U, stop_reason, stop_iter]
        
        Note : subset can be initialized here
        """
        state = self._calc_data[calc_name]["state"]
        keys = self.SAVE_ARRS
        data_type = {
            "Z": self.complex_type,
            "U": self.int_type,
            "stop_reason": self.termination_type,
            "stop_iter": self.int_type,
        }
        data_path = self.data_path(calc_name)

        pts_count = self.pts_count(calc_name) # the memmap 1st dim
        (complex_codes, int_codes, stop_codes) = state.codes
        # keep only the one which do not sart with "_"
        f_complex_codes = self.filter_stored_codes(complex_codes)
        f_int_codes = self.filter_stored_codes(int_codes)
        n_Z, n_U, n_stop = (len(codes) for codes in 
                            (f_complex_codes, f_int_codes, stop_codes))
        # Followin C row-major order --> arr[x, :] shall be fast
        data_dim = {
            "Z": (n_Z, pts_count),
            "U": (n_U, pts_count),
            "stop_reason": (1, pts_count),
            "stop_iter": (1, pts_count),
        }

        for key in keys:
            mmap = open_memmap(
                filename=data_path[key], 
                mode='w+',
                dtype=np.dtype(data_type[key]),
                shape=data_dim[key],
                fortran_order=False,
                version=None
            )
            del mmap

        # Store the subset (if there is one) at this stage : it is already
        # known
        # /!\ the size of the subset is always the same, irrespective of
        # the number of items masked
        state = self._calc_data[calc_name]["state"]
        subset = state.subset

        if subset is not None:
            mmap = open_memmap(
                filename=data_path["subset"], 
                mode='w+',
                dtype=np.bool,
                shape=(self.nx * self.ny,),
                fortran_order=False,
                version=None
            )
            for rank, chunk_slice in enumerate(self.chunk_slices()):
                beg, end = self.uncompressed_beg_end(rank) # indep of calc_name
                mmap[beg:end] = subset[chunk_slice]

            del mmap


    def get_data_memmap(self, calc_name, key, mode='r+'):
        # See "Development Note - Memory mapping "
        data_path = self.data_path(calc_name)
        mmap = open_memmap(
            filename=data_path[key], mode=mode
        )
        return mmap


    def update_data_mmaps(
            self, calc_name, chunk_slice, Z, U, stop_reason, stop_iter
    ):
        state = self._calc_data[calc_name]["state"]
        keys = self.SAVE_ARRS
        arr_map = {
            "Z": Z,
            "U": U,
            "stop_reason": stop_reason,
            "stop_iter": stop_iter,
        }
        rank = self.chunk_rank(chunk_slice)
        beg, end = self.compressed_beg_end(calc_name, rank)

        # codes mapping - taking into account suppressed fields
        # (those starting with "_")
        (complex_codes, int_codes, stop_codes) = state.codes
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
            mmap = self.get_data_memmap(calc_name, key, mode="r+")
            arr = arr_map[key]
            for (field, f_field) in zip(*codes_index_map[key]):
                mmap[field, beg:end] = arr[f_field, :]


    def reload_data(self, chunk_slice, calc_name):
        """ Reload all stored raw arrays for this chunk : 
        raw_data = subset, Z, U, stop_reason, stop_iter
        """
        keys = self.SAVE_ARRS
        rank = self.chunk_rank(chunk_slice)
        beg, end = self.compressed_beg_end(calc_name, rank)

        arr = dict()
        for key in keys:
            mmap = self.get_data_memmap(calc_name, key)
            arr[key] = mmap[:, beg:end]
            del mmap

        state = self._calc_data[calc_name]["state"]
        subset = state.subset

        if subset is not None:
            # /!\ fixed-size irrespective of the mask
            beg, end = self.uncompressed_beg_end(rank) # indep of calc_name
            mmap = self.get_data_memmap(calc_name, "subset")
            arr["subset"] = mmap[beg: end]
            del mmap
        else:
            arr["subset"] = None
        
        # Reload c_pic - Note: we are in a "not final" (aka developement)
        # rendering ; jitter / supersampling are desactivated
        c_pix = np.ravel(
            self.chunk_pixel_pos(chunk_slice, jitter=False, supersampling=None)
        )
        if subset is not None:
            c_pix = c_pix[arr["subset"]]

        return (
            arr["subset"], c_pix, arr["Z"], arr["U"], arr["stop_reason"],
            arr["stop_iter"]
        )


    @Multithreading_iterator(
        iterable_attr="chunk_slices", iter_kwargs="chunk_slice"
    )
    def compute_rawdata_dev(self, calc_name, chunk_slice):
        """ In dev mode we follow a 2-step approach : here the compute step
        """
        if self.is_interrupted():
            logger.debug(
                "Interrupted - skipping calc for:\n"
                f"  {calc_name} ; {chunk_slice}"
            )
            return

        if self.res_available(calc_name, chunk_slice):
            logger.debug(
                "Result already available - skipping calc for:\n"
                f"  {calc_name} ; {chunk_slice}"
            )
            self.incr_tiles_status(which="Calc tiles")
            return

        (cycle_dep_args, chunk_subset
         ) = self.get_cycling_dep_args(calc_name, chunk_slice)
        cycle_indep_args = self._calc_data[calc_name]["cycle_indep_args"]
        
        ret_code = self.numba_cycle_call(cycle_dep_args, cycle_indep_args)
        if ret_code == self.USER_INTERRUPTED:
            logger.warning("Interruption signal received")
            return

        (c_pix, Z, U, stop_reason, stop_iter) = cycle_dep_args
        self.update_report_mmap(calc_name, chunk_slice, stop_reason)
        self.update_data_mmaps(
                calc_name, chunk_slice, Z, U, stop_reason, stop_iter
        )
        self.incr_tiles_status(which="Calc tiles")


    def reload_rawdata_dev(self, calc_name, chunk_slice):
        """ In dev mode we follow a 2-step approach : here the reload step
        """
        if self.res_available(calc_name, chunk_slice):
            self.incr_tiles_status(which="Plot tiles")
            return self.reload_data(chunk_slice, calc_name)
        else:
            # Interrupted calculation ?
            if self.res_available(calc_name, None):
                return None
            raise RuntimeError(f"Results unavailable for {calc_name}")


    def evaluate_rawdata_final(self, calc_name, chunk_slice, postproc_options):
        # we ARE in final render
        # - take into account postproc_options
        # - do not save raw data arrays to avoid too much disk usage (e.g.,
        #   in case of supersampling...)
        jitter = float(postproc_options["jitter"]) # casting to float
        supersampling = SUPERSAMPLING_DIC[postproc_options["supersampling"]]
        (cycle_dep_args, chunk_subset) = self.get_cycling_dep_args(
            calc_name, chunk_slice,
            final=True, jitter=jitter, supersampling=supersampling
        )

        cycle_indep_args = self._calc_data[calc_name]["cycle_indep_args"]
        ret_code = self.numba_cycle_call(cycle_dep_args, cycle_indep_args)

        self.incr_tiles_status(which="Plot tiles")

        if ret_code == self.USER_INTERRUPTED:
            logger.error("Interruption signal received")
            return

        (c_pix, Z, U, stop_reason, stop_iter) = cycle_dep_args
        return (chunk_subset, c_pix, Z, U, stop_reason, stop_iter)


    def evaluate_data(self, calc_name, chunk_slice, postproc_options):
        """ Compute on the fly the raw arrays for this chunk : 
        raw_data = subset, Z, U, stop_reason, stop_iter
        Note: this is normally activated only for a final redering
        """
        if postproc_options["final_render"]:
            ret = self.evaluate_rawdata_final(
                calc_name, chunk_slice, postproc_options
            )
            # Post calc saving
            if hasattr(self, "_subset_hook"):
                self.save_subset_arrays(
                    calc_name, chunk_slice, postproc_options, ret
                )
        else:
            ret = self.reload_rawdata_dev(calc_name, chunk_slice)
        return ret


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
            raise KeyError(
                "raw data code unknow: " + code, complex_codes,
                int_codes, "stop_reason", "stop_iter"
            )
        return kind

    @staticmethod
    def reshape2d(chunk_array, subset, chunk_slice):
        """
        Returns 2d versions of the 1d stored vecs
               chunk_array of size (n_post, n_pts)
        
        # note : to get a 2-dimensionnal vec do:
                 if bool_mask is not None:
                 we need to inverse
                     subset = np.ravel(subset[chunk_size])
                     c = c[subset]
        """
        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
        n_post, n_pts = chunk_array.shape

        chunk_1d = Fractal.reshape1d(chunk_array, subset, chunk_slice)
        return np.reshape(chunk_1d, [n_post, nx, ny])
    
    @staticmethod
    def reshape1d(chunk_array, subset, chunk_slice, supersampling=None):
        """
        Returns unmasked 1d versions of the 1d stored vecs
        """
        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
        n_post, n_pts = chunk_array.shape

        if subset is None:
            chunk_1d = chunk_array
        else:
            npts = nx * ny
            if supersampling is not None:
                npts *= supersampling ** 2
            indices = np.arange(npts)[subset]
            chunk_1d = np.empty([n_post, npts], dtype=chunk_array.dtype)
            chunk_1d[:] = np.nan
            chunk_1d[:, indices] = chunk_array

        return chunk_1d
    
    @staticmethod
    def index2d(index_1d, subset, chunk_slice):
        """ Return the 2d-indexing from 1d + mask 
        subset = None | self.subset[chunk_slice]
        """
        # chunk_size = fssettings.chunk_size
        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
        ix, iy = np.indices((nx, ny))
        ix = np.ravel(ix)
        iy = np.ravel(iy)
        if subset is not None:
            ix = ix[subset]
            iy = iy[subset]
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


    def calc_raw(self, calc_name):
        """ Here we can create the memory mappings and launch the calculation
        loop"""
        # Note: at this point res_available(calc_name) IS True, however
        # mmaps might not have been created.
        if self._calc_data[calc_name]["need_new_mmap"]:
            self.init_report_mmap(calc_name)
            self.init_data_mmaps(calc_name)

        # Launching the calculation + mmap storing multithreading loop
        self.compute_rawdata_dev(calc_name, chunk_slice=None)


    def postproc(self, postproc_batch, chunk_slice, postproc_options):
        """ Computes the output of ``postproc_batch`` for chunk_slice
        Return
          post_array of shape(nposts, chunk_n_pts)
          subset
        """
        if postproc_batch.fractal is not self:
            raise ValueError("Postproc batch from a different factal provided")
        calc_name = postproc_batch.calc_name
        
        if postproc_options["final_render"]:
            self.set_status("Calc tiles", "No [final]", bool_log=True)

        ret = self.evaluate_data(calc_name, chunk_slice, postproc_options)
        if ret is None:
            # Unexpected interruption
            return None

        (subset, c_pix, Z, U, stop_reason, stop_iter) = ret
        codes = self._calc_data[calc_name]["saved_codes"]
        complex_dic, int_dic, termination_dic = self.codes_mapping(*codes)

        # Compute c from cpix
        c_pt = self.get_std_cpt(c_pix)

        postproc_batch.set_chunk_data(chunk_slice, subset, c_pt, Z, U,
            stop_reason, stop_iter, complex_dic, int_dic, termination_dic)

        # Output data
        n_pts = Z.shape[1]  # Z of shape [n_Z, n_pts]
        post_array = np.empty(
            (len(postproc_batch.posts), n_pts), dtype=self.float_postproc_type
        )

        for i, postproc in enumerate(postproc_batch.posts.values()):
            val, context_update = postproc[chunk_slice]
            post_array[i, :]  = val
            postproc_batch.update_context(chunk_slice, context_update)

        postproc_batch.clear_chunk_data(chunk_slice)

        return post_array, subset


    def get_std_cpt(self, c_pix):
        """ Return the c complex value from c_pix """
        n_pts, = c_pix.shape  # Z of shape [n_Z, n_pts]
        # Explicit casting to complex / float
        dx = float(self.dx)
        center = complex(self.x + 1j * self.y)
        xy_ratio = self.xy_ratio
        theta = self.theta_deg / 180. * np.pi # used for expmap
        projection = getattr(PROJECTION_ENUM, self.projection).value
        cpt = np.empty((n_pts,), dtype=c_pix.dtype)
        fill1d_c_from_pix(
            c_pix, dx, center, xy_ratio, theta, projection, cpt
        )
        return cpt
        

#==============================================================================
# GUI : "interactive options"
#==============================================================================
    def coords(self, x, y, pix, dps):
        """ x, y : coordinates of the event """
        x_str = str(x)
        y_str = str(y)
        res_str = f"""
coords = {{
    "x": "{x_str}"
    "y": "{y_str}"
}}
"""
        return res_str


#==============================================================================

class _Subset_temporary_array:
    def __init__(self, fractal, calc_name_from, calc_name_to, key, func, inv):
        """ Drop-in replacement for Fractal_array in case of on-the-fly 
        computation: used when final calculation & subset activated """
        self.fractal = fractal
        self.calc_name_from = calc_name_from
        self.calc_name_to = calc_name_to
        self.key = key
        self._func = func
        self.inv = inv

        # Parsing the func string if needed
        self.func = func
        if func is not None:
            if isinstance(func, str):
                self.func = fs_parser.func_parser(["x"], func)

    def path(self):
        """ Memory mapping used in case of on-fly calculation
        """
        return os.path.join(
            self.fractal.directory, "data", self.calc_name_to + "_subset._img"
        )

    def init_mmap(self, supersampling):
        """ Create the memory mapping
        Format: 1-d UNcompressed per chunk, takes into account supersampling
        as needed. 
        """
        subset_path = self.path()
        logger.debug(f"Opening subset mmap at: {subset_path}")

        npts = self.fractal.nx * self.fractal.ny
        if supersampling is not None:
            npts *= supersampling ** 2
        mmap = open_memmap(
            filename=subset_path, 
            mode='w+',
            dtype=np.dtype(bool),
            shape=(npts,),
            fortran_order=False,
            version=None
        )
        del mmap
        # Storing status (/!\ not thread safe)
        self.supersampling = supersampling
        # del mmap
        # self._mmap = mmap
        
    def close_mmap(self, supersampling):
        try:
            del self._mmap
        except AttributeError:
            pass

    def __setitem__(self, chunk_slice, bool_arr):
        f = self.fractal
        rank = f.chunk_rank(chunk_slice)
        beg, end = f.uncompressed_beg_end(rank)
        ssg = self.supersampling
        if ssg is not None:
            beg *= ssg ** 2
            end *= ssg ** 2
        _mmap = open_memmap(self.path(), mode='r+')
        _mmap[beg: end] = bool_arr
        del _mmap

    def __getitem__(self, chunk_slice):
        f = self.fractal
        rank = f.chunk_rank(chunk_slice)
        beg, end = f.uncompressed_beg_end(rank)
        ssg = self.supersampling
        if ssg is not None:
            beg *= ssg ** 2
            end *= ssg ** 2
        _mmap = open_memmap(self.path(), mode='r+')
        ret = _mmap[beg: end]
        del _mmap
        return ret


    def save_array(self, chunk_slice, ret):
        """ Compute on the fly the boolean array & save it in memory mapping"""
        (subset, c_pix, Z, U, stop_reason, stop_iter) = ret
        f = self.fractal

        if subset  is not None:
            raise NotImplementedError(
                "Chained subset 'on the fly', currently not implemented"
            )

        # Identify the base 1d array
        codes = f._calc_data[self.calc_name_from]["saved_codes"]
        kind = f.kind_from_code(self.key, codes)
        complex_dic, int_dic, termination_dic = f.codes_mapping(*codes)

        if kind == "complex":
            arr_base = Z[complex_dic[self.key]]
        elif kind == "int":
            arr_base = U[int_dic[self.key]]
        elif kind in "stop_reason":
            arr_base = stop_reason[0]
        elif kind == "stop_iter":
            arr_base = stop_iter[0]
        else:
            raise ValueError(kind)

        # Apply func + inv as needed
        if self.func is not None:
            arr_base = self.func(arr_base)
        if self.inv:
            arr_base = ~arr_base

        self[chunk_slice] = arr_base



#==============================================================================
# Numba JIT functions
#==============================================================================
        
USER_INTERRUPTED = 1

@numba.njit(nogil=True)
def numba_cycles(
    c_pix, Z, U, stop_reason, stop_iter,
    initialize, iterate,
    dx, center, xy_ratio, theta, projection,
    _interrupted
):
    # Full iteration for a set of points - calls numba_cycle
    npts = c_pix.size

    for ipt in range(npts):
        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        cpt = c_from_pix(
            c_pix[ipt], dx, center, xy_ratio, theta, projection
        )
        stop_pt = stop_reason[:, ipt]

        initialize(Zpt, Upt, cpt)
        n_iter = iterate(
            # Zpt, Upt, cpt, stop_pt, 0,
            cpt, Zpt, Upt, stop_pt
        )
        stop_iter[0, ipt] = n_iter
        stop_reason[0, ipt] = stop_pt[0]

        if _interrupted[0]:
            return USER_INTERRUPTED

    return 0

def numba_iterate(
    calc_orbit, i_znorbit, backshift, zn, iterate_once, zn_iterate
):
    """ Numba implementation - recompiled if options change """
    @numba.njit(nogil=True)
    def numba_impl(c, Z, U, stop):
        n_iter = 0

        if calc_orbit:
            div_shift = 0
            orbit_zn1 = Z[zn]
            orbit_zn2 = Z[zn]
            orbit_i1 = 0
            orbit_i2 = 0

        while True:
            n_iter += 1
            ret = iterate_once(c, Z, U, stop, n_iter)

            if calc_orbit:
                div = n_iter // backshift
                if div > div_shift:
                    div_shift = div
                    orbit_i2 = orbit_i1
                    orbit_zn2 = orbit_zn1
                    orbit_i1 = n_iter
                    orbit_zn1 = Z[zn]

            if ret != 0:
                break

        if calc_orbit:
            zn_orbit = orbit_zn2
            while orbit_i2 < n_iter - backshift:
                zn_orbit = zn_iterate(zn_orbit, c)
                orbit_i2 += 1
            Z[i_znorbit] = zn_orbit

        return n_iter
    return numba_impl


def numba_iterate_BS(
    calc_orbit, i_xnorbit, i_ynorbit, backshift, xn, yn,
    iterate_once, xnyn_iterate
):
    """ Numba implementation - recompiled if options change """
    @numba.njit(nogil=True)
    def numba_impl(c, Z, U, stop):
        n_iter = 0

        if calc_orbit:
            div_shift = 0
            orbit_xn1 = Z[xn]
            orbit_xn2 = Z[xn]
            orbit_yn1 = Z[yn]
            orbit_yn2 = Z[yn]
            orbit_i1 = 0
            orbit_i2 = 0
        
        a = c.real
        b = c.imag

        while True:
            n_iter += 1
            ret = iterate_once(c, Z, U, stop, n_iter)

            if calc_orbit:
                div = n_iter // backshift
                if div > div_shift:
                    div_shift = div
                    orbit_i2 = orbit_i1
                    orbit_xn2 = orbit_xn1
                    orbit_yn2 = orbit_yn1
                    orbit_i1 = n_iter
                    orbit_xn1 = Z[xn]
                    orbit_yn1 = Z[yn]

            if ret != 0:
                break

        if calc_orbit:
            xn_orbit = orbit_xn2
            yn_orbit = orbit_yn2
            while orbit_i2 < n_iter - backshift:
                xn_orbit, yn_orbit = xnyn_iterate(xn_orbit, yn_orbit, a, b)
                orbit_i2 += 1
            Z[i_xnorbit] = xn_orbit
            Z[i_ynorbit] = yn_orbit

        return n_iter
    return numba_impl


def numba_Newton(   
    known_orders, max_order, max_newton, eps_newton_cv,
    reason_max_order, reason_order_confirmed,
    izr, idzrdz, idzrdc, id2zrdzdc, i_partial,  i_zn, iorder,
    zn_iterate, iterate_newton_search
):
    """ Numba implementation 
    Run Newton search to find z0 so that f^n(z0) == z0 (n being the order)
    """
    @numba.njit(nogil=True)
    def numba_impl(c, Z, U, stop):
        
        n_iter = 0
        while True:
            n_iter += 1

            if n_iter > max_order:
                stop[0] = reason_max_order
                break

            # If n is not a 'partial' for this point it cannot be the 
            # cycle order : early exit
            Z[i_zn] = zn_iterate(Z[i_zn], c)
            m = np.abs(Z[i_zn])
            if m < Z[i_partial].real:
                # This is a new partial, might be an attracting cycle
                Z[i_partial] = m # Cannot assign to the real part 
            else:
                continue

            # Early exit if n it is not a multiple of one of the
            # known_orders (provided by the user)
            if known_orders is not None:
                valid = False
                for order in known_orders:
                    if  n_iter % order == 0:
                        valid = True
                if not valid:
                    continue

            z0_loop = complex(0.) # Z[0]
            dz0dc_loop = complex(0.)  # Z[2]

            for i_newton in range(max_newton):
                zr = z0_loop
                dzrdc = dz0dc_loop
                dzrdz = complex(1.)
                d2zrdzdc = complex(0.)

                for i in range(n_iter):
                    # It seems it is not possible to split and compute the
                    # derivatives in postprocssing when convergence is reached
                    # --> we do it during the Newton iterations
                    d2zrdzdc, dzrdc, dzrdz, zr = iterate_newton_search(
                            d2zrdzdc, dzrdc, dzrdz, zr, c
                    )

                delta = (zr - z0_loop) / (dzrdz - 1.)
                newton_cv = (np.abs(delta) < eps_newton_cv)
                zz = z0_loop - delta
                dz0dc_loop = dz0dc_loop - (
                    (dzrdc - dz0dc_loop) / (dzrdz - 1.)
                    - (zr - z0_loop) * d2zrdzdc / (dzrdz - 1.)**2
                )
                z0_loop = zz
                if newton_cv:
                    break

            # We have a candidate but is it the good one ?
            is_confirmed = (np.abs(dzrdz) <= 1.) & newton_cv
            if not(is_confirmed): # not found, try next
                continue

            Z[izr] = zr
            Z[idzrdz] = dzrdz # attr (cycle attractivity)
            Z[idzrdc] = dzrdc
            Z[id2zrdzdc] = d2zrdzdc # dattrdc
            U[iorder] = n_iter
            stop[0] = reason_order_confirmed
            break

        # End of while loop
        return n_iter

    return numba_impl


proj_cartesian = PROJECTION_ENUM.cartesian.value
proj_spherical = PROJECTION_ENUM.spherical.value
proj_expmap = PROJECTION_ENUM.expmap.value

@numba.njit
def apply_skew_2d(skew, arrx, arry):
    "Unskews the view"
    nx = arrx.shape[0]
    ny = arrx.shape[1]
    for ix in range(nx):
        for iy in range(ny):
            tmpx = arrx[ix, iy]
            tmpy = arry[ix, iy]
            arrx[ix, iy] = skew[0, 0] * tmpx + skew[0, 1] * tmpy
            arry[ix, iy] = skew[1, 0] * tmpx + skew[1, 1] * tmpy

@numba.njit
def apply_unskew_1d(skew, arrx, arry):
    "Unskews the view for contravariant coordinates e.g. normal vec"
    n = arrx.shape[0]
    for i in range(n):
        nx = arrx[i]
        ny = arry[i]
        arrx[i] = skew[0, 0] * nx + skew[1, 0] * ny
        arry[i] = skew[0, 1] * nx + skew[1, 1] * ny

@numba.njit
def apply_rot_2d(theta, arrx, arry):
    s = np.sin(theta)
    c = np.cos(theta)
    nx = arrx.shape[0]
    ny = arrx.shape[1]
    for ix in range(nx):
        for iy in range(ny):
            tmpx = arrx[ix, iy]
            tmpy = arry[ix, iy]
            arrx[ix, iy] = c * tmpx - s * tmpy
            arry[ix, iy] = s * tmpx + c * tmpy


@numba.njit
def c_from_pix(pix, dx, center, xy_ratio, theta, projection):
    """
    Returns the true c from the pixel coords - Note: to be reimplemnted for 
    pertubation theory, as C = cref + dc

    Parameters
    ----------
    pix :  complex
        pixel location in fraction of dx

    Returns
    -------
    c : c value as complex
    """
    # Case cartesian
    if projection == proj_cartesian:
        offset = (pix * dx)

    elif projection == proj_spherical:
        dr_sc = np.abs(pix) * np.pi
        if dr_sc >= np.pi * 0.5:
            k = np.nan
        elif dr_sc < 1.e-12:
            k = 1.
        else:
            k = np.tan(dr_sc) / dr_sc
        offset = (pix * k * dx)

    elif projection == proj_expmap:
        dy = dx * xy_ratio
        h_max = 2. * np.pi * xy_ratio # max h reached on the picture
        xbar = (pix.real + 0.5 - xy_ratio) * h_max  # 0 .. hmax
        ybar = pix.imag / dy * 2. * np.pi           # -pi .. +pi
        rho = dx * 0.5 * np.exp(xbar)
        phi = ybar + theta
        offset = rho * (np.cos(phi) + 1j * np.sin(phi))

    return offset + center

@numba.njit
def fill1d_c_from_pix(c_pix, dx, center, xy_ratio, theta, projection,
                               c_out):
    """ Same as c_from_pix but fills in-place a 1d vec """
    nx = c_pix.shape[0]
    for i in range(nx):
        c_out[i] = c_from_pix(
            c_pix[i], dx, center, xy_ratio, theta, projection
        )
