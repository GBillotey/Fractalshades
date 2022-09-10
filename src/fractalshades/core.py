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

import numpy as np
from numpy.lib.format import open_memmap
import PIL
import PIL.PngImagePlugin
import numba

import fractalshades as fs
import fractalshades.settings as fssettings
import fractalshades.utils as fsutils

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


class Fractal_plotter:
    
    SUPERSAMPLING_DIC = {
        "2x2": 2,
        "3x3": 3,
        "4x4": 4,
        "5x5": 5,
        "6x6": 6,
        "7x7": 7,
        None: None
    }

    def  __init__(self, postproc_batch, final_render=False, supersampling=None,
                  jitter=False, reload=False):
        """
        The base plotting class.
        
        A Fractal plotter is a container for 
        `fractalshades.postproc.Postproc_batch` and fractal layers.
         
        Parameters
        ----------
        postproc_batch
            A single `fractalshades.postproc.Postproc_batch` or a list of 
            these
        final_render: bool
            - If False, this is an exploration rendering, the raw arrays will be
              stored to allow fast modifcation of the plotting parameters
              (without recomputing)
            - If True, this is the final rendering, the RGB arrays will
              be directly computed by chunks on the fly to limit disk usage.
              high quality rendering options are available in this case
              (antialising, jitter)

        supersampling: None | "2x2" | ... | "7x7"
            Used only for the final render. if not None, the final image will
            leverage supersampling (from 4 to 49 pixels computed for 1 pixel in 
            the saved image)
        jitter: bool | float
            Used only for the final render. If not None, the final image will
            leverage jitter, default intensity is 1. This can help reduce moiré
            effect
        reload: bool
            Used only for the final render. If True, will attempt to reload
            the image tiles already computed. Allows to restart an interrupted
            calculation

        Notes
        -----

        .. warning::
            When passed a list of `fractalshades.postproc.Postproc_batch`
            objects, each postprocessing batch shall point to the same
            unique `Fractal` object.
        """
        self.postproc_options = {
            "final_render": final_render,
            "supersampling": supersampling,
            "jitter": jitter,
            "reload": reload
        }

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
        
        # Mem mapping dtype
        self.post_dtype = np.float32 # TODO check this - use float64 ??
    
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
            plotter batches.

        .. warning::
            When a layer is added, a link layer -> Fractal_plotter is 
            created ; a layer can only point to a single `Fractal_plotter`.
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

    def plot(self):
        """
        The base method to produce images.
        
        When called, it will got through all the instance-registered layers
        and plot each layer for which the `output` attribute is set to `True`.
        """
        # Refactoring with direct calculation
        logger.info(self.plotter_info_str()) 
        
        self.open_images()
        self._raw_arr = dict()
        self._current_tile = {
                "value": 0,
                "time": 0.  # time of last evt in seconds
        }

        # Here we follow a 2-step approach:
        # if "dev" (= not final) render, we do compute the 'raw' arrays.
        # This is done by pbatches, due to the univoque relationship
        # one postproc_batches <-> Fractal calculation
        # if "final" we do not store omst raw datas, but still need to create a
        # mmap for the "subset" array
        f = self.fractal
        for pbatch in self.postproc_batches:
            calc_name = pbatch.calc_name
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


#    def initialize_plot(self):
    @property
    def try_reload(self):
        """ Will we try to reopen saved image chunks ?"""
        return (
            self.postproc_options["reload"]
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
            return self.SUPERSAMPLING_DIC[
                    self.postproc_options["supersampling"]
            ]
        # Note: implicitely None by default...

    @Multithreading_iterator(
        iterable_attr="chunk_slices", iter_kwargs="chunk_slice"
    )
    def plot_tiles(self, chunk_slice):
        """
        
        """
        # early exit if already computed
        if self.try_reload:
            f = self.fractal
            rank = f.chunk_rank(chunk_slice)
            is_valid = (self._mmap_status[rank] > 0)
            if is_valid:
                for i, layer in enumerate(self.layers):
                    # We need the postproc
                    # TODO what about 'update scaling' ???
                    # It is invalid as we lost the data...
                    self.push_reloaded(
                        chunk_slice, layer=layer, im=self._im[i], ilayer=i
                    )
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
        for batch in self.postproc_batches:
            # Do the real job
            self.fill_raw_arr(
                raw_arr, chunk_slice, batch, batch_first_post
            )
            # Just count index
            batch_first_post += len(batch.posts)

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
        post_array, subset = f.postproc(
            batch, chunk_slice, self.postproc_options
        )
        arr_1d = f.reshape1d(post_array, subset, chunk_slice,
                             self.supersampling)
        # arr_2d = f.reshape2d(post_array, subset, chunk_slice)
        n_posts, _ = arr_1d.shape

        raw_arr[inc: (inc + n_posts), :] = arr_1d


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
            self._mmap_status[rank] = 1

        if bool_log:
            self._current_tile["time"] = curr_time
            str_val = str(curr_val)+ " / " + str(f.chunks_count)
            logger.info(f"Image output: {str_val}")



#        def set_status(self, key, str_val):
#        """ Just a simple text status """
#        # if hasattr(self, "_status_wget"): # always has...
#        self._status_wget.update_status(key, str_val)
#        logger.info(f"{key}: {str_val}")
#
#    def incr_tiles_status(self):
#        """ Dealing with more complex status : 
#        Increase by 1 the number of computed tiles reported in status bar
#        """
#        dic = self._status_wget._status
#        dic["Tiles"]["val"] += 1
#        str_val = str(dic["Tiles"]["val"]) + " / " + str(self.chunks_count)
#        self.set_status("Tiles", str_val)


#==============================================================================

#    def store_postprocs(self):
#        """ Computes and stores posprocessed data in a temporary mmap
#        :meta private:
#        """
#        if fs.settings.optimize_RAM:
#            self.has_memmap = True
#            self.open_temporary_mmap()
#        else:
#            self.has_memmap = False
#            self.open_RAM_data()
#
#        inc_postproc_rank = 0
#        self.postname_rank = dict()
#        
#        for batch in self.postproc_batches:
#            if self.has_memmap:
#                self.store_temporary_mmap(
#                        chunk_slice=None,
#                        batch=batch,
#                        inc_postproc_rank=inc_postproc_rank
#                )
#            else:
#                
#                self.store_data(
#                        chunk_slice=None,
#                        batch=batch,
#                        inc_postproc_rank=inc_postproc_rank
#                )
#            for i, postname in enumerate(batch.postnames):
#                self.postname_rank[postname] = inc_postproc_rank + i
#            inc_postproc_rank += len(batch.posts)
#        
#        self.fractal.close_data_mmaps()
#        self.fractal.close_report_mmaps()


#    def temporary_mmap_path(self):
#        """ Path to the temporary memmap used to stored plotting arrays"""
#        # from tempfile import mkdtemp
#        return os.path.join(
#            self.fractal.directory, "data", "_plotter.tpm")
#
#    def open_temporary_mmap(self):
#        """
#        Creates the memory mappings for postprocessed arrays
#        Note: We expand to 2d format
#        """
#        f = self.fractal
#        nx, ny = (f.nx, f.ny)
#        n_pp = len(self.posts)
#        open_memmap(
#            filename=self.temporary_mmap_path(), 
#            mode='w+',
#            dtype=self.post_dtype,
#            # shape=(n_pp, nx, ny),
#            shape=(n_pp, nx * ny),
#            fortran_order=False,
#            version=None)

#    def get_temporary_mmap(self, mode='r'):
## A word of caution: Every instance of numpy.memmap creates its own mmap
## of the whole file (even if it only creates an array from part of the
## file). The implications of this are A) you can't use numpy.memmap's
## offset parameter to get around file size limitations, and B) you
## shouldn't create many numpy.memmaps of the same file. To work around
## B, you should create a single memmap, and dole out views and slices.
#        # attr = "_temporary_mmap" # + mode
#        if hasattr(self, "_temporary_mmap"):
#            return self._temporary_mmap
#        else:
#            val = open_memmap(filename=self.temporary_mmap_path(), mode=mode)
#            self._temporary_mmap = val
#            return val
#
#    def close_temporary_mmap(self):
#        if hasattr(self, "_temporary_mmap"):
#            # self._temporary_mmap.flush()
#            del self._temporary_mmap
#
#
#    def open_RAM_data(self):
#        """
#        Same as open_temporary_mmap but in RAM
#        """
#        f = self.fractal
#        nx, ny = (f.nx, f.ny)
#        n_pp = len(self.posts)
#        self._RAM_data = np.zeros(
#            shape=(n_pp, nx * ny), dtype=self.post_dtype
#        )


#    # multithreading: OK
#    @Multithreading_iterator(
#        iterable_attr="chunk_slices", iter_kwargs="chunk_slice"
#    )
#    def store_temporary_mmap(self, chunk_slice, batch, inc_postproc_rank):
#        """ Compute & store temporary arrays for this postproc batch
#            Note : inc_postproc_rank rank shift to take into account potential
#            other batches for this plotter.
#            (memory mapping version)
#        """
#        f = self.fractal
#        inc = inc_postproc_rank
#        post_array, subset = self.fractal.postproc(
#                batch, chunk_slice, self.postproc_options
#        )
#        arr_1d = f.reshape1d(post_array, subset, chunk_slice)
#        # arr_2d = f.reshape2d(post_array, subset, chunk_slice)
#        n_posts, _ = arr_1d.shape
#        # (ix, ixx, iy, iyy) = chunk_slice
#        # mmap = open_memmap(filename=self.temporary_mmap_path(), mode='r+')
#        mmap = self.get_temporary_mmap(mode="r+")
#
#        # Mmap shall be contiguous by chunks to speed-up disk access
#        rank = f.chunk_rank(chunk_slice)
#        beg, end = f.uncompressed_beg_end(rank)
##        chunk1d_begin
##        chunk1d_end
#        # mmap[inc:inc+n_posts, ix:ixx, iy:iyy] = arr_2d
#        mmap[inc:inc+n_posts, beg:end] = arr_1d
#        # mmap.flush()
#        # del mmap
        


#    # multithreading: OK
#    @Multithreading_iterator(
#        iterable_attr="chunk_slices", iter_kwargs="chunk_slice"
#    )
#    def store_data(self, chunk_slice, batch, inc_postproc_rank):
#        """ Compute & store temporary arrays for this postproc batch
#            (in-RAM version -> shall not use multiprocessing)
#        """
#        f = self.fractal
#        inc = inc_postproc_rank
#        post_array, subset = self.fractal.postproc(
#                batch, chunk_slice, self.postproc_options
#        )
#        arr_1d = f.reshape1d(post_array, subset, chunk_slice)
#        # arr_2d = f.reshape2d(post_array, subset, chunk_slice)
#        # n_posts, cx, cy = arr_2d.shape
#        n_posts, _ = arr_1d.shape
#        # Use contiguous data by chunks to speed-up memory access
#        rank = f.chunk_rank(chunk_slice)
#        beg, end = f.uncompressed_beg_end(rank)
#
#        # (ix, ixx, iy, iyy) = chunk_slice
#        # self._RAM_data[inc:inc+n_posts, ix:ixx, iy:iyy] = arr_2d
#        self._RAM_data[inc:inc+n_posts, beg:end] = arr_1d
#
#    # All methods needed for plotting
#    def compute_scalings(self):
#        """ Compute the scaling for all layer field
#        (needed for mapping to color) """
#        for layer in self.layers:
#            self.compute_layer_scaling(chunk_slice=None, layer=layer)


    def get_2d_arr(self, post_index, chunk_slice):
        """
        Returns a 2d view of a chunk for the given post-processed field
        """
        try:
            arr = self._raw_arr[chunk_slice][post_index, :]
        except KeyError:
            raise ValueError("Data requested for an unexpected chunk")

        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
        
        ssg = self.supersampling
        if ssg is not None:
            nx *= ssg
            ny *= ssg

        return np.reshape(arr, (nx, ny))


    def write_postproc_report(self):
        txt_report_path = os.path.join(
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

        with open(txt_report_path, 'w', encoding='utf-8') as report:
            for i, layer in enumerate(self.layers):
                write_layer_report(i, layer, report)
        
        logger.info(textwrap.dedent(f"""\
            Plotting image - postprocessing fields info saved to:
              {txt_report_path}"""
        ))

#    # multithreading: OK
#    @Multithreading_iterator(
#        iterable_attr="chunk_slices", iter_kwargs="chunk_slice"
#    )
#    def compute_layer_scaling(self, chunk_slice, layer):
#        """ Compute the scaling for this layer """
#        layer.update_scaling(chunk_slice)
    def image_name(self, layer):
        return "{}_{}".format(type(layer).__name__, layer.postname)

    def open_images(self):
        """ Open 
         - the image files
         - the associated memory mappings in case of "final render"
        ("""
        self._im = []
        if self.final_render:
            self._mmaps = []
            self.open_mmap_status()
        
        for layer in self.layers:
            if layer.output:
                self._im += [PIL.Image.new(mode=layer.mode, size=self.size)]
                if self.final_render:
                    self._mmaps += [self.open_mmap(layer)]

            else:
                self._im += [None]    
                if self.final_render:
                    self._mmaps += [None]

        if self.try_reload:
            valid_chunks = np.count_nonzero(self._mmap_status)
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
        file_path = os.path.join(
                self.plot_dir,"data", "final_render" + ".arr"
        )

        try:
            # Does layer the mmap already exists, and does it seems to suit
            # our need ?
            if not(self.try_reload):
#                print("DEBUG no try_reload")
                raise ValueError("Invalidated mmap_status")
            _mmap_status = open_memmap(
                filename=file_path, mode="r+"
            )
            if (_mmap_status.shape != (n_chunk,)):
#                print("DEBUG shape")
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

        self._mmap_status = _mmap_status


    def open_mmap(self, layer):
        """ mmap filled in // of the actual image - to allow restart
        Only for final render 
        """
        mode = layer.mode
        dtype = fs.colors.layers.Virtual_layer.DTYPE_FROM_MODE[mode]
        channel = fs.colors.layers.Virtual_layer.N_CHANNEL_FROM_MODE[mode]
        nx, ny = self.size
        file_name = self.image_name(layer)
        file_path = os.path.join(self.plot_dir, "data", file_name + "._img")

        # Does the mmap already exists, and does it seems to suit our need ?
        try:
            # Does layer the mmap already exists, and does it seems to suit
            # our need ?
            if not(self.postproc_options["reload"]):
                raise ValueError("Invalidated mmap_status")
            mmap = open_memmap(
                filename=file_path, mode="r+"
            )
            if mmap.shape != (ny, nx, channel):
                raise ValueError("Incompatible shapes for mmap")
            if mmap.dtype != dtype:
                raise ValueError("Incompatible dtype for mmap")

        except (FileNotFoundError, ValueError):
            # Create a new one...
            mmap = open_memmap(
                    filename=file_path, 
                    mode='w+',
                    dtype=dtype,
                    shape=(ny, nx, channel),
                    fortran_order=False,
                    version=None
            )
            # Here as we didnt find information for this layer, sadly the whole
            # memory mapping is invalidated
            logger.debug(f"No valid data found for layer {file_name}")
            self._mmap_status[:] = 0

        return mmap


    def save_images(self):
        """ Writes the images to disk """
        for i, layer in enumerate(self.layers):
            if not(layer.output):
                continue
            file_name = self.image_name(layer) #"{}_{}".format(type(layer).__name__, layer.postname)
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
        if (fssettings.output_context["doc"]
            and not(fssettings.output_context["gui_iter"] > 0)):
            fssettings.add_figure(_Pillow_figure(img, pnginfo))
        else:
            img.save(img_path, pnginfo=pnginfo)
            
            logger.info(textwrap.dedent(f"""\
                Image of shape {self.size} saved to:
                  {img_path}"""
            ))

#    def push_layers_to_images(self):
#        for i, layer in enumerate(self.layers):
#            if not(layer.output):
#                continue
#            self.push_cropped(chunk_slice=None, layer=layer, im=self._im[i])

#    # multithreading: OK
#    @Multithreading_iterator(
#        iterable_attr="chunk_slices", iter_kwargs="chunk_slice"
#    )
    def push_cropped(self, chunk_slice, layer, im, ilayer):
        """ push "cropped image" from layer for this chunk to the image"""
        (ix, ixx, iy, iyy) = chunk_slice
        ny = self.fractal.ny
        crop_slice = (ix, ny-iyy, ixx, ny-iy)
        paste_crop = layer.crop(chunk_slice)
        
        if self.supersampling:
            # Here, we should apply a resizig filter
            # Image.resize(size, resample=None, box=None, reducing_gap=None)
            print("### resizing filter")
            print("paste_crop", paste_crop, paste_crop.size, paste_crop.mode)
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
            # print("paste_crop_arr:", paste_crop_arr.shape, paste_crop_arr.dtype)
            layer_mmap = self._mmaps[ilayer]
            layer_mmap[iy: iyy, ix: ixx, :] = paste_crop_arr
            # layer_mmap[ix: ixx, (ny-iyy): (ny-iy), :] = paste_crop_arr
            # ValueError: could not broadcast input array from shape (167,200,3) into shape (200,167,3)

    def push_reloaded(self, chunk_slice, layer, im, ilayer):
        """ Just grap the already computed pixels and paste them"""
        if im is None:
            return
        (ix, ixx, iy, iyy) = chunk_slice
        ny = self.fractal.ny
        crop_slice = (ix, ny-iyy, ixx, ny-iy)
        layer_mmap = self._mmaps[ilayer]
        paste_crop = PIL.Image.fromarray(layer_mmap[iy: iyy, ix: ixx, :])
        im.paste(paste_crop, box=crop_slice)


#    def clean_up(self):
#        if self.has_memmap:
#            os.unlink(self.temporary_mmap_path())
#        else:
#            del self._RAM_data

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


class Fractal:

    REPORT_ITEMS = [
        "chunk1d_begin",
        "chunk1d_end",
        "chunk_pts",
        "done",
    ]
    # Note : subset is pre-computed and saved also but not at the same 
    # stage (at begining of calculation)
    SAVE_ARRS = [
        "Z",
        "U",
        "stop_reason",
        "stop_iter"
    ]
    PROJECTION_ENUM = {
        "cartesian": 1,
        "spherical": 2,
        "expmap": 3
    }
    USER_INTERRUPTED = 1

    def __init__(self, directory):
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

Attributes
----------
directory : str
    the working directory
subset
    A boolean array-like of the size of the image, when False the
    calculation is skipped for this point. It is usually the result
    of a previous calculation that can be passed via a Fractal_array
    wrapper.
    If None (default) all points will be calculated.
complex_type :
    the datatype used for Z output arrays
codes :
    the string identifier codes for the saved arrays:
    (`complex_codes`, `int_codes`, `termination_codes`)

Notes
-----

These notes describe implementation details and should be useful mostly to
advanced users when subclassing.

.. note::

    **Special methods**
    
    this class and its subclasses may define several methods decorated with
    specific tags:
    
    `fractalshades.zoom_options`
        decorates the methods used to define the zoom
    `fractalshades.calc_options`
        decorates the methods defining the calculation inner-loop
    `fractalshades.interactive_options` 
        decorates the methods that can be called
        interactively from the GUI (right-click then context menu selection).
        The coordinates of the click are passed to the called method.

.. note::
    
    **Calculation parameters**

    To lanch a calculation, call `~fractalshades.Fractal.run`. The parameters
    from the last 
    @ `fractalshades.zoom_options` call and last 
    @ `fractalshades.calc_options` call will be used. 
    They are stored as class attributes, above a list of such attributes and
    their  meaning (non-exhaustive as derived class cmay define their own).
    Note that they should normally not be directly accessed but defined in
    derived class through zoom and calc methods.

.. note::

    **Saved data**
    
        The calculation results (raw output of the inner loop at exit) are
        saved to disk and internally accessed during plotting phase through
        memory-mapping. These are:

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
        self.float_postproc_type = np.float32
        self.termination_type = np.int8
        self.int_type = np.int32
        
        # Default listener
        self._status_wget = _Null_status_wget(self)

#    def init_data_types(self, complex_type):
#        if type(complex_type) is tuple:
#            raise RuntimeError("Xrange is deprecated")


    def _repr(self):
        # String used to generate a new instance in GUI-generated scripts
        return (
            "fsm."
            + self.__class__.__name__
            + "(plot_dir)"
        )

    @fsutils.zoom_options
    def zoom(self, *,
             x: float,
             y: float,
             dx: float,
             nx: int,
             xy_ratio: float,
             theta_deg: float,
             projection: str="cartesian",
             antialiasing: bool=False  # DEPRECATED
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
        antialiasing : bool
            Deprecated - Use the options provided by `Fractal_plotter`
            (This parameter might be removed in a future version)
        """
        if antialiasing:
            logger.warn(
                "antialiasing parameter for zoom is obsolete, "
                "Use the supersampling option provided by `Fractal_plotter`"
            )

        # Safeguard in case the GUI inputs were strings
        if isinstance(x, str) or isinstance(y, str) or isinstance(dx, str):
            raise RuntimeError("Float expected")

    def new_status(self, wget):
        """ Return a dictionnary that can hold the current progress status """
        self._status_wget = wget
        status = {
            "Tiles": {
                "val": 0,
                "str_val": "- / -",
                "last_log": 0.  # time of last logged change in seconds
            }
        }
        return status

    def set_status(self, key, str_val, bool_log=True):
        """ Just a simple text status """
        # if hasattr(self, "_status_wget"): # always has...
        self._status_wget.update_status(key, str_val)
        if bool_log:
            logger.info(f"{key}: {str_val}")

    def incr_tiles_status(self):
        """ Dealing with more complex status : 
        Increase by 1 the number of computed tiles reported in status bar
        """
        dic = self._status_wget._status
        curr_val = dic["Tiles"]["val"] + 1
        dic["Tiles"]["val"] = curr_val

        prev_event = dic["Tiles"]["last_log"]
        curr_time = time.time()
        time_diff = curr_time - prev_event

        ntiles = self.chunks_count
        bool_log = ((time_diff > 1) or (curr_val == 1) or (curr_val == ntiles))
        if bool_log:
            dic["Tiles"]["last_log"] = curr_time

        str_val = str(dic["Tiles"]["val"]) + " / " + str(ntiles)
        self.set_status("Tiles", str_val, bool_log)




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
             calc_name + ".params",
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
        if r != 0:
            cy += 1
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
        if r != 0:
            cy += 1
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

    def uncompressed_beg_end(self, rank):  # Should be uncompressed_beg_end
        """ Return the span for the boolean mask index for the chunk index
        `rank` """
        # Note: indep of the mask.
        valid = False
        if hasattr(self, "_uncompressed_beg_end"):  # TODO
            arr, in_nx, in_ny, in_csize = self._uncompressed_beg_end
            valid = (
                self.nx == in_nx,
                self.ny == in_ny,
                fssettings.chunk_size == in_csize
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
        self._uncompressed_beg_end = (arr, self.nx, self.ny, fssettings.chunk_size)
        return arr

    def compressed_beg_end(self, calc_name, rank):
        """ Return the span for a stored array
        """
        # Note : depend of the mask... hence the calc_name
        mmap = self.get_report_memmap(calc_name)

        items = self.REPORT_ITEMS
        beg = mmap[rank, items.index("chunk1d_begin")]
        end = mmap[rank, items.index("chunk1d_end")]

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


#    def chunk_pts(self, calc_name, chunk_slice):
#        """
#        Return the number of compressed 1d points for this chunk_slice
#        (taking into account 2d subset bool if available)
#        """
#        state = self._calc_data[calc_name]["state"]
#        subset = state.subset
#
#        # print("subset:", subset)
#        (ix, ixx, iy, iyy) = chunk_slice
#        if subset is not None:
#            subset_pts = np.count_nonzero(subset[chunk_slice])
#            return subset_pts
#        else:
#            return (ixx - ix) * (iyy - iy)

#    @property
#    def chunk_mask(self):
#        """ Legacy - simple alias """
#        return self.subset


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
            dx_vec += (0.5 - rand_x) * 0.5 / (nx - 1) * jitter
            dy_vec += (0.5 - rand_y) * 0.5 / (ny - 1) * jitter
            if supersampling is not None:
                dx_vec /= supersampling
                dy_vec /= supersampling

        dy_vec /= self.xy_ratio

        # Apply the Linear part of the tranformation
        if theta != 0 and self.projection != "expmap":
            apply_rot_2d(theta, dx_vec, dy_vec)

        if self.skew is not None:
            apply_skew_2d(self.skew, dx_vec, dy_vec)

        res = dx_vec + 1j * dy_vec

        return res


#    def param_matching(self, dparams):
#        """
#        Test if the stored parameters match those of new calculation
#        /!\ modified in subclass
#        """
#        UNTRACKED = ["datetime", "debug"] 
#        for key, val in self.params.items():
#            if not(key in UNTRACKED) and dparams[key] != val:
#                logger.debug(textwrap.dedent(f"""\
#                    Parameter mismatch ; will trigger a recalculation
#                      {key}, {val} --> {dparams[key]}"""
#                ))
#                return False
#        return True

    @property    
    def tag(self):
        """ Used to tag an output image
        """
        tag = {
            "Software": "fractalshades " + fs.__version__,
            "datetime": datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        }
        for calc_name in self._calc_data.keys():
            state = self._calc_data[calc_name]["state"]
            tag[calc_name] = state.fingerprint
        # tag = fs.utils.dic_flatten(self.fingerprint))
        return fs.utils.dic_flatten(tag)



    def fingerprint_matching(self, calc_name, test_fingerprint):
        """
        Test if the stored parameters match those of new calculation
        /!\ modified in subclass
        """
#        print("test_fingerprint:\n", test_fingerprint)
#        print("expected_fingerprint:\n", self.fingerprint)
#        print("test_fingerprint flatten:\n", fs.utils.dic_flatten(test_fingerprint))
#        print("expected fingerprint flatten:\n", fs.utils.dic_flatten(self.fingerprint))
        
        flatten_fp = fs.utils.dic_flatten(test_fingerprint)

        state = self._calc_data[calc_name]["state"]
        expected_fp = fs.utils.dic_flatten(state.fingerprint)

        UNTRACKED = ["datetime", "debug"]

        for key, val in expected_fp.items():
            if not(key in UNTRACKED) and flatten_fp[key] != val:
                logger.debug(textwrap.dedent(f"""\
                    Parameter mismatch ; will trigger a recalculation
                      {key}, {val} --> {flatten_fp[key]}"""
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
            print("######################## debug reload_fingerprint")
            fingerprint = self.reload_fingerprint(calc_name)
            print("######################## debug reload_fingerprint...   OK")
        except IOError:
            logger.debug(textwrap.dedent(f"""\
                No fingerprint file found for {calc_name};
                  will trigger a recalculation"""
                ))
            return False

        matching = self.fingerprint_matching(calc_name, fingerprint)
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
        Called by a calculation wrapper
        Prepares & stores the data needed for future calculation of tiles
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
        }

        # Takes a 'fingerprint' of the calculation parameters
        fp_items = (
            "calc_class", "calc_callable", "calc_kwargs", "zoom_kwargs"
        )
        state.fingerprint = {
            k: self._calc_data[calc_name][k] for k in fp_items
        }
#        {
#            "calc_class": type(self).__name__,
#            "calc_callable": calc_callable,
#            "calc_kwargs": calc_kwargs,
#            "zoom_kwargs": self.zoom_kwargs,
#        }
        # Adding a subset hook for future 'on the fly' computation
        if "subset" in calc_kwargs.keys():
            subset = calc_kwargs["subset"]
            if subset is not None:
                self.add_subset_hook(subset, calc_name)

        if self.res_available(calc_name):
            logger.info(
                f"Found raw results files for {calc_name}\n"
                "  -> only the missing tiles will be recomputed"
            )
        else:
            # NOTE THAT WE CANNOT OPEN MEM MAP at this early stage : because
            # subset may still be unknown. But, we shall clear them.
            self.del_report_memmap(calc_name)
            self.del_data_mmaps(calc_name)
            self.save_fingerprint(calc_name, state.fingerprint)


#    def mmap_init(self, calc_name):
#        self.init_report_mmap(calc_name)
#        self.init_data_mmaps(calc_name)


#    def ensure_state(self, calc_name):
#        # TODO : this is not thread safe ! as we are now mixing different 
#        # calculations
#        # need to define a mapping: calc_name -> state
#        """ Before running a tile for a specific calculation, ensure the
#        fractal is in the correct internal state
#        (subset & other internal state items)
#        """
#        if self._lazzy_calc_state == calc_name:
#            return
#        else:
#            dat = self._calc_data[calc_name]
#            calc_kwargs = dat["calc_kwargs"]
#            set_state = dat["set_state"]
#
#            # Storing calc calling paarmeters
#            for k, v in calc_kwargs.items():
#                setattr(self, k, v)
#
#            # setting additionnnal parameters
#            set_state(self)
#
#            # Storing the 'fingerprint' for comparison with stored data
#            fp_items = (
#                "calc_class", "calc_callable", "calc_kwargs", "zoom_kwargs"
#            )
#            self.fingerprint = {
#                k: self._calc_data[calc_name][k] for k in fp_items
#            }
#
#            # Logging
#            self._lazzy_calc_state = calc_name
#            logger.info(
#                f"Resetted fractal object internal state for calc: {calc_name}"
#            )

#        raise NotImplementedError()


#    def run(self):
#        """
#        Launch a full calculation.
#
#        The zoom and calculation options from the last
#        @ `fractalshades.zoom_options`\-tagged method called and last
#        @ `fractalshades.calc_options`\-tagged method called will be used.
#
#        If calculation results are already there, the parameters will be
#        compared and if identical, the calculation will be skipped. This is
#        done for each tile, so it enables calculation to restart from an
#        unfinished status.
#        
#        Parameters:
#        -----------
#        lazzy_evaluation: bool
#            If True, this is the final rendering, the RGB arrays will be
#            computed on the fly during plotting. Intermediate arrays will not
#            be stored.
#        """
#        # We use lazzy evaluation and postpone the actual calculation to the
#        # plotting step (during tiles computation)
#        # 
#        # At this stage we do not know which kind of plotting (final render ?
#        # antialisasing ? etc) So only save in a dic with key <calc_name>
#        # - zoom_kwargs
#        # - calc_kwargs
#        # - calc_callable
#
#        # codes & other stuff needed *before* looping should not be managed
#        # as side effects (as currently) but rather grouped in a function
#        # ? set_context ???  set_state ???
#        
#        # That way you can get
#        # - codes (This is needed to size the memory mapping that we need to
#        #   before looping)
#        # - M & other stuff needed for postproc
#
#        zoom_kwargs = self.zoom_kwargs
#        calc_callable = self.calc_callable
#        calc_params = self.calc_kwargs
#
#        if not(hasattr(self, "_lazzy_data")):
#            self._lazzy_data = dict()
#        cycle_indep_args = self.get_cycle_indep_args()
#        
#        self._lazzy_data[self.calc_name] = {
#            "zoom_kwargs": self.zoom_kwargs,
#            "set_state": self.set_state,
#            "iterate": self.initialyze(),
#            "iterate": self.iterate(),
#            "cycle_indep_args": cycle_indep_args,
#        }
#       self.save_params()
#            self._lazzy_cycle_indep_args[self.calc_name] = cycle_indep_args
#    
#            if not(hasattr(self, "_lazzy_params")):
#                self._lazzy_params = dict()
#            self._lazzy_params[self.calc_name] = self.params
#            calc_hook
            # What about the codes ??

#            # We still need a disk copy as it is used everywhere
#            self.save_params()
#            # This is a placeholder if a non-fianl is run after
#            self.init_report_mmap()

        # else:
            # Launch parallel computing of the inner-loop
            # (Multi-threading with GIL released)

#            self.initialize_cycles()
#            self.cycles(cycle_indep_args, chunk_slice=None)
#            self.finalize_cycles()

#
#    def initialize_cycles(self):
#        """ a few things to open """
#        if self.res_available():
#            logger.info("Tiles: Found existing raw results files\n"
#                        "  -> only the missing tiles will be recomputed")
#        else:
#            # We write the param file and initialize the
#            # memmaps for progress reports and calc arrays
#            fsutils.mkdir_p(os.path.join(self.directory, "data"))
#            self.init_report_mmap()# open_report_mmap()
#            self.init_data_mmaps() # #open_data_mmaps()
#            self.save_params()
#
#        # Open the report mmap in mode "r+" as we will need to modify it 
#        self.get_report_memmap(mode='r+')


    def finalize_cycles(self):
        """ A few clean-up stuff"""
        self.set_status("Tiles", "completed")

        # Export to human-readable format
        if fs.settings.inspect_calc:
            self.inspect_calc()

        # Close memory mappings
        self.close_data_mmaps()
        self.close_report_mmaps()


    def raise_interruption(self):
        self._interrupted[0] = True
        
    def lower_interruption(self):
        self._interrupted[0] = False
    
    def is_interrupted(self):
        """ Either programmatically 'interrupted' (from the GUI) or by the user 
        in batch mode through fs.settings.skip_calc """
        return (self._interrupted[0] or fs.settings.skip_calc)


#    @Multithreading_iterator(
#        iterable_attr="chunk_slices", iter_kwargs="chunk_slice")

#    def cycles(self, cycle_indep_args, chunk_slice):
#        """
#        Fast-looping for Julia and Mandelbrot sets computation.
#
#        Parameters
#        *initialize*  function(Z, U, c) modify in place Z, U (return None)
#        *iterate*   function(Z, U, c, n) modify place Z, U (return None)
#
#        *subset*   bool arrays, iteration is restricted to the subset of current
#                   chunk defined by this array. In the returned arrays the size
#                    of axis ":" is np.sum(subset[ix:ixx, iy:iyy]) - see below
#        *codes*  = complex_codes, int_codes, termination_codes
#        *calc_name* prefix identifig the data files
#        *chunk_slice_c* None - provided by the looping wrapper
#
#        Returns 
#        None - save to a file. 
#        *raw_data* = (subset, Z, U, stop_reason, stop_iter) where
#            *subset*    1d mask
#            *Z*             Final values of iterated complex fields shape [ncomplex, :]
#            *U*             Final values of int fields [nint, :]       np.int32
#            *stop_reason*   Byte codes -> reasons for termination [:]  np.int8
#            *stop_iter*     Numbers of iterations when stopped [:]     np.int32
#        """
#        if self.is_interrupted():
#            return
#        if self.res_available(chunk_slice):
#            self.incr_tiles_status()
#            return
#
#        cycle_dep_args = self.get_cycling_dep_args(chunk_slice)
#        # Customization hook (for child-classes) :
#        ret_code = self.numba_cycle_call(cycle_dep_args, cycle_indep_args)
#
#        if ret_code == self.USER_INTERRUPTED:
#            logger.error("Interruption signal received")
#            return
##        self.incr_tiles_status()
#
#        (c_pix, Z, U, stop_reason, stop_iter) = cycle_dep_args
#        self.update_report_mmap(chunk_slice, stop_reason)
#        self.update_data_mmaps(chunk_slice, Z, U, stop_reason, stop_iter)
#
#        return Z, U, stop_reason, stop_iter
        

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
        projection = self.PROJECTION_ENUM[self.projection]

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
            c_pix = c_pix[subset[chunk_slice]]

        # Initialise the result arrays
        (n_pts,) = c_pix.shape

        n_Z, n_U, n_stop = (len(code) for code in self.codes)

        Z = np.zeros([n_Z, n_pts], dtype=self.complex_type)
        U = np.zeros([n_U, n_pts], dtype=self.int_type)
        stop_reason = - np.ones([1, n_pts], dtype=self.termination_type)
        stop_iter = np.zeros([1, n_pts], dtype=self.int_type)

        return (c_pix, Z, U, stop_reason, stop_iter)

#==============================================================================

    def add_subset_hook(self, f_array, calc_name_to):
        """ Create a namespace for a hook that shall be filled during
        on-the fly computation
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
#            bool_arr = self.get_bool_arr(
#                it_calc_name, it_key, it_func, it_inv, ret 
#            )


#    def update_subset_mmap(self, calc_name_to, chunk_slice, bool_arr): # private
#        """
#        Updates the memory mapping with a newly computed bool subset array
#        """
#        rank = self.chunk_rank(chunk_slice)
#        beg, end = self.uncompressed_beg_end(rank)
#        ssg = self._subset_hook[calc_name_to][1]
#        if ssg is not None:
#            beg *= ssg ** 2
#            end *= ssg ** 2
#        mmap = self._subset_hook[calc_name_to][2]
#        mmap[beg: end] = bool_arr

#    def reload_subset_mmap(self, calc_name_to, chunk_slice):
#        """ Reload the stored subset array"""
#        rank = self.chunk_rank(chunk_slice)
#        beg, end = self.uncompressed_beg_end(rank)
#        ssg = self._subset_hook[calc_name_to][1]
#        if ssg is not None:
#            beg *= ssg ** 2
#            end *= ssg ** 2
#        mmap = self._subset_hook[calc_name_to][2]
#        return mmap[beg: end]



        



#    def get_bool_arr(self, calc_name, key, func, inv, ret):
#        """ A direct implementation of Fractal_array __get__ method
#        Used for "on-the-fly" image computation
#        """
#        (chunk_subset, Z, U, stop_reason, stop_iter) = ret
#
#        # Identify the base 1d array
#        codes = fractal._calc_data[calc_name]["saved_codes"]
#        kind = self.kind_from_code(key, codes)
#        if kind == "complex":
#            arr_base = Z[complex_dic[key]]
#        elif kind == "int":
#            arr_base = U[int_dic[key]]
#        elif kind in "stop_reason":
#            arr_base = stop_reason[0]
#        elif king == "stop_iter":
#            arr_base = stop_iter[0]
#        else:
#            raise ValueError(kind)
#
#        # Apply func + inv
#        if func is not None:
#            if isinstance(func, str):
#                func = fs_parser.func_parser(["x"], func)
#            arr_base = func(arr_base)
#        if func is not None:
#            arr_base = func(arr_base)


    # ======== The various storing files for a calculation ====================

#    def params_path(self, calc_name=None):
#        if calc_name is None:
#            calc_name = self.calc_name
#        return os.path.join(
#            self.directory, "data", calc_name + ".params")
    
#    def serializable_params(self, params):
#        """ Some params we do not want to save as-is but only keep partial
#        information. Saving them would duplicate information + require coding
#        ad-hoc __getstate__, __setstate__ methods.
#        """
#        unserializable = ("calc-param_subset",)
#        ret =calc_name_from + "_to_" +  {}
#        for k, v in params.items():
#            if k in unserializable:
#                if v is None:
#                    ret[k] = v
#                else:
#                    ret[k] = repr(v)
#            else:
#                ret[k] = v
#        # print("modified params ready to save", ret)
#        return ret
#
#    def save_params(self):
#        """
#        Save (pickle) current calculation parameters in data file,
#        Don't save temporary codes - i.e. those which startwith "_"
#        This should only be used to tag images and not to re-run a calculation.
#        """
#        (complex_codes, int_codes, stop_codes) = self.codes
#        f_complex_codes = self.filter_stored_codes(complex_codes)
#        f_int_codes = self.filter_stored_codes(int_codes)
#        saved_codes = (f_complex_codes, f_int_codes, stop_codes)
#
#        if self._lazzy_evaluation:
#            self._lazzy_data[self.calc_name]["params"] = self.params
#            self._lazzy_data[self.calc_name]["codes"] = saved_codes
#        else:
#            save_path = self.params_path()
#            fsutils.mkdir_p(os.path.dirname(save_path))
#            with open(save_path, 'wb+') as tmpfile:
#                s_params = self.serializable_params(self.params)
#                pickle.dump(s_params, tmpfile, pickle.HIGHEST_PROTOCOL)
#                pickle.dump(saved_codes, tmpfile, pickle.HIGHEST_PROTOCOL)
##        print("Saved calc params", save_path)
#
#    def reload_params(self, calc_name=None): # public
#        if self._lazzy_evaluation:
#            if calc_name is None:
#                calc_name = self.calc_name
#            params = self._lazzy_data[calc_name]["params"]
#            codes = self._lazzy_data[calc_name]["codes"]
#        else:
#            save_path = self.params_path(calc_name)
#            with open(save_path, 'rb') as tmpfile:
#                params = pickle.load(tmpfile)
#                codes = pickle.load(tmpfile)
#        return (params, codes)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fingerprint_path(self, calc_name):
        return os.path.join(
            self.directory, "data", calc_name + ".fingerprint"
        )

    def save_fingerprint(self, calc_name, fingerprint):
        save_path = self.fingerprint_path(calc_name)
        fsutils.mkdir_p(os.path.dirname(save_path))
        with open(save_path, 'wb+') as fp_file:
            pickle.dump(fingerprint, fp_file, pickle.HIGHEST_PROTOCOL)

    def reload_fingerprint(self, calc_name):
        """ Reloading the fingerprint from the saved files """
        save_path = self.fingerprint_path(calc_name)
        with open(save_path, 'rb') as tmpfile:
            fingerprint = pickle.load(tmpfile)
        # TODO here we should finalize fractal arrays
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
# Report path tracks the progress of the calculations
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

        save_path = self.report_path(calc_name)
        fsutils.mkdir_p(os.path.dirname(save_path))
        mmap = open_memmap(
            filename=save_path, 
            mode='w+',
            dtype=np.int32,
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
        
#        print("MMAP for calc_name", calc_name, ":")
#        print(mmap)

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
        attr = self.report_memmap_attr(calc_name)
        if hasattr(self, attr):
            return getattr(self, attr) #self._temporary_mmap
        else:
            val = open_memmap(
                filename=self.report_path(calc_name), mode=mode
            )
            setattr(self, attr, val)
            return val

    def report_memmap_attr(self, calc_name):
        return "__report_mmap_" + "_" + calc_name

    def close_report_mmaps(self):
        """ Deletes the file handles"""
        to_del = tuple()
        for item in filter(
                lambda x: x.startswith("__report_mmap_"),
                vars(self)
        ):  
            to_del += (item,)
        for item in to_del:
            delattr(self, item)

    def del_report_memmap(self, calc_name):
        """ Deletes the file itself"""
        file_name = self.report_path(calc_name)
        try:
            os.unlink(file_name)
            logger.debug(f"Obsolete report mmap deleted: {file_name}")
        except:
            pass # Nothing to do, file was already suppressed

    def update_report_mmap(self, calc_name, chunk_slice, stop_reason): # private
        """
        """
        mmap = self.get_report_memmap(calc_name, mode="r+")
        items = self.REPORT_ITEMS
        chunk_rank = self.chunk_rank(chunk_slice)
        mmap[chunk_rank, items.index("done")] = 1


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

        return report


    def inspect_calc(self, calc_name):
        """
        Outputs a report for the current calculation
        """
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
            subset, Z, U, stop_reason, stop_iter = self.reload_data(
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
        
        outpath = os.path.join(self.directory, self.calc_name + ".inspect")
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

        keys = self.SAVE_ARRS
        data_type = {
            "Z": self.complex_type,
            "U": self.int_type,
            "stop_reason": self.termination_type,
            "stop_iter": self.int_type,
        }

        data_path = self.data_path(calc_name)

        pts_count = self.pts_count(calc_name) # the memmap 1st dim
        (complex_codes, int_codes, stop_codes) = self.codes
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
            open_memmap(
                filename=data_path[key], 
                mode='w+',
                dtype=data_type[key],
                shape=data_dim[key],
                fortran_order=False,
                version=None
            )
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


    def get_data_memmap(self, calc_name, key, mode='r+'):
        # See "Development Note - Memory mapping "
        attr = self.dat_memmap_attr(key, calc_name)
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            data_path = self.data_path(calc_name)
            val = open_memmap(
                filename=data_path[key], mode=mode
            )
            setattr(self, attr, val)
            return val

    def dat_memmap_attr(self, key, calc_name):
        return "_@data_mmap_" + calc_name + "_" + key

    def close_data_mmaps(self):
        """ Close the files handles"""
        to_del = tuple()
        for item in filter(
                lambda x: x.startswith("_@data_mmap_"),
                vars(self)
        ):  
            to_del += (item,)
        for item in to_del:
            delattr(self, item)

    def del_data_mmaps(self, calc_name):
        """ Deletes the file itself"""
        keys = self.SAVE_ARRS + ["subset"]
        data_path = self.data_path(calc_name)
        for key in keys:
            file_name = data_path[key]
            try:
                os.unlink(file_name)  # TODO : try - catch
                logger.debug(f"Obsolete data mmap deleted: {file_name}")
            except FileNotFoundError:
                pass # Nothing to do, file xas not there


    def update_data_mmaps(
            self, calc_name, chunk_slice, Z, U, stop_reason, stop_iter):
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
            mmap = self.get_data_memmap(calc_name, key, mode="r+")
            arr = arr_map[key]
            for (field, f_field) in zip(*codes_index_map[key]):
                # print("update_data_mmaps", arr[f_field, :].shape, beg, end)
                # DEBUG beg, end seems not right
                mmap[field, beg:end] = arr[f_field, :]


    def reload_data(self, chunk_slice, calc_name):
        """ Reload all stored raw arrays for this chunk : 
        raw_data = subset, Z, U, stop_reason, stop_iter
        """

        keys = self.SAVE_ARRS
        rank = self.chunk_rank(chunk_slice)
        beg, end = self.compressed_beg_end(calc_name, rank)

        arr = dict()
        # data_path = self.data_path(calc_name)
        for key in keys:
            mmap = self.get_data_memmap(calc_name, key)
            arr[key] = mmap[:, beg:end]

        state = self._calc_data[calc_name]["state"]
        subset = state.subset

        if subset is not None:
            # /!\ fixed-size irrespective of the mask
            beg, end = self.uncompressed_beg_end(rank) # indep of calc_name
            mmap = self.get_data_memmap(calc_name, "subset")
            arr["subset"] = mmap[beg: end]
        else:
            arr["subset"] = None

        return (
            arr["subset"], arr["Z"], arr["U"], arr["stop_reason"],
            arr["stop_iter"]
        )


    @Multithreading_iterator(
        iterable_attr="chunk_slices", iter_kwargs="chunk_slice"
    )
    def compute_rawdata_dev(self, calc_name, chunk_slice):
        """ In dev mode we follow a 2-step approach : here the compute step
        """
        if self.res_available(calc_name, chunk_slice):
            return
        
        cycle_dep_args = self.get_cycling_dep_args(calc_name, chunk_slice)
        cycle_indep_args = self._calc_data[calc_name][
                "cycle_indep_args"
        ]
        ret_code = self.numba_cycle_call(cycle_dep_args, cycle_indep_args)
        (c_pix, Z, U, stop_reason, stop_iter) = cycle_dep_args

        if ret_code == self.USER_INTERRUPTED:
            logger.error("Interruption signal received")
            return

        self.update_report_mmap(calc_name, chunk_slice, stop_reason)
        self.update_data_mmaps(
                calc_name, chunk_slice, Z, U, stop_reason, stop_iter
        )

#        chunk_subset = None
#        state = self._calc_data[calc_name]["state"]
#        if state.subset is not None:
#            chunk_subset = state.subset[chunk_slice]
#        return (chunk_subset, Z, U, stop_reason, stop_iter)

        


    def reload_rawdata_dev(self, calc_name, chunk_slice):
        """ In dev mode we follow a 2-step approach : here the reload step
        """
        if self.res_available(calc_name, chunk_slice):
            return self.reload_data(chunk_slice, calc_name)
        else:
            raise RuntimeError(f"Results unavailable for {calc_name}")


    def evaluate_rawdata_final(self, calc_name, chunk_slice, postproc_options):
        # we ARE in final render
        # - take into account postproc_options
        # - do not save raw data arrays to avoid too much disk usage (e.g.,
        #   in case of supersampling...)
        jitter = float(postproc_options["jitter"]) # casting to float
        supersampling = Fractal_plotter.SUPERSAMPLING_DIC[
            postproc_options["supersampling"]
        ]
        cycle_dep_args = self.get_cycling_dep_args(
            calc_name, chunk_slice,
            final=True, jitter=jitter, supersampling=supersampling
        )
        cycle_indep_args = self._calc_data[calc_name]["cycle_indep_args"]
        ret_code = self.numba_cycle_call(cycle_dep_args, cycle_indep_args)

        if ret_code == self.USER_INTERRUPTED:
            logger.error("Interruption signal received")
            return

        (c_pix, Z, U, stop_reason, stop_iter) = cycle_dep_args
        
        chunk_subset = None
        state = self._calc_data[calc_name]["state"]
        if state.subset is not None:
            # Using the substitution
            chunk_subset = self._subset_hook[calc_name][chunk_slice]
            # state.subset[chunk_slice]
            # chunk_subset = state.subset[chunk_slice]
        
        return (chunk_subset, Z, U, stop_reason, stop_iter)


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
            chunk_1d = np.copy(chunk_array) # TODO: should we pass a reference?
        else:
            # Note: this will not work in case of antialiasing
            # Need a refactoring to be able to pass a fine mesh
            # For the moment we raise NotImplementedError upstream
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
        if not os.path.isfile(self.report_path(calc_name)):
            # Note that at this point res_available(calc_name) IS True, however
            # mmaps might not be created.
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
        
#        # Is it the final render ?
#        if postproc_options["final_render"]:
#            # This is the final render, we compute from scratch
        ret = self.evaluate_data(calc_name, chunk_slice, postproc_options)
        if ret is None:
            return
        (subset, Z, U, stop_reason, stop_iter) = ret
#        else:
#            # Not the final -> raw data shall have been precomputed & stored
#            (subset, Z, U, stop_reason, stop_iter
#             ) = self.reload_data(chunk_slice, calc_name)


        # params, codes = self.reload_params(calc_name)
        codes = self._calc_data[calc_name]["saved_codes"]
        complex_dic, int_dic, termination_dic = self.codes_mapping(*codes)

        postproc_batch.set_chunk_data(chunk_slice, subset, Z, U,
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
        computation """
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

    def init_mmap(self, supersampling): # private
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
            dtype=np.bool,
            shape=(npts,),
            fortran_order=False,
            version=None
        )
        # Storing status (/!\ not thread safe)
        self.supersampling = supersampling
        self._mmap = mmap

    def __setitem__(self, chunk_slice, bool_arr):
        f = self.fractal
        rank = f.chunk_rank(chunk_slice)
        beg, end = f.uncompressed_beg_end(rank)
        ssg = self.supersampling
        if ssg is not None:
            beg *= ssg ** 2
            end *= ssg ** 2
        self._mmap[beg: end] = bool_arr

    def __getitem__(self, chunk_slice):
        f = self.fractal
        rank = f.chunk_rank(chunk_slice)
        beg, end = f.uncompressed_beg_end(rank)
        ssg = self.supersampling
        if ssg is not None:
            beg *= ssg ** 2
            end *= ssg ** 2
        return self._mmap[beg: end]

#    def add_subset_hook(self, f_array, calc_name_to):
#        """ Create a namespace for a hook that shall be filled during
#        on-the fly computation
#           Hook structure:  dict 
#           hook[calcname_to] = [f_array_dat, supersampling_k, mmap]
#        """
#        logger.debug(f"Add Subset hook for {calc_name_to}")
#
#        if not(hasattr(self, "_subset_hook")):
#            self._subset_hook = dict()
#
#        self._subset_hook[calc_name_to] = [
#            (f_array.calc_name, f_array.key, f_array._func, f_array.inv),
#            None,
#            None
#        ]

    def save_array(self, chunk_slice, ret):
        """ Compute on the fly the boolean array & save it in memory mapping"""
        (subset, Z, U, stop_reason, stop_iter) = ret
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

#
#    def get_bool_arr(self, calc_name, key, func, inv, ret):
#        """ A direct implementation of Fractal_array __get__ method
#        Used for "on-the-fly" image computation
#        """
#        (chunk_subset, Z, U, stop_reason, stop_iter) = ret








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
    """ Run the standard cycles
    """
    npts = c_pix.size
    
    for ipt in range(npts):
        Zpt = Z[:, ipt]
        Upt = U[:, ipt]
        cpt = ref_path_c_from_pix(c_pix[ipt], dx, center, xy_ratio, theta,
                                  projection)
        stop_pt = stop_reason[:, ipt]

        initialize(Zpt, Upt, cpt)
        n_iter = iterate(
            Zpt, Upt, cpt, stop_pt, 0,
        )
        stop_iter[0, ipt] = n_iter
        stop_reason[0, ipt] = stop_pt[0]

        if _interrupted[0]:
            return USER_INTERRUPTED
    return 0


proj_cartesian = Fractal.PROJECTION_ENUM["cartesian"]
proj_spherical = Fractal.PROJECTION_ENUM["spherical"]
proj_expmap = Fractal.PROJECTION_ENUM["expmap"]

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
def ref_path_c_from_pix(pix, dx, center, xy_ratio, theta, projection): # TODO : rename this to proj_c_from_pix ??
    """
    Returns the true c (coords from ref point) from the pixel coords
    
    Parameters
    ----------
    pix :  complex
        pixel location in farction of dx
        
    Returns
    -------
    c, c_xr : c value as complex and as Xrange
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

#        elif self.projection == "spherical":
#            dr_sc = np.sqrt(dx_vec**2 + dy_vec**2) / max(dx, dy) * np.pi
#            k = np.where(dr_sc >= np.pi * 0.5, np.nan,  # outside circle
#                         np.where(dr_sc < 1.e-12, 1., np.tan(dr_sc) / dr_sc))
#            dx_vec *= k
#            dy_vec *= k
#            offset = [(dx_vec * np.cos(theta)) - (dy_vec * np.sin(theta)),
#                      (dx_vec * np.sin(theta)) + (dy_vec * np.cos(theta))]
#
#        elif self.projection == "mixed_exp_map":
#            # square + exp. map
#            h_max = 2. * np.pi * xy_ratio # max h reached on the picture
#            xbar = (dx_vec + 0.5 * dx - dy) / dx * h_max # 0 .. hmax
#            ybar = dy_vec / dy * 2. * np.pi              # -pi .. +pi
#            rho = dx * 0.5 * np.where(xbar > 0., np.exp(xbar), 0.)
#            phi = ybar + theta
#            dx_vec = (dx_vec + 0.5 * dx - 0.5 * dy) * xy_ratio
#            dy_vec = dy_vec * xy_ratio
#            offset = [np.where(xbar <= 0.,
#                          (dx_vec * np.cos(theta)) - (dy_vec * np.sin(theta)),
#                          rho * np.cos(phi)),
#                      np.where(xbar <= 0.,
#                          (dx_vec * np.sin(theta)) + (dy_vec * np.cos(theta)),
#                          rho * np.sin(phi))]
#
#        elif self.projection == "exp_map":
#            # only exp. map
#            h_max = 2. * np.pi * xy_ratio # max h reached on the picture
#            xbar = (dx_vec + 0.5 * dx - dy) / dx * h_max # 0 .. hmax
#            ybar = dy_vec / dy * 2. * np.pi              # -pi .. +pi
#            rho = dx * 0.5 * np.exp(xbar)
#            phi = ybar + theta
#            offset = [rho * np.cos(phi), rho * np.sin(phi)]
