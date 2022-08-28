# -*- coding: utf-8 -*-
import os
import fnmatch
import copy
import datetime
import pickle
import logging
import textwrap
import time

import numpy as np
from numpy.lib.format import open_memmap
import PIL
import PIL.PngImagePlugin
import numba

import fractalshades as fs
import fractalshades.settings as fssettings
import fractalshades.utils as fsutils

from fractalshades.mthreading import Multithreading_iterator


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
        Saves as a png with Lanczos antialiasing if exceeds the max width
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
    def  __init__(self, postproc_batch, final_render=False, antialiasing=None,
                  jitter=None, reload=False):
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
            If False, this is an exploration rendering based on already
            computed data
            If True, this is the final rendering, the RGB arrays will
            computed by chunks on the fly to limit disk usage
            saved during plot
        antialiasing: None | "2x2" | ... | "7x7"
            Used only for the final render. if not None, the final image will
            leverage antialiasing (from 4 to 49 pixels computed for 1 pixel in 
            the image)
        jitter: None | True | float
            Used only for the final render. If not None, the final image will
            leverage jitter
        reload: bool
            Used only for the final render. If True, will attempt to reload
            the tiles already computed

        Notes
        -----

        .. warning::
            When passed a list of `fractalshades.postproc.Postproc_batch`
            objects, each postprocessing batch shall point to the same
            unique `Fractal` object.
        """
        self.postproc_options = {
            "final_render": final_render,
            "antialiasing": antialiasing,
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
        str_info += ("\n  /!\\ antialiasing and jitter only activated "
                     + "for final render")
        return str_info

    def plot(self):
        """
        The base method to produce images.
        
        When called, it will got through all the instance-registered layers
        and plot each layer for which the `output` attribute is set to `True`.
        """
#        logger.info("Plotting image - postprocessing fields: computing")
#        self.store_postprocs()
#        self.close_temporary_mmap()
#        logger.info("Plotting image - postprocessing fields: done")
#
#        self.compute_scalings()
#        self.write_postproc_report()
#        self.open_images()
#
#        logger.info("Plotting image - colorized layers: computing")
#        self.push_layers_to_images()
#        logger.info("Plotting image - colorized layers: done")
#
#        self.save_images()
#        self.clean_up()
#        self.close_temporary_mmap()
        
        # Refactoring with direct calculation
        logger.info(self.plotter_info_str()) 
        logger.info("Plotting images: computing tiles")
        # self.postproc_options
        
        self.open_images()
        self._raw_arr = dict()
        self._current_tile = {
                "value": 0,
                "time": 0.  # time of last evt in seconds
        }
        
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
        arr_shp = (n_pp, (ixx - ix) * (iyy - iy))
        raw_arr = np.empty(shape= arr_shp,dtype=self.post_dtype)

        batch_first_post = 0 # index of the first post for the current batch
        for batch in self.postproc_batches:
            self.fill_raw_arr(raw_arr, chunk_slice, batch, batch_first_post)
            batch_first_post += len(batch.posts)

        self._raw_arr[chunk_slice] = raw_arr
        
        # 2) Push each layer crop to the relevant image 
        for i, layer in enumerate(self.layers):
            layer.update_scaling(chunk_slice)
            if layer.output:
                self.push_cropped(chunk_slice, layer=layer, im=self._im[i],
                                  ilayer=i)

        # clean-up
        self.incr_tiles_status(chunk_slice)
        del self._raw_arr[chunk_slice]
            
            
    def fill_raw_arr(self, raw_arr, chunk_slice, batch, batch_first_post):
        """ Compute & store temporary arrays for this postproc batch
            Note : inc_postproc_rank rank shift to take into account potential
            other batches for this plotter.
            (memory mapping version)
        """
        f = self.fractal
        inc = batch_first_post
        post_array, chunk_mask = f.postproc(
            batch, chunk_slice, self.postproc_options
        )
        arr_1d = f.reshape1d(post_array, chunk_mask, chunk_slice)
        # arr_2d = f.reshape2d(post_array, chunk_mask, chunk_slice)
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
#        post_array, chunk_mask = self.fractal.postproc(
#                batch, chunk_slice, self.postproc_options
#        )
#        arr_1d = f.reshape1d(post_array, chunk_mask, chunk_slice)
#        # arr_2d = f.reshape2d(post_array, chunk_mask, chunk_slice)
#        n_posts, _ = arr_1d.shape
#        # (ix, ixx, iy, iyy) = chunk_slice
#        # mmap = open_memmap(filename=self.temporary_mmap_path(), mode='r+')
#        mmap = self.get_temporary_mmap(mode="r+")
#
#        # Mmap shall be contiguous by chunks to speed-up disk access
#        rank = f.chunk_rank(chunk_slice)
#        beg, end = f.mask_beg_end(rank)
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
#        post_array, chunk_mask = self.fractal.postproc(
#                batch, chunk_slice, self.postproc_options
#        )
#        arr_1d = f.reshape1d(post_array, chunk_mask, chunk_slice)
#        # arr_2d = f.reshape2d(post_array, chunk_mask, chunk_slice)
#        # n_posts, cx, cy = arr_2d.shape
#        n_posts, _ = arr_1d.shape
#        # Use contiguous data by chunks to speed-up memory access
#        rank = f.chunk_rank(chunk_slice)
#        beg, end = f.mask_beg_end(rank)
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
#        f = self.fractal
#        if self.current_chunck != chunk_slice:
            
#        rank = f.chunk_rank(chunk_slice)
#        beg, end = f.mask_beg_end(rank)
        try:
            arr = self._raw_arr[chunk_slice][post_index, :]
        except KeyError:
            raise ValueError("Data requested for an unexpected chunk")

        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
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
                raise ValueError("Incompatible shapes for mmap_status")

        except (FileNotFoundError, ValueError):
            # Lets create it from scratch
            logger.debug(f"No valid status file found - recompute")
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
            self.save_tagged(self._im[i], base_img_path, self.fractal.params)

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
    # Note : chunk_mask is pre-computed and saved also but not at the same 
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

        chunk_mask    
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
        
        # Default listener
        self._status_wget = _Null_status_wget(self)

    def init_data_types(self, complex_type):
        if type(complex_type) is tuple:
            raise RuntimeError("Xrange is deprecated")
        self.complex_type = complex_type
        self.float_postproc_type = np.float32
        self.termination_type = np.int8
        self.int_type = np.int32

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
             antialiasing: bool=False  # TODO - this is not anti alisasing but jitter - move to finl render ??
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
            logger.warn("antialiasing parameter for zoom is obsolete, "
                          "Use the options provided by `Fractal_plotter`")

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

    @property    
    def params(self):
        """ Used to tag an output image or check if data is already computed
        and stored
        :meta private:
        """
        software_params = {
                "Software": "fractalshades " + fs.__version__,
                "fractal_type": type(self).__name__,
                # "debug": ("1234567890" * 10), # tested 10000 chars ok
                "datetime": datetime.datetime.today().strftime(
                        '%Y-%m-%d_%H:%M:%S')}
        zoom_params = self.zoom_options
        calc_function = self.calc_options_callable # TODO rename to calc_callable
        calc_params = self.calc_options

        res = dict(software_params)
        res.update(zoom_params)
        res["calc-function"] = calc_function
        res.update({"calc-param_" + k: v for (k, v) in calc_params.items()})

        return res


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

    @property
    def pts_count(self):
        """ Return the total number of points for the current calculation 
        taking into account the `subset` parameter
        :meta private:
        """
        if self.subset is not None:
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

    def mask_beg_end(self, rank):
        """ Return the span for the boolean mask index for the chunk index
        `rank` """
        # Note: indep of the mask.
        valid = False
        if hasattr(self, "_mask_beg_end"):
            arr, in_nx, in_ny, in_csize = self._mask_beg_end
            valid = (
                self.nx == in_nx,
                self.ny == in_ny,
                fssettings.chunk_size == in_csize
            )
        if not valid:
            arr = self.recompute_mask_beg_end(rank)
        return arr[rank], arr[rank + 1]

    def recompute_mask_beg_end(self, rank):
        """ A bit brutal but does the job
        """
        arr = np.empty((self.chunks_count + 1,), dtype=np.int32)
        arr[0] = 0
        for i, chunk_slice in enumerate(self.chunk_slices()):
            (ix, ixx, iy, iyy) = chunk_slice
            arr[i + 1] = arr[i] + (ixx - ix) * (iyy - iy)
        self._mask_beg_end = (arr, self.nx, self.ny, fssettings.chunk_size)
        return arr

    def arr_beg_end(self, rank, calc_name=None):
        """ Return the span for a stored array
        """
        # Note : depend of the mask... hence the calc_name
        mmap = self.get_report_memmap(calc_name)

        items = self.REPORT_ITEMS
        beg = mmap[rank, items.index("chunk1d_begin")]
        end = mmap[rank, items.index("chunk1d_end")]
        del mmap
        return beg, end

    def chunk_pts(self, chunk_slice):
        """
        Return the number of compressed 1d points for this chunk_slice
        (taking into account 2d subset bool if available)
        """
        subset = self.subset
        (ix, ixx, iy, iyy) = chunk_slice
        if subset is not None:
            subset_pts = np.count_nonzero(subset[chunk_slice])
            return subset_pts
        else:
            return (ixx - ix) * (iyy - iy)

    @property
    def chunk_mask(self):
        """ Legacy - simple alias """
        return self.subset


    def chunk_pixel_pos(self, chunk_slice):
        """
        Return the image pixels vector distance to center in fraction of image
        width, as a complex
           pix = center + (pix_frac_x * dx,  pix_frac_x * dy)
        """
        data_type = self.float_type

        theta = self.theta_deg / 180. * np.pi
        (nx, ny) = (self.nx, self.ny)
        (ix, ixx, iy, iyy) = chunk_slice

        kx = 0.5 / (nx - 1)
        x_1d = np.linspace(
                kx * (2 * ix - nx + 1), kx * (2 * ixx - nx - 1),
                num=(ixx - ix), dtype=data_type
        )

        ky = 0.5 / (ny - 1)
        y_1d = np.linspace(
                ky * (2 * iy - ny + 1), ky * (2 * iyy - ny - 1),
                num=(iyy - iy), dtype=data_type
        )

        # DEBUG - alternative definition
        if False:
            dx_grid = np.linspace(-0.5, 0.5, num=nx, dtype=data_type)
            dy_grid = np.linspace(-0.5, 0.5, num=ny, dtype=data_type)
            x_1dr = dx_grid[ix:ixx]
            y_1dr = dy_grid[iy:iyy]
            np.testing.assert_almost_equal(x_1dr, x_1d)
            np.testing.assert_almost_equal(y_1dr, y_1d)
        
        dy_vec, dx_vec  = np.meshgrid(y_1d, x_1d)

#        if antialiasing:
#            rg = np.random.default_rng(0)
#            rand_x = rg.random(dx_vec.shape, dtype=data_type)
#            rand_y = rg.random(dx_vec.shape, dtype=data_type)
#            dx_vec += (0.5 - rand_x) * 0.5 / nx
#            dy_vec += (0.5 - rand_y) * 0.5 / ny

        dy_vec /= self.xy_ratio

        # Apply the Linear part of the tranformation
        if theta != 0 and self.projection != "expmap":
            apply_rot_2d(theta, dx_vec, dy_vec)

        if self.skew is not None:
            apply_skew_2d(self.skew, dx_vec, dy_vec)
        
        res = dx_vec + 1j * dy_vec
        
        return res


    def param_matching(self, dparams):
        """
        Test if the stored parameters match those of new calculation
        /!\ modified in subclass
        """
        UNTRACKED = ["datetime", "debug"] 
        for key, val in self.params.items():
            if not(key in UNTRACKED) and dparams[key] != val:
                logger.debug(textwrap.dedent(f"""\
                    Parameter mismatch ; will trigger a recalculation
                      {key}, {val} --> {dparams[key]}"""
                ))
                return False
        return True

    def res_available(self, chunk_slice=None):
        """  
        If chunk_slice is None, check that stored calculation parameters
        matches.
        If chunk_slice is provided, checks that calculation results are
        available
        """
        try:
            params, codes = self.reload_params()
        except IOError:
            return False
        matching = self.param_matching(params)
        if not(matching):
            return False
        if chunk_slice is None:
            return matching # should be True

        try:
            report = self.reload_report(chunk_slice)
        except IOError:
            return False

        return report["done"] > 0


    def run(self, lazzy_evaluation=False):
        """
        Launch a full calculation.

        The zoom and calculation options from the last
        @ `fractalshades.zoom_options`\-tagged method called and last
        @ `fractalshades.calc_options`\-tagged method called will be used.

        If calculation results are already there, the parameters will be
        compared and if identical, the calculation will be skipped. This is
        done for each tile, so it enables calculation to restart from an
        unfinished status.
        
        Parameters:
        -----------
        lazzy_evaluation: bool
            If True, this is the final rendering, the RGB arrays will be
            computed on the fly during plotting. Intermediate arrays will not
            be stored.
        """
        self._lazzy_evaluation = lazzy_evaluation
        cycle_indep_args = self.get_cycle_indep_args()

        if lazzy_evaluation:
            # We just store these parameters for later - create the container
            # as needed
            if not(hasattr(self, "_lazzy_data")):
                self._lazzy_data = dict()
            
            self._lazzy_data[self.calc_name] = {
                "cycle_indep_args": cycle_indep_args,
            }
            self.save_params()
#            self._lazzy_cycle_indep_args[self.calc_name] = cycle_indep_args
#    
#            if not(hasattr(self, "_lazzy_params")):
#                self._lazzy_params = dict()
#            self._lazzy_params[self.calc_name] = self.params
            
            # What about the codes ??

#            # We still need a disk copy as it is used everywhere
#            self.save_params()
#            # This is a placeholder if a non-fianl is run after
#            self.init_report_mmap()

        else:
            # Launch parallel computing of the inner-loop
            # (Multi-threading with GIL released)
            self.initialize_cycles()
            self.cycles(cycle_indep_args, chunk_slice=None)
            self.finalize_cycles()


    def initialize_cycles(self):
        """ a few things to open """
        if self.res_available():
            logger.info("Tiles: Found existing raw results files\n"
                        "  -> only the missing tiles will be recomputed")
        else:
            # We write the param file and initialize the
            # memmaps for progress reports and calc arrays
            fsutils.mkdir_p(os.path.join(self.directory, "data"))
            self.init_report_mmap()# open_report_mmap()
            self.init_data_mmaps() # #open_data_mmaps()
            self.save_params()

        # Open the report mmap in mode "r+" as we will need to modify it 
        self.get_report_memmap(mode='r+')


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


    @Multithreading_iterator(
        iterable_attr="chunk_slices", iter_kwargs="chunk_slice")
    def cycles(self, cycle_indep_args, chunk_slice=None):
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

        *gliched* boolean Fractal_Data_array of pixels that should be updated
                  with a new ref point

        Returns 
        None - save to a file. 
        *raw_data* = (chunk_mask, Z, U, stop_reason, stop_iter) where
            *chunk_mask*    1d mask
            *Z*             Final values of iterated complex fields shape [ncomplex, :]
            *U*             Final values of int fields [nint, :]       np.int32
            *stop_reason*   Byte codes -> reasons for termination [:]  np.int8
            *stop_iter*     Numbers of iterations when stopped [:]     np.int32
        """
        if self.is_interrupted():
            return
        if self.res_available(chunk_slice):
            self.incr_tiles_status()
            return

        cycle_dep_args = self.get_cycling_dep_args(chunk_slice)
        # Customization hook :
        ret_code = self.numba_cycle_call(cycle_dep_args, cycle_indep_args)

        if ret_code == self.USER_INTERRUPTED:
            logger.error("Interruption signal received")
            return
        self.incr_tiles_status()

        (c_pix, Z, U, stop_reason, stop_iter) = cycle_dep_args
        self.update_report_mmap(chunk_slice, stop_reason)
        self.update_data_mmaps(chunk_slice, Z, U, stop_reason, stop_iter)

    @staticmethod
    def numba_cycle_call(cycle_dep_args, cycle_indep_args):
        # Just a thin wrapper to allow customization in derived classes
        return numba_cycles(*cycle_dep_args, *cycle_indep_args)

    def get_cycle_indep_args(self):
        """
        This is just a diggest of the zoom and calculation parameters
        """
        # Dev notes for "on the fly" calc -  In cycle you have :
        # - chunck dependant args -> Shall be stored once only !
        # - chunck dependant args ->  Shall be 
        initialize = self.initialize()
        iterate = self.iterate()

        dx = self.dx
        center = self.x + 1j * self.y
        xy_ratio = self.xy_ratio
        theta = self.theta_deg / 180. * np.pi # used for expmap
        projection = self.PROJECTION_ENUM[self.projection]

        return (
            initialize, iterate,  # INDEP
            dx, center, xy_ratio, theta, projection, # INDEP
            self._interrupted # INDEP
        )

    def get_cycling_dep_args(self, chunk_slice, use_current_subset=True,
                             subset=None):
        """
        The actual input / output arrays
        """
        c_pix = np.ravel(self.chunk_pixel_pos(chunk_slice))

        if use_current_subset:
            # This is the current calculation, we use class attribute
            # We need to be able to bypass this in case of lazzy eval only
            subset = self.subset

        if subset is not None:
            chunk_mask = self.subset[chunk_slice]
            c_pix = c_pix[chunk_mask]

        # Initialise the result arrays
        (n_pts,) = c_pix.shape

        n_Z, n_U, n_stop = (len(code) for code in self.codes)

        Z = np.zeros([n_Z, n_pts], dtype=self.complex_type)
        U = np.zeros([n_U, n_pts], dtype=self.int_type)
        stop_reason = - np.ones([1, n_pts], dtype=self.termination_type)
        stop_iter = np.zeros([1, n_pts], dtype=self.int_type)

        return (c_pix, Z, U, stop_reason, stop_iter)


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

        if self._lazzy_evaluation:
            self._lazzy_data[self.calc_name]["params"] = self.params
            self._lazzy_data[self.calc_name]["codes"] = saved_codes
        else:
            save_path = self.params_path()
            fsutils.mkdir_p(os.path.dirname(save_path))
            with open(save_path, 'wb+') as tmpfile:
                s_params = self.serializable_params(self.params)
                pickle.dump(s_params, tmpfile, pickle.HIGHEST_PROTOCOL)
                pickle.dump(saved_codes, tmpfile, pickle.HIGHEST_PROTOCOL)
#        print("Saved calc params", save_path)

    def reload_params(self, calc_name=None): # public
        if self._lazzy_evaluation:
            if calc_name is None:
                calc_name = self.calc_name
            params = self._lazzy_data[calc_name]["params"]
            codes = self._lazzy_data[calc_name]["codes"]
        else:
            save_path = self.params_path(calc_name)
            with open(save_path, 'rb') as tmpfile:
                params = pickle.load(tmpfile)
                codes = pickle.load(tmpfile)
        return (params, codes)

#==============================================================================
# Report path tracks the progress of the calculations
    def report_path(self, calc_name=None): # public
        if calc_name is None:
            calc_name = self.calc_name
        return os.path.join(
            self.directory, "data", calc_name + ".report")

    def init_report_mmap(self): # private
        """
        Create the memory mapping for calculation reports by chunks
        [chunk1d_begin, chunk1d_end, chunk_pts]
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
            version=None
        )

        mmap[:, items.index("done")] = 0 # Set to 1 when done

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
    

    def get_report_memmap(self, calc_name=None, mode='r'):
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
        if calc_name is None:
            calc_name = self.calc_name

        attr = self.report_memmap_attr(calc_name)
        if hasattr(self, attr):
            return getattr(self, attr) #self._temporary_mmap
        else:
            val = open_memmap(
                filename=self.report_path(calc_name), mode=mode
            )
            # self._temporary_mmap = val
            setattr(self, attr, val)
            return val

    def report_memmap_attr(self, calc_name):
        return "__report_mmap_" + "_" + calc_name

    def close_report_mmaps(self):
        to_del = tuple()
        for item in filter(
                lambda x: x.startswith("__report_mmap_"),
                vars(self)
        ):  
            to_del += (item,)
        for item in to_del:
            # getattr(self, item).flush()
            delattr(self, item) # self._temporary_mmap


    def update_report_mmap(self, chunk_slice, stop_reason): # private
        """
        """
        # mmap =self.report_mmap[self.calc_name]
        mmap = self.get_report_memmap(mode="r+")
        items = self.REPORT_ITEMS
        chunk_rank = self.chunk_rank(chunk_slice)
        mmap[chunk_rank, items.index("done")] = 1


    def reload_report(self, chunk_slice, calc_name=None): # public
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


    def inspect_calc(self):
        """
        Outputs a report for the current calculation
        """
        REPORT_ITEMS, report = self.reload_report(None)
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
            chunk_mask, Z, U, stop_reason, stop_iter = self.reload_data(
                chunk_slice)
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


    def data_path(self, calc_name=None):
        if calc_name is None:
            calc_name = self.calc_name
        keys = ["chunk_mask"] + self.SAVE_ARRS 
        def file_map(key):
            return os.path.join(self.directory, "data",
                                calc_name + "_" + key + ".arr")
        return dict(zip(keys, map(file_map, keys)))


    def init_data_mmaps(self):
        """
        Creates the memory mappings for calculated arrays
        [chunk_mask, Z, U, stop_reason, stop_iter]
        
        Note : chunk_mask can be initialized here
        """

        keys = self.SAVE_ARRS
        data_type = {
            "Z": self.complex_type,
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
                version=None
            )
            for rank, chunk_slice in enumerate(self.chunk_slices()):
                beg, end = self.mask_beg_end(rank)
                mmap[beg:end] = self.chunk_mask[chunk_slice]


    def get_data_memmap(self, key, calc_name=None, mode='r'):
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
        if calc_name is None:
            calc_name = self.calc_name
        return "__data_mmap_" + calc_name + "_" + key

    def close_data_mmaps(self):
        to_del = tuple()
        for item in filter(
                lambda x: x.startswith("__data_mmap_"),
                vars(self)
        ):  
            to_del += (item,)
        for item in to_del:
            # getattr(self, item).flush()
            delattr(self, item) # self._temporary_mmap


    def update_data_mmaps(self, chunk_slice, Z, U, stop_reason, stop_iter):
        keys = self.SAVE_ARRS
        arr_map = {
            "Z": Z,
            "U": U,
            "stop_reason": stop_reason,
            "stop_iter": stop_iter,
        }
        rank = self.chunk_rank(chunk_slice)
        beg, end = self.arr_beg_end(rank)

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
            mmap = self.get_data_memmap(key, mode="r+")
            arr = arr_map[key]
            for (field, f_field) in zip(*codes_index_map[key]):
                mmap[field, beg:end] = arr[f_field, :]

    def reload_data(self, chunk_slice, calc_name=None): # public
        """ Reload all stored raw arrays for this chunk : 
        raw_data = chunk_mask, Z, U, stop_reason, stop_iter
        """
        is_current_calc = False
        if calc_name is None:
            is_current_calc = True
            calc_name = self.calc_name

        keys = self.SAVE_ARRS
        rank = self.chunk_rank(chunk_slice)
        beg, end = self.arr_beg_end(rank, calc_name)

        arr = dict()
        # data_path = self.data_path(calc_name)
        for key in keys:
            mmap = self.get_data_memmap(key, calc_name)
            arr[key] = mmap[:, beg:end]

        subset = self.subset
        # Here we can t always rely on self.subset, it has to be consistent
        # with calc_name ie loaded from params options
        # TODO: seems not very reliable as subset is currently not Serialisable
        if not(is_current_calc):
            params, _ = self.reload_params(calc_name)
            subset = params["calc-param_subset"]

        if subset is not None:
            # /!\ fixed-size irrespective of the mask
            beg, end = self.mask_beg_end(rank)
            mmap = self.get_data_memmap("chunk_mask", calc_name)
            arr["chunk_mask"] = mmap[beg: end]
        else:
            arr["chunk_mask"] = None

        return (
            arr["chunk_mask"], arr["Z"], arr["U"], arr["stop_reason"],
            arr["stop_iter"]
        )

    def evaluate_data(self, chunk_slice, calc_name, postproc_options): # public
        """ Compute on the fly the raw arrays for this chunk : 
        raw_data = chunk_mask, Z, U, stop_reason, stop_iter
        Note: this is normally a final redering
        """
        # The subset is the most tricky...
        subset = self._lazzy_data[calc_name]["params"]["calc-param_subset"]
        if subset is not None:
            raise NotImplementedError(
                "'Final' high quality renderering mode not implemented for "
                "calculations with a subset defined."
                "Please run in standard mode."
            )

        cycle_dep_args = self.get_cycling_dep_args(
            chunk_slice, use_current_subset=False, subset=subset
        )
        cycle_indep_args = self._lazzy_data[calc_name]["cycle_indep_args"]

        ret_code = self.numba_cycle_call(cycle_dep_args, cycle_indep_args)

        if ret_code == self.USER_INTERRUPTED:
            logger.error("Interruption signal received")
            return

        # Just need to wrap it up
        (c_pix, Z, U, stop_reason, stop_iter) = cycle_dep_args
        chunk_mask = None
        if subset is not None:
            chunk_mask = subset[chunk_slice]
        return chunk_mask, Z, U, stop_reason, stop_iter


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

        chunk_1d = Fractal.reshape1d(chunk_array, chunk_mask, chunk_slice)
        return np.reshape(chunk_1d, [n_post, nx, ny])
    
    @staticmethod
    def reshape1d(chunk_array, chunk_mask, chunk_slice):
        """
        Returns unmasked 1d versions of the 1d stored vecs
        """
        (ix, ixx, iy, iyy) = chunk_slice
        nx, ny = ixx - ix, iyy - iy
        n_post, n_pts = chunk_array.shape

        if chunk_mask is None:
            chunk_1d = np.copy(chunk_array)
        else:
            indices = np.arange(nx * ny)[chunk_mask]
            chunk_1d = np.empty([n_post, nx * ny], dtype=chunk_array.dtype)
            chunk_1d[:] = np.nan
            chunk_1d[:, indices] = chunk_array

        return chunk_1d
    
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

    def postproc(self, postproc_batch, chunk_slice, postproc_options):
        """ Computes the output of ``postproc_batch`` for chunk_slice
        Return
          post_array of shape(nposts, chunk_n_pts)
          chunk_mask
        """
        if postproc_batch.fractal is not self:
            raise ValueError("Postproc batch from a different factal provided")

        # Input data
        calc_name = postproc_batch.calc_name
        
        # Is it the final render ?
        if postproc_options["final_render"]:
            # This is the final render, we compute from scratch
            (chunk_mask, Z, U, stop_reason, stop_iter
             ) = self.evaluate_data(chunk_slice, calc_name, postproc_options)
        else:
            # Not the final -> raw data shall have been precomputed & stored
            (chunk_mask, Z, U, stop_reason, stop_iter
             ) = self.reload_data(chunk_slice, calc_name)

        params, codes = self.reload_params(calc_name)
        complex_dic, int_dic, termination_dic = self.codes_mapping(*codes)
        postproc_batch.set_chunk_data(chunk_slice, chunk_mask, Z, U,
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

        return post_array, chunk_mask

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
