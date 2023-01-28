# -*- coding: utf-8 -*-
import copy
import os
import logging

import numpy as np
import PIL
from numpy.lib.format import open_memmap

import fractalshades as fs
import fractalshades.utils
from fractalshades.lib.fast_interp import interp2d
import fractalshades.numpy_utils.filters as fsfilters
from fractalshades.mthreading import Multithreading_iterator

logger = logging.getLogger(__name__)

SCIPY_INTERPOLANT = False
import scipy


# TODO: a frame could handle 2x supersampling + Lanczos filter ?
class Frame:
    def __init__(self, x, y, dx, nx, xy_ratio, supersampling):
        """
    A frame is used to describe a specific data window to extract / interpolate
    from a db
    
    Parameters:
    -----------
    x: float
        center x-coord of the window (in screen coordinates)
    y: float
        center y-coord of the window (in screen coordinates)
    dx: float
        width of the window (in screen coordinates)
    nx: int
        number of pixel for the interpolated frame
    xy_ratio: float
        width / height ratio of the interpolated frame 
    supersampling: bool
        If True, uses a 2x2 supersampling
    
    Note:
    -----
    For a simple pass-through db -> Frame:

        Frame(
            x=0., y=0., dx=1.0,
            nx=db.zoom_kwargs["nx"],
            xy_ratio=db.zoom_kwargs["xy_ratio"]
        )
    """
        self.x = x
        self.y = y
        self.dx = dx
        self.nx = nx
        self.xy_ratio = xy_ratio
        self.supersampling = supersampling
        
        self.ny = int(self.nx / self.xy_ratio + 0.5)
        self.dy = dy = self.dx / self.xy_ratio
        self.size = (self.nx, self.ny)

        xmin = x - 0.5 * dx
        xmax = x + 0.5 * dx
        
        ymin = y - 0.5 * dy
        ymax = y + 0.5 * dy
        
        if supersampling:
            xvec = np.linspace(xmin, xmax, 2 * self.nx)
            yvec = np.linspace(ymin, ymax, 2 * self.ny)
        else:
            xvec = np.linspace(xmin, xmax, self.nx)
            yvec = np.linspace(ymin, ymax, self.ny)

        y_grid, x_grid = np.meshgrid(yvec, xvec)
        self.pts = np.ravel(x_grid), np.ravel(y_grid)


class Plot_template:
    def __init__(self, plotter, db_loader, frame):
        """ A plotter with a pluggable get_2d_arr method
        -> Basically, an interface to avoid monkey-patching `get_2d_arr`
        
        Parameters
        ----------
        plotter: `fractalshades.Fractal_plotter`
            The wrapped plotter - will be copied
        db_loader: `fractalshades.db.Db_loader`
            A db data-loading class - implementing `get_2d_arr`
        frame: Frame
            The frame-specific data holder
        """
        # Internal plotting object, reparented for frame-specific
        # functionnality
        self.plotter = copy.deepcopy(plotter)
        for layer in self.plotter.layers:
            layer.link_plotter(self)

        self.db_loader = db_loader
        self.frame = frame

    def __getattr__(self, attr):
        return getattr(self.plotter, attr)

    @property
    def supersampling(self):
        """ supersampling is already done at db level so not relevant anymore
        """
        return None

    def get_2d_arr(self, post_index, chunk_slice):
        """ Forwards to the db-loader for frame-specific functionnality """
        return self.db_loader.get_2d_arr(
            post_index, self.frame, chunk_slice
        )


class Db:
    def __init__(self, path):
        """ Wrapper around the raw array data stored at ``path``.

        The array is of shape (nposts, nx, ny) where nposts is the number of 
        post-processing fields, and is usually stored  by a
        ``fractalshades.Fractal_plotter.save_db`` call.
        datatype might be np.float32 or np.float64

        Parameters
        ----------
        path: str
            The path for the raw data
        """
        self.path = path
        # self.zoom_kw = zoom_kw
        self.init_model()
        
        # Cache for interpolating classes
        self._interpolator = {}

        # Frozon db properties
        self.is_frozen = False


    def init_model(self):
        """ Build a description for the datapoints in the mmap """
        mmap = open_memmap(filename=self.path, mode="r+")
        nposts, nx, ny = mmap.shape
        del mmap

        # Points number
        self.nx = nx #  = self.zoom_kw["nx"]
        self.ny = ny # = int(nx / xy_ratio + 0.5)
        self.xy_ratio = xy_ratio = nx / ny

        # Points loc - in screen coordinates
        self.x0 = x0 = 0.
        self.y0 = y0 = 0.
        self.dx0 = dx0 = 1.0
        self.dy0 = dy0 = 1.0 / xy_ratio

        # xy grid
        self.xmin0 = xmin0 = x0 - 0.5 * dx0
        self.xmax0 = xmax0 = x0 + 0.5 * dx0
        self.xgrid0, self.xh0 = np.linspace(
            xmin0, xmax0, nx, endpoint=True, retstep=True
        )
        self.ymin0 = ymin0 = y0 - 0.5 * dy0
        self.ymax0 = ymax0 = y0 + 0.5 * dy0
        self.ygrid0, self.yh0 = np.linspace(
            ymin0, ymax0, ny,  endpoint=True, retstep=True
        )


    def get_interpolator(self, frame, post_index):
        """ Try first to reload if the interpolating domain is still valid
        If not, creates a new interpolator """
        if post_index in self._interpolator.keys():
            (interpolator, bounds) = self._interpolator[post_index]
            x = frame.x
            y = frame.y
            ddx = 0.5 * frame.dx
            ddy = 0.5 * frame.dy
            a, b = bounds
            valid = (
                (a[0] <= x - ddx) and (b[0] >= x + ddx) 
                and (a[1] <= y - ddy) and (b[1] >= y + ddy)
            )
            if valid:
                return interpolator

        logger.debug("New Interpolator needed")
        interpolator, bounds =  self.make_interpolator(frame, post_index)
        self._interpolator[post_index] = (interpolator, bounds)
        return interpolator


    def make_interpolator(self, frame, post_index):
        """
        Returns an interpolator for the field post_index and  pts inside
        the domain [x-dx, x+dx] x [y-dy, y+dy]
        in Screeen coordinates
        """
        x = frame.x
        y = frame.y
        dx = frame.dx
        dy = frame.dy

        # 1) load the local interpolation array for the region of interest
        # with a margin so it has a chance to remain valid for several frames
        k = 0.5 * 1.5  # 0.5 if no margin at all
        x_min = max(x - k * dx, self.xmin0)
        ind_xmin = np.searchsorted(self.xgrid0, x_min, side="right") - 1
        y_min = max(y - k * dy, self.ymin0)
        ind_ymin = np.searchsorted(self.ygrid0, y_min, side="right") - 1
        # max
        x_max = min(x + k * dx, self.xmax0)
        ind_xmax = np.searchsorted(self.xgrid0, x_max, side="left")
        y_max = min(y + k * dy, self.ymax0)
        ind_ymax = np.searchsorted(self.ygrid0, y_max, side="left")


        # 2) Creates and return the interpolator
        # a, b: the lower and upper bounds of the interpolation region
        # h:    the grid-spacing at which f is given
        # f:    data to be interpolated
        # k:    order of local taylor expansions (int, 1, 3, or 5)
        # p:    whether the dimension is taken to be periodic
        # c:    whether the array should be padded to allow accurate close eval
        # e:    extrapolation distance, how far to allow extrap, in units of h
        #       (needs to be an integer)
        a = [self.xgrid0[ind_xmin], self.ygrid0[ind_ymin]]
        b = [self.xgrid0[ind_xmax], self.ygrid0[ind_ymax]]
        h = [self.xh0, self.yh0]
        
        if self.is_frozen:
            fr_mmap = open_memmap(filename=self.frozen_path, mode="r")
            f = fr_mmap[ind_xmin:ind_xmax, ind_ymin:ind_ymax, post_index]
            del fr_mmap

        else:
            mmap = open_memmap(filename=self.path, mode="r")
            f = mmap[post_index, ind_xmin:ind_xmax, ind_ymin:ind_ymax]
            del mmap

        k = 1
        p = [False, False]
        c = [False, False]
        e = [0, 0]
        interpolator = interp2d(a, b, h, f, k, p, c, e)

        bounds = (a, b)
        return interpolator, bounds
    
# --------------- db freezing interface ---------------------------------------
    @property
    def frozen_path(self):
        """ path to the 'froozen' image database """
        root, ext = os.path.splitext(self.path)
        return root + ".frozen" + ext

    def freeze(self, plotter, layer_name, try_reload):
        """
        Freeze a database by storing the postprocessed layer image as a numpy
        array. The layer shall be a RGB layer.

        Parameters:
        -----------
        plotter: fs.Fractal_plotter
            A plotter to be used
        layer_name: str
            The layer name - shall be a RGB layer
        try_reload: bool
            if True, will try to reload before computing a new one
        """
        logger.info(
            f"Freezing db to {self.frozen_path} - try_reload: {try_reload}"
        )

        # Create a mmap for the layer
        layer = plotter[layer_name]
        mode = layer.mode
        if not(mode in "RGB", "rgba"):
            raise ValueError(
                f"Only a RGB(A) layer can be used to freeze a db"
                + f"found: {mode}"
            )
        dtype = fs.colors.layers.Virtual_layer.DTYPE_FROM_MODE[mode]
        n_channel = fs.colors.layers.Virtual_layer.N_CHANNEL_FROM_MODE[mode]

        mmap = open_memmap(filename=self.path, mode="r+")
        nposts, nx, ny = mmap.shape
        del mmap
        
        self.frozen_props = {
            "dtype": np.dtype(dtype),
            "n_channel": n_channel,
            "mode": mode
        }

        if try_reload:
            # Does the mmap already exists, does it seems to suit our need ?
            try:
                try_mmap = open_memmap(filename=self.frozen_path, mode='r')
                try_nx, try_ny, try_channel = try_mmap.shape
                valid = (
                    (nx == try_nx) and (ny == try_ny)
                    and (n_channel == try_channel) 
                )
                if not(valid):
                    raise ValueError("Invalid db")
                logger.info(
                    f"Reloading successful, using {self.frozen_path}"
                )
                self.is_frozen = True
                return
            except (ValueError, FileNotFoundError):
                logger.info(
                    f"Reloading failed, computing {self.frozen_path} from scratch"
                )

        # Create a new file...
        fr_mmap = open_memmap(
            filename=self.frozen_path, 
            mode='w+',
            dtype=np.dtype(dtype),
            shape=(nx, ny, n_channel),
            fortran_order=False,
            version=None
        )                
        del fr_mmap

        plot_template = Plot_template(plotter, self, frame=None)
        # Reparenting the layers to _Plot_template
        for layer in plot_template.layers:
            layer.link_plotter(plot_template)

        self.process_for_freeze(plot_template, out_postname=layer_name)
        self.is_frozen = True


    @Multithreading_iterator(
        iterable_attr="db_chunks", iter_kwargs="db_chunk"
    )
    def process_for_freeze(self, plot_template, out_postname, db_chunk=None):
        """ Freeze (stores) the RGB array for this layer"""

        (ix, ixx, iy, iyy) = db_chunk
        mmap = open_memmap(filename=self.path, mode='r+')
        fr_mmap = open_memmap(filename=self.frozen_path, mode='r+')

        pasted = False
        for i, layer in enumerate(plot_template.layers):
            if layer.postname == out_postname:
                if not(layer.output):
                    raise ValueError("No output for this layer!!")
                # PILLOW -> Numpy
                paste_crop_arr = np.swapaxes(
                    np.asarray(layer.crop(db_chunk)), 0 , 1
                )[:, ::-1]
                fr_mmap[ix: ixx, iy: iyy, :] = paste_crop_arr
                pasted = True

        if not pasted:
            raise ValueError(
                f"Layer missing: {out_postname} "
                + f"not found in {plot_template.postnames}"
            )

        del mmap
        del fr_mmap


    def get_2d_arr(self, post_index, frame, chunk_slice):
        """ get_2d_arr implementation direct db reloading

        Parameters:
        -----------
        post_index: int
            the index for this post-processing field in self.plotter
        frame:
            Not used - uses chunk_slice instead
        chunk_slice: 4-uplet float
            chunk_slice = (ix, ixx, iy, iyy) is the sub-array to reload
        """
        (ix, ixx, iy, iyy) = chunk_slice
        mmap = open_memmap(filename=self.path, mode="r")
        ret = mmap[post_index, ix:ixx, iy:iyy]
        del mmap
        return ret   


    def db_chunks(self): #, chunk_size=None):
        """
        Generator function
        Yields the chunks spans (ix, ixx, iy, iyy)
        with each chunk of size chunk_size x chunk_size
        """
        # if chunk_size is None:
        chunk_size = fs.settings.db_chunk_size

        for ix in range(0, self.nx, chunk_size):
            ixx = min(ix + chunk_size, self.nx)
            for iy in range(0, self.ny, chunk_size):
                iyy = min(iy + chunk_size, self.ny)
                yield  (ix, ixx, iy, iyy)


class Db_loader:
    """
    A basic loading / plotting class for a stored db, with interpolation 
    capacity.

    Note : “full HD” is 1920 x 1080 pixels.
    
    """
    def __init__(self, plotter, db, plot_dir):
        """
        Parameters:
        -----------
        plotter: fs.Fractal_plotter
            A plotter to be used as template
        db: fs.db.Db
            The stored db, in cartesian coordinates
        plot_dir: str
            The directory for the plots
        """
        assert isinstance(plotter, fs.Fractal_plotter)

        self.plotter = plotter
        self.db = db
        self.plot_dir = plot_dir
        
        self.lf2 = fsfilters.Lanczos_decimator().get_impl(2, 2)


    def get_2d_arr(self, post_index, frame, chunk_slice):
        """ get_2d_arr with frame-specific functionnality

        Parameters:
        -----------
        post_index: int
            the index for this post-processing field in self.plotter
        frame: Frame
            Frame localisation for interpolation
        chunk_slice: Not used
            Not used - uses frame instead
        """
#        mmap = open_memmap(filename=self.db.path, mode="r+")
        ret = self.db.get_interpolator(frame, post_index)(*frame.pts)
#        del mmap

        if frame.supersampling:
            sps_size = (2 * frame.nx, 2 * frame.ny)
            return self.lf2(ret.reshape(sps_size))
        else:
            return ret.reshape(frame.size)


    def plot(self, frame, out_postname=None):
        """
        Parameters:
        -----------
        frame_kwargs: "raw", (pts_x, pts_y)
            Frame-specific functionnality
        out_mode: "file", "data"
            If file, will create a png file
            If "data", will return the PIL.Image object for further processing
            (movie making tool, ...)
        out_postname: Optional str
            If not None, only the PIL.Image object from this layer will be 
            returned - Has no effect if out_mode is "file".

        Return:
        -------
        img
            PIL.Image object
        """
        # Is the db frozen ?
        if self.db.is_frozen:
            return self.plot_frozen(frame)

        plot_template = Plot_template(self.plotter, self, frame)
        # Reparenting the layers to _Plot_template
        for layer in plot_template.layers:
            layer.link_plotter(plot_template)

#        if out_mode == "file":
#            out_postname = None
        imgs = []
        im_layers = []

        if out_postname is None:
            # We store all layers images
            for layer in plot_template.layers:
                if layer.output:
                    imgs.append(
                        PIL.Image.new(mode=layer.mode, size=frame.size)
                    )
                    im_layers.append(layer)
                else:
                    imgs.append(None)
                    im_layers.append(None)
        else:
            # We store only the relevant images
            for i, layer in enumerate(plot_template.layers):
                if layer.postname == out_postname:
                    if not(layer.output):
                        raise ValueError("No output for this layer!!")
                    imgs.append(
                        PIL.Image.new(mode=layer.mode, size=frame.size)
                    )
                    im_layers.append(layer)
                    break
            if len(imgs) == 0:
                raise ValueError(
                    f"Layer missing: {out_postname} "
                    + f"not found in {plot_template.postnames}"
                )
        self.process(plot_template, frame, imgs, im_layers)
        return self.output(plot_template, imgs, out_postname)


    def plot_frozen(self, frame):
        """ Direct interpolation in a frozen db image data
        """
        dtype = self.db.frozen_props["dtype"]
        n_channel = self.db.frozen_props["n_channel"]
#        mode = self.db.frozen_props["mode"]

        nx, ny = frame.size
        ret = np.empty((ny, nx, n_channel), dtype=dtype)
        # print("dtype", dtype, "shape", nx, ny)

        for ic in range(n_channel): #3): #n_channel):
            channel_ret = self.db.get_interpolator(frame, ic)(*frame.pts)
            if frame.supersampling:
                sps_size = (2 * frame.nx, 2 * frame.ny)
                channel_ret = self.lf2(channel_ret.reshape(sps_size))
            else:
                channel_ret = channel_ret.reshape(frame.size)
                # print("channel_ret", channel_ret.dtype, channel_ret.shape)
            # Numpy -> PILLOW
            ret[:, :, ic] = np.swapaxes(channel_ret, 0 , 1)[::-1, :]

        im = PIL.Image.fromarray(ret)
        return im
#        chunk_slice = (0, nx, 0, ny)
#        crop_slice = (0, 0, nx, ny)

#        if out_mode == "file":

#        if out_mode == "data":
#            return im
#        else:
#            raise NotImplementedError(out_mode)


    def process(self, plot_template, frame, imgs, im_layers):
        """
        Just plot the Images interpolated data + plot_template
        1 db point -> 1 pixel
        """
        nx, ny = frame.size
        chunk_slice = (0, nx, 0, ny)
        crop_slice = (0, 0, nx, ny)
        for i, im in enumerate(imgs):
            paste_crop = im_layers[i].crop(chunk_slice)
            imgs[i].paste(paste_crop, box=crop_slice)


    def output(self, plot_template, imgs, out_postname):
        """ Saves as file or return the img array for further processing
        """
#        if out_mode == "file":
#            for i, layer in enumerate(plot_template.layers):
#                if not(layer.output):
#                    continue
#                file_name = plot_template.image_name(layer)
#                base_img_path = os.path.join(self.plot_dir, file_name + ".png")
#                fs.utils.mkdir_p(os.path.dirname(base_img_path))
#                imgs[i].save(base_img_path)

#        elif out_mode == "data":
        if out_postname is None:
            out = {}
            # Should just return the layer as dict mapping
            # layer_name -> PIL images
            for i, layer in enumerate(plot_template.layers):
                if layer.output:
                    out[layer.postname] = imgs[i]
        else:
            return imgs[0]

#        else:
#            raise NotImplementedError(out_mode)




class Exp_db:
    """ Database for an expmap plot """
    pass


class Exp_db_loader(Db_loader):

        def __init__(self, db_exp, db_final):
            """
            Note : “full HD” is 1920 x 1080 pixels. This means the vertical
            dim of db_exp is ~ 6032  and the width potentially much longer.
            """
            self.db_exp = db_exp
            self.db_final = db_final
            # We create cascading db with at least 3 resolutions 
            # with a 2**n-decimating Lanczos filter

        def plot_interp(self, x_grid, y_grid):
            pass
    
    

            
            

            
            
            
            
            


