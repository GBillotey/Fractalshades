# -*- coding: utf-8 -*-
import copy
import os
import logging
#import queue

import numpy as np
import numba
import PIL
from numpy.lib.format import open_memmap

import fractalshades as fs
import fractalshades.utils
# from fractalshades.lib.fast_interp import interp2d
from fractalshades.numpy_utils.interp2d import (
    Grid_lin_interpolator as fsGrid_lin_interpolator
)

import fractalshades.numpy_utils.filters as fsfilters
from fractalshades.mthreading import Multithreading_iterator

logger = logging.getLogger(__name__)


class Frame:
    def __init__(self, x, y, dx, nx, xy_ratio, supersampling=False,
                 t=None, plotting_modifier=None):
        """
    A frame is used to describe a specific data window and used to interpolate
    inside a db.

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
    supersampling: Optionnal bool
        If True, uses a 2x2 supersampling + Lanczos-2 decimation filter
    t: Optionnal float
        time [s] of this frame in the movie
    plotting_modifier: Optionnal callable
        a plotting_modifier associated with this Frame

    Note:
    -----
    A simple pass-through Frame, extracting the raw ``my_db`` data is:

        fs.db.Frame(
            x=0., y=0., dx=1.0,
            nx=my_db.zoom_kwargs["nx"],
            xy_ratio=my_db.zoom_kwargs["xy_ratio"]
        )
    """
        self.x = x
        self.y = y
        self.dx = dx
        self.nx = nx
        self.xy_ratio = xy_ratio
        self.supersampling = supersampling
        self.t = t
        self.plotting_modifier = plotting_modifier
        
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
        # Internal plotterobject, layers reparented for frame-specific
        # functionnality
        self.plotter = copy.deepcopy(plotter)
        for layer in self.plotter.layers:
            layer.link_plotter(self)

        self.db_loader = db_loader
        self.frame = frame

    def __getattr__(self, attr):
        return getattr(self.plotter, attr)

    def __getitem__(self, layer_name):
        """ Get the layer by its postname
        """
        return self.plotter.__getitem__(layer_name)

    @property
    def supersampling(self):
        """ supersampling has to be done at db saving level so not relevant 
        in post-processing
        """
        return None

    def get_2d_arr(self, post_index, chunk_slice):
        """ Forwards to the db-loader for frame-specific functionnality """
        return self.db_loader.get_2d_arr(
            post_index, self.frame, chunk_slice
        )


class Db:

    def __init__(self, path):
        """ Wrapper around the raw numpy-array stored at ``path``.

        The array is of shape (nposts, nx, ny) where nposts is the number of 
        post-processing fields, and is usually created through a
        ``fractalshades.Fractal_plotter.save_db`` call.

        Note: datatype might be ``np.float32`` or ``np.float64`` 

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

    @property
    def zoom_kwargs(self):
        return self.plotter.fractal.zoom_kwargs

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

        interpolator, bounds =  self.make_interpolator(frame, post_index)
        self._interpolator[post_index] = (interpolator, bounds)
        logger.debug(f"New Interpolator added for field #{post_index}")

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
        interpolator = fsGrid_lin_interpolator(a, b, h, f)
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
        Freeze a database by storing a postprocessed layer image as a numpy
        array (1 data point = 1 pixel). The layer shall be a RGB(A) layer.

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
        if not(mode in "RGB", "RGBA"):
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
                p = self.frozen_path
                logger.info(
                    f"Reloading failed, computing {p} from scratch"
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
        self.process_for_freeze(plot_template, out_postname=layer_name)
        self.is_frozen = True


    @Multithreading_iterator(
        iterable_attr="db_chunks", iter_kwargs="db_chunk"
    )
    def process_for_freeze(self, plot_template, out_postname, db_chunk=None):
        """ Freeze (stores) the RGB array for this layer"""
        (ix, ixx, iy, iyy) = db_chunk
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
                # fr_mmap[ix: ixx, iy: iyy, :] = paste_crop_arr

                if fr_mmap.shape[2] == 1:
                    fr_mmap[ix: ixx, iy: iyy, 0] = paste_crop_arr
                else:
                    fr_mmap[ix: ixx, iy: iyy, :] = paste_crop_arr

                pasted = True

        if not pasted:
            raise ValueError(
                f"Layer missing: {out_postname} "
                + f"not found in {plot_template.postnames}"
            )

        del fr_mmap


    def db_chunks(self):
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

# --------------- db plotting interface ---------------------------------------

    def set_plotter(self, plotter, postname,
                    plotting_modifier=None, reload_frozen=False):
        """
        Define the plotting properties

        Parameters:
        -----------
        plotter: fs.Fractal_plotter
            A plotter to be used as template
        postname: str
            The string indentifier of the layer used for plotting
        plotting_modifier: Optionnal, callable(plotter, time)
            A callback which will modify the plotter instance before each time
            step. Defaults to None, which allows to 'freeze' in place the
            database postprocessad image and interpolate directly in the image.
            Using this option open a lot of possibilities but is also much
            more computer-intensive
        reload_frozen: Optionnal, bool
            Used only if plotting_modifier is None
            If True, will try to reload any previously computed frozen db image
        """
        assert isinstance(plotter, fs.Fractal_plotter)

        self.plotter = plotter
        self.postname = postname
        self.plotting_modifier = plotting_modifier
        self.lf2 = fsfilters.Lanczos_decimator().get_impl(2, 2)

        if plotting_modifier is None:
            # we can freeze the db and interpolate in the frozen image
            logger.info("Database will be frozen for camera move")
            self.freeze(plotter, postname, try_reload=reload_frozen)


    def get_2d_arr(self, post_index, frame, chunk_slice):
        """ get_2d_arr with frame-specific functionnality

        Parameters:
        -----------
        post_index: int
            the index for this post-processing field in self.plotter
        frame: fs.db.Frame
            Frame localisation for interpolation
        chunk_slice: 4-uplet float
            chunk_slice = (ix, ixx, iy, iyy) is the sub-array to reload
            Used only if `frame` is None: direct reloading
        """
        if frame is None:
            # Used internally when computing the interpolation arrays
            # Direct output : uses chunk_slice
            (ix, ixx, iy, iyy) = chunk_slice
            mmap = open_memmap(filename=self.path, mode="r")
            ret = mmap[post_index, ix:ixx, iy:iyy]
            del mmap
            return ret   
            
        else:
            # Interpolated output : uses frame
            ret = self.get_interpolator(frame, post_index)(*frame.pts)
            if frame.supersampling:
                sps_size = (2 * frame.nx, 2 * frame.ny)
                return self.lf2(ret.reshape(sps_size))
            else:
                return ret.reshape(frame.size)


    def plot(self, frame=None):
        """
        Parameters:
        -----------
        frame: fs.db.Frame, Optional
            Defines the area to plot. If not provided, the full db will be
            plotted

        Return:
        -------
        img: PIL.Image
            The plotted image

        Notes:
        ------
        Plotting settings as defined by set_plotter method
        """
        if frame is None:
            # Default to plotting the whole db
            nx = self.plotter.fractal.zoom_kwargs["nx"]
            xy_ratio = self.plotter.fractal.zoom_kwargs["xy_ratio"]
            frame = fs.db.Frame(
                x=0., y=0., dx=1.0, nx=nx, xy_ratio=xy_ratio
            )
        
        # Is the db frozen ?
        if self.is_frozen:
            return self.plot_frozen(frame)

        plot_template = Plot_template(self.plotter, self, frame)

        plotting_modifier = frame.plotting_modifier
        if plotting_modifier is not None:
            plotting_modifier(plot_template, frame.t)


        img = None
        out_postname = self.postname

        for i, layer in enumerate(plot_template.layers):
            if layer.postname == out_postname:
                if not(layer.output):
                    raise ValueError("No output for this layer!!")
                img = PIL.Image.new(mode=layer.mode, size=frame.size)
                im_layer = layer
                break

        if img is None:
            raise ValueError(
                f"Layer missing: {out_postname} "
                + f"not found in {plot_template.postnames}"
            )

        self.process(plot_template, frame, img, im_layer)
        return img


    def plot_frozen(self, frame):
        """ Direct interpolation in a frozen db image data
        """
        dtype = self.frozen_props["dtype"]
        n_channel = self.frozen_props["n_channel"]
        nx, ny = frame.size
        if frame.supersampling:
            im_size = (2 * nx, 2 * ny)
        else:
            im_size = (nx, ny)
        ret = np.empty(im_size[::-1] + (n_channel,), dtype=dtype)

        for ic in range(n_channel): #3): #n_channel):
            channel_ret = self.get_interpolator(frame, ic)(*frame.pts
            ).reshape(im_size)
            # Numpy -> PILLOW
            ret[:, :, ic] = np.swapaxes(channel_ret, 0 , 1)[::-1, :]
            
            # Image.fromarray((ret * 255).astype(np.uint8))
        
        if n_channel == 1:
            im = PIL.Image.fromarray(ret[:, :, 0])
        else:
            im = PIL.Image.fromarray(ret)

        if frame.supersampling:
            resample = PIL.Image.LANCZOS
            im = im.resize(size=(nx, ny), resample=resample) 

        return im


    def process(self, plot_template, frame, img, im_layer):
        """
        Just plot the Images interpolated data + plot_template
        1 db point -> 1 pixel
        """
        nx, ny = frame.size
        chunk_slice = (0, nx, 0, ny)
        crop_slice = (0, 0, nx, ny)
        paste_crop = im_layer.crop(chunk_slice)
        img.paste(paste_crop, box=crop_slice)













#==============================================================================
#==============================================================================
# Exponential mapping to cartesian frame transform

class Exp_frame:
    def __init__(self, h, nx, xy_ratio, supersampling=False,
                 t=None, plotting_modifier=None, pts=None):
        """
    A Exp_frame is used to describe a specific data window and used to
    interpolate inside a Expmap db.

    Parameters:
    -----------
    h: float >= 0.
        zoom level. A zoom level of 0. denotes that dx = dx0 fully zoomed-in
        frame - for h > 0, dx = dx0 * np.exp(h) 
    nx: int
        number of pixels for the interpolated frame
    xy_ratio: float
        width / height ratio of the interpolated frame 
    supersampling: Optional bool
        If True, uses a 2x2 supersampling + Lanczos-2 decimation filter
    t: Optionnal float
        time [s] of this frame in the movie
    plotting_modifier: Optional callable
        a plotting_modifier associated with this Frame
    pts: Optional, 4-uplet of arrays
        The x, y, h, t grid as returned by make_exp_grid - if not provied it
        will be recomputed
    """
        self.h = h
        self.nx = nx
        self.xy_ratio = xy_ratio
        self.supersampling = supersampling
        self.t = t
        self.plotting_modifier = plotting_modifier
        
        self.ny = int(self.nx / self.xy_ratio + 0.5)
        self.size = (self.nx, self.ny)

#        xmin = -0.5
#        xmax = +0.5
#        ymin = -0.5 / self.xy_ratio
#        ymax = +0.5 / self.xy_ratio
#
#        if supersampling:
#            xvec = np.linspace(xmin, xmax, 2 * self.nx)
#            yvec = np.linspace(ymin, ymax, 2 * self.ny)
#        else:
#            xvec = np.linspace(xmin, xmax, self.nx)
#            yvec = np.linspace(ymin, ymax, self.ny)
#
#        y_grid, x_grid = np.meshgrid(yvec, xvec)
        # Note that self.real_pts is in fact the unevaluated product:
        # self.pts x * np.exp(self.h) 
        if pts is None:
            pts = self.make_exp_grid(
                self.nx, self.xy_ratio, self.supersampling
            )
        # Basic shape verification
        k_ss = 2 if supersampling else 1
        assert pts[0].size == (self.nx * self.ny * k_ss ** 2)

        self.pts = pts
    
    @staticmethod
    def make_exp_grid(nx, xy_ratio, supersampling):
        """ Return a base grid [-0.5, 0.5] x [-0.5/xy_ratio, 0.5/xy_ratio] in 
        both cartesian and expoential coordiantes """
        ny = int(nx / xy_ratio + 0.5)

        xmin = -0.5
        xmax = +0.5
        ymin = -0.5 / xy_ratio
        ymax = +0.5 / xy_ratio

        # Cartesian grid
        k_ss = 2 if supersampling else 1
        xvec = np.linspace(xmin, xmax, k_ss * nx)
        yvec = np.linspace(ymin, ymax, k_ss * ny)

        y_grid, x_grid = np.meshgrid(yvec, xvec)
        x_grid = x_grid.reshape(-1)
        y_grid = y_grid.reshape(-1)

        # Exponential grid coordinates
        tenth_pixw = 0.1 / nx
        h_grid = 0.5 * np.log(
            np.maximum(x_grid ** 2 + y_grid ** 2, tenth_pixw)
        )
        t_grid = np.arctan2(y_grid, x_grid)

        # Store the grid
        return (
            x_grid.astype(np.float32), y_grid.astype(np.float32),
            h_grid.astype(np.float32), t_grid.astype(np.float32)
        )



class Exp_db:
    """ Database for an expmap plot """
    HCHUNK = 400

    def __init__(self, path_expmap, path_final):
        """ Wrapper around the raw array data stored at ``path_expmap`` and
        ``path_final``.

        The expmap array is of shape (nposts, nh, nt) where nposts is the number
        of  post-processing fields, and is usually stored  by a
        ``fractalshades.Fractal_plotter.save_db`` called on a fractal using a
         ``Expmap`` projection

        The final array is of shape (nposts, nx, ny) where nposts is the number
        of  post-processing fields, and is usually stored  by a
        ``fractalshades.Fractal_plotter.save_db`` call (with a standard
        ``Cartesian`` projection). It shall be square (nx == ny).

        datatype might be np.float32 or np.float64

        Parameters
        ----------
        path_expmap: str
            The path for the expmap raw data
            The array is of shape (nposts, nh, nt) 
        path_expmap: str
            The path for the final raw data
            The array is of shape (nposts, nx, ny) with nx = ny
        """
        self.path_expmap = path_expmap
        self.path_final = path_final

        self.init_model()

        # Cache for interpolating classes
        self._interpolator = {}

        # Frozon db properties
        self.is_frozen = False


    def init_model(self):
        """ Build a description for the datapoints in the mmap """
        mmap = open_memmap(filename=self.path_expmap, mode="r+")
        nposts, nh, nt = mmap.shape
        dtype = mmap.dtype
        del mmap

        mmap = open_memmap(filename=self.path_final, mode="r+")
        _nposts, nx, ny = mmap.shape
        _dtype = mmap.dtype
        del mmap

        if _nposts != nposts:
            raise ValueError("Incompatible final image database for Exp_db: "
                             f"`nposts` not matching: {_nposts} vs {nposts}")
        if _dtype != dtype:
            raise ValueError("Incompatible final image database for Exp_db: "
                             f"`dtype` not matching: {_dtype} vs {dtype}")
        if nx != ny:
            raise ValueError("Final image database shall be square, found: "
                             f"{nx} x {ny}")

        self.nposts = nposts
        self.dtype = dtype

        # Points number
        self.nh = nh #  = self.zoom_kw["nx"]
        self.nt = nt # = int(nx / xy_ratio + 0.5)
        self.nx = nx # = int(nx / xy_ratio + 0.5)
        self.ny = ny # = int(nx / xy_ratio + 0.5)

        # Data span
        dh0 = 2. * np.pi * nh / nt
        self.hmin0 = 0.
        self.hmax0 = self.hmin0 + dh0
        self.xmin0, self.xmax0 = -0.5, 0.5
        self.ymin0, self.ymax0 = -0.5, 0.5

        # ht grid
        self.hgrid0, self.hh0 = np.linspace(
            self.hmin0, self.hmax0, nh, endpoint=True, retstep=True
        )
        self.tgrid0, self.th0 = np.linspace(
            -np.pi, np.pi, nt, endpoint=True, retstep=True
        )

        # xy grid
        self.xgrid0, self.xh0 = np.linspace(
            self.xmin0, self.xmax0, nx, endpoint=True, retstep=True
        )
        self.ygrid0, self.yh0 = np.linspace(
            self.ymin0, self.ymax0, ny, endpoint=True, retstep=True
        )

        # Lanczos-2 2-decimation routine
        self.lf2_stable = fsfilters.Lanczos_decimator().get_stable_impl(2)


    def path(self, kind, downsampling):
        """ Path to the database

        Parameters
        ----------
        kind: "exp" | "final"
            The underlying db
        downsampling: bool
            If true, this is the path for the multi-level downsampled db
        """
        if kind == "exp":
            if not(downsampling):
                return self.path_expmap
            root, ext = os.path.splitext(self.path_expmap)
            return root + "_downsampling" + ext

        elif kind == "final":
            if not(downsampling):
                return self.path_final
            root, ext = os.path.splitext(self.path_final)
            return root + "_downsampling" + ext

        else:
            raise ValueError(f"{kind = }")


    def get_interpolator(self, frame, post_index):
        """ Try first to reload if the interpolating domain is still valid
        If not, creates a new interpolator """
        if post_index in self._interpolator.keys():
            (interpolator, bounds) = self._interpolator[post_index]
            h = frame.h
            h1, h2 = bounds  # h1 > h2 validity range of the interpolator
            valid = ((h <= h1) and (h >= h2))
            if valid:
                return interpolator

        interpolator, bounds =  self.make_interpolator(frame, post_index)
        self._interpolator[post_index] = (interpolator, bounds)
        logger.debug(f"New Interpolator added for field #{post_index}")

        return interpolator


    def make_interpolator(self, frame, post_index):
        """
        Returns an interpolator for the field post_index and  pts inside
        the domain [x0 - dx, x0 + dx] x [y0 - dy, y0 - dy] where
        dx = exp(h) * dx0
        dy = exp(h) dy0
        in Screeen coordinates of the final pic
        
        Returns
        -------
        interpolator
            the interpolator
        bounds: (h1, h2) h1 > h2
            the validity range
        """
        ic = post_index
        h = frame.h # expansion factor from final pic is exp(h)
        nx = frame.nx

        margin = 2. # Shall remain valid for this zoom range (in and out)
        h_margin = np.log(margin)
        h_decimate = np.log(2.)      # Triggers factor-2 image decimation 

        info_dic = self._subsampling_info
        dtype = (
            self.frozen_props["dtype"] if self.is_frozen 
            else self.plotter.fractal.post_dtype
        )
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define parameters for multilevel exp_map interpolation
        # a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot
        kind = "exp"
        full_shape = info_dic[(kind, "ss_shapes")]
        full_slot = info_dic[(kind, "ss_slots")]
        full_bound = info_dic[(kind, "ss_bounds")] # (start_x, end_x, start_y, end_y) 
        lvl = full_shape.shape[0]
        
        # We fill as if full range first
        a_exp = np.copy(full_bound[:, 0::2])
        b_exp = np.copy(full_bound[:, 1::2])
        h_exp = (b_exp - a_exp) / (full_shape - 1)
        f_exp_shape = np.copy(full_shape)
        f_exp_slot = np.copy(full_slot)
        
        # we extract a subrange for the theta direction
        h_index = np.copy(full_shape)
        for ilvl in range(lvl):
            delta_h = h_exp[ilvl, 0]
            arr_hmin = a_exp[ilvl, 0]
            arr_hmax = b_exp[ilvl, 0]

            # Compute the extracted indices for this level
            pix_hmin = np.clip(
                h - h_margin - h_decimate * (ilvl + 1), self.hmin0, self.hmax0
            )
            pix_hmax = np.clip(
                h + h_margin - h_decimate * ilvl, self.hmin0, self.hmax0
            )
            ind_hmin = int(np.floor((pix_hmin - arr_hmin) / delta_h))
            ind_hmax = int(np.ceil((pix_hmax - arr_hmin) / delta_h)) + 1
            
#            print("HMIN / HMAX", ilvl, pix_hmin, pix_hmax)
            
            assert ind_hmin >= 0
            assert ind_hmax <= full_shape[ilvl, 0]
            
            h_index[ilvl, :] = ind_hmin, ind_hmax
            k_min = ind_hmin / (full_shape[ilvl, 0] - 1)
            k_max = ind_hmax / (full_shape[ilvl, 0] - 1)

            # Updates tables
            a_exp[ilvl, 0] = arr_hmin * (1. - k_min) + arr_hmax * k_min
            b_exp[ilvl, 0] = arr_hmin * (1. - k_max) + arr_hmax * k_max
            f_exp_shape[ilvl, 0] = ind_hmax - ind_hmin
            f_exp_slot[ilvl, 1] = f_exp_shape[ilvl, 0] * f_exp_shape[ilvl, 1]

        f_exp_slot[:, 1] = np.cumsum(f_exp_slot[:, 1])
        f_exp_slot[1:, 0] = f_exp_slot[:-1, 1]
        f_exp_slot[0, 0] = 0
        f_exp = np.empty((f_exp_slot[-1, 1],), dtype=dtype)
        
#        print("SLOTS recap:\n", f_exp_slot)
#        print("SHAPES recap:\n", f_exp_shape)


        if self.is_frozen:
            ilvl = 0
            filename = self.frozen_path(kind, downsampling=False)
            mmap = open_memmap(filename=filename, mode="r")
            ind_hmin, ind_hmax = h_index[ilvl, :]
            loc_arr = mmap[ind_hmin:ind_hmax, :, ic]
            loc_arr = loc_arr.reshape(-1)

#            print(f"{ilvl = }", "slot", f_exp_slot[ilvl, 0], f_exp_slot[ilvl, 1], f_exp_slot[ilvl, 1] - f_exp_slot[ilvl, 0])
#            print("loc_arr", loc_arr.shape, loc_arr.size)
#            print("f_exp", f_exp.shape)
            
            f_exp[f_exp_slot[ilvl, 0]: f_exp_slot[ilvl, 1]] = loc_arr
            del mmap

            filename = self.frozen_path(kind, downsampling=True)
            mmap = open_memmap(filename=filename, mode="r")
            for ilvl in range(1, lvl):
                ind_hmin, ind_hmax = h_index[ilvl, :]
                nx, ny = full_shape[ilvl, :]
                di = full_slot[ilvl, 0]
                loc_arr = mmap[ic, (ind_hmin * ny) + di: (ind_hmax * ny) + di]
                f_exp[f_exp_slot[ilvl, 0]: f_exp_slot[ilvl, 1]] = loc_arr
            del mmap

        else:
            raise NotImplementedError("TODO")


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define parameters for multilevel exp_map interpolation
        # a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot
        kind = "final"
        full_shape = info_dic[(kind, "ss_shapes")]
        full_slot = info_dic[(kind, "ss_slots")]
        full_bound = info_dic[(kind, "ss_bounds")] # (start_x, end_x, start_y, end_y) 
        lvl = full_shape.shape[0]

        # Easy, we just fill as full range - no subrange extraction step
        a_final = np.copy(full_bound[:, 0::2])
        b_final = np.copy(full_bound[:, 1::2])
        h_final = (b_final - a_final) / (full_shape - 1)
        f_final_shape = np.copy(full_shape)
        f_final_slot = np.copy(full_slot)

        # we still adjust the slots position as first level is now merged
        for ilvl in range(lvl):
            f_final_slot[ilvl, 1] = f_final_shape[ilvl, 0] * f_final_shape[ilvl, 1]
        f_final_slot[:, 1] = np.cumsum(f_final_slot[:, 1])
        f_final_slot[1:, 0] = f_final_slot[:-1, 1]
        f_final_slot[0, 0] = 0
        f_final = np.empty((f_final_slot[-1, 1],), dtype=dtype)


        if self.is_frozen:
            ilvl = 0
            filename = self.frozen_path(kind, downsampling=False)
            mmap = open_memmap(filename=filename, mode="r")
            loc_arr = mmap[:, :, ic]
            loc_arr = loc_arr.reshape(-1)
            f_final[f_final_slot[ilvl, 0]: f_final_slot[ilvl, 1]] = loc_arr
            del mmap

            filename = self.frozen_path(kind, downsampling=True)
            mmap = open_memmap(filename=filename, mode="r")
#            print("mmap", mmap.shape) # (4, )
            for ilvl in range(1, lvl):
                filename = self.frozen_path(kind, downsampling=True)
                mmap = open_memmap(filename=filename, mode="r")
                nx, ny = full_shape[ilvl, :]
#                print("ilvl", ilvl, nx, ny, nx * ny)
#                print("->", full_slot[ilvl, 0], full_slot[ilvl, 1], full_slot[ilvl, 1] - full_slot[ilvl, 0])
                loc_arr = mmap[ic, full_slot[ilvl, 0]: full_slot[ilvl, 1]]
                f_final[f_final_slot[ilvl, 0]: f_final_slot[ilvl, 1]] = loc_arr
            del mmap

        else:
            raise NotImplementedError("TODO")

        interpolator = Multilevel_exp_interpolator(
            a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot,
            a_final, b_final, h_final, f_final, f_final_shape, f_final_slot
        )
        bounds = (h + h_margin, h - h_margin)

        return interpolator, bounds


    def debug_frozen_interpolator(self, frame, path):
        """
        Debugging - plot the image of the raw nested data used for
        interpolation of the provided frame
        """
        dtype = self.frozen_props["dtype"]
        n_channel = self.frozen_props["n_channel"]
        frame_interp = {
            ic: self.get_interpolator(frame, ic) for ic in range(n_channel)
        }

        # Sizing the output the 
        C0_interp = frame_interp[0]
        f_exp_shape = C0_interp.f_exp_shape
        f_final_shape = C0_interp.f_final_shape
        
        exp_lvl = f_exp_shape.shape[0]
        final_lvl = f_final_shape.shape[0]
        
        for ilvl in range(exp_lvl):
            arr = np.empty(
                tuple(f_exp_shape[ilvl, :]) + (n_channel,), dtype=dtype
            )
            for ic in range(n_channel):
                c_interp = frame_interp[ic]
                lw = c_interp.f_exp_slot[ilvl, 0]
                hg = c_interp.f_exp_slot[ilvl, 1]
                nx = c_interp.f_exp_shape[ilvl, 0]
                ny = c_interp.f_exp_shape[ilvl, 1]
                channel_ret = c_interp.f_exp[lw:hg].reshape((nx, ny))
                arr[:, :, ic] = channel_ret

            im = PIL.Image.fromarray(arr)
            im_path = os.path.join(path, "debug_interp", f"exp_{ilvl}.png")
            fs.utils.mkdir_p(os.path.dirname(im_path))
            im.save(im_path)

        for ilvl in range(final_lvl):
            arr = np.empty(
                tuple(f_final_shape[ilvl, :]) + (n_channel,), dtype=dtype
            )
            for ic in range(n_channel):
                c_interp = frame_interp[ic]
                lw = c_interp.f_final_slot[ilvl, 0]
                hg = c_interp.f_final_slot[ilvl, 1]
                nx = c_interp.f_final_shape[ilvl, 0]
                ny = c_interp.f_final_shape[ilvl, 1]
                channel_ret = c_interp.f_final[lw:hg].reshape((nx, ny))
                arr[:, :, ic] = channel_ret

            im = PIL.Image.fromarray(arr)
            im_path = os.path.join(path, "debug_interp", f"final_{ilvl}.png")
            fs.utils.mkdir_p(os.path.dirname(im_path))
            im.save(im_path)


# --------------- db Subsampling interface ---------------------------------------
    def subsample(self):
        if self.is_frozen:
            self.subsample_frozen()
        else:
            self.subsample_db()
    

    def subsample_frozen(self):
        
        self._subsampling_info = {}

        # 1) Downsample the frozen exp db
        source = self.frozen_path("exp", downsampling=False)
        filename = self.frozen_path("exp", downsampling=True)
        init_bound = np.array((self.hmin0, self.hmax0, -np.pi, np.pi))
        self.populate_subsampling(
            filename, source, channel_dim=2, driving_dim="y",
            init_bound=init_bound, kind="exp"
        )
    
        # 1) Downsample the frozen final db
        source = self.frozen_path("final", downsampling=False)
        filename = self.frozen_path("final", downsampling=True)
        init_bound = np.array((self.xmin0, self.xmax0, self.ymin0, self.ymax0))
        self.populate_subsampling(
            filename, source, channel_dim=2, driving_dim="x",
            init_bound=init_bound, kind="final"
        )


    def subsample_db(self):
        raise NotImplementedError("TODO")

    def ss_lvl_count(self, kind):
        """
        Return the number of subsampling data levels stored for this db.

        Parameters:
        -----------
        kind: "exp" | "final"
            The source db
        """
        if not self.is_frozen:
            raise RuntimeError("Db is not frozen")

        ss_shapes = self._subsampling_info[((kind, "ss_shapes"))]
        return ss_shapes.shape[0]


    def ss_img(self, kind, lvl):
        """
        Return a subsampled raw image (used for debuging)

        Parameters:
        -----------
        kind: "exp" | "final"
            The source db
        lvl: int >= 0
            the subsampling level
        """
        if not self.is_frozen:
            raise RuntimeError("Db is not frozen")

        nc = self.frozen_props["n_channel"] # min(self.frozen_props["n_channel"], 3) # RGBA -> RGB ? TODO
        ss_shapes = self._subsampling_info[((kind, "ss_shapes"))]
        ss_slots = self._subsampling_info[((kind, "ss_slots"))]

        nx, ny = ss_shapes[lvl, :]
        lw, hg = ss_slots[lvl, :]
        arr = np.empty((nx, ny, nc), dtype=self.frozen_props["dtype"])

        filename = self.frozen_path(kind, downsampling=(lvl != 0))
        mmap = open_memmap(filename=filename, mode="r")

        # Returns the RGB(A) array for the exp mapping for level = lvl
        for ic in range(nc):
            if lvl == 0:
                loc_arr = mmap[:, :, ic]
            else:
                loc_arr = (mmap[ic, lw:hg]).reshape((nx, ny))
            arr[:, :, ic] = loc_arr
        del mmap

        # np -> PIL
        return PIL.Image.fromarray(np.swapaxes(arr, 0 , 1)[::-1, :, :])


    def populate_subsampling(self, filename, source, channel_dim, driving_dim,
                             init_bound, kind):
        """
        Creates a memory mapping at filename and populates it with subsampled
        data from source.

        Parameters:
        -----------
        filename: str
            path for the new mmap
        source: str
            path for the source mmap
        channel_dim: 0 or 2
            The dimension of source associated with postprocs (or channel)
        driving_dim: "x", "y"
            The dimension of the image that will be reduced to 2 (criteria for 
            the number of levels)
        init_bound: np.array([minx, maxx, miny, maxy])
            The initial range for data
        kind: "exp" | "final"
            The kinf of mem mapping

        Returns:
        --------
        ss_shapes:
            shapes (nx, ny) of the nested subsampled arrays
        ss_bounds
            flatten localisation of the nested subsampled arrays
            To recover for 
        """
        logger.info(
            "Writing a subsampled db\n"
            f"   source: {source}\n"
            f"   dest: {filename}"
        )
        lf2_stable = self.lf2_stable

        # We flatten the image however the source might imply several channels
        # or layers...
        # For a db, mmap.shape: (nposts, nh, nt) or  (nposts, nx, ny)
        # For a frozen db, mmap.shape: (nx, ny, n_channel)
        source_mmap = open_memmap(filename=source, mode="r")
        dtype = source_mmap.dtype
        if channel_dim == 0:
            (nposts, nx, ny) = source_mmap.shape
        elif channel_dim == 2:
            (nx, ny, nposts) = source_mmap.shape
        else:
            raise ValueError(channel_dim)
        ss_nx = nx
        ss_ny = ny
        ss_slotl = ss_sloth = 0

        # Number of subsampling levels, based on nt
        # 2->0 3->1 4->2 5->2 6->3 7->3 8->3 9->3 10->4 [..] 17->4 18->5
        nt = {"x": nx, "y": ny}[driving_dim]
        ss_lvls = (nt - 2).bit_length()
        ss_shapes = np.empty((ss_lvls + 1, 2), dtype=np.int32)
        ss_slots = np.empty((ss_lvls + 1, 2), dtype=np.int32)

        # Note that we need to store the x AND y bounds
        ss_bounds = np.tile(init_bound, (ss_lvls + 1, 1)).astype(
            np.dtype(fs.settings.postproc_dtype)
        )

        # Sizing the subsampling arrays
        ss_shapes[0, :] = [nx, ny]
        ss_slots[0, :] = [-1, -1] # not relevant as in another mmap

        for lvl in range(ss_lvls):
            ss_nx = ss_nx // 2 + 1
            ss_ny = ss_ny // 2 + 1
            ss_slotl = ss_sloth
            ss_sloth += ss_nx * ss_ny
            ss_shapes[lvl + 1, :] = [ss_nx, ss_ny]
            ss_slots[lvl + 1, :] = [ss_slotl, ss_sloth]

        ss_mmap = open_memmap(
            filename=filename, 
            mode='w+',
            dtype=dtype,
            shape=(nposts, ss_sloth),
            fortran_order=False,
            version=None
        )

        for ipost in range(nposts):
            ss_nx = nx
            for lvl in range(ss_lvls):
                ss_dix = 200
                ss_nx = ss_nx // 2 + 1
                self.x_range = lambda: np.arange(0, ss_nx, ss_dix)
                self.parallel_populate_subsampling(
                    ss_mmap, source_mmap, channel_dim, ipost, lvl,
                    ss_shapes, ss_slots, ss_bounds, lf2_stable,
                    ss_dix, ss_ixstart=None
                )

        del source_mmap
        del ss_mmap
        
        self._subsampling_info.update({
            (kind, "ss_shapes"): ss_shapes, # Shape of the full ss array, lvl = i - 1
            (kind, "ss_slots"): ss_slots,   # 1d-slot of the full ss array, lvl = i - 1
            (kind, "ss_bounds"): ss_bounds  # bounds for the full ss array
        })


    @Multithreading_iterator(
        iterable_attr="x_range", iter_kwargs="ss_ixstart"
    )
    def parallel_populate_subsampling(self,
        mmap, source_mmap, channel_dim, ipost, lvl,
        ss_shapes, ss_slots, ss_bounds, lf2_stable,
        ss_dix, ss_ixstart=None
    ):
        """
        In parallel, apply the subsampling for (ipost, lvl).

        Parameters:
        -----------
        mmap: memory mapping for the output
        source_mmap: memory mapping for the source (used if lvl == 0)
        channel_dim: the dim used for posts / channel in source
        ipost: the current post / channel index
        lvl: current level in the nested chain
        ss_shapes: (nx, ny) of the nested subsampled array - coords in source
        ss_slots: (ssl, ssh) of the nested subsampled array - as stored,
            flatten, in res
        ss_bounds: (start_x, end_x, start_y, end_y) of the nested ss arrays
        lf2_stable: decimation routine
        ssixstart: start ix index for this parallel calc in the destination
           array /!\ not the source
        ssdix: gap in x used for parallel calc
        """
        
        ss_nx, ss_ny = ss_shapes[lvl + 1, :]   # For full "subsampled" shape
        ss_l, ss_h = ss_slots[lvl + 1, :]      # For full "subsampled" slot

        # This // run extract slot is [ixstart:ixend, :]
        ss_ixend = min(ss_ixstart + ss_dix, ss_nx)
        ss_dix = ss_ixend - ss_ixstart

        # The 2d shapes / extract slot at source array - we map (2n+1) -> n+1
        nx, ny = ss_shapes[lvl, :]
        lw, hg = ss_slots[lvl, :]  # This is the full "subsampled" slot
        ixstart = 2 * ss_ixstart
        ixend = min(ixstart + 2 * ss_dix + 1, nx)
        dix = ixend - ixstart

        if lvl == 0:
            # Source arr is from the source_mmap
            if channel_dim == 0:
                # the 'source span' shall be 2n format
                # -> mapping to n (skip the last item)
                source_arr = source_mmap[ipost, ixstart:ixend, :]
            else:
                source_arr = source_mmap[ixstart:ixend, :, ipost]
        else:
            # Source arr is from the mmap, however a level higher
            l_loc = lw + ixstart * ny
            h_loc = lw + ixend * ny
            source_arr = mmap[ipost, l_loc:h_loc].reshape((dix, ny))

        ssl_loc = ss_l + ss_ixstart * ss_ny
        ssh_loc = ss_l + ss_ixend * ss_ny
        ss2d_full, k_spanx_loc, k_spany_loc = lf2_stable(source_arr)
        ss2d_full = ss2d_full[:ss_dix, :]

        # Flatten then store in slot
        mmap[ipost, ssl_loc:ssh_loc] = ss2d_full.reshape(-1)

        if (ipost == 0) and (ss_ixstart == 0):
            # We store data localisation information. coeff applies to the 
            # following levels
            pass
            ss_bounds[(lvl + 1):, 1] +=  (k_spanx_loc - 1.) * (
                ss_bounds[(lvl + 1):, 1] - ss_bounds[(lvl + 1):, 0]
            )
            ss_bounds[(lvl + 1):, 3] +=  (k_spany_loc - 1.) * (
                ss_bounds[(lvl + 1):, 3] - ss_bounds[(lvl + 1):, 2]
            )


# --------------- db freezing interface ---------------------------------------

    def frozen_path(self, kind, downsampling):
        """ Path to the 'froozen' image database 

        Parameters
        ----------
        kind: "exp" | "final"
            The underlying db
        downsampling: bool
            If true, this is the path for the multi-level downsampled db
        """
        root, ext = os.path.splitext(self.path(kind, downsampling=False))
        if not(downsampling):
            return root + "_expmap" + ".frozen" + ext
        return root + "_expmap" + "_downsampling" + ".frozen" + ext


    def freeze(self, plotter, layer_name, try_reload):
        """
        Freeze a database by storing a postprocessed layer image as a numpy
        array (1 data point = 1 pixel). The layer shall be a RGB(A) layer.

        Parameters:
        -----------
        plotter: fs.Fractal_plotter
            A plotter to be used
        layer_name: str
            The layer name - shall be a RGB layer
        try_reload: bool
            if True, will try to reload before computing a new one
        """
        p1 = self.frozen_path("exp", downsampling=False)
        p2 = self.frozen_path("final", downsampling=False)
        logger.info(
            f"Freezing expmap db - try_reload: {try_reload}"
        )

        # Create a mmap for the layer
        layer = plotter[layer_name]
        mode = layer.mode
        if not(mode in "RGB", "RGBA"):
            raise ValueError(
                f"Only a RGB(A) layer can be used to freeze a db"
                + f"found: {mode}"
            )
        dtype = fs.colors.layers.Virtual_layer.DTYPE_FROM_MODE[mode]
        n_channel = fs.colors.layers.Virtual_layer.N_CHANNEL_FROM_MODE[mode]

        exp_mmap = open_memmap(filename=self.path_expmap, mode="r")
        nposts, nh, nt = exp_mmap.shape
        del exp_mmap

        final_mmap = open_memmap(filename=self.path_final, mode="r")
        _nposts, nx, ny = final_mmap.shape
        del final_mmap

        self.frozen_props = {
            "dtype": np.dtype(dtype),
            "n_channel": n_channel,
            "mode": mode
        }

        if try_reload:
            # Does the mmap already exists, does it seems to suit our need ?
            try:
                # Expmap db
                try_exp_mmap = open_memmap(filename=p1, mode='r')
                try_nh, try_nt, try_channel = try_exp_mmap.shape
                valid = (
                    (nh == try_nh) and (nt == try_nt)
                    and (n_channel == try_channel) 
                )
                del try_exp_mmap
                if not(valid):
                    raise ValueError("Invalid db shape")
                # Final db
                try_final_mmap = open_memmap(filename=p2, mode='r')
                try_nx, try_ny, try_channel = try_final_mmap.shape
                valid = (
                    (nx == try_nx) and (ny == try_ny)
                    and (n_channel == try_channel) 
                )
                del try_final_mmap
                if not(valid):
                    raise ValueError("Invalid db shape")
                logger.info(
                    f"Reloading successful, using {p1} and {p2}"
                )
                self.is_frozen = True
                return
            except (ValueError, FileNotFoundError):
                logger.info(
                    f"Reloading failed, computing {p1} and {p2} from scratch"
                )

        # Create new files...
        exp_fr_mmap = open_memmap(
            filename=p1, 
            mode='w+',
            dtype=np.dtype(dtype),
            shape=(nh, nt, n_channel),
            fortran_order=False,
            version=None
        )

        del exp_fr_mmap
        final_fr_mmap = open_memmap(
            filename=p2, 
            mode='w+',
            dtype=np.dtype(dtype),
            shape=(nx, ny, n_channel),
            fortran_order=False,
            version=None
        )
        del final_fr_mmap

        plot_template = Plot_template(plotter, self, frame=None)
        self._raw_kind = "exp" # Used to specify the mmap to `get_2d_arr`
        self.exp_process_for_freeze(plot_template, out_postname=layer_name)
        self._raw_kind = "final"
        self.final_process_for_freeze(plot_template, out_postname=layer_name)
        del self._raw_kind
        
        self.is_frozen = True


    @Multithreading_iterator(
        iterable_attr="exp_db_chunks", iter_kwargs="db_chunk"
    )
    def exp_process_for_freeze(
            self, plot_template, out_postname, db_chunk=None
    ):
        """ Freeze (stores) the RGB array for this layer"""
        (ix, ixx, iy, iyy) = db_chunk
        p_exp = self.frozen_path("exp", downsampling=False)
        fr_mmap = open_memmap(filename=p_exp, mode='r+')

        pasted = False
        for i, layer in enumerate(plot_template.layers):
            if layer.postname == out_postname:
                if not(layer.output):
                    raise ValueError("No output for this layer!!")
                # PILLOW -> Numpy
                paste_crop_arr = np.swapaxes(
                    np.asarray(layer.crop(db_chunk)), 0 , 1
                )[:, ::-1]
                # fr_mmap[ix: ixx, iy: iyy, :] = paste_crop_arr

                if fr_mmap.shape[2] == 1:
                    fr_mmap[ix: ixx, iy: iyy, 0] = paste_crop_arr
                else:
                    fr_mmap[ix: ixx, iy: iyy, :] = paste_crop_arr

                pasted = True

        if not pasted:
            raise ValueError(
                f"Layer missing: {out_postname} "
                + f"not found in {plot_template.postnames}"
            )

        del fr_mmap

    @Multithreading_iterator(
        iterable_attr="final_db_chunks", iter_kwargs="db_chunk"
    )
    def final_process_for_freeze(
            self, plot_template, out_postname, db_chunk=None
    ):
        """ Freeze (stores) the RGB array for this layer"""
        (ix, ixx, iy, iyy) = db_chunk
        p_final = self.frozen_path("final", downsampling=False)
        fr_mmap = open_memmap(filename=p_final, mode='r+')

        pasted = False
        for i, layer in enumerate(plot_template.layers):
            if layer.postname == out_postname:
                if not(layer.output):
                    raise ValueError("No output for this layer!!")
                # PILLOW -> Numpy
                paste_crop_arr = np.swapaxes(
                    np.asarray(layer.crop(db_chunk)), 0 , 1
                )[:, ::-1]

                if fr_mmap.shape[2] == 1:
                    fr_mmap[ix: ixx, iy: iyy, 0] = paste_crop_arr
                else:
                    fr_mmap[ix: ixx, iy: iyy, :] = paste_crop_arr

                pasted = True

        if not pasted:
            raise ValueError(
                f"Layer missing: {out_postname} "
                + f"not found in {plot_template.postnames}"
            )

        del fr_mmap


    def exp_db_chunks(self):
        """
        Generator function
        Yields the chunks spans for the exp_db
        """
        # if chunk_size is None:
        chunk_size = fs.settings.db_chunk_size

        for ih in range(0, self.nh, chunk_size):
            ihh = min(ih + chunk_size, self.nh)
            for it in range(0, self.nt, chunk_size):
                itt = min(it + chunk_size, self.nt)
                yield  (ih, ihh, it, itt)

    def final_db_chunks(self):
        """
        Generator function
        Yields the chunks spans for the final_db
        """
        # if chunk_size is None:
        chunk_size = fs.settings.db_chunk_size

        for ix in range(0, self.nx, chunk_size):
            ixx = min(ix + chunk_size, self.nx)
            for iy in range(0, self.ny, chunk_size):
                iyy = min(iy + chunk_size, self.ny)
                yield  (ix, ixx, iy, iyy)

# --------------- db plotting interface ---------------------------------------
            
    def set_plotter(self, plotter, postname,
                    plotting_modifier=None, reload_frozen=False):
        """
        Define the plotting properties

        Parameters:
        -----------
        plotter: fs.Fractal_plotter
            A plotter to be used as template
        postname: str
            The string indentifier of the layer used for plotting
        plotting_modifier: Optionnal, callable(plotter, time)
            A callback which will modify the plotter instance before each time
            step. Defaults to None, which allows to 'freeze' in place the
            database postprocessad image and interpolate directly in the image.
            Using this option open a lot of possibilities but is also much
            more computer-intensive
        reload_frozen: Optionnal, bool
            Used only if plotting_modifier is None
            If True, will try to reload any previously computed frozen db image
        """
        assert isinstance(plotter, fs.Fractal_plotter)

        self.plotter = plotter
        self.postname = postname
        self.plotting_modifier = plotting_modifier
        self.lf2 = fsfilters.Lanczos_decimator().get_impl(2, 2)

        if plotting_modifier is None:
            # we can freeze the db and interpolate in the frozen image
            logger.info("Database will be frozen for camera move")
            self.freeze(plotter, postname, try_reload=reload_frozen)
        
        self.subsample()


    def get_2d_arr(self, post_index, frame, chunk_slice):
        """ get_2d_arr with frame-specific functionnality

        Parameters:
        -----------
        post_index: int
            the index for this post-processing field in self.plotter
        frame: fs.db.Frame
            Frame localisation for interpolation
        chunk_slice: 4-uplet float
            chunk_slice = (ix, ixx, iy, iyy) is the sub-array to reload
            Used only if `frame` is None: direct reloading
        """
        if frame is None:
            # Used internally when computing the interpolation arrays
            # Direct output : uses chunk_slice
            try:
                filename = self.path(self._raw_kind, downsampling=False)
            except AttributeError:
                raise RuntimeError(
                    "raw get_2d_arr for Exp_db, user should specify the"
                    "plotting kind through arribute `_raw_kind` "
                    "(expecting \"exp\" or \"final\""
                )

            mmap = open_memmap(filename=filename, mode="r")
            (ix, ixx, iy, iyy) = chunk_slice
            ret = mmap[post_index, ix:ixx, iy:iyy]
            del mmap
            return ret   

        else:
            # Interpolated output : uses frame
            ret = self.get_interpolator(frame, post_index)(*frame.pts)
            if frame.supersampling:
                sps_size = (2 * frame.nx, 2 * frame.ny)
                return self.lf2(ret.reshape(sps_size))
            else:
                return ret.reshape(frame.size)


    def plot(self, frame=None):
        """
        Parameters:
        -----------
        frame: fs.db.Exp_frame
            Defines the area to plot. 

        Return:
        -------
        img: PIL.Image
            The plotted image

        Notes:
        ------
        Plotting settings as defined by set_plotter method
        """
        
        # Is the db frozen ?
        if self.is_frozen:
            return self.plot_frozen(frame)

        plot_template = Plot_template(self.plotter, self, frame)

        plotting_modifier = frame.plotting_modifier
        if plotting_modifier is not None:
            plotting_modifier(plot_template, frame.t)

        img = None
        out_postname = self.postname

        for i, layer in enumerate(plot_template.layers):
            if layer.postname == out_postname:
                if not(layer.output):
                    raise ValueError("No output for this layer!!")
                img = PIL.Image.new(mode=layer.mode, size=frame.size)
                im_layer = layer
                break

        if img is None:
            raise ValueError(
                f"Layer missing: {out_postname} "
                + f"not found in {plot_template.postnames}"
            )

        self.process(plot_template, frame, img, im_layer)
        return img


    def plot_frozen(self, frame):
        """ Direct interpolation in a frozen db image data
        """
        dtype = self.frozen_props["dtype"]
        n_channel = self.frozen_props["n_channel"]
        nx, ny = frame.size
        if frame.supersampling:
            im_size = (2 * nx, 2 * ny)
        else:
            im_size = (nx, ny)
        ret = np.empty(im_size[::-1] + (n_channel,), dtype=dtype)

        for ic in range(n_channel): #3): #n_channel):
            
            
            
            channel_ret = self.get_interpolator(frame, ic)(
                *frame.pts, frame.h, frame.nx
            ).reshape(im_size)
            # Numpy -> PILLOW
            ret[:, :, ic] = np.swapaxes(channel_ret, 0 , 1)[::-1, :]
            
            # Image.fromarray((ret * 255).astype(np.uint8))
        
        if n_channel == 1:
            im = PIL.Image.fromarray(ret[:, :, 0])
        else:
            im = PIL.Image.fromarray(ret)

        if frame.supersampling:
            resample = PIL.Image.LANCZOS
            im = im.resize(size=(nx, ny), resample=resample) 

        return im


    def process(self, plot_template, frame, img, im_layer):
        """
        Just plot the Images interpolated data + plot_template
        1 db point -> 1 pixel
        """
        nx, ny = frame.size
        chunk_slice = (0, nx, 0, ny)
        crop_slice = (0, 0, nx, ny)
        paste_crop = im_layer.crop(chunk_slice)
        img.paste(paste_crop, box=crop_slice)











#==============================================================================
# Ad_hoc interpolating routines for exponential mapping

class Multilevel_exp_interpolator:

    def __init__(self,
        a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot,
        a_final, b_final, h_final, f_final, f_final_shape, f_final_slot
    ):
        """
        Multilevel interpolation inside 2 sets of grids:
            - nested set of multilvel exponential 2d grids
            - nested set of cartesian 2d grids for the final image (h_tot < 0)

        Parameters:
        -----------
        a_exp: float array of shape (lvl_exp, 2)
            The lower h, t bounds of the interpolation region for each exp mapping
            level
        b_exp: float array of shape (lvl_exp, 2)
            The upper h, t bounds of the interpolation region for each exp mapping
            level
        h_exp: float array of shape (lvl_exp, 2)
            The h, t grid-spacing at which f is given (2-uplet for each level)
        f_exp: 1d-array
            The base 2d-array data to be interpolated, flatened
        f_exp_shape : 2d-array of shape (lvl_exp, 2)
            The shape for level-ilvl f_exp is f_exp_shape[ilvl, :]
        f_exp_slot : 2d-array of shape (lvl_exp, 2)
            The slot for level-ilvl f_exp is f_exp_slot[ilvl, :]

        a_final: float array of shape (lvl_final, 2)
            The lower x, y bounds of the interpolation region for each cartesian
            mapping level (Note: y bound is identical)
        b_final: float array of shape (lvl_final, 2)
            The upper x, y bounds of the interpolation region for each cartesian
            mapping level (Note: y bound is identical)
        h_final: float array of shape (lvl_final, 2)
            The x, y grid-spacing at which f is given (2-uplet for each level)
        f_final: 1d-array
            The base 2d-array data to be interpolated, flatened
        f_final_shape : 2d-array of shape (lvl_final, 2)
            The shape for level-ilvl f_exp is f_exp_shape[ilvl, :]
        f_final_slot : 2d-array of shape (lvl_final, 2)
            The slot for level-ilvl f_exp is f_exp_slot[ilvl, :]
        """
        self.a_exp = np.asarray(a_exp)
        self.b_exp = np.asarray(b_exp)
        self.h_exp = np.asarray(h_exp)
        self.f_exp = np.asarray(f_exp)
        self.f_exp_shape = np.asarray(f_exp_shape)
        self.f_exp_slot = np.asarray(f_exp_slot)

        self.a_final = np.asarray(a_final)
        self.b_final = np.asarray(b_final)
        self.h_final = np.asarray(h_final)
        self.f_final = np.asarray(f_final)
        self.f_final_shape = np.asarray(f_final_shape)
        self.f_final_slot = np.asarray(f_final_slot)

        # The numba interpolating implementations
        self.numba_impl = self.get_grid_impl()

    def get_grid_impl(self):
        """
        Return a numba-jitted function for multilevel interpolation of 1 grid
        """
        a_exp = self.a_exp
        b_exp = self.b_exp
        h_exp = self.h_exp
        f_exp = self.f_exp
        f_exp_shape = self.f_exp_shape
        f_exp_slot = self.f_exp_slot

        a_final = self.a_final
        b_final = self.b_final
        h_final = self.h_final
        f_final = self.f_final
        f_final_shape = self.f_final_shape
        f_final_slot = self.f_final_slot
        
        @numba.njit(nogil=True, parallel=False)
        def numba_impl(x_out, y_out, pts_h, pts_t, h_out, nx_out, f_out):
            # Interpolation: f_out = finterp(x_out, y_out)
            f_out = multilevel_interpolate(
                x_out, y_out, pts_h, pts_t, h_out, nx_out, f_out,
                a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot,
                a_final, b_final, h_final, f_final, f_final_shape, f_final_slot
            )
            return f_out

        return numba_impl


    def __call__(self, pts_x, pts_y, pts_h, pts_t, pic_h, pic_nx, pts_res=None):
        """
        Interpolates at pts_x, pts_y

        Parameters:
        ----------
        pts_x: 1d-array
            x-coord of interpolating point location
        pts_y: 1d-array
            y-coord of interpolating point location
        pts_h: 1d-array
            h-coord of interpolating point location
        pts_t: 1d-array
            t-coord of interpolating point location
        pic_h: float
            The zoom level of the frame
        pic_nx: int
            The number of point in the frame along the x-direction (used to
            define of local pixel size) 
        pts_res: 1d-array, Optionnal
            Out array handle - if not provided, it will be created. 
        """
        assert np.ndim(pts_x) == 1

        if pts_res is None:
            pts_res = np.empty_like(pts_x)

        interp = self.numba_impl
        interp(pts_x, pts_y, pts_h, pts_t, pic_h, pic_nx, pts_res)

        return pts_res



@numba.njit(nogil=True, parallel=False)
def multilevel_interpolate(
    x_out, y_out, pts_h, pts_t, h_out, nx_out, f_out,
    a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot,
    a_final, b_final, h_final, f_final, f_final_shape, f_final_slot
):
    # Interpolation: f_out = finterp(x_out, y_out, h_out) - h_out constant
    # x_im, y_im position in the image ([-0.5, 0.5] for x)
    npts = x_out.size
    half32 = numba.float32(0.5)
    log_half32 = np.log(half32)
    # tenth_pixw = numba.float32(0.1 / nx_out)

    max_lvl_ht = f_exp_shape.shape[0] - 1
    max_lvl_xy = f_final_shape.shape[0] - 1

    lvl_xy_cache = numba.intp(-1) # Invalid will trigger recalc 
    lvl_ht_cache = numba.intp(-1) # Invalid will trigger recalc 
    
    k_out = np.exp(h_out)

    for i in numba.prange(npts):
        x_loc = x_out[i]
        y_loc = y_out[i]
        # Note: log(sqrt(.)) == 0.5 * log(.)
        h_img = pts_h[i]
        # np.log(max(x_loc ** 2 + y_loc ** 2, tenth_pixw)) * half32

        # exp mapping is defined as z -> np.exp(dh * z)
        h_tot = h_out + h_img - log_half32

        if h_tot < 0.:
            # the lvl is linked to the zoom scale: log2(exp(h_out))
            # zoom 1. -> 0, 2. -> 1, 4. -> 
            lvl_xy = np.intp(np.floor(-h_out / (log_half32)))
            lvl_xy = min(max(lvl_xy, 0), max_lvl_xy)


            # Define the local arrays according to lvlxy_loc
            if lvl_xy != lvl_xy_cache:
                lvl_xy_cache = lvl_xy

                ax = a_final[lvl_xy, 0]
                ay = a_final[lvl_xy, 1]
                bx = b_final[lvl_xy, 0]
                by = b_final[lvl_xy, 1]
                hx = h_final[lvl_xy, 0]
                hy = h_final[lvl_xy, 1]

                slot_xy_l = f_final_slot[lvl_xy, 0]
                slot_xy_h = f_final_slot[lvl_xy, 1]
                xy_nx = f_final_shape[lvl_xy, 0]
                xy_ny = f_final_shape[lvl_xy, 1]
                fxy = f_final[slot_xy_l: slot_xy_h]

            f_loc = grid_interpolate(
                x_loc * k_out, y_loc * k_out,
                fxy, ax, ay, bx, by, hx, hy, xy_nx, xy_ny
            )

        else:
            # the lvl is linked to the pixel position in image
            lvl_ht = np.intp(np.floor(h_img / (log_half32))) - 1
            lvl_ht = min(max(lvl_ht, 0), max_lvl_ht)

            # The angle (t_tot == t_loc)
            t_loc = pts_t[i]
            # np.arctan2(y_loc, x_loc)

            # Define the local arrays according to lvlht_loc
            if lvl_ht != lvl_ht_cache:
                lvl_ht_cache = lvl_ht

                ah = a_exp[lvl_ht, 0]
                at = a_exp[lvl_ht, 1]
                bh = b_exp[lvl_ht, 0]
                bt = b_exp[lvl_ht, 1]
                hh = h_exp[lvl_ht, 0]
                ht = h_exp[lvl_ht, 1]

                slot_ht_l = f_exp_slot[lvl_ht, 0]
                slot_ht_h = f_exp_slot[lvl_ht, 1]
                ht_nx = f_exp_shape[lvl_ht, 0]
                ht_ny = f_exp_shape[lvl_ht, 1]
                fht = f_exp[slot_ht_l: slot_ht_h]

            f_loc = grid_interpolate(
                h_tot, t_loc, fht, ah, at, bh, bt, hh, ht, ht_nx, ht_ny
            )

        f_out[i] = f_loc

    return f_out

CHECK_BOUNDS = True

@numba.njit(cache=True, nogil=True)
def grid_interpolate(x_out, y_out, f, ax, ay, bx, by, hx, hy, nx, ny):
    # Bilinear interpolation in a rectangular grid - f is passed flatten and
    # is of size (nx x ny)
    # Interpolation: f_out = finterp(x_out, y_out)
    
    if CHECK_BOUNDS:
        x_out = min(max(x_out, ax), bx)
        y_out =  min(max(y_out, ay), by)
#        assert x_out >= ax
#        assert x_out <= bx  (*)
#        assert y_out >= ay
#        assert y_out <= by
    
    ix, ratx = divmod(x_out - ax, hx)
    iy, raty = divmod(y_out - ay, hy)
    
    ix = np.intp(ix)
    iy = np.intp(iy)
    ratx /= hx
    raty /= hy

    cx0 = 1. - ratx
    cx1 = ratx
    cy0 = 1. - raty
    cy1 = raty
    
    id00 = ix * ny + iy #     ix,     iy
    id01 = id00 + 1     #     ix, iy + 1
    id10 = id00 + ny    # ix + 1,     iy
    id11 = id10 + 1     # ix + 1, iy + 1

    f_out = (
        (cx0 * cy0 * f[id00])
        + (cx0 * cy1 * f[id01])
        + (cx1 * cy0 * f[id10])
        + (cx1 * cy1 * f[id11])
    )
    return f_out
