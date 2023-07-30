# -*- coding: utf-8 -*-
import copy
import os
import logging

import numpy as np
import numba
import PIL
from numpy.lib.format import open_memmap

import fractalshades as fs
import fractalshades.utils
from fractalshades.numpy_utils.interp2d import (
    Grid_lin_interpolator as fsGrid_lin_interpolator
)

import fractalshades.numpy_utils.filters as fsfilters
from fractalshades.mthreading import Multithreading_iterator

logger = logging.getLogger(__name__)


class Frame:
    def __init__(self, x, y, dx, nx, xy_ratio,
                 t=None, plotting_modifier=None):
        """
    A frame is used to describe a specific data window and for interpolating
    inside a db.

    Parameters
    ----------
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
    t: Optionnal float
        time [s] of this frame in the movie
    plotting_modifier: Optionnal callable
        a plotting_modifier associated with this Frame

    Notes
    -----
    A simple pass-through Frame, extracting the raw ``my_db`` data is:

    ::

        fractalshades.db.Frame(
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
        self.t = t
        self.plotting_modifier = plotting_modifier
        
        self.ny = int(self.nx / self.xy_ratio + 0.5)
        self.dy = dy = self.dx / self.xy_ratio
        self.size = (self.nx, self.ny)
        self.db_size = (self.ny, self.nx) # PIL convention

        xmin = x - 0.5 * dx
        xmax = x + 0.5 * dx
        ymin = y - 0.5 * dy
        ymax = y + 0.5 * dy

        x_1d = np.linspace(xmin, xmax, self.nx, dtype=np.float32)
        y_1d = np.linspace(ymin, ymax, self.ny, dtype=np.float32)

        # Meshgrid of the frame with PIL convention
        x_grid, y_grid  = np.meshgrid(x_1d, y_1d[::-1], indexing='xy')
        self.pts = np.ravel(x_grid), np.ravel(y_grid)


    def upsampled(self, supersampling):
        """ Return the upsampled version of this Frame
        (if supersampling is None just pass through) """
        if supersampling is None:
            return self
        return Frame(
            self.x, self.y, self.dx, self.nx * supersampling, self.xy_ratio,
            self.t, self.plotting_modifier
        )


class Plot_template:
    def __init__(self, plotter, db_loader, frame):
        """ A plotter with a pluggable get_2d_arr method
        -> Basically, an interface to avoid monkey-patching `get_2d_arr`
        
        Parameters
        ----------
        plotter: `fractalshades.Fractal_plotter`
            The wrapped plotter - will be copied
        db_loader: `fractalshades.db.Db`
            A db data-loading class - implementing `get_2d_arr`
        frame: Frame
            The frame-specific data holder
        """
        # Internal plotter object, layers reparented for frame-specific
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

    def get_2d_arr(self, post_index, chunk_slice):
        """ Forwards to the db-loader for frame-specific functionnality """
        return self.db_loader.get_2d_arr(
            post_index, self.frame, chunk_slice
        )


class Db:

    def __init__(self, path):
        """ Wrapper around the memory-mapped numpy-array stored at ``path``\.

        Parameters
        ----------
        path: str
            The path for the data. This memory-mapped array is usually
            created through a `fractalshades.Fractal_plotter.save_db` call.
            Two format are available (\*.db and \*.post db, refer to the doc
            for this function for details)
        """
        # Development Note - on supersampling:
        #  - the .db data is supersampled
        #  - the .postdb rgb data so it is already downsampled
        # General rule, Lanczos filter is applied at image making stage

        self.path = path
        _, ext = os.path.splitext(path)
        if ext == ".db":
            self.postdb = False
        elif ext == ".postdb":
            self.postdb = True
        else:
            raise ValueError(f"Unknown Db extension: {ext}")
        
        self.init_model()
        # Cache for interpolating classes
        self._interpolator = {}

    @property
    def is_postdb(self):
        return self.postdb

    @property
    def zoom_kwargs(self):
        return self.plotter.fractal.zoom_kwargs

    def init_model(self):
        """ Build a description for the datapoints in the mmap """
        mmap = open_memmap(filename=self.path, mode="r+")
        if self.postdb:
            ny, nx, nposts = mmap.shape
        else:
            nposts, ny, nx = mmap.shape
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

        # xy grid used for interpolation.
        # Downsampled version implemented as part of freezing process
        self.xmin0 = xmin0 = x0 - 0.5 * dx0
        self.xmax0 = xmax0 = x0 + 0.5 * dx0
        self.xgrid0, self.xh0 = np.linspace(
            xmin0, xmax0, nx, endpoint=True, retstep=True, dtype=np.float32
        )
        self.ymin0 = ymin0 = y0 - 0.5 * dy0
        self.ymax0 = ymax0 = y0 + 0.5 * dy0
        self.ygrid0, self.yh0 = np.linspace(
            ymin0, ymax0, ny, endpoint=True, retstep=True, dtype=np.float32
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
        logger.info("Creating a new interpolator in database")

        x = frame.x
        y = frame.y
        dx = frame.dx
        dy = frame.dy

        # NOTE that if we are interpolating in the frozen image, then the
        # xgrid0 and ygrid0 might be downsampled
        xgrid0 = self.xgrid0
        ygrid0 = self.ygrid0
        xh0 = self.xh0
        yh0 = self.yh0
        nx = len(xgrid0)
        ny = len(ygrid0)

        # 1) load the local interpolation array for the region of interest
        # with a margin so it has a chance to remain valid for several frames
        if (x - 0.5 * dx) < self.xmin0:
            raise ValueError("Frame partly outside databse data: low x")
        if (y - 0.5 * dy) < self.ymin0:
            raise ValueError("Frame partly outside databse data: low y")
        if (x + 0.5 * dx) > self.xmax0:
            raise ValueError("Frame partly outside databse data: high x")
        if (y + 0.5 * dy) > self.ymax0:
            raise ValueError("Frame partly outside databse data: high y")

        k = 0.5 * 1.5  # 0.5 would be no margin at all
        # min vals for frame interpolation
        x_min = np.float32(max(x - k * dx, self.xmin0))
        ind_xmin = np.searchsorted(xgrid0, x_min, side="right") - 1
        y_min = np.float32(max(y - k * dy, self.ymin0))
        ind_ymin = np.searchsorted(ygrid0, y_min, side="right") - 1

        # max vals for frame interpolation
        x_max = np.float32(min(x + k * dx, self.xmax0))
        ind_xmax = min(np.searchsorted(xgrid0, x_max, side="left"), nx)
        y_max = np.float32(min(y + k * dy, self.ymax0))
        ind_ymax = min(np.searchsorted(ygrid0, y_max, side="left"), ny)

        # 2) Creates and return the interpolator
        # a, b: the lower and upper bounds of the interpolation region
        # h:    the grid-spacing at which f is given
        # f:    data to be interpolated
        # k:    order of local taylor expansions (int, 1, 3, or 5)
        # p:    whether the dimension is taken to be periodic
        # c:    whether the array should be padded to allow accurate close eval
        # e:    extrapolation distance, how far to allow extrap, in units of h
        #       (needs to be an integer)
        a = [xgrid0[ind_xmin], ygrid0[ind_ymin]]
        b = [xgrid0[ind_xmax], ygrid0[ind_ymax]]
        h = [xh0, yh0]
        assert xh0 > 0
        assert yh0 > 0
        assert ind_xmin < ind_xmax
        assert ind_ymin < ind_ymax

        if self.postdb:
            fr_mmap = open_memmap(filename=self.path, mode="r")
            f = fr_mmap[
                (ny - ind_ymax - 1):(ny - ind_ymin),
                ind_xmin:(ind_xmax + 1),
                post_index
            ]
            del fr_mmap

        else:
            mmap = open_memmap(filename=self.path, mode="r")
            f = mmap[
                post_index,
                (ny - ind_ymax - 1):(ny - ind_ymin),
                ind_xmin:(ind_xmax + 1)
            ]
            del mmap

        assert a[0] < b[0]
        assert a[1] < b[1]
        interpolator = fsGrid_lin_interpolator(
            a, b, h, f, PIL_order=True
        )
        bounds = (a, b)

        return interpolator, bounds


# --------------- db plotting interface ---------------------------------------

    def set_plotter(self, plotter, postname):
        """
        Define the plotting properties - Needed only if a \*.db is provided
        not (as opposed to a \*.postdb image array format)

        Parameters
        ----------
        plotter: `fractalshades.Fractal_plotter`
            A plotter to be used as template
        postname: str
            The string indentifier of the layer used for plotting
        """
        assert isinstance(plotter, fs.Fractal_plotter)

        if self.postdb:
            raise RuntimeError(
                    "`set_plotter` shall not be called for a .postdb"
            )

        self.plotter = plotter
        self.postname = postname


    def get_2d_arr(self, post_index, frame, chunk_slice):
        """ get_2d_arr with frame-specific functionnality

        Parameters
        ----------
        post_index: int
            the index for this post-processing field in self.plotter
        frame: `fractalshades.db.Frame`
            Frame localisation for interpolation
        chunk_slice: 4-uplet float
            chunk_slice = (ix, ixx, iy, iyy) is the sub-array to reload
            Not currently used as `frame` is never None (direct reloading)
        """
        assert frame is not None
        ret = self.get_interpolator(frame, post_index)(*frame.pts)
        return ret.reshape(frame.db_size)


    def plot(self, frame=None):
        """
        Parameters
        ----------
        frame: `fractalshades.db.Frame`, Optional
            Defines the area to plot. If not provided, the full db will be
            plotted.

        Returns
        -------
        img: `PIL.Image`
            The plotted image

        Notes
        -----
        Plotting settings are set by `fractalshades.db.Db.set_plotter` method, 
        they are used only for \*.db format (as opposed to a \*.postdb image
        array format)
        """
        if frame is None:
            # Default to plotting the whole db
            frame = fs.db.Frame(
                x=0., y=0., dx=1.0, nx=self.nx, xy_ratio=self.xy_ratio
            )

        # Is the db filled with raw rgb data ?
        if self.postdb:
            return self.plot_postdb(frame)

        # Here the plotter shall take into account the 'full frame' size
        full_frame = frame.upsampled(self.plotter.supersampling)
        plot_template = Plot_template(self.plotter, self, full_frame)

        plotting_modifier = frame.plotting_modifier
        if plotting_modifier is not None:
            plotting_modifier(plot_template, frame.t)

        img = None
        out_postname = self.postname

        for i, layer in enumerate(plot_template.layers):
            if layer.postname == out_postname:
                if not(layer.output):
                    raise ValueError(f"No output for this layer: {layer}")
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


    def plot_postdb(self, frame):
        """ Direct interpolation in a .postdb db image data
        """
        mmap = open_memmap(filename=self.path, mode="r+")
        _, _, n_channel = mmap.shape
        dtype = mmap.dtype
        del mmap

        db_size = frame.db_size
        ret = np.empty(db_size + (n_channel,), dtype=dtype)

        for ic in range(n_channel):
            channel_ret = self.get_interpolator(frame, ic)(*frame.pts)
            channel_ret = channel_ret.reshape(db_size)
            ret[:, :, ic] = channel_ret

        if n_channel == 1:
            im = PIL.Image.fromarray(ret[:, :, 0])
        else:
            im = PIL.Image.fromarray(ret)

        return im


    def process(self, plot_template, frame, img, im_layer):
        """
        Just plot the Images interpolated data + plot_template
        1 db point -> 1 pixel
        """
        nx, ny = full_nx, full_ny = frame.size
        ss = self.plotter.supersampling

        chunk_slice = (0, nx, 0, ny)
        crop_slice = (0, 0, nx, ny)
        # This line ultimately forwards to self.get_2d_arr(...) thanks
        # to Plot_template interface - frame is not None
        paste_crop = im_layer.crop(chunk_slice)

        if ss:
            # Here, we apply a resizing filter
            resample = PIL.Image.LANCZOS
            paste_crop = paste_crop.resize(
                size=(nx, ny),
                resample=resample,
                box=None,
                reducing_gap=None
            )

        img.paste(paste_crop, box=crop_slice)


#==============================================================================
#==============================================================================
# Exponential mapping to cartesian frame transform

class Exp_frame:
    def __init__(self, h, nx, xy_ratio,
                 t=None, plotting_modifier=None, pts=None):
        """
    This class is used to describe a specific data window for interpolation
    inside a `fractalshades.db.Exp_db`.

    Parameters
    ----------
    h: float >= 0.
        zoom level. A zoom level of 0. denotes that dx = dx0 fully zoomed-in
        frame - for h > 0, dx = dx0 * np.exp(h) 
    nx: int
        number of pixels for the interpolated frame
    xy_ratio: float
        width / height ratio of the interpolated frame 
    t: Optionnal float
        time [s] of this frame in the movie
    plotting_modifier: Optional callable
        a plotting_modifier associated with this frame
    pts: Optional, 4-uplet of arrays
        The x, y, h, t grid as returned by make_exp_grid - if not provided it
        will be recomputed - but more efficient to share between frames
    """
        self.h = h
        self.nx = nx
        self.xy_ratio = xy_ratio
        self.t = t
        self.plotting_modifier = plotting_modifier
        
        self.ny = int(self.nx / self.xy_ratio + 0.5)
        self.size = (self.nx, self.ny)
        self.db_size = (self.ny, self.nx) # PIL convention
 
        if pts is None:
            pts = self.make_exp_grid(self.nx, self.xy_ratio)
        # Basic shape verification
        assert pts[0].size == (self.nx * self.ny)
        self.pts = pts

    @staticmethod
    def make_exp_grid(nx, xy_ratio):
        """ Return a base grid [-0.5, 0.5] x [-0.5/xy_ratio, 0.5/xy_ratio] in 
        both cartesian and expoential coordinates """
        ny = int(nx / xy_ratio + 0.5)

        xmin = -0.5
        xmax = +0.5
        ymin = -0.5 / xy_ratio
        ymax = +0.5 / xy_ratio

        # Cartesian grid
        xvec = np.linspace(xmin, xmax, nx, dtype=np.float32)
        yvec = np.linspace(ymin, ymax, ny, dtype=np.float32)

        x_grid, y_grid = np.meshgrid(xvec, yvec[::-1], indexing='xy')
        x_grid = x_grid.reshape(-1)
        y_grid = y_grid.reshape(-1)

        # Exponential grid coordinates
        frac_pixw = 0.1 / nx
        h_grid = 0.5 * np.log(
            np.maximum(x_grid ** 2 + y_grid ** 2, frac_pixw ** 2)
        )
        t_grid = np.arctan2(y_grid, x_grid)

        # Store the grid
        return (x_grid, y_grid, h_grid, t_grid)



class Exp_db:

    def __init__(self, path_expmap, path_final):
        """ Wrapper around the raw array data stored at ``path_expmap`` and
        ``path_final``\.

        Parameters
        ----------
        path_expmap: str
            The path for the expmap database. Note that only \*.postdb format 
            is currently supported hence the expmap array is of shape
            (nt, nh, nchannels) and stores rgb data. It is usually saved  by a
            call to `fractalshades.Fractal_plotter.save_db`
            using a `fractalshades.projection.Expmap` projection. Note that the
            orientation parameter of this projection shall be "vertical".
        path_final: str
            The path for the final raw data. Note that only \*.postdb format is
            currently supported hence the expmap array is of shape
            (ny, nx, nchannels) and stores rgb data. It is usually saved by a
            `fractalshades.Fractal_plotter.save_db` call (with a standard
            `fractalshades.projection.Cartesian` projection). It shall be
            square (nx == ny) and only \*.postdb format is currently supported.
        """
        # Development Note
        # ----------------
        # Note on supersampling:
        #  - the .db data is supersampled
        #  - the .postdb data is the image so it is already downsampled
        # General rule, Lanczos filter is applied at image making stage
        self.path_expmap = path_expmap
        _, ext = os.path.splitext(path_expmap)
        if ext == ".db":
            self.postdb = False
            raise NotImplementedError(
                "Only .postdb files implemented for making exp zoom movies. "
                "Consider saving your database in this format."
            )
        elif ext == ".postdb":
            self.postdb = True
        else:
            raise ValueError(f"Unknown Db extension: {ext}")

        self.path_final = path_final
        _, ext2 = os.path.splitext(path_final)
        if ext != ext2:
            raise ValueError(
                "Extensions for path_expmap and path_final shall match ; "
                f"Found: {ext} and {ext2}"
            )

        self.init_model()
        self.subsample()

        # Cache for interpolating classes
        self._interpolator = {}

    @property
    def is_postdb(self):
        return self.postdb

    def init_model(self):
        """ Build a description for the datapoints in the mmap """
        assert self.postdb
        mmap = open_memmap(filename=self.path_expmap, mode="r+")
        # .postdb of an Expmap woth orientation = "vertical"
        nh, nt, nposts = mmap.shape
        dtype = mmap.dtype
        del mmap

        mmap = open_memmap(filename=self.path_final, mode="r+")
        ny, nx, _nposts = mmap.shape
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

        self.nposts = self.nchannels = nposts
        self.dtype = dtype

        # Points number
        self.nh = nh
        self.nt = nt
        self.nx = nx
        self.ny = ny

        # Data span
        dh0 = 2. * np.pi * nh / nt
        self.hmin0 = 0.
        self.hmax0 = self.hmin0 + dh0
        self.xmin0, self.xmax0 = -0.5, 0.5
        self.ymin0, self.ymax0 = -0.5, 0.5

        # ht grid
        self.hgrid0, self.hh0 = np.linspace(
            self.hmin0, self.hmax0, nh, endpoint=True, retstep=True,
            dtype=np.float32
        )
        self.tgrid0, self.th0 = np.linspace(
            -np.pi, np.pi, nt, endpoint=True, retstep=True,
            dtype=np.float32
        )

        # xy grid
        self.xgrid0, self.xh0 = np.linspace(
            self.xmin0, self.xmax0, nx, endpoint=True, retstep=True,
            dtype=np.float32
        )
        self.ygrid0, self.yh0 = np.linspace(
            self.ymin0, self.ymax0, ny, endpoint=True, retstep=True,
            dtype=np.float32
        )

        # Lanczos-2 2-decimation routine
        self.lf2_stable = fsfilters.Lanczos_decimator().get_stable_impl(2)


    def path(self, kind, downsampling):
        """ Path to the database, including the cascading subsampled db

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
        xy_ratio = frame.xy_ratio

        margin = 20. # Shall remain valid for this zoom range (in and out)
        h_margin = np.log(margin)
        h_decimate = np.log(2.) # Triggers factor-2 image decimation 

        info_dic = self._subsampling_info
        dtype = self.dtype

        # Checks Frame validity
        if h < 0.:
            raise ValueError(f"Frame outside databse data: h = {h} < 0")
        # Highest acceptable h:
        rpix_max = 0.5 * np.sqrt(1. + xy_ratio ** 2)
        allowed_hmax = self.hmax0 - np.log(rpix_max)
        if h > allowed_hmax:
            raise ValueError(
                    "Frame outside databse data: "
                    f"h = {h} > allowed_hmax = {allowed_hmax}\n"
                    f"hmax0: {self.hmax0} "
                    f"rpix_max: {rpix_max} "
                    f"xy_ratio: {xy_ratio} "
                    f"nh: {self.nh} "
                    f"nt: {self.nt} "
                    f"hmin: {self.hmin0}"
            )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define parameters for multilevel exp_map interpolation
        # a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot
        kind = "exp"
        full_shape = info_dic[(kind, "ss_shapes")] # ny, nx or nh, nt
        full_slot = info_dic[(kind, "ss_slots")]
        full_bound = info_dic[(kind, "ss_bounds")] # (start_x, end_x, start_y, end_y) 
        lvl = full_shape.shape[0]

        # We fill as if full range first
        # Note that the bounds are expressed as h, t even with PIL indexing
        a_exp = np.copy(full_bound[:, 0::2]) # Lower bound h, t
        b_exp = np.copy(full_bound[:, 1::2]) # Higher bound h, t
        h_exp = ((b_exp - a_exp) / (full_shape - 1)).astype(np.float32)
        
        f_exp_shape = np.copy(full_shape)
        f_exp_slot = np.copy(full_slot)
        
        # we extract a subrange for the h direction
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
            # Note: still, ascending sort order
            ind_hmin = int(np.floor((pix_hmin - arr_hmin) / delta_h))
            ind_hmax = int(np.ceil((pix_hmax - arr_hmin) / delta_h))
            
            # Avoids the 'lonely pixel' case
            if ind_hmax - ind_hmin == 1:
                if ind_hmax < f_exp_shape[ilvl, 0]:
                    ind_hmax += 1
                else:
                    ind_hmin -= 1

            h_index[ilvl, :] = ind_hmin, ind_hmax
            k_min = ind_hmin / (full_shape[ilvl, 0] - 1)
            k_max = ind_hmax / (full_shape[ilvl, 0] - 1)

            # Updates tables to extracted values
            a_exp[ilvl, 0] = arr_hmin * (1. - k_min) + arr_hmax * k_min
            b_exp[ilvl, 0] = arr_hmin * (1. - k_max) + arr_hmax * k_max
            f_exp_shape[ilvl, 0] = ind_hmax - ind_hmin
            # Temporarly, we store it as dim, then use cumsum
            f_exp_slot[ilvl, 1] = f_exp_shape[ilvl, 0] * f_exp_shape[ilvl, 1]

        f_exp_slot[:, 1] = np.cumsum(f_exp_slot[:, 1])
        f_exp_slot[1:, 0] = f_exp_slot[:-1, 1]
        f_exp_slot[0, 0] = 0
        f_exp = np.empty((f_exp_slot[-1, 1],), dtype=dtype) # storage vec


        ilvl = 0
        filename = self.path(kind, downsampling=False)
        mmap = open_memmap(filename=filename, mode="r")

        ind_hmin, ind_hmax = h_index[ilvl, :]
        loc_arr = mmap[ind_hmin:ind_hmax, :, ic] # Use the full theta range
        loc_arr = loc_arr.reshape(-1)

        f_exp[f_exp_slot[ilvl, 0]: f_exp_slot[ilvl, 1]] = loc_arr

        del mmap

        filename = self.path(kind, downsampling=True)
        mmap = open_memmap(filename=filename, mode="r")

        for ilvl in range(1, lvl):
            ind_hmin, ind_hmax = h_index[ilvl, :]
            ny, nx = full_shape[ilvl, :]
            di = full_slot[ilvl, 0]  # 0 or 1 ???
            # Here why we need to used "vertical" orientation for the expmap,
            # as the primary dim is nh...
            loc_arr = mmap[ic, (ind_hmin * nx) + di: (ind_hmax * nx) + di]
            f_exp[f_exp_slot[ilvl, 0]: f_exp_slot[ilvl, 1]] = loc_arr

        del mmap


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define parameters for multilevel final interpolation
        # a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot
        kind = "final"
        full_shape = info_dic[(kind, "ss_shapes")]
        full_slot = info_dic[(kind, "ss_slots")]
        full_bound = info_dic[(kind, "ss_bounds")] # (start_x, end_x, start_y, end_y) 
        lvl = full_shape.shape[0]

        # Easy, we just fill as full range - no subrange extraction step
        a_final = np.copy(full_bound[:, 0::2])
        b_final = np.copy(full_bound[:, 1::2])
        h_final = ((b_final - a_final) / (full_shape - 1)).astype(np.float32)
        f_final_shape = np.copy(full_shape)
        f_final_slot = np.copy(full_slot)

        # we still adjust the slots position as first level is now merged
        for ilvl in range(lvl):
            f_final_slot[ilvl, 1] = f_final_shape[ilvl, 0] * f_final_shape[ilvl, 1]
        f_final_slot[:, 1] = np.cumsum(f_final_slot[:, 1])
        f_final_slot[1:, 0] = f_final_slot[:-1, 1]
        f_final_slot[0, 0] = 0
        f_final = np.empty((f_final_slot[-1, 1],), dtype=dtype)


        ilvl = 0
        filename = self.path(kind, downsampling=False)
        mmap = open_memmap(filename=filename, mode="r")
        loc_arr = mmap[:, :, ic]
        loc_arr = loc_arr.reshape(-1)
        f_final[f_final_slot[ilvl, 0]: f_final_slot[ilvl, 1]] = loc_arr
        del mmap

        filename = self.path(kind, downsampling=True)
        mmap = open_memmap(filename=filename, mode="r")
        for ilvl in range(1, lvl):
            filename = self.path(kind, downsampling=True)
            mmap = open_memmap(filename=filename, mode="r")
            loc_arr = mmap[ic, full_slot[ilvl, 0]: full_slot[ilvl, 1]]
            f_final[f_final_slot[ilvl, 0]: f_final_slot[ilvl, 1]] = loc_arr
        del mmap



        interpolator = Multilevel_exp_interpolator(
            a_exp, b_exp, h_exp, f_exp, f_exp_shape, f_exp_slot,
            a_final, b_final, h_final, f_final, f_final_shape, f_final_slot
        )
        bounds = (h + h_margin, h - h_margin)

        return interpolator, bounds


# --------------- db Subsampling interface ---------------------------------------
    def subsample(self):
        """ Make a series of subsampled databases (either pain or frozen) """
        self._subsampling_info = {}

        # 1) Downsample the frozen exp db
        source = self.path("exp", downsampling=False)
        filename = self.path("exp", downsampling=True)
        init_bound = np.array((self.hmin0, self.hmax0, -np.pi, np.pi))
        self.populate_subsampling(
            filename, source, driving_dim="x", # i.e, the "t" Expmap dim
            init_bound=init_bound, kind="exp"
        )

        # 2) Downsample the frozen final db
        source = self.path("final", downsampling=False)
        filename = self.path("final", downsampling=True)
        init_bound = np.array((self.xmin0, self.xmax0, self.ymin0, self.ymax0))
        self.populate_subsampling(
            filename, source, driving_dim="x",
            init_bound=init_bound, kind="final"
        )

    def ss_lvl_count(self, kind):
        """
        Return the number of subsampling data levels stored for this db.

        Parameters
        ----------
        kind: "exp" | "final"
            The source db
        """
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
        assert self.postdb
        
        dtype = self.dtype
        nc = self.nchannels

        ss_shapes = self._subsampling_info[((kind, "ss_shapes"))]
        ss_slots = self._subsampling_info[((kind, "ss_slots"))]

        nx, ny = ss_shapes[lvl, :]
        lw, hg = ss_slots[lvl, :]
        arr = np.empty((nx, ny, nc), dtype=dtype)

        filename = self.path(kind, downsampling=(lvl != 0))
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
        return PIL.Image.fromarray(arr)


    def populate_subsampling(self, filename, source, driving_dim,
                             init_bound, kind):
        """
        Creates a memory mapping at filename and populates it with subsampled
        data from source.

        Parameters
        ----------
        filename: str
            path for the new mmap
        source: str
            path for the source mmap
        driving_dim: "x", "y"
            The dimension of the image that will be reduced to 2 (criteria for 
            the number of levels)
        init_bound: np.array([minx, maxx, miny, maxy])
            The initial range for data
        kind: "exp" | "final"
            The kind of mem mapping

        Returns
        -------
        ss_shapes:
            shapes (ny, nx) or (nh, nt) of the nested subsampled arrays
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
        # For a .postdb, mmap.shape: (ny, nx, n_channel)
        # Hence for a "vertical" Expmap: nh, nt
        source_mmap = open_memmap(filename=source, mode="r")
        dtype = source_mmap.dtype
        (ny, nx, nposts) = source_mmap.shape
        # Dev note: in case of expmap, ny == nh, nx == nt

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
        ss_bounds = np.tile(init_bound, (ss_lvls + 1, 1)).astype(np.float32)

        # Sizing the subsampling arrays
        ss_shapes[0, :] = [ny, nx] # or nh, nt if Expmap, "vertical"
        ss_slots[0, :] = [-1, -1] # not relevant as in another mmap

        for lvl in range(ss_lvls):
            ss_nx = ss_nx // 2 + 1
            ss_ny = ss_ny // 2 + 1
            ss_slotl = ss_sloth
            ss_sloth += ss_nx * ss_ny
            ss_shapes[lvl + 1, :] = [ss_ny, ss_nx]
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
            ss_ny = ny
            for lvl in range(ss_lvls):
                ss_diy = 200
                ss_ny = ss_ny // 2 + 1
                # The grouping range (in data columns) for parallel exec.
                self.y_range = lambda: np.arange(0, ss_ny - 1, ss_diy)
                self.parallel_populate_subsampling(
                    ss_mmap, source_mmap, ipost, lvl,
                    ss_shapes, ss_slots, ss_bounds, lf2_stable,
                    ss_diy, ss_iystart=None
                )

        del source_mmap
        del ss_mmap
        
        self._subsampling_info.update({
            (kind, "ss_shapes"): ss_shapes, # Shape of the full ss array, lvl = i - 1
            (kind, "ss_slots"): ss_slots,   # 1d-slot of the full ss array, lvl = i - 1
            (kind, "ss_bounds"): ss_bounds  # bounds for the full ss array
        })


    @Multithreading_iterator(
        iterable_attr="y_range", iter_kwargs="ss_iystart"
    )
    def parallel_populate_subsampling(self,
        mmap, source_mmap, ipost, lvl,
        ss_shapes, ss_slots, ss_bounds, lf2_stable,
        ss_diy, ss_iystart=None
    ):
        """
        In parallel, apply the subsampling for (ipost, lvl).

        Parameters
        ----------
        mmap: memory mapping for the output
        source_mmap: memory mapping for the source (used if lvl == 0)
        ipost: the current post / channel index
        lvl: current level in the nested chain
        ss_shapes: (nx, ny) of the nested subsampled array - coords in source
        ss_slots: (ssl, ssh) of the nested subsampled array - as stored,
            flatten, in res
        ss_bounds: (start_x, end_x, start_y, end_y) of the nested ss arrays
        lf2_stable: decimation routine
        ss_iystart: start iy index for this parallel calc in the destination
           array /!\ not the source
        ssdiy: gap in y used for parallel calc
        """        
        ss_ny, ss_nx = ss_shapes[lvl + 1, :]   # For full "subsampled" shape
        ss_l, ss_h = ss_slots[lvl + 1, :]      # For full "subsampled" slot

        assert ss_ny * ss_nx == ss_h - ss_l

        # This // run extract slot is [iy_start:iy_end, :]
        ss_iyend = min(ss_iystart + ss_diy, ss_ny)
        ss_diy = ss_iyend - ss_iystart

        # The 2d shapes / extract slot at source array - we map (2n+1) -> n+1
        ny, nx = ss_shapes[lvl, :]
        lw, hg = ss_slots[lvl, :]  # This is the full "subsampled" slot
        iystart = 2 * ss_iystart  # *2 due to the supersampling factor
        iyend = min(iystart + 2 * ss_diy + 1, ny)
        diy = iyend - iystart

        if lvl == 0:
            # Source arr is from the source_mmap
            source_arr = source_mmap[iystart:iyend, :, ipost]
        else:
            # Source arr is from the mmap, however a level higher
            l_loc = lw + iystart * nx
            h_loc = lw + iyend * nx
            source_arr = mmap[ipost, l_loc:h_loc].reshape((diy, nx))

        ssl_loc = ss_l + ss_iystart * ss_nx
        ssh_loc = ss_l + ss_iyend * ss_nx
        ss2d_full, k_spanx_loc, k_spany_loc = lf2_stable(source_arr)
        ss2d_full = ss2d_full[:ss_diy, :]

        # Flatten then store in slot
        mmap[ipost, ssl_loc:ssh_loc] = ss2d_full.reshape(-1)

        if (ipost == 0) and (ss_iystart == 0):
            # We store data localisation information. coeff applies to the 
            # following levels
            ss_bounds[(lvl + 1):, 1] +=  (k_spanx_loc - 1.) * (
                ss_bounds[(lvl + 1):, 1] - ss_bounds[(lvl + 1):, 0]
            )
            ss_bounds[(lvl + 1):, 3] +=  (k_spany_loc - 1.) * (
                ss_bounds[(lvl + 1):, 3] - ss_bounds[(lvl + 1):, 2]
            )


# --------------- db plotting interface ---------------------------------------
    def get_2d_arr(self, post_index, frame, chunk_slice):
        """ get_2d_arr with frame-specific functionnality

        Parameters
        ----------
        post_index: int
            the index for this post-processing field in self.plotter
        frame: fs.db.Frame
            Frame localisation for interpolation
        chunk_slice: 4-uplet float
            chunk_slice = (ix, ixx, iy, iyy) is the sub-array to reload
            Used only if `frame` is None: direct reloading
        """
        assert frame is not None
        # Interpolated output : uses frame data
        ret = self.get_interpolator(frame, post_index)(*frame.pts)
        return ret.reshape(frame.size)


    def plot(self, frame):
        """
        Parameters
        ----------
        frame: `fractalshades.db.Exp_frame`
            Defines the area to plot.

        Returns
        -------
        img: `PIL.Image`
            The plotted image

        Notes
        -----
        Plotting settings are defined by ``set_plotter`` method.
        """
        assert self.postdb
        
        dtype = self.dtype
        nchannels = self.nchannels

        db_size = frame.db_size
        ret = np.empty(db_size + (nchannels,), dtype=dtype)


        for ic in range(nchannels):

            channel_ret = self.get_interpolator(frame, ic)(
                *frame.pts, frame.h, frame.nx
            )
            channel_ret = channel_ret.reshape(db_size)
            # Numpy -> PILLOW
            ret[:, :, ic] = channel_ret

        if nchannels == 1:
            im = PIL.Image.fromarray(ret[:, :, 0])
        else:
            im = PIL.Image.fromarray(ret) 

        return im


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

        Parameters
        ----------
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

        Parameters
        ----------
        pts_x: 1d-array float32
            x-coord of interpolating point location
        pts_y: 1d-array float32
            y-coord of interpolating point location
        pts_h: 1d-array float32
            h-coord of interpolating point location
        pts_t: 1d-array float32
            t-coord of interpolating point location
        pic_h: float
            The zoom level of the frame
        pic_nx: int
            The number of point in the frame along the x-direction (used to
            define of local pixel size) 
        pts_res: 1d-array float32, Optional
            Out array handle - if not provided, it will be created. 
        """
        assert np.ndim(pts_x) == 1

        if pts_res is None:
            pts_res = np.empty_like(pts_x)

        interp = self.numba_impl
        pic_h32 = np.float32(pic_h)
        interp(pts_x, pts_y, pts_h, pts_t, pic_h32, pic_nx, pts_res)

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
            # Interpolating in the final image
            # the lvl is linked to the zoom scale: log2(exp(h_out))
            # zoom 1. -> 0, 2. -> 1, 4. -> 
            lvl_xy = np.intp(-h_out / (log_half32) + 0.7)
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

            f_loc = grid_interpolate_alt(
                x_loc * k_out, y_loc * k_out,
                fxy, ax, ay, bx, by, hx, hy, xy_nx, xy_ny
            )

        else:
            # the lvl is linked to the pixel position in image
            lvl_ht = np.intp((h_img / log_half32) - 1.0)
            lvl_ht = min(max(lvl_ht, 0), max_lvl_ht)
            # The angle (t_tot == t_loc)
            t_loc = pts_t[i]

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


# debugging options
CHECK_BOUNDS = False
CLIP_BOUNDS = True

@numba.njit(nogil=True)
def grid_interpolate(x_out, y_out, f, ax, ay, bx, by, hx, hy, nx, ny):
    # Bilinear interpolation in a rectangular grid - f is passed flatten and
    # is of size (nx x ny)
    # Interpolation: f_out = finterp(x_out, y_out)

    if CHECK_BOUNDS:
        assert  ax <= x_out <= bx
        assert  ay <= y_out <= by

    if CLIP_BOUNDS:
        x_out = min(max(x_out, ax), bx)
        y_out = min(max(y_out, ay), by)

    ix, ratx = np.divmod(x_out - ax, hx)
    iy, raty = np.divmod(y_out - ay, hy)
    
    ix = np.intp(ix)
    iy = np.intp(iy)
    ratx /= hx
    raty /= hy

    cx0 = np.float32(1.) - ratx
    cx1 = ratx
    cy0 = np.float32(1.) - raty
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

@numba.njit(nogil=True)
def grid_interpolate_alt(x_out, y_out, f, ax, ay, bx, by, hx, hy, nx, ny):
    # Bilinear interpolation in a rectangular grid - f is passed flatten and
    # is of size (ny x nx)   / PILLOW order convention
    # Interpolation: f_out = finterp(x_out, y_out)

    if CHECK_BOUNDS:
        assert  ax <= x_out <= bx
        assert  ay <= y_out <= by

    if CLIP_BOUNDS:
        x_out = min(max(x_out, ax), bx)
        y_out = min(max(y_out, ay), by)

    ix_float, ratx = np.divmod(x_out - ax, hx)
    iy_float, raty = np.divmod(by - y_out, hy)
    
    ix = np.intp(ix_float)
    iy = np.intp(iy_float)
    ratx /= hx
    raty /= hy

    cx0 = np.float32(1.) - ratx
    cx1 = ratx
    cy0 = np.float32(1.) - raty
    cy1 = raty

    id00 = iy * nx + ix #       iy, ix
    id01 = id00 + 1     #       iy, ix + 1
    id10 = id00 + nx    #   iy + 1, ix
    id11 = id10 + 1     #   iy + 1, ix

    f_out = (
        (cy0 * cx0 * f[id00])
        + (cy0 * cx1 * f[id01])
        + (cy1 * cx0 * f[id10])
        + (cy1 * cx1 * f[id11])
    )
    
    return f_out
