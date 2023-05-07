# -*- coding: utf-8 -*-
#import copy
import logging
import os
import concurrent.futures

import numpy as np
#import numba

import fractalshades as fs
import fractalshades.db
import fractalshades.utils


try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    raise RuntimeError("Scipy is needed for movie maker, install with: "
                       "'pip install scipy'")
# Note: Scipy is used only for its monotonic cubic interpolant, we could remove
# this dependency if we reimplement it:
# https://math.stackexchange.com/questions/45218/implementation-of-monotone-cubic-interpolation

try:
    import av
except ImportError:
    raise RuntimeError("PyAV is needed for movie maker- please install with: "
                       "'pip install av'")
# PyAV is a Pythonic binding for the FFmpeg libraries. We aim to provide all of
# the power and control of the underlying library, but manage the gritty
# details as much as possible.

logger = logging.getLogger(__name__)


class Movie():

    def __init__(self, plotter, postname, size=(720, 480), fps=24,
                 plotting_modifier=None,
                 reload_frozen=False):
        """
        A movie-making class
        
        Parameters:
        -----------
        plotter: `fs.Fractal_plotter`
            The base plotter for this movie
        postname: str
            The string indentifier for the layer used to make the movie
        size: (int, int)
            Movie screen size in pixels. Default to 720 x 480 
        fps: int
            movie frame-count per second
        plotting_modifier: Optionnal, callable(plotter, time)
            A callback which will modify the plotter instance before each time
            step. Defaults to None, which allows to 'freeze' in place the
            database postprocessad image and interpolate directly in the image.
            Using this option open a lot of possibilities but is also much
            more computer-intensive
        reload_frozen: bool
            Used only if plotting_modifier is None
            If True, will try to reload any previousy computed frozen db image
        """
        # Note:
        # Standard 16:9 resolutions can be:
        # - 3200 x 1800 (QHD+) (*)
        # - 2560 × 1440 (1440p, QHD) (*)
        # - 1920 × 1080 (1080p, FHD) (*)
        # -     1600 × 900 (HD+)
        # -     1366 × 768 (WXGA)
        # - 1280 × 720 (720p, HD) (*)
        # -     960 × 540 (qHD)
        # - 720 × 480 (480p, SD) (*)

        self.plotter = plotter
        self.postname = postname
        self.plotting_modifier = plotting_modifier
        self.reload_frozen = reload_frozen

        self.width = size[0]
        self.height = size[1]
        self.fps = fps
        self.cam_moves = []


    def add_camera_move(self, cm):
        """ Adds a new camera move """
        if cm.movie is not None:
            raise RuntimeError("Can only add a Camera move to one Movie_maker")
        cm.set_movie(self)
        self.cam_moves.append(cm)


    def picture(self, i):
        acc_frames = 0
        for cm in self.cam_moves:
            cm_nframe = cm.nframe
            local_i = i - acc_frames
            if local_i < cm_nframe:
                return cm.picture(local_i)
            acc_frames += cm_nframe
        raise KeyError(f"No frame at {i}")


    def pictures(self):
        """ Iterator for the movie frames - as PIL.Image
        """
        for cm in self.cam_moves:
            for pic in cm.pictures():
                yield pic


    def make(self, out_file, pix_fmt="yuv420p", crf=22):
        """
        Make a .mp4 movie - Uses Codec AVC standard (H.264)

        Parameters:
        -----------
        out_file: str
            The path to output file
        pix_fmt: "yuv444p" | "yuv420p"
            The pixel format’s name - default: "yuv420p"
        crf: int, 0 - 51
            The "Constant Rate Factor" - default: 22 (51 = lowest quality)
        """
        # See several examples:
# https://stackoverflow.com/questions/73609006/how-to-create-a-video-out-of-frames-without-saving-it-to-disk-using-python
# https://github.com/PyAV-Org/PyAV/blob/develop/scratchpad/encode_frames.py
# https://stackoverflow.com/questions/61260182/how-to-output-x265-compressed-video-with-cv2-videowriter
# https://github.com/kkroening/ffmpeg-python
# https://github.com/NVIDIA/cuda-python
# https://github.com/AiueoABC/CompressedVideoGenerationExample/issues/1
        fs.utils.mkdir_p(os.path.dirname(out_file))

        with open(out_file, "wb") as f:
            # Use PyAV to open out_file as MP4 video
            # output_memory_file = io.BytesIO()
            output = av.open(f, 'w', format="mp4")
            codec_name = 'h264'
    
            # Advanced Video Coding (AVC), also referred to as H.264 or
            # MPEG-4 Part 10, is a video compression standard based on
            # block-oriented, motion-compensated coding.
            stream = output.add_stream(codec_name, rate=str(self.fps))
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = pix_fmt
            # https://ffmpeg.org/ffmpeg.html#Video-Options
            stream.options = {'crf': str(crf)}

            # Iterate the PIL images, convert image to PyAV VideoFrame, encode,
            # and "Mux":
            for pic in self.pictures():
                frame = av.VideoFrame.from_image(pic)
                packet = stream.encode(frame)
                output.mux(packet)

            # Flush the encoder and close the output file
            packet = stream.encode(None)
            output.mux(packet)
            output.close()


    def debug(self, out_file, first, last):
        """
        Output a selection of frames to .png format for debuging 

        Parameters:
        -----------
        out_file: str
            The path to output file wiil be out_file_%i%.png where i is the 
            frame number
        first: int
            The first saved frame
        last: int
            The last saved frame

        """
        head, tail = os.path.split(out_file)
        fs.utils.mkdir_p(head)

        for i in range(first, last +1):
            pic = self.picture(i)
            suffixed = tail + "_" + str(i) + ".png"
            pic.save(os.path.join(head, suffixed))


class Camera_move:
    def __init__(self, db, tmin, tmax):
        """
        Base class for classes implementing a movie sequence
        
        Time is relative to the whole movie
        Frame index is relative to this Cam move
        """
        self.db = db
        self.tmin = tmin
        self.tmax = tmax
        self.dt = tmax - tmin
        self.movie = None # add to a movie with movie.add_frame


    def set_movie(self, movie):
        """
        Link this camera move to a movie objects and adjust its internals
        properties accordingly.

        Parameters:
        ----------
        movie: fs.movie.Movie
        """
        self.movie = movie
        self.db.set_plotter(
            movie.plotter, movie.postname,
            movie.plotting_modifier, movie.reload_frozen
        )
        self.fps = self.movie.fps
        self.nframe = int((self.tmax - self.tmin) * self.fps)

        self.nx = self.movie.width
        self.xy_ratio = self.movie.width / self.movie.height
        self.plotting_modifier = self.movie.plotting_modifier
        
        self.make_grids()

    def picture(self, iframe):
        """ Returns the ith-frame as a PIL.Image object """
        return self.db.plot(frame=self.get_frame(iframe))

    def async_picture(self, iframe):
        # keeps trac of the frame index for async accumulation & flushing
        return iframe, self.picture(iframe)

    def pictures(self, istart=None, istop=None):
        """
        yields successively the Camera_pan images.
        If provided, starts at frame istart and stops at istop otherwise yields
        the full frame range
        """
        if istart is None:
            istart = 0
        if istop is None:
            istop = self.nframe

        frame_iterable = range(istart, istop)
        fig_cache = {}
        awaited = istart
        max_workers = os.cpu_count() - 1 # Leave one CPU for encoding

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers
        ) as threadpool:
            futures = (
                threadpool.submit(self.async_picture, iframe)
                for iframe in frame_iterable
            )

            for fut in concurrent.futures.as_completed(futures):
                iframe, fig = fut.result()
                fig_cache[iframe] = fig

                # Flushing the cache to the video maker
                can_flush = (awaited in fig_cache.keys())
                while can_flush:
                    yield fig_cache[awaited]
                    del fig_cache[awaited]
                    awaited += 1
                    can_flush = (awaited in fig_cache.keys())


class Camera_pan(Camera_move):

    def __init__(self, db, x_evol, y_evol, dx_evol):
        """
        Parameters:
        ----------
        db: fs.db.Db
            The underlying database
        x: couple of 1d array-like - (x_t, x)
            trajectory of x in db screen coordinates
        y: couple of 1d array-like - (y_t, y)
            trajectory of y in db screen coordinates
        dx: couple of 1d array-like - (dx_t, dx)
            trajectory of dx in db screen coordinates
        """
        x_t, x = x_evol
        y_t, y = y_evol
        dx_t, dx = dx_evol

        x_t = np.asarray(x_t)
        y_t = np.asarray(y_t)
        dx_t = np.asarray(dx_t)
        x = np.asarray(x)
        y = np.asarray(y)
        dx = np.asarray(dx)

        tmin = x_t[0]
        tmax = x_t[-1]
        if (y_t[0] != tmin) or (y_t[-1] != tmax):
            raise ValueError("Unexpected t span for y_t")
        if (dx_t[0] != tmin) or (dx_t[-1] != tmax):
            raise ValueError("Unexpected t span for dx_t")

        super().__init__(db, tmin, tmax)

        self.x_func = PchipInterpolator(x_t, x)
        self.y_func = PchipInterpolator(y_t, y)
        self.dx_func = PchipInterpolator(dx_t, dx)


    def get_frame(self, iframe):
        """ Parameter for the ith-frame frame to be interpolated """
        t_frame = self.tmin + self.dt * iframe / self.nframe
        frame_x = self.x_func(t_frame)
        frame_y = self.y_func(t_frame)
        frame_dx = self.dx_func(t_frame)

        return fs.db.Frame(
            x=frame_x,
            y=frame_y,
            dx=frame_dx,
            nx=self.nx,
            xy_ratio=self.xy_ratio,
            t=t_frame,
            plotting_modifier=self.plotting_modifier
        )
    
    def make_grids(self):
        # No added value, faster to recompute
        pass


class Camera_zoom(Camera_move):
    
    def __init__(self, db, h_evol):
        """
        Continuous zomming, to use in conjonction with a fs.db.Exp_Db in order
        to unwrap a pre-stored Exponential mapping.

        Parameters:
        ----------
        db: fs.db.Exp_Db
            The underlying database
        h_evol: couple of 1d array-like - (h_t, h)
            Trajectory of zoom logarithmic factor h
            The screen is scaled by np.exp(h)
        """
        if not(isinstance(db, fs.db.Exp_db)):
            raise ValueError("Camera_zoom shall be used with fs.db.Exp_Db")

        h_t, h = h_evol
        h_t = np.asarray(h_t)
        h = np.asarray(h)
        tmin = h_t[0]
        tmax = h_t[-1]
        super().__init__(db, tmin, tmax)

        self.h_func = PchipInterpolator(h_t, h)


    def get_frame(self, iframe):
        """ Parameter for the ith-frame frame to be interpolated """
        t_frame = self.tmin + self.dt * iframe / self.nframe
        frame_h = self.h_func(t_frame)
        return fs.db.Exp_frame(
            h=frame_h,
            nx=self.nx,
            xy_ratio=self.xy_ratio,
            pts=self.pts
        )

    def make_grids(self):
        self.pts = fs.db.Exp_frame.make_exp_grid(
            self.nx, self.xy_ratio
        )

class Camera_custom(Camera_move):

    def __init__(self, db, tmin, tmax, zoom_evol):
        """
        A Camera move for which each frame is computed from scratch from the
        zoom inputs

        Parameters:
        ----------
        db: fs.db.Custom_db
            The underlying database
        zoom_evol: function float -> dict
            Trajectory of zoom_kwargs
        """
        super().__init__(db, tmin, tmax)


    def get_frame(self, iframe):
        """  Get the zoom for this frame and compute it """
