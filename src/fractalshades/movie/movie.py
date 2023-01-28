# -*- coding: utf-8 -*-
#import copy
import logging
import os

import numpy as np
#import numba

import fractalshades as fs
import fractalshades.db
#import fractalshades.settings
#import fractalshades.colors
import fractalshades.utils
#import fractalshades.postproc
#import fractalshades.projection

#from fractalshades.lib.fast_interp import interp1d


try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    raise RuntimeError("Scipy is needed for movie maker- please install with: "
                       "'pip install scipy'")
# Note: Scipy used for monotonic cubic interpolant, we could remove this 
# dependency with reimplementing:
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
                 supersampling=None):
        """
        A movie-making class
        
        Parameters:
        -----------
        plotter: fs.Fractal_plotter
            The base plotter for this movie
        postname: str
            The string indentifier for the layer used to make the movie
        size: (int, int)
            movie screen size in pixels. Default to 1920 x 1080 
        fps: int
            movie frame-count per second

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
        self.supersampling = supersampling
        
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

    def make(self, out_file, pix_fmt="yuv444p", crf=22):
        """
        Make a mp4 movie - Codec AVC standard (H.264)

        Parameters:
        -----------
        out_file: str
            The path to output file
        pix_fmt: "yuv444p" | "yuv420p"
            The pixel format’s name - default: "yuv444p"
        crf: int, 0 - 51
            The "Constant Rate Factor" - default: 22 (51 = lowest quality)
        """
        # Credit:
# https://stackoverflow.com/questions/73609006/how-to-create-a-video-out-of-frames-without-saving-it-to-disk-using-python

        fs.utils.mkdir_p(os.path.dirname(out_file))

        with open(out_file, "wb") as f:
            # Use PyAV to open out_file as MP4 video
            # output_memory_file = io.BytesIO()
            output = av.open(f, 'w', format="mp4")
    
            # Advanced Video Coding (AVC), also referred to as H.264 or
            # MPEG-4 Part 10, is a video compression standard based on
            # block-oriented, motion-compensated coding.
            stream = output.add_stream('h264', str(self.fps))
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = pix_fmt
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
    """ 
    Time is relative to the whole movie
    Frame index is relative to this Cam move
    """
    def __init__(self, db, t):
        self.db = db
        self.t = np.asarray(t, dtype=np.float64)
        self.movie = None # add to a movie with movie.add_frame

    def pictures(self):
        raise NotImplementedError("Derived frames shall implement")

    def set_movie(self, movie):
        """
        Parameters:
        ----------
        movie: fs.movie.Movie
        """
        self.movie = movie
        self.db_loader = fs.db.Db_loader(
            movie.plotter, self.db, plot_dir=None
        )
        self.out_postname = movie.postname

        t = self.t
        self.fps = self.movie.fps
        self.nframe = int((t[-1] - t[0]) * self.fps)
        self.nx = self.movie.width
        self.xy_ratio = self.movie.width / self.movie.height
        self.supersampling = self.movie.supersampling

#    @property
#    def nframe(self):
#        """ Number of frames for this camera move"""
#        t = self.t
#        return int((t[-1] - t[0]) * self.fps)
#
#    @property
#    def fps(self):
#        """ Number of frames for this camera move"""
#        return self.movie.fps
#
#    @property
#    def nx(self):
#        """ image width in pixels """
#        return self.movie.width
#
#    @property
#    def xy_ratio(self):
#        """ frame ratio width / height """
#        return self.movie.width / self.movie.height


class Camera_pan(Camera_move):

    def __init__(self, db, t, x, y, dx):
        """
        Parameters:
        ----------
        t: 1d array-like
            time for the trajectories
        x: 1d array-like
            trajectory of x in db screen coordinates
        y: 1d array-like
            trajectory of y in db screen coordinates
        dx: 1d array-like
            trajectory of dx in db screen coordinates
        """
        super().__init__(db, t)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.dx = np.asarray(dx, dtype=np.float64)

        self.x_func = PchipInterpolator(self.t, self.x)
        self.y_func = PchipInterpolator(self.t, self.y)
        self.dx_func = PchipInterpolator(self.t, self.dx)

    def get_frame(self, iframe):
        """ Parameter for the ith-frame frame to be interpolated """
        t_frame = self.t[0] + self.t[1] * iframe / self.nframe
        frame_x = self.x_func(t_frame)
        frame_y = self.y_func(t_frame)
        frame_dx = self.dx_func(t_frame)
        return fs.db.Frame(
            x=frame_x,
            y=frame_y,
            dx=frame_dx,
            nx=self.nx,
            xy_ratio=self.xy_ratio,
            supersampling=self.supersampling
        )

    def picture(self, iframe):
        """ Returns the ith-frame as a PIL.Image object """
        return self.db_loader.plot(
            frame=self.get_frame(iframe), out_postname=self.out_postname
        )

    def pictures(self):
        for iframe in range(self.nframe):
            yield self.picture(iframe)


#------------------------------------------------------------------------------
    # Zoom in 10 seconds
#    t = [0., 10.]
#    h = [10., 0.]   # exp(10.) is ~ 22000
#    movie.add_camera_move(
#        fs.movie.Camera_zoom(db, t, h)
#    )
#    
#class Camera_zoom(Camera_move):
#    
#    def __init__(self, db, t, h):
#        """
#        Parameters:
#        ----------
#        t: 1d array-like
#            time for the trajectories
#        h: 1d array-like, > 0
#            trajectory of zoom logarithmic factor h
#            The screen is scaled by np.exp(h) 
#        """
#        super().__init__(db, t)
#        self.x = np.asarray(x, dtype=np.float64)
#        self.y = np.asarray(y, dtype=np.float64)
#        self.dx = np.asarray(dx, dtype=np.float64)
#
#        self.x_func = PchipInterpolator(self.t, self.x)
#        self.y_func = PchipInterpolator(self.t, self.y)
#        self.dx_func = PchipInterpolator(self.t, self.dx)

            
#
#class Camera_zoom(Camera_move):
#
#    def __init__(self, movie, exp_arr):
#        """
#        Based on exponential mapping
#        
#        Parameters
#        ----------
#        time_arr:
#            The time points, strictly increasing
#        exp_arr:
#            The scaling coefficients (in log scale). The image at time time_arr[i]
#            will be at zoom_args + scaling of  exp(exp_arr[i])
#            Positive numbers (>= 0.)
#        """
#        self.time_arr = np.asarray(time_arr, dtype=np.float64)
#        self.exp_arr = np.asarray(exp_arr, dtype=np.float64)
#        # Will be set when adding to a Movie_maker
#        self.mov = None
#
#    def set_movie(self, mov):
#        """ mov: Movie_maker"""
#        self.mov = mov
#        self.compute_frame_data()
#
#    @property
#    def ref_zoom_kwargs(self):
#        return self.mov.ref_zoom_kwargs
#
#    def compute_frame_data(self):
#        frame_indices = np.arange(self.nframe)
#        interp = scipy.interpolate.PchipInterpolator(
#            (self.time_arr - self.t0) * self.mov.fps,
#            self.exp_arr
#        )
#        self.frame_exp = interp(frame_indices)
#        
#
#    @property
#    def t0(self):
#        """ Time at start of the zoom """
#        return self.time_arr[0]
#
#    @property
#    def tf(self):
#        """ Time at end of the zoom """
#        return self.time_arr[-1]
#
#    @property
#    def dh(self):
#        """ The needed h range for Expmap. [0, hmax]"""
#        fexp = self.frame_exp
#        zkw = self.ref_zoom_kwargs
#        # Final image computed as a square (size dx x dx) and used 
#        # for pixels inside the dx-diameter circle.
#        # zpix ->  exp(hmin) : hmin == 0.
#        # zpix ->  exp(hmax) : hmax == np.max(fexp) * np.log(diag)
#        diag = np.sqrt((1. / zkw["xy_ratio"]) ** 2 + 1.)
#        return np.max(fexp) * np.log(diag)
#
#
#    @property
#    def nframe(self):
#        "Number of frames used for this zoom sequence"
#        return (self.tf - self.t0) * self.mov.fps
#    
#    def iframe(self, t):
#        """ The frame just before time t - index for this local zoom sequence
#        """
#        return int((t - self.t0) * self.mov.fps)
#
##    def exp_scale(self, iframe):
##        """ exp_scale at frame i """
##        t = self.to + iframe / self.mov.fps
#
#
#    def expmap_zoom_kwargs(self):
#        """ Returns the kwargs needed for the zoom of the Expmap projection """
#        # We just need to adjust the dx and to set the projection
#        zkw = copy.deepcopy(self.ref_zoom_kwargs)
#
#        # lowest expmap pixel density in the cartesian image (in pt per unit)
#        diag = np.sqrt((1. / zkw["xy_ratio"]) ** 2 + 1.)
#        ny_expmap = zkw["nx"] * np.pi * diag # Adjusting to same pix density
#        # dh = 2. * np.pi * xy_ratio
#        xy_ratio_expmap = self.dh / (2. * np.pi)
#        nx_expmap = ny_expmap * xy_ratio_expmap
#
#        # First, the dx if given thrgh the required pixel numbers fpr [-pi, pi]
#        zkw["nx"] = nx_expmap
#        zkw["projection"] = fs.projection.Expmap(
#                hmin=0., hmax=self.dh, rotates_df=False
#        )
#        return zkw
#
#
#    def final_frame_kwargs(self):
#        """ Returns the kwargs needed for the zoom of the final projection """
#        # We just need to adjust the dx and to set the projection
#        zkw = copy.deepcopy(self.ref_zoom_kwargs)
#        zkw["xy_ratio"] = 1.0
#        zkw["projection"] = fs.projection.Cartesian()
#        return zkw
#
#
#    def frame_interpolator(self, iframe):
#        """
#        return a numba compiled function numba_impl for the frame i
#        numba_impl(frame, x_pix, y_pix, data_out)
#        """
#        fps = self.fps
#        exp = self.frame_exp[iframe]
#
#        # Need to open the needed mmap range, as numba cannot use it
#        # pixels from the 2 memory mappings available
#
#
#        @numba.njit
#        def numba_impl(mmap_exp, mmap_final, data_out):
#            
#            y_pix
#
#        return numba_impl





