# -*- coding: utf-8 -*-
import inspect
import copy
import os
#import enum

import numpy as np
import numba
import mpmath
import PIL
from numpy.lib.format import open_memmap

import fractalshades as fs
import fractalshades.settings
import fractalshades.utils
import fractalshades.numpy_utils.xrange as fsx




class Db:
    def __init__(self, path, zoom_kw):
        """ Small wrapper around the raw array data,
        adding localisation info to the datapoints through the zoom kwargs """
        self.path = path
        self.zoom_kw = zoom_kw


class _Plot_template:
    def __init__(self, plotter, arr_hook):
        """ A plotter with a pluggable get_2d_arr method
        -> Basically, an interface to avoid monkey-patching `get_2d_arr`
        """
        plotter = copy.deepcopy(plotter)
        self._plotter = plotter
        self.arr_hook = arr_hook

    def __getattr__(self, attr):
        return getattr(self._plotter, attr)

    def get_2d_arr(self, post_index, chunk_slice):
        return self.arr_hook(post_index, chunk_slice)


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
        db: Db instance
            the stored db, will be treated as cartesian.
        """
        self.plot_template = _Plot_template(
                copy.deepcopy(plotter), self.arr_hook
        )
        # Reparenting the layers to _Plot_template
        for layer in self.plot_template.layers:
            layer.link_plotter(self.plot_template)

        self.db = db
        self.plot_dir = plot_dir


    def arr_hook(self, post_index, chunk_slice):
        """
        Forwards the mmap data to the layers
        """
        mmap = open_memmap(filename=self.db.path, mode="r+")
        ret = mmap[post_index, :, :]
        return ret


    @property
    def layers(self):
        return self.plot_template._plotter.layers


    def plot_size(self, pts):
        if pts == "raw":
            nx = self.db.zoom_kw["nx"]
            xy_ratio = self.db.zoom_kw["xy_ratio"]
            ny = int(nx / xy_ratio + 0.5)
            return nx, ny
        else:
            (pts_x, pts_y) = pts
            return np.shape(pts_x)
            

    def plot(self, pts="raw", out_mode="file"):
        """
        Parameters:
        -----------
        pts: "raw", (pts_x, pts_y)
            If "raw", this simply the raw pts from the db
            If (pts_x, pts_y), this is the x and y coordinates of whare to 
            interpolate, passed as 2 2d-arrays (grid) in window coordinates
        out_mode: "file", "data"
            If file, will create a png file
            If "data", will return the PIL.Image object for further processing
            (movie making tool, ...)
        """
        _im = []
        for layer in self.plot_template.layers: # This is needed even if no file out
            if layer.output:
                _im += [
                    PIL.Image.new(mode=layer.mode, size=self.plot_size(pts))
                ]
            else:
                _im += [None]

        if pts == "raw":
            self.process_raw(_im)
        else:
            self.process_interpolated(pts, _im)

        return self.output(out_mode, _im)
        
    
    def output(self, out_mode, _im):
        """ Saves as file or return the img array for further processing
        """
        if out_mode == "file":
            for i, layer in enumerate(self.plot_template.layers):
                if not(layer.output):
                    continue
                file_name = self.plot_template.image_name(layer)
                base_img_path = os.path.join(self.plot_dir, file_name + ".png")
                fs.utils.mkdir_p(os.path.dirname(base_img_path))
                _im[i].save(base_img_path)

        elif out_mode == "data":
            raise NotImplementedError()


    def process_raw(self, _im):
        """
        just plot the raw db data - 1 db point == 1 pixel
        """
        for i, layer in enumerate(self.layers):
            if layer.output:
                nx, ny = self.plot_size("raw")
                chunk_slice = (0, nx, 0, ny)
                crop_slice = (0, 0, nx, ny)
                
                # Ultimately calls plotter.get_2d_arr
                paste_crop = layer.crop(chunk_slice)
                _im[i].paste(paste_crop, box=crop_slice)


    def process_interpolated(self, pts_grid, _im):
            raise NotImplementedError()


class Exmap_db_loader(Db_loader):

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
    
    

            
            
            
class Grid():
    
    def __init(x_val, y_val, periodic=(False, False)):
        pass
    
 
            
            
            
            
            
            
            


