# -*- coding: utf-8 -*-
"""
============================
Movie: writing the template
============================
"""
import os
import sys

import fractalshades as fs
import fractalshades.colors
import fractalshades.movie
import fractalshades.db

from M2_plotter_bis import get_plotter, plot_kwargs


def make_movie(plot_dir):

    plot_kwargs["recovery_mode"] = True
    plot_kwargs["nx"] = 4000

    plotter, layer_name =  get_plotter(**plot_kwargs)
    db_path = plotter.save_db()
    db = fs.db.Db(db_path)

    db.freeze(plotter, layer_name, try_reload=True)


    movie = fs.movie.Movie(plotter, layer_name, size=(1920, 1080), fps=30, supersampling=False) # 1920, 1080 | 720, 480 | 1280, 720
    # Pan in 10 seconds
    t = [0., 10.]
    x = [-.25, .25]
    y = [0., 0.]
    dx = [0.5, 0.5]
    movie.add_camera_move(
        fs.movie.Camera_pan(db, t, x, y, dx)
    )

    print("plot DIR", plot_dir)
    movie.make(os.path.join(plot_dir, "test.mp4"))
    # movie.debug(os.path.join(plot_dir, "debug2"), 0, 2)


if __name__ == "__main__":
    realpath = os.path.realpath(__file__)
    plot_dir = os.path.splitext(realpath)[0]
    make_movie(plot_dir)

