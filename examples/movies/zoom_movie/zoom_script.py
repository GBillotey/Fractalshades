# -*- coding: utf-8 -*-
"""
=============================
Movie making: a zoom template
=============================
A demonstration for the zoom movie tool, which uses a precomputed database
featuring an exponentional mapping ``fractalshades.db.Exp_db`` to generate
frames.
"""
import os

import numpy as np
import mpmath

import fractalshades as fs
import fractalshades.colors
import fractalshades.movie
import fractalshades.db
import fractalshades.projection

from M2_plotter_zoom import get_plotter, plot_kwargs


fs.settings.enable_multithreading = True

def make_movie(plot_dir):

    movie_nx = 720       # width of the movie frame in pixels
    movie_ny = 480       # width of the movie frame in pixels
    movie_fps = 60
    ratio = movie_nx / movie_ny

    unzoomed_width = 10. # width of the "unzoomed" image (in complex plane)
    zoomed_width = 7.60e-20
    h_view = np.log(unzoomed_width) - np.log(zoomed_width)
    h = h_view + 0.5 * np.log(1. + ratio) + 0.1


    # Expmap database size
    nt = 3 * movie_nx # ~ equal to the number of pixels along a circonference
    nh = int(nt / (2 * np.pi) * h + 0.5)

    # Creates the expmap database
    plot_kwargs["recovery_mode"] = True
    plot_kwargs["nx"] = nh
    plot_kwargs["final_render"] = True
    plot_kwargs["supersampling"] = "2x2"
    plot_kwargs["batch_params"] = {
        "projection": fs.projection.Expmap(hmin=0., hmax=h, rotates_df=False, orientation="vertical")
    }
    plotter, layer_name = get_plotter(**plot_kwargs)

    expdb_path = plotter.save_db(
        relpath="expmap.postdb",
        postdb_layer=layer_name,
        recovery_mode=True
    )

    # Debuging: plot the unwrapped expmap
    test_expmap_plot = False
    if test_expmap_plot:
        expdb = fs.db.Db(expdb_path)
        img = expdb.plot()
        img.save(os.path.join(
            os.path.dirname(expdb.path),
            "expmap_full2.png"
        ))

    # Make the 'final plot' database
    plot_kwargs["recovery_mode"] = True
    plot_kwargs["dx"] = str(mpmath.mpf(plot_kwargs["dx"]) * 2.0)
    plot_kwargs["nx"] = movie_nx * 2
    plot_kwargs["xy_ratio"] = 1.0
    plot_kwargs["final_render"] = True
    plot_kwargs["supersampling"] = "3x3"
    plot_kwargs["batch_params"] = {
        "projection": fs.projection.Cartesian()
    }
    plotter, layer_name =  get_plotter(**plot_kwargs)

    finaldb_path = plotter.save_db(
        relpath="final.postdb",
        postdb_layer=layer_name,
        recovery_mode=True
    )

    # Make a test final plot
    test_final_plot = True
    if test_final_plot:
        finaldb = fs.db.Db(finaldb_path)
        img = finaldb.plot()
        img.save(os.path.join(
            os.path.dirname(finaldb.path),
            "final_full2.png"
        ))

    # Build the Exp_db database "movie db"
    movie_db = fs.db.Exp_db(expdb_path, finaldb_path)
    movie = fs.movie.Movie(size=(movie_nx, movie_ny), fps=movie_fps)

    # Zoom during 10 seconds
    t = [0., 2.5, 32.5, 35.]
    h = [h_view, h_view, 0.0, 0.0]  # np.log(1.e20) == 46.05


    movie.add_sequence(
        fs.movie.Camera_zoom(movie_db, (t, h))
    )

    # debugging: plotting the subsampling arrays 
    test_plot_ss = False
    if test_plot_ss:
        for kind in ("exp", "final"):
            lvl = movie_db.ss_lvl_count(kind)
            for ilvl in range(lvl):
                img_path = os.path.join(
                    plot_dir, "subsampled", f"{kind}_ss{ilvl}.png"
                )
                im = movie_db.ss_img(kind, ilvl)
                fs.utils.mkdir_p(os.path.dirname(img_path))
                im.save(img_path)

    # Making the movie ad plotting a few test frames
    movie.make(os.path.join(plot_dir, "zoom.mp4"))
    movie.export_frame(os.path.join(plot_dir, "frames"), time=30.)


if __name__ == "__main__":
    realpath = os.path.realpath(__file__)
    plot_dir = os.path.splitext(realpath)[0]
    make_movie(plot_dir)

