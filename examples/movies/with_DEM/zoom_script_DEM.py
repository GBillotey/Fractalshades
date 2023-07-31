# -*- coding: utf-8 -*-
"""
==========================================================
Movie making: a DEM coloring template for zooming sequence
=========================================================
Demonstates the use of the `expmap_seam` keyword for
``fractalshades.projection.Cartesian`` projection, needed for seamless
DEM (Distance Estimation) coloring
"""
import os

import numpy as np
import mpmath

import fractalshades as fs
import fractalshades.colors
import fractalshades.movie
import fractalshades.db
import fractalshades.projection

from perpendicular_BS import get_plotter, plot_kwargs


fs.settings.enable_multithreading = True

def make_movie(plot_dir):
    fs.settings.newton_zoom_level = 100. # 1.e-5
    fs.settings.std_zoom_level: float = 100. #1.e-8

    movie_nx = 1280       # width of the movie frame in pixels
    movie_ny = 720       # height of the movie frame in pixels
    movie_fps = 60       # frames per second
    ratio = movie_nx / movie_ny


    unzoomed_width = 10. # width of the "unzoomed" image (in complex plane)
    zoomed_width = "7.032184999234219e-55" #7.60e-20
    h_view = float(mpmath.log(unzoomed_width) - mpmath.log(zoomed_width))
    h = h_view + 0.5 * np.log(1. + ratio) + 0.1
    
    print("h", h) #  116.74631940708427

    # Expmap database size
    nt = 3 * movie_nx # ~ equal to the number of pixels along a circonference
    nh = int(nt / (2 * np.pi) * h + 0.5)

    # Creates the expmap database
    plot_kwargs["x"] = '-1.929319698524937920226708049698305350754670432084006734339806946'
    plot_kwargs["y"] = '-0.0000000000000000007592779387989739090287550144163328879329853232537252481600401185'
    plot_kwargs["dx"] = zoomed_width
    plot_kwargs["dps"] = 70
    plot_kwargs["max_iter"] = 20000
    
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
    # del plot_kwargs["fractal"]._calc_data
    plot_kwargs["recovery_mode"] = True
    plot_kwargs["dx"] = str(mpmath.mpf(plot_kwargs["dx"]) * 2.0)
    plot_kwargs["nx"] = movie_nx * 2
    plot_kwargs["xy_ratio"] = 1.0
    plot_kwargs["final_render"] = True
    plot_kwargs["supersampling"] = "3x3"
    plot_kwargs["batch_params"] = {
        "projection": fs.projection.Cartesian(expmap_seam=1.0)
    }
    plotter, layer_name = get_plotter(**plot_kwargs)


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

    # Zoom during 30 seconds
    t = [0., 2.5, 122.5, 125.]
    h = [h_view, h_view, 0.0, 0.0]


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
    movie.export_frame(os.path.join(plot_dir, "frames"), time=31.)


if __name__ == "__main__":
    realpath = os.path.realpath(__file__)
    plot_dir = os.path.splitext(realpath)[0]
    make_movie(plot_dir)

