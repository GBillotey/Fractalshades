# -*- coding: utf-8 -*-
"""
===============================
Movie making: a custom template
===============================
Demonstrate the use of successive sequences in the movie making tool. 
"""
import os
import copy

import numpy as np
import mpmath

import fractalshades as fs
import fractalshades.colors
import fractalshades.movie
import fractalshades.db
import fractalshades.projection

from custom_zoom_template import get_plotter, plot_kwargs

fs.settings.enable_multithreading = True


def make_movie(plot_dir):
    
    plotter, layer_name = get_plotter(**plot_kwargs)
    
    movie_nx = 720       # width of the movie frame in pixels
    movie_ny = 480       # width of the movie frame in pixels
    ratio = movie_nx / movie_ny

    unzoomed_width = 10. # width of the "unzoomed" image (in complex plane)
    skew_width = 1.9962495749473e-5
    zoomed_width = 1.9962495749473e-36

    # zoom sequence 1
    h_view1 = np.log(unzoomed_width) - np.log(skew_width)
    h_db1 = h_view1 + 0.5 * np.log(1. + ratio) + 0.1

    # zoom sequence 2
    h_view2 = np.log(skew_width) - np.log(zoomed_width)
    h_db2 = h_view2 + 0.5 * np.log(1. + ratio) + 0.1
    

    # final skew
    s00 = plot_kwargs["skew_00"]
    s01 = plot_kwargs["skew_01"]
    s10 = plot_kwargs["skew_10"]
    s11 = plot_kwargs["skew_11"]
    t_skew_transition = 2.5
    t_start_skew = 7.5

    def plotter_from_time(time):
        """ A modifier for the zoom skew angle """
        plot_kwargs_t = copy.copy(plot_kwargs)
        plot_kwargs_t["dx"] = skew_width * 2
        plot_kwargs_t["nx"] = movie_nx
        plot_kwargs_t["xy_ratio"] = ratio
        plot_kwargs_t["final_render"] = True
        plot_kwargs_t["supersampling"] =  "2x2"

        t_frac = (time - t_start_skew) / t_skew_transition
        plot_kwargs_t["has_skew"] = True
        plot_kwargs_t["skew_00"] = s00 * t_frac + 1. * (1. - t_frac)
        plot_kwargs_t["skew_01"] = s01 * t_frac
        plot_kwargs_t["skew_10"] = s10 * t_frac
        plot_kwargs_t["skew_11"] = s11 * t_frac + 1. * (1. - t_frac)

        return get_plotter(**plot_kwargs_t)

    #==========================================================================
    # Precomputing Expmap database 1 (first zoom sequence)
    plot_kwargs_expmap1 = copy.copy(plot_kwargs)
    nt = 3 * movie_nx # ~ equal to the number of pixels along a circonference
    nh = int(nt / (2 * np.pi) * h_db1 + 0.5)

    # Creates the expmap database
    plot_kwargs_expmap1["dx"] = str(skew_width)
    plot_kwargs_expmap1["has_skew"] = False
    plot_kwargs_expmap1["recovery_mode"] = True
    plot_kwargs_expmap1["nx"] = nh
    plot_kwargs_expmap1["final_render"] = True
    plot_kwargs_expmap1["supersampling"] = "2x2"
    plot_kwargs_expmap1["batch_params"] = {
        "projection": fs.projection.Expmap(hmin=0., hmax=h_db1, rotates_df=False, orientation="vertical")
    }
    plotter, layer = get_plotter(**plot_kwargs_expmap1)
    expdb_path1 = plotter.save_db(
        relpath="expmap1.postdb",
        postdb_layer=layer_name,
        recovery_mode=True
    )

    # Debuging: plot the unwrapped expmap
    test_expmap_plot = True
    if test_expmap_plot:
        expdb1 = fs.db.Db(expdb_path1)
        img = expdb1.plot()
        img.save(os.path.join(
            os.path.dirname(expdb1.path),
            "expmap_full1.png"
        ))
        print("saved to", os.path.join(os.path.dirname(expdb1.path), "expmap_full1.png"))
    


    # Creates the 'final plot' database1
    plot_kwargs_finalmap1 = copy.copy(plot_kwargs)
    plot_kwargs_finalmap1["has_skew"] = False
    plot_kwargs_finalmap1["recovery_mode"] = True
    plot_kwargs_finalmap1["dx"] = str(mpmath.mpf(plot_kwargs_expmap1["dx"]) * 2.0)
    plot_kwargs_finalmap1["nx"] = movie_nx * 2
    plot_kwargs_finalmap1["xy_ratio"] = 1.0
    plot_kwargs_finalmap1["final_render"] = True
    plot_kwargs_finalmap1["supersampling"] = "2x2"
    plot_kwargs_finalmap1["batch_params"] = {
        "projection": fs.projection.Cartesian()
    }
    plotter, layer_name =  get_plotter(**plot_kwargs_finalmap1)

    finaldb_path1 = plotter.save_db(
        relpath="final1.postdb",
        postdb_layer=layer_name,
        recovery_mode=True
    )

    # Build the Exp_db database "zoom_seq_db1"
    zoom_seq_db1 = fs.db.Exp_db(expdb_path1, finaldb_path1)

    #==========================================================================
    # Precomputing Expmap database 2 (second zoom sequence)
    plot_kwargs_expmap2 = copy.copy(plot_kwargs)
    nt = 3 * movie_nx # ~ equal to the number of pixels along a circonference
    nh = int(nt / (2 * np.pi) * h_db2 + 0.5)

    # Creates the expmap database
    plot_kwargs_expmap2["dx"] = "1.9962495749473e-36"
    plot_kwargs_expmap2["has_skew"] = True
    plot_kwargs_expmap2["recovery_mode"] = True
    plot_kwargs_expmap2["nx"] = nh
    plot_kwargs_expmap2["final_render"] = True
    plot_kwargs_expmap2["supersampling"] = "2x2"
    plot_kwargs_expmap2["batch_params"] = {
        "projection": fs.projection.Expmap(hmin=0., hmax=h_db2, rotates_df=False, orientation="vertical")
    }
    plotter, layer = get_plotter(**plot_kwargs_expmap2)
    expdb_path2 = plotter.save_db(
        relpath="expmap2.postdb",
        postdb_layer=layer_name,
        recovery_mode=True
    )

    # Make the 'final plot' database1
    plot_kwargs_finalmap2 = copy.copy(plot_kwargs)
    plot_kwargs_finalmap1["has_skew"] = True
    plot_kwargs_finalmap2["recovery_mode"] = True
    plot_kwargs_finalmap2["dx"] = str(mpmath.mpf(plot_kwargs_expmap2["dx"]) * 2.0)
    plot_kwargs_finalmap2["nx"] = movie_nx * 2
    plot_kwargs_finalmap2["xy_ratio"] = 1.0
    plot_kwargs_finalmap2["final_render"] = True
    plot_kwargs_finalmap2["supersampling"] = "3x3"
    plot_kwargs_finalmap2["batch_params"] = {
        "projection": fs.projection.Cartesian()
    }
    plotter, layer_name =  get_plotter(**plot_kwargs_finalmap2)

    finaldb_path2 = plotter.save_db(
        relpath="final2.postdb",
        postdb_layer=layer_name,
        recovery_mode=True
    )

    # Build the Exp_db database "zoom_seq_db1"
    zoom_seq_db2 = fs.db.Exp_db(expdb_path2, finaldb_path2)


    #==========================================================================
    # Making the movie
    movie = fs.movie.Movie(size=(movie_nx, movie_ny), fps=30)

    # SEQUENCE1: Create a zoom sequence - zoom during 15 seconds
    t1 = [0., 2.5, t_start_skew - 0.1, t_start_skew]
    h1 = [h_view1, h_view1, 0.0, 0.0]
    movie.add_sequence(
        fs.movie.Camera_zoom(zoom_seq_db1, (t1, h1))
    )

    # SEQUENCE2: Create the plotter for a Custom Sequence which will modify
    # the skew
    custom_kw = plot_kwargs
    custom_kw["dx"] = "1.9962495749473e-35"
    custom_kw["nx"] = movie_nx
    custom_kw["xy_ratio"] = ratio
    plotter, layer = get_plotter(**custom_kw)
    movie.add_sequence(
        fs.movie.Custom_sequence(
            t_start_skew, t_start_skew + t_skew_transition, plotter_from_time,
            scratch_name="skew_tmp")
    )

    # SEQUENCE3: Create a zoom sequence - Zoom during 2 seconds
#    t_skew_transition = 0. # DEBUG
    t2 = [0., 0.1, 17.5, 20.]
    t2 = [tloc + t_skew_transition + t_start_skew for tloc in t2]
    h2 = [h_view2, h_view2, 0.0, 0.0]
    movie.add_sequence(
        fs.movie.Camera_zoom(zoom_seq_db2, (t2, h2))
    )

    # Making the movie and plotting a few test frames
    movie.make(os.path.join(plot_dir, "custom.mp4"))
    # movie.export_frames(os.path.join(plot_dir, "frames"), 28, 30)


if __name__ == "__main__":
    realpath = os.path.realpath(__file__)
    plot_dir = os.path.splitext(realpath)[0]
    make_movie(plot_dir)

