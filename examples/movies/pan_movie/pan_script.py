# -*- coding: utf-8 -*-
"""
============================
Movie making: a pan template
============================
A demonstration for the pan movie tool, which uses a precomputed cartesian
database ``fractalshades.db.Db`` to generate frames.
"""
import os

import fractalshades as fs
import fractalshades.colors
import fractalshades.movie
import fractalshades.db

from M2_plotter import get_plotter, plot_kwargs


# Pan in 10 seconds
t = [0., 10.]
x = [-.25, .25]
y = [0., 0.]
dx = [0.5, 0.5]

postdb = False
recovery_mode = True

# Movie frame size
movie_size = (720, 480)
movie_nx, movie_ny = movie_size

def plotter_modifier(plotter, time):
    """ A modifier for the lighting angle """
    main_layer = plotter['continuous_iter']
#    main_layer.probe_z += 6 * time
    a, b = main_layer._twin_field
    main_layer._twin_field = a, b * 0.5 * (5. - abs(5. - time)) 
    lighting = main_layer._modifiers[0][2]

    ls0 = lighting.light_sources[0]
    ls0['polar_angle'] = 36. * time


def make_movie(plot_dir):

    # Sets the calculation db pixel to match the movie frame
    plot_kwargs["nx"] = movie_nx * 2
    plot_kwargs["xy_ratio"] = movie_nx / movie_ny * 2.0
    plotter, layer_name =  get_plotter(**plot_kwargs)

    if postdb:
        db_path = plotter.save_db(
                postdb_layer=layer_name,
                recovery_mode=recovery_mode
        )
        db = fs.db.Db(db_path)
        pan_sequence = fs.movie.Camera_pan(db, (t, x), (t, y), (t, dx))
    else:
        db_path = plotter.save_db(
                recovery_mode=recovery_mode
        )
        db = fs.db.Db(db_path)
        db.set_plotter(plotter=plotter, postname=layer_name)
        pan_sequence = fs.movie.Camera_pan(
                db, (t, x), (t, y), (t, dx),
                plotting_modifier=plotter_modifier
        )

    movie = fs.movie.Movie(size=movie_size, fps=60)
    movie.add_sequence(pan_sequence)
    movie.make(os.path.join(plot_dir, "pan.mp4"), crf=20)
    movie.export_frame(os.path.join(plot_dir, "frames"), 30)
    movie.export_frame(os.path.join(plot_dir, "frames"), 32)

if __name__ == "__main__":
    plot_dir = os.path.splitext(os.path.realpath(__file__))[0]
    make_movie(plot_dir)

