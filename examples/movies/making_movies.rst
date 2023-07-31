Making movies
*************

Fractalshades defines several classes to help the user making fractal movies.
You can refer to the documentation of the movie API: :doc:`/API/movie`.
The purpose of this section is to give more practical details on the use
of the movie-making API.

The main idea is to start from a movie class ` fractalshades.movie.Movie` and
populate is with one or several sequences. Some available sequence kinds are:

 -  `fractalshades.movie.Camera_pan` : a specialised pan sequence (implementing
    efficient interpolation in a Catesian database)
 -  `fractalshades.movie.Camera_zoom` : a specialised zooming sequence
    (implementing efficient interpolation in an Exponentially mapped database)
 -  `fractalshades.movie.Custom_sequence` : a flexible sequence which will
    compute each frame on the fly.

The sequence will take as input either a plotting object 
(`fractalshades.Fractal_plotter`) or a database object (`fractalshades.db.Db`).
The GUI tool allows to create a *template* script to generate a plotter object
for any image created by the GUI: see the *Movie* section of the main menu.

This plotter *template* can then be imported from the main movie-making
*script* (to be provied by the user). Several examples are provided below for
guidance.


Example: pan sequence
~~~~~~~~~~~~~~~~~~~~~

Download the following scripts in the same directory:

  - :download:`the template
    <../examples/movies/pan_movie/M2_plotter.py>` (usually
    created from the GUI movie template tool)

  - :download:`the movie-making script
    <../examples/movies/pan_movie/pan_script.py>`
    (to be adapted for the specific wanted sequence)

Run the *script* ; it will import the template at runtime and use it
to create a db and make the movie itself.

Note: setting `postdb` parameter to ``False`` will trigger additionnal
plotting effects, however at the expense of a much longer calculation
and increased disk storage.
A low-quality export of this video is available at:

..  youtube:: 1uOVqOETkps


Example: zoom sequence
~~~~~~~~~~~~~~~~~~~~~~

Download the following scripts in the same directory:

  - :download:`the template
    <../examples/movies/zoom_movie/M2_plotter_zoom.py>` (usually
    created from the GUI movie template tool)

  - :download:`the movie-making script
    <../examples/movies/zoom_movie/zoom_script.py>`
    (to be adapted for the specific wanted sequence)

Run the *script* ; it will import the template at runtime and use it
to create a db and make the movie itself.

A low-quality export of this video is available at:

..  youtube:: cJxNg8bmKtE 


Example: "Custom" zoom sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This movie is composed of 2 successive zoom sequences, and a "custom"
sequence is used to generate a skew transition at an intermediate depth.
Download the following scripts in the same directory:

  - :download:`the template
    <../examples/movies/custom_movie/custom_zoom_template.py>` (usually
    created from the GUI movie template tool)

  - :download:`the movie-making script
    <../examples/movies/custom_movie/custom.py>`
    (to be adapted for the specific wanted sequence)

Run the *script* ; it will import the template at runtime and use it
to create a db and make the movie itself.
A low-quality export of this video is available at:

..  youtube:: M8wsrGuj7QU

Example: Using Distance Estimation plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This zoom sequence uses colors based on distance estimation. In order to
ensure a smooth "seam" between the exponential mapping and the final
(cartesian) image be sure to use the `expmap_seam` option for this kind
of postprocessing. This option can safely be used for all postprocessing
types but has a noticeable effect only for a few:

.. code-block:: python

    plot_kwargs["batch_params"] = {
        "projection": fs.projection.Cartesian(expmap_seam=1.0)
    }


Download:

  - :download:`the template
    <../examples/movies/with_DEM/perpendicular_BS.py>` (usually
    created from the GUI movie template tool)

  - :download:`the movie-making script
    <../examples/movies/with_DEM/zoom_script_DEM.py>`
    (to be adapted for the specific wanted sequence)

Run the *script* ; it will import the template at runtime and use it
to create a db and make the movie itself.
A low-quality export of this video is available at:

..  youtube:: TlfVj7K6YSg


