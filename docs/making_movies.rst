Making movies
*************

Fractalshades defines several classes to help the user making fractal movies.
You can refer to the documentation of the movie API: :doc:`/API/movie`.

The purpose of this section is to give more practical details on the use
of the movie-making API.
The main idea is to create a movie class ` fractalshades.movie.Movie` and
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

  - :download:`the template <movies/pan_movie/M2_plotter.py>` (usually
    created from the GUI movie template tool)
  - :download:`the movie-making script <movies/pan_movie/pan_script.py>`
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

  - :download:`the template <movies/zoom_movie/M2_plotter.py>` (usually
    created from the GUI movie template tool)
  - :download:`the movie-making script <movies/zoom_movie/zoom_script.py>`
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

  - :download:`the template <movies/zoom_movie/M2_plotter.py>` (usually
    created from the GUI movie template tool)
  - :download:`the movie-making script <movies/zoom_movie/zoom_script.py>`
    (to be adapted for the specific wanted sequence)

Run the *script* ; it will import the template at runtime and use it
to create a db and make the movie itself.

A low-quality export of this video is available at:

..  youtube:: M8wsrGuj7QU

