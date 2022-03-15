Overview
********

Fractalshades is a library for creating static and interactive visualizations 
of fractals in Python. It targets primarily the Mandelbrot_ and Burning_Ship_
sets, for which it allows ultra-deep zooming (*1.e-2000* scale and beyong)
thanks to state-of-the-art algorithm (based on perturbation technique and
chained bilinear approximations).

.. _Mandelbrot: https://en.wikipedia.org/wiki/Mandelbrot_set
.. _Burning_Ship: https://en.wikipedia.org/wiki/Burning_Ship_fractal

.. figure:: /_static/gaia.jpg

   A deep Mandelbrot zoom (credit: see
   :doc:`examples/batch_mode/11-run_perturbdeep`)

The time-critical core-loops are run in parallel on CPU and leverage
just-in-time compiling through numba.
Arbitrary precision calculations rely on a dedicated MPFR C-extension compiled
with Cython (Windows & Linux OS).

The GUI is implemented with PyQt.

The main drivers for this hobby project has been the mathematical interest
and the aesthetics, hence the post-processing that reveals the structure of
the mathematical objects have been priviledged.

