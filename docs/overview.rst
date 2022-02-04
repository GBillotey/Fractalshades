Overview
********

Fractalshades is a library for creating static and interactive visualizations 
or fractals in Python. It targets primarily the Mandelbrot_ set, however it 
is structured to allow easy generalisation (through subclassing) to the more 
general class of escape-time fractals.

.. _Mandelbrot: https://en.wikipedia.org/wiki/Mandelbrot_set

Fractalshades aims at:

  - providing built-in formulas for most common escape-time fractal, which 
    let you produce beautiful images with just a few lines of code

  - for more advanced users, providing easy ways of implementing custom 
    formulas or coloring options, which will then benefit of the same 
    time-optimisation and graphical user interface.

Some fractals are only available at double precision, while some also allow
ultra-deep zooming (*1.e-2000* scale and beyong) thanks to state-of-the-art
algorithm (based on perturbation technique).

The time-critical core-loops are run in parallel on CPU and leverage
just-in-time compiling through numba.
Arbitrary precision calculations use a dedicated C-extension compiled with
Cython.

The main drivers for this hobby project has been the mathematical interest
and the aesthetics, hence the post-processing that reveals the structure of
the mathematical objects have been priviledged.

