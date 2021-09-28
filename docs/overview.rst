Overview
********

Fractalshades is a library for creating static and interactive visualizations 
or fractals in Python. It targets primarily the escape-time fractals, a wide 
class of fractals among which the most famous is probably the Mandelbrot_ set.

.. _Mandelbrot: https://en.wikipedia.org/wiki/Mandelbrot_set

Fractalshades aims at:

  - providing built-in formulas for most common escape-time fractal, which 
    let you produce beautiful images with just a few lines of code

  - for more advanced users, providing easy ways of implementing custom 
    formulas or coloring options, which will then benefit of the same 
    time-optimisation and graphical user interface.

Some fractals are only available at double precision, while some also allow
ultra-deep zooming (1e-300 scale and beyong) thanks to state-of-the-art
algorithm (based on perturbation technique).

The time-critical core-loops benefit from multiprocesssing and leverage
just-in-time compiling through numba.



