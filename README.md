# Fractal-shades

The aim of this program is to generate a few aesthetically pleasing fractal images.

## Prerequisites

This program is written in python (python 2.7, I should have switched to python3 some time ago...) and relies on the following dependencies: numpy, matplotlib, Pillow.
(matplotlib is used for convenience for some extra outputs but is not a core dependency.)

## Getting started

Have a look first at the small gallery included (see the directory *Gallery* or the section below) ; the code used to generate these images is provided.
To generate one of these example simply launch one of :

   - run_mandelbrot.py
   - run_power_tower.py
   - run_nova.py
   
This files are intended as working examples and you should be able to easily adapt the zoom parameters or color scheme.

## Program architecture

fractal.py  is the main low-level part, defining 4 classes

  - Color_tools for color operation in CIE color space (shading, color gradient, ...)
  - Fractal_colormap maps floats to colors
  - Fractal_plotter plots a fractal object
  - Fractal abstract base class encapsulating a fractal object, to be subclassed

The base class Fractal is subclassed in the following 3 files : 

   - Classical_mandelbrot for the famous iteration `z(n+1) <- zn**2 +c`  (classical_mandelbrot.py)
   - Power_tower for `z(n+1) <- c**zn` (Power_tower.py)
   - Nova.py for `f(z) = z**p - 1 and z(n+1) <- zn - R * f / dfdz + c` (Nova.py), a fractal  familly derived from Newton fractals.

The 3 example scripts already discussed rely on those 3 classes.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
I am not a computer engineer so I guess a lot of basic "software" things could be improved.

## License
BSD-3

## Gallery

### From "classical" Mandelbrot set
![Screenshot](gallery/billard.jpg)
![Screenshot](gallery/medaillon.jpg)
![Screenshot](gallery/Emerauld_shield.jpg)

### From "Nova 6" Mandelbrot set
![Screenshot](gallery/Nova6_whole_set.jpg)
![Screenshot](gallery/Nova6_zoom.jpg)

### From "Power Tower" Mandelbrot set
![Screenshot](gallery/Power_tower_Ankh.jpg)
