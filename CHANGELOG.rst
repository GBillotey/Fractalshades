Changelog
*********

Rev 0.5.6
~~~~~~~~~
- Added Shark Fin escape-time fractal (incl. arbitrary precision implementation)
- "Show source" export button from the GUI now gives a ready-to-run script
- Fixed bug in GUI where a modified colormap was not updated correctly
- Early interior points detection for Mandelbrot deep zoom

Rev 0.5.5
~~~~~~~~~
- Quick estimation of skew when no mini (based on a diverging point)
- Improved documentation

Rev 0.5.4
~~~~~~~~~
- Dark mode for html doc
- Remove debugging test file ("knwon fails") from source distribution

Rev 0.5.3
~~~~~~~~~
- Added more examples and reorganised the documentation
- Fixed bug in Perpendicular Burning Ship

Rev 0.5.2
~~~~~~~~~
- Added Perpendicular Burning Ship in arbitrary precision
  Rev 0.5.1
- Added unskew option in interactive mode
- Added Tetration (power tower) fractal

Rev 0.5.0
~~~~~~~~~
- Added Burning ship deep explorer
- Added unskew option in batch mode
- Chained Bilinear interpolations for arbitrary precision zooms
- Glitch correction after Zhuoran
  (https://fractalforums.org/fractal-mathematics-and-new-theories/28/another-solution-to-perturbation-glitches/4360)
- Documentation: GUI now runs & output figures from Github headless runner
  for interactive script examples

Rev 0.4.3
~~~~~~~~~
- fixed typo in run_interactive.py

Rev 0.4.2
~~~~~~~~~
- fixed concurrent.futures import

Rev 0.4.1
~~~~~~~~~
- Improved Fieldlines default postproc when using mirrored cmap
- Typo name Disp_Layer -> Disp_layer
- added Collaz fractal
- use gmpy2 bindings through Cython C-extension for faster full-precision
  calculations
- improved glitch correction : use single-reference based method
- use multi-threading + NOGIL compilation to improve portability under Windows
- build under windows
- added progress status bar
- cleaner separation of parameters by themes in GUI
- rotation in GUI
- Newton search in GUI
- When one quit and relaunch the GUI, all previous parameters are reloaded


