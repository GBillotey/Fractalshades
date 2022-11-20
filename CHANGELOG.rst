Changelog
*********

Rev 1.0.0
~~~~~~~~~
This revision is a major refactoring of the code, with optimised
high-resolution rendering options. Several non backward-compatible changes
habe been introduced.
If you use the GUI pre-made scripts available in the documentation section,
you will need to update them (i.e., download) as scripts running with 0.5.x
are not compatible with version 1.x.y onwards.

- Fractal_plotter now implements `supersampling` and `jitter` parameters
- 'antialiasing' parameter from zoom method deprecated: use Fractal_plotter
  `supersampling` options instead
- `BLA_params` dict-like parameter replaced by a float parameter:
   `BLA_eps`
- in add_light_source method from Blin_lighting class:
      - float 2-uplet parameters 'angles' replaced by 2 float parameters 
        `polar_angle`, `azimuth_angle`
      - `coords` parameter removed
- Fractal class: data files `.params` renamed `.fingerprint`
- `run` method from Fractal removed. One consequence: to clean-up stored
  calculation files, one shall now use  `clean_up` method before
  any calculation methd call (previously it was to be used before `run` call)
- in Color_layer, parameter `probes_kind` suppressed, all probes are now
  "absolute".
- Normal map layer: A correction of the "max slope" calculation may impact
  shaded renderings (especially glossy), usually a smaller slope angle is now
  to be used to achieve similar result (e.g. 45. -> 30.).
- Normal map layer: A correction of the "max slope" calculation may impact
  shaded renderings (especially glossy), usually a smaller slope angle is now
  to be used to achieve similar result (e.g. 45. -> 30.).
- Color types: now use a dedicated type fscolors.Color instead of QtGui.QColor:
  this allows running in batch mode for systems without pyQt6 installed

Several changes also implemented in the GUI:
- Collapsible paramters groups (main panel) to ease access to parameters
- Dedicated editor for lighting effect
- Dedicated editor for Fractals
- Parameter "function" can now be passed (as string)

Due to the high number of changes deployed, it is expected that bugfix-revision
will need to be issued, please report any unexpected behavior at:
https://github.com/GBillotey/Fractalshades/issues

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


