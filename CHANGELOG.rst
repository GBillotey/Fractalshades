Changelog
*********

Rev 1.0.2
~~~~~~~~~
Yet another bugfix.

Rev 1.0.2
~~~~~~~~~
Bugfix revision, fixing OS different behavior with memory mappings #2.

Rev 1.0.1
~~~~~~~~~
Bugfix revision, fixing OS Windows different behavior with memory mappings.

Rev 1.0.0
~~~~~~~~~
This revision is a major refactoring of the code, with optimised
high-resolution rendering options. Several non backward-compatible changes
have been introduced.
If you use the GUI pre-made scripts available in the documentation section,
you will need to update them (i.e., download them again) as the scripts
running with 0.5.x will not be compatible with version 1.x.y onwards.

- `fractalshades.Fractal_plotter`
  now implements ``supersampling`` and ``jitter`` parameters
- ``antialiasing`` parameter from `fractalshades.Fractal.zoom` method
  deprecated: use `fractalshades.Fractal_plotter`
  ``supersampling`` parameter instead
- ``BLA_params`` dict-like parameter replaced by a float parameter:
  ``BLA_eps``
- in `fractalshades.colors.layers.Blinn_lighting.add_light_source` method:
  2-uplet float parameters ``angles`` replaced by 2 float parameters 
  ``polar_angle``, ``azimuth_angle``
- in `fractalshades.colors.layers.Blinn_lighting.add_light_source`:
  ``coords`` parameter deprecated
- `fractalshades.Fractal` saved data files: files ``*.params`` renamed
  as ``*.fingerprint``
- `fractalshades.postproc.Fieldlines_pp` class: implementation entirely
  rewritten, parameter ``damping_ratio`` renamed ``endpoint_k``
- ``run`` method from `fractalshades.Fractal` removed. Calculation is
  automatically trigered through `fractalshades.Fractal_plotter.plot`
  method
- in `fractalshades.colors.layers.Color_layer`, parameter
  ``probes_kind`` suppressed, all probes are now "absolute".
- `fractalshades.colors.layers.Normal_map_layer` : A correction of the
  ``max slope`` calculation may impact shaded renderings (especially glossy
  renders). Usually a smaller slope angle is now needed to be used to achieve
  visually similar result (e.g. 45. -> 30.).
- Color parameters typing : now use a dedicated type `fscolors.Color` type
  is used instead of `QtGui.QColor`:
  this allows to run batch files on systems without pyQt6 installed.
- Avoiding banding for very high iteration numbers (banding may occur,
  usually above 10 million iterations): it is now possible to suppress
  banding by switching to ``float64`` datatype for the postprocessing
  phase : see parameter `fractalshades.settings.postproc_dtype`

Regarding the Fractal models implemented:

- Power n mandelbrot is now implemented in both standard and arbitrary
  precision and accessible from the GUI (in standard precision it
  features calculation of the limit cycle period and attaractivity
  with Newton method.)
- The different Burning Ship variants have been factored into a single class
  with a ``flavor`` parameter to select the appropriate variant.

Regarding the Fractal post-processing implemented:

- The `fractalshades.postproc.Fieldlines_pp` post-processing has
  been re-implemented as a truncated
  orbit average. It shall be used together with the ``calc_orbit``
  and ``backshift`` parameters from the basic iteration computations (to
  define a starting point for the truncated orbit before divergence)

Several changes have also been implemented in the GUI:

- Collapsible parameters groups (main panel) for easier selection
- Dedicated GUI-editor for `fractalshades.colors.layers.Blinn_lighting`
  instances
- Dedicated GUI-editor for `fractalshades.Fractal` subclasses instances.
  This allows the interactive selection
  of different fractal variant: exponent for Mandelbrot power n > 2
  implementations (`fractalshades.models.Mandelbrot_N`, 
  or `fractalshades.models.Perturbation_mandelbrot_N`)
  variant -``flavor``- of the Burning ship implementations
  (`fractalshades.models.Burning_ship`,
  `fractalshades.models.Perturbation_mandelbrot_N`)
  `fractalshades.models.Perturbation_burning_ship`)
- Parametrized "functions" can now be passed (as string)
- A GUI template function `fractalshades.gui.guitemplates.std_zooming` 
  is now available and suitable for most of the implemented
  fractal models ; as a result lanching a GUI with all options for a
  fractal becomes almost a one-liner (see the updated GUI examples).
- Saving and reloading `fractalshades.colors.Fractal_colormap`
  in binary format (through Python standard serialization module
  pickle ) is now available from the GUI (in addition to the already
  available python source-code format output)

Application-level logging has also been implemented (it may target
both the standard output pipe and user-defined log files). Refer to
`fractalshades.settings.log_directory` and
`fractalshades.log.set_log_handlers` .

Please report any bug or unexpected behavior at:
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


