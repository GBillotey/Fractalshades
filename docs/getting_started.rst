
Getting started
***************

Installing
~~~~~~~~~~

Fractalshades is published on the Python Package Index (PyPI_). To install the
last published version and its dependencies, you should run: [#f1]_

.. code-block:: console

    python3 -m pip install --user fractalshades

or, to install directly the latest version from Github master:

.. code-block:: console

    python -m pip install https://github.com/GBillotey/Fractalshades.git

This package relies on the following dependencies:

- numpy_
- numba_
- mpmath_
- gmpy2_
- Pillow_
- PyQt5_

.. _numpy: https://numpy.org/
.. _numba: http://numba.pydata.org/
.. _mpmath: https://mpmath.org/
.. _gmpy2: https://gmpy2.readthedocs.io/en/latest/
.. _Pillow: https://pillow.readthedocs.io/en/stable/
.. _PyQt5: https://pypi.org/project/PyQt5/
.. _PyPI: https://pypi.org/

They should install automatically through `pip`. A special case is gmpy2 as it
needs the most recent versions of GMP, MPFR and MPC multi-precision
arithmetic libraries. If your distribution does not include them you will have
to install them manually. For instance under Linux:

.. code-block:: console

    sudo apt-get install libgmp-dev
    sudo apt-get install libmpfr-dev
    sudo apt-get install libmpc-dev

.. [#f1] These instructions describe installation to your Python home
         directory. You could also consider the installation of
         `Fractalshades` in a virtual environment (a self-contained directory
         tree that contains a Python installation for a particular version of
         Python), through venv_.

.. _venv: https://docs.python.org/3/tutorial/venv.html



A 5-minutes guide to fractalshades
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fractalshades exposes 3 kinds of component each implementing a different
functionality:

  - The core or calculation components runs under the hood all the
    calculations necessary for a plot and
    stores the intermediate raw results. This is typically done in a subclass
    of `fractalshades.Fractal` base class.
    The list of the available fractal models is found here :
    :doc:`API/models`.

  - The plotting components will open the raw results and apply user-selected
    post-processing, to generate the image output. The base class for this
    part is `fractalshades.Fractal_plotter`.
    Common post-processing routines are available, they are listed under
    the :doc:`API/postproc` section.

  - In order to explore a fractal and select a location, a GUI is necessary.
    Fractalshades comes with a very flexible and user-configurable graphical
    interface based on PyQt5 framework.

The best way to start is probably to have a look at the 
:doc:`examples/index` section.


