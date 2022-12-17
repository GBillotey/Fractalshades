import sys
import os
import setuptools

from Cython.Build import cythonize
import numpy as np
import gmpy2

# Built-time dependencies are listed in pyproject.toml
# [build-system]

# Note : Build-time dependency on NUMPY :
# https://numpy.org/doc/stable/user/depending_on_numpy.html
# If a package either uses the NumPy C API directly or it uses some other tool 
# that depends on it like Cython or Pythran, NumPy is a build-time dependency
# of the package. Because the NumPy ABI is only forward compatible, you must
# build your own binaries (wheels or other package formats) against the lowest
# NumPy version that you support (or an even older version).
# For that purpose fractalshades uses the package oldest-supported-numpy
# https://pypi.org/project/oldest-supported-numpy/
# see: pyproject.toml

# Note: A "Using deprecated NumPy API" deprecation warning will appear when 
# compiling C extensions with Cython
# https://github.com/numpy/numpy/issues/11653
# https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api


extra_link_args = []

include_dirs = (
    sys.path
    + [os.path.dirname(gmpy2.__file__)]
    + [np.get_include()]  
)

if sys.platform == "win32":
    # An import library is necessary when calling functions in a DLL; it
    # provides the stubs that hook up to the DLL at runtime.
    # This means, during build process the following files need to be added
    # to gmpy2 site-package directory: header files and .lib files for the dll
# https://stackoverflow.com/questions/9946322/how-to-generate-an-import-library-lib-file-from-a-dll
    ext_FP = setuptools.Extension(
        "fractalshades.mpmath_utils.FP_loop",
        [r"src/fractalshades/mpmath_utils/FP_loop.pyx"],
        include_dirs=include_dirs,
        library_dirs=include_dirs,
        libraries=[
            'libgcc_s_seh-1',
            'libgmp-10',
            'libmpc-3',
            'libmpfr-6',
            'libwinpthread-1',
        ],
        extra_link_args=extra_link_args
    )

else:
    # Building extension for UNIX-like platforms
    ext_FP = setuptools.Extension(
        "fractalshades.mpmath_utils.FP_loop",
        [r"src/fractalshades/mpmath_utils/FP_loop.pyx"],
        include_dirs=include_dirs,
        libraries=['gmp', 'mpfr', 'mpc'],
        extra_link_args=extra_link_args
    )

setuptools.setup(
    ext_modules=cythonize(
        [ext_FP],
        include_path=include_dirs,
        compiler_directives={'language_level' : "3"}
    )
)
