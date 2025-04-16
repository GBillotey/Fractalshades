import sys
import os
import setuptools

from Cython.Build import cythonize
import numpy as np
import gmpy2

# Built-time dependencies are listed in pyproject.toml
# [build-system]

# Note : Build-time dependency on NUMPY :
# https://numpy.org/doc/stable/dev/depending_on_numpy.html#adding-a-dependency-on-numpy
# see: pyproject.toml

# A "Using deprecated NumPy API" deprecation warning will appear when 
# compiling C extensions with Cython. This is a red herring.
# https://github.com/numpy/numpy/issues/11653
# https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api

gmpy2_dir = os.path.dirname(gmpy2.__file__)
gmpy2_libs = gmpy2_dir + ".libs"

extra_link_args = []
include_dirs = (
    sys.path
    + [gmpy2_dir]
    + [np.get_include()]  
)

# Note under Windows:
# An import library is necessary when calling functions in a DLL; it
# provides the stubs that hook up to the DLL at runtime.
# This means, during build process the following files are needed in
# gmpy2 or gmpy2.lib site-package directory: header files and .lib files for the librairies
# https://stackoverflow.com/questions/9946322/how-to-generate-an-import-library-lib-file-from-a-dll
# New versions of gmpy2 (2.2.1) ship these .lib files.

ext_FP = setuptools.Extension(
    "fractalshades.mpmath_utils.FP_loop",
    [r"src/fractalshades/mpmath_utils/FP_loop.pyx"],
    include_dirs=include_dirs,
    library_dirs=[gmpy2_libs],
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
