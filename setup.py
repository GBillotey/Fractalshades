import setuptools

# from setuptools import setup, find_packages, Extension
# from setuptools.command.build_ext import build_ext
# from distutils.core import Extension
from Cython.Build import cythonize
import sys
# import numpy as np

# import gmpy2
# https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy

# Note: Regarding "Using deprecated NumPy API" deprecation warning
# https://github.com/numpy/numpy/issues/11653
# https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api

# DEV setup - not installing
include_dirs = (
    sys.path
#    + [os.path.dirname(gmpy2.__file__)]
#    + [np.get_include()]
    + [r"/home/geoffroy/.local/lib/python3.8/site-packages/gmpy2"]
    + ['/home/geoffroy/.local/lib/python3.8/site-packages/numpy/core/include']    
)
print("#### include_dirs", include_dirs)

ext_FP = setuptools.Extension(
    "fractalshades.mpmath_utils.FP_loop",
    [r"src/fractalshades/mpmath_utils/FP_loop.pyx"],
    include_dirs=include_dirs,
    libraries=['gmp', 'mpfr', 'mpc'],
)



#ext_test = setuptools.Extension(
#    "fractalshades.numpy_tests.trp",
#    [r"src/fractalshades/numpy_tests/trp.c"],
#    include_dirs=include_dirs,
#)

setuptools.setup(
    ext_modules=cythonize(
        [ext_FP],
        include_path=include_dirs,
        compiler_directives={'language_level' : "3"}
    )
#    + [ext_test]
)
