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

# Note: Regarding "Using deprecated NumPy API" deprecation warning when 
# compiling C extensions with Cython
# https://github.com/numpy/numpy/issues/11653
# https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api

extra_link_args = []


if sys.platform == "win32":
    
    include_dirs = (
        sys.path
        + [os.path.dirname(gmpy2.__file__)]
        + [np.get_include()]  
    )
    
    ext_FP = setuptools.Extension(
        "fractalshades.mpmath_utils.FP_loop",
        [r"src/fractalshades/mpmath_utils/FP_loop.pyx"],
        include_dirs=include_dirs,
        libraries=['gmp', 'mpfr', 'mpc'],
        extra_link_args=extra_link_args
    )

#    # To build for Windows:
#    # 1. Install MingW-W64-builds from https://mingw-w64.org/doku.php/download
#    #    It is important to change the default to 64-bit when installing if a
#    #    64-bit Python is installed in windows.
#    # 2. Put the bin/ folder inside x86_64-8.1.0-posix-seh-rt_v6-rev0 in your
#    #    system PATH when compiling.
#    # 3. The code below will moneky-patch distutils to work.
#    import distutils.cygwinccompiler
#    distutils.cygwinccompiler.get_msvcr = lambda: []
#    # Make sure that pthreads is linked statically, otherwise we run into problems
#    # on computers where it is not installed.
#    extra_link_args = ["-Wl,-Bstatic", "-lpthread"]



# https://stackoverflow.com/questions/65334494/python-c-extension-packaging-dll-along-with-pyd
#    Create a directory within your package that will contain all required DLLs your extension will be using
#    Modify your build procedure to include this directory along with sdist and wheel distributions
#    Once user imports your package, first thing you do is dynamically modify paths for where DLLs will be searched for (two different methods depending if you are on 3.8 or lower)
#






else:

    include_dirs = (
        sys.path
        + [os.path.dirname(gmpy2.__file__)]
        + [np.get_include()]  
    )
    
    ext_FP = setuptools.Extension(
        "fractalshades.mpmath_utils.FP_loop",
        [r"src/fractalshades/mpmath_utils/FP_loop.pyx"],
        include_dirs=include_dirs,
        libraries=['gmp', 'mpfr', 'mpc'],
        extra_link_args=extra_link_args
    )

# Note : MPFR under windows
# https://github.com/AntonKueltz/fastecdsa/issues/11
# https://preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/
# Or: https://www.msys2.org/
# https://www.lucaswillems.com/fr/articles/59/installer-gmp-windows
# curl https://httpbin.org/json >> $GITHUB_ENV

# curl.exe -o msys2-x86_64-latest.exe https://github.com/msys2/msys2-installer/releases/download/2021-11-30/msys2-base-x86_64-20211130.sfx.exe
# .\msys2-x86_64-latest.exe in --confirm-command --accept-messages --root C:/msys64

#  Build binaries on Windows.
# Could not find a working solution online, but were able to make
# this work with mingw-w64. Visual Studio will not compile QuickJS.
# https://github.com/PetterS/quickjs/commit/67bc2428b8c0716538b4583f4f2b0a2a5a49106c


setuptools.setup(
    ext_modules=cythonize(
        [ext_FP],
        include_path=include_dirs,
        compiler_directives={'language_level' : "3"}
    )
)
