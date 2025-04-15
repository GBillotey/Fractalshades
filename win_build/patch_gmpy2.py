# -*- coding: utf-8 -*-
"""
This script patches an existing gmpy2 installation dir in site-package
as needed for Windows build process
"""
import os
import shutil
import glob

import gmpy2

# Find the directory for MS Visual studio cl.exe dumpbin.exe lib.exe
# Note: not needed with recent versions of gmpy2
# https://stackoverflow.com/questions/54305638/how-to-find-vswhere-exe-path
MVS_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.43.34808\bin\Hostx86\x64"

# Windows dll have been moved to gmpy2.libs
gmpy2_dir = os.path.dirname(gmpy2.__file__)
gmpy2_libs = gmpy2_dir + ".libs"

print("\n ====== Checking gmpy2 install...")
ctx = gmpy2.get_context()
a = gmpy2.mpc("1.0")
print("gmpy2 install DIR:\n", gmpy2_dir)
print("\n * listing files in site-package gmpy2 :\n")
print(os.listdir(gmpy2_dir))
print("\n * listing files in site-package gmpy2.libs :\n")
print(os.listdir(gmpy2_libs))
print("gmpy2 context:\n", ctx)
print("2 * 1.0j =", a * 2.)
print("====== Done\n")


# Headers to be patched, if needed
patched_headers = ["mpc.h"]
print("\n ====== Patching gmpy2 headers...")
for header in patched_headers:
    copied = shutil.copy2(
        os.path.join("win_build", "gmpy2_headers", header),
        gmpy2_dir
    )
    print(rf"Copied {os.path.join('win_build', 'gmpy2_headers', header)} to {copied}")
print("====== Done\n")
