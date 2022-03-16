# -*- coding: utf-8 -*-
"""
This script patches an existing gmpy2 installation dir in site-package by 
adding the following files : headers, .lib associated to the dll.
This then will allow us to link against GMP, MPFR and MPC dlls
"""
import os
import shutil
import glob

import gmpy2

gmpy2_dir = os.path.dirname(gmpy2.__file__)

# Find the directory for MS Visual studio cl.exe dumpbin.exe lib.exe
# Note : should we automate this ?
# https://stackoverflow.com/questions/54305638/how-to-find-vswhere-exe-path
MVS_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE"

print("\n * Checking gmpy2 install")
ctx = gmpy2.get_context()
a = gmpy2.mpc("1.0")
print("gmpy2 install DIR:\n", gmpy2_dir)
print("gmpy2 context:\n", ctx)
print("2 * 1.0j =", a * 2.)

print("\n * listing files in gmpy2 dir :")
print(os.listdir(gmpy2_dir))
print("\n * listing files in site-package dir :")
print(os.listdir(os.path.dirname(gmpy2_dir)))
# gmpy2_dir content :
#  'gmpy2.cp38-win_amd64.pyd'
#  'gmpy2.h',
#  'gmpy2.pxd',
#  'libgcc_s_seh-1.dll',
#  'libgmp-10.dll',
#  'libmpc-3.dll',
#  'libmpfr-6.dll',
#  'libwinpthread-1.dll',
#  '__init__.pxd',
#  '__init__.py',
#  '__pycache__'

print("\n * Adding necessary header files to gmpy2 insatll dir")
# Currently we store the heeaders locally ; we could also download them
# at runtime
for header in glob.glob("win_build/gmpy2_headers/*.h"):
    print("Copy header file", header, "-->", gmpy2_dir)
    shutil.copy2(header, gmpy2_dir)

print("\n * Listing files in Microsoft Visual Studio dir :")
print("MVS_dir", MVS_dir)
print(os.listdir(MVS_dir))
print("\n * Adding MVS_dir to system PATH")
os.environ["PATH"] += os.pathsep + MVS_dir
print("...", os.environ["PATH"][-600:])


print("\n * Generate import library from the dlls")
# Note : need dumpbin and lib in path
# Note path sep under windows : '\'
for dll in glob.glob(gmpy2_dir + "/" + "*.dll"):
    print(">>> dll file:", dll)
    os_exc = f"win_build\dll2lib.bat 64 {dll}"
    print("execute:", os_exc)
    os.system(os_exc)

# Move the created .lib files to gmpy2 install dir
for lib in glob.glob("*lib"):
    shutil.copy2(lib, gmpy2_dir)
    
print("executed, folder content:")
print(os.listdir(gmpy2_dir))
