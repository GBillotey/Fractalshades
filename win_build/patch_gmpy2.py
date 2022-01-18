import os
import sys
import shutil
import glob

import gmpy2


ctx = gmpy2.get_context()
a = gmpy2.mpc("1.0")

gmpy2_dir = os.path.dirname(gmpy2.__file__)
print("gmpy2 install DIR:\n", gmpy2_dir)
print("gmpy2 context:\n", ctx)
print("2 * 1.0j =", a * 2.)

print(os.listdir(gmpy2_dir))


print(os.listdir(os.path.dirname(gmpy2_dir)))

for header in glob.glob("gmpy2_headers/*"):
    print("Copy header file", header, "-->", gmpy2_dir)
    if sys.platform == "win32":
        shutil.copy2(header, gmpy2_dir)

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
# 