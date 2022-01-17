import os
import gmpy2


ctx = gmpy2.get_context()
a = gmpy2.mpc("1.0")
print("gmpy2 install DIR:\n", os.path.dirname(gmpy2.__file__))
print("gmpy2 context:\n", ctx)
print("2 * 1.0j =", a * 2.)

# Hack setup.cfg
hack_setup_str = (
"""[build]
compiler=mingw32

"""
)

with open("setup.cfg", "a") as setup_cfg:
    setup_cfg.write(hack_setup_str)

print("### modifed setup.cfg with:\n", hack_setup_str)

