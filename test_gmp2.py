import gmpy2
from gmpy2 import mpc

ctx = gmpy2.get_context()
a = mpc("1.0")
print("gmpy2 context:\n", ctx)
print("2 * 1.0j =", a * 2.)
