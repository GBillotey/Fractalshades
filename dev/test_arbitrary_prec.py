# -*- coding: utf-8 -*-
import numpy as np
import mpmath

def _pres_test_1():
    mpmath.mp.dps = 100     # higher precision for demonstration
    a = [mpmath.sin(mpmath.pi*n/3) for n in range(99)]
    b = np.array(a)
    b.dot(b)
    print(b)
    
def _pres_test_2():
    mpmath.mp.dps = 100     # higher precision for demonstration
    a = mpmath.mpf("1.123456891011121314151617181920") + np.linspace(0., 1.e-70, 150)
    b = mpmath.mpf("1.1234568901234567890123456789") + mpmath.mpf("0." + "0"* 99 + "1")
    print(b)
    print(b.__str__())
    print(repr(b))
    print(repr(1.1234567890123456789012345))


if __name__ == "__main__":
    _pres_test_2()