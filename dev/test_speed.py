# -*- coding: utf-8 -*-
import time
import numpy as np

def main():
    a = np.linspace(1, 1000., 10000000, dtype=np.float128)
    t0 = time.time()
    b = a**2
    print(time.time() - t0)
    




if __name__ == "__main__":
    main()