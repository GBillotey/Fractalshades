# -*- coding: utf-8 -*-
import os
#from run_mandelbrot import plot_classic_draft
import numpy as np

#def plot_tests():
#    """
#    Run a few sanity tests based on classical Mandelbrot.
#    (image size, x / y ratio, rotation, ...)
#    """
#    ntests = 5
#    test_params = [{"x": -0.75, "y": 0.0, "dx": 3., "xy_ratio": 1., 
#                        "theta_deg": 0, "nx": 400},
#                   {"x": -0.75, "y": 0.0, "dx": 3., "xy_ratio": 1., 
#                        "theta_deg": 0, "nx": 357}, 
#                   {"x": -0.75, "y": 0.0, "dx": 3., "xy_ratio": 1., 
#                        "theta_deg": 45, "nx": 400},
#                   {"x": -0.75, "y": 0.1, "dx": 3., "xy_ratio": 2., 
#                        "theta_deg": 0, "nx": 357}, 
#                   {"x": -0.75, "y": 0.1, "dx": 3., "xy_ratio": 0.5, 
#                        "theta_deg": -45, "nx": 633}]
#    test_main_dir = "/home/gby/Pictures/Mes_photos/math/fractal/tests"
#    test_dirs = [os.path.join(test_main_dir,
#                              str(itest)) for itest in range(ntests)]
#
#    for itest in range(ntests):
#        plot_classic_draft(test_dirs[itest], **test_params[itest])


def class_test():
    import tempfile

    class Wrapper(object):
        def __init__(self):
            self._ref = None

        def __getitem__(self, key):
            _ = self._ref[key].seek(0)
            return np.load(self._ref[key])

        def __setitem__(self, key, val):
            if self._ref is None:
                self._ref = dict()
            self._ref[key] = tempfile.SpooledTemporaryFile(mode='w+b')
            np.save(self._ref[key], val)#, allow_pickle=True, fix_imports=True)
            #self._ref[key].write(val)
            
        def __invert__(self):
            """
            Convenience for ~ notation
            """
            return self

  
    w = Wrapper()
    w["a"] = np.ones([100, 215], dtype=np.longcomplex)
    print("read\n", w["a"])




    
    
    
    

if __name__ == "__main__":
    #plot_tests()
    class_test()