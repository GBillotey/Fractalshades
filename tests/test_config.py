# -*- coding: utf-8 -*-
""" Gathers codes snippets used in the test suite.
"""
import unittest
from contextlib import contextmanager
from functools import wraps
import os
import sys
import numpy as np
import PIL


test_dir = os.path.dirname(__file__)
temporary_data_dir = os.path.join(test_dir, "_temporary_data")
ref_data_dir = os.path.join(test_dir, "REFERENCE_DATA")

def suite(testcases):
    """
    Parameters
    testcases : an iterable of unittest.TestCases
    
    Returns
    suite : a unittest.TestSuite combining all the individual tests routines
            from the input 'testcases' list (by default these are the method
            names beginning with test).
    """
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for testcase in testcases:
        suite.addTests(loader.loadTestsFromTestCase(testcase))
    return suite

@contextmanager 
def suppress_stdout():
    """ Temporarly suppress print statement during tests. """
    # Note: Only deals with Python level streams ; if need a more involving
    # version dealing also with C-level streams:
    # https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def no_stdout(func):
    """ Decorator, suppress output of the decorated function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with suppress_stdout():
            return func(*args, **kwargs)
    return wrapper

def compare_png(ref_file, test_file):
    """ Return a scalar value function of the difference between 2 images :
    arithmetic mean of the rgb deltas
    """
    ref_image = PIL.Image.open(ref_file)
    test_image = PIL.Image.open(test_file)
    
    root, ext = os.path.splitext(test_file)
    diff_file = root + ".diff" + ext
    diff_image = PIL.ImageChops.difference(ref_image, test_image)
    diff_image.save(diff_file)
    errors = np.asarray(diff_image) / 255.
    return np.mean(errors)
