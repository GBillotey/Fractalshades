# -*- coding: utf-8 -*-
import unittest
from contextlib import contextmanager
from functools import wraps
import os
import sys

test_dir = os.path.dirname(__file__)

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
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def no_stdout(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with suppress_stdout():
            return func(*args, **kwargs)
    return wrapper