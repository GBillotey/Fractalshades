# -*- coding: utf-8 -*-
""" Runs the test suite"""
import unittest
import os
import sys

# path of the test rep, relative to this file rep. 
test_rep = "tests"

# Adds the test directory to sys path
abs_test_rep = os.path.abspath(
    os.path.join(os.path.dirname(__file__), test_rep)
)
if abs_test_rep not in sys.path:
    sys.path.insert(0, abs_test_rep)

loader = unittest.TestLoader()
suite = loader.discover(test_rep)

# buffer option controls the stdout output stream
# use buffer=False to activate full output
runner = unittest.TextTestRunner(verbosity=2, buffer=True)

runner.run(suite)

