# -*- coding: utf-8 -*-
""" Runs the test suite"""
import unittest
import os
import sys

# path of the test rep, relative to this file rep. 
test_rep = "tests"

# Adds the test directory to sys path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), test_rep)))

loader = unittest.TestLoader()
suite = loader.discover(test_rep)

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
