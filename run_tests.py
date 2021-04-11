# -*- coding: utf-8 -*-
""" Runs the test suite"""
import unittest
import tests


start_dir = tests.test_dir
loader = unittest.TestLoader()
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)

