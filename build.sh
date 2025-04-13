#!/bin/bash
# This script will generate distribution packages for the package.
# These are archives that are uploaded to the Package Index and can be installed by pip.
# UNIX / MAC

source bin/activate
rm -rf build dist
python3 setup.py build bdist_wheel
python3 setup.py sdist

# https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html#summary
# These 2 commands shall be replaced with the 'official' 
# python -m build
sleep
