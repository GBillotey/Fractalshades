#!/bin/bash
# This script will generate distribution packages for the package.
# These are archives that are uploaded to the Package Index and can be installed by pip.
# UNIX / MAC
rm -rf build dist
python3 -m build