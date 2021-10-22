#!/bin/bash

# Dev mode installation : --force reinstall -e
# https://setuptools.readthedocs.io/en/latest/userguide/commands.html#develop
# Note that --user option is not recognized for editable installs, hence the use of --prefix to explicitely specify the installation directory
# python3 -m pip install --prefix=/home/geoffroy/.local/lib/python3.8/site-packages --force-reinstall --no-deps --editable .

# "Normal" installation
python3 -m pip install --user --force-reinstall --no-deps ./dist/fractalshades-0.3b1-py3-none-any.whl

# upload to test repository
# python3 -m twine upload --repository testpypi dist/*

