#!/bin/bash

# Fractalshades installation from locally built wheel (`dist` subdirectory)
# Note: first, run build.sh to build the wheel
python3 -m pip install fractalshades --user --force-reinstall --no-deps --no-index --find-links dist

sleep 2
