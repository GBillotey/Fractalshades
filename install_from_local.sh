#!/bin/bash

# Dev mode installation : --force reinstall -e
# https://stackoverflow.com/questions/30306099/pip-install-editable-vs-python-setup-py-develop
# sudo python3 -m pip install --user --force-reinstall --no-deps --editable .
python3 -m pip install --user --force-reinstall --no-deps ./dist/fractalshades_G_BILLOTEY-0.2.0-py3-none-any.whl


# upload to test
# python3 -m twine upload --repository testpypi dist/*
