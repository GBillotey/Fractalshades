#!/bin/bash
# Installation from linux locally built wheel fractalshades-*-cp38-cp38-linux_x86_64.whl
python3 -m pip install fractalshades --user --force-reinstall --no-deps --no-index --find-links dist

#file_pattern="*linux_x86_64.whl"

#for filename in ./dist/*; do
#    if [[ $filename = $file_pattern ]]
#    then
#        echo "*** Found wheel" "$filename"
#        python3 -m pip install --user --force-reinstall --no-deps  "$filename"
#    fi
#done

sleep 2
