#!/bin/bash

# upgrade python version
./python.sh

# install linux dependencies
./ubuntu.sh

# install python dependencies
pip install -r pre-requirements.txt
pip install -r requirements.txt

# install jupyter lab
./jupyter.sh

# make python install directory writable by staff group
# (already owned by staff group from python.sh)
chmod -R g+w /usr/local/lib /usr/local/bin
