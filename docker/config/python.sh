#!/bin/bash

apt-get update

# get preinstalled pip dependencies
pip freeze > preinstalled.txt

# delete preinstalled dependencies
rm -r /usr/local/lib/*

# make everything in python package folders owned by staff group
chgrp -R staff /usr/local/lib /usr/local/bin
chmod -R 2775  /usr/local/lib /usr/local/bin

# upgrade python
apt-get install -y python3.8
rm /usr/local/bin/python
rm /usr/bin/python3
ln -s $(which python3.8) /usr/local/bin/python
ln -s $(which python3.8) /usr/bin/python3
python -m pip install --upgrade pip setuptools six

# reinstall preinstalled dependencies
pip install -r preinstalled.txt
