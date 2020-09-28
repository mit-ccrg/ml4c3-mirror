#!/bin/bash

apt-get update

# get preinstalled pip dependencies
pip freeze > preinstalled.txt

# upgrade python
apt-get install -y python3.8
rm /usr/local/bin/python
ln -s $(which python3.8) /usr/local/bin/python
python -m pip install --upgrade pip setuptools six

# reinstall preinstalled dependencies
pip install -r preinstalled.txt
