#!/bin/bash

# Other necessities
apt-get update
echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
apt-get install -y python3.8-dev
apt-get install -y sudo wget unzip curl python-pydot python-pydot-ng graphviz ttf-mscorefonts-installer
apt-get install -y python3.8-tk libgl1-mesa-glx libxt-dev

# compression library setup
apt-get install -y git cmake libhdf5-dev libzstd-dev
git clone https://github.com/aparamon/HDF5Plugin-Zstandard.git zstd-hd5-linker
cd zstd-hd5-linker
sed -i '1s/^/#define ZSTD_CLEVEL_DEFAULT 3\n/' zstd_h5plugin.c
cmake .
make
make install
cd ..
