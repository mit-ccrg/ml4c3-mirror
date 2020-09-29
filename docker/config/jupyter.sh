#!/bin/bash

curl -sL https://deb.nodesource.com/setup_12.x -o nodesource_setup.sh
bash nodesource_setup.sh
apt-get install -y nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab build
