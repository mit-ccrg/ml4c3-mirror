ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SETUPTOOLS_USE_DISTUTILS=stdlib

LABEL maintainer="Steven Song <ssong20@mgh.harvard.edu>"

# Setup time zone (or else docker build hangs)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./config/* /app/
WORKDIR /app

# Basic setup
RUN ./ubuntu.sh

# Python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r pre_requirements.txt
RUN pip3 install -r requirements.txt

# Jupyter Lab
RUN curl -sL https://deb.nodesource.com/setup_12.x -o nodesource_setup.sh
RUN bash nodesource_setup.sh
RUN apt-get install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter lab build
