ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SETUPTOOLS_USE_DISTUTILS=stdlib

LABEL maintainer="Steven Song <ssong20@mgh.harvard.edu>"

# Setup time zone (or else docker build hangs)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./config/* /app/
WORKDIR /app

# Install dependencies
RUN ./install.sh
