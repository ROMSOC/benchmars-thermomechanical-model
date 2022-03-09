# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

FROM ubuntu:20.04
ENV LAST_UPDATED=2022-03-01

USER root
WORKDIR /tmp

# Install FEniCS
USER root

# Install system packages
RUN apt-get update --yes
RUN apt-get install --yes --no-install-recommends software-properties-common
RUN apt-add-repository ppa:fenics-packages/fenics
RUN apt-get install --yes --no-install-recommends \
        python3-pip \
        fenics

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install JupyterLab
USER root

RUN chmod 777 /tmp        && \
    DEBIAN_FRONTEND="noninteractive" && \
    pip3 install --upgrade --no-cache-dir \
      jupyterlab                    \
      jupytext                      \
      jupyter-book                  \
      ghp-import

# Install RBniCS
USER root
RUN apt-get update --yes
RUN apt-get -qq remove python3-pytest && \
    apt-get install --yes --no-install-recommends \
      libsuitesparse-dev git python3-dev && \
    apt-get --yes clean          && \
    apt-get --yes autoremove     && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade cvxopt "flake8<4" multipledispatch pylru "pytest<7" pytest-benchmark \
       pytest-dependency pytest-flake8 sympy toposort numpy scipy matplotlib 


RUN git clone https://github.com/RBniCS/RBniCS.git
WORKDIR /tmp/RBniCS
RUN python3 setup.py -q install
RUN rm -rf /tmp/RBniCS

# Install Pyvista
USER root
RUN apt-get update --yes && \
    apt-get install  -yq --no-install-recommends \
      libgl1 \
      libgl1-mesa-glx xvfb \
      libfontconfig1 \
      libxrender1 \
      libosmesa6  && \
    apt-get --yes clean && \ 
    rm -rf /var/lib/apt/lists/* && \
    pip3 -q install cmocean colorcet imageio-ffmpeg imageio ipygany \ 
      ipyvtklink meshio panel trimesh pythreejs pyvista \ 
      piglet pyvirtualdisplay

# allow jupyterlab for ipyvtk
ENV JUPYTER_ENABLE_LAB=yes
ENV PYVISTA_USE_IPYVTK=true

# Configure environment
ENV SHELL=/bin/bash \
    NB_USER="${NB_USER}" \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME="/home/${NB_USER}"

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

WORKDIR "${HOME}"