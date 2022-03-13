# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG OWNER=jupyter
ARG BASE_CONTAINER=$OWNER/datascience-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    fonts-wqy-zenhei && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ( rm -f /home/jovyan/.cache/matplotlib/fontlist-v330.json )

USER ${NB_UID}

WORKDIR "${HOME}"
