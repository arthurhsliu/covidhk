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
    fonts-arphic-ukai && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
#    fonts-arphic-uming \
#    fonts-cns11643-sung \
#    fonts-wqy-zenhei && \

RUN ( cd /usr/share/fonts ; \
      wget 'https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf' ; \
      rm -f /home/jovyan/.cache/matplotlib/fontlist-v330.json )

#RUN sed -ie 's/^#font.sans-serif: DejaVu Sans/font.sans-serif: Microsoft YaHei, DejaVu Sans/g' /opt/conda/lib/python3.9/site-packages/matplotlib/mpl-data/matplotlibrc

USER ${NB_UID}

WORKDIR "${HOME}"
