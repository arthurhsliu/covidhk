#! /bin/bash
set -e
export MPLBACKEND=Agg
cd /home/jovyan/work
python hkg.py
python hkg.py vax
