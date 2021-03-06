#!/bin/bash

# vars for jupyter
storage_dir=$PWD
export JUPYTER_PATH=${storage_dir}/.jupyter
export JUPYTER_RUNTIME_DIR=${storage_dir}/.local/share/jupyter/runtime
export JUPYTER_DATA_DIR=${storage_dir}/.local/share/jupyter
export IPYTHONDIR=${storage_dir}/.ipython

source ticlenv/bin/activate

# Additional system paths for local packages
export PYTHONPATH=${VIRTUAL_ENV}/lib/python3/site-packages/:${PYTHONPATH}
export PYTHONPATH=${PYTHONPATH}:${VIRTUAL_ENV}/../
