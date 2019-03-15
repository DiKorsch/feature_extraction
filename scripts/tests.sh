#!/usr/bin/env bash

source config.sh

export TF_CUDNN_USE_AUTOTUNE=0
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
$PYTHON -m unittest discover -s .. $@
