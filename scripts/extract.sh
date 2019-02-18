#!/usr/bin/env bash
# Script for generic feature extraction

MODEL_TYPE=${MODEL_TYPE:-"resnet"}
SUFFIX=${SUFFIX:-"20parts"}
WEIGHTS=${WEIGHTS:-"ft_cub200/model.npz"}
DATASET=${DATASET:-"NAC/2017-bilinear"}

source config.sh

$PYTHON $SCRIPT \
	$DATA \
	$MODEL \
	$OUTPUT \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	$OPTS \
	$@
