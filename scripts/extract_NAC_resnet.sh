#!/usr/bin/env bash
# Extracts NAC ResNet50-features pre-trained on ImageNet and finetuned on cub200

MODEL_TYPE="inception"
SUFFIX="20parts"
WEIGHTS="ft_cub200/model.npz"
DATASET="NAC/2017-bilinear"

source config.sh

$PYTHON $SCRIPT \
	$DATA \
	$MODEL \
	$OUTPUT \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	$OPTS \
	$@
