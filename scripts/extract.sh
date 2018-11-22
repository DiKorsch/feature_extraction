#!/usr/bin/env bash
PYTHON=python

BASE_DIR="/home/korsch/Data"

GPU=1

DATA="${BASE_DIR}/DATASETS/birds/NAC/2017-bilinear"
MODEL_TYPE="vgg19"
MODEL="${BASE_DIR}/MODELS/${MODEL_TYPE}/ft_cub200/model.npz"

OUTPUT="-o ../output/train_feats.npz ../output/val_feats.npz"

OPTS="--augment_positions"

$PYTHON ../run.py \
	$DATA \
	$MODEL \
	$OUTPUT \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	$OPTS \
	$@
