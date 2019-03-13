#!/usr/bin/env bash
# Extracts NAC ResNet50-features pre-trained on ImageNet and finetuned on cub200

FULL=${FULL:-0}

MODEL_TYPE="inception"
WEIGHTS="ft_cub200/sgd.inat_pretrain/g_avg_pooling/model.npz"
SUFFIX="5parts_L1"
DATASET="cub200_11_L1"

if [[ $FULL == "0" ]]; then
	SUFFIX="${SUFFIX}_pred"
	DATASET="${DATASET}_pred"
else
	SUFFIX="${SUFFIX}_full"
	DATASET="${DATASET}_full"
fi

BATCH_SIZE=32
OPTS="--is_bbox_parts"

source config.sh

$PYTHON $SCRIPT \
	$DATA \
	$MODEL \
	$OUTPUT \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	--part_rescale 299 \
	--scales -1 \
	$OPTS \
	$@
