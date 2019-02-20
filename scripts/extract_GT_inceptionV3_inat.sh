#!/usr/bin/env bash
# Extracts NAC InceptionV3-features pre-trained on iNaturalist and finetuned on cub200

MODEL_TYPE="inception"
SUFFIX="16parts_gt"
WEIGHTS="ft_cub200/sgd.inat_pretrain/g_avg_pooling/model.npz"
DATASET="cub200_11"
BATCH_SIZE=32
# N_LOADERS=2

source config.sh

$PYTHON $SCRIPT \
	$DATA \
	$MODEL \
	$OUTPUT \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	--scales 0.31 \
	$OPTS \
	$@
