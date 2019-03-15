#!/usr/bin/env bash
# Extracts NAC InceptionV3-features pre-trained on iNaturalist and finetuned on cub200

MODEL_TYPE="inception"
SUFFIX="5parts_gt"
# WEIGHTS="ft_cub200/sgd.inat_pretrain/g_avg_pooling/model.npz"
WEIGHTS="ft_inat/model.ckpt.npz"
DATASET="cub200_11_regrouped"
BATCH_SIZE=32
# N_LOADERS=2

OPTS="--prepare_type custom"
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

$CP $FEATURES $DATA/features
