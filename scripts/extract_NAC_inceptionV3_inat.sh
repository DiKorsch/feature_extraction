#!/usr/bin/env bash
# Extracts NAC InceptionV3-features pre-trained on iNaturalist and finetuned on cub200

MODEL_TYPE="inception"
SUFFIX="20parts"
# WEIGHTS="ft_cub200/sgd.inat_pretrain/g_avg_pooling/model.npz"
WEIGHTS="ft_inat/model.ckpt.npz"
DATASET="NAC/2017-bilinear"
BATCH_SIZE=24
# N_LOADERS=2

OPTS="--prepare_type custom"
source config.sh

$PYTHON $SCRIPT \
	$DATA \
	$MODEL \
	$OUTPUT \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	--part_rescale 227 \
	$OPTS \
	$@

# echo "Copying \"${FEATURES}\" to \"$DATA/features\""
$CP $FEATURES $DATA/features
