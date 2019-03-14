#!/usr/bin/env bash
# Extracts NAC InceptionV3-features pre-trained on iNaturalist and finetuned on cub200

MODEL_TYPE="inception_tf"
SUFFIX="20parts"
WEIGHTS="ft_inat/inception_v3_iNat_299.ckpt"
DATASET="NAC/2017-bilinear"
BATCH_SIZE=12
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

$CP $FEATURES $DATA/features
