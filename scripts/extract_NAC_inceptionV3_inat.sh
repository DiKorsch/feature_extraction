#!/usr/bin/env bash
# Extracts NAC InceptionV3-features pre-trained on iNaturalist and finetuned on cub200

MODEL_TYPE="inception"
SUFFIX="20parts"
WEIGHTS="ft_cub200/rmsprop.inat_pretrain/g_avg_pooling/clf_epoch30"
DATASET="NAC/2017-bilinear"

source config.sh


echo $OUTPUT

exit 0
$PYTHON $SCRIPT \
	$DATA \
	$MODEL \
	$OUTPUT \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	$OPTS \
	$@
