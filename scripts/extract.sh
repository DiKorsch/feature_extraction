#!/usr/bin/env bash
# Script for generic feature extraction

# resnet inception inception_tf
MODEL_TYPE=${MODEL_TYPE:-resnet}
# NAC GT GT2 L1_pred L1_full
PARTS=${PARTS:-NAC}
DATA=/home/korsch/Data/info.yml

source config.sh

$PYTHON $SCRIPT \
	$DATA \
	$PARTS \
	$OUTPUT \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	$OPTS \
	$@
