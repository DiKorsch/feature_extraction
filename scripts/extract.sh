#!/usr/bin/env bash
# Script for generic feature extraction

# resnet inception inception_tf
MODEL_TYPE=${MODEL_TYPE:-inception}
N_LOADERS=${N_LOADERS:-2}

# NAC GT GT2 L1_pred L1_full
# PARTS=${PARTS:-GT}
DATA=/home/korsch/Data/info.yml

if [[ -z $DATASET ]]; then
	echo "DATASET variable is missing!"
	exit -1
fi

if [[ -z $PARTS ]]; then
	echo "PARTS variable is missing!"
	exit -1
fi

WEIGHTS=${WEIGHTS:-"rmsprop.g_avg_pooling/model.inat.ckpt/model_final.npz"}

OUTPUT=${OUTPUT:-"../output/$DATASET"}

if [[ ! -d $OUTPUT ]]; then
	mkdir -p $OUTPUT
fi

source config.sh

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${DATASET}_${PARTS} \
	${OUTPUT} \
	--weights ${WEIGHTS} \
	-mt $MODEL_TYPE \
	--gpu $GPU \
	$OPTS \
	$@
