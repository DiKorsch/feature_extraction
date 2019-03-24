#!/usr/bin/env bash
# Script for generic feature extraction

# resnet inception inception_tf
MODEL_TYPE=${MODEL_TYPE:-inception}
N_JOBS=${N_JOBS:-2}

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
	${OPTS} \
	$@
