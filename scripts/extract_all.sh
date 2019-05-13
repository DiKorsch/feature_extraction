#!/usr/bin/env bash

# resnet inception inception_tf
export MODEL_TYPE=inception
export OMP_NUM_THREADS=4
export N_JOBS=3
export BATCH_SIZE=12

export DATASET=CUB200
export OUTPUT=/home/korsch/Data/DATASETS/birds/cub200/features

export WEIGHTS=$(realpath ../models/ft_${DATASET}_inceptionV3.npz)
# export WEIGHTS=/home/korsch1/korsch/models/inception/model.inat.ckpt.npz

PARAMS="--prepare_type model --input_size 427 --label_shift 1"

PARTS=L1_pred ./extract.sh $PARAMS
PARTS=L1_full ./extract.sh $PARAMS

