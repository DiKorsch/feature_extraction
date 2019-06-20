#!/usr/bin/env bash

# resnet inception inception_tf
export MODEL_TYPE=inception
export OMP_NUM_THREADS=4
export N_JOBS=3
export BATCH_SIZE=12

export DATASET=CUB200
export OUTPUT=/home/korsch/Data/DATASETS/birds/cub200/features

# export WEIGHTS=/home/korsch1/korsch/models/inception/ft_CUB200/rmsprop.g_avg_pooling/model.inat.ckpt/model_final.npz
export WEIGHTS=/home/korsch1/korsch/models/inception/model.inat.ckpt.npz

PARAMS="--prepare_type model --input_size 427 --label_shift 1"

PARTS=L1_pred_15 ./extract.sh $PARAMS
PARTS=L1_full_15 ./extract.sh $PARAMS

