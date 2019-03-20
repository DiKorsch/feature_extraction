#!/usr/bin/env bash

# resnet inception inception_tf
export MODEL_TYPE=inception
export N_LOADERS=2

export DATASET=CUB200
export OUTPUT=../output/$DATASET

PARTS=GLOBAL BATCH_SIZE=128 ./extract.sh

PARTS=NAC BATCH_SIZE=24 ./extract.sh

PARTS=GT BATCH_SIZE=32 ./extract.sh

PARTS=GT2 BATCH_SIZE=64 ./extract.sh

PARTS=L1_pred BATCH_SIZE=64 ./extract.sh

PARTS=L1_full BATCH_SIZE=64 ./extract.sh

