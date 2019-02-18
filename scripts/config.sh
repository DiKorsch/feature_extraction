source ${HOME}/.anaconda3/etc/profile.d/conda.sh
conda activate chainer4

PYTHON=python
SCRIPT="../run.py"
GPU=${GPU:-0}

BASE_DIR="/home/korsch/Data"

OPTS=""
# OPTS="${OPTS} --augment_positions"


DATA="${BASE_DIR}/DATASETS/birds/${DATASET}"
MODEL="${BASE_DIR}/MODELS/${MODEL_TYPE}/${WEIGHTS}"

OUTPUT="-o ../output/train_${SUFFIX}.${MODEL_TYPE}.npz ../output/val_${SUFFIX}.${MODEL_TYPE}.npz"

