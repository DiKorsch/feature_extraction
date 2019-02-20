source ${HOME}/.anaconda3/etc/profile.d/conda.sh
conda activate chainer4

PYTHON=python
SCRIPT="../run.py"
BASE_DIR="/home/korsch/Data"

GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
N_LOADERS=${N_LOADERS:-0}

OPTS=""
# OPTS="${OPTS} --augment_positions"
OPTS="${OPTS} --batch_size $BATCH_SIZE"
OPTS="${OPTS} --n_jobs $N_LOADERS"


DATA="${BASE_DIR}/DATASETS/birds/${DATASET}"
MODEL="${BASE_DIR}/MODELS/${MODEL_TYPE}/${WEIGHTS}"

OUTPUT="-o ../output/train_${SUFFIX}.${MODEL_TYPE}.npz ../output/val_${SUFFIX}.${MODEL_TYPE}.npz"

