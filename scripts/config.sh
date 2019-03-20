source ${HOME}/.anaconda3/etc/profile.d/conda.sh
conda activate chainer4

PYTHON=python
SCRIPT="../run.py"

GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
N_LOADERS=${N_LOADERS:-0}

OPTS=${OPTS:-""}
# OPTS="${OPTS} --augment_positions"
OPTS="${OPTS} --batch_size $BATCH_SIZE"
OPTS="${OPTS} --n_jobs $N_LOADERS"

CP="rsync -auh --info=progress2"
