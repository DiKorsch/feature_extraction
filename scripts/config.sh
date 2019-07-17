if [[ ! -f /.dockerenv ]]; then
	source ${HOME}/.anaconda3/etc/profile.d/conda.sh
	conda activate chainer4
fi

PYTHON=${PYTHON:-python}
SCRIPT="../run.py"

GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
N_JOBS=${N_JOBS:-0}
FINAL_POOLING=${FINAL_POOLING:-g_avg}

MODEL_TYPE=${MODEL_TYPE:-inception}
PREPARE_TYPE=${PREPARE_TYPE:-model}

OPTS=${OPTS:-""}
OPTS="${OPTS} --gpu $GPU"
OPTS="${OPTS} --batch_size $BATCH_SIZE"
OPTS="${OPTS} --n_jobs $N_JOBS"
OPTS="${OPTS} --pooling ${FINAL_POOLING}"
# OPTS="${OPTS} --augment_positions"

OPTS="${OPTS} --model_type $MODEL_TYPE"
OPTS="${OPTS} --prepare_type $PREPARE_TYPE"
OPTS="${OPTS} --weights $WEIGHTS"

CP="rsync -auh --info=progress2"
