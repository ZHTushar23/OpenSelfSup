#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

echo $WORK_DIR

#$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    tools/train.py $CFG --work_dir $WORK_DIR --seed 0 --launcher pytorch ${PY_ARGS}
    
#Generate training loss curve
python tools/gen_loss_graph_json.py --work_dir $WORK_DIR ${PY_ARGS}
