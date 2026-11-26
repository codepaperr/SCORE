#!/bin/bash
# One add-on run of the author response (Qwen3-1.7B, one GPU).
#   bash rebuttal/run.sh <data: v6|temporal> <method> <seed>
# v6       = the paper's setting: the 131 LiveCodeBench-v6 problems, 30 epochs (120 steps)
# temporal = 737 LiveCodeBench problems released before 2025-02, which share no problem with the
#            131 LCB-v6 test problems; the same 120 steps, so each problem is seen about 5 times
# method   = score | sdpo | grpo | score_kn (weights from the group success rate k/N, no self-prediction)
# Set DATA_ROOT to the folder that holds lcb_v6/ and lcb_temporal/ (train.parquet, test.parquet).
set -euo pipefail
DATA=$1; METHOD=$2; SEED=$3; shift 3
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-$REPO/data}
case $DATA in
  v6)       export DATA_DIR=$DATA_ROOT/lcb_v6;       EPOCHS=30; export EXP_PREFIX=v6_ ;;
  temporal) export DATA_DIR=$DATA_ROOT/lcb_temporal; EPOCHS=6;  export EXP_PREFIX=temporal_ ;;
  *) echo "unknown data $DATA"; exit 1 ;;
esac
export OUT_DIR=${OUT_DIR:-$REPO/rebuttal/runs}
bash $REPO/train.sh 1.7b $METHOD $SEED trainer.total_epochs=$EPOCHS "$@"
