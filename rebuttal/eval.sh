#!/bin/bash
# Evaluate one add-on run with the paper's eval.py on LCB-v6 (held-out tests) and ErrCalBench.
#   bash rebuttal/eval.sh <label> <model: rebuttal/runs/<exp>/global_step_120/actor | HF id>
set -euo pipefail
LABEL=$1; MODEL=$2
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-$REPO/data}
mkdir -p $REPO/rebuttal/results/$LABEL
cd $REPO
python eval.py --model "$MODEL" --benchmark lcb --data $DATA_ROOT/lcb_v6/test.parquet \
    --diagnosis binary --output rebuttal/results/$LABEL/lcb.json
python eval.py --model "$MODEL" --benchmark errcalbench --data $DATA_ROOT/ErrCalBench.jsonl \
    --diagnosis binary --output rebuttal/results/$LABEL/errcalbench.json
