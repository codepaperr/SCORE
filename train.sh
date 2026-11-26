#!/usr/bin/env bash
# Train one run of the paper.
#
#   bash train.sh <model> <method> <seed>
#
#   model : 1.7b | 4b | 8b                 (Qwen3, thinking mode)
#           0.6b                           (only for Fig. 2 / overhead table: THINKING=false, seed 42)
#   method: grpo | sdpo | score            (Table 2)
#           score_noamp | score_nodyn | score_noprefix | score_nocrit   (Table 3, ablations)
#           score_random | score_uniform   (Table 4, failure-weighting controls)
#   seed  : 10 | 20 | 30
#
# All shared hyperparameters live in verl/trainer/config/score.yaml.
# Env: DATA_DIR (default ./data/lcb_v6), OUT_DIR (default ./outputs), GPUS (e.g. 0,1),
#      THINKING (default true; false only for the 0.6B analysis runs).
set -euo pipefail

MODEL=${1:?model: 1.7b|4b|8b|0.6b}
METHOD=${2:?method: grpo|sdpo|score|score_noamp|score_nodyn|score_noprefix|score_nocrit|score_random|score_uniform}
SEED=${3:?seed: 10|20|30}
THINKING=${THINKING:-true}

REPO=$(cd "$(dirname "$0")" && pwd)
DATA_DIR=${DATA_DIR:-$REPO/data/lcb_v6}
OUT_DIR=${OUT_DIR:-$REPO/outputs}
EXP=${EXP_PREFIX:-}${METHOD}_qwen3-${MODEL}_s${SEED}

# ---- model geometry (Table 7: 1.7B 1xH100, 4B 2xH100, 8B 4xH100 with offloading) ----
OFFLOAD=False
case "$MODEL" in
  0.6b) HF=Qwen/Qwen3-0.6B; NGPU=1; TP=1; GPU_UTIL=0.55; DIAG=binary ;;
  1.7b) HF=Qwen/Qwen3-1.7B; NGPU=1; TP=1; GPU_UTIL=0.55; DIAG=binary ;;
  4b)   HF=Qwen/Qwen3-4B;   NGPU=2; TP=2; GPU_UTIL=0.2;  DIAG=five_class ;;
  8b)   HF=Qwen/Qwen3-8B;   NGPU=4; TP=2; GPU_UTIL=0.3;  DIAG=five_class; OFFLOAD=True ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

# ---- method switches (Sec. 3, Sec. 4.1, Sec. 4.3) ----
LOSS=score; QW=False; DYN=True; AMP=False; PREFIX=False; BETA=0.0; CONTROL=none
case "$METHOD" in
  grpo)           LOSS=vanilla ;;
  sdpo)           ;;
  score)          QW=True; AMP=True; PREFIX=True; BETA=0.1 ;;
  score_noamp)    QW=True; AMP=False; PREFIX=True; BETA=0.1 ;;
  score_nodyn)    QW=True; DYN=False; AMP=True; PREFIX=True; BETA=0.1 ;;
  score_noprefix) QW=True; AMP=True; PREFIX=False; BETA=0.1 ;;
  score_nocrit)   QW=True; AMP=True; PREFIX=True; BETA=0.0 ;;
  score_random)   QW=True; AMP=True; PREFIX=True; BETA=0.1; CONTROL=random ;;
  score_uniform)  QW=True; AMP=True; PREFIX=True; BETA=0.1; CONTROL=uniform ;;
  score_kn)       QW=True; AMP=True; PREFIX=True; BETA=0.1; CONTROL=kn ;;   # rebuttal: weights from group success rate k/N
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

if [ -n "${GPUS:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPUS; fi
if [ "$NGPU" -gt 1 ]; then export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; fi
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false
# vLLM sleep mode and expandable_segments both use the CUDA VMM API; keep it off
export PYTORCH_ALLOC_CONF=expandable_segments:False
# sandbox: persistent pool of plain subprocesses (see verl/workers/reward_manager/naive.py)
export VERL_SANDBOX_ISOLATION=${VERL_SANDBOX_ISOLATION:-subprocess}
export VERL_REWARD_WORKERS=${VERL_REWARD_WORKERS:-6}

mkdir -p "$OUT_DIR/$EXP"
cd "$REPO"
python3 -u -m verl.trainer.main_ppo --config-name score \
    data.train_files="['$DATA_DIR/train.parquet']" \
    data.val_files="['$DATA_DIR/test.parquet']" \
    data.seed=$SEED \
    "data.apply_chat_template_kwargs={enable_thinking: $THINKING}" \
    actor_rollout_ref.model.path=$HF \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS \
    actor_rollout_ref.actor.self_distillation.diagnosis_mode=$DIAG \
    actor_rollout_ref.actor.self_distillation.use_quadrant_weighting=$QW \
    actor_rollout_ref.actor.self_distillation.use_dynamic_weight=$DYN \
    actor_rollout_ref.actor.self_distillation.use_advantage_amplification=$AMP \
    actor_rollout_ref.actor.self_distillation.use_quadrant_prefix=$PREFIX \
    actor_rollout_ref.actor.self_distillation.critique_loss_beta=$BETA \
    actor_rollout_ref.actor.self_distillation.weighting_control=$CONTROL \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
    custom_reward_function.path=$REPO/verl/utils/reward_score/feedback/__init__.py \
    trainer.n_gpus_per_node=$NGPU \
    trainer.experiment_name=$EXP \
    trainer.default_local_dir=$OUT_DIR/$EXP \
    "${@:4}" \
    2>&1 | tee -a "$OUT_DIR/$EXP/train.log"
