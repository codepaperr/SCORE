# SCORE: Self-Evaluating On-Policy Distillation for Code LLMs

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Built on verl](https://img.shields.io/badge/Built%20on-verl-6f10c1.svg)](https://github.com/volcengine/verl)

</div>

Code for **SCORE** (Self-Correctness-Oriented REweighting). During sandbox execution, the model
predicts whether its own rollouts pass; failures it wrongly expected to pass (*blind spots*) are
upweighted in distillation and GRPO.

The code is a fork of [verl](https://github.com/volcengine/verl) (FSDP training + vLLM rollout).

## Layout

```
train.sh        one training run: bash train.sh <model> <method> <seed>
eval.py         pass@1 / pass@4 / BAcc / BSR on LCB-v6, HumanEval+, APPS-300, ErrCalBench
summarize.py    mean +- std over seeds and Welch t-tests (paper tables)
verl/           verl fork; the SCORE-specific code is listed below
```

| Paper | Code |
|---|---|
| All hyperparameters (Table 7) | `verl/trainer/config/score.yaml` |
| Self-prediction pass, diagnosis prompt (Eq. 4, Sec. 3.1, App. C) | `RayPPOTrainer._run_self_prediction`, `verl/trainer/ppo/score_utils.py` |
| Quadrants (Table 1), w (Eq. 7), running blind-spot rate b~ (Eq. 5-6), A-hat (Eq. 8), controls (Sec. 4.3) | `RayPPOTrainer._score_reweight` |
| Critique targets z (Eq. 9) | `RayPPOTrainer._build_critique_batch` |
| Teacher prompt x~ = [x; y; f] and quadrant prefixes (Eq. 2, App. C) | `RayPPOTrainer._build_teacher_batch` |
| Total loss L_PG + alpha L_distill + beta L_critique (Eq. 10) | `DataParallelPPOActor.update_policy` (`verl/workers/actor/dp_actor.py`) |
| Top-K reverse KL with IS clip (Eq. 3) | `compute_self_distillation_loss` (`verl/trainer/ppo/core_algos.py`) |
| Sandbox, outcome types, 10 s per test | `verl/utils/reward_score/feedback/code.py` |
| EMA teacher | `DataParallelPPOActor._update_teacher` |

## Install

Python 3.12, CUDA 12.x, vLLM 0.11, torch 2.8.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip sync requirements.txt
uv pip install -e . --no-deps
```

## Data

The data files are released at https://github.com/codepaperr/ErrCalBench. Place them under `data/`:

```
data/lcb_v6/train.parquet   131 LiveCodeBench-v6 problems; tests used for the training reward
data/lcb_v6/test.parquet    the same 131 problems with the held-out half of the tests
data/apps_300.jsonl         fixed 300-problem APPS subset (100 per difficulty)
data/ErrCalBench.jsonl      ErrCalBench, https://github.com/codepaperr/ErrCalBench
```

The LCB parquet files use the verl format: `prompt` (chat messages), `data_source="livecodebench"`,
`reward_model.ground_truth` (JSON with `inputs`, `outputs`, `testtype`, `fn_name`) and
`extra_info.split`. HumanEval+ is loaded by `evalplus`.

## Training

```bash
bash train.sh 1.7b score 10          # Qwen3-1.7B, SCORE, seed 10
GPUS=0,1 bash train.sh 4b sdpo 20
GPUS=0,1,2,3 bash train.sh 8b grpo 30
```

| Argument | Values |
|---|---|
| model | `1.7b` (1 GPU, binary self-prediction), `4b` (2 GPUs, 5-class), `8b` (4 GPUs, 5-class, offloading) |
| method | `grpo`, `sdpo`, `score` (Table 2) |
| | `score_noamp`, `score_nodyn`, `score_noprefix`, `score_nocrit` (Table 3, Qwen3-1.7B) |
| | `score_random`, `score_uniform` (Table 4, Qwen3-1.7B) |
| seed | `10`, `20`, `30` |

All methods run the self-prediction pass. GRPO and SDPO use it only for measurement and have no
critique loss. A run takes 120 steps (30 epochs, 32 problems x 16 rollouts per step). Checkpoints
go to `outputs/<method>_qwen3-<model>_s<seed>/global_step_*`, every logged scalar to
`metrics.jsonl` in the same directory.

Full grid of the paper (27 main runs plus 18 ablation and control runs):

```bash
for seed in 10 20 30; do
  for model in 1.7b 4b 8b; do for m in grpo sdpo score; do bash train.sh $model $m $seed; done; done
  for m in score_noamp score_nodyn score_noprefix score_nocrit score_random score_uniform; do
    bash train.sh 1.7b $m $seed
  done
done
```

The training-dynamics figure (Fig. 2) and the overhead table use single analysis runs, seed 10,
non-thinking, with the Table 7 hyperparameters (w_base = 2.0): SDPO and SCORE on Qwen3-0.6B for
the figure, SCORE on Qwen3-0.6B and 1.7B for the overhead table. They read `score/bsr`,
`score/bacc`, `score/quad_*` and `bubble/*` from `metrics.jsonl`.

```bash
THINKING=false bash train.sh 0.6b sdpo 10
THINKING=false bash train.sh 0.6b score 10     # Fig. 2 and overhead table
THINKING=false bash train.sh 1.7b score 10     # overhead table
```

## Evaluation

```bash
EXP=score_qwen3-1.7b_s10
CKPT=outputs/$EXP/global_step_120/actor
for b in humaneval_plus apps errcalbench lcb; do
  case $b in apps) D=data/apps_300.jsonl;; errcalbench) D=data/ErrCalBench.jsonl;; lcb) D=data/lcb_v6/test.parquet;; *) D=;; esac
  python eval.py --model $CKPT --benchmark $b ${D:+--data $D} --diagnosis binary --output results/$EXP/$b.json
done
python summarize.py --results results
```

Use `--diagnosis five_class --tp 2` for 4B and 8B. Evaluate the base models the same way with
`--model Qwen/Qwen3-1.7B`. Each run uses n = 16 samples per problem at temperature 0.6 and
top-p 0.95. pass@1 is avg@16 and pass@4 is the fraction of problems solved by at least one of four
samples (the first four of the 16). BAcc and BSR measure the pass/fail self-prediction; the
diagnosis pass uses the training settings (temperature 0.1, top-p 0.95, 64 tokens).

## Implementation choices not spelled out in the paper

- The teacher sees the rollout's final response with the thinking trace removed. The attempt is
  truncated so the teacher prompt stays within 10,240 tokens.
- If the diagnosis prompt exceeds 512 tokens, the solution is truncated first, keeping at least
  half of the budget, and then the problem.
- The predicted outcome is the first option letter in the answer. An answer without a valid
  letter counts as "pass". Diagnosis sampling uses temperature 0.1 and top-p 0.95, the rollout
  value.
- One update per mini-batch of one problem (16 rollouts). PPO clip 0.2. Token-level rollout
  importance sampling with threshold 2.0.
