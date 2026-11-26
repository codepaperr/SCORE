# Author-response add-ons

One folder per reviewer. Each folder lists the add-on experiments that answer that reviewer, the
commands that produce them, and the logs and results. The shared scripts are in this folder.

All add-ons use Qwen3-1.7B, seed 10, one GPU, and the hyperparameters of `verl/trainer/config/score.yaml`.

| Add-on | Command | Answers |
| --- | --- | --- |
| E1. SCORE in the paper's setting, with per-step logging | `bash rebuttal/run.sh v6 score 10` | TX5X W3, C2; gmkv W3; UfJ4 W3, C2 |
| E2. k/N-weighting baseline | `bash rebuttal/run.sh v6 score_kn 10` | UfJ4 W2 |
| E3. SCORE on a larger, problem-disjoint training set | `bash rebuttal/run.sh temporal score 10` | TX5X W1; gmkv W1; UfJ4 W2 |
| E4. SDPO on the same larger training set | `bash rebuttal/run.sh temporal sdpo 10` | TX5X W1; gmkv W1; UfJ4 W2 |

## What was added to the code

- `weighting_control=kn` in `RayPPOTrainer._score_reweight` (method `score_kn` in `train.sh`).
  Self-prediction is not used in the weights. Every failure of a problem with success rate
  `s = k/N` gets the distillation weight `(1 - s) + s * w_base * (1 + s)` and the advantage
  multiplier `1 + s * s`. The teacher prefix and the critique loss stay as in SCORE, as in the
  `random` and `uniform` controls of the paper.
- Per-step logging in `_score_reweight`. Each run writes `score_dynamics.jsonl` with the running
  blind-spot rate, the blind-spot and failure counts per error type, the composition of every
  problem's rollouts (A, B, C, D counts), the mean weight and multiplier on blind spots, and how
  often the 512-token diagnosis prompt was truncated for passing and for failing rollouts.

## Scripts

- `run.sh` starts one run, `eval.sh` evaluates a checkpoint on LCB-v6 and ErrCalBench.
- `analyze_dynamics.py` turns a `score_dynamics.jsonl` into the table of blind-spot rates, the
  share of problems that hold both a blind spot and a known failure, the AUROC of `k/N` for
  telling blind spots from known failures, and the truncation rates.

Logs and results are added to the reviewer folders as the runs finish.
