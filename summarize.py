#!/usr/bin/env python3
"""Aggregate eval.py outputs into the paper tables.

Expects results/<method>_qwen3-<model>_s<seed>/<benchmark>.json (the layout used in README).
Prints mean +- sample std (n-1) over seeds 10/20/30 and two-sided Welch t-tests of SCORE vs
SDPO and vs GRPO on pass@1 (paper Appendix "Significance Tests").

    python summarize.py --results results
"""

import argparse
import json
import os
import statistics

from scipy.stats import ttest_ind

SEEDS = [10, 20, 30]
MODELS = ["1.7b", "4b", "8b"]
METHODS = ["grpo", "sdpo", "score", "score_noamp", "score_nodyn", "score_noprefix", "score_nocrit",
           "score_random", "score_uniform"]
BENCHES = ["humaneval_plus", "apps", "errcalbench", "lcb"]
METRICS = ["pass@1", "pass@4", "bacc", "bsr"]


def load(root, method, model, seed, bench):
    p = os.path.join(root, f"{method}_qwen3-{model}_s{seed}", f"{bench}.json")
    return json.load(open(p))["metrics"] if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    vals = {}
    for model in MODELS:
        for method in METHODS:
            for bench in BENCHES:
                runs = [load(args.results, method, model, s, bench) for s in SEEDS]
                runs = [r for r in runs if r is not None]
                if not runs:
                    continue
                vals[(model, method, bench)] = runs
                cells = []
                for m in METRICS:
                    xs = [r[m] for r in runs]
                    sd = statistics.stdev(xs) if len(xs) > 1 else float("nan")
                    cells.append(f"{m}={statistics.mean(xs):.1f}+-{sd:.1f}")
                print(f"{model:5s} {method:15s} {bench:15s} seeds={len(runs)}  " + "  ".join(cells))

    print("\nWelch t-test on pass@1 (two-sided)")
    for model in MODELS:
        for bench in BENCHES:
            score = vals.get((model, "score", bench))
            for base in ("sdpo", "grpo"):
                other = vals.get((model, base, bench))
                if not score or not other or len(score) < 2 or len(other) < 2:
                    continue
                a, b = [r["pass@1"] for r in score], [r["pass@1"] for r in other]
                t, p = ttest_ind(a, b, equal_var=False)
                diff = statistics.mean(a) - statistics.mean(b)
                print(f"{model:5s} {bench:15s} score vs {base}: diff={diff:+.1f} t={t:.2f} p={p:.3f}")


if __name__ == "__main__":
    main()
