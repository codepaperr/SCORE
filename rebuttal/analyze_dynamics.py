"""Summarise score_dynamics.jsonl of one run (written by the rebuttal logging in _score_reweight).

  python rebuttal/analyze_dynamics.py rebuttal/runs/<exp>/score_dynamics.jsonl

Reports, per block of 20 steps and over the whole run:
  - running blind-spot rate b~ (before the step's update) and the mean weights / multipliers
  - number of blind spots (B) and known failures (D), blind-spot rate among failures
  - how many problems contain BOTH a blind spot and a known failure, and the share of failures
    that sit in such mixed problems (a weight that depends only on the problem cannot separate these)
  - AUROC of the group success rate k/N for telling blind spots from known failures
  - truncation rate of the 512-token diagnosis prompt, for passing and failing rollouts
"""
import json
import sys

import numpy as np


def auroc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count one half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):          # average ranks over ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def block(recs):
    B = sum(r["quad"]["B"] for r in recs)
    D = sum(r["quad"]["D"] for r in recs)
    fail_groups = mixed = fails_in_mixed = 0
    s_blind, s_known = [], []
    for r in recs:
        for a, b, c, d in r["groups"]:
            n = a + b + c + d
            if b + d == 0 or n == 0:
                continue
            fail_groups += 1
            s = (a + c) / n
            s_blind += [s] * b
            s_known += [s] * d
            if b > 0 and d > 0:
                mixed += 1
                fails_in_mixed += b + d
    rate = [np.mean(list(r["blind_rate_before"].values())) for r in recs]
    mult = [r["mult_blind_mean"] for r in recs if r.get("mult_blind_mean") is not None]
    w = [r["w_blind_mean"] for r in recs if r.get("w_blind_mean") is not None]
    tp = np.sum([r.get("trunc_pass", [0, 0]) for r in recs], axis=0)
    tf = np.sum([r.get("trunc_fail", [0, 0]) for r in recs], axis=0)
    return {
        "B": B, "D": D, "bsr": B / max(1, B + D), "b_tilde": float(np.mean(rate)),
        "w_blind": float(np.mean(w)) if w else float("nan"),
        "mult_blind": float(np.mean(mult)) if mult else float("nan"),
        "mixed_groups": mixed / max(1, fail_groups), "fails_in_mixed": fails_in_mixed / max(1, B + D),
        "auroc_kn": auroc(s_blind, s_known),
        "trunc_pass": tp[0] / max(1, tp[1]), "trunc_fail": tf[0] / max(1, tf[1]),
    }


def main():
    recs = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    recs = list({r["step"]: r for r in recs}.values())      # a resumed run may repeat a step
    recs.sort(key=lambda r: r["step"])
    print(f"{sys.argv[1]}: {len(recs)} steps")
    head = f"{'steps':>9s} {'B':>6s} {'D':>6s} {'B/(B+D)':>8s} {'b~':>6s} {'w_B':>6s} {'mult_B':>7s} {'mixed':>6s} {'fail_in_mixed':>14s} {'AUROC(k/N)':>11s} {'trunc_pass':>11s} {'trunc_fail':>11s}"
    print(head)
    rows = [(f"{recs[i]['step']}-{recs[min(i + 19, len(recs) - 1)]['step']}", block(recs[i:i + 20]))
            for i in range(0, len(recs), 20)] + [("all", block(recs))]
    for name, b in rows:
        print(f"{name:>9s} {b['B']:6d} {b['D']:6d} {b['bsr']:8.3f} {b['b_tilde']:6.3f} {b['w_blind']:6.2f} "
              f"{b['mult_blind']:7.3f} {b['mixed_groups']:6.3f} {b['fails_in_mixed']:14.3f} {b['auroc_kn']:11.3f} "
              f"{b['trunc_pass']:11.3f} {b['trunc_fail']:11.3f}")


if __name__ == "__main__":
    main()
