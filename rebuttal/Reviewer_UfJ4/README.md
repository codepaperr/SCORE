# Reviewer UfJ4

| Point | Add-on | Files |
| --- | --- | --- |
| W2. k/N-weighting baseline | E2: failures weighted by the group success rate k/N, without self-prediction, compared with E1 | `E2_v6_score_kn/`, `E1_v6_score/` |
| W2. Larger training set | E3 and E4: SCORE and SDPO trained on 737 older LiveCodeBench problems that share no problem with the test set | `E3_temporal_score/`, `E4_temporal_sdpo/` |
| W3. Running rate b~ and number of blind-spot samples over training for the main setting | E1: per-step log and its summary | `E1_v6_score/dynamics.txt`, `E1_v6_score/score_dynamics.jsonl` |
| C2. Truncation of the 512-token diagnosis prompt | E1: truncation rate for passing and for failing rollouts | `E1_v6_score/dynamics.txt` |

See `../README.md` for the commands.
