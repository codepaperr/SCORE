# Reviewer TX5X

| Point | Add-on | Files |
| --- | --- | --- |
| W1. Does the mechanism carry over to a regime with fewer repeats? | E3 and E4: SCORE and SDPO trained on 737 older LiveCodeBench problems that share no problem with the test set (each problem seen about 5 times, not 30) | `E3_temporal_score/`, `E4_temporal_sdpo/` |
| W1. Does a blind spot only encode which problems are hard? | E1: share of problems that hold both a blind spot and a known failure, and AUROC of the group success rate for telling the two apart | `E1_v6_score/dynamics.txt` |
| W3. Self-assessment during training in the main setting | E1: blind-spot rate and counts over the 120 steps of the 1.7B thinking-mode run | `E1_v6_score/dynamics.txt` |
| C2. Truncation of the 512-token diagnosis prompt | E1: truncation rate for passing and for failing rollouts | `E1_v6_score/dynamics.txt` |

See `../README.md` for the commands.
