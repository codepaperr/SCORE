# Reviewer gmkv

| Point | Add-on | Files |
| --- | --- | --- |
| W1. Small, repeatedly visited training set | E3 and E4: SCORE and SDPO trained on 737 older LiveCodeBench problems that share no problem with the test set (each problem seen about 5 times, not 30) | `E3_temporal_score/`, `E4_temporal_sdpo/` |
| W1. Is the weighting close to uniform when most failures are blind spots? | E3: blind-spot rate, weights and multipliers over training on the larger set | `E3_temporal_score/dynamics.txt` |
| W3. Training dynamics in the main setting | E1: blind-spot rate and counts over the 120 steps of the 1.7B thinking-mode run | `E1_v6_score/dynamics.txt` |

See `../README.md` for the commands.
