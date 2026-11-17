# HR1 report (2026-11-17T14:59:57Z)

Declared in `docs/HR1_HOPRETRIEVER_ROUTE1.md`. Verdicts: ment NO_SIGNAL (carry ['2wiki', 'hotpotqa'], below ['squad']); co NO_SIGNAL (carry ['hotpotqa'], below ['2wiki'])

| dataset | family | A | interval (A - 0.5) | call | counted / eligible | A_miss | golds covered | median deg gold / non-gold |
|---|---|---|---|---|---|---|---|---|
| squad | ment | 0.39293 | [-0.11957, -0.09526] | BELOW | 5758 / 5758 | 0.45688 | 1.0 | 57.0 / 64.0 |
| squad | co | 0.49416 | [-0.02061, 0.00882] | LEVEL | 4197 / 5758 | 0.49357 | 0.80531 | 13.0 / 14.0 |
| musique | ment | 0.54972 | [0.04003, 0.05948] | ABOVE | 4500 / 4599 | 0.60494 | 0.93316 | 16.0 / 19.0 |
| musique | co | 0.52161 | [0.0116, 0.03215] | ABOVE | 4444 / 4599 | 0.56692 | 0.884 | 15.0 / 16.0 |
| hotpotqa | ment | 0.69272 | [0.18455, 0.20089] | ABOVE | 5868 / 5918 | 0.77679 | 0.99665 | 14.0 / 10.0 |
| hotpotqa | co | 0.5798 | [0.06908, 0.09002] | ABOVE | 5262 / 5918 | 0.65042 | 0.86547 | 14.0 / 12.0 |
| 2wiki | ment | 0.59307 | [0.08464, 0.101] | ABOVE | 5890 / 5928 | 0.67911 | 0.99993 | 20.0 / 18.0 |
| 2wiki | co | 0.48578 | [-0.02408, -0.00453] | BELOW | 5523 / 5928 | 0.52164 | 0.85582 | 17.0 / 17.0 |

| dataset | AUC s_u | AUC twin0 | AUC F_ment | AUC F_co | R@5 s_u | R@5 F_ment | R@5 F_co |
|---|---|---|---|---|---|---|---|
| squad | 0.96744 | 0.99113 | 0.92909 (BELOW) | 0.92764 (BELOW) | 0.91542 | 0.83206 (BELOW) | 0.81695 (BELOW) |
| musique | 0.95008 | 0.98838 | 0.9317 (BELOW) | 0.93356 (BELOW) | 0.49802 | 0.35943 (BELOW) | 0.35843 (BELOW) |
| hotpotqa | 0.90176 | 0.99225 | 0.9362 (ABOVE) | 0.90294 (LEVEL) | 0.67176 | 0.61609 (BELOW) | 0.58009 (BELOW) |
| 2wiki | 0.7736 | 0.99316 | 0.81968 (ABOVE) | 0.74955 (BELOW) | 0.50333 | 0.29441 (BELOW) | 0.35261 (BELOW) |
