# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\step1\fits\L-2wiki (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.7793 | +0.1325 [+0.1270, +0.1377] | GAIN | 0.0047 | +0.1380 | +0.2607 |
| squad | in-domain | 11873 | 0.9112 | 0.9121 | +0.0008 [-0.0013, +0.0028] | WITHIN | 0.9054 | +0.0008 | -0.0025 |
| musique | in-domain | 2417 | 0.5601 | 0.5669 | +0.0068 [+0.0005, +0.0131] | WITHIN | 0.4734 | +0.0120 | +0.0099 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9001 | +0.0024 [-0.0006, +0.0052] | WITHIN | 0.6846 | +0.0043 | +0.0105 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.7937 | -0.0137 [-0.0162, -0.0110] | LOSS | 0.6079 | -0.0253 | +0.0099 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.3042 | +0.1966 [+0.1736, +0.2187] | GAIN | 0.0535 | +0.1564 | +0.0619 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
