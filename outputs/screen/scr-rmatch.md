# Screen scr-rmatch (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7805 | +0.1268 [+0.1217, +0.1320] | GAIN | 0.0047 | +0.1286 | +0.2550 |
| squad | in-domain | 11873 | 0.9100 | 0.9106 | +0.0007 [-0.0014, +0.0029] | WITHIN | 0.9054 | +0.0007 | +0.0001 |
| musique | zero-shot | 2417 | 0.2696 | 0.5020 | +0.2324 [+0.2212, +0.2436] | GAIN | 0.4734 | +0.1121 | +0.3579 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9038 | +0.0023 [+0.0000, +0.0045] | WITHIN | 0.6846 | +0.0051 | +0.0117 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8736 | +0.0003 [-0.0015, +0.0021] | WITHIN | 0.6079 | +0.0014 | -0.0004 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3155 | +0.1376 [+0.1152, +0.1600] | GAIN | 0.0535 | +0.1151 | +0.1011 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
