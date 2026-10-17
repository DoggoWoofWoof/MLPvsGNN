# Screen scr-p1c3-zspk-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8040 | +0.1500 [+0.1444, +0.1554] | GAIN | 0.0047 | +0.1538 | +0.2898 |
| squad | in-domain | 11873 | 0.9116 | 0.9135 | +0.0019 [-0.0003, +0.0040] | WITHIN | 0.9054 | +0.0019 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5754 | +0.0204 [+0.0128, +0.0281] | GAIN | 0.4734 | +0.0240 | +0.0501 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8499 | -0.0086 [-0.0133, -0.0038] | LOSS | 0.6846 | -0.0172 | -0.0396 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9333 | +0.0661 [+0.0628, +0.0697] | GAIN | 0.6079 | +0.1341 | +0.0736 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2680 | +0.1480 [+0.1237, +0.1724] | GAIN | 0.0535 | +0.1104 | +0.0912 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
