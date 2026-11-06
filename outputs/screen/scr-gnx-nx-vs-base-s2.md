# Screen scr-gnx-nx-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-ml1-zg6r-s2 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3461 | 0.3902 | +0.0440 [+0.0388, +0.0491] | GAIN | 0.0047 | +0.0381 | +0.0143 |
| squad | in-domain | 11873 | 0.9163 | 0.9146 | -0.0017 [-0.0035, +0.0003] | WITHIN | 0.9054 | -0.0017 | -0.0013 |
| musique | zero-shot | 2417 | 0.5134 | 0.5248 | +0.0113 [+0.0053, +0.0172] | GAIN | 0.4734 | +0.0157 | +0.0376 |
| hotpotqa | in-domain | 7405 | 0.8562 | 0.8586 | +0.0024 [-0.0002, +0.0051] | WITHIN | 0.6846 | +0.0059 | -0.0023 |
| 2wiki | in-domain | 12576 | 0.9215 | 0.9243 | +0.0029 [+0.0012, +0.0047] | WITHIN | 0.6079 | +0.0100 | +0.0101 |
| webqsp | in-domain | 1503 | 0.6593 | 0.6456 | -0.0138 [-0.0235, -0.0044] | LOSS | 0.0535 | -0.0146 | +0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
