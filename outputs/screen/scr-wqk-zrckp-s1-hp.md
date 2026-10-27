# Screen scr-wqk-zrckp-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4193 | +0.0267 [+0.0208, +0.0325] | GAIN | 0.0047 | +0.0221 | -0.0051 |
| squad | in-domain | 11873 | 0.9117 | 0.9115 | -0.0003 [-0.0024, +0.0017] | WITHIN | 0.9054 | -0.0003 | -0.0004 |
| musique | in-domain | 2417 | 0.5743 | 0.5768 | +0.0024 [-0.0023, +0.0074] | WITHIN | 0.4734 | +0.0025 | +0.0128 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8547 | +0.0022 [-0.0010, +0.0055] | WITHIN | 0.6846 | +0.0059 | -0.0212 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9284 | -0.0015 [-0.0031, +0.0001] | WITHIN | 0.6079 | -0.0050 | -0.0006 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6417 | +0.0035 [-0.0052, +0.0127] | WITHIN | 0.0535 | +0.0000 | +0.0093 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
