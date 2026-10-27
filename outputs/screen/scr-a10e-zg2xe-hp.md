# Screen scr-a10e-zg2xe-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8027 | -0.0012 [-0.0020, -0.0005] | WITHIN | 0.0047 | -0.0024 | -0.0056 |
| squad | in-domain | 11873 | 0.9129 | 0.9138 | +0.0008 [-0.0010, +0.0028] | WITHIN | 0.9054 | +0.0008 | -0.0035 |
| musique | in-domain | 2417 | 0.5901 | 0.5860 | -0.0041 [-0.0088, +0.0008] | WITHIN | 0.4734 | -0.0091 | -0.0062 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8537 | -0.0010 [-0.0035, +0.0016] | WITHIN | 0.6846 | -0.0030 | +0.0047 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9347 | +0.0034 [+0.0020, +0.0049] | WITHIN | 0.6079 | +0.0082 | -0.0013 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.2834 | -0.0080 [-0.0158, +0.0002] | WITHIN | 0.0535 | -0.0027 | -0.0033 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8027 | +0.1487 [+0.1430, +0.1541] | GAIN | 0.0047 | +0.1525 | +0.2869 |
| squad | in-domain | 11873 | 0.9116 | 0.9138 | +0.0021 [-0.0001, +0.0044] | WITHIN | 0.9054 | +0.0021 | -0.0017 |
| musique | in-domain | 2417 | 0.5550 | 0.5860 | +0.0310 [+0.0234, +0.0385] | GAIN | 0.4734 | +0.0331 | +0.0488 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8537 | -0.0049 [-0.0099, +0.0001] | WITHIN | 0.6846 | -0.0096 | -0.0190 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9347 | +0.0674 [+0.0641, +0.0710] | GAIN | 0.6079 | +0.1393 | +0.0735 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2834 | +0.1634 [+0.1393, +0.1878] | GAIN | 0.0535 | +0.1244 | +0.0878 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
