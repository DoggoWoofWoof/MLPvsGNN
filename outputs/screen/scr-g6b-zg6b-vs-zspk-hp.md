# Screen scr-g6b-zg6b-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8037 | 0.8115 | +0.0078 [+0.0061, +0.0094] | GAIN | 0.0047 | +0.0094 | +0.0266 |
| squad | in-domain | 11873 | 0.9139 | 0.9130 | -0.0009 [-0.0031, +0.0012] | WITHIN | 0.9054 | -0.0009 | +0.0029 |
| musique | in-domain | 2417 | 0.5754 | 0.5934 | +0.0180 [+0.0119, +0.0243] | GAIN | 0.4734 | +0.0331 | +0.0149 |
| hotpotqa | zero-shot | 7405 | 0.8550 | 0.8388 | -0.0162 [-0.0201, -0.0124] | LOSS | 0.6846 | -0.0181 | -0.0379 |
| 2wiki | in-domain | 12576 | 0.9265 | 0.9443 | +0.0177 [+0.0157, +0.0197] | GAIN | 0.6079 | +0.0567 | +0.0013 |
| webqsp | zero-shot | 1503 | 0.2640 | 0.2397 | -0.0243 [-0.0439, -0.0043] | LOSS | 0.0535 | -0.0166 | +0.0266 |

## Against outputs\screen\fits\scr-kg-base-zg6-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8115 | 0.8115 | -0.0000 [-0.0009, +0.0009] | WITHIN | 0.0047 | +0.0002 | -0.0009 |
| squad | in-domain | 11873 | 0.9122 | 0.9130 | +0.0008 [-0.0009, +0.0025] | WITHIN | 0.9054 | +0.0008 | -0.0024 |
| musique | in-domain | 2417 | 0.5953 | 0.5934 | -0.0019 [-0.0068, +0.0029] | WITHIN | 0.4734 | +0.0004 | -0.0008 |
| hotpotqa | zero-shot | 7405 | 0.8391 | 0.8388 | -0.0003 [-0.0032, +0.0026] | WITHIN | 0.6846 | +0.0000 | -0.0050 |
| 2wiki | in-domain | 12576 | 0.9463 | 0.9443 | -0.0020 [-0.0034, -0.0006] | WITHIN | 0.6079 | -0.0049 | +0.0122 |
| webqsp | zero-shot | 1503 | 0.2430 | 0.2397 | -0.0033 [-0.0153, +0.0092] | WITHIN | 0.0535 | -0.0020 | +0.0180 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
