# Screen scr-u2d-base-zspk-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8021 | 0.8018 | -0.0003 [-0.0014, +0.0007] | WITHIN | 0.0047 | +0.0003 | -0.0010 |
| squad | in-domain | 11873 | 0.9118 | 0.9124 | +0.0006 [-0.0013, +0.0025] | WITHIN | 0.9054 | +0.0006 | +0.0060 |
| musique | zero-shot | 2417 | 0.5143 | 0.5203 | +0.0060 [+0.0011, +0.0110] | WITHIN | 0.4734 | +0.0046 | +0.0103 |
| hotpotqa | in-domain | 7405 | 0.9088 | 0.9059 | -0.0030 [-0.0050, -0.0010] | WITHIN | 0.6846 | -0.0041 | -0.0035 |
| 2wiki | in-domain | 12576 | 0.9346 | 0.9343 | -0.0003 [-0.0017, +0.0011] | WITHIN | 0.6079 | -0.0013 | +0.0009 |
| webqsp | zero-shot | 1503 | 0.3162 | 0.3091 | -0.0071 [-0.0158, +0.0018] | WITHIN | 0.0535 | -0.0053 | +0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
