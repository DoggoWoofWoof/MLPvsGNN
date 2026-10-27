# Screen scr-r2t-zg6m (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zg6 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8127 | 0.8123 | -0.0004 [-0.0015, +0.0007] | WITHIN | 0.0047 | -0.0020 | -0.0028 |
| squad | in-domain | 11873 | 0.9120 | 0.9106 | -0.0014 [-0.0035, +0.0006] | WITHIN | 0.9054 | -0.0014 | -0.0007 |
| musique | zero-shot | 2417 | 0.5138 | 0.5244 | +0.0107 [+0.0053, +0.0163] | GAIN | 0.4734 | +0.0087 | -0.0294 |
| hotpotqa | in-domain | 7405 | 0.9127 | 0.9143 | +0.0016 [-0.0003, +0.0037] | WITHIN | 0.6846 | +0.0034 | -0.0076 |
| 2wiki | in-domain | 12576 | 0.9448 | 0.9417 | -0.0031 [-0.0045, -0.0017] | WITHIN | 0.6079 | -0.0097 | -0.0083 |
| webqsp | zero-shot | 1503 | 0.2398 | 0.2182 | -0.0216 [-0.0393, -0.0047] | LOSS | 0.0535 | -0.0173 | -0.0166 |

## Against outputs\screen\fits\scr-kg-base-zspk (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8013 | 0.8123 | +0.0110 [+0.0092, +0.0126] | GAIN | 0.0047 | +0.0112 | +0.0369 |
| squad | in-domain | 11873 | 0.9102 | 0.9106 | +0.0003 [-0.0021, +0.0028] | WITHIN | 0.9054 | +0.0003 | +0.0022 |
| musique | zero-shot | 2417 | 0.5146 | 0.5244 | +0.0098 [+0.0028, +0.0168] | GAIN | 0.4734 | +0.0165 | -0.0174 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9143 | +0.0057 [+0.0035, +0.0081] | WITHIN | 0.6846 | +0.0103 | +0.0066 |
| 2wiki | in-domain | 12576 | 0.9364 | 0.9417 | +0.0053 [+0.0037, +0.0069] | WITHIN | 0.6079 | +0.0148 | +0.0043 |
| webqsp | zero-shot | 1503 | 0.3151 | 0.2182 | -0.0969 [-0.1184, -0.0747] | LOSS | 0.0535 | -0.0752 | -0.0585 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
