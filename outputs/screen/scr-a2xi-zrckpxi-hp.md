# Screen scr-a2xi-zrckpxi-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8044 | 0.8061 | +0.0017 [+0.0006, +0.0028] | WITHIN | 0.0047 | +0.0022 | +0.0055 |
| squad | in-domain | 11873 | 0.9127 | 0.9096 | -0.0031 [-0.0054, -0.0007] | WITHIN | 0.9054 | -0.0031 | -0.0147 |
| musique | in-domain | 2417 | 0.5745 | 0.5564 | -0.0181 [-0.0243, -0.0118] | LOSS | 0.4734 | -0.0182 | -0.0294 |
| hotpotqa | zero-shot | 7405 | 0.8513 | 0.8364 | -0.0149 [-0.0186, -0.0113] | LOSS | 0.6846 | -0.0271 | -0.0258 |
| 2wiki | in-domain | 12576 | 0.9350 | 0.9340 | -0.0011 [-0.0027, +0.0008] | WITHIN | 0.6079 | -0.0030 | -0.0010 |
| webqsp | zero-shot | 1503 | 0.3475 | 0.3507 | +0.0032 [-0.0107, +0.0175] | WITHIN | 0.0535 | -0.0007 | +0.0286 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8061 | +0.1521 [+0.1465, +0.1575] | GAIN | 0.0047 | +0.1562 | +0.2939 |
| squad | in-domain | 11873 | 0.9116 | 0.9096 | -0.0020 [-0.0047, +0.0007] | WITHIN | 0.9054 | -0.0020 | -0.0123 |
| musique | in-domain | 2417 | 0.5550 | 0.5564 | +0.0013 [-0.0071, +0.0096] | WITHIN | 0.4734 | +0.0000 | +0.0037 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8364 | -0.0221 [-0.0272, -0.0173] | LOSS | 0.6846 | -0.0423 | -0.0602 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9340 | +0.0668 [+0.0635, +0.0703] | GAIN | 0.6079 | +0.1380 | +0.0636 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3507 | +0.2307 [+0.2048, +0.2559] | GAIN | 0.0535 | +0.1843 | +0.0971 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
