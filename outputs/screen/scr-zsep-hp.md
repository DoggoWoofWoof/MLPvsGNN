# Screen scr-zsep-hp (p@swa, s1eval carves)

## Against outputs\full_zret\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.6524 | +0.0104 [+0.0073, +0.0135] | GAIN | 0.0047 | +0.0099 | +0.0216 |
| squad | in-domain | 11873 | 0.9127 | 0.9112 | -0.0015 [-0.0037, +0.0007] | WITHIN | 0.9054 | -0.0015 | -0.0014 |
| musique | in-domain | 2417 | 0.5623 | 0.5522 | -0.0101 [-0.0163, -0.0043] | LOSS | 0.4734 | -0.0066 | -0.0012 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8419 | -0.0026 [-0.0061, +0.0009] | WITHIN | 0.6846 | -0.0078 | +0.0024 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8725 | -0.0003 [-0.0024, +0.0018] | WITHIN | 0.6079 | -0.0043 | +0.0120 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.1189 | -0.0006 [-0.0105, +0.0088] | WITHIN | 0.0535 | +0.0027 | -0.0073 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6524 | -0.0016 [-0.0053, +0.0018] | WITHIN | 0.0047 | -0.0001 | +0.0121 |
| squad | in-domain | 11873 | 0.9116 | 0.9112 | -0.0004 [-0.0029, +0.0019] | WITHIN | 0.9054 | -0.0004 | -0.0001 |
| musique | in-domain | 2417 | 0.5550 | 0.5522 | -0.0028 [-0.0093, +0.0041] | WITHIN | 0.4734 | +0.0108 | +0.0116 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8419 | -0.0167 [-0.0207, -0.0126] | LOSS | 0.6846 | -0.0338 | +0.0030 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8725 | +0.0053 [+0.0030, +0.0078] | WITHIN | 0.6079 | +0.0115 | +0.0085 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1189 | -0.0011 [-0.0108, +0.0095] | WITHIN | 0.0535 | +0.0007 | -0.0120 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
