# Screen scr-r1-zrckpr (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8019 | 0.8012 | -0.0007 [-0.0016, +0.0002] | WITHIN | 0.0047 | -0.0003 | +0.0019 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0011 [-0.0006, +0.0028] | WITHIN | 0.9054 | +0.0011 | -0.0019 |
| musique | zero-shot | 2417 | 0.5121 | 0.5197 | +0.0076 [+0.0028, +0.0125] | GAIN | 0.4734 | +0.0095 | +0.0004 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9083 | +0.0018 [-0.0003, +0.0038] | WITHIN | 0.6846 | +0.0051 | -0.0001 |
| 2wiki | in-domain | 12576 | 0.9347 | 0.9361 | +0.0014 [-0.0000, +0.0028] | WITHIN | 0.6079 | +0.0045 | -0.0022 |
| webqsp | zero-shot | 1503 | 0.3627 | 0.3762 | +0.0135 [+0.0032, +0.0239] | GAIN | 0.0535 | +0.0073 | +0.0106 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8012 | +0.1475 [+0.1418, +0.1529] | GAIN | 0.0047 | +0.1451 | +0.2831 |
| squad | in-domain | 11873 | 0.9100 | 0.9127 | +0.0028 [+0.0006, +0.0050] | WITHIN | 0.9054 | +0.0028 | -0.0029 |
| musique | zero-shot | 2417 | 0.2696 | 0.5197 | +0.2501 [+0.2382, +0.2619] | GAIN | 0.4734 | +0.1303 | +0.4137 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9083 | +0.0068 [+0.0034, +0.0101] | WITHIN | 0.6846 | +0.0093 | -0.0076 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9361 | +0.0629 [+0.0598, +0.0660] | GAIN | 0.6079 | +0.1308 | +0.0705 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3762 | +0.1984 [+0.1760, +0.2217] | GAIN | 0.0535 | +0.1657 | +0.0832 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
