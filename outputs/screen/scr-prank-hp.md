# Screen scr-prank-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6220 | -0.0320 [-0.0362, -0.0278] | LOSS | 0.0047 | -0.0174 | -0.0347 |
| squad | in-domain | 11873 | 0.9116 | 0.9081 | -0.0035 [-0.0061, -0.0008] | WITHIN | 0.9054 | -0.0035 | -0.0084 |
| musique | in-domain | 2417 | 0.5550 | 0.5484 | -0.0067 [-0.0133, -0.0000] | WITHIN | 0.4734 | +0.0083 | -0.0112 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8330 | -0.0255 [-0.0298, -0.0214] | LOSS | 0.6846 | -0.0481 | -0.0024 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8668 | -0.0004 [-0.0028, +0.0021] | WITHIN | 0.6079 | +0.0040 | -0.0060 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1253 | +0.0054 [-0.0051, +0.0167] | WITHIN | 0.0535 | +0.0067 | -0.0146 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
