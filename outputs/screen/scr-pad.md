# Screen scr-pad (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6502 | -0.0036 [-0.0063, -0.0007] | WITHIN | 0.0047 | -0.0066 | +0.0015 |
| squad | in-domain | 11873 | 0.9100 | 0.9095 | -0.0005 [-0.0026, +0.0016] | WITHIN | 0.9054 | -0.0005 | -0.0008 |
| musique | zero-shot | 2417 | 0.2696 | 0.3527 | +0.0831 [+0.0742, +0.0920] | GAIN | 0.4734 | +0.0422 | +0.1038 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8929 | -0.0086 [-0.0115, -0.0059] | LOSS | 0.6846 | -0.0180 | -0.0003 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8674 | -0.0058 [-0.0079, -0.0035] | WITHIN | 0.6079 | -0.0123 | -0.0060 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1780 | +0.0001 [-0.0095, +0.0099] | WITHIN | 0.0535 | +0.0020 | -0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
