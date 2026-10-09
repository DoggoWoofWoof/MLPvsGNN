# Screen scr-zkind (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7782 | 0.7786 | +0.0004 [-0.0006, +0.0014] | WITHIN | 0.0047 | +0.0014 | -0.0024 |
| squad | in-domain | 11873 | 0.9105 | 0.9110 | +0.0005 [-0.0012, +0.0022] | WITHIN | 0.9054 | +0.0005 | -0.0005 |
| musique | zero-shot | 2417 | 0.5235 | 0.5331 | +0.0096 [+0.0042, +0.0146] | GAIN | 0.4734 | +0.0141 | -0.0285 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9028 | -0.0013 [-0.0035, +0.0008] | WITHIN | 0.6846 | -0.0036 | -0.0062 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.8687 | -0.0046 [-0.0064, -0.0028] | WITHIN | 0.6079 | -0.0052 | -0.0116 |
| webqsp | zero-shot | 1503 | 0.3384 | 0.3296 | -0.0089 [-0.0245, +0.0075] | WITHIN | 0.0535 | -0.0086 | +0.0745 |

## Against outputs\screen\fits\scr-zrm (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7786 | -0.0003 [-0.0014, +0.0007] | WITHIN | 0.0047 | +0.0007 | -0.0040 |
| squad | in-domain | 11873 | 0.9111 | 0.9110 | -0.0001 [-0.0019, +0.0019] | WITHIN | 0.9054 | -0.0001 | +0.0010 |
| musique | zero-shot | 2417 | 0.5231 | 0.5331 | +0.0100 [+0.0046, +0.0148] | GAIN | 0.4734 | +0.0112 | -0.0381 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.9028 | -0.0012 [-0.0034, +0.0009] | WITHIN | 0.6846 | -0.0014 | +0.0057 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8687 | -0.0022 [-0.0040, -0.0004] | WITHIN | 0.6079 | +0.0002 | -0.0017 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3296 | +0.0168 [+0.0048, +0.0291] | GAIN | 0.0535 | +0.0113 | +0.0213 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7786 | +0.1249 [+0.1197, +0.1303] | GAIN | 0.0047 | +0.1281 | +0.2481 |
| squad | in-domain | 11873 | 0.9100 | 0.9110 | +0.0010 [-0.0011, +0.0032] | WITHIN | 0.9054 | +0.0010 | -0.0007 |
| musique | zero-shot | 2417 | 0.2696 | 0.5331 | +0.2635 [+0.2524, +0.2745] | GAIN | 0.4734 | +0.1609 | +0.3463 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9028 | +0.0013 [-0.0014, +0.0036] | WITHIN | 0.6846 | +0.0016 | +0.0045 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8687 | -0.0045 [-0.0064, -0.0025] | WITHIN | 0.6079 | -0.0064 | -0.0038 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3296 | +0.1517 [+0.1286, +0.1749] | GAIN | 0.0535 | +0.1211 | +0.1271 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
