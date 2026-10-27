# Screen scr-p1gc-zrckpr (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8012 | 0.8004 | -0.0008 [-0.0024, +0.0008] | WITHIN | 0.0047 | +0.0011 | -0.0022 |
| squad | in-domain | 11873 | 0.9127 | 0.9135 | +0.0008 [-0.0010, +0.0026] | WITHIN | 0.9054 | +0.0008 | +0.0008 |
| musique | zero-shot | 2417 | 0.5197 | 0.5053 | -0.0144 [-0.0199, -0.0085] | LOSS | 0.4734 | -0.0074 | -0.0021 |
| hotpotqa | in-domain | 7405 | 0.9083 | 0.9059 | -0.0024 [-0.0049, +0.0001] | WITHIN | 0.6846 | -0.0047 | -0.0138 |
| 2wiki | in-domain | 12576 | 0.9361 | 0.9365 | +0.0004 [-0.0012, +0.0019] | WITHIN | 0.6079 | +0.0015 | -0.0017 |
| webqsp | zero-shot | 1503 | 0.3762 | 0.3873 | +0.0111 [-0.0002, +0.0218] | WITHIN | 0.0535 | +0.0080 | +0.0200 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8004 | +0.1467 [+0.1412, +0.1519] | GAIN | 0.0047 | +0.1462 | +0.2808 |
| squad | in-domain | 11873 | 0.9100 | 0.9135 | +0.0035 [+0.0013, +0.0057] | WITHIN | 0.9054 | +0.0035 | -0.0022 |
| musique | zero-shot | 2417 | 0.2696 | 0.5053 | +0.2357 [+0.2238, +0.2477] | GAIN | 0.4734 | +0.1229 | +0.4117 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9059 | +0.0045 [+0.0011, +0.0077] | WITHIN | 0.6846 | +0.0046 | -0.0213 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9365 | +0.0633 [+0.0600, +0.0666] | GAIN | 0.6079 | +0.1323 | +0.0688 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3873 | +0.2095 [+0.1882, +0.2327] | GAIN | 0.0535 | +0.1737 | +0.1031 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
