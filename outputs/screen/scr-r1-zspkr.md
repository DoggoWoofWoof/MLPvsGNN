# Screen scr-r1-zspkr (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.8026 | +0.0017 [+0.0008, +0.0026] | WITHIN | 0.0047 | +0.0014 | +0.0114 |
| squad | in-domain | 11873 | 0.9114 | 0.9135 | +0.0021 [+0.0000, +0.0041] | WITHIN | 0.9054 | +0.0021 | -0.0017 |
| musique | zero-shot | 2417 | 0.5012 | 0.4933 | -0.0079 [-0.0141, -0.0017] | LOSS | 0.4734 | +0.0033 | -0.0170 |
| hotpotqa | in-domain | 7405 | 0.9080 | 0.9076 | -0.0005 [-0.0026, +0.0017] | WITHIN | 0.6846 | -0.0011 | +0.0128 |
| 2wiki | in-domain | 12576 | 0.9383 | 0.9342 | -0.0041 [-0.0055, -0.0027] | WITHIN | 0.6079 | -0.0092 | -0.0032 |
| webqsp | zero-shot | 1503 | 0.3229 | 0.3224 | -0.0005 [-0.0089, +0.0077] | WITHIN | 0.0535 | +0.0000 | -0.0013 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8026 | +0.1488 [+0.1433, +0.1542] | GAIN | 0.0047 | +0.1468 | +0.2864 |
| squad | in-domain | 11873 | 0.9100 | 0.9135 | +0.0035 [+0.0014, +0.0058] | WITHIN | 0.9054 | +0.0035 | -0.0032 |
| musique | zero-shot | 2417 | 0.2696 | 0.4933 | +0.2237 [+0.2119, +0.2351] | GAIN | 0.4734 | +0.1196 | +0.3699 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9076 | +0.0061 [+0.0027, +0.0093] | WITHIN | 0.6846 | +0.0088 | +0.0178 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9342 | +0.0610 [+0.0579, +0.0642] | GAIN | 0.6079 | +0.1262 | +0.0691 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3224 | +0.1445 [+0.1220, +0.1692] | GAIN | 0.0535 | +0.1104 | +0.1264 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
