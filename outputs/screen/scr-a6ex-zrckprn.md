# Screen scr-a6ex-zrckprn (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8012 | 0.8016 | +0.0003 [-0.0005, +0.0012] | WITHIN | 0.0047 | +0.0001 | -0.0010 |
| squad | in-domain | 11873 | 0.9127 | 0.9125 | -0.0003 [-0.0022, +0.0017] | WITHIN | 0.9054 | -0.0003 | +0.0008 |
| musique | zero-shot | 2417 | 0.5197 | 0.5099 | -0.0098 [-0.0151, -0.0048] | LOSS | 0.4734 | -0.0070 | -0.0087 |
| hotpotqa | in-domain | 7405 | 0.9083 | 0.9097 | +0.0014 [-0.0008, +0.0036] | WITHIN | 0.6846 | +0.0020 | -0.0022 |
| 2wiki | in-domain | 12576 | 0.9361 | 0.9308 | -0.0053 [-0.0068, -0.0036] | WITHIN | 0.6079 | -0.0157 | -0.0104 |
| webqsp | zero-shot | 1503 | 0.3762 | 0.3644 | -0.0118 [-0.0202, -0.0033] | LOSS | 0.0535 | -0.0140 | -0.0173 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8016 | +0.1478 [+0.1421, +0.1531] | GAIN | 0.0047 | +0.1452 | +0.2821 |
| squad | in-domain | 11873 | 0.9100 | 0.9125 | +0.0025 [+0.0003, +0.0047] | WITHIN | 0.9054 | +0.0025 | -0.0022 |
| musique | zero-shot | 2417 | 0.2696 | 0.5099 | +0.2403 [+0.2282, +0.2523] | GAIN | 0.4734 | +0.1233 | +0.4050 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9097 | +0.0082 [+0.0049, +0.0115] | GAIN | 0.6846 | +0.0113 | -0.0097 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9308 | +0.0576 [+0.0546, +0.0608] | GAIN | 0.6079 | +0.1151 | +0.0600 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3644 | +0.1865 [+0.1651, +0.2099] | GAIN | 0.0535 | +0.1517 | +0.0659 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
