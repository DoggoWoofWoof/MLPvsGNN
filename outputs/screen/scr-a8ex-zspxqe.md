# Screen scr-a8ex-zspxqe (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8033 | +0.0003 [-0.0006, +0.0012] | WITHIN | 0.0047 | +0.0014 | +0.0013 |
| squad | in-domain | 11873 | 0.9092 | 0.9109 | +0.0017 [-0.0002, +0.0036] | WITHIN | 0.9054 | +0.0017 | -0.0015 |
| musique | zero-shot | 2417 | 0.5073 | 0.5099 | +0.0027 [-0.0033, +0.0084] | WITHIN | 0.4734 | +0.0000 | +0.0211 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9064 | -0.0009 [-0.0032, +0.0011] | WITHIN | 0.6846 | -0.0003 | -0.0093 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9294 | -0.0057 [-0.0072, -0.0042] | WITHIN | 0.6079 | -0.0163 | +0.0037 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3221 | +0.0018 [-0.0049, +0.0084] | WITHIN | 0.0535 | +0.0060 | -0.0080 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8033 | +0.1496 [+0.1439, +0.1548] | GAIN | 0.0047 | +0.1488 | +0.2908 |
| squad | in-domain | 11873 | 0.9100 | 0.9109 | +0.0009 [-0.0013, +0.0032] | WITHIN | 0.9054 | +0.0009 | -0.0034 |
| musique | zero-shot | 2417 | 0.2696 | 0.5099 | +0.2403 [+0.2277, +0.2528] | GAIN | 0.4734 | +0.1274 | +0.3715 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9064 | +0.0049 [+0.0016, +0.0083] | WITHIN | 0.6846 | +0.0070 | +0.0038 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9294 | +0.0561 [+0.0530, +0.0594] | GAIN | 0.6079 | +0.1110 | +0.0686 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3221 | +0.1443 [+0.1205, +0.1685] | GAIN | 0.0535 | +0.1131 | +0.1218 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
