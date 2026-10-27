# Screen scr-a7p-zspks (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8004 | 0.8014 | +0.0010 [-0.0001, +0.0022] | WITHIN | 0.0047 | +0.0009 | -0.0020 |
| squad | in-domain | 11873 | 0.9129 | 0.9142 | +0.0013 [-0.0004, +0.0029] | WITHIN | 0.9054 | +0.0013 | -0.0028 |
| musique | zero-shot | 2417 | 0.5169 | 0.5180 | +0.0012 [-0.0042, +0.0066] | WITHIN | 0.4734 | +0.0041 | -0.0232 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9103 | +0.0017 [-0.0006, +0.0040] | WITHIN | 0.6846 | +0.0031 | -0.0030 |
| 2wiki | in-domain | 12576 | 0.9344 | 0.9388 | +0.0044 [+0.0030, +0.0059] | WITHIN | 0.6079 | +0.0134 | -0.0068 |
| webqsp | zero-shot | 1503 | 0.3354 | 0.3389 | +0.0036 [-0.0060, +0.0132] | WITHIN | 0.0535 | +0.0047 | +0.0033 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8014 | +0.1476 [+0.1422, +0.1528] | GAIN | 0.0047 | +0.1466 | +0.2769 |
| squad | in-domain | 11873 | 0.9100 | 0.9142 | +0.0042 [+0.0019, +0.0065] | WITHIN | 0.9054 | +0.0042 | -0.0024 |
| musique | zero-shot | 2417 | 0.2696 | 0.5180 | +0.2484 [+0.2364, +0.2603] | GAIN | 0.4734 | +0.1365 | +0.3943 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9103 | +0.0088 [+0.0055, +0.0122] | GAIN | 0.6846 | +0.0127 | -0.0188 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9388 | +0.0656 [+0.0624, +0.0689] | GAIN | 0.6079 | +0.1370 | +0.0643 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3389 | +0.1611 [+0.1373, +0.1853] | GAIN | 0.0535 | +0.1257 | +0.1271 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
