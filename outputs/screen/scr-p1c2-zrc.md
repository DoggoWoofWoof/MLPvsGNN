# Screen scr-p1c2-zrc (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7782 | 0.7994 | +0.0211 [+0.0183, +0.0238] | GAIN | 0.0047 | +0.0186 | +0.0239 |
| squad | in-domain | 11873 | 0.9105 | 0.9123 | +0.0019 [-0.0002, +0.0040] | WITHIN | 0.9054 | +0.0019 | -0.0052 |
| musique | zero-shot | 2417 | 0.5235 | 0.4789 | -0.0447 [-0.0521, -0.0372] | LOSS | 0.4734 | -0.0525 | -0.0186 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9036 | -0.0004 [-0.0034, +0.0026] | WITHIN | 0.6846 | -0.0030 | -0.0090 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.9128 | +0.0394 [+0.0367, +0.0423] | GAIN | 0.6079 | +0.0766 | +0.0228 |
| webqsp | zero-shot | 1503 | 0.3384 | 0.3466 | +0.0082 [-0.0065, +0.0232] | WITHIN | 0.0535 | +0.0080 | +0.0133 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7994 | +0.1456 [+0.1401, +0.1509] | GAIN | 0.0047 | +0.1452 | +0.2744 |
| squad | in-domain | 11873 | 0.9100 | 0.9123 | +0.0024 [+0.0002, +0.0046] | WITHIN | 0.9054 | +0.0024 | -0.0054 |
| musique | zero-shot | 2417 | 0.2696 | 0.4789 | +0.2093 [+0.1983, +0.2203] | GAIN | 0.4734 | +0.0943 | +0.3562 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9036 | +0.0022 [-0.0009, +0.0051] | WITHIN | 0.6846 | +0.0023 | +0.0016 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9128 | +0.0395 [+0.0368, +0.0424] | GAIN | 0.6079 | +0.0754 | +0.0306 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3466 | +0.1688 [+0.1476, +0.1916] | GAIN | 0.0535 | +0.1377 | +0.0659 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
