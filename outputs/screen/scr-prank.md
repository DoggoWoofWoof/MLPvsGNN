# Screen scr-prank (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6228 | -0.0310 [-0.0351, -0.0269] | LOSS | 0.0047 | -0.0210 | -0.0228 |
| squad | in-domain | 11873 | 0.9100 | 0.9111 | +0.0011 [-0.0017, +0.0037] | WITHIN | 0.9054 | +0.0011 | -0.0078 |
| musique | zero-shot | 2417 | 0.2696 | 0.4712 | +0.2016 [+0.1901, +0.2126] | GAIN | 0.4734 | +0.1241 | +0.2520 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8935 | -0.0080 [-0.0108, -0.0054] | LOSS | 0.6846 | -0.0163 | +0.0082 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8664 | -0.0068 [-0.0091, -0.0047] | WITHIN | 0.6079 | -0.0128 | -0.0003 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.2047 | +0.0269 [+0.0133, +0.0410] | GAIN | 0.0535 | +0.0259 | +0.0067 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
