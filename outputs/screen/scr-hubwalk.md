# Screen scr-hubwalk (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6579 | +0.0041 [+0.0010, +0.0071] | WITHIN | 0.0047 | +0.0005 | +0.0075 |
| squad | in-domain | 11873 | 0.9100 | 0.9098 | -0.0002 [-0.0022, +0.0019] | WITHIN | 0.9054 | -0.0002 | -0.0030 |
| musique | zero-shot | 2417 | 0.2696 | 0.2408 | -0.0288 [-0.0367, -0.0205] | LOSS | 0.4734 | -0.0153 | -0.0443 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9024 | +0.0009 [-0.0014, +0.0032] | WITHIN | 0.6846 | +0.0009 | +0.0078 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8723 | -0.0009 [-0.0028, +0.0010] | WITHIN | 0.6079 | -0.0050 | +0.0021 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1647 | -0.0131 [-0.0237, -0.0025] | LOSS | 0.0535 | -0.0106 | -0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
