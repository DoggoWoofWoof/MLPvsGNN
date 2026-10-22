# Screen scr-u2d-new-zrckpv-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-new-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.8002 | -0.0007 [-0.0018, +0.0003] | WITHIN | 0.0047 | -0.0005 | +0.0036 |
| squad | in-domain | 11873 | 0.9131 | 0.9128 | -0.0003 [-0.0019, +0.0015] | WITHIN | 0.9054 | -0.0003 | -0.0017 |
| musique | in-domain | 2417 | 0.5741 | 0.5737 | -0.0003 [-0.0059, +0.0049] | WITHIN | 0.4734 | -0.0066 | +0.0033 |
| hotpotqa | zero-shot | 7405 | 0.8064 | 0.8087 | +0.0023 [-0.0009, +0.0055] | WITHIN | 0.6846 | +0.0043 | +0.0018 |
| 2wiki | in-domain | 12576 | 0.9029 | 0.9060 | +0.0030 [+0.0010, +0.0052] | WITHIN | 0.6079 | +0.0093 | -0.0013 |
| webqsp | zero-shot | 1503 | 0.3870 | 0.3520 | -0.0350 [-0.0478, -0.0217] | LOSS | 0.0535 | -0.0299 | -0.0326 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
