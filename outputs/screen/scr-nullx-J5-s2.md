# Screen scr-nullx-J5-s2 (p@swa, s1eval carves)

## Against outputs\step1\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.6492 | +0.0049 [+0.0018, +0.0082] | WITHIN | 0.0047 | +0.0067 | -0.0007 |
| squad | in-domain | 11873 | 0.9095 | 0.9074 | -0.0021 [-0.0042, +0.0000] | WITHIN | 0.9054 | -0.0021 | -0.0011 |
| musique | in-domain | 2417 | 0.5609 | 0.5707 | +0.0099 [+0.0044, +0.0158] | GAIN | 0.4734 | +0.0232 | -0.0066 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9003 | -0.0010 [-0.0038, +0.0016] | WITHIN | 0.6846 | -0.0007 | -0.0036 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8697 | +0.0003 [-0.0015, +0.0021] | WITHIN | 0.6079 | -0.0001 | -0.0017 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.1128 | -0.0040 [-0.0125, +0.0050] | WITHIN | 0.0535 | -0.0027 | +0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
