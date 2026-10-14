# Screen J5 (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7796 | 0.7989 | +0.0193 [+0.0163, +0.0219] | GAIN | 0.0047 | +0.0164 | +0.0112 |
| squad | in-domain | 11873 | 0.9107 | 0.9112 | +0.0005 [-0.0015, +0.0025] | WITHIN | 0.9054 | +0.0005 | +0.0028 |
| musique | in-domain | 2417 | 0.5669 | 0.5728 | +0.0059 [-0.0018, +0.0139] | WITHIN | 0.4734 | -0.0033 | +0.0190 |
| hotpotqa | in-domain | 7405 | 0.9049 | 0.9065 | +0.0016 [-0.0018, +0.0047] | WITHIN | 0.6846 | +0.0001 | -0.0113 |
| 2wiki | in-domain | 12576 | 0.8697 | 0.9304 | +0.0607 [+0.0574, +0.0639] | GAIN | 0.6079 | +0.1245 | +0.0769 |
| webqsp | zero-shot | 1503 | 0.3071 | 0.3611 | +0.0540 [+0.0388, +0.0698] | GAIN | 0.0535 | +0.0472 | +0.0299 |

## Against outputs\full_zrck\fits\J5 (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7992 | 0.7989 | -0.0003 [-0.0012, +0.0006] | WITHIN | 0.0047 | +0.0002 | -0.0001 |
| squad | in-domain | 11873 | 0.9108 | 0.9112 | +0.0004 [-0.0017, +0.0024] | WITHIN | 0.9054 | +0.0004 | +0.0030 |
| musique | in-domain | 2417 | 0.5688 | 0.5728 | +0.0040 [-0.0016, +0.0096] | WITHIN | 0.4734 | +0.0029 | -0.0190 |
| hotpotqa | in-domain | 7405 | 0.9037 | 0.9065 | +0.0028 [+0.0004, +0.0051] | WITHIN | 0.6846 | +0.0045 | -0.0109 |
| 2wiki | in-domain | 12576 | 0.9269 | 0.9304 | +0.0035 [+0.0017, +0.0053] | WITHIN | 0.6079 | +0.0150 | -0.0054 |
| webqsp | zero-shot | 1503 | 0.3600 | 0.3611 | +0.0011 [-0.0084, +0.0110] | WITHIN | 0.0535 | -0.0027 | +0.0007 |

## Against outputs\step1\fits\J5 (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7989 | +0.1546 [+0.1492, +0.1603] | GAIN | 0.0047 | +0.1575 | +0.2691 |
| squad | in-domain | 11873 | 0.9095 | 0.9112 | +0.0017 [-0.0007, +0.0040] | WITHIN | 0.9054 | +0.0017 | +0.0019 |
| musique | in-domain | 2417 | 0.5609 | 0.5728 | +0.0119 [+0.0040, +0.0200] | GAIN | 0.4734 | +0.0099 | +0.0248 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9065 | +0.0052 [+0.0020, +0.0084] | WITHIN | 0.6846 | +0.0073 | -0.0111 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.9304 | +0.0610 [+0.0577, +0.0642] | GAIN | 0.6079 | +0.1221 | +0.0647 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.3611 | +0.2444 [+0.2192, +0.2706] | GAIN | 0.0535 | +0.1963 | +0.0852 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
