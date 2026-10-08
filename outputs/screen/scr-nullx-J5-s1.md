# Screen scr-nullx-J5-s1 (p@swa, s1eval carves)

## Against outputs\step1\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.6539 | +0.0097 [+0.0066, +0.0126] | GAIN | 0.0047 | +0.0112 | +0.0018 |
| squad | in-domain | 11873 | 0.9095 | 0.9071 | -0.0024 [-0.0045, -0.0004] | WITHIN | 0.9054 | -0.0024 | -0.0013 |
| musique | in-domain | 2417 | 0.5609 | 0.5661 | +0.0052 [-0.0007, +0.0108] | WITHIN | 0.4734 | +0.0153 | -0.0066 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9015 | +0.0002 [-0.0022, +0.0028] | WITHIN | 0.6846 | +0.0012 | +0.0028 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8722 | +0.0028 [+0.0010, +0.0047] | WITHIN | 0.6079 | +0.0061 | -0.0039 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.1373 | +0.0205 [+0.0120, +0.0296] | GAIN | 0.0535 | +0.0173 | +0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
