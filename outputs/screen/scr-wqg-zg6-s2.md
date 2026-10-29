# Screen scr-wqg-zg6-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zg6-s0 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.3964 | -0.0428 [-0.0477, -0.0378] | LOSS | 0.0047 | -0.0326 | -0.0477 |
| squad | in-domain | 11873 | 0.9097 | 0.9124 | +0.0027 [+0.0008, +0.0046] | WITHIN | 0.9054 | +0.0027 | +0.0004 |
| musique | zero-shot | 2417 | 0.5018 | 0.5111 | +0.0093 [+0.0029, +0.0156] | GAIN | 0.4734 | +0.0046 | +0.0252 |
| hotpotqa | in-domain | 7405 | 0.9109 | 0.9126 | +0.0016 [-0.0005, +0.0036] | WITHIN | 0.6846 | +0.0054 | +0.0043 |
| 2wiki | in-domain | 12576 | 0.9390 | 0.9414 | +0.0024 [+0.0011, +0.0038] | WITHIN | 0.6079 | +0.0044 | +0.0099 |
| webqsp | in-domain | 1503 | 0.6423 | 0.6330 | -0.0093 [-0.0201, +0.0014] | WITHIN | 0.0535 | -0.0106 | -0.0067 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
