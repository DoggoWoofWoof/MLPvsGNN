# Screen scr-psms-zrckpg1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4110 | +0.0183 [+0.0132, +0.0237] | GAIN | 0.0047 | +0.0137 | +0.0033 |
| squad | in-domain | 11873 | 0.9117 | 0.9115 | -0.0003 [-0.0023, +0.0018] | WITHIN | 0.9054 | -0.0003 | +0.0013 |
| musique | in-domain | 2417 | 0.5743 | 0.5735 | -0.0008 [-0.0057, +0.0042] | WITHIN | 0.4734 | -0.0021 | +0.0041 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8409 | -0.0116 [-0.0148, -0.0085] | LOSS | 0.6846 | -0.0228 | -0.0169 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9281 | -0.0018 [-0.0035, -0.0001] | WITHIN | 0.6079 | -0.0025 | -0.0038 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6277 | -0.0105 [-0.0187, -0.0022] | LOSS | 0.0535 | -0.0153 | -0.0220 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
