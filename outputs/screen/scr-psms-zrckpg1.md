# Screen scr-psms-zrckpg1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4617 | +0.0224 [+0.0172, +0.0273] | GAIN | 0.0047 | +0.0107 | +0.0321 |
| squad | in-domain | 11873 | 0.9106 | 0.9090 | -0.0015 [-0.0034, +0.0003] | WITHIN | 0.9054 | -0.0015 | +0.0007 |
| musique | zero-shot | 2417 | 0.4972 | 0.4613 | -0.0359 [-0.0427, -0.0294] | LOSS | 0.4734 | -0.0294 | -0.0319 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.9007 | -0.0044 [-0.0067, -0.0022] | WITHIN | 0.6846 | -0.0082 | -0.0182 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9287 | -0.0028 [-0.0043, -0.0013] | WITHIN | 0.6079 | -0.0096 | +0.0031 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6353 | -0.0080 [-0.0164, +0.0002] | WITHIN | 0.0535 | -0.0093 | -0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
