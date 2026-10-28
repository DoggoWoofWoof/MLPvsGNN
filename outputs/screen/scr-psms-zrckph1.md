# Screen scr-psms-zrckph1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4250 | -0.0144 [-0.0195, -0.0089] | LOSS | 0.0047 | -0.0174 | -0.0264 |
| squad | in-domain | 11873 | 0.9106 | 0.9114 | +0.0008 [-0.0010, +0.0026] | WITHIN | 0.9054 | +0.0008 | +0.0003 |
| musique | zero-shot | 2417 | 0.4972 | 0.4997 | +0.0025 [-0.0032, +0.0082] | WITHIN | 0.4734 | -0.0029 | +0.0381 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.9022 | -0.0030 [-0.0053, -0.0007] | WITHIN | 0.6846 | -0.0069 | -0.0018 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9306 | -0.0009 [-0.0024, +0.0007] | WITHIN | 0.6079 | -0.0056 | +0.0056 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6410 | -0.0023 [-0.0109, +0.0060] | WITHIN | 0.0535 | -0.0033 | -0.0120 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
