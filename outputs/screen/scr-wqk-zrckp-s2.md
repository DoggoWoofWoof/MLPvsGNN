# Screen scr-wqk-zrckp-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4004 | -0.0389 [-0.0445, -0.0333] | LOSS | 0.0047 | -0.0314 | -0.0557 |
| squad | in-domain | 11873 | 0.9106 | 0.9104 | -0.0002 [-0.0024, +0.0019] | WITHIN | 0.9054 | -0.0002 | +0.0008 |
| musique | zero-shot | 2417 | 0.4972 | 0.4753 | -0.0219 [-0.0284, -0.0157] | LOSS | 0.4734 | -0.0112 | -0.0269 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.9020 | -0.0031 [-0.0056, -0.0007] | WITHIN | 0.6846 | -0.0061 | +0.0020 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9260 | -0.0056 [-0.0073, -0.0040] | WITHIN | 0.6079 | -0.0197 | -0.0026 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6457 | +0.0023 [-0.0072, +0.0120] | WITHIN | 0.0535 | -0.0020 | +0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
