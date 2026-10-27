# Screen scr-wqk-zspk-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zspk-s0 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3984 | 0.3681 | -0.0303 [-0.0355, -0.0247] | LOSS | 0.0047 | -0.0308 | -0.0093 |
| squad | in-domain | 11873 | 0.9110 | 0.9104 | -0.0006 [-0.0028, +0.0016] | WITHIN | 0.9054 | -0.0006 | -0.0020 |
| musique | zero-shot | 2417 | 0.4590 | 0.4318 | -0.0273 [-0.0351, -0.0196] | LOSS | 0.4734 | -0.0228 | -0.0447 |
| hotpotqa | in-domain | 7405 | 0.9058 | 0.9043 | -0.0015 [-0.0037, +0.0007] | WITHIN | 0.6846 | -0.0038 | -0.0092 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9337 | -0.0008 [-0.0022, +0.0007] | WITHIN | 0.6079 | -0.0009 | +0.0052 |
| webqsp | in-domain | 1503 | 0.6334 | 0.6212 | -0.0123 [-0.0227, -0.0013] | LOSS | 0.0535 | -0.0166 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
