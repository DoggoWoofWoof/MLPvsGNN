# Screen scr-r2eg-zg6 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zg6-s0 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.3949 | -0.0443 [-0.0505, -0.0383] | LOSS | 0.0047 | -0.0376 | +0.0014 |
| squad | in-domain | 11873 | 0.9097 | 0.9101 | +0.0004 [-0.0018, +0.0026] | WITHIN | 0.9054 | +0.0004 | -0.0034 |
| musique | zero-shot | 2417 | 0.5018 | 0.4838 | -0.0180 [-0.0250, -0.0104] | LOSS | 0.4734 | -0.0095 | -0.0219 |
| hotpotqa | in-domain | 7405 | 0.9109 | 0.9145 | +0.0036 [+0.0013, +0.0058] | WITHIN | 0.6846 | +0.0065 | +0.0031 |
| 2wiki | in-domain | 12576 | 0.9390 | 0.9380 | -0.0010 [-0.0027, +0.0006] | WITHIN | 0.6079 | -0.0039 | +0.0051 |
| webqsp | in-domain | 1503 | 0.6423 | 0.6439 | +0.0015 [-0.0106, +0.0149] | WITHIN | 0.0535 | +0.0060 | -0.0106 |

## Against outputs\screen\fits\scr-wqk-zspk-s0 (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3984 | 0.3949 | -0.0034 [-0.0100, +0.0032] | WITHIN | 0.0047 | -0.0093 | -0.0330 |
| squad | in-domain | 11873 | 0.9110 | 0.9101 | -0.0008 [-0.0033, +0.0016] | WITHIN | 0.9054 | -0.0008 | -0.0020 |
| musique | zero-shot | 2417 | 0.4590 | 0.4838 | +0.0248 [+0.0162, +0.0332] | GAIN | 0.4734 | +0.0236 | +0.1258 |
| hotpotqa | in-domain | 7405 | 0.9058 | 0.9145 | +0.0087 [+0.0061, +0.0111] | GAIN | 0.6846 | +0.0153 | +0.0146 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9380 | +0.0034 [+0.0017, +0.0051] | WITHIN | 0.6079 | +0.0104 | +0.0072 |
| webqsp | in-domain | 1503 | 0.6334 | 0.6439 | +0.0104 [-0.0033, +0.0240] | WITHIN | 0.0535 | +0.0166 | +0.0106 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
