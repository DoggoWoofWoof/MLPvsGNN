# Screen scr-g6rw-zg6r (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zspk-s0 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3984 | 0.4105 | +0.0121 [+0.0058, +0.0184] | GAIN | 0.0047 | +0.0019 | -0.0025 |
| squad | in-domain | 11873 | 0.9110 | 0.9105 | -0.0005 [-0.0025, +0.0014] | WITHIN | 0.9054 | -0.0005 | -0.0004 |
| musique | zero-shot | 2417 | 0.4590 | 0.4106 | -0.0484 [-0.0562, -0.0410] | LOSS | 0.4734 | -0.0327 | +0.0095 |
| hotpotqa | in-domain | 7405 | 0.9058 | 0.9134 | +0.0076 [+0.0053, +0.0097] | GAIN | 0.6846 | +0.0140 | +0.0220 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9369 | +0.0024 [+0.0009, +0.0038] | WITHIN | 0.6079 | +0.0108 | -0.0102 |
| webqsp | in-domain | 1503 | 0.6334 | 0.6501 | +0.0167 [+0.0039, +0.0288] | GAIN | 0.0535 | +0.0186 | +0.0379 |

## Against outputs\screen\fits\scr-wqk-zg6-s0 (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4105 | -0.0288 [-0.0345, -0.0230] | LOSS | 0.0047 | -0.0264 | +0.0320 |
| squad | in-domain | 11873 | 0.9097 | 0.9105 | +0.0008 [-0.0013, +0.0028] | WITHIN | 0.9054 | +0.0008 | -0.0018 |
| musique | zero-shot | 2417 | 0.5018 | 0.4106 | -0.0912 [-0.1007, -0.0819] | LOSS | 0.4734 | -0.0658 | -0.1382 |
| hotpotqa | in-domain | 7405 | 0.9109 | 0.9134 | +0.0024 [+0.0003, +0.0045] | WITHIN | 0.6846 | +0.0053 | +0.0105 |
| 2wiki | in-domain | 12576 | 0.9390 | 0.9369 | -0.0021 [-0.0037, -0.0005] | WITHIN | 0.6079 | -0.0035 | -0.0123 |
| webqsp | in-domain | 1503 | 0.6423 | 0.6501 | +0.0078 [-0.0051, +0.0209] | WITHIN | 0.0535 | +0.0080 | +0.0166 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
