# Screen scr-wqk-zrckp-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4189 | -0.0204 [-0.0254, -0.0152] | LOSS | 0.0047 | -0.0150 | -0.0290 |
| squad | in-domain | 11873 | 0.9106 | 0.9091 | -0.0014 [-0.0035, +0.0007] | WITHIN | 0.9054 | -0.0014 | +0.0002 |
| musique | zero-shot | 2417 | 0.4972 | 0.4637 | -0.0335 [-0.0406, -0.0269] | LOSS | 0.4734 | -0.0306 | +0.0157 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.8998 | -0.0053 [-0.0078, -0.0030] | WITHIN | 0.6846 | -0.0104 | -0.0041 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9281 | -0.0035 [-0.0052, -0.0018] | WITHIN | 0.6079 | -0.0129 | +0.0068 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6421 | -0.0012 [-0.0114, +0.0091] | WITHIN | 0.0535 | -0.0047 | -0.0033 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
