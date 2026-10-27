# Screen scr-wqk-zspk-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zspk-s0 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3984 | 0.3916 | -0.0068 [-0.0120, -0.0013] | WITHIN | 0.0047 | -0.0071 | +0.0107 |
| squad | in-domain | 11873 | 0.9110 | 0.9103 | -0.0007 [-0.0029, +0.0015] | WITHIN | 0.9054 | -0.0007 | -0.0041 |
| musique | zero-shot | 2417 | 0.4590 | 0.4862 | +0.0272 [+0.0204, +0.0341] | GAIN | 0.4734 | +0.0190 | -0.0157 |
| hotpotqa | in-domain | 7405 | 0.9058 | 0.9064 | +0.0006 [-0.0016, +0.0027] | WITHIN | 0.6846 | +0.0026 | -0.0096 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9333 | -0.0013 [-0.0027, +0.0001] | WITHIN | 0.6079 | -0.0034 | -0.0114 |
| webqsp | in-domain | 1503 | 0.6334 | 0.6327 | -0.0007 [-0.0111, +0.0101] | WITHIN | 0.0535 | +0.0000 | +0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
