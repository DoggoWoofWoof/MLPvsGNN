# Screen scr-psmu-zrckpu1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4361 | -0.0032 [-0.0082, +0.0018] | WITHIN | 0.0047 | +0.0007 | -0.0226 |
| squad | in-domain | 11873 | 0.9106 | 0.9097 | -0.0008 [-0.0027, +0.0009] | WITHIN | 0.9054 | -0.0008 | -0.0015 |
| musique | zero-shot | 2417 | 0.4972 | 0.4936 | -0.0036 [-0.0090, +0.0018] | WITHIN | 0.4734 | -0.0050 | +0.0157 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.9032 | -0.0019 [-0.0040, +0.0002] | WITHIN | 0.6846 | -0.0035 | -0.0107 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9314 | -0.0001 [-0.0015, +0.0013] | WITHIN | 0.6079 | -0.0010 | -0.0019 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6388 | -0.0046 [-0.0133, +0.0040] | WITHIN | 0.0535 | -0.0040 | -0.0140 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
