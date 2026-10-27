# Screen scr-wqk-zg6-s0-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zspk-s0-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3384 | 0.4167 | +0.0783 [+0.0713, +0.0853] | GAIN | 0.0047 | +0.0730 | -0.0507 |
| squad | in-domain | 11873 | 0.9114 | 0.9114 | +0.0000 [-0.0019, +0.0019] | WITHIN | 0.9054 | +0.0000 | +0.0027 |
| musique | in-domain | 2417 | 0.5794 | 0.5961 | +0.0167 [+0.0109, +0.0230] | GAIN | 0.4734 | +0.0248 | +0.0376 |
| hotpotqa | zero-shot | 7405 | 0.8496 | 0.8183 | -0.0313 [-0.0354, -0.0269] | LOSS | 0.6846 | -0.0438 | -0.0227 |
| 2wiki | in-domain | 12576 | 0.9330 | 0.9414 | +0.0084 [+0.0065, +0.0102] | GAIN | 0.6079 | +0.0256 | +0.0015 |
| webqsp | in-domain | 1503 | 0.6153 | 0.6421 | +0.0268 [+0.0149, +0.0387] | GAIN | 0.0535 | +0.0299 | +0.0080 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
