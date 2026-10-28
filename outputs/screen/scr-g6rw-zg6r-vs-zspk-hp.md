# Screen scr-g6rw-zg6r-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zspk-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3384 | 0.3834 | +0.0450 [+0.0380, +0.0519] | GAIN | 0.0047 | +0.0428 | +0.0103 |
| squad | in-domain | 11873 | 0.9114 | 0.9111 | -0.0003 [-0.0022, +0.0015] | WITHIN | 0.9054 | -0.0003 | +0.0043 |
| musique | in-domain | 2417 | 0.5794 | 0.5941 | +0.0147 [+0.0086, +0.0207] | GAIN | 0.4734 | +0.0219 | +0.0265 |
| hotpotqa | zero-shot | 7405 | 0.8496 | 0.8529 | +0.0033 [-0.0001, +0.0068] | WITHIN | 0.6846 | +0.0119 | -0.0382 |
| 2wiki | in-domain | 12576 | 0.9330 | 0.9420 | +0.0090 [+0.0074, +0.0107] | GAIN | 0.6079 | +0.0243 | +0.0004 |
| webqsp | in-domain | 1503 | 0.6153 | 0.6536 | +0.0383 [+0.0259, +0.0508] | GAIN | 0.0535 | +0.0426 | +0.0579 |

## Against outputs\screen\fits\scr-wqk-zg6-s0-hp (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4167 | 0.3834 | -0.0333 [-0.0393, -0.0275] | LOSS | 0.0047 | -0.0301 | +0.0610 |
| squad | in-domain | 11873 | 0.9114 | 0.9111 | -0.0003 [-0.0024, +0.0016] | WITHIN | 0.9054 | -0.0003 | +0.0016 |
| musique | in-domain | 2417 | 0.5961 | 0.5941 | -0.0020 [-0.0075, +0.0034] | WITHIN | 0.4734 | -0.0029 | -0.0112 |
| hotpotqa | zero-shot | 7405 | 0.8183 | 0.8529 | +0.0346 [+0.0309, +0.0386] | GAIN | 0.6846 | +0.0556 | -0.0155 |
| 2wiki | in-domain | 12576 | 0.9414 | 0.9420 | +0.0006 [-0.0010, +0.0023] | WITHIN | 0.6079 | -0.0014 | -0.0011 |
| webqsp | in-domain | 1503 | 0.6421 | 0.6536 | +0.0116 [-0.0010, +0.0242] | WITHIN | 0.0535 | +0.0126 | +0.0499 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
