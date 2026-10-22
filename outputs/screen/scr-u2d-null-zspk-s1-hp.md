# Screen scr-u2d-base-zspk-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8043 | 0.8005 | -0.0038 [-0.0049, -0.0028] | WITHIN | 0.0047 | -0.0039 | -0.0146 |
| squad | in-domain | 11873 | 0.9117 | 0.9122 | +0.0005 [-0.0015, +0.0027] | WITHIN | 0.9054 | +0.0005 | +0.0023 |
| musique | in-domain | 2417 | 0.5800 | 0.5860 | +0.0060 [+0.0009, +0.0114] | WITHIN | 0.4734 | +0.0099 | +0.0178 |
| hotpotqa | zero-shot | 7405 | 0.8512 | 0.8494 | -0.0018 [-0.0049, +0.0012] | WITHIN | 0.6846 | -0.0022 | -0.0001 |
| 2wiki | in-domain | 12576 | 0.9290 | 0.9385 | +0.0094 [+0.0077, +0.0112] | GAIN | 0.6079 | +0.0244 | +0.0050 |
| webqsp | zero-shot | 1503 | 0.2773 | 0.2498 | -0.0275 [-0.0381, -0.0172] | LOSS | 0.0535 | -0.0180 | -0.0173 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
