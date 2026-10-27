# Screen scr-kg-base-zg6 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8013 | 0.8127 | +0.0114 [+0.0097, +0.0131] | GAIN | 0.0047 | +0.0133 | +0.0397 |
| squad | in-domain | 11873 | 0.9102 | 0.9120 | +0.0018 [-0.0005, +0.0040] | WITHIN | 0.9054 | +0.0018 | +0.0029 |
| musique | zero-shot | 2417 | 0.5146 | 0.5138 | -0.0009 [-0.0079, +0.0064] | WITHIN | 0.4734 | +0.0079 | +0.0120 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9127 | +0.0041 [+0.0017, +0.0065] | WITHIN | 0.6846 | +0.0069 | +0.0142 |
| 2wiki | in-domain | 12576 | 0.9364 | 0.9448 | +0.0084 [+0.0067, +0.0100] | GAIN | 0.6079 | +0.0245 | +0.0126 |
| webqsp | zero-shot | 1503 | 0.3151 | 0.2398 | -0.0753 [-0.0957, -0.0527] | LOSS | 0.0535 | -0.0579 | -0.0419 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
