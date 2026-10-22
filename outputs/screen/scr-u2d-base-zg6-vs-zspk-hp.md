# Screen scr-u2d-base-zg6-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8043 | 0.8112 | +0.0068 [+0.0054, +0.0083] | WITHIN | 0.0047 | +0.0083 | +0.0251 |
| squad | in-domain | 11873 | 0.9117 | 0.9118 | +0.0001 [-0.0020, +0.0023] | WITHIN | 0.9054 | +0.0001 | +0.0034 |
| musique | in-domain | 2417 | 0.5800 | 0.5937 | +0.0138 [+0.0070, +0.0204] | GAIN | 0.4734 | +0.0236 | +0.0393 |
| hotpotqa | zero-shot | 7405 | 0.8512 | 0.8309 | -0.0203 [-0.0248, -0.0161] | LOSS | 0.6846 | -0.0221 | -0.0408 |
| 2wiki | in-domain | 12576 | 0.9290 | 0.9391 | +0.0101 [+0.0080, +0.0123] | GAIN | 0.6079 | +0.0297 | +0.0091 |
| webqsp | zero-shot | 1503 | 0.2773 | 0.2189 | -0.0584 [-0.0783, -0.0381] | LOSS | 0.0535 | -0.0406 | -0.0067 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
