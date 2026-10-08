# Screen scr-nullx-L-2wiki-s2 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.6505 | +0.0037 [+0.0008, +0.0065] | WITHIN | 0.0047 | +0.0049 | +0.0002 |
| squad | in-domain | 11873 | 0.9112 | 0.9066 | -0.0046 [-0.0070, -0.0024] | WITHIN | 0.9054 | -0.0046 | -0.0010 |
| musique | in-domain | 2417 | 0.5601 | 0.5613 | +0.0012 [-0.0050, +0.0074] | WITHIN | 0.4734 | +0.0041 | -0.0099 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.8976 | -0.0001 [-0.0028, +0.0024] | WITHIN | 0.6846 | +0.0009 | -0.0008 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.8000 | -0.0073 [-0.0097, -0.0049] | WITHIN | 0.6079 | -0.0029 | -0.0036 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.1306 | +0.0230 [+0.0141, +0.0319] | GAIN | 0.0535 | +0.0193 | +0.0113 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
