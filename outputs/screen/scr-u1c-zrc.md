# Screen scr-u1c-zrc (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7782 | 0.7843 | +0.0061 [+0.0041, +0.0081] | WITHIN | 0.0047 | +0.0054 | -0.0105 |
| squad | in-domain | 11873 | 0.9105 | 0.9115 | +0.0010 [-0.0011, +0.0032] | WITHIN | 0.9054 | +0.0010 | -0.0014 |
| musique | zero-shot | 2417 | 0.5235 | 0.4912 | -0.0323 [-0.0406, -0.0241] | LOSS | 0.4734 | -0.0397 | -0.1121 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.8444 | -0.0596 [-0.0641, -0.0550] | LOSS | 0.6846 | -0.1195 | -0.0072 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.8537 | -0.0196 [-0.0228, -0.0164] | LOSS | 0.6079 | -0.0457 | -0.0044 |
| webqsp | zero-shot | 1503 | 0.3384 | 0.3301 | -0.0084 [-0.0217, +0.0054] | WITHIN | 0.0535 | -0.0040 | -0.0060 |

## Against outputs\step1\fits\L-musique (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7843 | +0.1306 [+0.1254, +0.1358] | GAIN | 0.0047 | +0.1320 | +0.2400 |
| squad | in-domain | 11873 | 0.9100 | 0.9115 | +0.0015 [-0.0007, +0.0037] | WITHIN | 0.9054 | +0.0015 | -0.0016 |
| musique | zero-shot | 2417 | 0.2696 | 0.4912 | +0.2216 [+0.2099, +0.2333] | GAIN | 0.4734 | +0.1072 | +0.2627 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8444 | -0.0571 [-0.0617, -0.0527] | LOSS | 0.6846 | -0.1142 | +0.0035 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8537 | -0.0195 [-0.0228, -0.0162] | LOSS | 0.6079 | -0.0469 | +0.0034 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3301 | +0.1522 [+0.1313, +0.1741] | GAIN | 0.0535 | +0.1257 | +0.0466 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
