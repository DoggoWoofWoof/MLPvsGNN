# Screen scr-r2ws-zrckpb-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s2 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4004 | 0.4695 | +0.0690 [+0.0635, +0.0752] | GAIN | 0.0047 | +0.0468 | +0.0706 |
| squad | in-domain | 11873 | 0.9104 | 0.9093 | -0.0011 [-0.0033, +0.0011] | WITHIN | 0.9054 | -0.0011 | -0.0043 |
| musique | zero-shot | 2417 | 0.4753 | 0.4994 | +0.0241 [+0.0173, +0.0308] | GAIN | 0.4734 | +0.0099 | +0.0472 |
| hotpotqa | in-domain | 7405 | 0.9020 | 0.9021 | +0.0001 [-0.0024, +0.0027] | WITHIN | 0.6846 | +0.0015 | +0.0007 |
| 2wiki | in-domain | 12576 | 0.9260 | 0.9295 | +0.0035 [+0.0019, +0.0052] | WITHIN | 0.6079 | +0.0113 | -0.0057 |
| webqsp | in-domain | 1503 | 0.6457 | 0.6523 | +0.0066 [-0.0047, +0.0185] | WITHIN | 0.0535 | +0.0047 | +0.0220 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
