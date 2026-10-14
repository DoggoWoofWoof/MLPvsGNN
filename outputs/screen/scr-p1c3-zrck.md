# Screen scr-p1c3-zrck (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7782 | 0.8007 | +0.0225 [+0.0196, +0.0253] | GAIN | 0.0047 | +0.0194 | +0.0267 |
| squad | in-domain | 11873 | 0.9105 | 0.9117 | +0.0013 [-0.0008, +0.0034] | WITHIN | 0.9054 | +0.0013 | -0.0036 |
| musique | zero-shot | 2417 | 0.5235 | 0.5059 | -0.0176 [-0.0260, -0.0097] | LOSS | 0.4734 | -0.0331 | +0.0534 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9063 | +0.0022 [-0.0010, +0.0057] | WITHIN | 0.6846 | -0.0003 | -0.0186 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.9365 | +0.0632 [+0.0600, +0.0666] | GAIN | 0.6079 | +0.1316 | +0.0625 |
| webqsp | zero-shot | 1503 | 0.3384 | 0.3756 | +0.0372 [+0.0220, +0.0528] | GAIN | 0.0535 | +0.0306 | +0.0246 |

## Against outputs\screen\fits\scr-p1c2-zrc (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7994 | 0.8007 | +0.0014 [+0.0002, +0.0027] | WITHIN | 0.0047 | +0.0008 | +0.0028 |
| squad | in-domain | 11873 | 0.9123 | 0.9117 | -0.0006 [-0.0025, +0.0013] | WITHIN | 0.9054 | -0.0006 | +0.0016 |
| musique | zero-shot | 2417 | 0.4789 | 0.5059 | +0.0271 [+0.0185, +0.0358] | GAIN | 0.4734 | +0.0194 | +0.0720 |
| hotpotqa | in-domain | 7405 | 0.9036 | 0.9063 | +0.0026 [-0.0001, +0.0053] | WITHIN | 0.6846 | +0.0027 | -0.0096 |
| 2wiki | in-domain | 12576 | 0.9128 | 0.9365 | +0.0238 [+0.0217, +0.0259] | GAIN | 0.6079 | +0.0550 | +0.0397 |
| webqsp | zero-shot | 1503 | 0.3466 | 0.3756 | +0.0290 [+0.0188, +0.0404] | GAIN | 0.0535 | +0.0226 | +0.0113 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8007 | +0.1470 [+0.1415, +0.1523] | GAIN | 0.0047 | +0.1460 | +0.2772 |
| squad | in-domain | 11873 | 0.9100 | 0.9117 | +0.0018 [-0.0003, +0.0039] | WITHIN | 0.9054 | +0.0018 | -0.0038 |
| musique | zero-shot | 2417 | 0.2696 | 0.5059 | +0.2363 [+0.2244, +0.2480] | GAIN | 0.4734 | +0.1138 | +0.4282 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9063 | +0.0048 [+0.0014, +0.0081] | WITHIN | 0.6846 | +0.0050 | -0.0080 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9365 | +0.0633 [+0.0602, +0.0666] | GAIN | 0.6079 | +0.1304 | +0.0703 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3756 | +0.1978 [+0.1758, +0.2211] | GAIN | 0.0535 | +0.1603 | +0.0772 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
