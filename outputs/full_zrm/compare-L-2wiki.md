# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\full_zret\fits\L-2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6464 | 0.7794 | +0.1330 [+0.1276, +0.1381] | GAIN | 0.0047 | +0.1382 | +0.2741 |
| squad | in-domain | 11873 | 0.9109 | 0.9093 | -0.0016 [-0.0035, +0.0003] | WITHIN | 0.9054 | -0.0016 | +0.0007 |
| musique | in-domain | 2417 | 0.5656 | 0.5684 | +0.0027 [-0.0025, +0.0082] | WITHIN | 0.4734 | +0.0103 | +0.0008 |
| hotpotqa | in-domain | 7405 | 0.9035 | 0.9034 | -0.0001 [-0.0024, +0.0022] | WITHIN | 0.6846 | -0.0005 | +0.0011 |
| 2wiki | zero-shot | 12576 | 0.7993 | 0.7972 | -0.0020 [-0.0044, +0.0003] | WITHIN | 0.6079 | +0.0013 | +0.0060 |
| webqsp | zero-shot | 1503 | 0.1338 | 0.2644 | +0.1306 [+0.1091, +0.1527] | GAIN | 0.0535 | +0.1031 | +0.0685 |

## Against outputs\full_rmatch\fits\L-2wiki (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7803 | 0.7794 | -0.0009 [-0.0020, +0.0003] | WITHIN | 0.0047 | -0.0001 | -0.0039 |
| squad | in-domain | 11873 | 0.9100 | 0.9093 | -0.0007 [-0.0029, +0.0015] | WITHIN | 0.9054 | -0.0007 | +0.0021 |
| musique | in-domain | 2417 | 0.5576 | 0.5684 | +0.0107 [+0.0049, +0.0165] | GAIN | 0.4734 | +0.0190 | +0.0070 |
| hotpotqa | in-domain | 7405 | 0.9006 | 0.9034 | +0.0028 [+0.0003, +0.0053] | WITHIN | 0.6846 | +0.0058 | -0.0014 |
| 2wiki | zero-shot | 12576 | 0.7937 | 0.7972 | +0.0035 [+0.0009, +0.0059] | WITHIN | 0.6079 | +0.0072 | +0.0045 |
| webqsp | zero-shot | 1503 | 0.2607 | 0.2644 | +0.0037 [-0.0078, +0.0144] | WITHIN | 0.0535 | +0.0007 | +0.0120 |

## Against outputs\step1\fits\L-2wiki (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.7794 | +0.1326 [+0.1272, +0.1378] | GAIN | 0.0047 | +0.1379 | +0.2624 |
| squad | in-domain | 11873 | 0.9112 | 0.9093 | -0.0019 [-0.0043, +0.0003] | WITHIN | 0.9054 | -0.0019 | -0.0003 |
| musique | in-domain | 2417 | 0.5601 | 0.5684 | +0.0082 [+0.0017, +0.0146] | GAIN | 0.4734 | +0.0153 | +0.0037 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9034 | +0.0057 [+0.0030, +0.0084] | WITHIN | 0.6846 | +0.0101 | +0.0057 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.7972 | -0.0101 [-0.0125, -0.0076] | LOSS | 0.6079 | -0.0164 | +0.0052 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.2644 | +0.1569 [+0.1352, +0.1794] | GAIN | 0.0535 | +0.1264 | +0.0832 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
