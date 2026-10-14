# Screen scr-p1c3-zrck-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.8052 | +0.0259 [+0.0230, +0.0287] | GAIN | 0.0047 | +0.0218 | +0.0381 |
| squad | in-domain | 11873 | 0.9112 | 0.9122 | +0.0009 [-0.0013, +0.0029] | WITHIN | 0.9054 | +0.0009 | -0.0023 |
| musique | in-domain | 2417 | 0.5645 | 0.5694 | +0.0050 [-0.0037, +0.0132] | WITHIN | 0.4734 | +0.0008 | +0.0546 |
| hotpotqa | zero-shot | 7405 | 0.8451 | 0.8564 | +0.0113 [+0.0065, +0.0160] | GAIN | 0.6846 | +0.0205 | -0.0424 |
| 2wiki | in-domain | 12576 | 0.8731 | 0.9128 | +0.0397 [+0.0362, +0.0434] | GAIN | 0.6079 | +0.0589 | +0.0765 |
| webqsp | zero-shot | 1503 | 0.2865 | 0.3686 | +0.0820 [+0.0644, +0.0985] | GAIN | 0.0535 | +0.0665 | +0.0266 |

## Against outputs\screen\fits\scr-p1c2-zrc-hp (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8002 | 0.8052 | +0.0050 [+0.0038, +0.0062] | WITHIN | 0.0047 | +0.0034 | +0.0208 |
| squad | in-domain | 11873 | 0.9134 | 0.9122 | -0.0013 [-0.0032, +0.0008] | WITHIN | 0.9054 | -0.0013 | -0.0020 |
| musique | in-domain | 2417 | 0.5651 | 0.5694 | +0.0044 [-0.0036, +0.0122] | WITHIN | 0.4734 | +0.0008 | +0.0376 |
| hotpotqa | zero-shot | 7405 | 0.8363 | 0.8564 | +0.0201 [+0.0159, +0.0245] | GAIN | 0.6846 | +0.0365 | -0.0401 |
| 2wiki | in-domain | 12576 | 0.9125 | 0.9128 | +0.0003 [-0.0023, +0.0029] | WITHIN | 0.6079 | -0.0155 | +0.0337 |
| webqsp | zero-shot | 1503 | 0.3142 | 0.3686 | +0.0544 [+0.0404, +0.0685] | GAIN | 0.0535 | +0.0466 | +0.0226 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8052 | +0.1512 [+0.1455, +0.1566] | GAIN | 0.0047 | +0.1552 | +0.2934 |
| squad | in-domain | 11873 | 0.9116 | 0.9122 | +0.0005 [-0.0016, +0.0026] | WITHIN | 0.9054 | +0.0005 | +0.0008 |
| musique | in-domain | 2417 | 0.5550 | 0.5694 | +0.0144 [+0.0069, +0.0223] | GAIN | 0.4734 | +0.0165 | +0.0550 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8564 | -0.0022 [-0.0069, +0.0024] | WITHIN | 0.6846 | -0.0038 | -0.0424 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9128 | +0.0456 [+0.0423, +0.0491] | GAIN | 0.6079 | +0.0763 | +0.0677 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3686 | +0.2486 [+0.2232, +0.2734] | GAIN | 0.0535 | +0.2029 | +0.0739 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
