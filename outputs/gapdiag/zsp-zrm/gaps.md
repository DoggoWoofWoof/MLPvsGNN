# Gap diagnostics: zsp (GNN track) against zrm (MLP), and both against the pool and the published numbers

Diagnostic only (docs/DIAG_GAPS.md): it decides nothing. Development numbers, p@swa, s1eval carves. zsp and zrm are the two tracks' bases; zsp's numbers are message passing and never the MLP's.

## 1. GNN against MLP, R@5 (delta with its 95% interval; wins / losses by question)

| split | dataset | read | questions | rrf | MLP | GNN | delta R@5 | delta FC@5 | delta hit@1 | wins / losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J5 | metaqa | in-domain | 9785 | 0.0047 | 0.7778 | 0.7794 | +0.0016 [+0.0004, +0.0028] | +0.0020 | +0.0018 | 235 / 202 |
| J5 | squad | in-domain | 11873 | 0.9054 | 0.9106 | 0.9099 | -0.0007 [-0.0026, +0.0013] | -0.0007 | -0.0007 | 63 / 71 |
| J5 | musique | in-domain | 2417 | 0.4734 | 0.5730 | 0.5657 | -0.0073 [-0.0131, -0.0016] | -0.0012 | -0.0099 | 135 / 181 |
| J5 | hotpotqa | in-domain | 7405 | 0.6846 | 0.9025 | 0.9051 | +0.0026 [+0.0000, +0.0051] | +0.0049 | -0.0018 | 187 / 150 |
| J5 | 2wiki | in-domain | 12576 | 0.6079 | 0.8684 | 0.8726 | +0.0041 [+0.0021, +0.0061] | +0.0165 | -0.0025 | 662 / 466 |
| J5 | webqsp | zero-shot | 1503 | 0.0535 | 0.2408 | 0.2677 | +0.0268 [+0.0166, +0.0375] | +0.0186 | +0.0153 | 128 / 47 |
| L-metaqa | metaqa | zero-shot | 9785 | 0.0047 | 0.1607 | 0.1910 | +0.0303 [+0.0252, +0.0355] | +0.0291 | +0.0110 | 787 / 412 |
| L-metaqa | squad | in-domain | 11873 | 0.9054 | 0.9101 | 0.9117 | +0.0016 [-0.0004, +0.0036] | +0.0016 | +0.0016 | 91 / 72 |
| L-metaqa | musique | in-domain | 2417 | 0.4734 | 0.5696 | 0.5689 | -0.0007 [-0.0062, +0.0053] | -0.0029 | -0.0066 | 160 / 156 |
| L-metaqa | hotpotqa | in-domain | 7405 | 0.6846 | 0.9040 | 0.9066 | +0.0026 [+0.0003, +0.0053] | +0.0055 | +0.0059 | 204 / 163 |
| L-metaqa | 2wiki | in-domain | 12576 | 0.6079 | 0.8663 | 0.8768 | +0.0105 [+0.0087, +0.0125] | +0.0243 | +0.0041 | 697 / 314 |
| L-metaqa | webqsp | zero-shot | 1503 | 0.0535 | 0.1082 | 0.1174 | +0.0092 [+0.0020, +0.0166] | +0.0060 | +0.0047 | 53 / 33 |
| L-squad | metaqa | in-domain | 9785 | 0.0047 | 0.7787 | 0.7790 | +0.0003 [-0.0009, +0.0014] | +0.0004 | +0.0039 | 279 / 179 |
| L-squad | squad | zero-shot | 11873 | 0.9054 | 0.8886 | 0.8894 | +0.0008 [-0.0019, +0.0035] | +0.0008 | -0.0052 | 134 / 124 |
| L-squad | musique | in-domain | 2417 | 0.4734 | 0.5709 | 0.5721 | +0.0012 [-0.0048, +0.0070] | +0.0008 | -0.0083 | 166 / 163 |
| L-squad | hotpotqa | in-domain | 7405 | 0.6846 | 0.9034 | 0.9065 | +0.0032 [+0.0009, +0.0056] | +0.0068 | +0.0045 | 180 / 132 |
| L-squad | 2wiki | in-domain | 12576 | 0.6079 | 0.8724 | 0.8717 | -0.0007 [-0.0027, +0.0013] | +0.0044 | +0.0083 | 579 / 554 |
| L-squad | webqsp | zero-shot | 1503 | 0.0535 | 0.2640 | 0.2794 | +0.0154 [+0.0046, +0.0263] | +0.0140 | -0.0067 | 101 / 53 |
| L-2wiki | metaqa | in-domain | 9785 | 0.0047 | 0.7781 | 0.7792 | +0.0011 [-0.0000, +0.0022] | +0.0013 | +0.0055 | 239 / 191 |
| L-2wiki | squad | in-domain | 11873 | 0.9054 | 0.9111 | 0.9111 | +0.0000 [-0.0019, +0.0020] | +0.0000 | -0.0006 | 71 / 71 |
| L-2wiki | musique | in-domain | 2417 | 0.4734 | 0.5620 | 0.5648 | +0.0028 [-0.0025, +0.0081] | +0.0000 | -0.0025 | 141 / 126 |
| L-2wiki | hotpotqa | in-domain | 7405 | 0.6846 | 0.8921 | 0.9020 | +0.0099 [+0.0071, +0.0128] | +0.0196 | +0.0061 | 289 / 143 |
| L-2wiki | 2wiki | zero-shot | 12576 | 0.6079 | 0.7825 | 0.7966 | +0.0141 [+0.0115, +0.0167] | +0.0201 | -0.0033 | 1144 / 591 |
| L-2wiki | webqsp | zero-shot | 1503 | 0.0535 | 0.2429 | 0.2592 | +0.0163 [+0.0071, +0.0268] | +0.0106 | +0.0053 | 103 / 49 |
| L-musique | metaqa | in-domain | 9785 | 0.0047 | 0.7780 | 0.7786 | +0.0006 [-0.0005, +0.0017] | +0.0002 | +0.0020 | 243 / 185 |
| L-musique | squad | in-domain | 11873 | 0.9054 | 0.9118 | 0.9100 | -0.0018 [-0.0039, +0.0003] | -0.0018 | -0.0017 | 72 / 93 |
| L-musique | musique | zero-shot | 2417 | 0.4734 | 0.5126 | 0.5131 | +0.0004 [-0.0052, +0.0059] | -0.0008 | -0.0248 | 146 / 141 |
| L-musique | hotpotqa | in-domain | 7405 | 0.6846 | 0.9038 | 0.9051 | +0.0013 [-0.0012, +0.0036] | +0.0028 | +0.0020 | 183 / 165 |
| L-musique | 2wiki | in-domain | 12576 | 0.6079 | 0.8704 | 0.8756 | +0.0052 [+0.0031, +0.0072] | +0.0127 | +0.0012 | 627 / 421 |
| L-musique | webqsp | zero-shot | 1503 | 0.0535 | 0.3212 | 0.3247 | +0.0035 [-0.0062, +0.0136] | +0.0040 | +0.0053 | 77 / 76 |
| L-hotpotqa | metaqa | in-domain | 9785 | 0.0047 | 0.7787 | 0.7800 | +0.0013 [+0.0001, +0.0024] | +0.0009 | +0.0059 | 265 / 164 |
| L-hotpotqa | squad | in-domain | 11873 | 0.9054 | 0.9114 | 0.9148 | +0.0035 [+0.0014, +0.0055] | +0.0035 | +0.0040 | 101 / 60 |
| L-hotpotqa | musique | in-domain | 2417 | 0.4734 | 0.5714 | 0.5748 | +0.0034 [-0.0020, +0.0088] | +0.0099 | +0.0083 | 162 / 147 |
| L-hotpotqa | hotpotqa | zero-shot | 7405 | 0.6846 | 0.8481 | 0.8406 | -0.0074 [-0.0111, -0.0037] | -0.0136 | +0.0004 | 318 / 428 |
| L-hotpotqa | 2wiki | in-domain | 12576 | 0.6079 | 0.8663 | 0.8763 | +0.0100 [+0.0080, +0.0122] | +0.0278 | +0.0064 | 846 / 393 |
| L-hotpotqa | webqsp | zero-shot | 1503 | 0.0535 | 0.2364 | 0.2333 | -0.0030 [-0.0121, +0.0065] | -0.0007 | -0.0007 | 73 / 73 |

## 2. Where the delta comes from (pooled over splits, R@5 delta by bucket; share of the summed delta)

### golds

| dataset | read | bucket | questions | MLP | GNN | delta | share | ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | in-domain | 1 | 19330 | 0.9740 | 0.9738 | -0.0002 | -0.06 | 0.9904 |
| metaqa | in-domain | 2 | 6720 | 0.9486 | 0.9496 | +0.0010 | 0.15 | 0.9762 |
| metaqa | in-domain | 3+ | 22875 | 0.5629 | 0.5648 | +0.0019 | 0.92 | 0.6108 |
| metaqa | zero-shot | 1 | 3866 | 0.2838 | 0.3262 | +0.0424 | 0.55 | 0.9904 |
| metaqa | zero-shot | 2 | 1344 | 0.1656 | 0.2072 | +0.0417 | 0.19 | 0.9762 |
| metaqa | zero-shot | 3+ | 4575 | 0.0552 | 0.0719 | +0.0167 | 0.26 | 0.6108 |
| squad | in-domain | 1 | 59365 | 0.9110 | 0.9115 | +0.0005 | 1.00 | 0.9798 |
| squad | zero-shot | 1 | 11873 | 0.8886 | 0.8894 | +0.0008 | 1.00 | 0.9798 |
| musique | in-domain | 2 | 6260 | 0.6756 | 0.6754 | -0.0002 | 0.93 | 0.9665 |
| musique | in-domain | 3+ | 5825 | 0.4552 | 0.4552 | -0.0000 | 0.07 | 0.8716 |
| musique | zero-shot | 2 | 1252 | 0.6202 | 0.6198 | -0.0004 | -0.44 | 0.9665 |
| musique | zero-shot | 3+ | 1165 | 0.3970 | 0.3984 | +0.0014 | 1.44 | 0.8716 |
| hotpotqa | in-domain | 2 | 37025 | 0.9012 | 0.9051 | +0.0039 | 1.00 | 0.9800 |
| hotpotqa | zero-shot | 2 | 7405 | 0.8481 | 0.8406 | -0.0074 | 1.00 | 0.9800 |
| 2wiki | in-domain | 2 | 49125 | 0.8986 | 0.9002 | +0.0016 | 0.21 | 0.9719 |
| 2wiki | in-domain | 3+ | 13755 | 0.7622 | 0.7833 | +0.0211 | 0.79 | 0.9344 |
| 2wiki | zero-shot | 2 | 9825 | 0.8489 | 0.8557 | +0.0068 | 0.38 | 0.9719 |
| 2wiki | zero-shot | 3+ | 2751 | 0.5455 | 0.5858 | +0.0403 | 0.62 | 0.9344 |
| webqsp | zero-shot | 1 | 4578 | 0.3406 | 0.3554 | +0.0149 | 0.66 | 0.8886 |
| webqsp | zero-shot | 2 | 1194 | 0.2337 | 0.2470 | +0.0134 | 0.16 | 0.8744 |
| webqsp | zero-shot | 3+ | 3246 | 0.0883 | 0.0940 | +0.0057 | 0.18 | 0.5501 |

### all_in_pool

| dataset | read | bucket | questions | MLP | GNN | delta | share | ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | in-domain | all | 43785 | 0.8514 | 0.8523 | +0.0010 | 0.89 | 0.8828 |
| metaqa | in-domain | none | 620 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| metaqa | in-domain | some | 4520 | 0.1770 | 0.1781 | +0.0011 | 0.11 | 0.2265 |
| metaqa | zero-shot | all | 8757 | 0.1775 | 0.2112 | +0.0337 | 1.00 | 0.8828 |
| metaqa | zero-shot | none | 124 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| metaqa | zero-shot | some | 904 | 0.0195 | 0.0206 | +0.0011 | 0.00 | 0.2265 |
| squad | in-domain | all | 58165 | 0.9298 | 0.9303 | +0.0005 | 1.00 | 1.0000 |
| squad | in-domain | none | 1200 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| squad | zero-shot | all | 11633 | 0.9069 | 0.9078 | +0.0009 | 1.00 | 1.0000 |
| squad | zero-shot | none | 240 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| musique | in-domain | all | 9700 | 0.6247 | 0.6252 | +0.0005 | -3.14 | 1.0000 |
| musique | in-domain | none | 60 | 0.0000 | 0.0000 | +0.0000 | -0.00 | 0.0000 |
| musique | in-domain | some | 2325 | 0.3532 | 0.3504 | -0.0029 | 4.14 | 0.6138 |
| musique | zero-shot | all | 1940 | 0.5641 | 0.5655 | +0.0014 | 2.61 | 1.0000 |
| musique | zero-shot | none | 12 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| musique | zero-shot | some | 465 | 0.3109 | 0.3073 | -0.0036 | -1.61 | 0.6138 |
| hotpotqa | in-domain | all | 35855 | 0.9244 | 0.9285 | +0.0040 | 1.00 | 1.0000 |
| hotpotqa | in-domain | none | 310 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| hotpotqa | in-domain | some | 860 | 0.2546 | 0.2546 | -0.0000 | -0.00 | 0.5000 |
| hotpotqa | zero-shot | all | 7171 | 0.8700 | 0.8620 | -0.0079 | 1.04 | 1.0000 |
| hotpotqa | zero-shot | none | 62 | 0.0000 | 0.0000 | +0.0000 | -0.00 | 0.0000 |
| hotpotqa | zero-shot | some | 172 | 0.2413 | 0.2529 | +0.0116 | -0.04 | 0.5000 |
| 2wiki | in-domain | all | 56960 | 0.9039 | 0.9102 | +0.0063 | 0.98 | 1.0000 |
| 2wiki | in-domain | none | 20 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| 2wiki | in-domain | some | 5900 | 0.5330 | 0.5341 | +0.0010 | 0.02 | 0.6161 |
| 2wiki | zero-shot | all | 11392 | 0.8165 | 0.8313 | +0.0148 | 0.95 | 1.0000 |
| 2wiki | zero-shot | none | 4 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| 2wiki | zero-shot | some | 1180 | 0.4568 | 0.4648 | +0.0081 | 0.05 | 0.6161 |
| webqsp | zero-shot | all | 6204 | 0.3212 | 0.3363 | +0.0151 | 0.91 | 0.9853 |
| webqsp | zero-shot | none | 708 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| webqsp | zero-shot | some | 2106 | 0.0626 | 0.0668 | +0.0041 | 0.09 | 0.3727 |

### deepest_gold

| dataset | read | bucket | questions | MLP | GNN | delta | share | ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | in-domain | hop 1 | 13605 | 0.9405 | 0.9400 | -0.0006 | -0.17 | 0.9647 |
| metaqa | in-domain | hop 2 | 18035 | 0.8286 | 0.8299 | +0.0013 | 0.51 | 0.8501 |
| metaqa | in-domain | hop 3 | 16250 | 0.6210 | 0.6232 | +0.0022 | 0.75 | 0.6729 |
| metaqa | in-domain | none in pool | 620 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| metaqa | in-domain | seed | 250 | 0.8720 | 0.8560 | -0.0160 | -0.08 | 1.0000 |
| metaqa | in-domain | unreached | 165 | 0.1707 | 0.1697 | -0.0010 | -0.00 | 0.2253 |
| metaqa | zero-shot | hop 1 | 2721 | 0.4494 | 0.5895 | +0.1402 | 1.29 | 0.9647 |
| metaqa | zero-shot | hop 2 | 3607 | 0.0684 | 0.0460 | -0.0224 | -0.27 | 0.8501 |
| metaqa | zero-shot | hop 3 | 3250 | 0.0186 | 0.0180 | -0.0006 | -0.01 | 0.6729 |
| metaqa | zero-shot | none in pool | 124 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| metaqa | zero-shot | seed | 50 | 0.8400 | 0.8000 | -0.0400 | -0.01 | 1.0000 |
| metaqa | zero-shot | unreached | 33 | 0.0041 | 0.0041 | +0.0000 | 0.00 | 0.2253 |
| squad | in-domain | hop 1 | 665 | 0.1338 | 0.1714 | +0.0376 | 0.81 | 1.0000 |
| squad | in-domain | hop 2 | 195 | 0.0359 | 0.0461 | +0.0102 | 0.06 | 1.0000 |
| squad | in-domain | hop 3 | 15 | 0.0667 | 0.0667 | +0.0000 | 0.00 | 1.0000 |
| squad | in-domain | none in pool | 1200 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| squad | in-domain | seed | 55300 | 0.9725 | 0.9726 | +0.0001 | 0.25 | 1.0000 |
| squad | in-domain | unreached | 1990 | 0.1045 | 0.1025 | -0.0020 | -0.13 | 1.0000 |
| squad | zero-shot | hop 1 | 133 | 0.2632 | 0.2406 | -0.0226 | -0.31 | 1.0000 |
| squad | zero-shot | hop 2 | 39 | 0.0256 | 0.0769 | +0.0513 | 0.21 | 1.0000 |
| squad | zero-shot | hop 3 | 3 | 0.0000 | 0.3333 | +0.3333 | 0.10 | 1.0000 |
| squad | zero-shot | none in pool | 240 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| squad | zero-shot | seed | 11060 | 0.9469 | 0.9485 | +0.0015 | 1.73 | 1.0000 |
| squad | zero-shot | unreached | 398 | 0.1030 | 0.0854 | -0.0176 | -0.73 | 1.0000 |
| musique | in-domain | hop 1 | 2965 | 0.6310 | 0.6396 | +0.0086 | -15.30 | 0.9614 |
| musique | in-domain | hop 2 | 3440 | 0.4394 | 0.4353 | -0.0041 | 8.59 | 0.9400 |
| musique | in-domain | hop 3 | 1575 | 0.3955 | 0.3968 | +0.0014 | -1.31 | 0.8929 |
| musique | in-domain | none in pool | 60 | 0.0000 | 0.0000 | +0.0000 | -0.00 | 0.0000 |
| musique | in-domain | seed | 2975 | 0.8169 | 0.8150 | -0.0019 | 3.34 | 0.9060 |
| musique | in-domain | unreached | 1070 | 0.4160 | 0.4073 | -0.0088 | 5.69 | 0.8797 |
| musique | zero-shot | hop 1 | 593 | 0.6230 | 0.6218 | -0.0011 | -0.58 | 0.9614 |
| musique | zero-shot | hop 2 | 688 | 0.3802 | 0.3780 | -0.0022 | -1.35 | 0.9400 |
| musique | zero-shot | hop 3 | 315 | 0.3558 | 0.3553 | -0.0005 | -0.14 | 0.8929 |
| musique | zero-shot | none in pool | 12 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| musique | zero-shot | seed | 595 | 0.7132 | 0.7214 | +0.0083 | 4.42 | 0.9060 |
| musique | zero-shot | unreached | 214 | 0.3345 | 0.3275 | -0.0070 | -1.34 | 0.8797 |
| hotpotqa | in-domain | hop 1 | 16665 | 0.8591 | 0.8672 | +0.0081 | 0.93 | 0.9929 |
| hotpotqa | in-domain | hop 2 | 430 | 0.3302 | 0.3430 | +0.0128 | 0.04 | 0.8140 |
| hotpotqa | in-domain | hop 3 | 90 | 0.5111 | 0.5500 | +0.0389 | 0.02 | 1.0000 |
| hotpotqa | in-domain | none in pool | 310 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| hotpotqa | in-domain | seed | 18950 | 0.9749 | 0.9750 | +0.0002 | 0.02 | 0.9908 |
| hotpotqa | in-domain | unreached | 580 | 0.6664 | 0.6629 | -0.0034 | -0.01 | 0.9009 |
| hotpotqa | zero-shot | hop 1 | 3333 | 0.7591 | 0.7501 | -0.0090 | 0.54 | 0.9929 |
| hotpotqa | zero-shot | hop 2 | 86 | 0.3023 | 0.3198 | +0.0174 | -0.03 | 0.8140 |
| hotpotqa | zero-shot | hop 3 | 18 | 0.5000 | 0.5556 | +0.0556 | -0.02 | 1.0000 |
| hotpotqa | zero-shot | none in pool | 62 | 0.0000 | 0.0000 | +0.0000 | -0.00 | 0.0000 |
| hotpotqa | zero-shot | seed | 3790 | 0.9620 | 0.9546 | -0.0074 | 0.51 | 0.9908 |
| hotpotqa | zero-shot | unreached | 116 | 0.5948 | 0.5991 | +0.0043 | -0.01 | 0.9009 |
| 2wiki | in-domain | hop 1 | 36435 | 0.8690 | 0.8794 | +0.0104 | 1.03 | 0.9903 |
| 2wiki | in-domain | hop 2 | 1115 | 0.5399 | 0.5361 | -0.0038 | -0.01 | 0.8957 |
| 2wiki | in-domain | hop 3 | 65 | 0.4885 | 0.5231 | +0.0346 | 0.01 | 0.9423 |
| 2wiki | in-domain | none in pool | 20 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| 2wiki | in-domain | seed | 21970 | 0.9153 | 0.9141 | -0.0012 | -0.07 | 0.9390 |
| 2wiki | in-domain | unreached | 3275 | 0.6784 | 0.6830 | +0.0046 | 0.04 | 0.8626 |
| 2wiki | zero-shot | hop 1 | 7287 | 0.7423 | 0.7730 | +0.0307 | 1.26 | 0.9903 |
| 2wiki | zero-shot | hop 2 | 223 | 0.4776 | 0.4619 | -0.0157 | -0.02 | 0.8957 |
| 2wiki | zero-shot | hop 3 | 13 | 0.4231 | 0.4808 | +0.0577 | 0.00 | 0.9423 |
| 2wiki | zero-shot | none in pool | 4 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| 2wiki | zero-shot | seed | 4394 | 0.9009 | 0.8897 | -0.0112 | -0.28 | 0.9390 |
| 2wiki | zero-shot | unreached | 655 | 0.5508 | 0.5599 | +0.0092 | 0.03 | 0.8626 |
| webqsp | zero-shot | hop 1 | 4032 | 0.3084 | 0.3259 | +0.0176 | 0.69 | 0.8927 |
| webqsp | zero-shot | hop 2 | 2154 | 0.1204 | 0.1329 | +0.0126 | 0.26 | 0.7282 |
| webqsp | zero-shot | hop 3 | 1140 | 0.1357 | 0.1527 | +0.0170 | 0.19 | 0.7171 |
| webqsp | zero-shot | none in pool | 708 | 0.0000 | 0.0000 | +0.0000 | 0.00 | 0.0000 |
| webqsp | zero-shot | seed | 834 | 0.5424 | 0.5229 | -0.0195 | -0.16 | 0.9786 |
| webqsp | zero-shot | unreached | 150 | 0.0990 | 0.1085 | +0.0094 | 0.01 | 0.6416 |

### rrf_r5

| dataset | read | bucket | questions | MLP | GNN | delta | share | ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | in-domain | full | 125 | 0.8240 | 0.7920 | -0.0320 | -0.08 | 1.0000 |
| metaqa | in-domain | partial | 545 | 0.5293 | 0.5305 | +0.0013 | 0.01 | 0.5996 |
| metaqa | in-domain | zero | 48255 | 0.7810 | 0.7820 | +0.0011 | 1.07 | 0.8129 |
| metaqa | zero-shot | full | 25 | 0.9200 | 0.9600 | +0.0400 | 0.00 | 1.0000 |
| metaqa | zero-shot | partial | 109 | 0.2019 | 0.2071 | +0.0052 | 0.00 | 0.5996 |
| metaqa | zero-shot | zero | 9651 | 0.1582 | 0.1888 | +0.0306 | 0.99 | 0.8129 |
| squad | in-domain | full | 53750 | 0.9859 | 0.9857 | -0.0002 | -0.39 | 1.0000 |
| squad | in-domain | zero | 5615 | 0.1943 | 0.2018 | +0.0075 | 1.39 | 0.7863 |
| squad | zero-shot | full | 10750 | 0.9639 | 0.9660 | +0.0020 | 2.27 | 1.0000 |
| squad | zero-shot | zero | 1123 | 0.1674 | 0.1567 | -0.0107 | -1.27 | 0.7863 |
| musique | in-domain | full | 1635 | 0.9459 | 0.9473 | +0.0014 | -1.51 | 1.0000 |
| musique | in-domain | partial | 8820 | 0.5708 | 0.5714 | +0.0006 | -3.20 | 0.9264 |
| musique | in-domain | zero | 1630 | 0.1838 | 0.1784 | -0.0054 | 5.71 | 0.8103 |
| musique | zero-shot | full | 327 | 0.8445 | 0.8558 | +0.0112 | 3.46 | 1.0000 |
| musique | zero-shot | partial | 1764 | 0.5194 | 0.5209 | +0.0015 | 2.50 | 0.9264 |
| musique | zero-shot | zero | 326 | 0.1431 | 0.1270 | -0.0161 | -4.95 | 0.8103 |
| hotpotqa | in-domain | full | 15700 | 0.9911 | 0.9912 | +0.0001 | 0.01 | 1.0000 |
| hotpotqa | in-domain | partial | 19295 | 0.8929 | 0.8999 | +0.0070 | 0.93 | 0.9908 |
| hotpotqa | in-domain | zero | 2030 | 0.2838 | 0.2877 | +0.0039 | 0.06 | 0.7229 |
| hotpotqa | zero-shot | full | 3140 | 0.9847 | 0.9791 | -0.0056 | 0.32 | 1.0000 |
| hotpotqa | zero-shot | partial | 3859 | 0.8009 | 0.7901 | -0.0108 | 0.75 | 0.9908 |
| hotpotqa | zero-shot | zero | 406 | 0.2401 | 0.2500 | +0.0099 | -0.07 | 0.7229 |
| 2wiki | in-domain | full | 15945 | 0.9834 | 0.9829 | -0.0005 | -0.02 | 1.0000 |
| 2wiki | in-domain | partial | 45975 | 0.8374 | 0.8457 | +0.0083 | 1.03 | 0.9526 |
| 2wiki | in-domain | zero | 960 | 0.4648 | 0.4594 | -0.0055 | -0.01 | 0.8893 |
| 2wiki | zero-shot | full | 3189 | 0.9757 | 0.9691 | -0.0066 | -0.12 | 1.0000 |
| 2wiki | zero-shot | partial | 9195 | 0.7240 | 0.7459 | +0.0220 | 1.14 | 0.9526 |
| 2wiki | zero-shot | zero | 192 | 0.3763 | 0.3594 | -0.0169 | -0.02 | 0.8893 |
| webqsp | zero-shot | full | 342 | 0.5819 | 0.5599 | -0.0219 | -0.07 | 1.0000 |
| webqsp | zero-shot | partial | 546 | 0.1499 | 0.1530 | +0.0030 | 0.02 | 0.6108 |
| webqsp | zero-shot | zero | 8130 | 0.2268 | 0.2401 | +0.0133 | 1.06 | 0.7653 |

### pool_size

| dataset | read | bucket | questions | MLP | GNN | delta | share | ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | in-domain | Q1 | 9185 | 0.7301 | 0.7319 | +0.0018 | 0.34 | 0.7568 |
| metaqa | in-domain | Q2 | 12430 | 0.7176 | 0.7188 | +0.0012 | 0.31 | 0.7579 |
| metaqa | in-domain | Q3 | 13405 | 0.7888 | 0.7887 | -0.0001 | -0.03 | 0.8235 |
| metaqa | in-domain | Q4 | 13905 | 0.8542 | 0.8555 | +0.0013 | 0.38 | 0.8821 |
| metaqa | zero-shot | Q1 | 1837 | 0.1618 | 0.2050 | +0.0432 | 0.27 | 0.7568 |
| metaqa | zero-shot | Q2 | 2486 | 0.1434 | 0.1596 | +0.0162 | 0.14 | 0.7579 |
| metaqa | zero-shot | Q3 | 2681 | 0.1630 | 0.1821 | +0.0191 | 0.17 | 0.8235 |
| metaqa | zero-shot | Q4 | 2781 | 0.1730 | 0.2183 | +0.0453 | 0.42 | 0.8821 |
| squad | in-domain | Q4 | 59365 | 0.9110 | 0.9115 | +0.0005 | 1.00 | 0.9798 |
| squad | zero-shot | Q4 | 11873 | 0.8886 | 0.8894 | +0.0008 | 1.00 | 0.9798 |
| musique | in-domain | Q1 | 2960 | 0.5268 | 0.5236 | -0.0032 | 5.84 | 0.9727 |
| musique | in-domain | Q2 | 2965 | 0.5364 | 0.5382 | +0.0018 | -3.29 | 0.9271 |
| musique | in-domain | Q3 | 3105 | 0.6077 | 0.6095 | +0.0018 | -3.49 | 0.9120 |
| musique | in-domain | Q4 | 3055 | 0.6037 | 0.6027 | -0.0010 | 1.94 | 0.8732 |
| musique | zero-shot | Q1 | 592 | 0.4655 | 0.4738 | +0.0083 | 4.28 | 0.9727 |
| musique | zero-shot | Q2 | 593 | 0.4852 | 0.4879 | +0.0027 | 1.40 | 0.9271 |
| musique | zero-shot | Q3 | 621 | 0.5511 | 0.5459 | -0.0052 | -2.82 | 0.9120 |
| musique | zero-shot | Q4 | 611 | 0.5457 | 0.5421 | -0.0035 | -1.86 | 0.8732 |
| hotpotqa | in-domain | Q1 | 8745 | 0.9253 | 0.9277 | +0.0024 | 0.14 | 0.9911 |
| hotpotqa | in-domain | Q2 | 9735 | 0.9159 | 0.9200 | +0.0041 | 0.28 | 0.9828 |
| hotpotqa | in-domain | Q3 | 9080 | 0.9044 | 0.9085 | +0.0041 | 0.25 | 0.9777 |
| hotpotqa | in-domain | Q4 | 9465 | 0.8605 | 0.8656 | +0.0050 | 0.33 | 0.9691 |
| hotpotqa | zero-shot | Q1 | 1749 | 0.8719 | 0.8699 | -0.0020 | 0.06 | 0.9911 |
| hotpotqa | zero-shot | Q2 | 1947 | 0.8636 | 0.8549 | -0.0087 | 0.31 | 0.9828 |
| hotpotqa | zero-shot | Q3 | 1816 | 0.8535 | 0.8480 | -0.0055 | 0.18 | 0.9777 |
| hotpotqa | zero-shot | Q4 | 1893 | 0.8048 | 0.7919 | -0.0129 | 0.45 | 0.9691 |
| 2wiki | in-domain | Q1 | 14830 | 0.8888 | 0.8934 | +0.0046 | 0.18 | 0.9751 |
| 2wiki | in-domain | Q2 | 16430 | 0.8698 | 0.8759 | +0.0060 | 0.27 | 0.9662 |
| 2wiki | in-domain | Q3 | 15465 | 0.8684 | 0.8749 | +0.0065 | 0.27 | 0.9604 |
| 2wiki | in-domain | Q4 | 16155 | 0.8497 | 0.8559 | +0.0062 | 0.27 | 0.9537 |
| 2wiki | zero-shot | Q1 | 2966 | 0.8267 | 0.8386 | +0.0119 | 0.20 | 0.9751 |
| 2wiki | zero-shot | Q2 | 3286 | 0.7880 | 0.8003 | +0.0123 | 0.23 | 0.9662 |
| 2wiki | zero-shot | Q3 | 3093 | 0.7796 | 0.7919 | +0.0123 | 0.21 | 0.9604 |
| 2wiki | zero-shot | Q4 | 3231 | 0.7392 | 0.7590 | +0.0198 | 0.36 | 0.9537 |
| webqsp | zero-shot | Q1 | 2178 | 0.2382 | 0.2397 | +0.0014 | 0.03 | 0.7702 |
| webqsp | zero-shot | Q2 | 2244 | 0.2474 | 0.2653 | +0.0178 | 0.39 | 0.7756 |
| webqsp | zero-shot | Q3 | 2292 | 0.2229 | 0.2334 | +0.0104 | 0.23 | 0.7400 |
| webqsp | zero-shot | Q4 | 2304 | 0.2341 | 0.2495 | +0.0154 | 0.35 | 0.7741 |

### hops

| dataset | read | bucket | questions | MLP | GNN | delta | share | ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| musique | in-domain | 2 hops | 6260 | 0.6756 | 0.6754 | -0.0002 | 0.98 | 0.9665 |
| musique | in-domain | 3 hops | 3800 | 0.5212 | 0.5225 | +0.0012 | -3.08 | 0.9088 |
| musique | in-domain | 4 hops | 2025 | 0.3312 | 0.3289 | -0.0023 | 3.10 | 0.8019 |
| musique | zero-shot | 2 hops | 1252 | 0.6202 | 0.6198 | -0.0004 | -0.47 | 0.9665 |
| musique | zero-shot | 3 hops | 760 | 0.4553 | 0.4557 | +0.0004 | 0.29 | 0.9088 |
| musique | zero-shot | 4 hops | 405 | 0.2877 | 0.2907 | +0.0031 | 1.19 | 0.8019 |

## 3. Where every model misses: outside the pool, or in it and ranked below five (1 - R@5)

| split | dataset | read | ceiling R@5 | FC@5 reachable | outside pool | in pool, ranked out: rrf / MLP / GNN |
| --- | --- | --- | --- | --- | --- | --- |
| J5 | metaqa | in-domain | 0.8110 | 0.6987 | 0.1890 | 0.8062 / 0.0332 / 0.0316 |
| J5 | squad | in-domain | 0.9798 | 0.9798 | 0.0202 | 0.0744 / 0.0692 / 0.0699 |
| J5 | musique | in-domain | 0.9207 | 0.8026 | 0.0793 | 0.4473 / 0.3478 / 0.3551 |
| J5 | hotpotqa | in-domain | 0.9800 | 0.9684 | 0.0200 | 0.2954 / 0.0775 / 0.0749 |
| J5 | 2wiki | in-domain | 0.9637 | 0.9059 | 0.0363 | 0.3558 / 0.0952 / 0.0911 |
| J5 | webqsp | zero-shot | 0.7649 | 0.6587 | 0.2351 | 0.7113 / 0.5240 / 0.4972 |
| L-metaqa | metaqa | zero-shot | 0.8110 | 0.6987 | 0.1890 | 0.8062 / 0.6503 / 0.6200 |
| L-metaqa | squad | in-domain | 0.9798 | 0.9798 | 0.0202 | 0.0744 / 0.0697 / 0.0681 |
| L-metaqa | musique | in-domain | 0.9207 | 0.8026 | 0.0793 | 0.4473 / 0.3512 / 0.3519 |
| L-metaqa | hotpotqa | in-domain | 0.9800 | 0.9684 | 0.0200 | 0.2954 / 0.0760 / 0.0734 |
| L-metaqa | 2wiki | in-domain | 0.9637 | 0.9059 | 0.0363 | 0.3558 / 0.0973 / 0.0868 |
| L-metaqa | webqsp | zero-shot | 0.7649 | 0.6587 | 0.2351 | 0.7113 / 0.6566 / 0.6475 |
| L-squad | metaqa | in-domain | 0.8110 | 0.6987 | 0.1890 | 0.8062 / 0.0323 / 0.0320 |
| L-squad | squad | zero-shot | 0.9798 | 0.9798 | 0.0202 | 0.0744 / 0.0912 / 0.0904 |
| L-squad | musique | in-domain | 0.9207 | 0.8026 | 0.0793 | 0.4473 / 0.3498 / 0.3486 |
| L-squad | hotpotqa | in-domain | 0.9800 | 0.9684 | 0.0200 | 0.2954 / 0.0766 / 0.0735 |
| L-squad | 2wiki | in-domain | 0.9637 | 0.9059 | 0.0363 | 0.3558 / 0.0913 / 0.0919 |
| L-squad | webqsp | zero-shot | 0.7649 | 0.6587 | 0.2351 | 0.7113 / 0.5008 / 0.4854 |
| L-2wiki | metaqa | in-domain | 0.8110 | 0.6987 | 0.1890 | 0.8062 / 0.0329 / 0.0318 |
| L-2wiki | squad | in-domain | 0.9798 | 0.9798 | 0.0202 | 0.0744 / 0.0686 / 0.0686 |
| L-2wiki | musique | in-domain | 0.9207 | 0.8026 | 0.0793 | 0.4473 / 0.3587 / 0.3560 |
| L-2wiki | hotpotqa | in-domain | 0.9800 | 0.9684 | 0.0200 | 0.2954 / 0.0879 / 0.0780 |
| L-2wiki | 2wiki | zero-shot | 0.9637 | 0.9059 | 0.0363 | 0.3558 / 0.1812 / 0.1670 |
| L-2wiki | webqsp | zero-shot | 0.7649 | 0.6587 | 0.2351 | 0.7113 / 0.5220 / 0.5056 |
| L-musique | metaqa | in-domain | 0.8110 | 0.6987 | 0.1890 | 0.8062 / 0.0330 / 0.0324 |
| L-musique | squad | in-domain | 0.9798 | 0.9798 | 0.0202 | 0.0744 / 0.0680 / 0.0697 |
| L-musique | musique | zero-shot | 0.9207 | 0.8026 | 0.0793 | 0.4473 / 0.4081 / 0.4077 |
| L-musique | hotpotqa | in-domain | 0.9800 | 0.9684 | 0.0200 | 0.2954 / 0.0762 / 0.0749 |
| L-musique | 2wiki | in-domain | 0.9637 | 0.9059 | 0.0363 | 0.3558 / 0.0933 / 0.0880 |
| L-musique | webqsp | zero-shot | 0.7649 | 0.6587 | 0.2351 | 0.7113 / 0.4437 / 0.4401 |
| L-hotpotqa | metaqa | in-domain | 0.8110 | 0.6987 | 0.1890 | 0.8062 / 0.0323 / 0.0310 |
| L-hotpotqa | squad | in-domain | 0.9798 | 0.9798 | 0.0202 | 0.0744 / 0.0684 / 0.0649 |
| L-hotpotqa | musique | in-domain | 0.9207 | 0.8026 | 0.0793 | 0.4473 / 0.3493 / 0.3460 |
| L-hotpotqa | hotpotqa | zero-shot | 0.9800 | 0.9684 | 0.0200 | 0.2954 / 0.1319 / 0.1394 |
| L-hotpotqa | 2wiki | in-domain | 0.9637 | 0.9059 | 0.0363 | 0.3558 / 0.0974 / 0.0874 |
| L-hotpotqa | webqsp | zero-shot | 0.7649 | 0.6587 | 0.2351 | 0.7113 / 0.5285 / 0.5315 |

## 4. Against the published numbers (never a direct comparison: each row's setup differs)

Ours: the GNN track's J5 read (in-domain; webqsp zero-shot) and its zero-shot read from the leave-out fit. Step 4e's pools (every gold in the pool, s1eval) are listed beside today's ceiling.

| dataset | ours J5: R@5 / FC@5 / hit@1 | ours zero-shot: R@5 / FC@5 / hit@1 | ceiling R@5 today | step 4e all-golds | published |
| --- | --- | --- | --- | --- | --- |
| metaqa | 0.779 / 0.667 / 0.888 | 0.191 / 0.160 / 0.019 | 0.811 | all 0.953, golds 0.988 | NuTrea (2-hop / 3-hop) answer Hit@1 [0.9999, 0.9889] (topic entities assigned (an oracle we refuse)) |
| squad | 0.910 / 0.910 / 0.749 | 0.889 / 0.889 / 0.723 | 0.980 | all 0.992, golds 0.992 | none filed |
| musique | 0.566 / 0.259 / 0.695 | 0.513 / 0.204 / 0.641 | 0.921 | all 0.937, golds 0.979 | HippoRAG 2 (NV-Embed-v2 7B, Llama-3.3-70B graph) R@5 0.747 (1,000 dev questions over 11,656 passages); NV-Embed-v2 alone R@5 0.697 (as above); HippoRAG R@5 0.532 (as above); GraphER GAT / GCS / MLP FC@5 [0.256, 0.254, 0.216] (2,000 sampled dev questions, induced corpus, 200 candidates) |
| hotpotqa | 0.905 / 0.849 / 0.856 | 0.841 / 0.722 / 0.848 | 0.980 | all 0.979, golds 0.986 | GraphER GAT / GCS / MLP FC@5 [0.78, 0.788, 0.789] (2,000 sampled dev questions, induced corpus) |
| 2wiki | 0.873 / 0.721 / 0.898 | 0.797 / 0.581 / 0.881 | 0.964 | all 0.945, golds 0.979 | GraphER GAT / GCS / MLP FC@5 [0.441, 0.438, 0.425] (2,000 sampled dev questions, induced corpus) |
| webqsp | 0.268 / 0.209 / 0.134 | - | 0.765 | all 0.909, golds 0.966 | NuTrea answer Hit@1 0.7743 (topic entities assigned (an oracle we refuse), 2-hop subgraph) |

Metric traps (docs/M3A_SOTA_ARCHAEOLOGY.md): GraphER's PR@K is set coverage, our FC@5, never R@5. NuTrea's Hit@1 is answer accuracy with topic entities assigned, an oracle we refuse; set beside our hit@1 only as context. HippoRAG 2's R@5 comes from a 7B encoder over a corpus a tenth of ours.
