# U2d's screen verdict: the lean pair's full run: **none: the lean pair stays on P1c2's pools**; zg6: **P1c2's pools, as filed**

Rule: ADOPT_FOR_FULL_RUN when no arm's re-call under U2d's seed null has a LOSS on any of its twelve reads and the arms together have at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a re-call is missing, is not U2d's, does not hold twelve reads, or its fits ran on different devices (docs/U2D_REFIT_ON_U2E.md, section 7).

- **A** (the pools, for the lean pair; zrckp on the new build against zrckp on the base build; zspk on the new build against zspk on the base build): **NOT_ADOPTED**, 3 GAIN (zrckp L-musique webqsp, zrckp L-hotpotqa webqsp, zspk L-musique musique), 8 LOSS (zrckp L-musique hotpotqa, zrckp L-musique 2wiki, zrckp L-hotpotqa hotpotqa, zrckp L-hotpotqa 2wiki, zspk L-musique hotpotqa, zspk L-musique 2wiki, zspk L-hotpotqa hotpotqa, zspk L-hotpotqa 2wiki).
- **B** (via, on the new pools; zrckpv on the new build against zrckp on the new build; zspkv on the new build against zspk on the new build): **NOT_ADOPTED**, 0 GAIN (none), 2 LOSS (zrckpv L-musique webqsp, zrckpv L-hotpotqa webqsp).
- **C** (the pools and via together; zrckpv on the new build against zrckp on the base build; zspkv on the new build against zspk on the base build): **NOT_ADOPTED**, 1 GAIN (zspkv L-musique musique), 8 LOSS (zrckpv L-musique hotpotqa, zrckpv L-musique 2wiki, zrckpv L-hotpotqa hotpotqa, zrckpv L-hotpotqa 2wiki, zspkv L-musique hotpotqa, zspkv L-musique 2wiki, zspkv L-hotpotqa hotpotqa, zspkv L-hotpotqa 2wiki).
- **G** (the pools, for zg6; zg6 on the new build against zg6 on the base build): **NOT_ADOPTED**, 1 GAIN (zg6 L-musique musique), 5 LOSS (zg6 L-musique hotpotqa, zg6 L-musique 2wiki, zg6 L-musique webqsp, zg6 L-hotpotqa hotpotqa, zg6 L-hotpotqa webqsp).

C decides only when A is NOT_ADOPTED and B is ADOPT_FOR_FULL_RUN; G decides only zg6's pools.

Next forms (section 9, fixed before any number):

- the 2 x 2: U2e's rule on today's graph, which separates the graph from the pool rule; then via on P1c2's pools
- via in the block encoder's own input layer, in place of the heads; then the link-level form: the parent's own first-stage weight and cosine per link
- zg6 with the via columns (zg6v) on the new build
- the hit@1 diagnostic on metaqa and webqsp: where the gold sits in the order when hit@1 misses

## Beside, deciding nothing

R@5 / FC@5 / hit@1 per fit (seed 0) and read; rrf from the new fit's read of each build:

| build | split | dataset | rrf | zrckp | zspk | zg6 | zrckpv | zspkv |
|---|---|---|---|---|---|---|---|---|
| base | L-musique | metaqa | 0.0047 / 0.0026 / 0.0008 | 0.8008 / 0.6842 / 0.9139 | 0.8021 / 0.6862 / 0.9197 | 0.8123 / 0.6979 / 0.9551 |  |  |
| base | L-musique | squad | 0.9054 / 0.9054 / 0.7335 | 0.9121 / 0.9121 / 0.7483 | 0.9118 / 0.9118 / 0.7461 | 0.9106 / 0.9106 / 0.7523 |  |  |
| base | L-musique | musique | 0.4734 / 0.1353 / 0.6330 | 0.5153 / 0.1953 / 0.6947 | 0.5143 / 0.1982 / 0.6731 | 0.5207 / 0.2102 / 0.6847 |  |  |
| base | L-musique | hotpotqa | 0.6846 / 0.4240 / 0.8277 | 0.9055 / 0.8454 / 0.8344 | 0.9088 / 0.8521 / 0.8573 | 0.9118 / 0.8597 / 0.8632 |  |  |
| base | L-musique | 2wiki | 0.6079 / 0.2536 / 0.8747 | 0.9345 / 0.8402 / 0.9712 | 0.9346 / 0.8424 / 0.9703 | 0.9416 / 0.8616 / 0.9726 |  |  |
| base | L-musique | webqsp | 0.0535 / 0.0379 / 0.0459 | 0.3561 / 0.2841 / 0.1450 | 0.3162 / 0.2448 / 0.1823 | 0.2368 / 0.1850 / 0.1304 |  |  |
| base | L-hotpotqa | metaqa | 0.0047 / 0.0026 / 0.0008 | 0.8033 / 0.6863 / 0.9239 | 0.8043 / 0.6880 / 0.9290 | 0.8112 / 0.6963 / 0.9541 |  |  |
| base | L-hotpotqa | squad | 0.9054 / 0.9054 / 0.7335 | 0.9124 / 0.9124 / 0.7500 | 0.9117 / 0.9117 / 0.7492 | 0.9118 / 0.9118 / 0.7525 |  |  |
| base | L-hotpotqa | musique | 0.4734 / 0.1353 / 0.6330 | 0.5697 / 0.2511 / 0.7195 | 0.5800 / 0.2590 / 0.7286 | 0.5937 / 0.2826 / 0.7679 |  |  |
| base | L-hotpotqa | hotpotqa | 0.6846 / 0.4240 / 0.8277 | 0.8538 / 0.7463 / 0.8047 | 0.8512 / 0.7429 / 0.8099 | 0.8309 / 0.7207 / 0.7691 |  |  |
| base | L-hotpotqa | 2wiki | 0.6079 / 0.2536 / 0.8747 | 0.9303 / 0.8244 / 0.9699 | 0.9290 / 0.8232 / 0.9738 | 0.9391 / 0.8528 / 0.9828 |  |  |
| base | L-hotpotqa | webqsp | 0.0535 / 0.0379 / 0.0459 | 0.3370 / 0.2748 / 0.1238 | 0.2773 / 0.2122 / 0.1477 | 0.2189 / 0.1717 / 0.1411 |  |  |
| new | L-musique | metaqa | 0.0047 / 0.0026 / 0.0008 | 0.7999 / 0.6841 / 0.9098 | 0.7991 / 0.6837 / 0.9090 | 0.8134 / 0.6995 / 0.9569 | 0.7986 / 0.6828 / 0.9039 | 0.7997 / 0.6839 / 0.9051 |
| new | L-musique | squad | 0.9054 / 0.9054 / 0.7335 | 0.9129 / 0.9129 / 0.7510 | 0.9137 / 0.9137 / 0.7524 | 0.9141 / 0.9141 / 0.7539 | 0.9114 / 0.9114 / 0.7520 | 0.9127 / 0.9127 / 0.7525 |
| new | L-musique | musique | 0.4734 / 0.1353 / 0.6330 | 0.5189 / 0.2048 / 0.7025 | 0.5257 / 0.2135 / 0.7058 | 0.5426 / 0.2247 / 0.7029 | 0.5195 / 0.2098 / 0.7025 | 0.5264 / 0.2106 / 0.6843 |
| new | L-musique | hotpotqa | 0.6846 / 0.4240 / 0.8277 | 0.8484 / 0.7341 / 0.8280 | 0.8483 / 0.7327 / 0.8348 | 0.8604 / 0.7530 / 0.8681 | 0.8450 / 0.7260 / 0.8278 | 0.8464 / 0.7284 / 0.8347 |
| new | L-musique | 2wiki | 0.6079 / 0.2536 / 0.8747 | 0.9204 / 0.8040 / 0.9757 | 0.9187 / 0.7982 / 0.9776 | 0.9306 / 0.8331 / 0.9772 | 0.9147 / 0.7894 / 0.9754 | 0.9159 / 0.7875 / 0.9729 |
| new | L-musique | webqsp | 0.0535 / 0.0379 / 0.0459 | 0.3951 / 0.3194 / 0.1750 | 0.3223 / 0.2468 / 0.1963 | 0.1560 / 0.1218 / 0.0931 | 0.3729 / 0.3034 / 0.1570 | 0.3150 / 0.2435 / 0.1870 |
| new | L-hotpotqa | metaqa | 0.0047 / 0.0026 / 0.0008 | 0.8009 / 0.6838 / 0.9150 | 0.7999 / 0.6830 / 0.9117 | 0.8134 / 0.6999 / 0.9586 | 0.8002 / 0.6833 / 0.9185 | 0.7990 / 0.6824 / 0.9110 |
| new | L-hotpotqa | squad | 0.9054 / 0.9054 / 0.7335 | 0.9131 / 0.9131 / 0.7500 | 0.9159 / 0.9159 / 0.7485 | 0.9137 / 0.9137 / 0.7555 | 0.9128 / 0.9128 / 0.7483 | 0.9143 / 0.9143 / 0.7506 |
| new | L-hotpotqa | musique | 0.4734 / 0.1353 / 0.6330 | 0.5741 / 0.2586 / 0.7228 | 0.5765 / 0.2573 / 0.7323 | 0.5985 / 0.2859 / 0.7604 | 0.5737 / 0.2520 / 0.7261 | 0.5727 / 0.2507 / 0.7158 |
| new | L-hotpotqa | hotpotqa | 0.6846 / 0.4240 / 0.8277 | 0.8064 / 0.6536 / 0.8244 | 0.8018 / 0.6456 / 0.8185 | 0.7810 / 0.6196 / 0.7670 | 0.8087 / 0.6579 / 0.8262 | 0.8086 / 0.6589 / 0.8216 |
| new | L-hotpotqa | 2wiki | 0.6079 / 0.2536 / 0.8747 | 0.9029 / 0.7564 / 0.9770 | 0.9095 / 0.7712 / 0.9793 | 0.9267 / 0.8224 / 0.9863 | 0.9060 / 0.7657 / 0.9757 | 0.9107 / 0.7770 / 0.9783 |
| new | L-hotpotqa | webqsp | 0.0535 / 0.0379 / 0.0459 | 0.3870 / 0.3120 / 0.1590 | 0.2942 / 0.2269 / 0.1630 | 0.1781 / 0.1377 / 0.1091 | 0.3520 / 0.2821 / 0.1264 | 0.2867 / 0.2202 / 0.1663 |

The MLP share (zrckp - rrf) / (GNN - rrf) on R@5, where the GNN's gain over rrf exceeds 0.002:

| read | share |
|---|---:|
| base L-musique metaqa zrckp/zspk | 0.9984 |
| base L-musique metaqa zrckp/zg6 | 0.9858 |
| base L-musique squad zrckp/zspk | 1.0469 |
| base L-musique squad zrckp/zg6 | 1.2885 |
| base L-musique musique zrckp/zspk | 1.0244 |
| base L-musique musique zrckp/zg6 | 0.8858 |
| base L-musique hotpotqa zrckp/zspk | 0.9853 |
| base L-musique hotpotqa zrckp/zg6 | 0.9723 |
| base L-musique 2wiki zrckp/zspk | 0.9997 |
| base L-musique 2wiki zrckp/zg6 | 0.9787 |
| base L-musique webqsp zrckp/zspk | 1.1519 |
| base L-musique webqsp zrckp/zg6 | 1.6508 |
| base L-hotpotqa metaqa zrckp/zspk | 0.9987 |
| base L-hotpotqa metaqa zrckp/zg6 | 0.9902 |
| base L-hotpotqa squad zrckp/zspk | 1.1111 |
| base L-hotpotqa squad zrckp/zg6 | 1.0937 |
| base L-hotpotqa musique zrckp/zspk | 0.9034 |
| base L-hotpotqa musique zrckp/zg6 | 0.8005 |
| base L-hotpotqa hotpotqa zrckp/zspk | 1.0156 |
| base L-hotpotqa hotpotqa zrckp/zg6 | 1.1565 |
| base L-hotpotqa 2wiki zrckp/zspk | 1.0040 |
| base L-hotpotqa 2wiki zrckp/zg6 | 0.9734 |
| base L-hotpotqa webqsp zrckp/zspk | 1.2668 |
| base L-hotpotqa webqsp zrckp/zg6 | 1.7140 |
| new L-musique metaqa zrckp/zspk | 1.0010 |
| new L-musique metaqa zrckp/zg6 | 0.9833 |
| new L-musique squad zrckp/zspk | 0.9036 |
| new L-musique squad zrckp/zg6 | 0.8621 |
| new L-musique musique zrckp/zspk | 0.8700 |
| new L-musique musique zrckp/zg6 | 0.6575 |
| new L-musique hotpotqa zrckp/zspk | 1.0006 |
| new L-musique hotpotqa zrckp/zg6 | 0.9317 |
| new L-musique 2wiki zrckp/zspk | 1.0055 |
| new L-musique 2wiki zrckp/zg6 | 0.9684 |
| new L-musique webqsp zrckp/zspk | 1.2708 |
| new L-musique webqsp zrckp/zg6 | 3.3327 |
| new L-hotpotqa metaqa zrckp/zspk | 1.0013 |
| new L-hotpotqa metaqa zrckp/zg6 | 0.9845 |
| new L-hotpotqa squad zrckp/zspk | 0.7333 |
| new L-hotpotqa squad zrckp/zg6 | 0.9277 |
| new L-hotpotqa musique zrckp/zspk | 0.9767 |
| new L-hotpotqa musique zrckp/zg6 | 0.8050 |
| new L-hotpotqa hotpotqa zrckp/zspk | 1.0392 |
| new L-hotpotqa hotpotqa zrckp/zg6 | 1.2635 |
| new L-hotpotqa 2wiki zrckp/zspk | 0.9781 |
| new L-hotpotqa 2wiki zrckp/zg6 | 0.9253 |
| new L-hotpotqa webqsp zrckp/zspk | 1.3855 |
| new L-hotpotqa webqsp zrckp/zg6 | 2.6766 |

metaqa and webqsp: hit@1 is the headline; R@5 against its cap (the pool's, s1eval):

| build and dataset | hit@1 cap | R@5 cap | fits (R@5, hit@1) |
|---|---:|---:|---|
| base metaqa | 0.9976 | 0.8273 | zrckp L-musique: 0.8008, 0.9139; zspk L-musique: 0.8021, 0.9197; zg6 L-musique: 0.8123, 0.9551; zrckp L-hotpotqa: 0.8033, 0.9239; zspk L-hotpotqa: 0.8043, 0.9290; zg6 L-hotpotqa: 0.8112, 0.9541 |
| base webqsp | 0.9667 | 0.8342 | zrckp L-musique: 0.3561, 0.1450; zspk L-musique: 0.3162, 0.1823; zg6 L-musique: 0.2368, 0.1304; zrckp L-hotpotqa: 0.3370, 0.1238; zspk L-hotpotqa: 0.2773, 0.1477; zg6 L-hotpotqa: 0.2189, 0.1411 |
| new metaqa | 0.9993 | 0.8289 | zrckp L-musique: 0.7999, 0.9098; zspk L-musique: 0.7991, 0.9090; zg6 L-musique: 0.8134, 0.9569; zrckpv L-musique: 0.7986, 0.9039; zspkv L-musique: 0.7997, 0.9051; zrckp L-hotpotqa: 0.8009, 0.9150; zspk L-hotpotqa: 0.7999, 0.9117; zg6 L-hotpotqa: 0.8134, 0.9586; zrckpv L-hotpotqa: 0.8002, 0.9185; zspkv L-hotpotqa: 0.7990, 0.9110 |
| new webqsp | 0.9854 | 0.8617 | zrckp L-musique: 0.3951, 0.1750; zspk L-musique: 0.3223, 0.1963; zg6 L-musique: 0.1560, 0.0931; zrckpv L-musique: 0.3729, 0.1570; zspkv L-musique: 0.3150, 0.1870; zrckp L-hotpotqa: 0.3870, 0.1590; zspk L-hotpotqa: 0.2942, 0.1630; zg6 L-hotpotqa: 0.1781, 0.1091; zrckpv L-hotpotqa: 0.3520, 0.1264; zspkv L-hotpotqa: 0.2867, 0.1663 |

Each carve's pool (questions, mean rows, ALL and recall of I and the pool, caps):

| build | carve | questions | rows | ALL I | ALL pool | recall pool | R@5 cap | hit@1 cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 2wiki__fit | 5928 | 106.0 | 0.9025 | 0.9501 | 0.9791 | 0.9791 | 0.9997 |
| base | 2wiki__s1eval | 12576 | 105.7 | 0.9059 | 0.9587 | 0.9839 | 0.9839 | 1.0000 |
| base | hotpotqa__fit | 5930 | 94.0 | 0.9681 | 0.9627 | 0.9788 | 0.9788 | 0.9949 |
| base | hotpotqa__s1eval | 7405 | 93.7 | 0.9684 | 0.9594 | 0.9761 | 0.9761 | 0.9928 |
| base | metaqa__fit | 5960 | 2016.9 | 0.9010 | 0.9622 | 0.9887 | 0.8304 | 0.9970 |
| base | metaqa__s1eval | 9785 | 2017.6 | 0.8949 | 0.9606 | 0.9889 | 0.8273 | 0.9976 |
| base | musique__s1fit | 4601 | 2091.5 | 0.8185 | 0.9213 | 0.9686 | 0.9686 | 0.9987 |
| base | musique__s1eval | 2417 | 2092.4 | 0.8026 | 0.8999 | 0.9628 | 0.9628 | 0.9992 |
| base | squad__fit | 5856 | 50.0 | 0.9833 | 0.9833 | 0.9833 | 0.9833 | 0.9833 |
| base | squad__s1eval | 11873 | 50.0 | 0.9798 | 0.9798 | 0.9798 | 0.9798 | 0.9798 |
| base | webqsp__s1eval | 1503 | 2119.9 | 0.6880 | 0.8410 | 0.9190 | 0.8342 | 0.9667 |
| new | 2wiki__fit | 5928 | 838.3 | 0.8463 | 0.9573 | 0.9841 | 0.9841 | 1.0000 |
| new | 2wiki__s1eval | 12576 | 848.3 | 0.8519 | 0.9671 | 0.9893 | 0.9893 | 1.0000 |
| new | hotpotqa__fit | 5930 | 554.4 | 0.9017 | 0.9631 | 0.9804 | 0.9804 | 0.9978 |
| new | hotpotqa__s1eval | 7405 | 555.7 | 0.8883 | 0.9530 | 0.9756 | 0.9756 | 0.9981 |
| new | metaqa__fit | 5960 | 4390.6 | 0.9010 | 0.9760 | 0.9953 | 0.8322 | 0.9992 |
| new | metaqa__s1eval | 9785 | 4387.5 | 0.8949 | 0.9752 | 0.9956 | 0.8289 | 0.9993 |
| new | musique__s1fit | 4601 | 5390.3 | 0.8246 | 0.9611 | 0.9847 | 0.9847 | 0.9998 |
| new | musique__s1eval | 2417 | 5217.2 | 0.8089 | 0.9450 | 0.9815 | 0.9815 | 1.0000 |
| new | squad__fit | 5856 | 50.0 | 0.9833 | 0.9833 | 0.9833 | 0.9833 | 0.9833 |
| new | squad__s1eval | 11873 | 50.0 | 0.9798 | 0.9798 | 0.9798 | 0.9798 | 0.9798 |
| new | webqsp__s1eval | 1503 | 4754.6 | 0.6880 | 0.8929 | 0.9592 | 0.8617 | 0.9854 |

zg6 against zspk on each build (delta R@5 [95% CI], FC@5, hit@1, call at 0.0075):

- base L-musique: metaqa +0.0102 [+0.0085, +0.0118] +0.0118 +0.0355 GAIN; squad -0.0012 [-0.0033, +0.0010] -0.0012 +0.0061 WITHIN; musique +0.0063 [-0.0010, +0.0131] +0.0120 +0.0116 WITHIN; hotpotqa +0.0030 [+0.0005, +0.0053] +0.0076 +0.0059 WITHIN; 2wiki +0.0070 [+0.0055, +0.0087] +0.0192 +0.0024 WITHIN; webqsp -0.0795 [-0.1003, -0.0596] -0.0599 -0.0519 LOSS
- base L-hotpotqa: metaqa +0.0068 [+0.0054, +0.0083] +0.0083 +0.0251 WITHIN; squad +0.0001 [-0.0020, +0.0023] +0.0001 +0.0034 WITHIN; musique +0.0138 [+0.0070, +0.0204] +0.0236 +0.0393 GAIN; hotpotqa -0.0203 [-0.0248, -0.0161] -0.0221 -0.0408 LOSS; 2wiki +0.0101 [+0.0080, +0.0123] +0.0297 +0.0091 GAIN; webqsp -0.0584 [-0.0783, -0.0381] -0.0406 -0.0067 LOSS
- new L-musique: metaqa +0.0143 [+0.0124, +0.0162] +0.0158 +0.0478 GAIN; squad +0.0004 [-0.0016, +0.0024] +0.0004 +0.0015 WITHIN; musique +0.0170 [+0.0104, +0.0230] +0.0112 -0.0029 GAIN; hotpotqa +0.0121 [+0.0092, +0.0151] +0.0203 +0.0332 GAIN; 2wiki +0.0119 [+0.0101, +0.0138] +0.0349 -0.0004 GAIN; webqsp -0.1663 [-0.1876, -0.1451] -0.1251 -0.1031 LOSS
- new L-hotpotqa: metaqa +0.0135 [+0.0117, +0.0153] +0.0170 +0.0469 GAIN; squad -0.0023 [-0.0045, +0.0000] -0.0023 +0.0070 WITHIN; musique +0.0219 [+0.0161, +0.0279] +0.0285 +0.0281 GAIN; hotpotqa -0.0208 [-0.0251, -0.0164] -0.0261 -0.0515 LOSS; 2wiki +0.0172 [+0.0149, +0.0194] +0.0511 +0.0070 GAIN; webqsp -0.1161 [-0.1364, -0.0939] -0.0892 -0.0539 LOSS

The base build's KT arrays against P1c3's: 2wiki/fit REPRODUCED, 2wiki/s1eval REPRODUCED, hotpotqa/fit REPRODUCED, hotpotqa/s1eval REPRODUCED, metaqa/fit REPRODUCED, metaqa/s1eval REPRODUCED, musique/s1eval REPRODUCED, musique/s1fit REPRODUCED, squad/fit REPRODUCED, squad/s1eval REPRODUCED, webqsp/s1eval REPRODUCED.

U2d's floors beside the 7 October null's: see the json (beside.floors).
Rows per question and the bytes each fit's training data took: see the json (beside.sizes).
