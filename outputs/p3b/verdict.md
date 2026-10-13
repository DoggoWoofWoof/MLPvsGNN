# P3b: the gap ledger (s1eval)

Shares are question means, weighted as R@5 (each gold 1 / gt). A missed gold's bucket: POOL (outside the pool), BRIDGE (in it, one link from a found gold), RANKED (the rest); CAPACITY: golds past five slots. Records: outputs/p3b/.

## metaqa

Bars: R@5 LINKED, FC@5 LINKED, hit@1 A1. The bar's golds outside our pools: 0.0006 (R@5 share).

| ranking | R@5 | POOL | BRIDGE | RANKED | of which 6-10 | 11-50 | 51+ | CAPACITY | top rank only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zrc | 0.7796 | 0.0206 | 0.0013 | 0.0288 | 0.0138 | 0.0126 | 0.0025 | 0.1697 | 0.0680 |
| zsp | 0.7794 | 0.0206 | 0.0014 | 0.0289 | 0.0133 | 0.0131 | 0.0024 | 0.1697 | 0.0719 |
| twin0 | 0.7212 | 0.0212 | 0.0038 | 0.0841 | 0.0418 | 0.0380 | 0.0042 | 0.1697 | 0.1679 |
| gnn0 | 0.7839 | 0.0204 | 0.0008 | 0.0252 | 0.0136 | 0.0102 | 0.0014 | 0.1697 | 0.0738 |
| A1 | 0.0096 | 0.0528 | 0.0002 | 0.7677 | 0.0564 | 0.3889 | 0.3223 | 0.1697 | 0.0181 |
| LINKED | 0.2406 | 0.0062 | 0.0021 | 0.5813 | 0.2068 | 0.2061 | 0.1684 | 0.1697 | 0.3158 |

| family | gap (R@5) | lost: POOL | BRIDGE | RANKED | won back: POOL | BRIDGE | RANKED | reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lean | bar>GNN (LINKED) | 0.0006 | 0.0002 | 0.0038 | 0.0037 | 0.0021 | 0.5376 | LEAD RANKED |
| lean | bar>MLP (LINKED) | 0.0006 | 0.0002 | 0.0041 | 0.0036 | 0.0021 | 0.5382 |  |
| lean | GNN>MLP (zsp) | 0.0000 | 0.0010 | 0.0192 | 0.0000 | 0.0021 | 0.0183 | LEAD RANKED |
| universal | bar>GNN (LINKED) | 0.0006 | 0.0003 | 0.0030 | 0.0041 | 0.0023 | 0.5407 | LEAD RANKED |
| universal | bar>MLP (LINKED) | 0.0006 | 0.0003 | 0.0056 | 0.0036 | 0.0019 | 0.4815 |  |
| universal | GNN>MLP (gnn0) | 0.0000 | 0.0060 | 0.0898 | 0.0000 | 0.0031 | 0.0300 | LEAD RANKED |

Beside the bar, seeds removed: 17 of 9785 questions read (51 with no gold left; undecided by a ranking's first ten: A1 9714, A2 9714, DENSE 1162, SPLADE 2802, RRF 4194, A1_all 9714, A2_all 9714, twin0 1, gnn0 4); 9.08 nodes taken out per question, 0.0045 of the golds. Bars: R@5 LINKED, FC@5 LINKED, hit@1 LINKED.

| ranking | R@5 | FC@5 | hit@1 | against the bar (R@5, FC@5, hit@1) | call |
| --- | ---: | ---: | ---: | --- | --- |
| A1 | 0.3578 | 0.3529 | 0.0000 |  |  |
| A2 | 0.3578 | 0.3529 | 0.0000 |  |  |
| LINKED | 0.5098 | 0.4706 | 0.1765 |  |  |
| DENSE | 0.0000 | 0.0000 | 0.0000 |  |  |
| SPLADE | 0.0000 | 0.0000 | 0.0000 |  |  |
| RRF | 0.0000 | 0.0000 | 0.0000 |  |  |
| A1_all | 0.3529 | 0.3529 | 0.0588 |  |  |
| A2_all | 0.3529 | 0.3529 | 0.1176 |  |  |
| zrc | 0.8608 | 0.8235 | 0.7647 | +0.3510, +0.3529, +0.5882 | AT_OR_ABOVE |
| zsp | 0.8608 | 0.8235 | 0.7647 | +0.3510, +0.3529, +0.5882 | AT_OR_ABOVE |
| twin0 | 0.8018 | 0.7647 | 0.7059 | +0.2920, +0.2941, +0.5294 | AT_OR_ABOVE |
| gnn0 | 0.8608 | 0.8235 | 0.8824 | +0.3510, +0.3529, +0.7059 | AT_OR_ABOVE |

## squad

Bars: R@5 A1, FC@5 A1, hit@1 A1. The bar's golds outside our pools: 0.0000 (R@5 share).

| ranking | R@5 | POOL | BRIDGE | RANKED | of which 6-10 | 11-50 | 51+ | CAPACITY | top rank only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zrc | 0.9107 | 0.0202 | 0.0000 | 0.0691 | 0.0319 | 0.0371 | 0.0001 | 0.0000 | 0.1602 |
| zsp | 0.9099 | 0.0202 | 0.0000 | 0.0699 | 0.0307 | 0.0392 | 0.0000 | 0.0000 | 0.1606 |
| twin0 | 0.8971 | 0.0202 | 0.0000 | 0.0827 | 0.0323 | 0.0503 | 0.0001 | 0.0000 | 0.1608 |
| gnn0 | 0.9038 | 0.0202 | 0.0000 | 0.0760 | 0.0309 | 0.0451 | 0.0000 | 0.0000 | 0.1607 |
| A1 | 0.9053 | 0.0193 | 0.0000 | 0.0754 | 0.0344 | 0.0410 | 0.0000 | 0.0000 | 0.1704 |

| family | gap (R@5) | lost: POOL | BRIDGE | RANKED | won back: POOL | BRIDGE | RANKED | reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lean | bar>GNN (A1) | 0.0000 | 0.0000 | 0.0136 | 0.0000 | 0.0000 | 0.0182 | LEAD RANKED |
| lean | bar>MLP (A1) | 0.0000 | 0.0000 | 0.0125 | 0.0000 | 0.0000 | 0.0179 |  |
| lean | GNN>MLP (zsp) | 0.0000 | 0.0000 | 0.0061 | 0.0000 | 0.0000 | 0.0069 | LEAD RANKED |
| universal | bar>GNN (A1) | 0.0000 | 0.0000 | 0.0198 | 0.0000 | 0.0000 | 0.0183 | LEAD RANKED |
| universal | bar>MLP (A1) | 0.0000 | 0.0000 | 0.0267 | 0.0000 | 0.0000 | 0.0184 |  |
| universal | GNN>MLP (gnn0) | 0.0000 | 0.0000 | 0.0196 | 0.0000 | 0.0000 | 0.0129 | LEAD RANKED |

## musique

Bars: R@5 RRF, FC@5 DENSE, hit@1 RRF. The bar's golds outside our pools: 0.0000 (R@5 share).

| ranking | R@5 | POOL | BRIDGE | RANKED | of which 6-10 | 11-50 | 51+ | CAPACITY | top rank only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zrc | 0.5669 | 0.0793 | 0.1213 | 0.2326 | 0.0469 | 0.0943 | 0.0914 | 0.0000 | 0.1961 |
| zsp | 0.5657 | 0.0793 | 0.1173 | 0.2378 | 0.0525 | 0.0936 | 0.0917 | 0.0000 | 0.2031 |
| twin0 | 0.5119 | 0.0793 | 0.1345 | 0.2744 | 0.0471 | 0.1013 | 0.1259 | 0.0000 | 0.2218 |
| gnn0 | 0.5380 | 0.0793 | 0.1072 | 0.2756 | 0.0558 | 0.1112 | 0.1087 | 0.0000 | 0.2189 |
| DENSE | 0.4572 | 0.1275 | 0.1154 | 0.2999 | 0.0539 | 0.1052 | 0.1408 | 0.0000 | 0.2354 |
| RRF | 0.4734 | 0.1029 | 0.1270 | 0.2967 | 0.0562 | 0.1020 | 0.1385 | 0.0000 | 0.2305 |

| family | gap (R@5) | lost: POOL | BRIDGE | RANKED | won back: POOL | BRIDGE | RANKED | reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lean | bar>GNN (RRF) | 0.0000 | 0.0057 | 0.0206 | 0.0145 | 0.0486 | 0.0555 | LEAD RANKED |
| lean | bar>MLP (RRF) | 0.0000 | 0.0056 | 0.0179 | 0.0121 | 0.0471 | 0.0578 |  |
| lean | GNN>MLP (zsp) | 0.0000 | 0.0103 | 0.0123 | 0.0000 | 0.0066 | 0.0172 | SPLIT RANKED+BRIDGE |
| universal | bar>GNN (RRF) | 0.0000 | 0.0082 | 0.0441 | 0.0153 | 0.0539 | 0.0476 | LEAD RANKED |
| universal | bar>MLP (RRF) | 0.0000 | 0.0164 | 0.0452 | 0.0132 | 0.0381 | 0.0488 |  |
| universal | GNN>MLP (gnn0) | 0.0000 | 0.0398 | 0.0324 | 0.0000 | 0.0131 | 0.0330 | SPLIT BRIDGE+RANKED |

## hotpotqa

Bars: R@5 RRF, FC@5 A1, hit@1 RRF. The bar's golds outside our pools: 0.0000 (R@5 share).

| ranking | R@5 | POOL | BRIDGE | RANKED | of which 6-10 | 11-50 | 51+ | CAPACITY | top rank only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zrc | 0.9049 | 0.0200 | 0.0472 | 0.0279 | 0.0112 | 0.0147 | 0.0020 | 0.0000 | 0.1106 |
| zsp | 0.9051 | 0.0200 | 0.0451 | 0.0298 | 0.0113 | 0.0165 | 0.0020 | 0.0000 | 0.1056 |
| twin0 | 0.8786 | 0.0200 | 0.0716 | 0.0298 | 0.0120 | 0.0158 | 0.0020 | 0.0000 | 0.1299 |
| gnn0 | 0.8993 | 0.0200 | 0.0522 | 0.0286 | 0.0120 | 0.0145 | 0.0020 | 0.0000 | 0.0957 |
| A1 | 0.6766 | 0.0172 | 0.2082 | 0.0979 | 0.0502 | 0.0367 | 0.0110 | 0.0000 | 0.4433 |
| RRF | 0.6845 | 0.1438 | 0.1171 | 0.0546 | 0.0192 | 0.0300 | 0.0054 | 0.0000 | 0.1160 |

| family | gap (R@5) | lost: POOL | BRIDGE | RANKED | won back: POOL | BRIDGE | RANKED | reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lean | bar>GNN (RRF) | 0.0000 | 0.0047 | 0.0057 | 0.1001 | 0.1024 | 0.0284 | SPLIT RANKED+BRIDGE |
| lean | bar>MLP (RRF) | 0.0000 | 0.0053 | 0.0051 | 0.1007 | 0.1007 | 0.0293 |  |
| lean | GNN>MLP (zsp) | 0.0000 | 0.0087 | 0.0026 | 0.0000 | 0.0070 | 0.0041 | LEAD BRIDGE |
| universal | bar>GNN (RRF) | 0.0000 | 0.0034 | 0.0060 | 0.0948 | 0.0991 | 0.0302 | LEAD RANKED |
| universal | bar>MLP (RRF) | 0.0000 | 0.0094 | 0.0062 | 0.0894 | 0.0927 | 0.0275 |  |
| universal | GNN>MLP (gnn0) | 0.0000 | 0.0335 | 0.0075 | 0.0000 | 0.0141 | 0.0062 | LEAD BRIDGE |

## 2wiki

Bars: R@5 RRF, FC@5 SPLADE, hit@1 RRF. The bar's golds outside our pools: 0.0000 (R@5 share).

| ranking | R@5 | POOL | BRIDGE | RANKED | of which 6-10 | 11-50 | 51+ | CAPACITY | top rank only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zrc | 0.8697 | 0.0363 | 0.0619 | 0.0320 | 0.0157 | 0.0153 | 0.0010 | 0.0000 | 0.0912 |
| zsp | 0.8726 | 0.0363 | 0.0537 | 0.0375 | 0.0172 | 0.0190 | 0.0013 | 0.0000 | 0.0847 |
| twin0 | 0.8487 | 0.0363 | 0.0826 | 0.0324 | 0.0166 | 0.0145 | 0.0013 | 0.0000 | 0.1151 |
| gnn0 | 0.8906 | 0.0363 | 0.0438 | 0.0293 | 0.0155 | 0.0130 | 0.0008 | 0.0000 | 0.0685 |
| RRF | 0.6079 | 0.2548 | 0.0776 | 0.0597 | 0.0284 | 0.0279 | 0.0033 | 0.0000 | 0.1093 |
| SPLADE | 0.6055 | 0.2738 | 0.0642 | 0.0564 | 0.0231 | 0.0277 | 0.0057 | 0.0000 | 0.1167 |

| family | gap (R@5) | lost: POOL | BRIDGE | RANKED | won back: POOL | BRIDGE | RANKED | reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lean | bar>GNN (RRF) | 0.0000 | 0.0040 | 0.0117 | 0.1790 | 0.0625 | 0.0389 | LEAD RANKED |
| lean | bar>MLP (RRF) | 0.0000 | 0.0049 | 0.0093 | 0.1727 | 0.0632 | 0.0401 |  |
| lean | GNN>MLP (zsp) | 0.0000 | 0.0164 | 0.0027 | 0.0000 | 0.0094 | 0.0067 | LEAD BRIDGE |
| universal | bar>GNN (RRF) | 0.0000 | 0.0034 | 0.0090 | 0.1873 | 0.0655 | 0.0424 | LEAD RANKED |
| universal | bar>MLP (RRF) | 0.0000 | 0.0096 | 0.0096 | 0.1667 | 0.0531 | 0.0402 |  |
| universal | GNN>MLP (gnn0) | 0.0000 | 0.0508 | 0.0099 | 0.0000 | 0.0118 | 0.0069 | LEAD BRIDGE |

## webqsp

Bars: R@5 DENSE, FC@5 DENSE, hit@1 DENSE. The bar's golds outside our pools: 0.0000 (R@5 share).

| ranking | R@5 | POOL | BRIDGE | RANKED | of which 6-10 | 11-50 | 51+ | CAPACITY | top rank only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zrc | 0.3071 | 0.1346 | 0.0198 | 0.4251 | 0.0980 | 0.2089 | 0.1182 | 0.1135 | 0.3546 |
| zsp | 0.2677 | 0.1345 | 0.0185 | 0.4658 | 0.0926 | 0.2196 | 0.1536 | 0.1135 | 0.2741 |
| twin0 | 0.6015 | 0.1281 | 0.0123 | 0.1447 | 0.0494 | 0.0726 | 0.0226 | 0.1135 | 0.2309 |
| gnn0 | 0.6289 | 0.1260 | 0.0099 | 0.1218 | 0.0356 | 0.0595 | 0.0268 | 0.1135 | 0.1790 |
| DENSE | 0.1095 | 0.4142 | 0.0053 | 0.3575 | 0.0239 | 0.0844 | 0.2493 | 0.1135 | 0.0971 |

| family | gap (R@5) | lost: POOL | BRIDGE | RANKED | won back: POOL | BRIDGE | RANKED | reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lean | bar>GNN (DENSE) | 0.0000 | 0.0038 | 0.0592 | 0.0728 | 0.0018 | 0.1466 | LEAD RANKED |
| lean | bar>MLP (DENSE) | 0.0000 | 0.0032 | 0.0539 | 0.0784 | 0.0020 | 0.1745 |  |
| lean | GNN>MLP (zsp) | 0.0000 | 0.0024 | 0.0369 | 0.0000 | 0.0029 | 0.0759 | LEAD RANKED |
| universal | bar>GNN (DENSE) | 0.0000 | 0.0015 | 0.0051 | 0.2456 | 0.0045 | 0.2760 | LEAD RANKED |
| universal | bar>MLP (DENSE) | 0.0000 | 0.0019 | 0.0089 | 0.2321 | 0.0035 | 0.2672 |  |
| universal | GNN>MLP (gnn0) | 0.0000 | 0.0080 | 0.0634 | 0.0000 | 0.0061 | 0.0380 | LEAD RANKED |

Beside the bar, seeds removed: 12 of 1503 questions read (136 with no gold left; undecided by a ranking's first ten: A1 1353, A2 1353, LINKED 239, DENSE 744, SPLADE 677, RRF 782, A1_all 1353, A2_all 1353, zrc 27, zsp 19, twin0 1, gnn0 5); 10.84 nodes taken out per question, 0.0324 of the golds. Bars: R@5 A1, FC@5 A1, hit@1 A1.

| ranking | R@5 | FC@5 | hit@1 | against the bar (R@5, FC@5, hit@1) | call |
| --- | ---: | ---: | ---: | --- | --- |
| A1 | 0.3333 | 0.3333 | 0.2500 |  |  |
| A2 | 0.3333 | 0.3333 | 0.2500 |  |  |
| LINKED | 0.0833 | 0.0833 | 0.0000 |  |  |
| DENSE | 0.0000 | 0.0000 | 0.0000 |  |  |
| SPLADE | 0.0000 | 0.0000 | 0.0000 |  |  |
| RRF | 0.0000 | 0.0000 | 0.0000 |  |  |
| A1_all | 0.3333 | 0.3333 | 0.2500 |  |  |
| A2_all | 0.3333 | 0.3333 | 0.2500 |  |  |
| zrc | 0.4583 | 0.4167 | 0.2500 | +0.1250, +0.0833, +0.0000 | AT_OR_ABOVE |
| zsp | 0.5833 | 0.5000 | 0.1667 | +0.2500, +0.1667, -0.0833 | AT_OR_ABOVE |
| twin0 | 0.8333 | 0.8333 | 0.7500 | +0.5000, +0.5000, +0.5000 | AT_OR_ABOVE |
| gnn0 | 0.8472 | 0.8333 | 0.7500 | +0.5139, +0.5000, +0.5000 | AT_OR_ABOVE |

