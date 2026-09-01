# Package B: edge families against their common candidate ceiling

Package B changes only the global edge family, so every family receives the same frozen candidate pool and one ceiling per dataset governs all of them. The absolute level is therefore an upstream candidate constraint that no edge family can move, while the difference between families is a topology and ranking effect, because the ceiling cancels between them.

## Families against the shared ceiling

| Dataset | Family | Edges | Ceil@5 | QLS R@5 | GNN R@5 | Ceil-QLS | Ceil-GNN | QLS att. | GNN att. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | sealed_a_multigraph | 855146 | 0.7967 | 0.6840 | 0.6985 | 0.1126 | 0.0982 | 0.859 | 0.877 |
| 2wiki_clean | baseline_a_simple | 521614 | 0.7967 | 0.6781 | 0.7005 | 0.1186 | 0.0961 | 0.851 | 0.879 |
| 2wiki_clean | symbolic_b | 1240748 | 0.7967 | 0.6813 | 0.6964 | 0.1154 | 0.1003 | 0.855 | 0.874 |
| 2wiki_clean | knn_only | 269474 | 0.7967 | 0.6611 | 0.6559 | 0.1356 | 0.1408 | 0.830 | 0.823 |
| 2wiki_clean | full_union_c | 1457378 | 0.7967 | 0.6716 | 0.6944 | 0.1250 | 0.1023 | 0.843 | 0.872 |
| musique_clean | sealed_a_multigraph | 280108 | 0.9405 | 0.8028 | 0.8124 | 0.1377 | 0.1281 | 0.854 | 0.864 |
| musique_clean | baseline_a_simple | 157898 | 0.9405 | 0.8018 | 0.8125 | 0.1387 | 0.1280 | 0.852 | 0.864 |
| musique_clean | symbolic_b | 290086 | 0.9405 | 0.8098 | 0.8153 | 0.1308 | 0.1252 | 0.861 | 0.867 |
| musique_clean | knn_only | 54812 | 0.9405 | 0.8023 | 0.8100 | 0.1382 | 0.1306 | 0.853 | 0.861 |
| musique_clean | full_union_c | 327446 | 0.9405 | 0.8055 | 0.8173 | 0.1350 | 0.1232 | 0.856 | 0.869 |
| webqsp | sealed_a_multigraph | 13379166 | 0.4501 | 0.3337 | 0.3309 | 0.1164 | 0.1192 | 0.741 | 0.735 |
| webqsp | baseline_a_simple | 6621594 | 0.4501 | 0.3297 | 0.3276 | 0.1204 | 0.1225 | 0.732 | 0.728 |
| webqsp | symbolic_b | 5228122 | 0.4501 | 0.3239 | 0.3392 | 0.1262 | 0.1109 | 0.720 | 0.754 |
| webqsp | knn_only | 3345462 | 0.4501 | 0.2973 | 0.2855 | 0.1528 | 0.1646 | 0.660 | 0.634 |
| webqsp | full_union_c | 8309152 | 0.4501 | 0.3305 | 0.3264 | 0.1196 | 0.1237 | 0.734 | 0.725 |
| hotpotqa_clean | sealed_a_multigraph | 16223058 | 0.9295 | 0.7713 | 0.7766 | 0.1582 | 0.1529 | 0.830 | 0.835 |
| hotpotqa_clean | baseline_a_simple | 9118344 | 0.9295 | 0.7673 | 0.7723 | 0.1623 | 0.1572 | 0.825 | 0.831 |
| hotpotqa_clean | symbolic_b | 14971304 | 0.9295 | 0.7769 | 0.7793 | 0.1527 | 0.1502 | 0.836 | 0.838 |
| hotpotqa_clean | knn_only | 2333304 | 0.9295 | 0.7354 | 0.7333 | 0.1941 | 0.1962 | 0.791 | 0.789 |
| hotpotqa_clean | full_union_c | 16880560 | 0.9295 | 0.7646 | 0.7705 | 0.1649 | 0.1590 | 0.823 | 0.829 |
| squad_clean | sealed_a_multigraph | 2857316 | 0.9949 | 0.8923 | 0.8933 | 0.1025 | 0.1015 | 0.897 | 0.898 |
| squad_clean | baseline_a_simple | 1445712 | 0.9949 | 0.8929 | 0.8936 | 0.1020 | 0.1013 | 0.897 | 0.898 |
| squad_clean | symbolic_b | 1642136 | 0.9949 | 0.8931 | 0.8935 | 0.1018 | 0.1014 | 0.898 | 0.898 |
| squad_clean | knn_only | 56552 | 0.9949 | 0.8933 | 0.8933 | 0.1016 | 0.1015 | 0.898 | 0.898 |
| squad_clean | full_union_c | 1682702 | 0.9949 | 0.8932 | 0.8948 | 0.1017 | 0.1001 | 0.898 | 0.899 |
| metaqa | sealed_a_multigraph | 585728 | 0.3262 | 0.3011 | 0.3013 | 0.0251 | 0.0249 | 0.923 | 0.924 |
| metaqa | baseline_a_simple | 329374 | 0.3262 | 0.2994 | 0.3012 | 0.0268 | 0.0250 | 0.918 | 0.923 |
| metaqa | symbolic_b | 683970 | 0.3262 | 0.3044 | 0.3030 | 0.0218 | 0.0232 | 0.933 | 0.929 |
| metaqa | knn_only | 110414 | 0.3262 | 0.2368 | 0.2368 | 0.0894 | 0.0894 | 0.726 | 0.726 |
| metaqa | full_union_c | 775576 | 0.3262 | 0.3002 | 0.3027 | 0.0261 | 0.0235 | 0.920 | 0.928 |

## Every reported cut-off

| Dataset | Family | Ceil@1 | QLS R@1 | GNN R@1 | Ceil@5 | QLS R@5 | GNN R@5 | Ceil@20 | QLS R@20 | GNN R@20 | QLS MRR | GNN MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | sealed_a_multigraph | 0.4472 | 0.2923 | 0.2855 | 0.7967 | 0.6840 | 0.6985 | 0.7967 | 0.7775 | 0.7785 | 0.7966 | 0.7861 |
| 2wiki_clean | baseline_a_simple | 0.4472 | 0.2865 | 0.2877 | 0.7967 | 0.6781 | 0.7005 | 0.7967 | 0.7769 | 0.7779 | 0.7897 | 0.7887 |
| 2wiki_clean | symbolic_b | 0.4472 | 0.2884 | 0.2860 | 0.7967 | 0.6813 | 0.6964 | 0.7967 | 0.7806 | 0.7824 | 0.7894 | 0.7883 |
| 2wiki_clean | knn_only | 0.4472 | 0.2914 | 0.2909 | 0.7967 | 0.6611 | 0.6559 | 0.7967 | 0.7350 | 0.7328 | 0.7982 | 0.7976 |
| 2wiki_clean | full_union_c | 0.4472 | 0.2863 | 0.2829 | 0.7967 | 0.6716 | 0.6944 | 0.7967 | 0.7788 | 0.7821 | 0.7871 | 0.7843 |
| musique_clean | sealed_a_multigraph | 0.4487 | 0.3577 | 0.3654 | 0.9405 | 0.8028 | 0.8124 | 0.9405 | 0.9007 | 0.9023 | 0.8780 | 0.8909 |
| musique_clean | baseline_a_simple | 0.4487 | 0.3563 | 0.3658 | 0.9405 | 0.8018 | 0.8125 | 0.9405 | 0.9002 | 0.9013 | 0.8775 | 0.8919 |
| musique_clean | symbolic_b | 0.4487 | 0.3603 | 0.3693 | 0.9405 | 0.8098 | 0.8153 | 0.9405 | 0.9047 | 0.9038 | 0.8830 | 0.8958 |
| musique_clean | knn_only | 0.4487 | 0.3599 | 0.3664 | 0.9405 | 0.8023 | 0.8100 | 0.9405 | 0.8981 | 0.9002 | 0.8816 | 0.8934 |
| musique_clean | full_union_c | 0.4487 | 0.3576 | 0.3673 | 0.9405 | 0.8055 | 0.8173 | 0.9405 | 0.9036 | 0.9048 | 0.8787 | 0.8943 |
| webqsp | sealed_a_multigraph | 0.3894 | 0.1738 | 0.1749 | 0.4501 | 0.3337 | 0.3309 | 0.4762 | 0.4139 | 0.4231 | 0.4180 | 0.4255 |
| webqsp | baseline_a_simple | 0.3894 | 0.1810 | 0.1740 | 0.4501 | 0.3297 | 0.3276 | 0.4762 | 0.4158 | 0.4224 | 0.4177 | 0.4190 |
| webqsp | symbolic_b | 0.3894 | 0.1565 | 0.1788 | 0.4501 | 0.3239 | 0.3392 | 0.4762 | 0.4028 | 0.4301 | 0.3957 | 0.4293 |
| webqsp | knn_only | 0.3894 | 0.1394 | 0.1551 | 0.4501 | 0.2973 | 0.2855 | 0.4762 | 0.4017 | 0.3973 | 0.3625 | 0.3704 |
| webqsp | full_union_c | 0.3894 | 0.1667 | 0.1781 | 0.4501 | 0.3305 | 0.3264 | 0.4762 | 0.4078 | 0.4259 | 0.3984 | 0.4277 |
| hotpotqa_clean | sealed_a_multigraph | 0.4989 | 0.3649 | 0.3584 | 0.9295 | 0.7713 | 0.7766 | 0.9295 | 0.8856 | 0.8895 | 0.8240 | 0.8165 |
| hotpotqa_clean | baseline_a_simple | 0.4989 | 0.3641 | 0.3583 | 0.9295 | 0.7673 | 0.7723 | 0.9295 | 0.8836 | 0.8880 | 0.8233 | 0.8167 |
| hotpotqa_clean | symbolic_b | 0.4989 | 0.3691 | 0.3605 | 0.9295 | 0.7769 | 0.7793 | 0.9295 | 0.8872 | 0.8893 | 0.8297 | 0.8202 |
| hotpotqa_clean | knn_only | 0.4989 | 0.3544 | 0.3472 | 0.9295 | 0.7354 | 0.7333 | 0.9295 | 0.8440 | 0.8409 | 0.8117 | 0.8030 |
| hotpotqa_clean | full_union_c | 0.4989 | 0.3633 | 0.3573 | 0.9295 | 0.7646 | 0.7705 | 0.9295 | 0.8840 | 0.8877 | 0.8231 | 0.8163 |
| squad_clean | sealed_a_multigraph | 0.9949 | 0.5435 | 0.5347 | 0.9949 | 0.8923 | 0.8933 | 0.9949 | 0.9578 | 0.9555 | 0.6867 | 0.6815 |
| squad_clean | baseline_a_simple | 0.9949 | 0.5490 | 0.5377 | 0.9949 | 0.8929 | 0.8936 | 0.9949 | 0.9574 | 0.9551 | 0.6907 | 0.6829 |
| squad_clean | symbolic_b | 0.9949 | 0.5456 | 0.5394 | 0.9949 | 0.8931 | 0.8935 | 0.9949 | 0.9574 | 0.9546 | 0.6881 | 0.6840 |
| squad_clean | knn_only | 0.9949 | 0.5521 | 0.5326 | 0.9949 | 0.8933 | 0.8933 | 0.9949 | 0.9576 | 0.9554 | 0.6927 | 0.6801 |
| squad_clean | full_union_c | 0.9949 | 0.5546 | 0.5395 | 0.9949 | 0.8932 | 0.8948 | 0.9949 | 0.9585 | 0.9558 | 0.6937 | 0.6853 |
| metaqa | sealed_a_multigraph | 0.2638 | 0.2015 | 0.2096 | 0.3262 | 0.3011 | 0.3013 | 0.3349 | 0.3274 | 0.3263 | 0.4073 | 0.4104 |
| metaqa | baseline_a_simple | 0.2638 | 0.1970 | 0.2087 | 0.3262 | 0.2994 | 0.3012 | 0.3349 | 0.3272 | 0.3262 | 0.4016 | 0.4093 |
| metaqa | symbolic_b | 0.2638 | 0.1999 | 0.2090 | 0.3262 | 0.3044 | 0.3030 | 0.3349 | 0.3288 | 0.3270 | 0.4132 | 0.4153 |
| metaqa | knn_only | 0.2638 | 0.1352 | 0.1360 | 0.3262 | 0.2368 | 0.2368 | 0.3349 | 0.2948 | 0.2944 | 0.3142 | 0.3152 |
| metaqa | full_union_c | 0.2638 | 0.1930 | 0.2087 | 0.3262 | 0.3002 | 0.3027 | 0.3349 | 0.3282 | 0.3270 | 0.4036 | 0.4145 |

## Relational topology against embedding similarity

`symbolic_b` is structural and NER edges: genuine relational topology. `knn_only` is the embedding-similarity edges alone. A positive difference means the family built from relations outperforms the family built from embedding neighborhoods on the same candidates and the same ceiling. A difference near zero means the message passing was exploiting embedding similarity reintroduced as edges rather than relational structure, and must be reported as such.

| Dataset | Model | Relational R@5 | Similarity R@5 | Relational - Similarity | Relational edges | Similarity edges |
|---|---|---:|---:|---:|---:|---:|
| 2wiki_clean | QLS-MLP | 0.6813 | 0.6611 | +0.0202 | 1240748 | 269474 |
| 2wiki_clean | Seed-aware GNN | 0.6964 | 0.6559 | +0.0405 | 1240748 | 269474 |
| musique_clean | QLS-MLP | 0.8098 | 0.8023 | +0.0075 | 290086 | 54812 |
| musique_clean | Seed-aware GNN | 0.8153 | 0.8100 | +0.0053 | 290086 | 54812 |
| webqsp | QLS-MLP | 0.3239 | 0.2973 | +0.0266 | 5228122 | 3345462 |
| webqsp | Seed-aware GNN | 0.3392 | 0.2855 | +0.0537 | 5228122 | 3345462 |
| hotpotqa_clean | QLS-MLP | 0.7769 | 0.7354 | +0.0415 | 14971304 | 2333304 |
| hotpotqa_clean | Seed-aware GNN | 0.7793 | 0.7333 | +0.0460 | 14971304 | 2333304 |
| squad_clean | QLS-MLP | 0.8931 | 0.8933 | -0.0002 | 1642136 | 56552 |
| squad_clean | Seed-aware GNN | 0.8935 | 0.8933 | +0.0002 | 1642136 | 56552 |
| metaqa | QLS-MLP | 0.3044 | 0.2368 | +0.0676 | 683970 | 110414 |
| metaqa | Seed-aware GNN | 0.3030 | 0.2368 | +0.0662 | 683970 | 110414 |

The ceiling is a diagnostic and is never given to a model. It is identical across the families of a dataset by construction, so it explains none of the differences in this report.
