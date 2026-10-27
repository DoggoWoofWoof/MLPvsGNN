# G6b2's screen verdict: zg6b2 **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zg6b2's re-call against KG's zspk under KG's zspk seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when KG's base verdict is not BASE_READY, G6b's verdict is not NOT_ADOPTED with G6b2 next, a fit or compare is missing or not G6b2's, the fits are not on KG's device, or a re-call does not hold the twelve reads (G6's rule, docs/G6_FREE_GNN.md section 4, on the T4).

- **zg6b2** (zg6's network over the edges and without them, mixed per row by a gate on the question's edge-family shares and the row's own log1p in-degree per family) against KG's zspk on the T4, re-called under KG's zspk seed null: 3 GAIN (L-musique metaqa, L-hotpotqa metaqa, L-hotpotqa musique), 3 LOSS (L-musique musique, L-musique webqsp, L-hotpotqa hotpotqa).
- **Next form** (fixed in section 6 before any number): G6r, relation-aware messages from text: the relation embeddings (outputs/m3b/relations/metaqa_rel_embeddings.npy, webqsp_rel_embeddings.npy) through one shared low-rank map in place of the per-family maps on the KB datasets, the encoder started from zspk's trained encoder, still unbounded with its own readout; one declared file. No verdict here closes the free GNN.
- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.

Beside, deciding nothing.

- zg6b2 against KG's zg6 under zspk's floors: 0 GAIN (none), 2 LOSS (L-musique musique, L-musique webqsp).
- zg6b2 against G6b's zg6b under zspk's floors: 0 GAIN (none), 2 LOSS (L-musique musique, L-musique webqsp).

R@5 per read on the T4; the MLP share = (zrckp - rrf) / (zg6b2 - rrf) where zg6b2's gain over rrf exceeds 0.002:

| read | rrf | zspk | zg6 | zg6b | zg6b2 | zrckp (MLP) | MLP share | floor | call | vs zg6 | vs zg6b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| L-musique metaqa (in-domain) | 0.0047 | 0.8013 | 0.8127 | 0.8126 | 0.8112 | 0.7995 | 0.9855 | 0.0075 | GAIN | WITHIN | WITHIN |
| L-musique squad (in-domain) | 0.9054 | 0.9102 | 0.9120 | 0.9113 | 0.9119 | 0.9116 | 0.9538 | 0.0075 | WITHIN | WITHIN | WITHIN |
| L-musique musique (zero-shot) | 0.4734 | 0.5146 | 0.5138 | 0.5052 | 0.4780 | 0.5129 | 8.5870 | 0.0138 | LOSS | LOSS | LOSS |
| L-musique hotpotqa (in-domain) | 0.6846 | 0.9086 | 0.9127 | 0.9128 | 0.9113 | 0.9041 | 0.9682 | 0.0075 | WITHIN | WITHIN | WITHIN |
| L-musique 2wiki (in-domain) | 0.6079 | 0.9364 | 0.9448 | 0.9424 | 0.9431 | 0.9319 | 0.9666 | 0.0075 | WITHIN | WITHIN | WITHIN |
| L-musique webqsp (zero-shot) | 0.0535 | 0.3151 | 0.2398 | 0.2784 | 0.2222 | 0.3570 | 1.7991 | 0.0075 | LOSS | LOSS | LOSS |
| L-hotpotqa metaqa (in-domain) | 0.0047 | 0.8037 | 0.8115 | 0.8115 | 0.8115 | 0.8038 | 0.9905 | 0.0075 | GAIN | WITHIN | WITHIN |
| L-hotpotqa squad (in-domain) | 0.9054 | 0.9139 | 0.9122 | 0.9130 | 0.9127 | 0.9117 | 0.8630 | 0.0075 | WITHIN | WITHIN | WITHIN |
| L-hotpotqa musique (in-domain) | 0.4734 | 0.5754 | 0.5953 | 0.5934 | 0.6002 | 0.5743 | 0.7957 | 0.0105 | GAIN | WITHIN | WITHIN |
| L-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8550 | 0.8391 | 0.8388 | 0.8367 | 0.8504 | 1.0901 | 0.0075 | LOSS | WITHIN | WITHIN |
| L-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9265 | 0.9463 | 0.9443 | 0.9420 | 0.9354 | 0.9802 | 0.0219 | WITHIN | WITHIN | WITHIN |
| L-hotpotqa webqsp (zero-shot) | 0.0535 | 0.2640 | 0.2430 | 0.2397 | 0.2501 | 0.3289 | 1.4008 | 0.0242 | WITHIN | WITHIN | WITHIN |

The gate of each fit's p@swa: g = sigmoid(a . sh_q + c . log1p(deg_r) + b) on each row. Its value on a row of a question of one family alone, unreached and reached by k edges of that family, and on a row of a question without edges; on each dataset's s1eval questions the question part sigmoid(a . sh + b) (mean, 10th / 50th / 90th percentile), and over the rows the edges reach their share of the carve's rows, the row gate (mean and percentiles) and their mean log1p in-degree per family:

- L-musique: a = [-0.17955, -0.421954, -0.655105], b = -0.496769, c = [0.299489, 0.102914, 0.153157]; one family alone, unreached [0.3371, 0.2852, 0.2401]; reached by 1: [0.3849, 0.3, 0.26]; 3: [0.4351, 0.3152, 0.281]; 10: [0.5105, 0.3381, 0.3133]; 30: [0.5871, 0.3623, 0.3484]; 100: [0.6695, 0.3908, 0.3905]; no edges 0.3783.
  - metaqa: question part 0.3241 (0.3211 / 0.3254 / 0.3266); reached rows 0.9958 of 19742463, row gate 0.4557 (0.3942 / 0.4497 / 0.5253), log1p in-degree [1.6099, 0.0685, 0.4361]; questions without edges 0.0000.
  - squad: question part 0.2890 (0.2612 / 0.2846 / 0.3229); reached rows 0.8164 of 594063, row gate 0.4126 (0.2995 / 0.3692 / 0.6180), log1p in-degree [1.015, 0.7979, 0.9518]; questions without edges 0.0000.
  - musique: question part 0.3220 (0.3153 / 0.3229 / 0.3274); reached rows 0.9757 of 5057436, row gate 0.5528 (0.3950 / 0.5542 / 0.7048), log1p in-degree [2.3772, 1.1416, 0.9132]; questions without edges 0.0000.
  - hotpotqa: question part 0.2972 (0.2873 / 0.2968 / 0.3083); reached rows 0.8533 of 693828, row gate 0.4183 (0.3354 / 0.4090 / 0.5136), log1p in-degree [1.1355, 0.6886, 0.7235]; questions without edges 0.0000.
  - 2wiki: question part 0.2989 (0.2913 / 0.2987 / 0.3071); reached rows 0.8070 of 1328715, row gate 0.4172 (0.3368 / 0.4058 / 0.5157), log1p in-degree [1.1208, 0.8035, 0.6073]; questions without edges 0.0000.
  - webqsp: question part 0.3027 (0.2929 / 0.3009 / 0.3164); reached rows 0.9649 of 3186165, row gate 0.4151 (0.3463 / 0.4054 / 0.4918), log1p in-degree [1.1627, 0.2351, 0.7511]; questions without edges 0.0000.
- L-hotpotqa: a = [-0.118062, -0.473619, -0.871081], b = -0.532776, c = [0.160765, 0.103266, 0.123875]; one family alone, unreached [0.3428, 0.2677, 0.1972]; reached by 1: [0.3683, 0.2819, 0.2112]; 3: [0.3946, 0.2967, 0.2258]; 10: [0.4341, 0.3189, 0.2485]; 30: [0.4753, 0.3426, 0.2732]; 100: [0.5228, 0.3706, 0.3032]; no edges 0.3699.
  - metaqa: question part 0.3224 (0.3177 / 0.3243 / 0.3262); reached rows 0.9958 of 19742463, row gate 0.3963 (0.3548 / 0.3922 / 0.4395), log1p in-degree [1.6099, 0.0685, 0.4361]; questions without edges 0.0000.
  - squad: question part 0.2701 (0.2283 / 0.2635 / 0.3207); reached rows 0.8164 of 594063, row gate 0.3520 (0.2615 / 0.3302 / 0.4937), log1p in-degree [1.015, 0.7979, 0.9518]; questions without edges 0.0000.
  - musique: question part 0.3198 (0.3098 / 0.3211 / 0.3279); reached rows 0.9757 of 5057436, row gate 0.4654 (0.3686 / 0.4624 / 0.5643), log1p in-degree [2.3772, 1.1416, 0.9132]; questions without edges 0.0000.
  - hotpotqa: question part 0.2821 (0.2671 / 0.2816 / 0.2986); reached rows 0.8533 of 693828, row gate 0.3591 (0.3025 / 0.3525 / 0.4251), log1p in-degree [1.1355, 0.6886, 0.7235]; questions without edges 0.0000.
  - 2wiki: question part 0.2851 (0.2739 / 0.2849 / 0.2970); reached rows 0.8070 of 1328715, row gate 0.3604 (0.3066 / 0.3522 / 0.4270), log1p in-degree [1.1208, 0.8035, 0.6073]; questions without edges 0.0000.
  - webqsp: question part 0.2895 (0.2746 / 0.2866 / 0.3105); reached rows 0.9649 of 3186165, row gate 0.3567 (0.3114 / 0.3508 / 0.4083), log1p in-degree [1.1627, 0.2351, 0.7511]; questions without edges 0.0000.

Seconds and memory: zg6b2, and in brackets G6b's zg6b and KG's zg6 of the same split, all on the T4:

| split | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---:|---:|---|---:|
| L-musique | 2236 (2239, 1942) | 818 (875, 773) | 4.8919 / 5.2354 (4.8919 / 5.2354, 4.6552 / 5.2370) | 6.0400 (4.5800, 4.5700) |
| L-hotpotqa | 4321 (4347, 3998) | 800 (786, 687) | 10.2888 / 10.2888 (10.2888 / 10.2888, 10.2883 / 10.2883) | 6.1100 (4.5500, 4.5800) |
