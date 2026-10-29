# R2GW, R2g under WQ on Kaggle's T4: zrckpbg **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zrckpbg's re-call against WQK's seed-0 zrckp under WQK's zrckp seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not BASE_READY, a fit or compare is missing or not R2GW's, R2GW's fits and WQK's seed-0 zrckp fits are not on one device, a fit or read names a gate that is not a filed PASS gate of an R2GW session, a fit's R2 constants or EM term are not as declared, zrckpbg's gate is not as declared or its read holds no gate per question, WQK's floors are not their compares', or a re-call does not hold the twelve reads (docs/R2GW_R2G_UNDER_WQ.md section 5).

- **zrckpbg** (zrckp + R2's bridge head, the head multiplied per question by 2 sigmoid(a . c_q + b)) against WQK's seed-0 zrckp on the T4, re-called under WQK's zrckp seed null: 1 GAIN (WQ-hotpotqa webqsp), 1 LOSS (WQ-hotpotqa musique).
- **Next form** (fixed in section 7 before any number): R2e under WQ on the T4, its own declared file (R2's next form after R2g: R2e, the edge form: bridge scores only for bridges linked to the row by zlink's edges (B1d: the edge is the bridge), as fixed per-row link flags for the MLP and inside zg6's messages for the GNN; one declared file).
- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.
- Gates: 4b6f360be563, 7576c1848176.
- Naming: zrckpb is called "zrckp + R2's bridge head" until the user rules whether reading the five bridges' vectors keeps it inside the MLP track (R2, section 2); zrckpbg likewise, and no claim sets either against the GNN as the MLP.

Beside, deciding nothing.

- zrckpb against WQK's seed-0 zrckp under the same null: 3 GAIN (WQ-musique webqsp, WQ-hotpotqa metaqa, WQ-hotpotqa webqsp), 0 LOSS (none).
- zrckpbg against zrckpb (the null's seed-0 R@5 is R2GW's zrckpb read): 0 GAIN (none), 1 LOSS (WQ-hotpotqa musique).

R@5 per read on the T4, and zrckpbg's gate per question on the read (mean, share of questions below 1):

| read | rrf | zspk | zrckp | zrckpb | zrckpbg | floor | call | zrckpb vs zrckp | zrckpbg vs zrckpb | gate mean | share below 1 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| WQ-musique metaqa (zero-shot) | 0.0047 | 0.3984 | 0.4393 | 0.4578 | 0.4697 | 0.0621 | WITHIN | WITHIN | WITHIN | 0.7140 | 0.8958 |
| WQ-musique squad (in-domain) | 0.9054 | 0.9110 | 0.9106 | 0.9080 | 0.9071 | 0.0075 | WITHIN | WITHIN | WITHIN | 1.0394 | 0.4439 |
| WQ-musique musique (zero-shot) | 0.4734 | 0.4590 | 0.4972 | 0.4999 | 0.4990 | 0.0566 | WITHIN | WITHIN | WITHIN | 1.0800 | 0.3835 |
| WQ-musique hotpotqa (in-domain) | 0.6846 | 0.9058 | 0.9051 | 0.8994 | 0.9002 | 0.0087 | WITHIN | WITHIN | WITHIN | 0.9386 | 0.5928 |
| WQ-musique 2wiki (in-domain) | 0.6079 | 0.9345 | 0.9315 | 0.9274 | 0.9308 | 0.0093 | WITHIN | WITHIN | WITHIN | 0.8598 | 0.6619 |
| WQ-musique webqsp (in-domain) | 0.0535 | 0.6334 | 0.6433 | 0.6577 | 0.6545 | 0.0075 | WITHIN | GAIN | WITHIN | 1.3033 | 0.1224 |
| WQ-hotpotqa metaqa (zero-shot) | 0.0047 | 0.3384 | 0.3927 | 0.4459 | 0.4131 | 0.0489 | WITHIN | GAIN | WITHIN | 0.7123 | 0.8746 |
| WQ-hotpotqa squad (in-domain) | 0.9054 | 0.9114 | 0.9117 | 0.9090 | 0.9069 | 0.0075 | WITHIN | WITHIN | WITHIN | 1.2226 | 0.1556 |
| WQ-hotpotqa musique (in-domain) | 0.4734 | 0.5794 | 0.5743 | 0.5689 | 0.5613 | 0.0075 | LOSS | WITHIN | LOSS | 1.2954 | 0.1121 |
| WQ-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8496 | 0.8525 | 0.8498 | 0.8456 | 0.0089 | WITHIN | WITHIN | WITHIN | 1.1544 | 0.2941 |
| WQ-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9330 | 0.9299 | 0.9304 | 0.9290 | 0.0075 | WITHIN | WITHIN | WITHIN | 0.9355 | 0.5719 |
| WQ-hotpotqa webqsp (in-domain) | 0.0535 | 0.6153 | 0.6382 | 0.6488 | 0.6492 | 0.0075 | GAIN | GAIN | WITHIN | 1.3561 | 0.1025 |

R2's EM term on each fit (the share of training questions it reached, L_br at the first and the last epoch):

| fit | questions | with the term | share | L_br first | L_br last |
|---|---:|---:|---:|---:|---:|
| scr-r2gw-zrckpbg | 151328 | 150024 | 0.9914 | 4.3904 | 3.7146 |
| scr-r2gw-zrckpbg-hp | 140696 | 139584 | 0.9921 | 5.1570 | 4.2340 |
| scr-r2gw-zrckpb | 151328 | 150024 | 0.9914 | 4.3895 | 3.7112 |
| scr-r2gw-zrckpb-hp | 140696 | 139584 | 0.9921 | 5.1576 | 4.2325 |

Seconds and memory: zrckpbg, and in brackets zrckpb and WQK's seed-0 zrckp of the same split, all on the T4:

| split | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---:|---:|---|---:|
| WQ-musique | 1020 (991, 831) | 303 (300, 297) | 1.8704 / 4.4332 (1.8704 / 4.4332, 1.8704 / 4.4311) | 4.5000 (4.5000, 4.4800) |
| WQ-hotpotqa | 1026 (985, 837) | 302 (299, 283) | 3.9910 / 4.4328 (3.9910 / 4.4328, 3.9910 / 4.4306) | 4.5800 (4.5500, 4.6200) |
