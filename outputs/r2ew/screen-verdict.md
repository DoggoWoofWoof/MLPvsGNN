# R2EW, R2e (the MLP half) under WQ on Kaggle's T4: zrckpbe **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zrckpbe's re-call against WQK's seed-0 zrckp under WQK's zrckp seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not BASE_READY, a fit or compare is missing or not R2EW's (for R2GW's zrckpb, not R2GW's), R2EW's fits, R2GW's zrckpb fits and WQK's seed-0 zrckp fits are not on one device, a fit or read names a gate that is not a filed PASS gate of its session, a fit's R2 constants or EM term are not as declared, zrckpbe's flag is not as declared or its read holds no link statistics on the six carves, WQK's floors are not their compares', or a re-call does not hold the twelve reads (docs/R2EW_R2E_UNDER_WQ.md section 5).

- **zrckpbe** (zrckp + R2's bridge head on linked bridges: R2's head on the bridges that zlink's edges link to the row) against WQK's seed-0 zrckp on the T4, re-called under WQK's zrckp seed null: 0 GAIN (none), 1 LOSS (WQ-hotpotqa musique).
- **Next form** (fixed in section 7 before any number): R2e's GNN half under WQ on the T4, its own declared file: bridge scores only for linked bridges inside zg6's messages (R2's text: R2e, the edge form: bridge scores only for bridges linked to the row by zlink's edges (B1d: the edge is the bridge), as fixed per-row link flags for the MLP and inside zg6's messages for the GNN; one declared file); the MLP half closes with this verdict.
- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.
- Gates: 7196c841e9c8.
- Naming: zrckpbe is called "zrckp + R2's bridge head on linked bridges" until the user rules whether reading the five bridges' vectors keeps it inside the MLP track (R2, section 2); the edges answer R2's flag 2 (graph-free) only, and no claim sets zrckpbe against the GNN as the MLP.

Beside, deciding nothing.

- zrckpbe against R2GW's zrckpb (the null's seed-0 R@5 is R2GW's zrckpb read): 0 GAIN (none), 0 LOSS (none).

R@5 per read on the T4, and zrckpbe's link statistics on the read (share of rows with a linked bridge, linked bridges per row, share of questions with a linked row):

| read | rrf | zspk | zrckp | zrckpb | zrckpbe | floor | call | vs zrckpb | rows linked | flags per row | questions linked |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| WQ-musique metaqa (zero-shot) | 0.0047 | 0.3984 | 0.4393 | 0.4578 | 0.4488 | 0.0621 | WITHIN | WITHIN | 0.0127 | 0.0152 | 1.0000 |
| WQ-musique squad (in-domain) | 0.9054 | 0.9110 | 0.9106 | 0.9080 | 0.9080 | 0.0075 | WITHIN | WITHIN | 0.4251 | 1.0759 | 0.9993 |
| WQ-musique musique (zero-shot) | 0.4734 | 0.4590 | 0.4972 | 0.4999 | 0.4911 | 0.0566 | WITHIN | WITHIN | 0.0957 | 0.1382 | 1.0000 |
| WQ-musique hotpotqa (in-domain) | 0.6846 | 0.9058 | 0.9051 | 0.8994 | 0.9034 | 0.0087 | WITHIN | WITHIN | 0.3862 | 0.5574 | 1.0000 |
| WQ-musique 2wiki (in-domain) | 0.6079 | 0.9345 | 0.9315 | 0.9274 | 0.9322 | 0.0093 | WITHIN | WITHIN | 0.3536 | 0.4578 | 1.0000 |
| WQ-musique webqsp (in-domain) | 0.0535 | 0.6334 | 0.6433 | 0.6577 | 0.6506 | 0.0075 | WITHIN | WITHIN | 0.0800 | 0.0893 | 1.0000 |
| WQ-hotpotqa metaqa (zero-shot) | 0.0047 | 0.3384 | 0.3927 | 0.4459 | 0.4403 | 0.0489 | WITHIN | WITHIN | 0.0127 | 0.0152 | 1.0000 |
| WQ-hotpotqa squad (in-domain) | 0.9054 | 0.9114 | 0.9117 | 0.9090 | 0.9105 | 0.0075 | WITHIN | WITHIN | 0.4251 | 1.0759 | 0.9993 |
| WQ-hotpotqa musique (in-domain) | 0.4734 | 0.5794 | 0.5743 | 0.5689 | 0.5639 | 0.0075 | LOSS | WITHIN | 0.0957 | 0.1382 | 1.0000 |
| WQ-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8496 | 0.8525 | 0.8498 | 0.8531 | 0.0089 | WITHIN | WITHIN | 0.3862 | 0.5574 | 1.0000 |
| WQ-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9330 | 0.9299 | 0.9304 | 0.9307 | 0.0075 | WITHIN | WITHIN | 0.3536 | 0.4578 | 1.0000 |
| WQ-hotpotqa webqsp (in-domain) | 0.0535 | 0.6153 | 0.6382 | 0.6488 | 0.6425 | 0.0075 | WITHIN | WITHIN | 0.0800 | 0.0893 | 1.0000 |

R2's EM term on each fit (the share of training questions it reached, L_br at the first and the last epoch):

| fit | questions | with the term | share | L_br first | L_br last |
|---|---:|---:|---:|---:|---:|
| scr-r2ew-zrckpbe | 151328 | 127032 | 0.8394 | 2.1619 | 1.3961 |
| scr-r2ew-zrckpbe-hp | 140696 | 111368 | 0.7916 | 2.4757 | 1.5051 |

Seconds and memory: zrckpbe, and in brackets R2GW's zrckpb and WQK's seed-0 zrckp of the same split, all on the T4:

| split | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---:|---:|---|---:|
| WQ-musique | 1111 (991, 831) | 341 (300, 297) | 1.8704 / 4.4336 (1.8704 / 4.4332, 1.8704 / 4.4311) | 4.5100 (4.5000, 4.4800) |
| WQ-hotpotqa | 1267 (985, 837) | 322 (299, 283) | 3.9910 / 4.4331 (3.9910 / 4.4328, 3.9910 / 4.4306) | 4.6100 (4.5500, 4.6200) |
