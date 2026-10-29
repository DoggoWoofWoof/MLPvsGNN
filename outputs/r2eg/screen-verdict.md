# R2EG, R2e (the GNN half) under WQ on Kaggle's T4: zg6e **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zg6e's re-call against WQK's seed-0 zg6 under WQK's zspk seed null, its seed-0 R@5 replaced by zg6's, has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not BASE_READY, a fit or compare is missing or not R2EG's, R2EG's fits and WQK's seed-0 zg6 and zspk fits are not on one device, a fit or read names a gate that is not a filed PASS gate of an R2EG session, a fit's R2 constants, parameters or EM term are not as declared, zg6e's message is not as declared or its read holds no link statistics on the six carves, WQK's floors are not their compares', or a re-call does not hold the twelve reads (docs/R2EG_R2E_GNN_UNDER_WQ.md section 5).

- **zg6e** (zg6 + R2's linked-bridge scores in its hidden update) against WQK's seed-0 zg6 on the T4, re-called under WQK's zspk seed null with zg6's seed-0 R@5: 1 GAIN (WQ-hotpotqa hotpotqa), 2 LOSS (WQ-musique metaqa, WQ-hotpotqa metaqa).
- **Next form** (fixed in section 7 before any number): R2's GNN line closes: zg6m and zg6e are NOT_ADOPTED, and a verdict closes only its form (R2WS's confirmation of zrckpb at the null seeds is the MLP line's, in its own file). The next declared forms are those waiting on PKO's pools (U2n's zrckpt on U2s's pick, G6RW's zg6r on PKO's pools) and the typed channels of docs/PROGRAM_2026_10_24.md section 4.3 item 5 (G6 with EM-typed messages over EL, R and kNN), each its own declared file.
- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.
- Gates: f143a359e6a5.
- Naming: zg6e is a GNN arm, called "zg6 + R2's linked-bridge scores in its hidden update"; the user's open ruling on zrckpb (docs/R2_BRIDGE_EM.md section 2: whether reading the five bridges' vectors keeps an arm inside the MLP track) is about the MLP track and does not apply to it.
- Numbers added to zg6: 59648 (59456 in R2's bridge logits, 192 in the three directions).

Beside, deciding nothing.

- zg6e against WQK's seed-0 zspk (G6's call, under WQK's zspk null): 4 GAIN (WQ-musique hotpotqa, WQ-hotpotqa metaqa, WQ-hotpotqa musique, WQ-hotpotqa webqsp), 1 LOSS (WQ-hotpotqa hotpotqa).

R@5 per read on the T4, and zg6e's link statistics on the read (share of rows with a linked bridge, linked bridges per row, share of questions with a linked row):

| read | rrf | zspk | zg6 | zg6e | floor | call | vs zspk | rows linked | flags per row | questions linked |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| WQ-musique metaqa (zero-shot) | 0.0047 | 0.3984 | 0.4393 | 0.3949 | 0.0439 | LOSS | WITHIN | 0.0127 | 0.0152 | 1.0000 |
| WQ-musique squad (in-domain) | 0.9054 | 0.9110 | 0.9097 | 0.9101 | 0.0075 | WITHIN | WITHIN | 0.4251 | 1.0759 | 0.9993 |
| WQ-musique musique (zero-shot) | 0.4734 | 0.4590 | 0.5018 | 0.4838 | 0.0545 | WITHIN | WITHIN | 0.0957 | 0.1382 | 1.0000 |
| WQ-musique hotpotqa (in-domain) | 0.6846 | 0.9058 | 0.9109 | 0.9145 | 0.0075 | WITHIN | GAIN | 0.3862 | 0.5574 | 1.0000 |
| WQ-musique 2wiki (in-domain) | 0.6079 | 0.9345 | 0.9390 | 0.9380 | 0.0075 | WITHIN | WITHIN | 0.3536 | 0.4578 | 1.0000 |
| WQ-musique webqsp (in-domain) | 0.0535 | 0.6334 | 0.6423 | 0.6439 | 0.0174 | WITHIN | WITHIN | 0.0800 | 0.0893 | 1.0000 |
| WQ-hotpotqa metaqa (zero-shot) | 0.0047 | 0.3384 | 0.4167 | 0.3847 | 0.0308 | LOSS | GAIN | 0.0127 | 0.0152 | 1.0000 |
| WQ-hotpotqa squad (in-domain) | 0.9054 | 0.9114 | 0.9114 | 0.9108 | 0.0075 | WITHIN | WITHIN | 0.4251 | 1.0759 | 0.9993 |
| WQ-hotpotqa musique (in-domain) | 0.4734 | 0.5794 | 0.5961 | 0.5927 | 0.0090 | WITHIN | GAIN | 0.0957 | 0.1382 | 1.0000 |
| WQ-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8496 | 0.8183 | 0.8280 | 0.0075 | GAIN | LOSS | 0.3862 | 0.5574 | 1.0000 |
| WQ-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9330 | 0.9414 | 0.9399 | 0.0075 | WITHIN | WITHIN | 0.3536 | 0.4578 | 1.0000 |
| WQ-hotpotqa webqsp (in-domain) | 0.0535 | 0.6153 | 0.6421 | 0.6343 | 0.0089 | WITHIN | GAIN | 0.0800 | 0.0893 | 1.0000 |

R2's EM term on each fit (the share of training questions it reached, L_br at the first and the last epoch):

| fit | questions | with the term | share | L_br first | L_br last |
|---|---:|---:|---:|---:|---:|
| scr-r2eg-zg6 | 151328 | 127032 | 0.8394 | 2.1623 | 1.4434 |
| scr-r2eg-zg6-hp | 140696 | 111368 | 0.7916 | 2.4751 | 1.5263 |

Seconds and memory: zg6e, and in brackets WQK's zg6 and zspk of the same split, all on the T4:

| split | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---:|---:|---|---:|
| WQ-musique | 1515 (1364, 933) | 763 (706, 358) | 2.2984 / 5.2383 (2.2806 / 5.2366, 1.8704 / 4.4436) | 4.5200 (4.5400, 4.5100) |
| WQ-hotpotqa | 3564 (3356, 1056) | 742 (715, 348) | 9.1571 / 9.1571 (9.1280 / 9.1280, 3.9910 / 4.4449) | 4.5600 (4.5600, 4.5400) |
