# R2WS, zrckpb under WQ at seeds 1 and 2 on Kaggle's T4: **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when, over the seeds fitted (1 and 2), no read of zrckpb's re-call against WQK's zrckp of the same seed under WQK's zrckp seed null is a LOSS at any seed and at least one read is a GAIN at every seed (the same read); NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not BASE_READY, a fit or compare is missing, not R2WS's or of another seed or split than its name, R2WS's fits and the WQK zrckp fits they are compared with are not on one device, a fit or read names a gate that is not a filed PASS gate of an R2WS session, a fit's R2 constants or EM term are not as declared or its head names a gate, a compare's zrckp column is not WQK's read of that seed, WQK's floors are not their compares', or a re-call does not hold the twelve reads (docs/R2WS_R2_SEEDS_UNDER_WQ.md section 4).

- **zrckpb** (zrckp + R2's bridge head) against WQK's zrckp of the same seed on the T4, re-called under WQK's zrckp seed null. GAIN by seed: seed 1: WQ-musique webqsp; seed 2: WQ-musique metaqa, WQ-hotpotqa webqsp. LOSS by seed: seed 1: WQ-hotpotqa musique, WQ-hotpotqa hotpotqa; seed 2: WQ-hotpotqa musique, WQ-hotpotqa hotpotqa. GAIN at every seed (the same read): none.
- **Next form** (fixed in section 6 before any number): the seed-0 GAINs are called noise (they did not reproduce under this rule); zrckpb is not taken further and the R2 line goes on with R2e (R2EW, docs/R2EW_R2E_UNDER_WQ.md) only, whose GNN half stays its own later file.
- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.
- Gates: 8e0f3db91ebe.
- Naming: zrckpb is called "zrckp + R2's bridge head" until the user rules whether reading the five bridges' vectors keeps it inside the MLP track (R2, section 2); no claim sets zrckpb against the GNN as the MLP, and none rests on the head until that ruling.

Beside, deciding nothing.

- Seeds with at least one GAIN: [1, 2] of [1, 2] (the weaker form; it does not adopt). Reads that are a GAIN at one seed only: WQ-hotpotqa webqsp, WQ-musique metaqa, WQ-musique webqsp.
- R2GW's seed-0 call per read (filed re-call outputs/r2gw/r2gw-zrckpb-pair-recall.json), read before this file was written; it is not in the rule.

R@5 per read on the T4 (zrckp and zrckpb of each seed, their difference and the call under the floor):

| read | rrf | zspk | floor | zrckp s1 | zrckpb s1 | diff s1 | call s1 | zrckp s2 | zrckpb s2 | diff s2 | call s2 | seed 0 (R2GW) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| WQ-musique metaqa (zero-shot) | 0.0047 | 0.3984 | 0.0621 | 0.4189 | 0.4455 | +0.0266 | WITHIN | 0.4004 | 0.4695 | +0.0690 | GAIN | WITHIN |
| WQ-musique squad (in-domain) | 0.9054 | 0.9110 | 0.0075 | 0.9091 | 0.9087 | -0.0004 | WITHIN | 0.9104 | 0.9093 | -0.0011 | WITHIN | WITHIN |
| WQ-musique musique (zero-shot) | 0.4734 | 0.4590 | 0.0566 | 0.4637 | 0.5010 | +0.0373 | WITHIN | 0.4753 | 0.4994 | +0.0241 | WITHIN | WITHIN |
| WQ-musique hotpotqa (in-domain) | 0.6846 | 0.9058 | 0.0087 | 0.8998 | 0.9013 | +0.0015 | WITHIN | 0.9020 | 0.9021 | +0.0001 | WITHIN | WITHIN |
| WQ-musique 2wiki (in-domain) | 0.6079 | 0.9345 | 0.0093 | 0.9281 | 0.9317 | +0.0036 | WITHIN | 0.9260 | 0.9295 | +0.0035 | WITHIN | WITHIN |
| WQ-musique webqsp (in-domain) | 0.0535 | 0.6334 | 0.0075 | 0.6421 | 0.6555 | +0.0134 | GAIN | 0.6457 | 0.6523 | +0.0066 | WITHIN | GAIN |
| WQ-hotpotqa metaqa (zero-shot) | 0.0047 | 0.3384 | 0.0489 | 0.4193 | 0.4281 | +0.0087 | WITHIN | 0.4147 | 0.4572 | +0.0426 | WITHIN | GAIN |
| WQ-hotpotqa squad (in-domain) | 0.9054 | 0.9114 | 0.0075 | 0.9115 | 0.9089 | -0.0026 | WITHIN | 0.9116 | 0.9090 | -0.0027 | WITHIN | WITHIN |
| WQ-hotpotqa musique (in-domain) | 0.4734 | 0.5794 | 0.0075 | 0.5768 | 0.5593 | -0.0175 | LOSS | 0.5786 | 0.5571 | -0.0215 | LOSS | WITHIN |
| WQ-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8496 | 0.0089 | 0.8547 | 0.8422 | -0.0125 | LOSS | 0.8583 | 0.8469 | -0.0115 | LOSS | WITHIN |
| WQ-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9330 | 0.0075 | 0.9284 | 0.9306 | +0.0022 | WITHIN | 0.9313 | 0.9302 | -0.0011 | WITHIN | WITHIN |
| WQ-hotpotqa webqsp (in-domain) | 0.0535 | 0.6153 | 0.0075 | 0.6417 | 0.6442 | +0.0024 | WITHIN | 0.6393 | 0.6514 | +0.0122 | GAIN | GAIN |

R2's EM term on each fit (the share of training questions it reached, L_br at the first and the last epoch):

| fit | questions | with the term | share | L_br first | L_br last |
|---|---:|---:|---:|---:|---:|
| scr-r2ws-zrckpb-s1 | 151328 | 150024 | 0.9914 | 4.3965 | 3.7030 |
| scr-r2ws-zrckpb-s1-hp | 140696 | 139584 | 0.9921 | 5.1704 | 4.2143 |
| scr-r2ws-zrckpb-s2 | 151328 | 150024 | 0.9914 | 4.3693 | 3.6863 |
| scr-r2ws-zrckpb-s2-hp | 140696 | 139584 | 0.9921 | 5.1429 | 4.1920 |

Seconds and memory: zrckpb, and in brackets WQK's zrckp of the same seed and split, all on the T4:

| fit | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---:|---:|---|---:|
| scr-r2ws-zrckpb-s1 | 1078 (873) | 306 (288) | 1.8704 / 4.4321 (1.8704 / 4.4300) | 4.6100 (4.5500) |
| scr-r2ws-zrckpb-s1-hp | 1087 (928) | 303 (298) | 3.9910 / 4.4323 (3.9910 / 4.4301) | 4.6600 (4.5200) |
| scr-r2ws-zrckpb-s2 | 1067 (876) | 301 (287) | 1.8704 / 4.4327 (1.8704 / 4.4305) | 4.6100 (4.5600) |
| scr-r2ws-zrckpb-s2-hp | 1047 (885) | 303 (291) | 3.9910 / 4.4328 (3.9910 / 4.4306) | 4.5700 (4.6000) |
