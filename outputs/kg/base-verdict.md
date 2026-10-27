# KG's base set on Kaggle's T4: **BASE_READY**

Rule: BASE_READY when the fourteen fits are trained and read under KG's stamp on one device, a T4 on KG's image, and the ten compares (the null's eight, zg6 against zspk on each split) are filed under KG's stamp from those fits; INCOMPLETE otherwise, naming what is missing.

- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.

Beside, deciding nothing.

R@5 of each fit's p@swa on the six s1eval carves: the T4 (KG) and, in brackets, U2d's fit of the same arm, seed and split on the L40S:

| fit | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---:|---:|---:|---:|---:|---:|
| scr-kg-base-zrckp | 0.7995 (0.8008) | 0.9116 (0.9121) | 0.5129 (0.5153) | 0.9041 (0.9055) | 0.9319 (0.9345) | 0.3570 (0.3561) |
| scr-kg-base-zrckp-hp | 0.8038 (0.8033) | 0.9117 (0.9124) | 0.5743 (0.5697) | 0.8504 (0.8538) | 0.9354 (0.9303) | 0.3289 (0.3370) |
| scr-kg-base-zrckp-s1 | 0.8036 (0.8026) | 0.9111 (0.9116) | 0.5114 (0.5057) | 0.9072 (0.9049) | 0.9375 (0.9325) | 0.3522 (0.3596) |
| scr-kg-base-zrckp-s1-hp | 0.8013 (0.8024) | 0.9108 (0.9123) | 0.5828 (0.5781) | 0.8562 (0.8502) | 0.9352 (0.9338) | 0.3415 (0.3212) |
| scr-kg-base-zrckp-s2 | 0.8038 (0.8043) | 0.9106 (0.9111) | 0.5117 (0.5157) | 0.9055 (0.9066) | 0.9381 (0.9362) | 0.3448 (0.3686) |
| scr-kg-base-zrckp-s2-hp | 0.8023 (0.8023) | 0.9127 (0.9122) | 0.5713 (0.5710) | 0.8516 (0.8492) | 0.9354 (0.9324) | 0.3399 (0.3534) |
| scr-kg-base-zspk | 0.8013 (0.8021) | 0.9102 (0.9118) | 0.5146 (0.5143) | 0.9086 (0.9088) | 0.9364 (0.9346) | 0.3151 (0.3162) |
| scr-kg-base-zspk-hp | 0.8037 (0.8043) | 0.9139 (0.9117) | 0.5754 (0.5800) | 0.8550 (0.8512) | 0.9265 (0.9290) | 0.2640 (0.2773) |
| scr-kg-base-zspk-s1 | 0.8038 (0.8018) | 0.9109 (0.9124) | 0.5080 (0.5203) | 0.9075 (0.9059) | 0.9334 (0.9343) | 0.3160 (0.3091) |
| scr-kg-base-zspk-s1-hp | 0.8010 (0.8005) | 0.9140 (0.9122) | 0.5762 (0.5860) | 0.8503 (0.8494) | 0.9379 (0.9385) | 0.2474 (0.2498) |
| scr-kg-base-zspk-s2 | 0.8044 (0.8039) | 0.9114 (0.9110) | 0.5074 (0.5124) | 0.9068 (0.9085) | 0.9353 (0.9358) | 0.3158 (0.3141) |
| scr-kg-base-zspk-s2-hp | 0.8026 (0.8014) | 0.9138 (0.9116) | 0.5828 (0.5845) | 0.8551 (0.8614) | 0.9370 (0.9261) | 0.2681 (0.2852) |
| scr-kg-base-zg6 | 0.8127 (0.8123) | 0.9120 (0.9106) | 0.5138 (0.5207) | 0.9127 (0.9118) | 0.9448 (0.9416) | 0.2398 (0.2368) |
| scr-kg-base-zg6-hp | 0.8115 (0.8112) | 0.9122 (0.9118) | 0.5953 (0.5937) | 0.8391 (0.8309) | 0.9463 (0.9391) | 0.2430 (0.2189) |

Floors, max(0.0075, twice the seed-only spread): KG's on the T4 and, in brackets, U2d's on the L40S:

| arm | split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---:|---:|---:|---:|---:|---:|
| zrckp | L-musique | 0.0084 (0.0075) | 0.0075 (0.0075) | 0.0075 (0.0136) | 0.0075 (0.0075) | 0.0117 (0.0075) | 0.0184 (0.0182) |
| zrckp | L-hotpotqa | 0.0075 (0.0075) | 0.0075 (0.0075) | 0.0127 (0.0120) | 0.0084 (0.0083) | 0.0075 (0.0075) | 0.0237 (0.0322) |
| zspk | L-musique | 0.0075 (0.0075) | 0.0075 (0.0075) | 0.0138 (0.0089) | 0.0075 (0.0075) | 0.0075 (0.0075) | 0.0075 (0.0105) |
| zspk | L-hotpotqa | 0.0075 (0.0075) | 0.0075 (0.0075) | 0.0105 (0.0107) | 0.0075 (0.0145) | 0.0219 (0.0139) | 0.0242 (0.0405) |

G6's call replicated on the T4 (zg6 against zspk under KG's zspk null, G6's rule): it would read **NOT_ADOPTED** (4 GAIN: L-musique metaqa, L-musique 2wiki, L-hotpotqa metaqa, L-hotpotqa musique; 2 LOSS: L-musique webqsp, L-hotpotqa hotpotqa). G6 stays as filed on the L40S: NOT_ADOPTED (docs/GL_GPU_LINE_ON_L40S.md, section 7).

Seconds and memory per fit: the T4 (KG) and, in brackets, the L40S (U2d):

| fit | train s | read s | peak GPU GB, train / read | data GB | peak RAM GB |
|---|---:|---:|---|---:|---:|
| scr-kg-base-zrckp | 960 (795) | 291 (311) | 3.6978 / 4.4323 | 3.0142 | 4.6000 |
| scr-kg-base-zrckp-hp | 963 (813) | 279 (306) | 5.5511 / 5.5511 | 4.8693 | 4.5900 |
| scr-kg-base-zrckp-s1 | 959 (644) | 296 (77) | 3.6978 / 4.4309 | 3.0142 | 4.5800 |
| scr-kg-base-zrckp-s1-hp | 1019 (662) | 286 (75) | 5.5511 / 5.5511 | 4.8693 | 4.5800 |
| scr-kg-base-zrckp-s2 | 967 (651) | 287 (76) | 3.6978 / 4.4329 | 3.0142 | 4.6200 |
| scr-kg-base-zrckp-s2-hp | 1018 (655) | 286 (77) | 5.5511 / 5.5511 | 4.8693 | 4.5900 |
| scr-kg-base-zspk | 1086 (892) | 337 (232) | 3.6978 / 4.4434 | 3.0142 | 4.5300 |
| scr-kg-base-zspk-hp | 1202 (938) | 340 (186) | 5.5511 / 5.5511 | 4.8693 | 4.5100 |
| scr-kg-base-zspk-s1 | 1096 (579) | 335 (93) | 3.6978 / 4.4423 | 3.0142 | 4.5300 |
| scr-kg-base-zspk-s1-hp | 1199 (814) | 334 (87) | 5.5511 / 5.5511 | 4.8693 | 4.5100 |
| scr-kg-base-zspk-s2 | 1043 (712) | 330 (84) | 3.6978 / 4.4432 | 3.0142 | 4.5300 |
| scr-kg-base-zspk-s2-hp | 1147 (768) | 330 (87) | 5.5511 / 5.5511 | 4.8693 | 4.5100 |
| scr-kg-base-zg6 | 1942 (1309) | 773 (244) | 4.6552 / 5.2370 | 3.0142 | 4.5700 |
| scr-kg-base-zg6-hp | 3998 (2419) | 687 (194) | 10.2883 / 10.2883 | 4.8693 | 4.5800 |

Next: a later stage may declare the T4 as its device: KG's fits are then its bases and KG's seed nulls its floors (G6b first, G6's next form).
