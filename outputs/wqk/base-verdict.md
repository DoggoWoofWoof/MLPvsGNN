# WQK, WQ's base set on Kaggle's T4: **BASE_READY**

Rule: BASE_READY when every gate a fit or read names is a session's PASS gate, filed with that session's overlay, on the reproduced check; the fourteen fits are trained, stamped and read on the six s1eval carves under WQK's stamp on one device, a T4 on KG's image; the ten compares (the null's eight, zg6 against zspk on each split) are WQK's, from those fits on that device; and both floors records are OK and are their compares' floors. INCOMPLETE otherwise, naming what is missing.

- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.
- Gates: 357da77c695d, a38d40778e82.

Beside, deciding nothing (section 7).

R@5 of each fit's p@swa on the six s1eval carves on the T4; in brackets KG's fit of the same arm and seed on the old split with the same wiki side (L-musique for WQ-musique, L-hotpotqa for WQ-hotpotqa), also on the T4; then WQ's host twin, once filed:

| fit | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---:|---:|---:|---:|---:|---:|
| scr-wqk-zrckp-s0 | 0.4393 (0.7995) | 0.9106 (0.9116) | 0.4972 (0.5129) | 0.9051 (0.9041) | 0.9315 (0.9319) | 0.6433 (0.3570) |
| scr-wqk-zrckp-s0-hp | 0.3927 (0.8038) | 0.9117 (0.9117) | 0.5743 (0.5743) | 0.8525 (0.8504) | 0.9299 (0.9354) | 0.6382 (0.3289) |
| scr-wqk-zrckp-s1 | 0.4189 (0.8036) | 0.9091 (0.9111) | 0.4637 (0.5114) | 0.8998 (0.9072) | 0.9281 (0.9375) | 0.6421 (0.3522) |
| scr-wqk-zrckp-s1-hp | 0.4193 (0.8013) | 0.9115 (0.9108) | 0.5768 (0.5828) | 0.8547 (0.8562) | 0.9284 (0.9352) | 0.6417 (0.3415) |
| scr-wqk-zrckp-s2 | 0.4004 (0.8038) | 0.9104 (0.9106) | 0.4753 (0.5117) | 0.9020 (0.9055) | 0.9260 (0.9381) | 0.6457 (0.3448) |
| scr-wqk-zrckp-s2-hp | 0.4147 (0.8023) | 0.9116 (0.9127) | 0.5786 (0.5713) | 0.8583 (0.8516) | 0.9313 (0.9354) | 0.6393 (0.3399) |
| scr-wqk-zspk-s0 | 0.3984 (0.8013) | 0.9110 (0.9102) | 0.4590 (0.5146) | 0.9058 (0.9086) | 0.9345 (0.9364) | 0.6334 (0.3151) |
| scr-wqk-zspk-s0-hp | 0.3384 (0.8037) | 0.9114 (0.9139) | 0.5794 (0.5754) | 0.8496 (0.8550) | 0.9330 (0.9265) | 0.6153 (0.2640) |
| scr-wqk-zspk-s1 | 0.3681 (0.8038) | 0.9104 (0.9109) | 0.4318 (0.5080) | 0.9043 (0.9075) | 0.9337 (0.9334) | 0.6212 (0.3160) |
| scr-wqk-zspk-s1-hp | 0.3599 (0.8010) | 0.9107 (0.9140) | 0.5766 (0.5762) | 0.8484 (0.8503) | 0.9345 (0.9379) | 0.6174 (0.2474) |
| scr-wqk-zspk-s2 | 0.3916 (0.8044) | 0.9103 (0.9114) | 0.4862 (0.5074) | 0.9064 (0.9068) | 0.9333 (0.9353) | 0.6327 (0.3158) |
| scr-wqk-zspk-s2-hp | 0.3417 (0.8026) | 0.9114 (0.9138) | 0.5850 (0.5828) | 0.8460 (0.8551) | 0.9353 (0.9370) | 0.6213 (0.2681) |
| scr-wqk-zg6-s0 | 0.4393 (0.8127) | 0.9097 (0.9120) | 0.5018 (0.5138) | 0.9109 (0.9127) | 0.9390 (0.9448) | 0.6423 (0.2398) |
| scr-wqk-zg6-s0-hp | 0.4167 (0.8115) | 0.9114 (0.9122) | 0.5961 (0.5953) | 0.8183 (0.8391) | 0.9414 (0.9463) | 0.6421 (0.2430) |

Floors, max(0.0075, twice the seed-only spread): WQK's and, in brackets, KG's, both on the T4:

| arm | split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---:|---:|---:|---:|---:|---:|
| zrckp | WQ-musique | 0.0621 (0.0084) | 0.0075 (0.0075) | 0.0566 (0.0075) | 0.0087 (0.0075) | 0.0093 (0.0117) | 0.0075 (0.0184) |
| zrckp | WQ-hotpotqa | 0.0489 (0.0075) | 0.0075 (0.0075) | 0.0075 (0.0127) | 0.0089 (0.0084) | 0.0075 (0.0075) | 0.0075 (0.0237) |
| zspk | WQ-musique | 0.0439 (0.0075) | 0.0075 (0.0075) | 0.0545 (0.0138) | 0.0075 (0.0075) | 0.0075 (0.0075) | 0.0174 (0.0075) |
| zspk | WQ-hotpotqa | 0.0308 (0.0075) | 0.0075 (0.0075) | 0.0090 (0.0105) | 0.0075 (0.0075) | 0.0075 (0.0219) | 0.0089 (0.0242) |

G6's call under WQ on the T4 (zg6 against zspk under WQK's zspk null, G6's rule): it would read **NOT_ADOPTED** (4 GAIN: WQ-hotpotqa metaqa, WQ-hotpotqa musique, WQ-hotpotqa 2wiki, WQ-hotpotqa webqsp; 1 LOSS: WQ-hotpotqa hotpotqa). G6 stays as filed: NOT_ADOPTED (docs/G6_FREE_GNN.md).

Seconds and memory per fit on the T4; in brackets KG's fit beside it:

| fit | train s | read s | peak GPU GB, train / read | data GB | peak RAM GB |
|---|---:|---:|---|---:|---:|
| scr-wqk-zrckp-s0 | 831 (960) | 297 (291) | 1.8704 / 4.4311 | 1.4532 | 4.4800 |
| scr-wqk-zrckp-s0-hp | 837 (963) | 283 (279) | 3.9910 / 4.4306 | 3.3082 | 4.6200 |
| scr-wqk-zrckp-s1 | 873 (959) | 288 (296) | 1.8704 / 4.4300 | 1.4532 | 4.5500 |
| scr-wqk-zrckp-s1-hp | 928 (1019) | 298 (286) | 3.9910 / 4.4301 | 3.3082 | 4.5200 |
| scr-wqk-zrckp-s2 | 876 (967) | 287 (287) | 1.8704 / 4.4305 | 1.4532 | 4.5600 |
| scr-wqk-zrckp-s2-hp | 885 (1018) | 291 (286) | 3.9910 / 4.4306 | 3.3082 | 4.6000 |
| scr-wqk-zspk-s0 | 933 (1086) | 358 (337) | 1.8704 / 4.4436 | 1.4532 | 4.5100 |
| scr-wqk-zspk-s0-hp | 1056 (1202) | 348 (340) | 3.9910 / 4.4449 | 3.3082 | 4.5400 |
| scr-wqk-zspk-s1 | 990 (1096) | 348 (335) | 1.8704 / 4.4425 | 1.4532 | 4.4800 |
| scr-wqk-zspk-s1-hp | 1126 (1199) | 363 (334) | 3.9910 / 4.4422 | 3.3082 | 4.5400 |
| scr-wqk-zspk-s2 | 987 (1043) | 343 (330) | 1.8704 / 4.4435 | 1.4532 | 4.5800 |
| scr-wqk-zspk-s2-hp | 1054 (1147) | 347 (330) | 3.9910 / 4.4449 | 3.3082 | 4.5400 |
| scr-wqk-zg6-s0 | 1364 (1942) | 706 (773) | 2.2806 / 5.2366 | 1.4532 | 4.5400 |
| scr-wqk-zg6-s0-hp | 3356 (3998) | 715 (687) | 9.1280 / 9.1280 | 3.3082 | 4.5600 |

Next: a later WQ stage may declare the T4 as its device: WQK's seed-0 fits are then its bases and WQK's floors its floors (section 9: G6r, R2g and pool attention under WQ, each with its own declared file first).
