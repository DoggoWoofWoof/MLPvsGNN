# U2n's screen verdict: zg6e **NOT_ADOPTED**, zrckpt **ADOPT_FOR_FULL_RUN**

Rule: per arm: ADOPT_FOR_FULL_RUN when the arm's re-call against its KG base (zg6e: zspk; zrckpt: zrckp) under KG's seed null of that base has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when KG's base verdict is not BASE_READY, a fit or compare of the arm is missing or not U2n's, the arm's fits are not on KG's device, or the re-call does not hold the twelve reads (section 6).

- **zg6e** (zg6 with typed channels in its layers, against KG's zspk) on the T4, re-called under KG's zspk seed null: 3 GAIN (L-musique metaqa, L-hotpotqa metaqa, L-hotpotqa musique), 3 LOSS (L-musique webqsp, L-hotpotqa hotpotqa, L-hotpotqa webqsp).
  - Next form (fixed in section 6 before any number): zg6e under G6b's gate (its edge pass and its no-edge pass mixed per question by the question's edge-family shares); one declared file.
- **zrckpt** (zrckp with the typed columns, against KG's zrckp) on the T4, re-called under KG's zrckp seed null: 1 GAIN (L-musique musique), 0 LOSS (none).
  - Next form (fixed in section 6 before any number): zrckpt fitted again on U2s's carried pools, then its full run on all six splits beside zrckp's, then Claim 1 and Claim 3 measured again; each its own declared file.
- KG's device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.

Beside, deciding nothing.

- zg6e against KG's zg6 under zspk's floors: 0 GAIN (none), 2 LOSS (L-musique webqsp, L-hotpotqa hotpotqa).

R@5 per read on the T4; the MLP shares (zrckp - rrf) / (zg6 - rrf) and (zrckpt - rrf) / (zg6e - rrf) where the GNN's gain over rrf exceeds 0.002:

| read | rrf | zspk | zg6 | zg6e | zrckp | zrckpt | share zrckp/zg6 | share zrckpt/zg6e | zg6e call | vs zg6 | zrckpt call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| L-musique metaqa (in-domain) | 0.0047 | 0.8013 | 0.8127 | 0.8130 | 0.7995 | 0.7993 | 0.9837 | 0.9831 | GAIN | WITHIN | WITHIN |
| L-musique squad (in-domain) | 0.9054 | 0.9102 | 0.9120 | 0.9133 | 0.9116 | 0.9116 | 0.9394 | 0.7848 | WITHIN | WITHIN | WITHIN |
| L-musique musique (zero-shot) | 0.4734 | 0.5146 | 0.5138 | 0.5129 | 0.5129 | 0.5228 | 0.9777 | 1.2506 | WITHIN | WITHIN | GAIN |
| L-musique hotpotqa (in-domain) | 0.6846 | 0.9086 | 0.9127 | 0.9120 | 0.9041 | 0.9066 | 0.9623 | 0.9763 | WITHIN | WITHIN | WITHIN |
| L-musique 2wiki (in-domain) | 0.6079 | 0.9364 | 0.9448 | 0.9438 | 0.9319 | 0.9361 | 0.9617 | 0.9771 | WITHIN | WITHIN | WITHIN |
| L-musique webqsp (zero-shot) | 0.0535 | 0.3151 | 0.2398 | 0.2163 | 0.3570 | 0.3517 | 1.6291 | 1.8317 | LOSS | LOSS | WITHIN |
| L-hotpotqa metaqa (in-domain) | 0.0047 | 0.8037 | 0.8115 | 0.8113 | 0.8038 | 0.8039 | 0.9905 | 0.9908 | GAIN | WITHIN | WITHIN |
| L-hotpotqa squad (in-domain) | 0.9054 | 0.9139 | 0.9122 | 0.9106 | 0.9117 | 0.9128 | 0.9265 | 1.4231 | WITHIN | WITHIN | WITHIN |
| L-hotpotqa musique (in-domain) | 0.4734 | 0.5754 | 0.5953 | 0.5985 | 0.5743 | 0.5762 | 0.8277 | 0.8217 | GAIN | WITHIN | WITHIN |
| L-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8550 | 0.8391 | 0.8238 | 0.8504 | 0.8583 | 1.0731 | 1.2478 | LOSS | LOSS | WITHIN |
| L-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9265 | 0.9463 | 0.9437 | 0.9354 | 0.9349 | 0.9678 | 0.9738 | WITHIN | WITHIN | WITHIN |
| L-hotpotqa webqsp (zero-shot) | 0.0535 | 0.2640 | 0.2430 | 0.2254 | 0.3289 | 0.3181 | 1.4533 | 1.5393 | LOSS | WITHIN | WITHIN |

The types of each fit's p@swa (both kappas; each direction's cosine to its tau0):

- scr-u2n-zg6e: kappa_d 11.0032, kappa_r 11.062; cosine to tau0 min -0.1635, median -0.0210.
  - metaqa (top three of the first four directions): t0: +has imdb votes 0.30, -has genre 0.27, -release year 0.12; t1: -directed by 0.14, +release year 0.08, +in language 0.08; t2: -directed by 0.14, +in language 0.08, +has tags 0.07; t3: -directed by 0.14, +release year 0.08, +in language 0.07.
  - webqsp (top three of the first four directions): t0: -game genre 0.00, -content genre 0.00, -genre content 0.00; t1: -possible cabinet member cabinet position 0.00, -flag referent flag 0.00, -possibly correlated thing possible correlations 0.00; t2: -possible cabinet member cabinet position 0.00, -flag referent flag 0.00, -possibly correlated thing possible correlations 0.00; t3: -possible cabinet member cabinet position 0.00, -flag referent flag 0.00, -possibly correlated thing possible correlations 0.00.
- scr-u2n-zg6e-hp: kappa_d 21.9734, kappa_r 15.1214; cosine to tau0 min -0.3234, median 0.0146.
  - metaqa (top three of the first four directions): t0: -release year 0.09, +has genre 0.09, -written by 0.07; t1: -release year 0.09, +has genre 0.09, -written by 0.07; t2: -release year 0.09, +has genre 0.09, -written by 0.07; t3: +has genre 0.23, +has tags 0.15, -release year 0.13.
  - webqsp (top three of the first four directions): t0: +flag referent flag 0.00, +possibly correlated thing possible correlations 0.00, +possible cabinet member cabinet position 0.00; t1: +flag referent flag 0.00, +possibly correlated thing possible correlations 0.00, +possible cabinet member cabinet position 0.00; t2: +flag referent flag 0.00, +possibly correlated thing possible correlations 0.00, +possible cabinet member cabinet position 0.00; t3: +flag referent flag 0.00, +possibly correlated thing possible correlations 0.00, +force strength in fiction unit if known 0.00.
- scr-u2n-zrckpt: kappa_d 10.0, kappa_r 10.0; cosine to tau0 min 1.0000, median 1.0000.
  - metaqa (top three of the first four directions): t0: +written by 0.24, +directed by 0.14, -has tags 0.14; t1: -has genre 0.18, +release year 0.11, +directed by 0.11; t2: -in language 0.19, +written by 0.11, +release year 0.09; t3: -has imdb votes 0.21, +in language 0.12, +has genre 0.12.
  - webqsp (top three of the first four directions): t0: +quotation author 0.00, +character played by 0.00, +genre artist 0.00; t1: +pest attacked by 0.00, +battle result of conflict 0.00, +ceasefire w 0.00; t2: -capital category capitals of administrative divisions 0.00, -form of government countries 0.00, -country form of government 0.00; t3: +content genre 0.00, +genre content 0.00, +language dialect language 0.00.
- scr-u2n-zrckpt-hp: kappa_d 10.0, kappa_r 10.0; cosine to tau0 min 1.0000, median 1.0000.
  - metaqa (top three of the first four directions): t0: -written by 0.25, +has genre 0.14, -directed by 0.11; t1: -has genre 0.30, +release year 0.10, +has imdb rating 0.07; t2: +release year 0.17, +written by 0.14, -has tags 0.10; t3: -has imdb votes 0.21, +has genre 0.13, +in language 0.10.
  - webqsp (top three of the first four directions): t0: -pest attacked by 0.00, -execution person executed 0.00, -executed person execution 0.00; t1: +ceasefire w 0.00, +pest attacked by 0.00, +conflict result battles with this result 0.00; t2: +settlement founded by 0.01, +religion branched from 0.00, +dated location succession succeeded by 0.00; t3: +content genre 0.00, +genre content 0.00, +content language 0.00.

zg6e's EM term per epoch (share of questions with the term, its mean cross entropy, both kappas):

- scr-u2n-zg6e: ep 0: 1.000, 2.1072, 12.330 / 10.708; ep 1: 1.000, 2.6340, 11.325 / 10.633; ep 2: 1.000, 2.6699, 9.733 / 10.442; ep 3: 1.000, 2.6770, 9.037 / 10.201; ep 4: 1.000, 2.6101, 8.801 / 10.290; ep 5: 1.000, 2.4655, 10.610 / 10.962; ep 6: 1.000, 2.0797, 13.611 / 12.235; ep 7: 1.000, 2.1897, 14.228 / 12.242.
- scr-u2n-zg6e-hp: ep 0: 1.000, 1.7382, 14.499 / 11.266; ep 1: 1.000, 1.5520, 18.688 / 12.411; ep 2: 1.000, 1.6639, 20.026 / 13.254; ep 3: 1.000, 1.6512, 21.429 / 13.860; ep 4: 1.000, 1.6124, 22.349 / 14.629; ep 5: 1.000, 1.7297, 21.755 / 15.475; ep 6: 1.000, 1.5605, 23.981 / 16.374; ep 7: 1.000, 1.7871, 22.301 / 17.136.

Seconds and memory: each U2n fit and, in brackets, its KG base of the same split, both on the T4:

| fit | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---:|---:|---|---:|
| scr-u2n-zg6e | 2872 (1086) | 1180 (337) | 4.1086 / 4.6669 (3.6978 / 4.4434) | 4.6400 (4.5300) |
| scr-u2n-zg6e-hp | 6055 (1202) | 1067 (340) | 8.2068 / 8.2068 (5.5511 / 5.5511) | 4.6200 (4.5100) |
| scr-u2n-zrckpt | 1060 (960) | 323 (291) | 3.7054 / 4.7716 (3.6978 / 4.4323) | 4.5500 (4.6000) |
| scr-u2n-zrckpt-hp | 1280 (963) | 353 (279) | 5.5591 / 5.5591 (5.5511 / 5.5511) | 4.5400 (4.5900) |
