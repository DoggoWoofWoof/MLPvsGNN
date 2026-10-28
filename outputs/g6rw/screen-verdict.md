# G6RW, G6r under WQ on Kaggle's T4: zg6r **ADOPT_FOR_FULL_RUN**

Rule: ADOPT_FOR_FULL_RUN when zg6r's re-call against WQK's seed-0 zspk under WQK's zspk seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not BASE_READY, G6b2's verdict is not NOT_ADOPTED with G6r next, a fit or compare is missing or not G6RW's, G6RW's fits and WQK's seed-0 zspk and zg6 fits are not on one device, a fit or read names a gate that is not a filed PASS gate of a G6RW session, a fit was not trained from WQK's zspk encoder and typed.json's typed links, WQK's floors are not their compares', or a re-call does not hold the twelve reads (G6's rule, docs/G6_FREE_GNN.md section 4, under WQ on the T4; docs/G6RW_G6R_UNDER_WQ.md section 5).

- **zg6r** (zg6 with each typed KB edge's message from its relations' shared low-rank map, its encoder started from WQK's seed-0 zspk fit) against WQK's seed-0 zspk on the T4, re-called under WQK's zspk seed null: 5 GAIN (WQ-musique hotpotqa, WQ-hotpotqa metaqa, WQ-hotpotqa musique, WQ-hotpotqa 2wiki, WQ-hotpotqa webqsp), 0 LOSS (none).
- **Next form** (fixed in section 7 before any number): zg6r under WQ fitted again on the pools WQP carried (U2s's pick), then its full run under WQ beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) measured again; each its own declared file.
- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.
- Gates: fd5c2834b389.

Beside, deciding nothing.

- zg6r against WQK's seed-0 zg6 under zspk's floors: 1 GAIN (WQ-hotpotqa hotpotqa), 2 LOSS (WQ-musique musique, WQ-hotpotqa metaqa).

R@5 per read on the T4; the MLP share = (zrckp - rrf) / (zg6r - rrf) where zg6r's gain over rrf exceeds 0.002:

| read | rrf | zspk | zg6 | zg6r | zrckp (MLP) | MLP share | floor | call | vs zg6 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| WQ-musique metaqa (zero-shot) | 0.0047 | 0.3984 | 0.4393 | 0.4105 | 0.4393 | 1.0710 | 0.0439 | WITHIN | WITHIN |
| WQ-musique squad (in-domain) | 0.9054 | 0.9110 | 0.9097 | 0.9105 | 0.9106 | 1.0196 | 0.0075 | WITHIN | WITHIN |
| WQ-musique musique (zero-shot) | 0.4734 | 0.4590 | 0.5018 | 0.4106 | 0.4972 | n/a | 0.0545 | WITHIN | LOSS |
| WQ-musique hotpotqa (in-domain) | 0.6846 | 0.9058 | 0.9109 | 0.9134 | 0.9051 | 0.9637 | 0.0075 | GAIN | WITHIN |
| WQ-musique 2wiki (in-domain) | 0.6079 | 0.9345 | 0.9390 | 0.9369 | 0.9315 | 0.9836 | 0.0075 | WITHIN | WITHIN |
| WQ-musique webqsp (in-domain) | 0.0535 | 0.6334 | 0.6423 | 0.6501 | 0.6433 | 0.9886 | 0.0174 | WITHIN | WITHIN |
| WQ-hotpotqa metaqa (zero-shot) | 0.0047 | 0.3384 | 0.4167 | 0.3834 | 0.3927 | 1.0246 | 0.0308 | GAIN | LOSS |
| WQ-hotpotqa squad (in-domain) | 0.9054 | 0.9114 | 0.9114 | 0.9111 | 0.9117 | 1.1053 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa musique (in-domain) | 0.4734 | 0.5794 | 0.5961 | 0.5941 | 0.5743 | 0.8360 | 0.0090 | GAIN | WITHIN |
| WQ-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8496 | 0.8183 | 0.8529 | 0.8525 | 0.9976 | 0.0075 | WITHIN | GAIN |
| WQ-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9330 | 0.9414 | 0.9420 | 0.9299 | 0.9638 | 0.0075 | GAIN | WITHIN |
| WQ-hotpotqa webqsp (in-domain) | 0.0535 | 0.6153 | 0.6421 | 0.6536 | 0.6382 | 0.9743 | 0.0089 | GAIN | WITHIN |

The relation map of each fit's p@swa (zg6r's table, unchanged): P's norms and top singular values, the share of their norm in metaqa's centred relation span (under WQ the unseen KB's: P learns from webqsp's relations only), each KB's codes |P_f r| / |P_b r|, the layers' delta and norms, and the encoder's drift from WQK's zspk:

- WQ-musique: |P_f| 11.8012, |P_b| 12.3044; in metaqa's centred span 0.2252 and 0.2165 of their norms; top singular values [7.866493, 5.5325, 4.038832, 3.314684] and [7.720826, 5.662606, 4.99286, 3.998829]; the encoder's drift from zspk's 0.7453.
  - metaqa (9 relations, span rank 8): |P_f r| / |P_b r| in language 1.5951 / 1.3348; has imdb rating 1.0992 / 1.1073; has tags 0.9489 / 0.8964; written by 0.8930 / 1.0285; starred actors 0.8438 / 0.6269; has imdb votes 0.8283 / 0.8552; release year 0.6861 / 0.7859; has genre 0.6353 / 1.6411; directed by 0.5789 / 0.8009.
  - webqsp (7058 relations, 0.0427 of each relation inside metaqa's span on average): |P_f r| / |P_b r| location containedby 3.4441 / 2.9937; country languages spoken 2.9879 / 2.1441; location primarily containedby 2.9219 / 2.3804; human language countries spoken in 2.8170 / 2.0594; location contains 2.6986 / 2.7573.
  - per layer: 0: delta 0.6680 (0.4532 to 0.8518), |A| 3.3771, |B| 2.8284, |C| 1.7552; 1: delta 0.7702 (0.5884 to 0.9788), |A| 4.0928, |B| 3.0138, |C| 2.1630; 2: delta 0.6729 (0.5848 to 0.7768), |A| 3.0807, |B| 2.6307, |C| 2.7392.
- WQ-hotpotqa: |P_f| 11.5244, |P_b| 11.4594; in metaqa's centred span 0.2239 and 0.3096 of their norms; top singular values [7.648607, 5.291651, 4.117141, 2.941547] and [7.315592, 6.003742, 3.351555, 3.214106]; the encoder's drift from zspk's 0.7470.
  - metaqa (9 relations, span rank 8): |P_f r| / |P_b r| starred actors 1.5557 / 2.1790; in language 1.1082 / 0.6994; written by 1.0957 / 1.3628; has tags 0.9018 / 1.5432; release year 0.7642 / 0.7966; directed by 0.6716 / 1.1048; has imdb rating 0.6059 / 1.1126; has genre 0.6012 / 2.0216; has imdb votes 0.5157 / 1.1144.
  - webqsp (7058 relations, 0.0427 of each relation inside metaqa's span on average): |P_f r| / |P_b r| location containedby 2.9012 / 1.2484; cause of death people 2.7389 / 0.9913; deceased person cause of death 2.7260 / 0.9792; human language countries spoken in 2.5949 / 1.3191; location contains 2.5728 / 0.9082.
  - per layer: 0: delta 0.7454 (0.6528 to 0.8991), |A| 3.4577, |B| 2.6197, |C| 1.6208; 1: delta 0.7940 (0.5371 to 1.0273), |A| 4.1615, |B| 3.2154, |C| 2.4058; 2: delta 0.7682 (0.6302 to 0.9337), |A| 3.6205, |B| 2.9325, |C| 2.1537.

Seconds and memory: zg6r, and in brackets WQK's seed-0 zg6 and zspk of the same split, all on the T4:

| split | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---:|---:|---|---:|
| WQ-musique | 1819 (1364, 933) | 1088 (706, 358) | 2.0508 / 4.5888 (2.2806 / 5.2366, 1.8704 / 4.4436) | 4.5600 (4.5400, 4.5100) |
| WQ-hotpotqa | 3958 (3356, 1056) | 962 (715, 348) | 6.0079 / 6.0079 (9.1280 / 9.1280, 3.9910 / 4.4449) | 4.6700 (4.5600, 4.5400) |
