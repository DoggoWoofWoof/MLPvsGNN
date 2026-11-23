# MHR1 report (2026-11-23T06:01:06Z)

Declared in docs/MHR1_MULTIHOPRAG_PACKAGE.md. Package: 609 articles, 2255 kept questions ({"train": {"comparison_query": 342, "inference_query": 326, "temporal_query": 233}, "dev": {"comparison_query": 85, "inference_query": 81, "temporal_query": 58}, "test": {"comparison_query": 429, "inference_query": 409, "temporal_query": 292}}).

## First stage: **e3v LOSES ON MULTIHOP-RAG**

| list | d2 | d4 | d5 | d10 | d20 | rrf5 | hits4 | hits10 | map10 | mrr10 |
|---|---|---|---|---|---|---|---|---|---|---|
| e3v | 41.8 | 58.9 | 64.29 | 79.41 | 92.14 | 61.87 | 89.58 | 98.31 | 42.26 | 74.55 |
| frozen | 42.41 | 61.07 | 66.91 | 81.59 | 93.14 | 63.43 | 91.84 | 98.54 | 42.93 | 75.19 |
| splade | 36.68 | 52.1 | 57.46 | 71.19 | 83.92 |  | 84.04 | 96.76 | 37.63 | 69.15 |
| e3v+splade | 41.7 | 57.15 | 61.87 | 77.58 | 91.28 |  | 88.69 | 98.31 | 41.79 | 74.2 |
| frozen+splade | 42.53 | 58.35 | 63.43 | 78.62 | 92.05 |  | 90.64 | 98.23 | 42.31 | 75.17 |

| e3v - frozen (all) | points [95%] | call |
|---|---|---|
| d2 | -0.61 [-1.66, +0.44] | LEVEL |
| d4 | -2.18 [-3.23, -1.10] | BELOW |
| d5 | -2.62 [-3.70, -1.56] | BELOW |
| d10 | -2.18 [-3.03, -1.35] | BELOW |
| d20 | -1.00 [-1.68, -0.34] | BELOW |
| hits4 | -2.26 [-3.46, -0.98] | BELOW |
| hits10 | -0.22 [-0.80, +0.40] | LEVEL |
| map10 | -0.67 [-1.32, -0.04] | BELOW |
| mrr10 | -0.64 [-1.93, +0.59] | LEVEL |
| rrf5 | -1.56 [-2.30, -0.84] | BELOW |

| subset | n | e3v d5 | frozen d5 | gap d5 [95%] | gap rrf5 [95%] |
|---|---|---|---|---|---|
| all | 2255 | 64.29 | 66.91 | -2.62 [-3.70, -1.56] BELOW | -1.56 [-2.30, -0.84] BELOW |
| test | 1130 | 63.89 | 66.08 | -2.19 [-3.79, -0.61] BELOW | -2.08 [-3.17, -1.02] BELOW |
| comparison_query | 856 | 67.02 | 69.9 | -2.88 [-4.75, -1.07] BELOW | -1.60 [-2.82, -0.37] BELOW |
| inference_query | 816 | 57.02 | 60.76 | -3.75 [-5.36, -2.10] BELOW | -1.93 [-3.10, -0.77] BELOW |
| temporal_query | 583 | 70.47 | 71.13 | -0.66 [-2.80, +1.43] LEVEL | -1.00 [-2.49, +0.49] LEVEL |
| golds2 | 1169 | 72.63 | 74.59 | -1.97 [-3.68, -0.34] BELOW | -1.20 [-2.27, -0.17] BELOW |
| golds3 | 774 | 57.62 | 61.28 | -3.66 [-5.30, -2.07] BELOW | -2.20 [-3.45, -0.95] BELOW |
| golds4 | 312 | 49.6 | 52.08 | -2.48 [-4.81, -0.08] BELOW | -1.36 [-3.04, +0.40] LEVEL |
| a_gold_cut_by_e3v | 164 | 60.01 | 60.32 | -0.30 [-4.52, +4.07] LEVEL | +3.61 [+1.17, +6.15] ABOVE |
| no_gold_cut_by_e3v | 2091 | 64.63 | 67.43 | -2.80 [-3.87, -1.65] BELOW | -1.97 [-2.75, -1.20] BELOW |

Cut at the window: {"e3v": {"articles": 13, "gold_slots": 169, "gold_slots_pct": 2.86, "questions_with_a_cut_gold": 164}, "frozen": {"articles": 0, "gold_slots": 0, "gold_slots_pct": 0.0, "questions_with_a_cut_gold": 0}, "splade": {"articles": 609, "gold_slots": 5908, "gold_slots_pct": 100.0, "questions_with_a_cut_gold": 2255}, "cuts_json_sha256": "041bb031e7c4dd41b207f272de8e5dfd3c4c2cd62632b9a5b40d39a9fccc2309"}.

## Links: **NOT_CARRIED (before H)**

| form | pairs | density % | GP linked | GP share % [95%] | lift | call | GS % | pairs per linked slot |
|---|---|---|---|---|---|---|---|---|
| F0T | 25,080 | 13.55 | 4,541 | 84.67 [83.44, 85.89] | 6.25 | POINTS | 80.58 | 5.5 |
| W2_S@e3v | 2 | 0.0 | 1 | 0.02 [0.0, 0.06] | 17.26 | NO_LIFT | 0.04 | 2.0 |
| R@e3v | 25,080 | 13.55 | 4,541 | 84.67 [83.44, 85.89] | 6.25 | POINTS | 80.58 | 5.5 |
| K@e3v | 1,366 | 0.74 | 1,531 | 28.55 [27.23, 29.93] | 38.69 | POINTS | 24.3 | 0.9 |
| RK@e3v | 25,343 | 13.69 | 4,564 | 85.1 [83.92, 86.31] | 6.22 | POINTS | 81.55 | 5.6 |
| W2_S@frozen | 2 | 0.0 | 1 | 0.02 [0.0, 0.06] | 17.26 | NO_LIFT | 0.04 | 2.0 |
| R@frozen | 25,080 | 13.55 | 4,541 | 84.67 [83.44, 85.89] | 6.25 | POINTS | 80.58 | 5.5 |
| K@frozen | 1,339 | 0.72 | 1,571 | 29.29 [27.89, 30.73] | 40.5 | POINTS | 25.28 | 0.9 |
| RK@frozen | 25,304 | 13.67 | 4,665 | 86.98 [85.88, 88.08] | 6.36 | POINTS | 83.33 | 5.4 |

| difference | GP points [95%] | GS points [95%] |
|---|---|---|
| R@e3v - F0T | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL |
| RK@e3v - R@e3v | +0.43 [+0.26, +0.61] ABOVE | +0.98 [+0.58, +1.42] ABOVE |
| RK@frozen - R@frozen | +2.31 [+1.93, +2.73] ABOVE | +2.75 [+2.08, +3.46] ABOVE |
| RK@e3v - RK@frozen | -1.88 [-2.27, -1.49] BELOW | -1.77 [-2.44, -1.15] BELOW |
| K@e3v - K@frozen | -0.75 [-1.64, +0.13] LEVEL | -0.98 [-2.13, +0.27] LEVEL |

W2_S (one build on the frozen rows, section 5.1): {"tie_break_rows": "frozen (section 5.1)", "mention_links": 3, "sibling_links": 0, "pairs": 2, "descriptive_nodes": 609, "surfaces": 2578, "common_forms": 166, "W2_surface_links": 3, "W2_ambiguous": 0}

Stores (R > K):
- e3v: R 25,080; K 1,366 (1,103 already in R, 263 in K's store); RK 25,343; labels {"S": 0, "R.mention": 2, "R.name_comention": 25005, "R.title_comention": 75, "K.knn": 1366}; K {"groups": 609, "edges": 1366, "tie_rows": 62}
- frozen: R 25,080; K 1,339 (1,115 already in R, 224 in K's store); RK 25,304; labels {"S": 0, "R.mention": 2, "R.name_comention": 25005, "R.title_comention": 75, "K.knn": 1339}; K {"groups": 609, "edges": 1339, "tie_rows": 39}
