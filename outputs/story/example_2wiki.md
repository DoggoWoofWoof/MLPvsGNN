# Worked examples on 2wiki s1eval (J5 fits)

Rules (in s1eval order): A, zspk's top five holds both golds and zrckp's one, the two golds linked; B, the reverse; C, both models hold both. Counts on the carve: A 198, B 94, C 8564.

## Rule A: What is the date of death of Conan Ii, Duke Of Brittany's mother?

Answer: 1080. Golds: Conan II, Duke of Brittany, Bertha of Blois.
Pool: 73 rows (50 base and seeds, the rest KIND_T's fill); entity question False; 748 links inside the pool.

| row | title / text | gold | pool | rrf z | zrckp score (rank) | zspk score (rank) | links |
|---:|---|---|---|---:|---:|---:|---:|
| 50 | Conan II, Duke of Brittany | yes | base or seed | +1.815 | +1.301 (1) | +0.963 (1) | 68 |
| 4 | Hawise of Rennes (1037 – 19 August 1072) was sovereign Duchess of Brittany from ... |  | base or seed | +1.370 | +0.277 (2) | +0.326 (2) | 48 |
| 49 | Alan III of Rennes (997–1 October 1040) (French: Alain III de Bretagne) was Count ... |  | KIND_T fill 3 | -1.682 | -0.759 (5) | -0.535 (3) | 18 |
| 25 | The Breton–Norman War of 1064–66 was fought between the Duchy of Brittany and the ... |  | base or seed | +1.129 | -0.589 (3) | -0.711 (4) | 30 |
| 32 | Bertha of Blois | yes | KIND_T fill 1 | -1.359 | -1.331 (6) | -1.047 (5) | 10 |
| 53 | Conan III, also known as Conan of Cornouaille and Conan the Fat (c. 1093–1096 ... |  | base or seed | +1.613 | -1.423 (7) | -1.845 (6) | 24 |
| 56 | Conan IV (1138 – February 20, 1171), called the Young, was the Duke of ... |  | base or seed | +1.746 | -0.723 (4) | -2.160 (7) | 54 |
| 52 | Alan IV (born circa 1063; died 13 October 1119) was Duke of Brittany from ... |  | base or seed | +0.324 | -3.170 (15) | -2.660 (8) | 52 |
| 17 | Conan I (d. 27 June 992) nicknamed "Le Tort" (The Crooked) was the Duke ... |  | base or seed | +1.677 | -2.162 (8) | -3.607 (14) | 16 |

Links among the rows shown: 4-25 named-entity; 4-25 structural; 4-32 structural; 4-49 structural; 4-50 kNN; 4-50 named-entity; 4-50 structural; 4-52 kNN; 4-52 named-entity; 4-52 structural; 17-50 kNN; 17-53 kNN; 17-56 kNN; 25-50 named-entity; 25-50 structural; 32-50 structural; 49-50 kNN; 49-50 structural; 49-52 kNN; 50-52 named-entity; 50-53 kNN; 50-56 kNN; 50-56 named-entity; 52-53 structural; 52-56 kNN; 52-56 named-entity; 53-56 kNN.

Features of row 4 (not gold):
- rank: dense_rr 0.125, splade_rr 0.125, rrf 0.0294, agreement 1.0, is_seed 0.0
- dense_cos: dense_cos 0.624
- topo_STRUCT: topo_STRUCT[0] 1.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.3333, topo_STRUCT[5] 2.1973, topo_STRUCT[6] 2.3984, topo_STRUCT[7] 1.6094, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.2031
- depth_STRUCT: depth_STRUCT[0] 0.5435, depth_STRUCT[1] 0.6934, depth_STRUCT[2] 0.1243, depth_STRUCT[3] 1.0, depth_STRUCT[4] 0.5142, depth_STRUCT[5] 0.6748, depth_STRUCT[6] 2.5645, depth_STRUCT[7] 0.5713, depth_STRUCT[8] 1.0986, depth_STRUCT[9] 1.8828, depth_STRUCT[10] 3.2578, depth_STRUCT[11] 2.1973, depth_STRUCT[12] 0.1107, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.5522, depth_STRUCT[15] 0.7134, depth_STRUCT[16] 3.1777, depth_STRUCT[17] 0.7041
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.5728, max_cos_seed 0.6562, max_cos_rank1_seed 0.6562
- WALK: WALK[0] 0.6934, WALK[1] 1.6094, WALK[2] 3.2578, WALK[3] 0.6934, WALK[4] 0.6934, WALK[5] 1.6094, WALK[6] 0.6934, WALK[7] 0.0, WALK[8] 1.6094, WALK[9] 0.6934, WALK[10] 1.3867, WALK[11] 2.1973, WALK[12] 1.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 0.0
- DISTS: DISTS[0] 0.6934, DISTS[1] 1.0, DISTS[2] 0.7676, DISTS[3] 0.6934, DISTS[4] 0.9536, DISTS[5] 0.5913, DISTS[6] 1.0986, DISTS[7] 1.8828, DISTS[8] 0.7012, DISTS[9] 0.125, DISTS[10] 0.1243, DISTS[11] 0.1107, DISTS[12] 1.6094, DISTS[13] 2.1973
- WALKF: WALKF[0] 1.3867, WALKF[1] 4.0938, WALKF[2] 6.8984, WALKF[3] 1.3867, WALKF[4] 3.8926, WALKF[5] 6.5508, WALKF[6] 1.3867, WALKF[7] 3.9121, WALKF[8] 6.6289, WALKF[9] 1.3867, WALKF[10] 3.1777, WALKF[11] 3.2188, WALKF[12] 1.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 0.0, WALKF[16] 0.6934, WALKF[17] 0.6934
- _kt: _kt[0] 0.1241, _kt[1] 0.2898, _kt[2] 0.2808, _kt[3] 0.1241, _kt[4] 0.1241, _kt[5] 0.0783, _kt[6] 0.0, _kt[7] 0.0
- rrf z-score (the base score): 1.37

Features of row 32 (gold):
- rank: dense_rr 0.0039, splade_rr 0.0085, rrf 0.0088, agreement 1.0, is_seed 0.0
- dense_cos: dense_cos 0.4863
- topo_STRUCT: topo_STRUCT[0] 1.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.3333, topo_STRUCT[5] 1.3867, topo_STRUCT[6] 1.9463, topo_STRUCT[7] 1.0986, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.2031
- depth_STRUCT: depth_STRUCT[0] 0.5815, depth_STRUCT[1] 0.6934, depth_STRUCT[2] 0.0667, depth_STRUCT[3] 1.0, depth_STRUCT[4] 0.5225, depth_STRUCT[5] 0.6577, depth_STRUCT[6] 2.9453, depth_STRUCT[7] 0.6636, depth_STRUCT[8] 1.3867, depth_STRUCT[9] 2.7969, depth_STRUCT[10] 2.9453, depth_STRUCT[11] 1.3867, depth_STRUCT[12] 0.1666, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.5576, depth_STRUCT[15] 0.7134, depth_STRUCT[16] 3.332, depth_STRUCT[17] 0.6865
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.5186, max_cos_seed 0.6084, max_cos_rank1_seed 0.5396
- WALK: WALK[0] 0.6934, WALK[1] 1.0986, WALK[2] 2.9453, WALK[3] 0.6934, WALK[4] 0.6934, WALK[5] 1.0986, WALK[6] 0.0, WALK[7] 0.0, WALK[8] 0.6934, WALK[9] 0.6934, WALK[10] 0.6934, WALK[11] 1.3867, WALK[12] 1.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 0.0
- DISTS: DISTS[0] 0.6934, DISTS[1] 1.0, DISTS[2] 0.6313, DISTS[3] 0.6934, DISTS[4] 0.8989, DISTS[5] 0.6562, DISTS[6] 1.3867, DISTS[7] 2.7988, DISTS[8] 0.6675, DISTS[9] 0.3333, DISTS[10] 0.0667, DISTS[11] 0.1666, DISTS[12] 1.0986, DISTS[13] 1.3867
- WALKF: WALKF[0] 1.3867, WALKF[1] 2.709, WALKF[2] 5.5352, WALKF[3] 1.3867, WALKF[4] 2.3027, WALKF[5] 5.1484, WALKF[6] 1.0986, WALKF[7] 1.9463, WALKF[8] 4.3047, WALKF[9] 0.6934, WALKF[10] 1.3867, WALKF[11] 1.792, WALKF[12] 1.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 0.0, WALKF[16] 0.6934, WALKF[17] 0.6934
- _kt: _kt[0] 0.3325, _kt[1] 0.3723, _kt[2] 0.3201, _kt[3] 0.1566, _kt[4] 0.2024, _kt[5] 0.3264, _kt[6] 1.0, _kt[7] 0.0
- rrf z-score (the base score): -1.3586

Features of row 50 (gold):
- rank: dense_rr 1.0, splade_rr 1.0, rrf 0.0328, agreement 1.0, is_seed 1.0
- dense_cos: dense_cos 0.7461
- topo_STRUCT: topo_STRUCT[0] 0.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.1666, topo_STRUCT[5] 2.0801, topo_STRUCT[6] 2.0801, topo_STRUCT[7] 2.0801, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.2031
- depth_STRUCT: depth_STRUCT[0] 0.5039, depth_STRUCT[1] 0.0, depth_STRUCT[2] 0.281, depth_STRUCT[3] 0.0, depth_STRUCT[4] 0.5059, depth_STRUCT[5] 0.5596, depth_STRUCT[6] 2.3984, depth_STRUCT[7] 0.0, depth_STRUCT[8] 1.3867, depth_STRUCT[9] 2.8359, depth_STRUCT[10] 2.709, depth_STRUCT[11] 1.9463, depth_STRUCT[12] 0.0634, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.5586, depth_STRUCT[15] 0.7134, depth_STRUCT[16] 3.2188, depth_STRUCT[17] 0.8184
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.6099, max_cos_seed 0.731, max_cos_rank1_seed 0.731
- WALK: WALK[0] 0.0, WALK[1] 2.0801, WALK[2] 2.709, WALK[3] 0.0, WALK[4] 0.6934, WALK[5] 0.6934, WALK[6] 0.0, WALK[7] 0.6934, WALK[8] 0.0, WALK[9] 0.0, WALK[10] 2.0801, WALK[11] 2.0801, WALK[12] 0.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 1.0
- DISTS: DISTS[0] 0.0, DISTS[1] 0.0, DISTS[2] 0.0, DISTS[3] 0.0, DISTS[4] 0.0, DISTS[5] 0.0, DISTS[6] 1.3867, DISTS[7] 2.8359, DISTS[8] 0.7437, DISTS[9] 0.0, DISTS[10] 0.281, DISTS[11] 0.0634, DISTS[12] 2.0801, DISTS[13] 1.9463
- WALKF: WALKF[0] 1.6094, WALKF[1] 4.543, WALKF[2] 7.1875, WALKF[3] 1.6094, WALKF[4] 4.3555, WALKF[5] 6.9219, WALKF[6] 1.6094, WALKF[7] 4.3828, WALKF[8] 6.9297, WALKF[9] 0.0, WALKF[10] 4.0078, WALKF[11] 3.5547, WALKF[12] 0.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 1.0, WALKF[16] 0.6934, WALKF[17] 1.3867
- _kt: _kt[0] 0.0, _kt[1] 0.2024, _kt[2] 0.0783, _kt[3] 0.0, _kt[4] 0.0, _kt[5] 0.0, _kt[6] 0.0, _kt[7] 0.0
- rrf z-score (the base score): 1.815

## Rule B: Who is the mother of Olaf (Ii) Haraldsen?

Answer: Ragnild Magnusdotter. Golds: Olaf (II) Haraldsen, Harald Kesja.
Pool: 119 rows (50 base and seeds, the rest KIND_T's fill); entity question False; 1140 links inside the pool.

| row | title / text | gold | pool | rrf z | zrckp score (rank) | zspk score (rank) | links |
|---:|---|---|---|---:|---:|---:|---:|
| 57 | Olaf (II) Haraldsen | yes | base or seed | +1.988 | +0.831 (1) | +0.441 (1) | 52 |
| 102 | Olaf II Haakonsson (December 1370 – 3 August 1387) was King of Denmark as ... |  | base or seed | +1.897 | +0.426 (2) | +0.191 (2) | 56 |
| 8 | Olaf II Haraldsson (c. 995 – 29 July 1030), later known as St. Olaf ... |  | base or seed | +1.448 | -0.930 (3) | -1.788 (3) | 72 |
| 94 | Åsta Gudbrandsdatter (c. 975/980 – c. 1020/1030) was the mother of two Norwegian kings, ... |  | base or seed | +1.941 | -1.172 (4) | -1.926 (4) | 28 |
| 96 | Margaret I (15 March 1353 – 28 October 1412) was queen consort of Norway ... |  | KIND_T fill 23 | -1.261 | -3.070 (10) | -2.212 (5) | 20 |
| 30 | Haakon VI of Norway (, ; 1340–1380), also known as "Håkan Magnusson", was King ... |  | KIND_T fill 2 | -0.863 | -2.577 (6) | -2.309 (6) | 22 |
| 50 | Harald Kesja | yes | KIND_T fill 1 | -0.697 | -2.304 (5) | -2.349 (7) | 26 |
| 113 | Eric II the Memorable( – 18 September 1137) was king of Denmark between 1134 ... |  | KIND_T fill 3 | -1.261 | -3.311 (13) | -2.669 (8) | 24 |
| 19 | Sigurd Syr (Old Norse: "Sigurðr Sýr") (died c. 1018) was a Norwegian petty king ... |  | base or seed | +1.438 | -2.855 (8) | -3.824 (13) | 30 |
| 18 | Olaf Haraldssøn Geirstadalf Digerbein (Agder, ca. 877 - Tønsberg, ca. 934), was a reputed ... |  | base or seed | +1.938 | -2.786 (7) | -4.316 (21) | 24 |

Links among the rows shown: 8-19 structural; 8-94 structural; 19-94 kNN; 19-94 named-entity; 19-94 structural; 30-96 structural; 30-102 kNN; 30-102 structural; 50-57 kNN; 50-57 named-entity; 50-57 structural; 50-113 structural; 57-102 structural; 57-113 named-entity; 57-113 structural; 96-102 structural.

Features of row 50 (gold):
- rank: dense_rr 0.0056, splade_rr 0.0015, rrf 0.0055, agreement 1.0, is_seed 0.0
- dense_cos: dense_cos 0.5474
- topo_STRUCT: topo_STRUCT[0] 1.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.2222, topo_STRUCT[5] 1.9463, topo_STRUCT[6] 2.4844, topo_STRUCT[7] 1.3867, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.6133
- depth_STRUCT: depth_STRUCT[0] 0.4836, depth_STRUCT[1] 0.6934, depth_STRUCT[2] 0.1246, depth_STRUCT[3] 1.0, depth_STRUCT[4] 0.4048, depth_STRUCT[5] 0.7065, depth_STRUCT[6] 2.709, depth_STRUCT[7] 0.5093, depth_STRUCT[8] 0.6934, depth_STRUCT[9] 0.9043, depth_STRUCT[10] 3.2578, depth_STRUCT[11] 1.9463, depth_STRUCT[12] 0.151, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.4985, depth_STRUCT[15] 0.6582, depth_STRUCT[16] 3.4336, depth_STRUCT[17] 0.5518
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.4773, max_cos_seed 0.5708, max_cos_rank1_seed 0.4802
- WALK: WALK[0] 0.6934, WALK[1] 1.3867, WALK[2] 3.2578, WALK[3] 0.6934, WALK[4] 0.0, WALK[5] 0.6934, WALK[6] 0.0, WALK[7] 0.0, WALK[8] 0.0, WALK[9] 0.0, WALK[10] 0.6934, WALK[11] 1.9463, WALK[12] 1.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 0.0
- DISTS: DISTS[0] 0.6934, DISTS[1] 1.0, DISTS[2] 0.6538, DISTS[3] 0.6934, DISTS[4] 0.9722, DISTS[5] 0.5303, DISTS[6] 0.6934, DISTS[7] 0.9038, DISTS[8] 0.5601, DISTS[9] 0.1666, DISTS[10] 0.1246, DISTS[11] 0.1509, DISTS[12] 1.3867, DISTS[13] 1.9463
- WALKF: WALKF[0] 1.3867, WALKF[1] 2.6387, WALKF[2] 5.4492, WALKF[3] 1.3867, WALKF[4] 2.0801, WALKF[5] 4.7773, WALKF[6] 1.0986, WALKF[7] 2.3027, WALKF[8] 4.6992, WALKF[9] 0.0, WALKF[10] 1.6094, WALKF[11] 2.6387, WALKF[12] 1.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 0.0, WALKF[16] 0.6934, WALKF[17] 0.6934
- _kt: _kt[0] 0.2024, _kt[1] 0.4221, _kt[2] 0.3542, _kt[3] 0.2024, _kt[4] 0.1241, _kt[5] 0.1566, _kt[6] 1.0, _kt[7] 0.0
- rrf z-score (the base score): -0.6968

Features of row 57 (gold):
- rank: dense_rr 0.25, splade_rr 0.5, rrf 0.0318, agreement 1.0, is_seed 1.0
- dense_cos: dense_cos 0.6689
- topo_STRUCT: topo_STRUCT[0] 0.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.2222, topo_STRUCT[5] 2.5645, topo_STRUCT[6] 2.5645, topo_STRUCT[7] 2.5645, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.6133
- depth_STRUCT: depth_STRUCT[0] 0.3547, depth_STRUCT[1] 0.0, depth_STRUCT[2] 0.4036, depth_STRUCT[3] 0.0, depth_STRUCT[4] 0.4773, depth_STRUCT[5] 0.6309, depth_STRUCT[6] 3.3672, depth_STRUCT[7] 0.0, depth_STRUCT[8] 1.3867, depth_STRUCT[9] 2.7207, depth_STRUCT[10] 3.8711, depth_STRUCT[11] 2.5645, depth_STRUCT[12] 0.1436, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.5723, depth_STRUCT[15] 0.6899, depth_STRUCT[16] 3.6113, depth_STRUCT[17] 0.7041
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.5156, max_cos_seed 0.7622, max_cos_rank1_seed 0.5303
- WALK: WALK[0] 0.6934, WALK[1] 2.5645, WALK[2] 3.8711, WALK[3] 0.0, WALK[4] 0.0, WALK[5] 0.0, WALK[6] 0.6934, WALK[7] 0.0, WALK[8] 1.6094, WALK[9] 0.6934, WALK[10] 0.0, WALK[11] 2.5645, WALK[12] 0.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 1.0
- DISTS: DISTS[0] 0.6934, DISTS[1] 0.9722, DISTS[2] 0.6094, DISTS[3] 0.0, DISTS[4] 0.0, DISTS[5] 0.0, DISTS[6] 1.3867, DISTS[7] 2.7188, DISTS[8] 0.6646, DISTS[9] 0.081, DISTS[10] 0.4036, DISTS[11] 0.1436, DISTS[12] 2.5645, DISTS[13] 2.5645
- WALKF: WALKF[0] 1.0986, WALKF[1] 3.6641, WALKF[2] 5.8281, WALKF[3] 0.6934, WALKF[4] 3.0449, WALKF[5] 4.7617, WALKF[6] 1.0986, WALKF[7] 3.1348, WALKF[8] 5.1836, WALKF[9] 0.6934, WALKF[10] 0.6934, WALKF[11] 3.2949, WALKF[12] 0.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 1.0, WALKF[16] 0.0, WALKF[17] 0.6934
- _kt: _kt[0] 0.0, _kt[1] 0.1818, _kt[2] 0.1566, _kt[3] 0.0, _kt[4] 0.0, _kt[5] 0.0, _kt[6] 0.0, _kt[7] 0.0
- rrf z-score (the base score): 1.9877

Features of row 102 (not gold):
- rank: dense_rr 1.0, splade_rr 0.1111, rrf 0.0309, agreement 1.0, is_seed 1.0
- dense_cos: dense_cos 0.7065
- topo_STRUCT: topo_STRUCT[0] 0.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.2222, topo_STRUCT[5] 2.6387, topo_STRUCT[6] 2.709, topo_STRUCT[7] 2.6387, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.6133
- depth_STRUCT: depth_STRUCT[0] 0.446, depth_STRUCT[1] 0.0, depth_STRUCT[2] 0.2124, depth_STRUCT[3] 0.0, depth_STRUCT[4] 0.385, depth_STRUCT[5] 0.5913, depth_STRUCT[6] 2.9453, depth_STRUCT[7] 0.0, depth_STRUCT[8] 1.792, depth_STRUCT[9] 4.418, depth_STRUCT[10] 4.3047, depth_STRUCT[11] 2.6387, depth_STRUCT[12] 0.1472, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.5659, depth_STRUCT[15] 0.6899, depth_STRUCT[16] 3.6113, depth_STRUCT[17] 0.7041
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.5454, max_cos_seed 0.7563, max_cos_rank1_seed 0.7563
- WALK: WALK[0] 0.6934, WALK[1] 2.6387, WALK[2] 4.3047, WALK[3] 0.6934, WALK[4] 1.0986, WALK[5] 1.792, WALK[6] 0.0, WALK[7] 1.0986, WALK[8] 0.6934, WALK[9] 0.0, WALK[10] 2.6387, WALK[11] 2.6387, WALK[12] 0.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 1.0
- DISTS: DISTS[0] 0.6934, DISTS[1] 1.0, DISTS[2] 0.6074, DISTS[3] 0.0, DISTS[4] 0.0, DISTS[5] 0.0, DISTS[6] 1.792, DISTS[7] 4.4141, DISTS[8] 0.6733, DISTS[9] 0.0769, DISTS[10] 0.2123, DISTS[11] 0.1471, DISTS[12] 2.6387, DISTS[13] 2.6387
- WALKF: WALKF[0] 0.6934, WALKF[1] 3.8711, WALKF[2] 6.1406, WALKF[3] 0.6934, WALKF[4] 3.4961, WALKF[5] 5.5, WALKF[6] 0.0, WALKF[7] 3.332, WALKF[8] 5.0742, WALKF[9] 0.0, WALKF[10] 3.5547, WALKF[11] 3.3672, WALKF[12] 0.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 1.0, WALKF[16] 0.0, WALKF[17] 0.0
- _kt: _kt[0] 0.0783, _kt[1] 0.1566, _kt[2] 0.2024, _kt[3] 0.2898, _kt[4] 0.0783, _kt[5] 0.1241, _kt[6] 0.0, _kt[7] 0.0
- rrf z-score (the base score): 1.8972

## Rule C: Who is the mother of the director of film Polish-Russian War (Film)?

Answer: Małgorzata Braunek. Golds: Polish-Russian War (film), Xawery Żuławski.
Pool: 110 rows (50 base and seeds, the rest KIND_T's fill); entity question False; 496 links inside the pool.

| row | title / text | gold | pool | rrf z | zrckp score (rank) | zspk score (rank) | links |
|---:|---|---|---|---:|---:|---:|---:|
| 88 | Polish-Russian War (film) | yes | base or seed | +2.537 | +1.388 (2) | +1.163 (1) | 18 |
| 107 | Xawery Żuławski | yes | base or seed | +1.005 | +1.419 (1) | +0.781 (2) | 44 |
| 29 | Dorota Masłowska (; born 3 July 1983) is a Polish writer, playwright, columnist and ... |  | KIND_T fill 2 | -1.316 | -0.852 (3) | -1.742 (3) | 10 |
| 106 | Mirosław Żuławski (January 16, 1913 – February 17, 1995) was a Polish writer, prosaist, ... |  | KIND_T fill 3 | +0.268 | -3.070 (7) | -2.099 (4) | 16 |
| 81 | Andrzej Witold Wajda ( 6 March 1926 – 9 October 2016) was a Polish ... |  | KIND_T fill 4 | -0.345 | -1.951 (4) | -2.495 (5) | 18 |
| 79 | Andrzej Żuławski (22 November 1940 – 17 February 2016) was a Polish film director ... |  | KIND_T fill 8 | -0.880 | -4.004 (13) | -2.966 (6) | 28 |
| 7 | Janusz Zaorski (born 19 September 1947) is a Polish film director, scenarist and actor, ... |  | KIND_T fill 12 | -0.935 | -3.733 (12) | -3.689 (7) | 14 |
| 84 | Mother ("Mat") is a 1926 Soviet drama film directed by Vsevolod Pudovkin depicting one ... |  | base or seed | +2.302 | -2.974 (6) | -3.885 (8) | 26 |
| 101 | Mark Semyonovich Donskoy (– 21 March 1981) was a Soviet film director and screenwriter. ... |  | KIND_T fill 23 | -1.316 | -3.095 (8) | -4.297 (11) | 12 |
| 64 | The Mother (Matka in Czech) is an anti-war drama written by Czech novelist and ... |  | KIND_T fill 6 | -0.558 | -2.836 (5) | -4.732 (17) | 14 |

Links among the rows shown: 29-88 structural; 29-107 structural; 79-106 kNN; 79-106 named-entity; 79-106 structural; 79-107 kNN; 79-107 named-entity; 79-107 structural; 88-107 named-entity; 88-107 structural; 106-107 kNN; 106-107 named-entity.

Features of row 29 (not gold):
- rank: dense_rr 0.0, splade_rr 0.0, rrf 0.0, agreement 0.0, is_seed 0.0
- dense_cos: dense_cos 0.3604
- topo_STRUCT: topo_STRUCT[0] 1.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.1111, topo_STRUCT[5] 1.6094, topo_STRUCT[6] 1.6094, topo_STRUCT[7] 0.6934, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.4531
- depth_STRUCT: depth_STRUCT[0] 0.4841, depth_STRUCT[1] 0.0, depth_STRUCT[2] 0.0278, depth_STRUCT[3] 0.0, depth_STRUCT[4] 0.3513, depth_STRUCT[5] 0.4895, depth_STRUCT[6] 2.1973, depth_STRUCT[7] 0.0, depth_STRUCT[8] 1.0986, depth_STRUCT[9] 1.2246, depth_STRUCT[10] 2.709, depth_STRUCT[11] 1.6094, depth_STRUCT[12] 0.1377, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.4907, depth_STRUCT[15] 0.5957, depth_STRUCT[16] 3.6641, depth_STRUCT[17] 0.3398
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.3118, max_cos_seed 0.3904, max_cos_rank1_seed 0.3633
- WALK: WALK[0] 0.6934, WALK[1] 0.6934, WALK[2] 2.709, WALK[3] 0.6934, WALK[4] 0.6934, WALK[5] 0.0, WALK[6] 0.0, WALK[7] 0.0, WALK[8] 0.0, WALK[9] 0.6934, WALK[10] 0.6934, WALK[11] 1.6094, WALK[12] 1.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 0.0
- DISTS: DISTS[0] 0.6934, DISTS[1] 1.0, DISTS[2] 0.4001, DISTS[3] 0.0, DISTS[4] 0.0, DISTS[5] 0.0, DISTS[6] 1.0986, DISTS[7] 1.2246, DISTS[8] 0.3716, DISTS[9] 0.25, DISTS[10] 0.0278, DISTS[11] 0.1377, DISTS[12] 0.6934, DISTS[13] 1.6094
- WALKF: WALKF[0] 0.6934, WALKF[1] 1.3867, WALKF[2] 3.4648, WALKF[3] 0.6934, WALKF[4] 1.3867, WALKF[5] 2.5645, WALKF[6] 0.0, WALKF[7] 0.0, WALKF[8] 0.0, WALKF[9] 0.6934, WALKF[10] 1.3867, WALKF[11] 1.792, WALKF[12] 1.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 0.0, WALKF[16] 0.0, WALKF[17] 0.0
- _kt: _kt[0] 0.2024, _kt[1] 0.3132, _kt[2] 0.3635, _kt[3] 0.1818, _kt[4] 0.2197, _kt[5] 0.3325, _kt[6] 1.0, _kt[7] 0.0
- rrf z-score (the base score): -1.3164

Features of row 88 (gold):
- rank: dense_rr 1.0, splade_rr 1.0, rrf 0.0328, agreement 1.0, is_seed 1.0
- dense_cos: dense_cos 0.6675
- topo_STRUCT: topo_STRUCT[0] 0.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.3333, topo_STRUCT[5] 1.3867, topo_STRUCT[6] 1.3867, topo_STRUCT[7] 1.792, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.4531
- depth_STRUCT: depth_STRUCT[0] 0.4104, depth_STRUCT[1] 1.0986, depth_STRUCT[2] 0.15, depth_STRUCT[3] 1.0, depth_STRUCT[4] 0.4575, depth_STRUCT[5] 0.5957, depth_STRUCT[6] 3.332, depth_STRUCT[7] 0.5254, depth_STRUCT[8] 1.3867, depth_STRUCT[9] 1.8633, depth_STRUCT[10] 2.6387, depth_STRUCT[11] 1.3867, depth_STRUCT[12] 0.0345, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.4839, depth_STRUCT[15] 0.5679, depth_STRUCT[16] 3.3672, depth_STRUCT[17] 0.4604
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.4458, max_cos_seed 0.8237, max_cos_rank1_seed 0.8237
- WALK: WALK[0] 0.0, WALK[1] 1.792, WALK[2] 2.6387, WALK[3] 0.0, WALK[4] 0.0, WALK[5] 0.0, WALK[6] 0.0, WALK[7] 0.0, WALK[8] 0.0, WALK[9] 0.0, WALK[10] 1.3867, WALK[11] 1.3867, WALK[12] 0.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 1.0
- DISTS: DISTS[0] 0.0, DISTS[1] 0.0, DISTS[2] 0.0, DISTS[3] 1.0986, DISTS[4] 1.2246, DISTS[5] 0.5483, DISTS[6] 1.3867, DISTS[7] 1.8633, DISTS[8] 0.4917, DISTS[9] 0.0, DISTS[10] 0.15, DISTS[11] 0.0345, DISTS[12] 1.3867, DISTS[13] 1.3867
- WALKF: WALKF[0] 0.0, WALKF[1] 2.6387, WALKF[2] 3.8496, WALKF[3] 0.0, WALKF[4] 2.0801, WALKF[5] 1.6094, WALKF[6] 0.0, WALKF[7] 2.0801, WALKF[8] 1.6094, WALKF[9] 0.0, WALKF[10] 2.4844, WALKF[11] 2.3027, WALKF[12] 0.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 1.0, WALKF[16] 0.0, WALKF[17] 0.0
- _kt: _kt[0] 0.0, _kt[1] 0.0, _kt[2] 0.1566, _kt[3] 0.0, _kt[4] 0.0, _kt[5] 0.0, _kt[6] 0.0, _kt[7] 0.0
- rrf z-score (the base score): 2.5367

Features of row 107 (gold):
- rank: dense_rr 0.0667, splade_rr 0.0104, rrf 0.0197, agreement 1.0, is_seed 0.0
- dense_cos: dense_cos 0.5659
- topo_STRUCT: topo_STRUCT[0] 1.0, topo_STRUCT[1] 0.0, topo_STRUCT[2] 0.0, topo_STRUCT[3] 0.0, topo_STRUCT[4] 0.3333, topo_STRUCT[5] 2.3027, topo_STRUCT[6] 2.3027, topo_STRUCT[7] 1.6094, topo_STRUCT[8] 1.0, topo_STRUCT[9] 1.0, topo_STRUCT[10] 4.4531
- depth_STRUCT: depth_STRUCT[0] 0.3921, depth_STRUCT[1] 1.0986, depth_STRUCT[2] 0.0377, depth_STRUCT[3] 1.0, depth_STRUCT[4] 0.4871, depth_STRUCT[5] 0.5957, depth_STRUCT[6] 3.7129, depth_STRUCT[7] 0.6001, depth_STRUCT[8] 1.792, depth_STRUCT[9] 3.4629, depth_STRUCT[10] 3.7129, depth_STRUCT[11] 2.3027, depth_STRUCT[12] 0.0683, depth_STRUCT[13] 1.0, depth_STRUCT[14] 0.4897, depth_STRUCT[15] 0.5894, depth_STRUCT[16] 2.7734, depth_STRUCT[17] 0.5322
- SEMB: learned product of the question's embedding and the node's projection (257)
- SEED: cos_seed_mean 0.458, max_cos_seed 0.5503, max_cos_rank1_seed 0.5317
- WALK: WALK[0] 0.6934, WALK[1] 1.6094, WALK[2] 3.7129, WALK[3] 0.6934, WALK[4] 0.0, WALK[5] 0.0, WALK[6] 0.0, WALK[7] 0.0, WALK[8] 0.0, WALK[9] 0.6934, WALK[10] 1.0986, WALK[11] 2.3027, WALK[12] 1.0, WALK[13] 0.0, WALK[14] 0.0, WALK[15] 0.0
- DISTS: DISTS[0] 0.6934, DISTS[1] 1.0, DISTS[2] 0.5859, DISTS[3] 1.0986, DISTS[4] 1.2246, DISTS[5] 0.6343, DISTS[6] 1.792, DISTS[7] 3.4648, DISTS[8] 0.5674, DISTS[9] 0.1111, DISTS[10] 0.0377, DISTS[11] 0.0684, DISTS[12] 1.0986, DISTS[13] 2.3027
- WALKF: WALKF[0] 1.0986, WALKF[1] 1.792, WALKF[2] 5.1055, WALKF[3] 1.0986, WALKF[4] 0.6934, WALKF[5] 4.1445, WALKF[6] 0.6934, WALKF[7] 0.6934, WALKF[8] 3.3672, WALKF[9] 1.0986, WALKF[10] 1.3867, WALKF[11] 3.1348, WALKF[12] 1.0, WALKF[13] 0.0, WALKF[14] 0.0, WALKF[15] 0.0, WALKF[16] 0.6934, WALKF[17] 0.0
- _kt: _kt[0] 0.0783, _kt[1] 0.2981, _kt[2] 0.26, _kt[3] 0.1241, _kt[4] 0.1241, _kt[5] 0.1241, _kt[6] 0.0, _kt[7] 0.0
- rrf z-score (the base score): 1.0048

