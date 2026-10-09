# B1d: training-free chain slots with query offsets

| setting | R0 (RRF) R@5 | chosen arm | its R@5 (all) | odd-half delta [95% CI] | call | HippoRAG 2 |
| --- | --- | --- | --- | --- | --- | --- |
| musique | 56.4 | s1m4/ner/Q | 60.4 | +4.0 [+2.5, +5.7] | ABOVE | 74.7 |
| 2wiki | 71.2 | s2m3/structural/Q | 92.8 | +21.2 [+18.9, +23.2] | ABOVE | 90.4 |
| hotpotqa | 85.0 | s2m3/structural/Q | 95.2 | +10.0 [+8.0, +11.9] | ABOVE | 96.3 |

R@5 (all 1,000) of every arm:

| arm | musique | 2wiki | hotpotqa |
| --- | ---: | ---: | ---: |
| R0 | 56.4 | 71.2 | 85.0 |
| s1m3/all/Q | 58.6 | 78.6 | 88.0 |
| s1m3/all/QR | 55.8 | 79.7 | 87.1 |
| s1m3/structural/Q | 57.6 | 89.5 | 94.6 |
| s1m3/structural/QR | 57.1 | 89.4 | 94.5 |
| s1m3/ner/Q | 60.0 | 80.6 | 86.1 |
| s1m3/ner/QR | 58.1 | 81.8 | 85.7 |
| s1m3/knn/Q | 54.7 | 76.9 | 85.4 |
| s1m3/knn/QR | 54.9 | 78.6 | 86.0 |
| s1m3/none/Q | 57.5 | 71.5 | 85.5 |
| s1m3/none/QR | 55.2 | 76.6 | 86.2 |
| s1m4/all/Q | 58.4 | 76.1 | 87.2 |
| s1m4/all/QR | 57.2 | 77.7 | 86.8 |
| s1m4/structural/Q | 59.0 | 88.3 | 94.2 |
| s1m4/structural/QR | 58.9 | 88.2 | 94.0 |
| s1m4/ner/Q | 60.4 | 78.3 | 86.6 |
| s1m4/ner/QR | 58.9 | 79.7 | 86.7 |
| s1m4/knn/Q | 56.2 | 75.1 | 85.7 |
| s1m4/knn/QR | 56.8 | 77.5 | 86.2 |
| s1m4/none/Q | 57.6 | 71.7 | 85.6 |
| s1m4/none/QR | 56.9 | 75.7 | 86.3 |
| s2m3/all/Q | 57.7 | 75.3 | 86.5 |
| s2m3/all/QR | 54.6 | 75.5 | 85.4 |
| s2m3/structural/Q | 57.0 | 92.8 | 95.2 |
| s2m3/structural/QR | 56.5 | 92.4 | 94.5 |
| s2m3/ner/Q | 59.9 | 78.5 | 86.5 |
| s2m3/ner/QR | 57.0 | 79.9 | 85.5 |
| s2m3/knn/Q | 55.0 | 74.0 | 85.2 |
| s2m3/knn/QR | 54.2 | 75.4 | 85.0 |
| s2m3/none/Q | 57.5 | 71.5 | 85.5 |
| s2m3/none/QR | 54.3 | 74.6 | 85.1 |
| s2m4/all/Q | 57.8 | 74.1 | 86.2 |
| s2m4/all/QR | 56.6 | 74.5 | 86.0 |
| s2m4/structural/Q | 58.7 | 87.1 | 94.0 |
| s2m4/structural/QR | 58.2 | 85.7 | 92.8 |
| s2m4/ner/Q | 59.4 | 76.2 | 86.6 |
| s2m4/ner/QR | 57.9 | 77.3 | 86.4 |
| s2m4/knn/Q | 56.6 | 73.0 | 85.5 |
| s2m4/knn/QR | 56.4 | 74.6 | 85.7 |
| s2m4/none/Q | 57.6 | 71.7 | 85.6 |
| s2m4/none/QR | 56.1 | 74.0 | 85.7 |

QR minus Q, R@5 points (all):

| rule | musique | 2wiki | hotpotqa |
| --- | ---: | ---: | ---: |
| s1m3/all | -2.8 | +1.0 | -0.9 |
| s1m3/structural | -0.5 | -0.1 | -0.1 |
| s1m3/ner | -1.9 | +1.3 | -0.4 |
| s1m3/knn | +0.2 | +1.7 | +0.6 |
| s1m3/none | -2.4 | +5.1 | +0.6 |
| s1m4/all | -1.2 | +1.6 | -0.4 |
| s1m4/structural | -0.1 | -0.0 | -0.2 |
| s1m4/ner | -1.5 | +1.4 | +0.1 |
| s1m4/knn | +0.6 | +2.4 | +0.5 |
| s1m4/none | -0.8 | +4.0 | +0.7 |
| s2m3/all | -3.1 | +0.2 | -1.1 |
| s2m3/structural | -0.5 | -0.4 | -0.7 |
| s2m3/ner | -2.8 | +1.4 | -1.1 |
| s2m3/knn | -0.8 | +1.4 | -0.2 |
| s2m3/none | -3.3 | +3.1 | -0.4 |
| s2m4/all | -1.3 | +0.4 | -0.3 |
| s2m4/structural | -0.5 | -1.4 | -1.1 |
| s2m4/ner | -1.5 | +1.1 | -0.2 |
| s2m4/knn | -0.2 | +1.6 | +0.2 |
| s2m4/none | -1.5 | +2.3 | +0.1 |

E2 (the chosen rule on SPLADE's list; reported, never used to choose):

- musique: {'splade_R5': 0.5082, 'rule_on_splade_R5': 0.5653, 'gain': 0.0571, 'gain_on_rrf': 0.0394}
- 2wiki: {'splade_R5': 0.7045, 'rule_on_splade_R5': 0.9255, 'gain': 0.221, 'gain_on_rrf': 0.2158}
- hotpotqa: {'splade_R5': 0.8035, 'rule_on_splade_R5': 0.941, 'gain': 0.1375, 'gain_on_rrf': 0.102}

R@5 by question kind (R0 -> chosen):

- musique: 2hop 0.652 -> 0.715; 3hop1 0.525 -> 0.531; 3hop2 0.539 -> 0.543; 4hop1 0.333 -> 0.377; 4hop2 0.361 -> 0.370; 4hop3 0.435 -> 0.444
- 2wiki: bridge_comparison 0.531 -> 0.879; comparison 0.978 -> 0.975; compositional 0.654 -> 0.925; inference 0.727 -> 0.935
- hotpotqa: bridge 0.822 -> 0.953; comparison 0.971 -> 0.947
