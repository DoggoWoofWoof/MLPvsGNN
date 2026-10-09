# Step 4g: where step 4e's pools miss golds

| carve | questions missing | misses | pool | F | W | H1 | H2 | H3 | X | linked | deg 0 | <=2 hops of a found gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa__s1sel | 60 / 1497 | 799 | 7,093 | 0.000 | 0.093 | 0.886 | 0.021 | 0.000 | 0.000 | 0.000 | 0.000 | 0.536 |
| metaqa__s1eval | 464 / 9785 | 4446 | 7,082 | 0.000 | 0.114 | 0.879 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.532 |
| squad__s1sel | 12 / 1498 | 12 | 150 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 | 1.000 | 0.000 |
| squad__s1eval | 94 / 11873 | 94 | 150 | 0.894 | 0.000 | 0.000 | 0.000 | 0.000 | 0.106 | 0.000 | 1.000 | 0.000 |
| musique__s1sel | 77 / 1534 | 83 | 6,921 | 0.012 | 0.193 | 0.783 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.711 |
| musique__s1eval | 153 / 2417 | 180 | 6,899 | 0.000 | 0.189 | 0.794 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.589 |
| hotpotqa__s1sel | 34 / 1508 | 44 | 188 | 0.682 | 0.023 | 0.227 | 0.045 | 0.023 | 0.000 | 0.000 | 0.000 | 0.545 |
| hotpotqa__s1eval | 158 / 7405 | 208 | 188 | 0.688 | 0.067 | 0.178 | 0.043 | 0.024 | 0.000 | 0.000 | 0.000 | 0.514 |
| 2wiki__s1sel | 80 / 1496 | 82 | 220 | 0.159 | 0.098 | 0.415 | 0.244 | 0.061 | 0.024 | 0.000 | 0.012 | 0.354 |
| 2wiki__s1eval | 692 / 12576 | 708 | 221 | 0.234 | 0.133 | 0.302 | 0.239 | 0.082 | 0.010 | 0.000 | 0.010 | 0.386 |
| webqsp__s1eval | 137 / 1503 | 2310 | 7,213 | 0.001 | 0.067 | 0.877 | 0.055 | 0.000 | 0.000 | 0.000 | 0.000 | 0.920 |

H1 first-hop families and buckets by question kind:

- metaqa__s1sel: H1 via {'structural': 708}; F median dense rank 0; W median rank/B 2.475; by kind {'2': {'W': 8, 'H1': 2}, '3': {'H1': 706, 'W': 66, 'H2': 17}}
- metaqa__s1eval: H1 via {'structural': 3909}; F median dense rank 0; W median rank/B 2.491; by kind {'1': {'W': 3, 'H1': 1}, '2': {'W': 27, 'H1': 16}, '3': {'H1': 3892, 'W': 478, 'H2': 29}}
- squad__s1sel: H1 via {}; F median dense rank 374; W median rank/B 0; by kind {'': {'F': 9, 'X': 3}}
- squad__s1eval: H1 via {}; F median dense rank 350.5; W median rank/B 0; by kind {'': {'F': 84, 'X': 10}}
- musique__s1sel: H1 via {'structural': 64, 'ner': 35, 'knn': 33}; F median dense rank 0; W median rank/B 2.2935; by kind {'2hop': {'H1': 11, 'W': 3, 'H2': 1}, '3hop': {'H1': 36, 'W': 10, 'F': 1}, '4hop': {'H1': 18, 'W': 3}}
- musique__s1eval: H1 via {'structural': 133, 'knn': 70, 'ner': 78}; F median dense rank 0; W median rank/B 2.4755; by kind {'2hop': {'H1': 11, 'W': 1}, '3hop': {'H1': 27, 'W': 6}, '4hop': {'H1': 105, 'W': 27, 'H2': 3}}
- hotpotqa__s1sel: H1 via {'structural': 10}; F median dense rank 224; W median rank/B 7.53; by kind {'bridge': {'H1': 10, 'F': 30, 'W': 1, 'H2': 2, 'H3': 1}}
- hotpotqa__s1eval: H1 via {'structural': 37}; F median dense rank 269; W median rank/B 7.463; by kind {'bridge': {'F': 142, 'H3': 5, 'W': 14, 'H2': 9, 'H1': 37}, 'comparison': {'F': 1}}
- 2wiki__s1sel: H1 via {'structural': 34}; F median dense rank 589.5; W median rank/B 2.794; by kind {'bridge_comparison': {'H1': 15, 'F': 4, 'H3': 2, 'H2': 7, 'W': 5, 'X': 2}, 'compositional': {'H2': 11, 'F': 8, 'H1': 16, 'W': 2, 'H3': 3}, 'inference': {'H1': 3, 'H2': 2, 'W': 1, 'F': 1}}
- 2wiki__s1eval: H1 via {'structural': 214}; F median dense rank 282; W median rank/B 4.1845; by kind {'bridge_comparison': {'H1': 104, 'W': 77, 'H2': 88, 'H3': 26, 'F': 38, 'X': 4}, 'comparison': {'F': 1}, 'compositional': {'H1': 94, 'H2': 78, 'F': 108, 'H3': 31, 'W': 10, 'X': 3}, 'inference': {'H1': 16, 'F': 19, 'H3': 1, 'H2': 3, 'W': 7}}
- webqsp__s1eval: H1 via {'structural': 2012, 'knn': 291, 'ner': 44}; F median dense rank 0; W median rank/B 2.573; by kind {'': {'H1': 2026, 'W': 155, 'H2': 126, 'F': 3}}
