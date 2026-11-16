# D84 report (declared in docs/D84_MULTIHOPRAG_LINK_ORACLE.md)

MultiHop-RAG revision 71ac0d0bd1f9; 609 articles; 2255 kept queries of 2556 (301 null dropped); gold counts {'2': 1169, '3': 774, '4': 312}.

**Call: CARRIED F0T OFF WIKIPEDIA**

| form | pairs | ratio to N_sp | density % | GP linked | GP share % [95%] | lift | lift call | GS share % | price |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 0.0 | 0.0 | 0 | 0.0 [0.0, 0.0] | None | NO_LIFT | 0.0 | - |
| N_sp | 21,817 | 1.0 | 11.78 | 4334 | 80.81 [79.35, 82.21] | 6.86 | POINTS | 77.07 | 5.0 |
| CN_all | 25,005 | 1.1461 | 13.51 | 4525 | 84.37 [83.11, 85.61] | 6.25 | POINTS | 79.91 | 5.5 |
| F0T | 25,080 | 1.1496 | 13.55 | 4541 | 84.67 [83.42, 85.91] | 6.25 | POINTS | 80.58 | 5.5 |

GP: 5363 slots over 2255 questions. GS: 2255 questions.

| minus N_sp | GP diff [95%] call | GS diff [95%] call |
|---|---|---|
| F0T | +3.86 [+3.1, +4.59] ABOVE | +3.5 [+2.3, +4.66] ABOVE |
| CN_all | +3.56 [+2.85, +4.27] ABOVE | +2.84 [+1.69, +3.9] ABOVE |

Beside, by question type and gold count (GP share N_sp, F0T; F0T minus N_sp; GS share N_sp / F0T):

| group | questions | slots | N_sp GP % | F0T GP % | F0T - N_sp | GS N_sp / F0T % |
|---|---|---|---|---|---|---|
| comparison_query | 856 | 1212 | 73.43 | 75.33 | +1.9 [+0.33, +3.49] ABOVE | 68.46 / 70.09 |
| inference_query | 816 | 3094 | 83.68 | 87.88 | +4.2 [+3.18, +5.2] ABOVE | 82.72 / 88.73 |
| temporal_query | 583 | 1057 | 80.89 | 86.0 | +5.11 [+3.43, +6.76] ABOVE | 81.82 / 84.56 |
| 2 golds | 1169 | 1169 | 72.11 | 73.48 | +1.37 [-0.17, +2.99] LEVEL | 72.11 / 73.48 |
| 3 golds | 774 | 2322 | 79.8 | 86.09 | +6.29 [+5.17, +7.49] ABOVE | 79.46 / 86.56 |
| 4 golds | 312 | 1872 | 87.5 | 89.9 | +2.4 [+1.23, +3.58] ABOVE | 89.74 / 92.31 |

Rule build: 26,532 name keys; 44,495 name entries; 233 title-run entries added on 163 articles; LT 25,077 pairs. spaCy entities: 21,213.
Script sha256 dac9b0035300; 245.1s.
