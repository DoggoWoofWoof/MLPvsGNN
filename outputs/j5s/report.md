# J5S report (declared in docs/J5S_J5_SEEDS.md)

R@5 x100, candidate p@swa, 1,000 questions per setting. Three seeds: each question's R@5 is averaged over seeds 0, 1 and 2, and the 95% interval is B1b's bootstrap over the questions. Seed 0 is this stage's re-read of the filed fit. b1 = the corpus hyperlinks (musique: its title links), b1w = W2_S's rule links, b1t = text links only, b1u = U1a's title rule. Calls: ABOVE when the interval's low end is above the reference, BELOW when its high end is below, AT otherwise.

## Controls

- **zrc** C1 (seed-0 control fit against the filed fit, tensor by tensor): REPRODUCED; 9 states compared, differing [], fields []; curve {'epochs': [8, 8], 'same_steps': True, 'max_abs_loss_delta': 0.0}
  - C2 (seed-0 re-read = filed read): 12 of 12 EQUAL 
  - C3 (control's b1 read = filed read): 3 of 3 EQUAL 
- **zsp** C1 (seed-0 control fit against the filed fit, tensor by tensor): REPRODUCED; 9 states compared, differing [], fields []; curve {'epochs': [8, 8], 'same_steps': True, 'max_abs_loss_delta': 0.0}
  - C2 (seed-0 re-read = filed read): 12 of 12 EQUAL 
  - C3 (control's b1 read = filed read): 3 of 3 EQUAL 

## The rows on each carve

| setting | carve | arm | s0 | s1 | s2 | mean [95% CI] | spread | vs HippoRAG 2 (s0 / 3 seeds) | calls changed |
|---|---|---|---:|---:|---:|---|---:|---|---|
| musique | b1 | zrc | 66.1 | 66.1 | 66.6 | 66.3 [64.5, 68.0] | 0.5 | BELOW / BELOW | none |
| musique | b1 | zsp | 66.3 | 65.5 | 66.9 | 66.2 [64.5, 68.0] | 1.3 | BELOW / BELOW | none |
| musique | b1w | zrc | 65.3 | 65.1 | 66.2 | 65.5 [63.9, 67.2] | 1.1 | BELOW / BELOW | none |
| musique | b1w | zsp | 65.7 | 65.6 | 65.7 | 65.7 [63.9, 67.5] | 0.1 | BELOW / BELOW | none |
| musique | b1t | zrc | 63.4 | 62.8 | 64.1 | 63.4 [61.7, 65.2] | 1.3 | BELOW / BELOW | none |
| musique | b1t | zsp | 62.9 | 63.1 | 63.7 | 63.2 [61.6, 65.0] | 0.8 | BELOW / BELOW | none |
| musique | b1u | zrc | 66.1 | 66.1 | 66.6 | 66.3 [64.5, 68.0] | 0.5 | BELOW / BELOW | none |
| musique | b1u | zsp | 66.3 | 65.5 | 66.9 | 66.2 [64.5, 68.0] | 1.3 | BELOW / BELOW | none |
| 2wiki | b1 | zrc | 96.4 | 96.4 | 96.0 | 96.3 [95.6, 96.9] | 0.3 | ABOVE / ABOVE | none |
| 2wiki | b1 | zsp | 97.2 | 97.0 | 97.5 | 97.2 [96.5, 97.8] | 0.5 | ABOVE / ABOVE | none |
| 2wiki | b1w | zrc | 94.2 | 93.9 | 93.6 | 93.9 [93.0, 94.7] | 0.6 | ABOVE / ABOVE | none |
| 2wiki | b1w | zsp | 95.2 | 94.8 | 95.6 | 95.2 [94.4, 96.0] | 0.8 | ABOVE / ABOVE | none |
| 2wiki | b1t | zrc | 75.6 | 75.8 | 76.0 | 75.8 [74.3, 77.3] | 0.4 | BELOW / BELOW | none |
| 2wiki | b1t | zsp | 77.3 | 77.3 | 77.9 | 77.5 [76.0, 79.0] | 0.6 | BELOW / BELOW | none |
| 2wiki | b1u | zrc | 95.3 | 95.3 | 95.0 | 95.2 [94.4, 96.0] | 0.4 | ABOVE / ABOVE | none |
| 2wiki | b1u | zsp | 95.9 | 96.0 | 96.2 | 96.0 [95.3, 96.7] | 0.4 | ABOVE / ABOVE | none |
| hotpotqa | b1 | zrc | 98.2 | 98.0 | 97.9 | 98.0 [97.4, 98.6] | 0.2 | ABOVE / ABOVE | PropRAG: ABOVE -> AT; BridgeRAG: AT -> BELOW |
| hotpotqa | b1 | zsp | 98.2 | 97.9 | 98.1 | 98.1 [97.4, 98.7] | 0.4 | ABOVE / ABOVE | PropRAG: ABOVE -> AT; BridgeRAG: AT -> BELOW |
| hotpotqa | b1w | zrc | 94.8 | 94.3 | 94.2 | 94.5 [93.5, 95.4] | 0.5 | BELOW / BELOW | HGRAG: AT -> BELOW |
| hotpotqa | b1w | zsp | 94.3 | 94.3 | 94.7 | 94.4 [93.5, 95.4] | 0.4 | BELOW / BELOW | none |
| hotpotqa | b1t | zrc | 88.7 | 88.4 | 88.7 | 88.6 [87.4, 89.9] | 0.2 | BELOW / BELOW | none |
| hotpotqa | b1t | zsp | 90.0 | 89.5 | 90.0 | 89.8 [88.6, 91.1] | 0.4 | BELOW / BELOW | none |
| hotpotqa | b1u | zrc | 94.8 | 94.5 | 94.5 | 94.6 [93.6, 95.6] | 0.3 | BELOW / BELOW | none |
| hotpotqa | b1u | zsp | 95.0 | 94.7 | 94.8 | 94.8 [93.9, 95.8] | 0.3 | BELOW / BELOW | none |

## Against each published row (three seeds, by the interval)

- musique b1 zrc: HippoRAG 2 BELOW, G-reasoner BELOW, CoopRAG BELOW, PropRAG BELOW, HGRAG BELOW, EviReform BELOW, SAG2 BELOW
- musique b1 zsp: HippoRAG 2 BELOW, G-reasoner BELOW, CoopRAG BELOW, PropRAG BELOW, HGRAG BELOW, EviReform BELOW, SAG2 BELOW
- 2wiki b1 zrc: HippoRAG 2 ABOVE, G-reasoner BELOW, PROPEX-RAG BELOW, CoopRAG BELOW, PropRAG ABOVE, HGRAG ABOVE, EviReform BELOW, BridgeRAG ABOVE, SAG2 BELOW
- 2wiki b1 zsp: HippoRAG 2 ABOVE, G-reasoner BELOW, PROPEX-RAG BELOW, CoopRAG AT, PropRAG ABOVE, HGRAG ABOVE, EviReform BELOW, BridgeRAG ABOVE, SAG2 AT
- hotpotqa b1 zrc: HippoRAG 2 ABOVE, PROPEX-RAG ABOVE, CoopRAG ABOVE, PropRAG AT, HGRAG ABOVE, EviReform ABOVE, BridgeRAG BELOW, SAG2 AT
- hotpotqa b1 zsp: HippoRAG 2 ABOVE, PROPEX-RAG ABOVE, CoopRAG ABOVE, PropRAG AT, HGRAG ABOVE, EviReform ABOVE, BridgeRAG BELOW, SAG2 AT

## Paired differences (three-seed means, per question)

| setting | difference | R@5 | 95% CI |
|---|---|---:|---|
| musique | b1-b1t zrc | +2.8 | [+1.9, +3.7] |
| musique | b1-b1t zsp | +3.0 | [+2.0, +4.0] |
| musique | b1w-b1t zrc | +2.1 | [+1.3, +2.9] |
| musique | b1w-b1t zsp | +2.4 | [+1.5, +3.4] |
| musique | b1u-b1t zrc | +2.8 | [+1.9, +3.7] |
| musique | b1u-b1t zsp | +3.0 | [+2.0, +4.0] |
| musique | b1-b1w zrc | +0.7 | [+0.1, +1.3] |
| musique | b1-b1w zsp | +0.6 | [-0.1, +1.3] |
| musique | b1w-b1u zrc | -0.7 | [-1.3, -0.1] |
| musique | b1w-b1u zsp | -0.6 | [-1.3, +0.1] |
| musique | zsp-zrc on b1 | -0.0 | [-0.6, +0.6] |
| musique | zsp-zrc on b1w | +0.1 | [-0.5, +0.7] |
| musique | zsp-zrc on b1t | -0.2 | [-0.7, +0.3] |
| musique | zsp-zrc on b1u | -0.0 | [-0.6, +0.6] |
| musique | share of b1-b1t, zrc | b1u 1.0, b1w 0.75 | |
| musique | share of b1-b1t, zsp | b1u 1.0, b1w 0.814 | |
| musique | (zrc - RRF) / (zsp - RRF) on b1 | 1.002 | |
| 2wiki | b1-b1t zrc | +20.5 | [+19.1, +21.9] |
| 2wiki | b1-b1t zsp | +19.7 | [+18.3, +21.1] |
| 2wiki | b1w-b1t zrc | +18.1 | [+16.8, +19.4] |
| 2wiki | b1w-b1t zsp | +17.7 | [+16.3, +19.1] |
| 2wiki | b1u-b1t zrc | +19.4 | [+18.0, +20.8] |
| 2wiki | b1u-b1t zsp | +18.5 | [+17.1, +20.0] |
| 2wiki | b1-b1w zrc | +2.4 | [+1.6, +3.2] |
| 2wiki | b1-b1w zsp | +2.0 | [+1.2, +2.8] |
| 2wiki | b1w-b1u zrc | -1.3 | [-1.7, -0.9] |
| 2wiki | b1w-b1u zsp | -0.8 | [-1.1, -0.5] |
| 2wiki | zsp-zrc on b1 | +0.9 | [+0.7, +1.2] |
| 2wiki | zsp-zrc on b1w | +1.3 | [+0.9, +1.7] |
| 2wiki | zsp-zrc on b1t | +1.7 | [+1.2, +2.2] |
| 2wiki | zsp-zrc on b1u | +0.8 | [+0.5, +1.1] |
| 2wiki | share of b1-b1t, zrc | b1u 0.948, b1w 0.884 | |
| 2wiki | share of b1-b1t, zsp | b1u 0.939, b1w 0.898 | |
| 2wiki | (zrc - RRF) / (zsp - RRF) on b1 | 0.963 | |
| hotpotqa | b1-b1t zrc | +9.4 | [+8.2, +10.5] |
| hotpotqa | b1-b1t zsp | +8.2 | [+7.1, +9.3] |
| hotpotqa | b1w-b1t zrc | +5.9 | [+4.8, +6.9] |
| hotpotqa | b1w-b1t zsp | +4.6 | [+3.6, +5.5] |
| hotpotqa | b1u-b1t zrc | +6.0 | [+5.0, +7.0] |
| hotpotqa | b1u-b1t zsp | +5.0 | [+4.0, +5.9] |
| hotpotqa | b1-b1w zrc | +3.5 | [+2.7, +4.3] |
| hotpotqa | b1-b1w zsp | +3.6 | [+2.9, +4.4] |
| hotpotqa | b1w-b1u zrc | -0.1 | [-0.6, +0.3] |
| hotpotqa | b1w-b1u zsp | -0.4 | [-0.8, +0.0] |
| hotpotqa | zsp-zrc on b1 | +0.1 | [-0.2, +0.4] |
| hotpotqa | zsp-zrc on b1w | -0.0 | [-0.4, +0.3] |
| hotpotqa | zsp-zrc on b1t | +1.2 | [+0.7, +1.8] |
| hotpotqa | zsp-zrc on b1u | +0.2 | [-0.1, +0.5] |
| hotpotqa | share of b1-b1t, zrc | b1u 0.639, b1w 0.623 | |
| hotpotqa | share of b1-b1t, zsp | b1u 0.605, b1w 0.559 | |
| hotpotqa | (zrc - RRF) / (zsp - RRF) on b1 | 0.995 | |

Missing: 0 []
