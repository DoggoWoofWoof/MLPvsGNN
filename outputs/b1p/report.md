# B1P report (declared in docs/B1P_PKO_CHAIN_ON_B1.md)

R@5 x100 of the p@swa candidate on HippoRAG 2's settings (1,000 questions each), no LLM and no hyperlink: b1w = U2b's W2_S rule cut to the setting, b1t = the setting's text links only. In-domain = the seed mean (and split mean on 2wiki) of the fits that trained on the dataset; 95% bootstrap intervals over the questions; calls against the best published row (any / peer-reviewed): ABOVE, AT or BELOW.

| setting | read | arm | b1t | b1w | w - t (CI) | best (any) | best (peer) | calls b1w (any/peer) |
|---|---|---|---:|---:|---|---|---|---|
| musique | in-domain | zg6r | 66.0 | 69.0 (67.2-70.7) | +3.0 (+1.8, +4.0) | SAG2 80.68 | PropRAG 78.3 | BELOW/BELOW |
| musique | in-domain | zrckp | 62.5 | 66.9 (65.2-68.6) | +4.4 (+3.3, +5.6) | SAG2 80.68 | PropRAG 78.3 | BELOW/BELOW |
| musique | zero-shot | zg6r | 62.1 | 60.8 (59.1-62.5) | -1.3 (-2.6, +0.1) | SAG2 80.68 | PropRAG 78.3 | -- |
| musique | zero-shot | zrckp | 58.6 | 60.9 (59.2-62.6) | +2.3 (+0.9, +3.5) | SAG2 80.68 | PropRAG 78.3 | -- |
| 2wiki | in-domain | zg6r | 78.1 | 94.9 (94.1-95.7) | +16.8 (+15.5, +18.2) | EviReform 98.45 | G-reasoner 98.2 | BELOW/BELOW |
| 2wiki | in-domain | zrckp | 75.9 | 94.0 (93.1-94.8) | +18.1 (+16.8, +19.4) | EviReform 98.45 | G-reasoner 98.2 | BELOW/BELOW |
| hotpotqa | in-domain | zg6r | 91.1 | 94.3 (93.3-95.3) | +3.1 (+2.4, +4.0) | BridgeRAG 98.75 | PropRAG 97.4 | BELOW/BELOW |
| hotpotqa | in-domain | zrckp | 90.0 | 93.4 (92.4-94.5) | +3.4 (+2.5, +4.3) | BridgeRAG 98.75 | PropRAG 97.4 | BELOW/BELOW |
| hotpotqa | zero-shot | zg6r | 89.4 | 92.9 (91.9-94.0) | +3.5 (+2.6, +4.4) | BridgeRAG 98.75 | PropRAG 97.4 | -- |
| hotpotqa | zero-shot | zrckp | 87.9 | 92.2 (91.2-93.3) | +4.3 (+3.4, +5.2) | BridgeRAG 98.75 | PropRAG 97.4 | -- |

Setting questions found in the in-domain splits' training carves (PKO's KT ids): musique {'WQ-hotpotqa:musique/s1fit': 0}; 2wiki {'WQ-musique:2wiki/fit': 0, 'WQ-hotpotqa:2wiki/fit': 0}; hotpotqa {'WQ-musique:hotpotqa/fit': 0}

Missing reads: 0.
