# B1L-U report (declared in docs/B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md, section 15)

R@5 x100 of the p@swa candidate on HippoRAG 2's setting (B1: 1,000 questions), no new fit. The two models of record, zg6r and zrckpn, fitted without LLM links, read on four link sets: b1t (the text links only), b1w (the rule's W2_S links, W2_S-typed), b1o (the text links plus the P pairs of the setting's LLM-built reference, mostly NULL-typed) and b1u (the text links plus the union of the rule's pairs and the reference's: the rule's entries W2_S-typed, the reference's own NULL). b1u - b1w is what the reference's pair set adds to the rule's graph; b1u - b1o is what the rule's pairs add to the reference's. Nothing here says more than these twelve fits on these four graphs.

## In-domain

| setting | model | b1t | b1w | b1o | b1u | b1u - b1w | b1u - b1o | b1u - b1t | words (u-w / u-o / u-t) |
|---|---|---:|---:|---:|---:|---|---|---|---|
| musique | zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | 67.4 (65.8-69.3) | 68.8 (67.1-70.6) | -0.1 (-1.1, +0.9) | +1.4 (+0.8, +2.0) | +2.9 (+1.8, +3.9) | FLAT / UP / UP |
| musique | zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | 67.3 (65.6-69.2) | 67.7 (66.0-69.5) | +0.3 (-0.7, +1.2) | +0.4 (-0.1, +0.9) | +3.2 (+2.1, +4.4) | FLAT / FLAT / UP |

## Held-out

| setting | model | b1t | b1w | b1o | b1u | b1u - b1w | b1u - b1o | b1u - b1t | words (u-w / u-o / u-t) |
|---|---|---:|---:|---:|---:|---|---|---|---|
| musique | zg6r | 62.1 (60.4-63.7) | 60.8 (59.1-62.5) | 62.8 (61.1-64.5) | 63.5 (61.7-65.2) | +2.7 (+1.4, +4.0) | +0.7 (-0.1, +1.4) | +1.4 (+0.1, +2.6) | UP / FLAT / UP |
| musique | zrckpn | 60.0 (58.4-61.7) | 61.8 (60.1-63.6) | 64.0 (62.3-65.8) | 65.0 (63.3-66.8) | +3.2 (+2.0, +4.4) | +1.1 (+0.4, +1.7) | +5.0 (+3.8, +6.3) | UP / UP / UP |

## The call per setting (in-domain, both models)

| setting | b1u - b1w | b1u - b1o |
|---|---|---|
| musique | NOTHING | NOT_DOWN |

Missing reads: 0.
