# B1f report (declared in docs/B1F_W2S_ON_B1.md)

R@5 x100, candidate p@swa; b1 = hyperlinks (musique: title links), b1t = text only, b1u = U1a's title rule, b1w = W2_S.

| setting | read | b1 | b1t | b1u | b1w | w-u (CI) | HippoRAG 2 |
|---|---|---:|---:|---:|---:|---|---:|
| musique | zrc/J5 | 66.1 | 63.4 | 66.1 | 65.3 | -0.8 (-1.5, +0.0) | 74.7 |
| musique | zrc/L-musique | 61.3 | 58.4 | 61.3 | 61.6 | +0.3 (-0.7, +1.2) | 74.7 |
| musique | zsp/J5 | 66.3 | 62.9 | 66.3 | 65.7 | -0.6 (-1.4, +0.2) | 74.7 |
| musique | zsp/L-musique | 60.2 | 57.8 | 60.2 | 61.3 | +1.0 (+0.1, +2.0) | 74.7 |
| 2wiki | zrc/J5 | 96.4 | 75.6 | 95.3 | 94.2 | -1.1 (-1.6, -0.7) | 90.4 |
| 2wiki | zrc/L-2wiki | 93.3 | 75.6 | 92.2 | 89.8 | -2.3 (-2.9, -1.8) | 90.4 |
| 2wiki | zsp/J5 | 97.2 | 77.3 | 95.9 | 95.2 | -0.6 (-1.0, -0.3) | 90.4 |
| 2wiki | zsp/L-2wiki | 93.4 | 77.1 | 92.2 | 90.5 | -1.7 (-2.2, -1.2) | 90.4 |
| hotpotqa | zrc/J5 | 98.2 | 88.7 | 94.8 | 94.8 | +0.0 (-0.4, +0.5) | 96.3 |
| hotpotqa | zrc/L-hotpotqa | 96.5 | 88.5 | 94.2 | 92.3 | -1.9 (-2.7, -1.2) | 96.3 |
| hotpotqa | zsp/J5 | 98.2 | 90.0 | 95.0 | 94.3 | -0.6 (-1.1, -0.1) | 96.3 |
| hotpotqa | zsp/L-hotpotqa | 95.9 | 89.1 | 93.8 | 91.9 | -1.8 (-2.6, -1.1) | 96.3 |

In-domain reads (J5) against the best published row of the setting, by the 95% interval of R@5 (ABOVE, AT or BELOW the row's value).

| setting | best row (any) | best peer-reviewed row | read | b1 | b1u | b1w | calls (any/peer) for b1, b1u, b1w |
|---|---|---|---|---:|---:|---:|---|
| musique | SAG2 80.7 | PropRAG 78.3 | zrc/J5 | 66.1 | 66.1 | 65.3 | BELOW/BELOW, BELOW/BELOW, BELOW/BELOW |
| musique | SAG2 80.7 | PropRAG 78.3 | zsp/J5 | 66.3 | 66.3 | 65.7 | BELOW/BELOW, BELOW/BELOW, BELOW/BELOW |
| 2wiki | EviReform 98.5 | G-reasoner 98.2 | zrc/J5 | 96.4 | 95.3 | 94.2 | BELOW/BELOW, BELOW/BELOW, BELOW/BELOW |
| 2wiki | EviReform 98.5 | G-reasoner 98.2 | zsp/J5 | 97.2 | 95.9 | 95.2 | BELOW/BELOW, BELOW/BELOW, BELOW/BELOW |
| hotpotqa | BridgeRAG 98.8 | PropRAG 97.4 | zrc/J5 | 98.2 | 94.8 | 94.8 | AT/ABOVE, BELOW/BELOW, BELOW/BELOW |
| hotpotqa | BridgeRAG 98.8 | PropRAG 97.4 | zsp/J5 | 98.2 | 95.0 | 94.3 | AT/ABOVE, BELOW/BELOW, BELOW/BELOW |
