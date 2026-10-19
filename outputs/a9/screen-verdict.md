# A9's screen verdict: zspxd **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zspxd's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A9's, or not twelve reads (docs/A9_EDGE_FAMILY_DROPOUT.md, section 4).

- **zspxd** (zspx trained with edge-family dropout) against zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0019), L-musique webqsp WITHIN (-0.0034), L-hotpotqa metaqa WITHIN (+0.0007), L-hotpotqa webqsp WITHIN (-0.0037).
- **Next form** (fixed in section 5 before any number): A9t, the MP-unified track's whole form: the dropout with zspx's three increments bounded together, s0 + 0.5 tanh((s3 - s0) / 0.5) (qd_gnn13's tb5, G2's bound); one declared file.

R@5 difference against zspx per read:

| read | zspxd |
| --- | --- |
| L-musique metaqa | -0.0019 WITHIN |
| L-musique squad | +0.0026 WITHIN |
| L-musique musique | +0.0024 WITHIN |
| L-musique hotpotqa | -0.0015 WITHIN |
| L-musique 2wiki | -0.0005 WITHIN |
| L-musique webqsp | -0.0034 WITHIN |
| L-hotpotqa metaqa | +0.0007 WITHIN |
| L-hotpotqa squad | -0.0008 WITHIN |
| L-hotpotqa musique | -0.0059 WITHIN |
| L-hotpotqa hotpotqa | -0.0075 WITHIN |
| L-hotpotqa 2wiki | +0.0026 WITHIN |
| L-hotpotqa webqsp | -0.0037 WITHIN |
