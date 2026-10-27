# A9b's screen verdict: zspxt **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zspxt's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A9b's, or not twelve reads (docs/A9B_BOUND_ALONE.md, section 4).

- **zspxt** (zspx with its three increments bounded together, no dropout) against zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp), after the seed-null re-call: 0 GAIN (none), 2 LOSS (L-hotpotqa musique, L-hotpotqa webqsp). Beside: L-musique metaqa WITHIN (-0.0012), L-musique webqsp WITHIN (-0.0054), L-hotpotqa metaqa WITHIN (-0.0008), L-hotpotqa webqsp LOSS (-0.0237).
- **Next form** (fixed in section 5 before any number): none of this track: the MP-unified track's form closes on the GNN. Its dropout (A9), its bound (A9b) and both together (A9t) have all reached the adopted GNN without a GAIN, and the MLP takes no message, so the idea has reached the one model that can take it (docs/SCREENS.md section 3b). This closing is the step recorded; no further form of this track is declared.

R@5 difference against zspx per read; zspxdt's (A9t, the dropout and the bound) and zspxd's (A9, the dropout alone) beside, deciding nothing:

| read | zspxt | zspxdt (A9t) | zspxd (A9) |
| --- | --- | --- | --- |
| L-musique metaqa | -0.0012 WITHIN | -0.0019 WITHIN | -0.0019 WITHIN |
| L-musique squad | +0.0024 WITHIN | +0.0035 WITHIN | +0.0026 WITHIN |
| L-musique musique | +0.0124 WITHIN | -0.0012 WITHIN | +0.0024 WITHIN |
| L-musique hotpotqa | -0.0004 WITHIN | -0.0002 WITHIN | -0.0015 WITHIN |
| L-musique 2wiki | -0.0023 WITHIN | -0.0010 WITHIN | -0.0005 WITHIN |
| L-musique webqsp | -0.0054 WITHIN | -0.0012 WITHIN | -0.0034 WITHIN |
| L-hotpotqa metaqa | -0.0008 WITHIN | -0.0023 WITHIN | +0.0007 WITHIN |
| L-hotpotqa squad | +0.0005 WITHIN | +0.0007 WITHIN | -0.0008 WITHIN |
| L-hotpotqa musique | -0.0116 LOSS | -0.0125 LOSS | -0.0059 WITHIN |
| L-hotpotqa hotpotqa | -0.0034 WITHIN | -0.0035 WITHIN | -0.0075 WITHIN |
| L-hotpotqa 2wiki | -0.0004 WITHIN | -0.0047 WITHIN | +0.0026 WITHIN |
| L-hotpotqa webqsp | -0.0237 LOSS | -0.0208 WITHIN | -0.0037 WITHIN |
