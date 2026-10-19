# A9t's screen verdict: zspxdt **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zspxdt's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A9t's, or not twelve reads (docs/A9T_BOUNDED_DROPOUT.md, section 4).

- **zspxdt** (zspx trained with edge-family dropout, its three increments bounded together) against zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp), after the seed-null re-call: 0 GAIN (none), 1 LOSS (L-hotpotqa musique). Beside: L-musique metaqa WITHIN (-0.0019), L-musique webqsp WITHIN (-0.0012), L-hotpotqa metaqa WITHIN (-0.0023), L-hotpotqa webqsp WITHIN (-0.0208).
- **Next form** (fixed in section 5 before any number): A9b, the bound alone: zspx's three increments bounded together at 0.5 with no dropout, the one part of the track's form not yet on zspx; one declared file. If A9b finds no GAIN either, the MP-unified track's form closes on the GNN: its dropout, its bound and both together will have reached it, and the MLP takes no message.

R@5 difference against zspx per read; zspxd's (A9, the dropout alone) beside, deciding nothing:

| read | zspxdt | zspxd (A9) |
| --- | --- | --- |
| L-musique metaqa | -0.0019 WITHIN | -0.0019 WITHIN |
| L-musique squad | +0.0035 WITHIN | +0.0026 WITHIN |
| L-musique musique | -0.0012 WITHIN | +0.0024 WITHIN |
| L-musique hotpotqa | -0.0002 WITHIN | -0.0015 WITHIN |
| L-musique 2wiki | -0.0010 WITHIN | -0.0005 WITHIN |
| L-musique webqsp | -0.0012 WITHIN | -0.0034 WITHIN |
| L-hotpotqa metaqa | -0.0023 WITHIN | +0.0007 WITHIN |
| L-hotpotqa squad | +0.0007 WITHIN | -0.0008 WITHIN |
| L-hotpotqa musique | -0.0125 LOSS | -0.0059 WITHIN |
| L-hotpotqa hotpotqa | -0.0035 WITHIN | -0.0075 WITHIN |
| L-hotpotqa 2wiki | -0.0047 WITHIN | +0.0026 WITHIN |
| L-hotpotqa webqsp | -0.0208 WITHIN | -0.0037 WITHIN |
