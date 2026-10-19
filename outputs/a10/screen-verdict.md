# A10's screen verdict: zg2x **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zg2x's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A10's, or not twelve reads (docs/A10_G2_ON_ZSPX.md, section 4).

- **zg2x** (zspx with G2's hidden-state layers over its score) against zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0012), L-musique webqsp WITHIN (+0.0050), L-hotpotqa metaqa WITHIN (+0.0001), L-hotpotqa webqsp WITHIN (-0.0062).
- **Next form** (fixed in section 5 before any number): A10e, the stack end to end: its z-score of zspx's score with the gradient kept (G4's zspkg, which found that depth pays when the step acts on the score), everything else G2's; one declared file.

R@5 difference against zspx per read:

| read | zg2x |
| --- | --- |
| L-musique metaqa | +0.0012 WITHIN |
| L-musique squad | +0.0019 WITHIN |
| L-musique musique | +0.0011 WITHIN |
| L-musique hotpotqa | -0.0018 WITHIN |
| L-musique 2wiki | -0.0056 WITHIN |
| L-musique webqsp | +0.0050 WITHIN |
| L-hotpotqa metaqa | +0.0001 WITHIN |
| L-hotpotqa squad | +0.0013 WITHIN |
| L-hotpotqa musique | -0.0033 WITHIN |
| L-hotpotqa hotpotqa | -0.0014 WITHIN |
| L-hotpotqa 2wiki | +0.0054 WITHIN |
| L-hotpotqa webqsp | -0.0062 WITHIN |
