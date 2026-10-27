# A10e's screen verdict: zg2xe **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zg2xe's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A10e's, or not twelve reads (docs/A10E_G2_END_TO_END.md, section 4).

- **zg2xe** (zspx with G2's hidden-state layers over its score, trained end to end) against zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0001), L-musique webqsp WITHIN (-0.0037), L-hotpotqa metaqa WITHIN (-0.0012), L-hotpotqa webqsp WITHIN (-0.0080).
- **Next form** (fixed in section 5 before any number): A10p, the stack before the steps: zg2xe's stack (its z-score with the gradient kept) reading zspx's pre-step score (zspk's scorer and KT head), its bounded readout added there, so zspx's three steps act on the stack's correction (G4: depth pays when the step acts on the score); G2, A10 and A10e all put the stack after the steps; one declared file.

R@5 difference against zspx per read; beside, deciding nothing: zg2x's (A10, its stack's z-score detached) and the two deltas' difference (rounded deltas):

| read | zg2xe | zg2x (A10) | zg2xe - zg2x |
| --- | --- | --- | --- |
| L-musique metaqa | +0.0001 WITHIN | +0.0012 WITHIN | -0.0011 |
| L-musique squad | +0.0024 WITHIN | +0.0019 WITHIN | +0.0005 |
| L-musique musique | -0.0169 WITHIN | +0.0011 WITHIN | -0.0180 |
| L-musique hotpotqa | +0.0011 WITHIN | -0.0018 WITHIN | +0.0029 |
| L-musique 2wiki | -0.0054 WITHIN | -0.0056 WITHIN | +0.0002 |
| L-musique webqsp | -0.0037 WITHIN | +0.0050 WITHIN | -0.0087 |
| L-hotpotqa metaqa | -0.0012 WITHIN | +0.0001 WITHIN | -0.0013 |
| L-hotpotqa squad | +0.0008 WITHIN | +0.0013 WITHIN | -0.0005 |
| L-hotpotqa musique | -0.0041 WITHIN | -0.0033 WITHIN | -0.0008 |
| L-hotpotqa hotpotqa | -0.0010 WITHIN | -0.0014 WITHIN | +0.0004 |
| L-hotpotqa 2wiki | +0.0034 WITHIN | +0.0054 WITHIN | -0.0020 |
| L-hotpotqa webqsp | -0.0080 WITHIN | -0.0062 WITHIN | -0.0018 |
