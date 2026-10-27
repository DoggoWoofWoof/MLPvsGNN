# P5c5's screen verdict: zrckpo5 **NOT_ADOPTED**, zspko5 **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not P5c5's re-call of the arm, or not twelve reads (docs/P5C5_FIVE_FOLDS.md, section 4).

- **zrckpo5** (zrckp plus P5c's head over the four columns, g cross-fitted over five folds) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0006), L-musique webqsp WITHIN (-0.0076), L-hotpotqa metaqa WITHIN (-0.0008), L-hotpotqa webqsp WITHIN (-0.0120). Next: none: the learned offsets close as a fixed input to the lean pair, this file named as the close (docs/VERDICT_AUDIT_2026_10_17.md, section 6): they reached both models at two folds and at five, the GNN also with P5b's normalised log p, with no GAIN on any read.
- **zspko5** (zspk plus the head over the four five-fold columns and P5b's log p, before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 0 GAIN (none), 1 LOSS (L-musique webqsp). Beside: L-musique metaqa WITHIN (+0.0043), L-musique webqsp LOSS (-0.0142), L-hotpotqa metaqa WITHIN (+0.0020), L-hotpotqa webqsp WITHIN (+0.0046). Next: none: the learned offsets close as a fixed input to the lean pair, this file named as the close (docs/VERDICT_AUDIT_2026_10_17.md, section 6): they reached both models at two folds and at five, the GNN also with P5b's normalised log p, with no GAIN on any read.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckpo5 WITHIN (+0.0030), zspko5 WITHIN (+0.0053).

This verdict closes only this form (the five-fold columns through a pointwise head, and lp on the GNN); docs/SCREENS.md, 3b.
