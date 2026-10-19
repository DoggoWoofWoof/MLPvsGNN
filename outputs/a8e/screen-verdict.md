# A8e's screen verdict: zrckpqe **NOT_ADOPTED**, zspkqe **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A8e's re-call of the arm, or not twelve reads (docs/A8E_QDEPTH_PRIOR.md, section 4).

- **zrckpqe** (zrckp plus round 11's depth prior) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0000), L-musique webqsp WITHIN (+0.0024), L-hotpotqa metaqa WITHIN (+0.0002), L-hotpotqa webqsp WITHIN (+0.0030). Next: A8ex, the same prior on the adopted arms (G5's zspx, added before its propagation reads the score, and R1's zrckpr, added to its final score), on both models, its own declared file; the depth prior closes as a model term only if A8ex finds no GAIN either.
- **zspkqe** (zspk plus round 11's depth prior before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0008), L-musique webqsp WITHIN (-0.0007), L-hotpotqa metaqa WITHIN (-0.0008), L-hotpotqa webqsp WITHIN (+0.0133). Next: A8ex, the same prior on the adopted arms (G5's zspx, added before its propagation reads the score, and R1's zrckpr, added to its final score), on both models, its own declared file; the depth prior closes as a model term only if A8ex finds no GAIN either.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckpqe WITHIN (+0.0002), zspkqe WITHIN (-0.0072).

This verdict closes only this form (round 11's prior inside each model, trained by its listwise loss); docs/SCREENS.md, 3b.
