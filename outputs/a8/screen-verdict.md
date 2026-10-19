# A8's screen verdict: zrckpq **NOT_ADOPTED**, zspkq **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A8's re-call of the arm, or not twelve reads (docs/A8_QDEPTH_INPUT.md, section 4).

- **zrckpq** (zrckp plus the A8 head) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0003), L-musique webqsp WITHIN (-0.0066), L-hotpotqa metaqa WITHIN (+0.0001), L-hotpotqa webqsp WITHIN (-0.0012). Next: A8e, round 11's own form on both lean models: the prior l_q[depth(v)] added to each row's score (qdepth.QDepth's U, W, b at zero), trained by each model's listwise loss, on zrckp and on zspk before zsp's propagation reads the score; one declared file.
- **zspkq** (zspk plus the A8 head before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0006), L-musique webqsp WITHIN (-0.0022), L-hotpotqa metaqa WITHIN (+0.0009), L-hotpotqa webqsp WITHIN (+0.0101). Next: A8e, round 11's own form on both lean models: the prior l_q[depth(v)] added to each row's score (qdepth.QDepth's U, W, b at zero), trained by each model's listwise loss, on zrckp and on zspk before zsp's propagation reads the score; one declared file.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckpq WITHIN (-0.0007), zspkq WITHIN (+0.0011).

This verdict closes only this form (the depth prior's six columns through a pointwise head); docs/SCREENS.md, 3b.
