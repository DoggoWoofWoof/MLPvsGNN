# R1's screen verdict: zrckpr **ADOPT_FOR_FULL_RUN**, zspkr **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not R1's re-call of the arm, or not twelve reads (docs/R1_RESIDUAL_INPUT.md, section 4).

- **zrckpr** (zrckp plus the R1 head) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 1 GAIN (L-musique webqsp), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0007), L-musique webqsp GAIN (+0.0135), L-hotpotqa metaqa WITHIN (-0.0004), L-hotpotqa webqsp WITHIN (-0.0064).
- **zspkr** (zspk plus the R1 head before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0017), L-musique webqsp WITHIN (-0.0005), L-hotpotqa metaqa WITHIN (-0.0016), L-hotpotqa webqsp WITHIN (+0.0078).

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckpr WITHIN (+0.0034), zspkr WITHIN (+0.0011).
