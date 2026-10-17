# G4's screen verdict: zspk2 **ADOPT_FOR_FULL_RUN**, zspkg **ADOPT_FOR_FULL_RUN**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not G4's re-call of the arm, or not twelve reads (docs/G4_DEEPER_PROPAGATION.md, section 3).

- **zspk2** (zspk plus a second propagation step) against zspk's card fit of each split under lean_gpu's loss (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 1 GAIN (L-hotpotqa musique), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0012), L-musique webqsp WITHIN (-0.0081), L-hotpotqa metaqa WITHIN (+0.0000), L-hotpotqa webqsp WITHIN (+0.0030).
- **zspkg** (zspk with its propagation trained end to end) against zspk's card fit of each split under lean_gpu's loss (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 1 GAIN (L-hotpotqa musique), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0006), L-musique webqsp WITHIN (-0.0067), L-hotpotqa metaqa WITHIN (+0.0004), L-hotpotqa webqsp WITHIN (+0.0152).
