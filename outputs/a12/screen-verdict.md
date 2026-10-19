# A12's screen verdict: zspkgs **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zspkgs's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A12's, or not twelve reads (docs/A12_GSURG_GNN.md, section 4).

- **zspkgs** (zspk trained with gsurg's loop) against zspk's card fit of each split under lean_gpu's loop (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 0 GAIN (none), 1 LOSS (L-musique webqsp). Beside: L-musique metaqa WITHIN (+0.0016), L-musique webqsp LOSS (-0.0137), L-hotpotqa metaqa WITHIN (-0.0003), L-hotpotqa webqsp WITHIN (+0.0040).
- **Next form** (fixed in section 5 before any number): A12x, gsurg's loop on zspx (the adopted GNN, three steps, where depth paid); one declared file.

R@5 difference against zspk per read:

| read | zspkgs |
| --- | --- |
| L-musique metaqa | +0.0016 WITHIN |
| L-musique squad | +0.0006 WITHIN |
| L-musique musique | -0.0190 WITHIN |
| L-musique hotpotqa | +0.0010 WITHIN |
| L-musique 2wiki | -0.0014 WITHIN |
| L-musique webqsp | -0.0137 LOSS |
| L-hotpotqa metaqa | -0.0003 WITHIN |
| L-hotpotqa squad | -0.0034 WITHIN |
| L-hotpotqa musique | +0.0007 WITHIN |
| L-hotpotqa hotpotqa | +0.0019 WITHIN |
| L-hotpotqa 2wiki | +0.0023 WITHIN |
| L-hotpotqa webqsp | +0.0040 WITHIN |
