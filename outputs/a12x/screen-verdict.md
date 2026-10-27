# A12x's screen verdict: zspxgs **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zspxgs's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A12x's, or not twelve reads (docs/A12X_GSURG_ZSPX.md, section 4).

- **zspxgs** (zspx trained with gsurg's loop) against zspx's card fit of each split under lean_gpu's loop (scr-g5-zspx, scr-g5-zspx-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0012), L-musique webqsp WITHIN (-0.0004), L-hotpotqa metaqa WITHIN (-0.0014), L-hotpotqa webqsp WITHIN (-0.0094).
- **Next form** (fixed in section 5 before any number): A12m, gsurg's loop on zrckp (the MLP's lean model): the loop's only GAINs came on the MLP's line (zrg on zrm: webqsp zero-shot +0.0273, musique +0.0087) and the MLP's model has changed since; one declared file, whose own section 5 names what follows it.

R@5 difference against zspx per read:

| read | zspxgs |
| --- | --- |
| L-musique metaqa | +0.0012 WITHIN |
| L-musique squad | -0.0003 WITHIN |
| L-musique musique | -0.0133 WITHIN |
| L-musique hotpotqa | +0.0027 WITHIN |
| L-musique 2wiki | -0.0010 WITHIN |
| L-musique webqsp | -0.0004 WITHIN |
| L-hotpotqa metaqa | -0.0014 WITHIN |
| L-hotpotqa squad | -0.0001 WITHIN |
| L-hotpotqa musique | -0.0050 WITHIN |
| L-hotpotqa hotpotqa | -0.0033 WITHIN |
| L-hotpotqa 2wiki | +0.0051 WITHIN |
| L-hotpotqa webqsp | -0.0094 WITHIN |
