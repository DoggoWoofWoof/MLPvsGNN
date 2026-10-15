# G2's screen verdict: **NOT_ADOPTED**

Rule: SCREEN_PASSED when zg2's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when the re-call is missing, not G2's, or not twelve reads. A pass is fitted again on P1f's final pools before anything is adopted (docs/G2_HIDDEN_STATE_GNN.md, section 4).

zg2 (zspk plus a 3-layer, 64-wide stack over the encoder's hidden states) against zspk, after the seed-null re-call: 0 GAIN (none), 0 LOSS (none).

Beside, deciding nothing.

The twin (zg2t, the same stack with no edges) against zspk, after its re-call: 0 GAIN (none), 1 LOSS (L-musique webqsp).

zg2 minus zg2t (what the stack's message passing adds), R@5 / FC@5 / hit@1, with the 95% question-bootstrap interval of the R@5 difference:

| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075 floor) |
|---|---|---|---|---|---|---|
| L-musique | metaqa | in-domain | -0.0013 [-0.0022, -0.0005] | -0.0017 | -0.0042 | WITHIN |
| L-musique | squad | in-domain | -0.0003 [-0.0023, +0.0016] | -0.0003 | +0.0049 | WITHIN |
| L-musique | musique | zero-shot | -0.0141 [-0.0194, -0.0089] | -0.0108 | -0.0079 | LOSS |
| L-musique | hotpotqa | in-domain | -0.0004 [-0.0024, +0.0016] | +0.0003 | -0.0061 | WITHIN |
| L-musique | 2wiki | in-domain | -0.0034 [-0.0048, -0.0019] | -0.0103 | +0.0096 | WITHIN |
| L-musique | webqsp | zero-shot | +0.0131 [+0.0048, +0.0214] | +0.0133 | +0.0140 | GAIN |
| L-hotpotqa | metaqa | in-domain | -0.0028 [-0.0037, -0.0019] | -0.0026 | -0.0039 | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0020 [+0.0002, +0.0040] | +0.0020 | +0.0018 | WITHIN |
| L-hotpotqa | musique | in-domain | -0.0034 [-0.0088, +0.0018] | -0.0058 | +0.0054 | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | +0.0068 [+0.0038, +0.0097] | +0.0099 | +0.0113 | WITHIN |
| L-hotpotqa | 2wiki | in-domain | -0.0078 [-0.0094, -0.0063] | -0.0233 | +0.0003 | LOSS |
| L-hotpotqa | webqsp | zero-shot | +0.0084 [-0.0000, +0.0171] | +0.0060 | +0.0073 | WITHIN |
