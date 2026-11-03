# LT2: zrckp2 (zrckp + LT2's next-hop columns (NEXT within two links: each anchor's learned next-hop term on the pool rows within two links of it outside the first stage's top five, with the two links' families, the middle rows' count and the best middle row's rrf rank as inputs, cross-fitted; uniform negatives)) against LTM's zrckpn, on PKO's pools, on the host's CPU

## Screen: NO_GAIN

counts: {"GAIN": 0, "LOSS": 0, "WITHIN": 10}

| split | read | zrckpn s0, s1, s2 | zrckp2 s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | squad | 0.9138, 0.9125, 0.9149 | 0.9140 | +0.0002 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8460, 0.8469, 0.8491 | 0.8466 | +0.0006 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9155, 0.9159, 0.9133 | 0.9182 | +0.0027 | 0.0075 | WITHIN |
| WQ-musique | webqsp | 0.6457, 0.6498, 0.6552 | 0.6523 | +0.0066 | 0.0146 | WITHIN |
| WQ-hotpotqa | squad | 0.9132, 0.9120, 0.9132 | 0.9136 | +0.0004 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5806, 0.5803, 0.5777 | 0.5806 | +0.0000 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9135, 0.9166, 0.9135 | 0.9130 | -0.0005 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6469, 0.6597, 0.6512 | 0.6609 | +0.0140 | 0.0191 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5307 | -0.0044 | 0.0127 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3580 | +0.0006 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5190, 0.5227, 0.5381 | 0.5153 | -0.0037 | 0.0275 | WITHIN (beside) |
| WQ-musique | metaqa | 0.4126, 0.4256, 0.4167 | 0.4176 | +0.0050 | 0.0193 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8193, 0.8182, 0.8136 | 0.8147 | -0.0046 | 0.0082 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3668, 0.4261, 0.4131 | 0.3954 | +0.0286 | 0.1064 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4544, 0.4596, 0.4702 | 0.4425 | -0.0118 | 0.0235 | WITHIN (beside) |
| WQ-musique | musique-4hop | 0.3086, 0.3148, 0.3352 | 0.2994 | -0.0093 | 0.0385 | WITHIN (beside) |

next (section 7): the two-link lists close; next form: the path's vectors as inputs (each two-link row's middle row nearest the question, with its cosines to the question, the anchor and the row, beside LT2's nine) (LT2's scorer does not rank the two-link targets above rrf on WQ-hotpotqa's musique/s1eval)
section 7's read (WQ-hotpotqa musique/s1eval): two-link targets' T2 under LT2's scorer 0.082, under rrf among the same rows 0.099

Beside, deciding nothing: the A cases' T2 under bil. LT2's scorer ranks among the rows within two links of the anchor (targets beyond them last; linked and two-link targets apart; rrf among the same rows); LTH's (ranked negatives) and LTA's (uniform negatives) among every row outside the top five; LTM's among linked rows (carve records):

| split | carve | A ranked | LT2 bil | linked | two links | rrf, two links | beyond share | LTH bil | LTA bil | rrf, every row | LTM bil | musique 2/3/4-hop, LT2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| WQ-musique | webqsp/fit | 663 | 0.009 | 0.045 | 0.004 | 0.020 | 0.030 | 0.005 | 0.009 | 0.032 | 0.023 |  |
| WQ-musique | squad/fit | 0 | - | - | - | - | - | - | - | - | - |  |
| WQ-musique | hotpotqa/fit | 2830 | 0.643 | 0.802 | 0.111 | 0.172 | 0.094 | 0.574 | 0.639 | 0.088 | 0.793 |  |
| WQ-musique | 2wiki/fit | 5661 | 0.566 | 0.904 | 0.188 | 0.091 | 0.320 | 0.579 | 0.599 | 0.046 | 0.915 |  |
| WQ-musique | metaqa/s1eval | 1495 | 0.002 | 0.053 | 0.002 | 0.013 | 0.361 | 0.000 | 0.001 | 0.007 | 0.105 |  |
| WQ-musique | squad/s1eval | 0 | - | - | - | - | - | - | - | - | - |  |
| WQ-musique | musique/s1eval | 2547 | 0.168 | 0.449 | 0.018 | 0.099 | 0.304 | 0.131 | 0.172 | 0.067 | 0.454 | 0.349, 0.112, 0.055 |
| WQ-musique | hotpotqa/s1eval | 3632 | 0.631 | 0.804 | 0.084 | 0.156 | 0.102 | 0.555 | 0.621 | 0.096 | 0.809 |  |
| WQ-musique | 2wiki/s1eval | 12096 | 0.539 | 0.841 | 0.138 | 0.110 | 0.300 | 0.529 | 0.557 | 0.057 | 0.865 |  |
| WQ-musique | webqsp/s1eval | 1113 | 0.004 | 0.030 | 0.001 | 0.012 | 0.033 | 0.007 | 0.005 | 0.015 | 0.061 |  |
| WQ-hotpotqa | webqsp/fit | 663 | 0.002 | 0.011 | 0.000 | 0.020 | 0.030 | 0.006 | 0.005 | 0.032 | 0.023 |  |
| WQ-hotpotqa | squad/fit | 0 | - | - | - | - | - | - | - | - | - |  |
| WQ-hotpotqa | musique/s1fit | 3798 | 0.217 | 0.488 | 0.146 | 0.184 | 0.288 | 0.128 | 0.174 | 0.066 | 0.542 | 0.305, 0.138, 0.064 |
| WQ-hotpotqa | 2wiki/fit | 5661 | 0.550 | 0.885 | 0.120 | 0.091 | 0.320 | 0.532 | 0.580 | 0.046 | 0.915 |  |
| WQ-hotpotqa | metaqa/s1eval | 1495 | 0.003 | 0.053 | 0.004 | 0.013 | 0.361 | 0.001 | 0.001 | 0.007 | 0.053 |  |
| WQ-hotpotqa | squad/s1eval | 0 | - | - | - | - | - | - | - | - | - |  |
| WQ-hotpotqa | musique/s1eval | 2547 | 0.194 | 0.459 | 0.082 | 0.099 | 0.304 | 0.151 | 0.179 | 0.067 | 0.493 | 0.391, 0.137, 0.064 |
| WQ-hotpotqa | hotpotqa/s1eval | 3632 | 0.504 | 0.647 | 0.043 | 0.156 | 0.102 | 0.373 | 0.473 | 0.096 | 0.679 |  |
| WQ-hotpotqa | 2wiki/s1eval | 12096 | 0.529 | 0.827 | 0.116 | 0.110 | 0.300 | 0.476 | 0.533 | 0.057 | 0.855 |  |
| WQ-hotpotqa | webqsp/s1eval | 1113 | 0.002 | 0.010 | 0.001 | 0.012 | 0.033 | 0.005 | 0.002 | 0.015 | 0.061 |  |

LTA's zrckpa (uniform negatives; screen NOT_ADOPTED, full run None) - zrckpn at seed 0, as filed: WQ-musique squad -0.0001 WITHIN; WQ-musique hotpotqa -0.0012 WITHIN; WQ-musique 2wiki -0.0036 WITHIN; WQ-musique webqsp +0.0013 WITHIN; WQ-hotpotqa squad +0.0006 WITHIN; WQ-hotpotqa musique -0.0045 WITHIN; WQ-hotpotqa 2wiki -0.0013 WITHIN; WQ-hotpotqa webqsp +0.0029 WITHIN; WQ-hotpotqa musique-3hop -0.0053 WITHIN; WQ-hotpotqa musique-4hop -0.0111 LOSS

LTH's zrckph (ranked negatives; screen NOT_ADOPTED, full run None) - zrckpn at seed 0, as filed: WQ-musique squad -0.0007 WITHIN; WQ-musique hotpotqa -0.0019 WITHIN; WQ-musique 2wiki -0.0015 WITHIN; WQ-musique webqsp +0.0019 WITHIN; WQ-hotpotqa squad -0.0004 WITHIN; WQ-hotpotqa musique -0.0104 LOSS; WQ-hotpotqa 2wiki -0.0025 WITHIN; WQ-hotpotqa webqsp +0.0024 WITHIN; WQ-hotpotqa musique-3hop -0.0123 WITHIN; WQ-hotpotqa musique-4hop -0.0160 LOSS

| fit | train s | read s |
|---|---:|---:|
| scr-lt2-zrckp2-s0 | 3314.1 | 3740.3 |
| scr-lt2-zrckp2-s0-hp | 5201.8 | 4175.8 |
