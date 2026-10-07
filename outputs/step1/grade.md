# Step 1 grade (seed 0)

Verdict over the eleven primary reads: **NOT_ADOPTED**. Development numbers; the paper's numbers come from one declared confirmation run.

## Picks

| fit | M-pick (score) | D-pick (score) |
| --- | --- | --- |
| J5 | p@swa (0.7450) | p@swa (0.7078) |
| L-metaqa | p@ep7 (0.7792) | pf@swa (0.7401) |
| L-squad | p@swa (0.7068) | p@swa (0.6628) |
| L-musique | p@swa (0.8130) | p@swa (0.7895) |
| L-hotpotqa | p@ep7 (0.7141) | pf@swa (0.6716) |
| L-2wiki | pf@swa (0.7238) | p@ep5 (0.6875) |

## Primary reads (graded on R@5)

| fit | read | role | R@5 M | R@5 D | D - M R@5 [95%] | label | D - M FC@5 [95%] | D - M hit@1 [95%] | twin0 R@5 | gnn0 R@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J5 | metaqa | in-domain | 0.6443 | 0.6443 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.7212 | 0.7839 |
| J5 | squad | in-domain | 0.9095 | 0.9095 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8971 | 0.9038 |
| J5 | musique | in-domain | 0.5609 | 0.5609 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.5119 | 0.5380 |
| J5 | hotpotqa | in-domain | 0.9013 | 0.9013 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8786 | 0.8993 |
| J5 | 2wiki | in-domain | 0.8694 | 0.8694 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8487 | 0.8906 |
| J5 | webqsp | zero-shot | 0.1167 | 0.1167 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.6015 | 0.6289 |
| L-metaqa | metaqa | zero-shot | 0.0888 | 0.0709 | -0.0179 [-0.0213, -0.0145] | BELOW | -0.0144 [-0.0178, -0.0110] | -0.0058 [-0.0075, -0.0043] | 0.7212 | 0.7839 |
| L-squad | squad | zero-shot | 0.8797 | 0.8797 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8971 | 0.9038 |
| L-musique | musique | zero-shot | 0.2696 | 0.2696 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.5119 | 0.5380 |
| L-hotpotqa | hotpotqa | zero-shot | 0.8510 | 0.8563 | +0.0053 [+0.0013, +0.0090] | ABOVE | +0.0109 [+0.0036, +0.0181] | +0.0024 [-0.0028, +0.0077] | 0.8786 | 0.8993 |
| L-2wiki | 2wiki | zero-shot | 0.8070 | 0.7897 | -0.0173 [-0.0201, -0.0146] | BELOW | -0.0181 [-0.0230, -0.0134] | -0.0165 [-0.0216, -0.0117] | 0.8487 | 0.8906 |

## Secondary reads (not graded)

| fit | read | role | R@5 M | R@5 D | D - M R@5 [95%] | label | D - M FC@5 [95%] | D - M hit@1 [95%] | twin0 R@5 | gnn0 R@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L-metaqa | squad | in-domain | 0.9068 | 0.9118 | +0.0050 [+0.0028, +0.0072] | ABOVE | +0.0050 [+0.0028, +0.0072] | +0.0058 [+0.0019, +0.0096] | 0.8971 | 0.9038 |
| L-metaqa | musique | in-domain | 0.5639 | 0.5698 | +0.0059 [-0.0003, +0.0120] | AT | +0.0103 [+0.0012, +0.0199] | +0.0186 [+0.0062, +0.0306] | 0.5119 | 0.5380 |
| L-metaqa | hotpotqa | in-domain | 0.8978 | 0.9057 | +0.0080 [+0.0051, +0.0109] | ABOVE | +0.0130 [+0.0077, +0.0181] | +0.0054 [-0.0005, +0.0117] | 0.8786 | 0.8993 |
| L-metaqa | 2wiki | in-domain | 0.8674 | 0.8720 | +0.0045 [+0.0024, +0.0067] | ABOVE | +0.0083 [+0.0040, +0.0127] | +0.0172 [+0.0128, +0.0220] | 0.8487 | 0.8906 |
| L-metaqa | webqsp | zero-shot | 0.1039 | 0.0858 | -0.0180 [-0.0275, -0.0093] | BELOW | -0.0140 [-0.0233, -0.0053] | -0.0160 [-0.0240, -0.0086] | 0.6015 | 0.6289 |
| L-squad | metaqa | in-domain | 0.6535 | 0.6535 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.7212 | 0.7839 |
| L-squad | musique | in-domain | 0.5743 | 0.5743 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.5119 | 0.5380 |
| L-squad | hotpotqa | in-domain | 0.9045 | 0.9045 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8786 | 0.8993 |
| L-squad | 2wiki | in-domain | 0.8724 | 0.8724 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8487 | 0.8906 |
| L-squad | webqsp | zero-shot | 0.1302 | 0.1302 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.6015 | 0.6289 |
| L-musique | metaqa | in-domain | 0.6537 | 0.6537 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.7212 | 0.7839 |
| L-musique | squad | in-domain | 0.9100 | 0.9100 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8971 | 0.9038 |
| L-musique | hotpotqa | in-domain | 0.9015 | 0.9015 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8786 | 0.8993 |
| L-musique | 2wiki | in-domain | 0.8732 | 0.8732 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.8487 | 0.8906 |
| L-musique | webqsp | zero-shot | 0.1779 | 0.1779 | +0.0000 [+0.0000, +0.0000] | SAME | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.6015 | 0.6289 |
| L-hotpotqa | metaqa | in-domain | 0.6450 | 0.6420 | -0.0030 [-0.0066, +0.0006] | AT | +0.0000 [-0.0044, +0.0047] | +0.0157 [+0.0076, +0.0240] | 0.7212 | 0.7839 |
| L-hotpotqa | squad | in-domain | 0.9080 | 0.9131 | +0.0051 [+0.0026, +0.0076] | ABOVE | +0.0051 [+0.0026, +0.0076] | +0.0091 [+0.0051, +0.0131] | 0.8971 | 0.9038 |
| L-hotpotqa | musique | in-domain | 0.5587 | 0.5616 | +0.0029 [-0.0032, +0.0090] | AT | +0.0012 [-0.0083, +0.0116] | +0.0149 [+0.0025, +0.0269] | 0.5119 | 0.5380 |
| L-hotpotqa | 2wiki | in-domain | 0.8689 | 0.8742 | +0.0053 [+0.0030, +0.0076] | ABOVE | +0.0016 [-0.0033, +0.0064] | +0.0076 [+0.0034, +0.0118] | 0.8487 | 0.8906 |
| L-hotpotqa | webqsp | zero-shot | 0.1214 | 0.1077 | -0.0138 [-0.0238, -0.0043] | BELOW | -0.0146 [-0.0240, -0.0053] | +0.0226 [+0.0133, +0.0326] | 0.6015 | 0.6289 |
| L-2wiki | metaqa | in-domain | 0.6415 | 0.6519 | +0.0104 [+0.0068, +0.0138] | ABOVE | +0.0102 [+0.0058, +0.0142] | +0.0134 [+0.0054, +0.0210] | 0.7212 | 0.7839 |
| L-2wiki | squad | in-domain | 0.9113 | 0.9091 | -0.0022 [-0.0047, +0.0003] | AT | -0.0022 [-0.0047, +0.0003] | -0.0003 [-0.0037, +0.0033] | 0.8971 | 0.9038 |
| L-2wiki | musique | in-domain | 0.5593 | 0.5569 | -0.0024 [-0.0089, +0.0041] | AT | -0.0021 [-0.0124, +0.0083] | -0.0124 [-0.0244, +0.0008] | 0.5119 | 0.5380 |
| L-2wiki | hotpotqa | in-domain | 0.8995 | 0.8984 | -0.0011 [-0.0041, +0.0018] | AT | +0.0009 [-0.0045, +0.0064] | -0.0095 [-0.0159, -0.0028] | 0.8786 | 0.8993 |
| L-2wiki | webqsp | zero-shot | 0.1275 | 0.1436 | +0.0161 [+0.0043, +0.0273] | ABOVE | +0.0140 [+0.0027, +0.0246] | -0.0013 [-0.0106, +0.0080] | 0.6015 | 0.6289 |

## Primary reads by stratum (diagnostic)

| fit | read | stratum | questions | R@5 M | R@5 D | D - M R@5 [95%] | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| J5 | metaqa | hop1 | 2498 | 0.9215 | 0.9215 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | metaqa | hop2 | 3718 | 0.7513 | 0.7513 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | metaqa | hop3 | 3569 | 0.3387 | 0.3387 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | squad | answerable=False | 5945 | 0.8782 | 0.8782 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | squad | answerable=True | 5928 | 0.9410 | 0.9410 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | musique | hop2 | 1252 | 0.6637 | 0.6637 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | musique | hop3 | 760 | 0.5136 | 0.5136 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | musique | hop4 | 405 | 0.3315 | 0.3315 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | hotpotqa | hard/bridge | 5918 | 0.8853 | 0.8853 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | hotpotqa | hard/comparison | 1487 | 0.9650 | 0.9650 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | 2wiki | bridge_comparison | 2751 | 0.7605 | 0.7605 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | 2wiki | comparison | 3040 | 0.9531 | 0.9531 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | 2wiki | compositional | 5236 | 0.8876 | 0.8876 | +0.0000 [+0.0000, +0.0000] | SAME |
| J5 | 2wiki | inference | 1549 | 0.8373 | 0.8373 | +0.0000 [+0.0000, +0.0000] | SAME |
| L-metaqa | metaqa | hop1 | 2498 | 0.2408 | 0.1966 | -0.0442 [-0.0553, -0.0332] | BELOW |
| L-metaqa | metaqa | hop2 | 3718 | 0.0411 | 0.0303 | -0.0108 [-0.0153, -0.0066] | BELOW |
| L-metaqa | metaqa | hop3 | 3569 | 0.0320 | 0.0252 | -0.0068 [-0.0094, -0.0042] | BELOW |
| L-squad | squad | answerable=False | 5945 | 0.8421 | 0.8421 | +0.0000 [+0.0000, +0.0000] | SAME |
| L-squad | squad | answerable=True | 5928 | 0.9175 | 0.9175 | +0.0000 [+0.0000, +0.0000] | SAME |
| L-musique | musique | hop2 | 1252 | 0.3446 | 0.3446 | +0.0000 [+0.0000, +0.0000] | SAME |
| L-musique | musique | hop3 | 760 | 0.2123 | 0.2123 | +0.0000 [+0.0000, +0.0000] | SAME |
| L-musique | musique | hop4 | 405 | 0.1451 | 0.1451 | +0.0000 [+0.0000, +0.0000] | SAME |
| L-hotpotqa | hotpotqa | hard/bridge | 5918 | 0.8271 | 0.8317 | +0.0046 [+0.0002, +0.0090] | ABOVE |
| L-hotpotqa | hotpotqa | hard/comparison | 1487 | 0.9462 | 0.9543 | +0.0081 [+0.0020, +0.0141] | ABOVE |
| L-2wiki | 2wiki | bridge_comparison | 2751 | 0.5902 | 0.5531 | -0.0372 [-0.0428, -0.0317] | BELOW |
| L-2wiki | 2wiki | comparison | 3040 | 0.9174 | 0.9197 | +0.0023 [-0.0021, +0.0072] | AT |
| L-2wiki | 2wiki | compositional | 5236 | 0.8541 | 0.8387 | -0.0154 [-0.0198, -0.0107] | BELOW |
| L-2wiki | 2wiki | inference | 1549 | 0.8163 | 0.7892 | -0.0271 [-0.0362, -0.0184] | BELOW |

## Which select carve predicts the eval (Spearman over the 36 candidates; diagnostic)

| fit | read | pooled M | pooled D | own M | own D |
| --- | --- | --- | --- | --- | --- |
| J5 | metaqa | +0.963 | +0.963 | +0.989 | +0.984 |
| J5 | squad | -0.018 | +0.031 | +0.604 | +0.784 |
| J5 | musique | +0.687 | +0.684 | +0.690 | +0.761 |
| J5 | hotpotqa | +0.707 | +0.702 | +0.907 | +0.881 |
| J5 | 2wiki | +0.973 | +0.975 | +0.956 | +0.941 |
| J5 | webqsp | +0.670 | +0.632 | - | - |
| L-metaqa | metaqa | +0.082 | +0.036 | - | - |
| L-metaqa | squad | +0.037 | +0.104 | +0.263 | +0.760 |
| L-metaqa | musique | +0.585 | +0.504 | +0.723 | +0.677 |
| L-metaqa | hotpotqa | +0.808 | +0.824 | +0.922 | +0.883 |
| L-metaqa | 2wiki | +0.950 | +0.942 | +0.913 | +0.882 |
| L-metaqa | webqsp | +0.432 | +0.366 | - | - |
| L-squad | metaqa | +0.962 | +0.956 | +0.980 | +0.982 |
| L-squad | squad | -0.383 | -0.325 | - | - |
| L-squad | musique | +0.583 | +0.573 | +0.741 | +0.805 |
| L-squad | hotpotqa | +0.840 | +0.858 | +0.923 | +0.848 |
| L-squad | 2wiki | +0.980 | +0.972 | +0.964 | +0.968 |
| L-squad | webqsp | +0.599 | +0.568 | - | - |
| L-musique | metaqa | +0.962 | +0.960 | +0.983 | +0.987 |
| L-musique | squad | +0.092 | +0.117 | +0.558 | +0.671 |
| L-musique | musique | +0.162 | +0.162 | - | - |
| L-musique | hotpotqa | +0.779 | +0.778 | +0.898 | +0.905 |
| L-musique | 2wiki | +0.923 | +0.931 | +0.918 | +0.921 |
| L-musique | webqsp | +0.210 | +0.231 | - | - |
| L-hotpotqa | metaqa | +0.977 | +0.971 | +0.992 | +0.995 |
| L-hotpotqa | squad | -0.081 | -0.071 | +0.493 | +0.578 |
| L-hotpotqa | musique | +0.615 | +0.648 | +0.652 | +0.789 |
| L-hotpotqa | hotpotqa | -0.485 | -0.475 | - | - |
| L-hotpotqa | 2wiki | +0.910 | +0.926 | +0.964 | +0.964 |
| L-hotpotqa | webqsp | +0.594 | +0.593 | - | - |
| L-2wiki | metaqa | +0.968 | +0.971 | +0.992 | +0.989 |
| L-2wiki | squad | +0.058 | +0.040 | +0.384 | +0.575 |
| L-2wiki | musique | +0.749 | +0.737 | +0.837 | +0.906 |
| L-2wiki | hotpotqa | +0.643 | +0.634 | +0.907 | +0.826 |
| L-2wiki | 2wiki | +0.638 | +0.621 | - | - |
| L-2wiki | webqsp | +0.590 | +0.588 | - | - |

## Eval-best candidate (an oracle, never a result)

| fit | read | eval-best | R@5 | M-pick R@5 | D-pick R@5 |
| --- | --- | --- | --- | --- | --- |
| J5 | metaqa | p@ep6 | 0.6509 | 0.6443 | 0.6443 |
| J5 | squad | nf@ep1 | 0.9132 | 0.9095 | 0.9095 |
| J5 | musique | nf@ep6 | 0.5626 | 0.5609 | 0.5609 |
| J5 | hotpotqa | pf@ep4 | 0.9036 | 0.9013 | 0.9013 |
| J5 | 2wiki | p@swa | 0.8694 | 0.8694 | 0.8694 |
| J5 | webqsp | pf@ep3 | 0.1472 | 0.1167 | 0.1167 |
| L-metaqa | metaqa | n@ep4 | 0.1482 | 0.0888 | 0.0709 |
| L-metaqa | squad | nf@ep5 | 0.9138 | 0.9068 | 0.9118 |
| L-metaqa | musique | pf@swa | 0.5698 | 0.5639 | 0.5698 |
| L-metaqa | hotpotqa | pf@swa | 0.9057 | 0.8978 | 0.9057 |
| L-metaqa | 2wiki | pf@swa | 0.8720 | 0.8674 | 0.8720 |
| L-metaqa | webqsp | p@ep4 | 0.1172 | 0.1039 | 0.0858 |
| L-squad | metaqa | p@swa | 0.6535 | 0.6535 | 0.6535 |
| L-squad | squad | nf@swa | 0.8962 | 0.8797 | 0.8797 |
| L-squad | musique | p@swa | 0.5743 | 0.5743 | 0.5743 |
| L-squad | hotpotqa | p@swa | 0.9045 | 0.9045 | 0.9045 |
| L-squad | 2wiki | p@swa | 0.8724 | 0.8724 | 0.8724 |
| L-squad | webqsp | pf@ep7 | 0.1473 | 0.1302 | 0.1302 |
| L-musique | metaqa | p@swa | 0.6537 | 0.6537 | 0.6537 |
| L-musique | squad | nf@ep0 | 0.9138 | 0.9100 | 0.9100 |
| L-musique | musique | p@ep6 | 0.3116 | 0.2696 | 0.2696 |
| L-musique | hotpotqa | pf@swa | 0.9049 | 0.9015 | 0.9015 |
| L-musique | 2wiki | p@swa | 0.8732 | 0.8732 | 0.8732 |
| L-musique | webqsp | p@ep1 | 0.2126 | 0.1779 | 0.1779 |
| L-hotpotqa | metaqa | p@swa | 0.6540 | 0.6450 | 0.6420 |
| L-hotpotqa | squad | pf@ep1 | 0.9137 | 0.9080 | 0.9131 |
| L-hotpotqa | musique | nf@ep4 | 0.5631 | 0.5587 | 0.5616 |
| L-hotpotqa | hotpotqa | nf@swa | 0.8718 | 0.8510 | 0.8563 |
| L-hotpotqa | 2wiki | pf@swa | 0.8742 | 0.8689 | 0.8742 |
| L-hotpotqa | webqsp | p@ep6 | 0.1685 | 0.1214 | 0.1077 |
| L-2wiki | metaqa | p@ep5 | 0.6519 | 0.6415 | 0.6519 |
| L-2wiki | squad | nf@ep1 | 0.9132 | 0.9113 | 0.9091 |
| L-2wiki | musique | p@swa | 0.5601 | 0.5593 | 0.5569 |
| L-2wiki | hotpotqa | pf@ep7 | 0.9053 | 0.8995 | 0.8984 |
| L-2wiki | 2wiki | p@swa | 0.8074 | 0.8070 | 0.7897 |
| L-2wiki | webqsp | pf@ep3 | 0.1609 | 0.1275 | 0.1436 |

## Per-variant picks (state only; not graded)

| fit | variant | M | D |
| --- | --- | --- | --- |
| J5 | p | p@swa | p@swa |
| J5 | pf | pf@ep3 | pf@ep5 |
| J5 | n | n@swa | n@swa |
| J5 | nf | nf@ep6 | nf@swa |
| L-metaqa | p | p@ep7 | p@ep5 |
| L-metaqa | pf | pf@swa | pf@swa |
| L-metaqa | n | n@swa | n@swa |
| L-metaqa | nf | nf@swa | nf@swa |
| L-squad | p | p@swa | p@swa |
| L-squad | pf | pf@ep7 | pf@ep6 |
| L-squad | n | n@ep6 | n@ep7 |
| L-squad | nf | nf@ep6 | nf@swa |
| L-musique | p | p@swa | p@swa |
| L-musique | pf | pf@swa | pf@ep7 |
| L-musique | n | n@swa | n@swa |
| L-musique | nf | nf@swa | nf@swa |
| L-hotpotqa | p | p@ep7 | p@ep7 |
| L-hotpotqa | pf | pf@ep7 | pf@swa |
| L-hotpotqa | n | n@ep6 | n@ep6 |
| L-hotpotqa | nf | nf@swa | nf@swa |
| L-2wiki | p | p@ep6 | p@ep5 |
| L-2wiki | pf | pf@swa | pf@ep7 |
| L-2wiki | n | n@swa | n@swa |
| L-2wiki | nf | nf@swa | nf@swa |

carves.json 53cfb89f1e41..., freeze 58958f33a3af..., script fe85dda8ed7a..., 2026-10-07T06:52:09Z
