# LTC chain columns (declared in docs/LTC_CHAINED_BEAM_MLP.md)

Per depth d (1 = LTM's NEXT, 2 and 3 = the chain): the share of rows reached, the share of golds outside the top five reached, and the AUC of the depth's max column over the reached rows (gold against not).

| split | carve | training | rows d1 | d2 | d3 | missed golds d1 | d2 | d3 | any | AUC d1 | d2 | d3 | hop one | s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WQ-musique | webqsp/fit | True | 0.050 | 0.164 | 0.144 | 0.525 | 0.484 | 0.553 | 0.859 | 0.570 | 0.698 | 0.675 | 0.0020 | 151 |
| WQ-musique | squad/fit | True | 0.473 | 0.493 | 0.495 | 0.634 | 0.626 | 0.624 | 0.690 | 0.654 | 0.687 | 0.685 | 0.0020 | 58 |
| WQ-musique | hotpotqa/fit | True | 0.115 | 0.290 | 0.247 | 0.766 | 0.635 | 0.786 | 0.893 | 0.927 | 0.800 | 0.905 | 0.0020 | 225 |
| WQ-musique | 2wiki/fit | True | 0.089 | 0.223 | 0.172 | 0.730 | 0.371 | 0.733 | 0.794 | 0.980 | 0.719 | 0.950 | 0.0020 | 245 |
| WQ-musique | metaqa/s1eval | False | 0.008 | 0.107 | 0.076 | 0.115 | 0.284 | 0.190 | 0.441 | 0.613 | 0.701 | 0.621 | 0.0020 | 755 |
| WQ-musique | squad/s1eval | False | 0.430 | 0.452 | 0.456 | 0.586 | 0.590 | 0.613 | 0.670 | 0.661 | 0.654 | 0.662 | 0.0020 | 105 |
| WQ-musique | musique/s1eval | False | 0.047 | 0.173 | 0.142 | 0.511 | 0.595 | 0.590 | 0.736 | 0.813 | 0.841 | 0.852 | 0.0020 | 356 |
| WQ-musique | hotpotqa/s1eval | False | 0.114 | 0.294 | 0.250 | 0.755 | 0.651 | 0.787 | 0.887 | 0.926 | 0.802 | 0.903 | 0.0020 | 284 |
| WQ-musique | 2wiki/s1eval | False | 0.093 | 0.211 | 0.170 | 0.775 | 0.470 | 0.770 | 0.833 | 0.975 | 0.779 | 0.944 | 0.0020 | 541 |
| WQ-musique | webqsp/s1eval | False | 0.048 | 0.163 | 0.139 | 0.442 | 0.455 | 0.470 | 0.764 | 0.589 | 0.689 | 0.703 | 0.0020 | 152 |
| WQ-hotpotqa | webqsp/fit | True | 0.050 | 0.147 | 0.122 | 0.525 | 0.445 | 0.432 | 0.835 | 0.491 | 0.612 | 0.678 | 0.0020 | 130 |
| WQ-hotpotqa | squad/fit | True | 0.473 | 0.494 | 0.497 | 0.634 | 0.629 | 0.634 | 0.690 | 0.628 | 0.669 | 0.656 | 0.0020 | 57 |
| WQ-hotpotqa | musique/s1fit | True | 0.045 | 0.160 | 0.128 | 0.469 | 0.592 | 0.551 | 0.702 | 0.870 | 0.886 | 0.890 | 0.0020 | 678 |
| WQ-hotpotqa | 2wiki/fit | True | 0.089 | 0.218 | 0.171 | 0.730 | 0.390 | 0.732 | 0.795 | 0.979 | 0.740 | 0.945 | 0.0020 | 260 |
| WQ-hotpotqa | metaqa/s1eval | False | 0.008 | 0.101 | 0.048 | 0.115 | 0.237 | 0.151 | 0.370 | 0.570 | 0.610 | 0.555 | 0.0020 | 688 |
| WQ-hotpotqa | squad/s1eval | False | 0.430 | 0.454 | 0.458 | 0.586 | 0.593 | 0.613 | 0.668 | 0.642 | 0.642 | 0.635 | 0.0020 | 112 |
| WQ-hotpotqa | musique/s1eval | False | 0.047 | 0.153 | 0.127 | 0.511 | 0.574 | 0.580 | 0.727 | 0.851 | 0.864 | 0.871 | 0.0020 | 337 |
| WQ-hotpotqa | hotpotqa/s1eval | False | 0.114 | 0.289 | 0.254 | 0.755 | 0.649 | 0.750 | 0.882 | 0.889 | 0.794 | 0.863 | 0.0020 | 325 |
| WQ-hotpotqa | 2wiki/s1eval | False | 0.093 | 0.208 | 0.168 | 0.775 | 0.490 | 0.768 | 0.834 | 0.974 | 0.785 | 0.938 | 0.0020 | 549 |
| WQ-hotpotqa | webqsp/s1eval | False | 0.048 | 0.147 | 0.124 | 0.442 | 0.417 | 0.425 | 0.737 | 0.514 | 0.615 | 0.651 | 0.0020 | 186 |
