# P2a, reach: what each link family is worth to the pools (s1eval)

With minus without the family, at every N and both measures; the regime is the dataset's frozen graph. Labels use the tie 0.002: CARRIES (above it somewhere, never below), HARMS, MIXED, NOTHING.

## metaqa (regime S, 9785 questions)

| restart | family | kind | label | largest | smallest | at I_q (ALL) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A3 | structural | LOO (S vs none) | CARRIES | +0.6397 | +0.2273 | +0.5395 |
| A3 | ner | ADD (SN vs S) | HARMS | -0.0049 | -0.0204 | -0.0143 |
| A3 | knn | ADD (SK vs S) | HARMS | -0.0367 | -0.1971 | -0.0626 |
| A1 | structural | LOO (S vs none) | CARRIES | +0.6397 | +0.2222 | +0.5510 |
| A1 | ner | ADD (SN vs S) | HARMS | -0.0049 | -0.0221 | -0.0179 |
| A1 | knn | ADD (SK vs S) | HARMS | -0.0369 | -0.1724 | -0.0734 |
| LINKED | structural | LOO (S vs none) | CARRIES | +0.9937 | +0.5457 | +0.9608 |
| LINKED | ner | ADD (SN vs S) | HARMS | -0.0004 | -0.0214 | -0.0160 |
| LINKED | knn | ADD (SK vs S) | HARMS | -0.0094 | -0.1600 | -0.0782 |

ALL by subset, A3:

| subset | 25 | 100 | 500 | 1000 | 2000 | 7000 | at I_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.009 | 0.021 | 0.103 | 0.186 | 0.262 | 0.262 | 0.261 |
| S (regime) | 0.237 | 0.515 | 0.670 | 0.743 | 0.800 | 0.901 | 0.800 |
| N | 0.013 | 0.023 | 0.087 | 0.182 | 0.262 | 0.264 | 0.261 |
| K | 0.011 | 0.029 | 0.102 | 0.184 | 0.264 | 0.404 | 0.264 |
| SN | 0.224 | 0.499 | 0.649 | 0.726 | 0.785 | 0.893 | 0.786 |
| SK | 0.143 | 0.355 | 0.560 | 0.655 | 0.737 | 0.849 | 0.738 |
| NK | 0.010 | 0.028 | 0.094 | 0.184 | 0.264 | 0.402 | 0.264 |
| SNK | 0.139 | 0.349 | 0.554 | 0.644 | 0.729 | 0.843 | 0.730 |

Candidates: structural KEEP, ner LEAVE_OUT, knn LEAVE_OUT

## squad (regime none, 11873 questions)

| restart | family | kind | label | largest | smallest | at I_q (ALL) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A3 | structural | ADD (S vs none) | HARMS | +0.0001 | -0.0253 | -0.0253 |
| A3 | ner | ADD (N vs none) | HARMS | +0.0007 | -0.0201 | -0.0200 |
| A3 | knn | ADD (K vs none) | HARMS | +0.0004 | -0.0216 | -0.0192 |
| A1 | structural | ADD (S vs none) | HARMS | +0.0000 | -0.0208 | -0.0158 |
| A1 | ner | ADD (N vs none) | HARMS | +0.0006 | -0.0235 | -0.0189 |
| A1 | knn | ADD (K vs none) | HARMS | +0.0004 | -0.0209 | -0.0189 |

ALL by subset, A3:

| subset | 25 | 100 | 500 | 1000 | 2000 | 7000 | at I_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none (regime) | 0.970 | 0.989 | 0.997 | 0.999 | 0.999 | 0.999 | 0.981 |
| S | 0.949 | 0.969 | 0.994 | 0.997 | 0.999 | 0.999 | 0.955 |
| N | 0.950 | 0.974 | 0.996 | 0.998 | 0.999 | 1.000 | 0.961 |
| K | 0.949 | 0.981 | 0.997 | 0.999 | 0.999 | 1.000 | 0.962 |
| SN | 0.947 | 0.976 | 0.995 | 0.998 | 0.999 | 1.000 | 0.960 |
| SK | 0.950 | 0.976 | 0.995 | 0.998 | 0.999 | 1.000 | 0.963 |
| NK | 0.951 | 0.979 | 0.997 | 0.998 | 0.999 | 1.000 | 0.963 |
| SNK | 0.951 | 0.979 | 0.996 | 0.998 | 0.999 | 1.000 | 0.964 |

Candidates: structural LEAVE_OUT, ner LEAVE_OUT, knn LEAVE_OUT

## musique (regime SNK, 2417 questions)

| restart | family | kind | label | largest | smallest | at I_q (ALL) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A3 | structural | LOO (SNK vs NK) | CARRIES | +0.0339 | +0.0025 | +0.0154 |
| A3 | ner | LOO (SNK vs SK) | CARRIES | +0.0538 | +0.0101 | +0.0323 |
| A3 | knn | LOO (SNK vs SN) | CARRIES | +0.0161 | +0.0023 | +0.0079 |
| A1 | structural | LOO (SNK vs NK) | CARRIES | +0.0285 | +0.0041 | +0.0137 |
| A1 | ner | LOO (SNK vs SK) | CARRIES | +0.0559 | +0.0083 | +0.0307 |
| A1 | knn | LOO (SNK vs SN) | MIXED | +0.0174 | -0.0067 | +0.0100 |

ALL by subset, A3:

| subset | 25 | 100 | 500 | 1000 | 2000 | 7000 | at I_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.303 | 0.463 | 0.637 | 0.718 | 0.760 | 0.760 | 0.760 |
| S | 0.293 | 0.475 | 0.710 | 0.782 | 0.833 | 0.899 | 0.837 |
| N | 0.313 | 0.511 | 0.748 | 0.811 | 0.853 | 0.891 | 0.855 |
| K | 0.295 | 0.493 | 0.705 | 0.768 | 0.808 | 0.883 | 0.814 |
| SN | 0.336 | 0.539 | 0.783 | 0.834 | 0.875 | 0.925 | 0.876 |
| SK | 0.317 | 0.506 | 0.739 | 0.802 | 0.848 | 0.921 | 0.852 |
| NK | 0.308 | 0.545 | 0.773 | 0.826 | 0.867 | 0.919 | 0.869 |
| SNK (regime) | 0.342 | 0.548 | 0.788 | 0.843 | 0.882 | 0.940 | 0.884 |

Candidates: structural KEEP, ner KEEP, knn KEEP

## hotpotqa (regime S, 7405 questions)

| restart | family | kind | label | largest | smallest | at I_q (ALL) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A3 | structural | LOO (S vs none) | MIXED | +0.1813 | -0.0615 | +0.1653 |
| A3 | ner | ADD (SN vs S) | HARMS | +0.0000 | -0.0378 | -0.0439 |
| A3 | knn | ADD (SK vs S) | MIXED | +0.0407 | -0.0324 | -0.0345 |
| A1 | structural | LOO (S vs none) | CARRIES | +0.2654 | +0.0585 | +0.2529 |
| A1 | ner | ADD (SN vs S) | HARMS | +0.0000 | -0.0259 | -0.0083 |
| A1 | knn | ADD (SK vs S) | HARMS | +0.0002 | -0.0717 | -0.0158 |

ALL by subset, A3:

| subset | 25 | 100 | 500 | 1000 | 2000 | 7000 | at I_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.554 | 0.703 | 0.816 | 0.852 | 0.880 | 0.880 | 0.691 |
| S (regime) | 0.512 | 0.868 | 0.982 | 0.992 | 0.995 | 0.999 | 0.857 |
| N | 0.402 | 0.685 | 0.828 | 0.867 | 0.900 | 0.921 | 0.675 |
| K | 0.524 | 0.678 | 0.829 | 0.866 | 0.896 | 0.930 | 0.670 |
| SN | 0.484 | 0.831 | 0.966 | 0.987 | 0.994 | 0.999 | 0.813 |
| SK | 0.535 | 0.838 | 0.969 | 0.987 | 0.994 | 0.998 | 0.822 |
| NK | 0.455 | 0.688 | 0.839 | 0.879 | 0.911 | 0.944 | 0.676 |
| SNK | 0.511 | 0.816 | 0.958 | 0.983 | 0.993 | 0.998 | 0.799 |

Candidates: structural KEEP, ner LEAVE_OUT, knn LEAVE_OUT

## 2wiki (regime S, 12576 questions)

| restart | family | kind | label | largest | smallest | at I_q (ALL) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A3 | structural | LOO (S vs none) | CARRIES | +0.4157 | +0.0161 | +0.3103 |
| A3 | ner | ADD (SN vs S) | MIXED | +0.0116 | -0.1275 | -0.1136 |
| A3 | knn | ADD (SK vs S) | MIXED | +0.0050 | -0.1220 | -0.1107 |
| A1 | structural | LOO (S vs none) | CARRIES | +0.4413 | +0.1402 | +0.4426 |
| A1 | ner | ADD (SN vs S) | MIXED | +0.0112 | -0.0395 | -0.0116 |
| A1 | knn | ADD (SK vs S) | MIXED | +0.0035 | -0.1339 | -0.0256 |

ALL by subset, A3:

| subset | 25 | 100 | 500 | 1000 | 2000 | 7000 | at I_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.371 | 0.462 | 0.530 | 0.561 | 0.591 | 0.591 | 0.462 |
| S (regime) | 0.415 | 0.757 | 0.946 | 0.962 | 0.969 | 0.976 | 0.772 |
| N | 0.353 | 0.478 | 0.601 | 0.657 | 0.713 | 0.757 | 0.478 |
| K | 0.384 | 0.502 | 0.583 | 0.611 | 0.644 | 0.696 | 0.503 |
| SN | 0.391 | 0.651 | 0.893 | 0.957 | 0.976 | 0.988 | 0.658 |
| SK | 0.410 | 0.653 | 0.912 | 0.960 | 0.973 | 0.980 | 0.661 |
| NK | 0.368 | 0.500 | 0.614 | 0.662 | 0.723 | 0.800 | 0.502 |
| SNK | 0.399 | 0.619 | 0.851 | 0.944 | 0.975 | 0.989 | 0.623 |

Candidates: structural KEEP, ner LEAVE_OUT, knn LEAVE_OUT

## webqsp (regime SNK, 1503 questions)

| restart | family | kind | label | largest | smallest | at I_q (ALL) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A3 | structural | LOO (SNK vs NK) | CARRIES | +0.3007 | +0.1358 | +0.2654 |
| A3 | ner | LOO (SNK vs SK) | HARMS | -0.0046 | -0.0200 | -0.0180 |
| A3 | knn | LOO (SNK vs SN) | HARMS | -0.0175 | -0.0567 | -0.0486 |
| A1 | structural | LOO (SNK vs NK) | CARRIES | +0.3061 | +0.1623 | +0.2941 |
| A1 | ner | LOO (SNK vs SK) | HARMS | -0.0060 | -0.0193 | -0.0113 |
| A1 | knn | LOO (SNK vs SN) | HARMS | -0.0204 | -0.0732 | -0.0353 |
| LINKED | structural | LOO (SNK vs NK) | CARRIES | +0.5660 | +0.0193 | +0.5183 |
| LINKED | ner | LOO (SNK vs SK) | HARMS | +0.0013 | -0.0118 | -0.0080 |
| LINKED | knn | LOO (SNK vs SN) | HARMS | +0.0016 | -0.1437 | -0.0439 |

ALL by subset, A3:

| subset | 25 | 100 | 500 | 1000 | 2000 | 7000 | at I_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.106 | 0.178 | 0.313 | 0.371 | 0.412 | 0.412 | 0.412 |
| S | 0.335 | 0.492 | 0.673 | 0.760 | 0.824 | 0.894 | 0.829 |
| N | 0.134 | 0.216 | 0.368 | 0.419 | 0.474 | 0.481 | 0.475 |
| K | 0.115 | 0.167 | 0.309 | 0.375 | 0.436 | 0.511 | 0.441 |
| SN | 0.307 | 0.454 | 0.641 | 0.722 | 0.799 | 0.880 | 0.808 |
| SK | 0.274 | 0.436 | 0.605 | 0.681 | 0.769 | 0.860 | 0.777 |
| NK | 0.122 | 0.202 | 0.362 | 0.428 | 0.492 | 0.565 | 0.494 |
| SNK (regime) | 0.258 | 0.422 | 0.590 | 0.667 | 0.752 | 0.852 | 0.759 |

Candidates: structural KEEP, ner DROP_CANDIDATE, knn DROP_CANDIDATE

## One subset for all six (A3, s1eval, against each dataset's regime)

| subset | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | HARMS | REGIME | HARMS | MIXED | HARMS | HARMS | no |
| S | REGIME | HARMS | HARMS | REGIME | REGIME | CARRIES | no |
| N | HARMS | HARMS | HARMS | HARMS | HARMS | HARMS | no |
| K | HARMS | HARMS | HARMS | MIXED | HARMS | HARMS | no |
| SN | HARMS | HARMS | HARMS | HARMS | MIXED | CARRIES | no |
| SK | HARMS | HARMS | HARMS | MIXED | MIXED | CARRIES | no |
| NK | HARMS | HARMS | HARMS | HARMS | HARMS | HARMS | no |
| SNK | HARMS | HARMS | REGIME | MIXED | MIXED | REGIME | no |
