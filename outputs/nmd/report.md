# NMD: the passages' names at the next hop (declared in docs/NMD_NAME_LINK_DIAG.md)

## Call on musique/s1fit: NO_SIGNAL (RARE)

No names term is built. U2r's untitled names do not reach musique's unlinked next hop: the names line moves to its open forms (prefix-merged titled names, names as entity nodes), each read by this census before anything is built from it.

Rule: SIGNAL iff the share of unlinked targets that share a CN_u name with their anchor >= 0.1 and its rrf-decile-matched lift >= 2.0, with at least 200 A cases.

## CN_u from the anchor (the rule's read)

| carve | A cases | unlinked targets | named | candidates named | lift | matched lift | named at two links | named beyond two | linked targets also named | reach with two links | AUC weight | matched AUC | AUC rrf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| musique/s1fit | 3022 | 2536 | 0.0197 | 0.0005 | 38.685 | 12.331 | 0.0284 | 0.0082 | 0.3677 | 0.5726 | 0.5087 | 0.5083 | 0.8843 |
| 2wiki/fit | 4354 | 2197 | 0.0032 | 0.0006 | 5.434 | 3.335 | 0.0182 | 0.0000 | 0.0592 | 0.1748 | 0.5015 | 0.5013 | 0.5482 |
| hotpotqa/fit | 2830 | 608 | 0.0263 | 0.0007 | 35.632 | 13.894 | 0.0466 | 0.0000 | 0.0594 | 0.5641 | 0.5133 | 0.5129 | 0.8406 |

## Every form (shares of the unlinked targets and candidates)

| carve | names | from | named | candidates named | lift | matched lift | named beyond two links |
|---|---|---|---:|---:|---:|---:|---:|
| musique/s1fit | CN_u | anchor | 0.0197 | 0.0005 | 38.685 | 12.331 | 0.0082 |
| musique/s1fit | CN_u | top five | 0.1092 | 0.0074 | 14.801 | 5.636 | 0.0924 |
| musique/s1fit | CN_all | anchor | 0.0225 | 0.0006 | 35.396 | 11.331 | 0.0082 |
| musique/s1fit | CN_all | top five | 0.1266 | 0.0092 | 13.817 | - | 0.1043 |
| 2wiki/fit | CN_u | anchor | 0.0032 | 0.0006 | 5.434 | 3.335 | 0.0000 |
| 2wiki/fit | CN_u | top five | 0.0205 | 0.0046 | 4.474 | 2.275 | 0.0182 |
| 2wiki/fit | CN_all | anchor | 0.0046 | 0.0023 | 1.989 | 1.090 | 0.0000 |
| 2wiki/fit | CN_all | top five | 0.1579 | 0.0227 | 6.965 | - | 0.1721 |
| hotpotqa/fit | CN_u | anchor | 0.0263 | 0.0007 | 35.632 | 13.894 | 0.0000 |
| hotpotqa/fit | CN_u | top five | 0.0395 | 0.0064 | 6.127 | 3.011 | 0.0075 |
| hotpotqa/fit | CN_all | anchor | 0.0444 | 0.0029 | 15.558 | 7.976 | 0.0038 |
| hotpotqa/fit | CN_all | top five | 0.1201 | 0.0252 | 4.771 | - | 0.0604 |

## CN_u weight from the anchor (shares of the unlinked rows; lift = targets over candidates)

| carve | <0.1: targets, candidates, lift | 0.1-0.5: targets, candidates, lift | >=0.5: targets, candidates, lift |
|---|---|---|---|
| musique/s1fit | 0.0142, 0.0003, 41.312 | 0.0055, 0.0002, 36.063 | 0.0000, 0.0000, 0.000 |
| 2wiki/fit | 0.0014, 0.0004, 3.609 | 0.0018, 0.0002, 9.363 | 0.0000, 0.0000, 0.000 |
| hotpotqa/fit | 0.0099, 0.0005, 20.167 | 0.0132, 0.0002, 57.073 | 0.0033, 0.0000, 176.233 |

## Per case

| carve | candidates (median) | unlinked (median) | named from the anchor (median, p90) | named from the top five (median, p90) |
|---|---:|---:|---|---|
| musique/s1fit | 5068.5 | 5018.0 | 0.0, 9.0 | 35.0, 75.9 |
| 2wiki/fit | 1126.0 | 1086.0 | 0.0, 1.0 | 2.0, 12.0 |
| hotpotqa/fit | 718.0 | 685.0 | 0.0, 1.0 | 2.0, 13.0 |

## Fixed orders of the unlinked candidates (the unlinked targets' ranks)

| carve | order | ranked | T2 | T10 | median | MRR |
|---|---|---:|---:|---:|---:|---:|
| musique/s1fit | rrf | 2536 | 0.0974 | 0.2756 | 49.0 | 0.1239 |
| musique/s1fit | names_first | 2536 | 0.0690 | 0.2449 | 47.0 | 0.1005 |
| musique/s1fit | names_w | 2536 | 0.0686 | 0.2461 | 47.0 | 0.1006 |
| 2wiki/fit | rrf | 2197 | 0.1124 | 0.3136 | 310.0 | 0.1354 |
| 2wiki/fit | names_first | 2197 | 0.1061 | 0.3077 | 295.0 | 0.1292 |
| 2wiki/fit | names_w | 2197 | 0.1061 | 0.3077 | 295.0 | 0.1292 |
| hotpotqa/fit | rrf | 608 | 0.2056 | 0.4457 | 14.0 | 0.2290 |
| hotpotqa/fit | names_first | 608 | 0.2039 | 0.4474 | 13.0 | 0.2286 |
| hotpotqa/fit | names_w | 608 | 0.2023 | 0.4474 | 13.0 | 0.2266 |

## By musique hop (CN_u from the anchor)

| carve | hop | A cases | linked | unlinked | named | lift |
|---|---|---:|---:|---:|---:|---:|
| musique/s1fit | 2hop | 2017 | 0.4100 | 1190 | 0.0227 | 42.010 |
| musique/s1fit | 3hop | 805 | 0.2683 | 949 | 0.0221 | 45.361 |
| musique/s1fit | 4hop | 200 | 0.1798 | 397 | 0.0050 | 15.321 |
