# Universal-GNN v2, one checkpoint over six datasets

**STAGE: authorization_stage_2_2026_09_22 (amendment_5_2026_09_23)**

**ORIGINAL PILOT STATUS: PILOT_FAILED** -- the pilot gate was read once and both arms failed; that status is
terminal and is not reopened here.

**POST-PILOT REPLICATION STATUS: GNN_REPLICATION_ONLY** -- the three-seed replication of `u_gnn_v2_ef` held
every frozen cell; that is what opened this stage.

**THIS STAGE HAS NO GATE.** stage 2 is a measurement, not a selection. There is no threshold, no pass or fail and no new arm to choose. A per-dataset regression against the M3B GAT is REPORTED as it falls, never repaired inside this stage; what to do about it is a new dated declaration filed before its fit.

One checkpoint, one contract, six datasets: `u_gnn_v2_ef` (420932 parameters, hidden 128,
K_REL 4) fitted from scratch three times (seeds 0, 1, 2) on the union of the six fit carves, each batch slot
drawing a dataset uniformly among the six and then a query of that dataset. No dataset identity feature, no
per-dataset head, no router, no per-dataset loss weight. Early stopping on the macro select recall@5 over the
six select carves.

## 1. What was fitted

| checkpoint | seed | best epoch | epochs run | select macro R@5 by epoch | fit hours | state sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| `u_gnn_v2_ef__H128__six__s0` | 0 | 3 | 6 | 0.8008 / 0.8099 / 0.8158 / **0.8166** / 0.8158 / 0.8091 | 29.77 | `88f56eeffbe5` |
| `u_gnn_v2_ef__H128__six__s1` | 1 | 5 | 6 | 0.7965 / 0.8087 / 0.8147 / 0.8163 / 0.8169 / **0.8173** | 29.55 | `aabe9110122b` |
| `u_gnn_v2_ef__H128__six__s2` | 2 | 4 | 6 | 0.7980 / 0.8111 / 0.8135 / 0.8150 / **0.8163** / 0.8154 | 25.35 | `ac4bf9c2b186` |

Epochs count from 0, so epoch 5 is the last of the 6 the rule allows. The select-carve
numbers are the early-stopping signal of the fit and never a result.
Seed 1 was selected at that last epoch: still improving on the select carves when the cap ended the fit. No epoch was added.

## 2. The six datasets, whole populations

Three-seed mean +/- sd. The reference is the frozen M3B universal GAT on the same queries (three seeds,
not refitted); the paired column is the mean per-query difference of the two three-seed means with its 95%
percentile interval over 1000 paired bootstrap resamples (numpy default_rng(0)).

| dataset | split | queries | band | metric | this stage | M3B GAT | paired delta [95% CI] | pool ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | dev | 39138 | NOT_READ | hit@1 | 0.9090 +/- 0.0127 | 0.8411 +/- 0.0091 | +0.0679 [+0.0654, +0.0703] |  |
| metaqa | dev | 39138 | NOT_READ | recall@5 | 0.7859 +/- 0.0018 | 0.7645 +/- 0.0011 | +0.0213 [+0.0204, +0.0223] | 0.8094 |
| 2wiki | dev | 12576 | READ | recall@5 | 0.8886 +/- 0.0029 | 0.8836 +/- 0.0040 | +0.0050 [+0.0033, +0.0066] | 0.9637 |
| 2wiki | dev | 12576 | READ | full_coverage@5 | 0.7609 +/- 0.0041 | 0.7495 +/- 0.0072 | +0.0114 [+0.0082, +0.0144] |  |
| squad | dev | 11873 | CONTROL | recall@5 | 0.8908 +/- 0.0095 | 0.8922 +/- 0.0061 | -0.0014 [-0.0036, +0.0010] | 0.9798 |
| hotpotqa | validation | 7405 | READ | recall@5 | 0.8967 +/- 0.0018 | 0.8975 +/- 0.0044 | -0.0008 [-0.0032, +0.0016] | 0.9800 |
| hotpotqa | validation | 7405 | READ | full_coverage@5 | 0.8343 +/- 0.0030 | 0.8367 +/- 0.0074 | -0.0024 [-0.0064, +0.0018] |  |
| musique | dev | 2417 | READ | recall@5 | 0.5242 +/- 0.0099 | 0.5289 +/- 0.0069 | -0.0046 [-0.0095, +0.0003] | 0.9207 |
| musique | dev | 2417 | READ | full_coverage@5 | 0.2317 +/- 0.0126 | 0.2322 +/- 0.0079 | -0.0006 [-0.0077, +0.0068] |  |
| webqsp | train_holdout | 1503 | NOT_READ | hit@1 | 0.6299 +/- 0.0050 | 0.5928 +/- 0.0096 | +0.0370 [+0.0233, +0.0510] |  |
| webqsp | train_holdout | 1503 | NOT_READ | recall@5 | 0.6194 +/- 0.0088 | 0.6073 +/- 0.0049 | +0.0120 [+0.0034, +0.0212] | 0.7649 |

The same cells per seed. Each column is headed by the epoch at which that checkpoint was selected on the
select carves; the spread in the table above is the spread of these three values.

| dataset | metric | seed 0, epoch 3 | seed 1, epoch 5, at the cap | seed 2, epoch 4 | mean +/- sd | max - min |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | hit@1 | 0.8913 | 0.9159 | 0.9200 | 0.9090 +/- 0.0127 | 0.0287 |
| metaqa | recall@5 | 0.7834 | 0.7863 | 0.7878 | 0.7859 +/- 0.0018 | 0.0044 |
| 2wiki | recall@5 | 0.8906 | 0.8907 | 0.8845 | 0.8886 +/- 0.0029 | 0.0062 |
| 2wiki | full_coverage@5 | 0.7588 | 0.7666 | 0.7573 | 0.7609 +/- 0.0041 | 0.0093 |
| squad | recall@5 | 0.9038 | 0.8812 | 0.8873 | 0.8908 +/- 0.0095 | 0.0226 |
| hotpotqa | recall@5 | 0.8993 | 0.8954 | 0.8955 | 0.8967 +/- 0.0018 | 0.0039 |
| hotpotqa | full_coverage@5 | 0.8363 | 0.8300 | 0.8365 | 0.8343 +/- 0.0030 | 0.0065 |
| musique | recall@5 | 0.5380 | 0.5150 | 0.5198 | 0.5242 +/- 0.0099 | 0.0230 |
| musique | full_coverage@5 | 0.2491 | 0.2197 | 0.2263 | 0.2317 +/- 0.0126 | 0.0294 |
| webqsp | hit@1 | 0.6367 | 0.6281 | 0.6248 | 0.6299 +/- 0.0050 | 0.0119 |
| webqsp | recall@5 | 0.6289 | 0.6215 | 0.6076 | 0.6194 +/- 0.0088 | 0.0213 |

Every reported metric, whole populations, against the M3B GAT on the same queries. MRR is the audited value:
per query it was recomputed from the stored first-gold rank, matched the stored value within 1e-12, and lay
between hit@1 and 1, for every checkpoint (`mrr_audit` in each eval record; the audit column of section 9).
full_coverage@K is the GraphER-compatible PR@K (section 8).

| dataset | metric | this stage | M3B GAT | paired delta [95% CI] |
| --- | --- | --- | --- | --- |
| metaqa | recall@1 | 0.4883 +/- 0.0015 | 0.4541 +/- 0.0017 | +0.0342 [+0.0326, +0.0359] |
| metaqa | recall@5 | 0.7859 +/- 0.0018 | 0.7645 +/- 0.0011 | +0.0213 [+0.0204, +0.0223] |
| metaqa | recall@10 | 0.8626 +/- 0.0020 | 0.8465 +/- 0.0013 | +0.0161 [+0.0153, +0.0170] |
| metaqa | recall@20 | 0.9148 +/- 0.0022 | 0.9024 +/- 0.0013 | +0.0124 [+0.0117, +0.0130] |
| metaqa | hit@1 | 0.9090 +/- 0.0127 | 0.8411 +/- 0.0091 | +0.0679 [+0.0654, +0.0703] |
| metaqa | mrr | 0.9357 +/- 0.0084 | 0.8937 +/- 0.0059 | +0.0420 [+0.0404, +0.0436] |
| metaqa | ndcg@5 | 0.9225 +/- 0.0090 | 0.8717 +/- 0.0062 | +0.0508 [+0.0494, +0.0524] |
| metaqa | ndcg@20 | 0.9289 +/- 0.0060 | 0.8874 +/- 0.0042 | +0.0415 [+0.0402, +0.0427] |
| metaqa | full_coverage@5 | 0.6670 +/- 0.0004 | 0.6414 +/- 0.0002 | +0.0256 [+0.0241, +0.0272] |
| metaqa | full_coverage@20 | 0.8359 +/- 0.0025 | 0.8192 +/- 0.0010 | +0.0167 [+0.0155, +0.0179] |
| 2wiki | recall@1 | 0.3902 +/- 0.0134 | 0.3920 +/- 0.0037 | -0.0018 [-0.0034, -0.0001] |
| 2wiki | recall@5 | 0.8886 +/- 0.0029 | 0.8836 +/- 0.0040 | +0.0050 [+0.0033, +0.0066] |
| 2wiki | recall@10 | 0.9339 +/- 0.0015 | 0.9316 +/- 0.0026 | +0.0024 [+0.0012, +0.0034] |
| 2wiki | recall@20 | 0.9522 +/- 0.0009 | 0.9513 +/- 0.0012 | +0.0009 [+0.0001, +0.0016] |
| 2wiki | hit@1 | 0.8834 +/- 0.0280 | 0.8867 +/- 0.0086 | -0.0033 [-0.0067, +0.0001] |
| 2wiki | mrr | 0.9280 +/- 0.0160 | 0.9296 +/- 0.0047 | -0.0016 [-0.0035, +0.0003] |
| 2wiki | ndcg@5 | 0.8653 +/- 0.0091 | 0.8614 +/- 0.0011 | +0.0039 [+0.0024, +0.0054] |
| 2wiki | ndcg@20 | 0.8917 +/- 0.0084 | 0.8894 +/- 0.0008 | +0.0022 [+0.0010, +0.0034] |
| 2wiki | full_coverage@5 | 0.7609 +/- 0.0041 | 0.7495 +/- 0.0072 | +0.0114 [+0.0082, +0.0144] |
| 2wiki | full_coverage@20 | 0.8843 +/- 0.0021 | 0.8827 +/- 0.0023 | +0.0015 [-0.0000, +0.0031] |
| squad | recall@1 | 0.7237 +/- 0.0140 | 0.7304 +/- 0.0079 | -0.0067 [-0.0096, -0.0035] |
| squad | recall@5 | 0.8908 +/- 0.0095 | 0.8922 +/- 0.0061 | -0.0014 [-0.0036, +0.0010] |
| squad | recall@10 | 0.9275 +/- 0.0054 | 0.9288 +/- 0.0052 | -0.0013 [-0.0031, +0.0007] |
| squad | recall@20 | 0.9548 +/- 0.0035 | 0.9569 +/- 0.0023 | -0.0021 [-0.0036, -0.0006] |
| squad | hit@1 | 0.7237 +/- 0.0140 | 0.7304 +/- 0.0079 | -0.0067 [-0.0096, -0.0035] |
| squad | mrr | 0.7991 +/- 0.0114 | 0.8035 +/- 0.0070 | -0.0044 [-0.0062, -0.0025] |
| squad | ndcg@5 | 0.8164 +/- 0.0115 | 0.8201 +/- 0.0071 | -0.0037 [-0.0053, -0.0019] |
| squad | ndcg@20 | 0.8353 +/- 0.0097 | 0.8392 +/- 0.0061 | -0.0038 [-0.0053, -0.0023] |
| squad | full_coverage@5 | 0.8908 +/- 0.0095 | 0.8922 +/- 0.0061 | -0.0014 [-0.0036, +0.0010] |
| squad | full_coverage@20 | 0.9548 +/- 0.0035 | 0.9569 +/- 0.0023 | -0.0021 [-0.0036, -0.0006] |
| hotpotqa | recall@1 | 0.4160 +/- 0.0153 | 0.4229 +/- 0.0034 | -0.0069 [-0.0092, -0.0047] |
| hotpotqa | recall@5 | 0.8967 +/- 0.0018 | 0.8975 +/- 0.0044 | -0.0008 [-0.0032, +0.0016] |
| hotpotqa | recall@10 | 0.9380 +/- 0.0027 | 0.9379 +/- 0.0024 | +0.0001 [-0.0017, +0.0019] |
| hotpotqa | recall@20 | 0.9605 +/- 0.0023 | 0.9609 +/- 0.0011 | -0.0004 [-0.0017, +0.0008] |
| hotpotqa | hit@1 | 0.8320 +/- 0.0306 | 0.8458 +/- 0.0069 | -0.0138 [-0.0184, -0.0093] |
| hotpotqa | mrr | 0.8898 +/- 0.0171 | 0.8969 +/- 0.0044 | -0.0071 [-0.0098, -0.0045] |
| hotpotqa | ndcg@5 | 0.8537 +/- 0.0086 | 0.8582 +/- 0.0035 | -0.0045 [-0.0066, -0.0024] |
| hotpotqa | ndcg@20 | 0.8774 +/- 0.0089 | 0.8817 +/- 0.0027 | -0.0043 [-0.0059, -0.0027] |
| hotpotqa | full_coverage@5 | 0.8343 +/- 0.0030 | 0.8367 +/- 0.0074 | -0.0024 [-0.0064, +0.0018] |
| hotpotqa | full_coverage@20 | 0.9382 +/- 0.0040 | 0.9391 +/- 0.0020 | -0.0009 [-0.0031, +0.0014] |
| musique | recall@1 | 0.2566 +/- 0.0081 | 0.2671 +/- 0.0005 | -0.0106 [-0.0149, -0.0059] |
| musique | recall@5 | 0.5242 +/- 0.0099 | 0.5289 +/- 0.0069 | -0.0046 [-0.0095, +0.0003] |
| musique | recall@10 | 0.6078 +/- 0.0099 | 0.6164 +/- 0.0090 | -0.0085 [-0.0132, -0.0033] |
| musique | recall@20 | 0.6846 +/- 0.0080 | 0.6902 +/- 0.0080 | -0.0056 [-0.0100, -0.0008] |
| musique | hit@1 | 0.6267 +/- 0.0213 | 0.6529 +/- 0.0018 | -0.0262 [-0.0368, -0.0150] |
| musique | mrr | 0.7321 +/- 0.0147 | 0.7489 +/- 0.0027 | -0.0168 [-0.0234, -0.0099] |
| musique | ndcg@5 | 0.5243 +/- 0.0118 | 0.5342 +/- 0.0051 | -0.0099 [-0.0143, -0.0053] |
| musique | ndcg@20 | 0.5866 +/- 0.0109 | 0.5974 +/- 0.0056 | -0.0107 [-0.0146, -0.0068] |
| musique | full_coverage@5 | 0.2317 +/- 0.0126 | 0.2322 +/- 0.0079 | -0.0006 [-0.0077, +0.0068] |
| musique | full_coverage@20 | 0.4075 +/- 0.0131 | 0.4099 +/- 0.0135 | -0.0023 [-0.0099, +0.0063] |
| webqsp | recall@1 | 0.3652 +/- 0.0059 | 0.3397 +/- 0.0031 | +0.0256 [+0.0151, +0.0354] |
| webqsp | recall@5 | 0.6194 +/- 0.0088 | 0.6073 +/- 0.0049 | +0.0120 [+0.0034, +0.0212] |
| webqsp | recall@10 | 0.6873 +/- 0.0055 | 0.6768 +/- 0.0036 | +0.0105 [+0.0032, +0.0175] |
| webqsp | recall@20 | 0.7342 +/- 0.0057 | 0.7198 +/- 0.0039 | +0.0145 [+0.0085, +0.0203] |
| webqsp | hit@1 | 0.6299 +/- 0.0050 | 0.5928 +/- 0.0096 | +0.0370 [+0.0233, +0.0510] |
| webqsp | mrr | 0.7050 +/- 0.0063 | 0.6826 +/- 0.0056 | +0.0223 [+0.0122, +0.0319] |
| webqsp | ndcg@5 | 0.6601 +/- 0.0083 | 0.6342 +/- 0.0029 | +0.0259 [+0.0170, +0.0349] |
| webqsp | ndcg@20 | 0.6699 +/- 0.0071 | 0.6457 +/- 0.0014 | +0.0242 [+0.0169, +0.0308] |
| webqsp | full_coverage@5 | 0.5045 +/- 0.0086 | 0.4875 +/- 0.0053 | +0.0171 [+0.0064, +0.0277] |
| webqsp | full_coverage@20 | 0.6094 +/- 0.0033 | 0.5921 +/- 0.0030 | +0.0173 [+0.0100, +0.0248] |

Per-seed values, every reported metric (recall@1/5/10/20, hit@1, mrr, ndcg@5/20, full_coverage@5/20),
the non-MP references (`gat_no_mp_v1`, `qls_u_sota_v1`) and the fixed rrf are in
`outputs/universal_v2/six/read_record.json`.

## 3. The trio checkpoint beside the joint one

Declared in amendment_6_2026_09_23 before any stage-2 weight was fitted. The trio column is the pilot's
already-filed `u_gnn_v2_ef__H128__s0` read from `outputs/universal_v2/eval/*.npz`; those arrays are not
rescored here. This is a different training set, not a seed comparison: three datasets against six,
and the single pilot seed that was scored on the whole population against this stage's three seeds.
webqsp, hotpotqa and musique have no trio checkpoint, so they have no column here rather than a
substitute one. No threshold is applied: the stage has no gate.

| dataset | metric | six datasets, 3 seeds | trio pilot, 1 seed, 3 datasets | paired six - trio [95% CI] |
| --- | --- | --- | --- | --- |
| metaqa | hit@1 | 0.9090 +/- 0.0127 | 0.9167 | -0.0076 [-0.0096, -0.0056] |
| metaqa | recall@5 | 0.7859 +/- 0.0018 | 0.7889 | -0.0030 [-0.0038, -0.0023] |
| 2wiki | recall@5 | 0.8886 +/- 0.0029 | 0.8948 | -0.0062 [-0.0086, -0.0039] |
| 2wiki | full_coverage@5 | 0.7609 +/- 0.0041 | 0.7665 | -0.0055 [-0.0107, -0.0007] |
| squad | recall@5 | 0.8908 +/- 0.0095 | 0.9014 | -0.0106 [-0.0134, -0.0079] |

Every reported metric and both trio halves are in the read record under `trio_context`.

## 4. The trio halves

The three pilot datasets carry the V2_GATE / V2_HELD_CONFIRMATION split of the pilot. Both halves were
already read there, so neither is a first reading here; they are printed separately because the stage
reports what it measures, and nothing in this stage is compared to a threshold.

| dataset | metric | V2_GATE | V2_HELD_CONFIRMATION | whole |
| --- | --- | --- | --- | --- |
| metaqa | hit@1 | 0.9092 +/- 0.0123 | 0.9088 +/- 0.0131 | 0.9090 +/- 0.0127 |
| metaqa | recall@5 | 0.7814 +/- 0.0017 | 0.7904 +/- 0.0019 | 0.7859 +/- 0.0018 |
| 2wiki | recall@5 | 0.8891 +/- 0.0032 | 0.8880 +/- 0.0026 | 0.8886 +/- 0.0029 |
| 2wiki | full_coverage@5 | 0.7607 +/- 0.0047 | 0.7611 +/- 0.0035 | 0.7609 +/- 0.0041 |
| squad | recall@5 | 0.8869 +/- 0.0091 | 0.8946 +/- 0.0102 | 0.8908 +/- 0.0095 |

## 5. Slices

Each slice is a mask over the same population: metaqa by hop, the gold BFS-distance buckets in the STRUCT
view, and the single-gold / multi-gold split. The paired column is against the M3B GAT on the same slice.

| dataset | slice | queries | metric | this stage | M3B GAT | paired delta [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | 1hop | 9992 | hit@1 | 0.9339 +/- 0.0048 | 0.8963 +/- 0.0056 | +0.0377 [+0.0336, +0.0418] |
| metaqa | 1hop | 9992 | recall@5 | 0.9668 +/- 0.0012 | 0.9556 +/- 0.0006 | +0.0112 [+0.0094, +0.0128] |
| metaqa | 2hop | 14872 | hit@1 | 0.9499 +/- 0.0056 | 0.8863 +/- 0.0042 | +0.0636 [+0.0599, +0.0674] |
| metaqa | 2hop | 14872 | recall@5 | 0.8484 +/- 0.0008 | 0.8316 +/- 0.0017 | +0.0168 [+0.0155, +0.0181] |
| metaqa | 3hop | 14274 | hit@1 | 0.8490 +/- 0.0296 | 0.7554 +/- 0.0231 | +0.0936 [+0.0886, +0.0981] |
| metaqa | 3hop | 14274 | recall@5 | 0.5940 +/- 0.0045 | 0.5609 +/- 0.0047 | +0.0331 [+0.0310, +0.0350] |
| metaqa | gold_at_seed | 1122 | hit@1 | 0.8898 +/- 0.0191 | 0.8134 +/- 0.0198 | +0.0764 [+0.0609, +0.0936] |
| metaqa | gold_at_seed | 1122 | recall@5 | 0.5872 +/- 0.0055 | 0.5571 +/- 0.0048 | +0.0302 [+0.0239, +0.0365] |
| metaqa | gold_1_hop | 20241 | hit@1 | 0.9224 +/- 0.0082 | 0.8729 +/- 0.0092 | +0.0494 [+0.0464, +0.0525] |
| metaqa | gold_1_hop | 20241 | recall@5 | 0.8081 +/- 0.0010 | 0.7919 +/- 0.0013 | +0.0162 [+0.0149, +0.0175] |
| metaqa | gold_2_hops | 14171 | hit@1 | 0.9399 +/- 0.0093 | 0.8666 +/- 0.0068 | +0.0732 [+0.0691, +0.0770] |
| metaqa | gold_2_hops | 14171 | recall@5 | 0.8286 +/- 0.0007 | 0.8091 +/- 0.0011 | +0.0195 [+0.0180, +0.0209] |
| metaqa | gold_3_or_more | 3118 | hit@1 | 0.8311 +/- 0.0578 | 0.6594 +/- 0.0445 | +0.1717 [+0.1593, +0.1836] |
| metaqa | gold_3_or_more | 3118 | recall@5 | 0.6414 +/- 0.0125 | 0.5782 +/- 0.0112 | +0.0632 [+0.0571, +0.0690] |
| metaqa | no_gold_in_pool | 486 | hit@1 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| metaqa | no_gold_in_pool | 486 | recall@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| metaqa | single_gold | 15061 | hit@1 | 0.9178 +/- 0.0007 | 0.8509 +/- 0.0034 | +0.0669 [+0.0631, +0.0713] |
| metaqa | single_gold | 15061 | recall@5 | 0.9803 +/- 0.0002 | 0.9725 +/- 0.0011 | +0.0078 [+0.0065, +0.0094] |
| metaqa | multi_gold | 24077 | hit@1 | 0.9036 +/- 0.0202 | 0.8350 +/- 0.0137 | +0.0686 [+0.0652, +0.0718] |
| metaqa | multi_gold | 24077 | recall@5 | 0.6642 +/- 0.0031 | 0.6345 +/- 0.0025 | +0.0297 [+0.0284, +0.0311] |
| 2wiki | gold_at_seed | 12510 | recall@5 | 0.8919 +/- 0.0030 | 0.8869 +/- 0.0039 | +0.0050 [+0.0032, +0.0067] |
| 2wiki | gold_at_seed | 12510 | full_coverage@5 | 0.7645 +/- 0.0042 | 0.7531 +/- 0.0071 | +0.0114 [+0.0080, +0.0147] |
| 2wiki | gold_1_hop | 21 | recall@5 | 0.4286 +/- 0.0389 | 0.3968 +/- 0.0405 | +0.0317 [-0.0238, +0.0952] |
| 2wiki | gold_1_hop | 21 | full_coverage@5 | 0.2381 +/- 0.0389 | 0.1905 +/- 0.0389 | +0.0476 [+0.0000, +0.0952] |
| 2wiki | gold_2_hops | 15 | recall@5 | 0.1111 +/- 0.0314 | 0.0889 +/- 0.0157 | +0.0222 [-0.0333, +0.0778] |
| 2wiki | gold_2_hops | 15 | full_coverage@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| 2wiki | gold_3_or_more | 26 | recall@5 | 0.2404 +/- 0.0208 | 0.2596 +/- 0.0720 | -0.0192 [-0.0641, +0.0256] |
| 2wiki | gold_3_or_more | 26 | full_coverage@5 | 0.0385 +/- 0.0000 | 0.0513 +/- 0.0181 | -0.0128 [-0.0641, +0.0256] |
| 2wiki | no_gold_in_pool | 4 | recall@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| 2wiki | no_gold_in_pool | 4 | full_coverage@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| 2wiki | multi_gold | 12576 | recall@5 | 0.8886 +/- 0.0029 | 0.8836 +/- 0.0040 | +0.0050 [+0.0033, +0.0066] |
| 2wiki | multi_gold | 12576 | full_coverage@5 | 0.7609 +/- 0.0041 | 0.7495 +/- 0.0072 | +0.0114 [+0.0082, +0.0144] |
| squad | gold_at_seed | 11060 | recall@5 | 0.9483 +/- 0.0104 | 0.9498 +/- 0.0066 | -0.0016 [-0.0037, +0.0006] |
| squad | gold_1_hop | 133 | recall@5 | 0.1905 +/- 0.0128 | 0.1880 +/- 0.0162 | +0.0025 [-0.0376, +0.0451] |
| squad | gold_2_hops | 39 | recall@5 | 0.1197 +/- 0.0436 | 0.0684 +/- 0.0121 | +0.0513 [-0.0085, +0.1111] |
| squad | gold_3_or_more | 401 | recall@5 | 0.1455 +/- 0.0124 | 0.1496 +/- 0.0061 | -0.0042 [-0.0283, +0.0200] |
| squad | no_gold_in_pool | 240 | recall@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| squad | single_gold | 11873 | recall@5 | 0.8908 +/- 0.0095 | 0.8922 +/- 0.0061 | -0.0014 [-0.0036, +0.0010] |
| hotpotqa | gold_at_seed | 7160 | recall@5 | 0.9219 +/- 0.0019 | 0.9224 +/- 0.0041 | -0.0005 [-0.0028, +0.0017] |
| hotpotqa | gold_at_seed | 7160 | full_coverage@5 | 0.8612 +/- 0.0028 | 0.8636 +/- 0.0071 | -0.0024 [-0.0066, +0.0015] |
| hotpotqa | gold_1_hop | 109 | recall@5 | 0.3119 +/- 0.0065 | 0.3119 +/- 0.0389 | +0.0000 [-0.0260, +0.0275] |
| hotpotqa | gold_1_hop | 109 | full_coverage@5 | 0.0948 +/- 0.0229 | 0.0856 +/- 0.0426 | +0.0092 [-0.0153, +0.0336] |
| hotpotqa | gold_2_hops | 47 | recall@5 | 0.0674 +/- 0.0133 | 0.1028 +/- 0.0050 | -0.0355 [-0.0816, +0.0106] |
| hotpotqa | gold_2_hops | 47 | full_coverage@5 | 0.0213 +/- 0.0000 | 0.0567 +/- 0.0100 | -0.0355 [-0.0851, +0.0071] |
| hotpotqa | gold_3_or_more | 27 | recall@5 | 0.0802 +/- 0.0231 | 0.1173 +/- 0.0231 | -0.0370 [-0.0864, +0.0062] |
| hotpotqa | gold_3_or_more | 27 | full_coverage@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| hotpotqa | no_gold_in_pool | 62 | recall@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| hotpotqa | no_gold_in_pool | 62 | full_coverage@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| hotpotqa | multi_gold | 7405 | recall@5 | 0.8967 +/- 0.0018 | 0.8975 +/- 0.0044 | -0.0008 [-0.0032, +0.0016] |
| hotpotqa | multi_gold | 7405 | full_coverage@5 | 0.8343 +/- 0.0030 | 0.8367 +/- 0.0074 | -0.0024 [-0.0064, +0.0018] |
| musique | gold_at_seed | 2197 | recall@5 | 0.5659 +/- 0.0116 | 0.5725 +/- 0.0075 | -0.0066 [-0.0118, -0.0012] |
| musique | gold_at_seed | 2197 | full_coverage@5 | 0.2535 +/- 0.0140 | 0.2540 +/- 0.0085 | -0.0005 [-0.0080, +0.0076] |
| musique | gold_1_hop | 110 | recall@5 | 0.1336 +/- 0.0190 | 0.1230 +/- 0.0246 | +0.0106 [-0.0124, +0.0326] |
| musique | gold_1_hop | 110 | full_coverage@5 | 0.0212 +/- 0.0113 | 0.0303 +/- 0.0043 | -0.0091 [-0.0333, +0.0091] |
| musique | gold_2_hops | 74 | recall@5 | 0.0920 +/- 0.0222 | 0.0702 +/- 0.0037 | +0.0218 [-0.0038, +0.0503] |
| musique | gold_2_hops | 74 | full_coverage@5 | 0.0045 +/- 0.0064 | 0.0000 +/- 0.0000 | +0.0045 [+0.0000, +0.0135] |
| musique | gold_3_or_more | 24 | recall@5 | 0.0972 +/- 0.0177 | 0.0741 +/- 0.0269 | +0.0231 [-0.0231, +0.0718] |
| musique | gold_3_or_more | 24 | full_coverage@5 | 0.0139 +/- 0.0196 | 0.0000 +/- 0.0000 | +0.0139 [+0.0000, +0.0417] |
| musique | no_gold_in_pool | 12 | recall@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| musique | no_gold_in_pool | 12 | full_coverage@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| musique | multi_gold | 2417 | recall@5 | 0.5242 +/- 0.0099 | 0.5289 +/- 0.0069 | -0.0046 [-0.0095, +0.0003] |
| musique | multi_gold | 2417 | full_coverage@5 | 0.2317 +/- 0.0126 | 0.2322 +/- 0.0079 | -0.0006 [-0.0077, +0.0068] |
| webqsp | gold_at_seed | 260 | hit@1 | 0.8167 +/- 0.0390 | 0.7667 +/- 0.0179 | +0.0500 [+0.0141, +0.0872] |
| webqsp | gold_at_seed | 260 | recall@5 | 0.7298 +/- 0.0106 | 0.7004 +/- 0.0157 | +0.0294 [+0.0066, +0.0540] |
| webqsp | gold_1_hop | 924 | hit@1 | 0.6988 +/- 0.0077 | 0.6620 +/- 0.0194 | +0.0368 [+0.0188, +0.0552] |
| webqsp | gold_1_hop | 924 | recall@5 | 0.7044 +/- 0.0087 | 0.6989 +/- 0.0011 | +0.0055 [-0.0052, +0.0156] |
| webqsp | gold_2_hops | 182 | hit@1 | 0.4597 +/- 0.0052 | 0.4066 +/- 0.0179 | +0.0531 [+0.0128, +0.0952] |
| webqsp | gold_2_hops | 182 | recall@5 | 0.4551 +/- 0.0237 | 0.4298 +/- 0.0164 | +0.0253 [-0.0066, +0.0581] |
| webqsp | gold_3_or_more | 19 | hit@1 | 0.2632 +/- 0.0430 | 0.3158 +/- 0.0744 | -0.0526 [-0.1228, +0.0175] |
| webqsp | gold_3_or_more | 19 | recall@5 | 0.3918 +/- 0.0538 | 0.3538 +/- 0.0789 | +0.0380 [-0.0791, +0.1637] |
| webqsp | no_gold_in_pool | 118 | hit@1 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| webqsp | no_gold_in_pool | 118 | recall@5 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] |
| webqsp | single_gold | 763 | hit@1 | 0.5426 +/- 0.0123 | 0.4985 +/- 0.0054 | +0.0441 [+0.0249, +0.0633] |
| webqsp | single_gold | 763 | recall@5 | 0.7322 +/- 0.0152 | 0.7256 +/- 0.0107 | +0.0066 [-0.0074, +0.0205] |
| webqsp | multi_gold | 740 | hit@1 | 0.7198 +/- 0.0045 | 0.6901 +/- 0.0189 | +0.0297 [+0.0081, +0.0505] |
| webqsp | multi_gold | 740 | recall@5 | 0.5030 +/- 0.0061 | 0.4853 +/- 0.0015 | +0.0177 [+0.0072, +0.0274] |

## 6. What the one contract carries on each substrate

The same 129 columns on six graphs. The three added datasets were scanned in full during this stage; the
trio rows are the pilot screen's sampled availability, which is the number filed for them. Nothing is
dropped, reordered or specialised anywhere: a column that no query of a dataset can populate stays in the
layout and arrives as its declared absent value with its availability mask.

| dataset | rows scanned | mean availability | min | unavailable (kept) | constant (kept) | source |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | 12021002 | 0.6102 | 0.0000 | 10 |  | pilot screen, sampled (199114 rows at stride 65) |
| 2wiki | 628354 | 0.4485 | 0.0000 | 27 |  | pilot screen, sampled (199114 rows at stride 65) |
| squad | 292994 | 0.4029 | 0.0000 | 26 |  | pilot screen, sampled (199114 rows at stride 65) |
| hotpotqa | 557195 | 0.4870 | 0.0000 | 26 | 1 | this stage, full scan of the fit carve |
| musique | 9616637 | 0.5558 | 0.0000 | 26 | 2 | this stage, full scan of the fit carve |
| webqsp | 2546884 | 0.5941 | 0.0028 | 0 | 2 | this stage, full scan of the fit carve |

## 7. Mechanism readouts

Per checkpoint, over the whole population: the mean |delta_s| over |base_z| ratio, the fraction of queries
whose top-1 leaves the fixed base score, the message-passing step gates and, where the arm has them, the
evidence-flow gates. These are the readouts measurement.mechanism_readouts declares; they are reported as
they fell and nothing is tuned on them.

The evidence-flow step moves its state only along typed STRUCT relation edges, and reports a gate of 0 for a
node with no incoming typed edge, whose state then stays at its initial value (`EvidenceFlow.step`). A gate2
column of 0.0000 therefore means the block found no typed edge to flow over on that substrate, not a gate the
fit learned to close; the initial evidence state still enters the readout there.
Every checkpoint reads 0.0000 on every evidence-flow step on: 2wiki, squad, hotpotqa, musique.

| dataset | checkpoint | delta_ratio | top1_changed | gate_step1 | gate_step2 | gate_step3 | gate2_step1 | gate2_step2 | gate2_step3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | `u_gnn_v2_ef__H128__six__s0` | 27.4945 | 0.9994 | 0.6293 | 0.5246 | 0.5025 | 0.8599 | 0.5971 | 0.5623 |
| metaqa | `u_gnn_v2_ef__H128__six__s1` | 35.4840 | 0.9981 | 0.5271 | 0.5522 | 0.5209 | 0.6491 | 0.7769 | 0.6627 |
| metaqa | `u_gnn_v2_ef__H128__six__s2` | 35.8481 | 0.9995 | 0.4892 | 0.4430 | 0.4203 | 0.8535 | 0.8964 | 0.9256 |
| 2wiki | `u_gnn_v2_ef__H128__six__s0` | 10.7681 | 0.2665 | 0.4663 | 0.3470 | 0.2863 | 0.0000 | 0.0000 | 0.0000 |
| 2wiki | `u_gnn_v2_ef__H128__six__s1` | 15.9802 | 0.4892 | 0.3941 | 0.3805 | 0.3696 | 0.0000 | 0.0000 | 0.0000 |
| 2wiki | `u_gnn_v2_ef__H128__six__s2` | 14.5343 | 0.5538 | 0.3752 | 0.3021 | 0.2801 | 0.0000 | 0.0000 | 0.0000 |
| squad | `u_gnn_v2_ef__H128__six__s0` | 11.8709 | 0.1351 | 0.2671 | 0.2050 | 0.1735 | 0.0000 | 0.0000 | 0.0000 |
| squad | `u_gnn_v2_ef__H128__six__s1` | 19.5396 | 0.1894 | 0.2748 | 0.2699 | 0.2642 | 0.0000 | 0.0000 | 0.0000 |
| squad | `u_gnn_v2_ef__H128__six__s2` | 16.4931 | 0.1796 | 0.2081 | 0.1868 | 0.1806 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | `u_gnn_v2_ef__H128__six__s0` | 11.0412 | 0.2392 | 0.4659 | 0.3360 | 0.2744 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | `u_gnn_v2_ef__H128__six__s1` | 16.5169 | 0.4328 | 0.3989 | 0.3775 | 0.3533 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | `u_gnn_v2_ef__H128__six__s2` | 15.0619 | 0.5615 | 0.3540 | 0.2903 | 0.2710 | 0.0000 | 0.0000 | 0.0000 |
| musique | `u_gnn_v2_ef__H128__six__s0` | 21.7505 | 0.3839 | 0.5740 | 0.3655 | 0.2861 | 0.0000 | 0.0000 | 0.0000 |
| musique | `u_gnn_v2_ef__H128__six__s1` | 29.4633 | 0.4688 | 0.4649 | 0.4253 | 0.3675 | 0.0000 | 0.0000 | 0.0000 |
| musique | `u_gnn_v2_ef__H128__six__s2` | 27.1628 | 0.4824 | 0.4212 | 0.3413 | 0.3211 | 0.0000 | 0.0000 | 0.0000 |
| webqsp | `u_gnn_v2_ef__H128__six__s0` | 23.1673 | 0.9674 | 0.5438 | 0.3651 | 0.3059 | 0.6590 | 0.4288 | 0.3930 |
| webqsp | `u_gnn_v2_ef__H128__six__s1` | 31.3160 | 0.9561 | 0.4153 | 0.3696 | 0.3247 | 0.4716 | 0.5825 | 0.5261 |
| webqsp | `u_gnn_v2_ef__H128__six__s2` | 29.4576 | 0.9634 | 0.3595 | 0.2816 | 0.2575 | 0.6236 | 0.6703 | 0.7086 |

## 8. Calibration against published systems

the M3B band verdicts carry over unchanged and are reprinted with every calibration row

PR@K in the GraphER tables is set coverage, not recall, and is not compared to our recall@K without saying so; the webqsp comparator to a published coverage number is the pool any_gold_at_pool column, never an average; webqsp is on train_holdout and its corpus ceiling (0.5255 reference level) is printed on the row.

Our `full_coverage@K` is the GraphER-compatible PR@K: per query it is 1 only when every gold is ranked in the
top K, counted over the whole gold set, so a gold the pool never reached counts against it
(`m3b_train.rank_metrics`). It is reported at K = 5 and K = 20 for every dataset, per seed in
`outputs/universal_v2/six/read_record.json`, beside recall and never merged with it; hotpotqa, 2wiki and
musique carry `full_coverage@5` as a headline metric.

| dataset | band verdict | headline | this stage | M3B GAT | any_gold_at_pool | recall ceiling@5 | candidates mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | **NOT_READ** | hit@1 | 0.9090 +/- 0.0127 | 0.8411 +/- 0.0091 | 0.9876 | 0.8094 | 2017 |
| metaqa | **NOT_READ** | recall@5 | 0.7859 +/- 0.0018 | 0.7645 +/- 0.0011 | 0.9876 | 0.8094 | 2017 |
| 2wiki | **READ** | recall@5 | 0.8886 +/- 0.0029 | 0.8836 +/- 0.0040 | 0.9997 | 0.9637 | 106 |
| 2wiki | **READ** | full_coverage@5 | 0.7609 +/- 0.0041 | 0.7495 +/- 0.0072 | 0.9997 | 0.9637 | 106 |
| squad | **CONTROL** | recall@5 | 0.8908 +/- 0.0095 | 0.8922 +/- 0.0061 | 0.9798 | 0.9798 | 50 |
| hotpotqa | **READ** | recall@5 | 0.8967 +/- 0.0018 | 0.8975 +/- 0.0044 | 0.9916 | 0.9800 | 94 |
| hotpotqa | **READ** | full_coverage@5 | 0.8343 +/- 0.0030 | 0.8367 +/- 0.0074 | 0.9916 | 0.9800 | 94 |
| musique | **READ** | recall@5 | 0.5242 +/- 0.0099 | 0.5289 +/- 0.0069 | 0.9950 | 0.9207 | 2092 |
| musique | **READ** | full_coverage@5 | 0.2317 +/- 0.0126 | 0.2322 +/- 0.0079 | 0.9950 | 0.9207 | 2092 |
| webqsp | **NOT_READ** | hit@1 | 0.6299 +/- 0.0050 | 0.5928 +/- 0.0096 | 0.9215 | 0.7649 | 2120 |
| webqsp | **NOT_READ** | recall@5 | 0.6194 +/- 0.0088 | 0.6073 +/- 0.0049 | 0.9215 | 0.7649 | 2120 |

WebQSP is on `train_holdout` and its corpus reference level is 0.5255; the comparator to a published
coverage number is the `any_gold_at_pool` column of the pool, never an average. MetaQA and WebQSP stay
NOT_READ: the published KB systems assign topic entities while this pipeline seeds the graph by
inference-safe retrieval, so no number here is presented as beating or approaching a published system.

## 9. What the stage cost

- compile of the three added carves: 16724483 rows in 2.89 hours
- the measured joint epoch: 20282.9 s at 6 threads, peak RSS 9.4 GB
- the three fits: 84.67 fit-hours at 8 threads (peak RSS 10.38 GB); 131.03 of the 150 fit-hour ceiling spent in total
- the one eval pass: 15.69 hours at 4 threads, one dataset at a time (peak RSS 10.97 GB)
- placement: the laptop, one fit lane at the declared threads

Per dataset. The eval pass scored the three checkpoints together, so its wall clock per query covers compiling
the pool once, three forward passes and the fixed scorers; it is not one model's latency. The latency columns
are single-query timings, p50 / p95 in ms at the eval threads, on the sample the last column names: compile is
the per-query feature compile, pack the single-query batch, forward one checkpoint's forward pass (seeds 0, 1, 2).

| dataset | queries | pass hours | pass ms per query | compile | pack | forward p50 | forward p95 | peak RSS GB | MRR audit | latency sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | 39138 | 12.71 | 1160 | 764 / 963 | 17 / 23 | 207, 202, 203 | 313, 306, 300 | 8.12 | ok | shard 0 of 4, first 500 of its queries |
| 2wiki | 12576 | 0.29 | 79 | 36 / 56 | 7 / 23 | 46, 45, 44 | 67, 68, 65 | 8.22 | ok | the first 500 queries |
| squad | 11873 | 0.18 | 53 | 17 / 40 | 2 / 4 | 40, 40, 39 | 63, 60, 60 | 7.17 | ok | the first 500 queries |
| hotpotqa | 7405 | 0.16 | 69 | 71 / 105 | 6 / 21 | 43, 42, 42 | 64, 62, 62 | 7.81 | ok | the first 500 queries |
| musique | 2417 | 1.83 | 2713 | 604 / 875 | 37 / 72 | 245, 240, 239 | 437, 422, 421 | 10.97 | ok | the first 500 queries |
| webqsp | 1503 | 0.53 | 1250 | 675 / 934 | 50 / 93 | 156, 154, 155 | 357, 359, 350 | 8.98 | ok | the first 500 queries |

## 10. Incidents

- 2026-09-23T01:37Z to 02:03Z -- the stage-2 compile was queued, not launched: another session of the same user held a SPLADE job of 4.8 to 6.3 GB on the same 16.85 GB laptop and free memory was between 0.5 and 2.7 GB. The replication's logged mistake was launching into that contention, so the compile waited. Nothing scientific waited on this: no cache, weight or number existed yet.
- 2026-09-23T02:03Z -- the wait was replaced by a driver that compiles the three added carves one process per dataset (hotpotqa, then webqsp, then musique: smallest graph first) behind a 2.5 GB free-memory pre-flight, beside the other session's job rather than after it. The trio compile peaked at 2.73 GB on metaqa (329,282 nodes) and the three added graphs are smaller (90,447 / 1,549 / 19,938 nodes), and the row streams go to disk through NpyAppender, so the footprint of each added compile is bounded well below that. Placement only: no science moves.
- 2026-09-24T20:45Z to 20:50Z -- free disk was observed oscillating at the 8 GB floor declared in compute.abort_criteria during seed 1 of the stage-2 fit: 7.97 GB at 20:45Z, 8.58 GB at 20:50Z, 8.58-8.58 GB on re-measurement. The floor was crossed at the first reading only, transiently. Nothing halted, and nothing should have: SixDiskGuard is constructed only in stage_compile_six, which finished 2026-09-23T05:04Z, and the fit arms only fit_hours_guard_six; stage_eval_six, stage_read_six, stage_doc_six and stage_file_six never construct it, so no remaining stage-2 stage can be halted by this floor. The pipeline is not the consumer -- outputs/universal_v2/six totals 8.5 MB, 8.4 MB of it checkpoints; the movement is the machine's pagefile and other sessions. Recorded so the report can state the floor was touched and why no stage stopped. No science, no placement.
- 2026-09-25, between 20:48Z on the 24th and 09:47Z on the 25th -- a session restart killed both background shells (exit 4): the stage-2 driver and the epoch watcher. Seed 1 died during epoch 2; no stage-2 process was alive when this was checked. Seed 0 was already complete and its record and weights were untouched on disk. Seed 1 was RESUMED, not restarted, from the .ckpt m3b_train.fit_operator writes after every epoch: it restores weights, optimiser, both RNG states, the per-dataset draw cursors, best_state, the record, the patience counter and elapsed_s, and it refuses a checkpoint whose arm, seed or config do not match, so the resumed run sees the same batches the uninterrupted run would have and the recorded seconds exclude the idle gap. This is continuation of the declared fit; the_freeze.no_warm_start bars loading the TRIO checkpoint as an initialization, which nothing here does. The replacement driver runs seeds (1, 2) only and refuses to start if seed 0's artifacts are missing -- drive_stage2b.py looped (0, 1, 2) unconditionally and would have overwritten a completed fit. Its per-seed summary line also read a filename missing the __six segment, so it never printed; that is fixed. Placement and recovery only: no science moves, no config is touched, and the compute guard is unchanged because the killed partial wrote no fit record.

## 11. What this stage does not do

stage 2 is a measurement, not a selection. There is no threshold, no pass or fail and no new arm to choose. A per-dataset regression against the M3B GAT is REPORTED as it falls, never repaired inside this stage; what to do about it is a new dated declaration filed before its fit.

nothing in this block authorises an MLP fit, an MLP number or a new MLP reading; u_mlp_v2_mix stays REPLICATION_FAIL as filed
