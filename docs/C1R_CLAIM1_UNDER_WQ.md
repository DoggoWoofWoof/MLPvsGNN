# C1R: Claim 1 measured again under WQ on PKO's pools, with P3c's intervals

Declared 2 November 2026, before any number. Stage module: `outputs/mp_unified/zc1r.py`. It imports
`scripts/p3c_claim1.py` and does not edit it. ML1 section 6 names this file: "Claim 1 re-measured with P3c's intervals against
the strongest GNN".

## 1. What is read

**The fits.** The seed-0 fits of each model on the two WQ splits, on PKO's roots and the host's GPU:

| role | WQ-musique | WQ-hotpotqa |
| --- | --- | --- |
| MLP (zrckp, the MLP of record) | `outputs/screen/fits/scr-ml1-zrckp-s0` | `scr-ml1-zrckp-s0-hp` |
| GNN (zg6r, the GNN of record after ML1) | `scr-ml1-zg6r-s0` | `scr-ml1-zg6r-s0-hp` |

`--gnn zg6rm` reads ISO's zg6r-m fits (`scr-iso-zg6rm-s0`, `-hp`) in the GNN's place. Both records are written. The one that
speaks for Claim 1 is the one of the GNN of record when ISO's verdict is filed: zg6r if ISO says KEEP, zg6r-m if it says
REPLACE.

**The reads.** Each fit's `reads/<dataset>__s1eval.npz` on the six datasets, as in P3c section 2: per question, the
candidate's (p@swa) top-5 gold count and top-1 hit, and RRF's.

**The checks come first.** The stage refuses, and writes nothing, unless all of these hold:

- Every fit's stamp names its arm, seed 0 and split under its stage's key: `ml1` for ML1's fits, `iso` for ISO's (with its
  own start and no copies).
- Every read reproduces its filed compare's row. That means its question count, the candidate's means (R@5, FC@5, hit@1, to
  four decimals) and RRF's means.
  - For zrckp, the filed compare is ML1's `scr-ml1-zrckpt-vs-zrckp-s0{,-hp}`, where zrckp is the base.
  - For zg6r, it is `scr-ml1-zg6r-vs-zspk-s0{,-hp}`, where zg6r is new.
  - For zg6r-m, it is ISO's `scr-iso-zg6rm-vs-zg6r-s0{,-hp}`.
- On each split and dataset, the MLP's and the GNN's reads hold the same questions, in the same order, with the same RRF
  columns.
- A dataset's two reads paired by the zero-shot clause hold the same questions.

## 2. The rule (P3c's, with the regime's map; fixed now)

**What is P3c's, unchanged.** The bootstrap (2,000 resamples, drawn as lean_screen's, seeded by the dataset's index), the
share (MLP - RRF) / (GNN - RRF) with -infinity where the GNN's gain is 0 or below, the bar (0.90), the margin (0.002), the
in-domain call (NO_GNN_GAIN, HOLDS, NOT_SHOWN, MISSES) and the zero-shot call (BELOW, HOLDS).

**The map.** WQ has two splits, not J5 and five leave-outs:

- **In-domain clauses: eight, every in-domain read.**
  - WQ-musique: webqsp, squad, hotpotqa, 2wiki.
  - WQ-hotpotqa: webqsp, squad, musique, 2wiki.
  - No two reads of a dataset are pooled, because they come from different models.
- **Zero-shot clauses: four.**
  - musique, read zero-shot under WQ-musique; its allowed gap comes from its in-domain read under WQ-hotpotqa.
  - hotpotqa, read zero-shot under WQ-hotpotqa; its allowed gap comes from its read under WQ-musique.
  - metaqa under each split, allowed gap 0. It never trains, as P3c's webqsp.
- **The call.** CLAIM1_HOLDS when all eight in-domain clauses and all four zero-shot clauses HOLD. Otherwise CLAIM1_NOT_ALL,
  with every clause that does not hold named, with its numbers.

**Reported beside, deciding nothing:**

- the FC@5 and hit@1 shares with their intervals on every read;
- R@5 shares on the zero-shot reads;
- the R@5 share's point value at seeds 1 and 2 on every read, from the filed compares' means.

## 3. What it means

- These are development reads on s1eval, as P3c's were. The paper's numbers come from a confirmation run on questions never
  used for selection.
- The MLP is MP-free and the GNN is its own model. The claim is about how much of the GNN's gain over RRF the MLP keeps.
- Whatever the call, it changes no model, pool or input.

## 4. Running it

```
python outputs/mp_unified/zc1r.py --selftest
python outputs/mp_unified/zc1r.py run --gnn zg6r --limit 50     (development: steps and checks only, no number)
python outputs/mp_unified/zc1r.py run --gnn zg6r                (outputs/c1r/claim1_zg6r.json and .md)
python outputs/mp_unified/zc1r.py run --gnn zg6rm               (after ISO's fits and compares)
```

It runs on the host, where the reads are, on one CPU, in minutes.

The self-test passed before the declaring commit. It builds a made-up root of fits, reads and filed compares and checks:

- the map (12 reads, 8 in-domain and 4 zero-shot clauses);
- metaqa's allowed gap of 0;
- a missing seed-1 or seed-2 compare is reported, not fatal;
- a read that does not reproduce its compare is refused;
- a wrong stamp is refused.

## 5. Results, zg6r (filed 2 November 2026)

`c1r-zg6r` ran on the host. All of section 1's checks passed: the stamps, every read against its filed compare, and the same
questions between the paired reads. Records: `outputs/c1r/claim1_zg6r.json` and `.md`. zg6r-m's record follows ISO's
compares. ISO's verdict decides which record speaks for Claim 1 (section 1).

**CLAIM1_NOT_ALL.** All four zero-shot clauses HOLD. Four of the eight in-domain clauses HOLD: 2wiki and webqsp on both
splits. The four that do not:

| split | dataset | GNN gain over RRF | share [lo, hi], seed 0 | clause | share at seeds 1, 2 (beside) |
| --- | --- | ---: | --- | --- | --- |
| WQ-hotpotqa | musique | +0.1177 | 0.886 [0.832, 0.940] | MISSES | 0.858, 0.840 |
| WQ-musique | hotpotqa | +0.1747 | 0.903 [0.885, 0.920] | NOT_SHOWN | 0.889, 0.925 |
| WQ-musique | squad | +0.0088 | 0.971 [0.761, 1.239] | NOT_SHOWN | 0.868, 0.670 |
| WQ-hotpotqa | squad | +0.0085 | 0.990 [0.735, 1.309] | NOT_SHOWN | 0.965, 1.206 |

- **musique in-domain** misses at all three seeds. This is the MLP's real gap to the GNN, and the next MLP form has to work
  there.
- **hotpotqa in-domain** straddles the bar across the seeds.
- **squad.** The GNN's gain over RRF is under 0.01, so the share's interval is too wide to show anything with these questions.
- **The zero-shot reads.** All four hold. The MLP is above the GNN on three of them:
  - metaqa's shares are 1.106 and 1.124;
  - hotpotqa's is 1.091.

  On musique the MLP is 0.0035 below the GNN (0.5051 against 0.5086; share 0.901), and its excess is below the allowed gap.
  (Corrected on 2 November 2026: the first filing said the MLP was at or above the GNN on all four. The table's numbers are
  unchanged.)
- The call changes no model, pool or input (section 3).

## 6. Amendment A1 (3 November 2026, before any number it reads): the MLP after LTM

LTM's full run adopted zrckpn (docs/LTM_NEXT_HOP_MLP.md section 12, 7874e4d), and LTM's section 8 names this amendment. LTC's
zrckpc is added at the same time, so its record needs no second amendment.

**What changes.** `--mlp` picks the MLP's family. zrckp stays the default, and its record keeps its name.

| `--mlp` | seed-0 fits | filed compare (zrckpn and zrckpc are new in it, against DOB's CPU zrckp) | stamp key | record |
| --- | --- | --- | --- | --- |
| zrckp | `scr-ml1-zrckp-s0{,-hp}` | `scr-ml1-zrckpt-vs-zrckp-s0{,-hp}` (zrckp is the base) | `ml1` | `claim1_<gnn>` |
| zrckpn | `scr-ltm-zrckpn-s0{,-hp}` | `scr-ltm-zrckpn-vs-zrckp-s0{,-hp}` | `ltm` | `claim1_<gnn>_zrckpn` |
| zrckpc | `scr-ltc-zrckpc-s0{,-hp}` | `scr-ltc-zrckpc-vs-zrckp-s0{,-hp}` | `ltc` | `claim1_<gnn>_zrckpc` |

- Section 1's checks hold as written. Two are added:
  - an LTM fit's stamp must hold its NEXT block (the scorer's sha256);
  - an LTC fit's stamp must also hold its CHAIN block.
- The shares at seeds 1 and 2, reported beside, come from the same family's compares at those seeds.
- zrckpn and zrckpc were trained and read on DOB's CPU device, zg6r on the host's GPU. The rule does not need one device. The
  checks hold the two reads to the same questions, in the same order, with the same RRF columns.
- zrckpn reads NEXT and zg6r does not. The record compares the two models as they stand. A GNN that reads NEXT is read
  through its own family once a stage adopts one.
- **Which record speaks for Claim 1:** the one pairing the MLP of record with the GNN of record at the freeze. That is
  zrckpn now, or zrckpc if LTC's head-to-head against zrckpn adopts it.
- Section 2's rule and bar are unchanged.

```
python outputs/mp_unified/zc1r.py run --gnn zg6r --mlp zrckpn     (outputs/c1r/claim1_zg6r_zrckpn.json and .md)
python outputs/mp_unified/zc1r.py run --gnn zg6r --mlp zrckpc     (after LTC's seed-0 compares)
```

The self-test passed before this commit. It runs section 4's checks for each MLP family on the made-up root, and confirms
that each family uses its own stamp key and record name. It also refuses an LTM stamp without NEXT and an LTC stamp without
CHAIN.
