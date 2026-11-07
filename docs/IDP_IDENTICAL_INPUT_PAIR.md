# IDP: the best MLP and the best GNN on exactly the same inputs

Declared 6 November 2026, before any IDP fit, compare or read. Stage code: none new for the fits and the compare; they are
GNX's (`outputs/mp_unified/zgnx.py`, docs/GNX_NEXT_COLUMNS_IN_GNN.md), unchanged. The user asked for the pair of record to read
the same inputs ("the best mlp and the best gnn should have exactly the same inputs", 6 November).

## 1. The question

- **The pair.** The MLP of record, zrckpn (LTM: zrckp plus seven cross-fitted NEXT columns), and a GNN that reads the same
  columns: GNX, zg6r with LTM's seven NEXT columns as one more input of its encoder (docs/GNX_NEXT_COLUMNS_IN_GNN.md). Both read
  the same nine DOB blocks, the same KT input and the same NEXT columns; the GNN also receives the link graph as edges (its
  message passing), which the MLP does not.
- **What exists.** GNX's seed-0 fits on both splits (`scr-gnx-nx-s0`, `scr-gnx-nx-s0-hp`). Their screen is NO_GAIN against zg6r
  (outputs/gnx/calls.md: 0 GAIN, 0 LOSS, 10 WITHIN). Seeds 1 and 2 were never run: their items waited on `gnx-gate`, which could
  pass only on ADOPT_FOR_FULL_RUN.
- **What this stage does.** It runs those four fits so that the pair has three seeds on both splits, and files the reads. It
  decides no adoption: GNX's screen verdict stands, zg6r stays the GNN of record for Claim 3 and for the stack, and which pair
  headlines Claim 1 is the user's call once the tables are filed.

## 2. The fits

- `idp-gnx-s1`, `idp-gnx-s2` (WQ-musique) and `idp-gnx-s1-hp`, `idp-gnx-s2-hp` (WQ-hotpotqa): GNX's own items for seeds 1 and 2,
  with the same commands, fit names (`scr-gnx-nx-s1`, `-s2`, `-s1-hp`, `-s2-hp`), sizes and GPU shares (0.42 and 0.83), under
  new names and without the dependency on `gnx-gate` and `stk-own-gnx`, because the old items cannot start. Their other
  dependencies are GNX's: `pkoh-gate`, `ml1start-check`, `readswa-check`, `gnx-check-cols`.
- `idp-gnx-compare` (`zgnx.py compare-all`) after the four fits, then `idp-gnx-calls` (`zgnx.py calls`). Each GNX fit is read
  against the base's fit of its seed and split (ML1's zg6r) as in GNX section 3; the reads are the six datasets of C1R under both
  splits.
- ISB's `isbf-s2-hp` (running since 20:49 host time) is cancelled to free the card, because the hp fits need 0.83 of it alone.
  ISB's compare and calls items (`isbf-compare-full`, `isbf-calls-full`) are parked on a dependency no job has
  (`hold-isb-s2-hp`), so the feeder keeps them. Whether and when ISB's seed-2 hp fit is requeued is decided after IDP's calls;
  ISB's doc, module and filed records are untouched.

## 3. The reads this stage files

- **Dev tables.** For each of the six datasets and each split: R@5 of zrckpn, GNX and zg6r, the 3-seed means, GNX minus zrckpn
  with the number of fits where GNX is above, and the first stage's R@5 (RRF) beside. In-domain = the splits that train on the
  dataset; zero-shot = the others. Computed from the filed reads (C1R's zrckpn record and GNX's compares).
- **HippoRAG 2's three settings.** GNX read on B1P's six carves with B1M's columns is a separate amendment (A1), written after
  the fits, with its own module that edits none of zb1p, zb1m and zgnx. Until it is filed, those three settings carry zrckpn
  against zg6r (B1M, filed) and zrckp against zg6r (B1P, filed), and the tables say that the pair is not yet on identical
  inputs there.

## 4. Stop and disclosure

- A fit that ends INCOMPLETE or with a read missing is re-run by name, not skipped. No threshold is declared: the stage reports
  differences and counts, it does not call ADOPT or NOT_ADOPTED.
- The seed-0 numbers already filed are not recomputed. The cancelled ISB job lost about 1 h 10 min of a 6.5 h fit.

## 5. Amendment A1 (7 Nov 2026, 20:03 IST, before any GNX read on B1P's carves)

The user, 7 Nov: finish the ablations and the same-input model. Section 3 left one part open: GNX read on HippoRAG 2's three
settings with B1M's columns, so that the pair is on identical inputs there too. This amendment declares it.

- **What runs.** GNX's six fits (`scr-gnx-nx-s{0,1,2}` and `-hp`, filed by GNX and IDP; no training) are forked to
  `outputs/idpa/fits/<v>/<fit>` without their reads (models.pt checked), once per variant (`w` = b1w, `t` = b1t), and each is
  read on B1P's carve of the variant for musique, 2wiki and hotpotqa: B1P's roots, gate and carves, B1M's columns (LTM's own
  scorers and fold fits, built on those carves by B1M; `outputs/b1m/ltm`, read only), the card, p@swa alone, the rows cap and the
  device of the fit's own train. The read is zml1s's `run_arm(read)` under GNX's patches (`zgnx.on_gnx`) with B1P's patches for
  the roots, the gate, the eval carves and the read flags; `zgnx.on_split` is given B1M's column folder. A read that loads no
  columns for one of its three carves, or loads columns of another scorer than the fit's, stops unwritten.
- **Module.** `outputs/mp_unified/zidpa.py` (`fork`, `read --name FIT --variant w|t`, `report`, `--selftest`). It imports
  `zb1m.py`, `zb1p.py`, `zgnx.py`, `zml1s.py`, `zp1c3.py`, `b1b.py` and `b1f.py` and edits none of them, and writes under
  `outputs/idpa/` only. Twelve reads (6 fits x 2 variants), each about the length of B1P's zg6r reads (2 to 4 min).
- **The report.** Per setting, variant and domain (in-domain = the WQ splits that train on the dataset, zero-shot = the others),
  the 3-seed R@5 of GNX, zrckpn (B1M's reads) and zg6r (B1P's reads), each with its 95% bootstrap interval (b1b's: 2,000
  resamples of the questions, seed 0); GNX minus zrckpn and GNX minus zg6r with their intervals and a label by the interval
  (GNX_ABOVE, OTHER_ABOVE, LEVEL; the same rule as B1M's GNN_ABOVE, MLP_ABOVE, LEVEL), with the number of fit pairs where GNX is
  above; the b1w in-domain gap to the best published rows and the interval calls against them, as B1M files them. No threshold
  is declared and no ADOPT or NOT_ADOPTED is called: the stage reports differences, intervals and counts, like section 4's.
- **What it does not change.** zg6r stays the GNN of record; the tables of section 3 and the seed-0 numbers already filed stand.
  Until `outputs/idpa/report.json` is filed, the b1 settings carry zrckpn against zg6r as B1M filed them.
- **Stop.** A read that ends INCOMPLETE, or with a file missing, is re-run by name, not skipped.
