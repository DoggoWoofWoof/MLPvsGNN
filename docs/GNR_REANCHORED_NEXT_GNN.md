# GNR: the GNN of record with NEXT columns scored against its own top five rows (a first pass), on PKO's pools

Declared 3 November 2026, before any GNR number. Stage modules: `outputs/mp_unified/gnrcols.py` (the first pass, the lists
and the columns) and `outputs/mp_unified/zgnr.py` (the fits, compares and calls). They import `zgnx.py`, `ltmcols.py`,
`ltwdiag.py`, `zltm.py`, `zml1s.py`, `zdep.py`, `ziso.py`, `readswa.py`, `lean_gpu.py` and `s0stop.py`, and edit none of
them.

## 1. The question

- **GNX closed its form.** GNX (docs/GNX_NEXT_COLUMNS_IN_GNN.md) gave the GNN of record LTM's seven NEXT columns: LTW's
  learned next-hop score of each pool row against the first stage's top five rows. Its screen is NO_GAIN (ec09b5a): all ten
  in-domain reads WITHIN. Beside, the zero-shot musique read of WQ-musique rose +0.0147 and its 4-hop slice +0.0173, both
  over their floors.
- **GNX section 7 names this form:** "the NEXT columns against the GNN's own top five rows. A first pass of the GNN ranks the
  pool, LTM's split scorer scores each row against that pass's top five, and a second pass reads those columns (an
  iterative refinement of the anchors)."
- **Why the anchors matter.** A NEXT column scores a row as the next hop of an anchor. The first stage's top five are poor
  anchors where the first stage is poor. The base GNN's top five hold far more of the golds (s1eval R@5, p@swa, seed 0):

  | carve | rrf | base, WQ-musique | base, WQ-hotpotqa |
  | --- | ---: | ---: | ---: |
  | musique | 0.4734 | 0.5086 | 0.5912 |
  | hotpotqa | 0.6846 | 0.8593 | 0.8004 |
  | 2wiki | 0.6079 | 0.9269 | 0.9296 |
  | webqsp | 0.0535 | 0.6632 | 0.6658 |
  | metaqa | 0.0047 | 0.3615 | 0.3519 |
  | squad | 0.9054 | 0.9142 | 0.9139 |

  On webqsp and metaqa, rrf's top five hold almost no gold, so GNX's columns there scored next hops of wrong rows. LTA's
  report (outputs/lta/report.md) shows the split scorer ranks the gold next hop well among an anchor's linked rows (musique
  T2 0.33-0.36, rrf 0.07), so the scorer has signal when the anchor is right.
- **The user asked for this** ("imthinking of iterative refinement ... isonet and isonet++ they do that", and "try
  whatever can fix it ... iterative refinement of the vectors, relationship, pools").
- **The GNN stays its own model.** The anchors come from the base GNN of the split and seed; the scorer is LTW's (LTM's
  fold fits, read only). No MLP and no MLP fit enters a GNR fit.
- **So the question is:** with the NEXT columns re-anchored on the base GNN's own top five, does the GNN of record gain
  in-domain R@5, above all on musique and its 3- and 4-hop questions?

## 2. What runs

### 2.1 The first pass (gnrcols.py `first`, host GPU)

- **The model.** The base of GNX's arm for the split and seed (`zdep.iso_call`, `zdep.base_fit`). ISO's filed verdict is
  KEEP, so the base is ML1's zg6r fit (`scr-ml1-zg6r-s0`, and `scr-ml1-zg6r-s0-hp` on WQ-hotpotqa), candidate p@swa. On
  any other ISO verdict the first pass refuses to run.
- **The carves.** Every carve a GNR fit of the split loads: the split's four training carves and the six s1eval carves
  (`ltmcols.col_carves`). The training carves are read **in sample**: the base was trained on them (section 8).
- **The read.** zml1s's read path for the base (`zml1s.run_arm`, its contexts unchanged), on the fit's device (cuda), p@swa
  alone (readswa), rows cap 32,768. Two in-process changes, put back afterwards:
  - `lean_gpu.read_cmd` reads from and writes to GNR's copy of the fit: `outputs/gnr/s<seed>/first/<base>/` holds a byte
    copy of the base's `models.pt` and `screen.json` (sha256 checked). Nothing is written into ML1's fit folders.
  - `lean_gpu.read_carve` keeps the p@swa scores of each batch and, per question, its five highest rows (pool positions)
    in `lean_gpu.top_hit`'s order (descending score, NaN last, ties by pool position: `chndiag.order_of`). The golds among
    those five must equal top_hit's count for every question, or the pass stops. The five go into the carve's read file
    as `top5` (questions x 5, -1 past a question's rows).
- **The check.** On each s1eval carve, the first pass's R@5 must lie within 0.002 of the base's own filed read (the GPU's
  read noise moves a few near-ties; another model or a misaligned carve moves far more). `outputs/gnr/s<seed>/first/<split>.json`
  records, per carve, the R@5, the share of questions whose golds among the five equal the base's own read file's, and
  the read file's sha256. The lists refuse a first pass whose check failed.
- **Time and card.** The base's p@swa reads of the six s1eval carves took 718 s (WQ-musique, peak 9.36 GB) and 477 s
  (WQ-hotpotqa, peak 18.45 GB) on a shared card; the training carves add more rows. About 30-40 minutes per split:
  WQ-musique's pass at a 0.42 share of the card (cuda_alloc 0.4), WQ-hotpotqa's at 0.83 (cuda_alloc 0.8), with the same
  rows cap as the base's reads.

### 2.2 The lists (gnrcols.py `lists`, CPU)

LTM's T lists (ltmcols.lists) with the first pass's five rows as the anchors, per split, seed, dataset and carve:

- each anchor's candidates are its neighbours by PKO's links (LTW's `fam_bits`) outside the five;
- each candidate's features are LTM's: rrf's rank (the scorer's `log1p_rank`), the rank block, dense_cos and the link
  families;
- the questions' components and folds are LTM's, and must equal the folds in LTM's lists of the carve (the scorer's
  cross-fitting depends on it);
- the A lists are not built: the scorer is not refit.

### 2.3 The columns (gnrcols.py `cols`, CPU)

`ltmcols.cols` on GNR's lists, with LTM's split scorer (its five fold fits, `outputs/ltm/fit_<split>.json`, read only):
seven float16 columns per row (`next_1`..`next_5`, `next_max`, `next_n`), log-ratios to uniform among each anchor's
candidates, clipped to +-8, -8 off them. A training question's columns come from the fold fit without its fold; every
other question's from the fit its index draws. The records carry GNR's stage and declaration, so `zltm.load_next` loads
them only under GNR's names. Layout: `outputs/gnr/s<seed>/cols/<split>/<dataset>/<carve>/`.

### 2.4 The arm (zgnr.py)

GNX's arm unchanged, reading GNR's columns: the base's model and start, `gnx_w` (hidden x 15, zero-started) applied to each
row's [raw, z, 1] and added to l1's output, 1,920 parameters; the trainer, the card, the epochs and the read are GNX's.
For a call, zgnx's names (fits, declaration, stage, stamp key, output root, fit and pair names, code, term record and NEXT
block) are GNR's, and ltmcols' root, stage and declaration are GNR's for the fit's seed (as zgna does for LTA's columns).
The fits: `scr-gnr-nx-s0` (WQ-musique, 0.42 of the card) and `scr-gnr-nx-s0-hp` (WQ-hotpotqa, 0.83).

## 3. The reads

DEP's section 3, unchanged: ML1's twelve reads (two splits by six datasets, p@swa on the s1eval carves) and musique's 3-hop
and 4-hop slices on both splits.

## 4. The rule (fixed now)

LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4), unchanged, through `zltg.decide_screen` and `zltg.decide_full`:

- **Deciding (ten):** the eight reads of the datasets each split trains on, and the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices. **Beside (six):** the four zero-shot reads and the WQ-musique fits' musique slices.
- **The screen (seed 0):** floor max(0.0075, 2 x the base's spread over its three seeds). LOSS at or below minus the floor,
  GAIN at or above it. **NOT_ADOPTED:** any LOSS. **ADOPT_FOR_FULL_RUN:** no LOSS and at least one GAIN. **NO_GAIN:** every
  read WITHIN.

**INCOMPLETE** holds on GNX's conditions (section 4 of its file, under GNR's names) and when:

- a fit's train or read NEXT block does not name this declaration, or its train's code is not this zgnr.py and gnrcols.py;
- the first pass of the fit's split and seed is missing, is not of the base fit `zdep.base_fit` names, its models.pt sha256
  differs from the base's, or its s1eval check failed.

A verdict closes only the form it tested.

## 5. The seed-0 stop

DEP's seed-0 stop through `s0stop.py` with GNR's entry (as GNA's): `gnr-s0stop` after `scr-gnr-nx-s0`, on the four
in-domain reads of WQ-musique; STOP on a LOSS among them, GO otherwise. `scr-gnr-nx-s0-hp` runs inside `zgnr.py s0stop-run`.

## 6. Reported beside, deciding nothing

- DEP's sixteen-read rule on the same numbers.
- GNX's filed seed-0 deltas (rrf's anchors), read by read beside GNR's.
- Per carve: the first pass's R@5 (in sample on the training carves), and the lists' census: rows next to an anchor, the
  golds among the candidates, and the share of the golds outside the anchors that sit next to one, beside LTM's lists'
  (rrf's anchors).
- Each fit's parameters, seconds and GPU peak; the floor beside each delta.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | GNR's input goes to the stack through an STK amendment (a form beside GNX's slot, its columns per seed). Seeds 1 and 2 need their own first passes and columns, built from the base fits of those seeds before any stack fit reads them. No GNR-alone full run before the 6-7 November freeze. |
| NO_GAIN or NOT_ADOPTED | The in-sample anchors close. The next form, in its own file: the first pass cross-fitted (two half fits of the base per split), so that the training carves' anchors are out of sample as the s1eval carves' are. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

## 8. Risks, stated before any number

- **In-sample anchors.** The base was trained on the training carves, so its top five there hold more golds than on the
  s1eval carves. The second pass learns from columns whose anchors are better than the ones it reads at s1eval, and may
  lean on them too much. The report prints the first pass's R@5 on each training carve beside the s1eval ones. Section 7
  names the cross-fitted first pass as the next form.
- **The scorer was fit on rrf's anchors.** LTM's fold fits learned from A cases whose anchor is a gold row in rrf's top
  five. The features hold no anchor rank, but GNN anchors outside rrf's top five can sit further from the question.
- **Seed 0 only.** Each fit's anchors are its own seed's base; the screen is one seed.
- **Time.** WQ-musique's first pass shares the card with GNW's seed-0 fit before LTX's starts (about 40 minutes);
  WQ-hotpotqa's takes 0.83 of the card for about 30 minutes before the first WQ-hotpotqa fit at 0.83. The GNR fits run
  after them: the seed-0 fit at 0.42 of the card after LTX's seed-0 fit, the WQ-hotpotqa fit after the four hp fits of
  GNA, GNP, GNW and LTX. The stack's fits move later by the same time.
- **Memory.** A first-pass read holds one carve on the card at a time (metaqa s1eval has 42.8 million rows); the base's
  reads peaked at 9.36 GB (WQ-musique, under a 0.4 cap) and 18.45 GB (WQ-hotpotqa).

## 9. How it runs

Feeder items, LF:

- `gnr-selftest-cols` and `gnr-selftest` (CPU);
- `gnr-first-wqm` (gpu 0.42, cuda_alloc frac 0.4) before `gnw-mw-s0` in the file, and `gnr-first-wqh` (gpu 0.83,
  cuda_alloc frac 0.8) before `gna-nx-s0-hp`, both after `pkoh-gate`, `iso-verdict`, `readswa-check` and
  `gnr-selftest-cols`;
- `gnr-lists-<split>-<dataset>-<carve>` (twenty, CPU) after the split's first pass and LTM's list of the carve, then
  `gnr-cols-...` (twenty, CPU, two threads) after each list and LTM's columns of the carve, then `gnr-report` and
  `gnr-check-cols`;
- `gnr-nx-s0` (gpu 0.42), `gnr-s0stop`, `gnr-nx-s0-hp` (gpu 0.83, after the four hp fits), `gnr-compare`, `gnr-calls`,
  `gnr-gate`.

Writes: `outputs/gnr` and the fits `scr-gnr-*` only. LTM's lists, fold fits and columns, and ML1's fit folders, are read
only.
