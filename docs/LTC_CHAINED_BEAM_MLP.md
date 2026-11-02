# LTC: zrckp with LTM's next-hop scorer chained two and three links out, as fourteen more input columns, on PKO's pools, on the host's CPU

Declared 2 November 2026, before any LTC column or MLP fit, and before LTM's screen is read. Stage modules:

- `outputs/mp_unified/ltccols.py` (the columns), committed with this file;
- `outputs/mp_unified/zltc.py` (the MLP's fits, compares and calls), committed after this file and before any LTC fit
  (section 10).

They import `ltmcols.py`, `ltwdiag.py`, `chndiag.py`, `zltm.py`, `zwq.py`, `zml1s.py`, `zdob.py`, `zp1c3p.py`, `zdep.py`
and `zltg.py`, and edit none of them. LTM's lists, fold fits and NEXT columns are read only. LTC writes only under
`outputs/ltc`.

## 1. The question

- **Where LTM stops.** LTM (docs/LTM_NEXT_HOP_MLP.md) gives zrckp LTW's learned next-hop score one link out from rrf's
  top five rows. Its own risks name the limit ("One hop"): a 3- or 4-hop question's third passage may be two links from a
  found one.
- **What is filed:**
  - M1 (docs/M1_MUSIQUE_HOPS.md): musique's first sub-question's paragraph is found (0.70 to 0.84), and the chain breaks
    at the bridge.
  - CHN's D1: musique's missed golds sit next to found ones, with a lift of about 14 over non-golds.
  - The knee (DEPTH_FLAT): the pools reach more with depth (0.922 to 0.976) while R@5 stays flat.
  - LTW, SIGNAL: a learned score of (question, found passage, their residual) picks the next hop where no fixed score
    does (CHN, NO_SIGNAL).
  - PFS's A2 PRF (docs/PKO_FEATURE_SCREEN.md): fixed embedding-slot columns found nothing on all five datasets. So the
    chain is scored by LTW's learned scorer, not by a fixed vector score.
- **The user's direction (2 November):** "declare the chained beam for the mlp now, i think hop based softmax makes
  sense".
- **So the question is:** as inputs of zrckp beside LTM's NEXT, do the scorer's two- and three-link chained columns raise
  in-domain R@5, above all on musique's 3- and 4-hop questions?
- **Still not message passing.** Every input is fixed before the MLP scores any row: rrf's top five, PKO's links, the
  frozen vectors and LTM's fixed fold fits. The beam is chosen by the fixed scorer, never by the MLP. The user ruled
  (2 November) that reading the frozen vectors of a fixed beam "isnt message passing it is more akin to beam search".

## 2. The columns (`ltccols.py`)

- **"Hop-based softmax", as read here.** At each hop, a softmax over that hop's own candidates under the fixed scorer. A
  path's weight is the product of its hops' softmaxes. A row's mass at a depth is the sum of the weights of the beam's
  paths that end at it.
- **Anchors and hop one are LTM's.** The anchors are each question's top five rows by rrf. Anchor k's candidates are its
  neighbours by any of PKO's link families, outside the top five. p(.|a_k) is the softmax of the split's fold scorer over
  them.
- **The beam.** At each hop, each anchor keeps the 3 rows of most mass (ties go to the lower pool row).
- **An expansion.** A beam row x is scored as an anchor is:
  - the same scorer runs over x's own neighbours outside the top five, with x in the anchor's place;
  - LTW's features are taken against x: x's cosine to the question, and the question's residual against x;
  - p(.|x) is the softmax over them.

  This is LTW's single-anchor residual, as in the scorer's training.
- **Two links.** P2_k(r) = the sum over the beam's x of p(x|a_k) p(r|x).
- **Three links.** The 3 rows of most P2_k mass are expanded the same way: P3_k(s) = the sum over them of
  P2_k(r) p(s|r).
- **Not renormalised.** Mass the beam leaves out stays out.
- **Fourteen columns** (`chain2_*`, `chain3_*`).
  - For depth d and anchor k, the column is log P_d,k(row) + log |R_d,k|, clipped to +-8. R_d,k is the set of rows
    P_d,k reaches, so this is LTM's log-ratio to uniform.
  - A row off R_d,k reads -8.
  - Then, per depth: the largest of the five, and the number of anchors that reach the row.
- **No new fit.** Every score comes from the fold fit that LTM's NEXT columns use for the question: out of fold on a
  training carve, the drawn fold on any other carve.
- **Checks in every build:**
  - hop one, recomputed in LTM's own loop, gives LTM's NEXT columns back: the values within 0.01 (float16's step at |x|
    in [4, 8) is 2^-8) and the counts exactly;
  - every list passes LTW's cosine identity;
  - the cache parts, the links parts, LTM's lists and LTM's column files match by sha256;
  - LTM's column record names the split's present fit.
- **The chain report.** It decides nothing beyond section 7's choice of next form. Per carve and per depth 1, 2 and 3 it
  gives:
  - the share of rows reached;
  - the share of missed golds (golds outside the top five) reached;
  - the AUC of the depth's max column over the reached rows.

## 3. What runs

- **The columns.** The twenty carves LTM built (ten per split): `ltc-chain-<wqm|wqh>-<dataset>-<carve>`, on LTM's
  present fits and columns.
- **The arm, zrckpc.** zrckp (zp1c3p's ZRCKP on zp1c3's chain carve) with two more blocks:
  - NEXT: LTM's seven columns, as LTM's fits read them;
  - CHAIN: these fourteen.

  The trainer reads each block raw, z-scored on the training rows, and with a mask. That is 15 + 29 = 44 more inputs to
  the first layer than zrckp has. The output layer starts at zero, as zrckp's does.
- **Everything else is DOB's base, as for LTM:**
  - the trainer, the card, PKO's roots and typed links, and the KT input;
  - the read cap (65,536) and the candidate p@swa;
  - the pick: DOB's nine blocks, then NEXT, then CHAIN. A fit whose trainer drops NEXT or CHAIN as dead is refused.
- **The device is DOB's:** the host's CPU at 6 threads, torch 2.8.0+cpu, with the GPU hidden. The base is DOB's CPU zrckp
  at seeds 0, 1 and 2 on both splits.
- **The fits.**
  - Screen: seed 0 on both splits, `scr-ltc-zrckpc-s0` (WQ-musique) and `scr-ltc-zrckpc-s0-hp` (WQ-hotpotqa).
  - Full run: seeds 1 and 2, queued now behind the screen's gate.
  - Each train and read is stamped under the key `ltc`. The stamp names the split's scorer (LTM's fit record sha256),
    every carve's NEXT and CHAIN records (sha256), the pick, the threads and the device.
- **The compares,** through zwq's `run_compare`:
  - each LTC fit against DOB's base fit of its seed and split (the rule);
  - DOB's seeds 1 and 2 against its seed 0 (the null), filed again under LTC's names;
  - each LTC fit against LTM's zrckpn of its seed and split, once that fit is read (beside, and section 7's
    head-to-head).

## 4. The reads

DEP's section 3, unchanged, as in LTM's section 5:

- ML1's twelve reads;
- musique's 3-hop and 4-hop slices on both splits.

## 5. The rule (fixed now)

LTM's section 6, unchanged, with zrckpc in zrckpn's place: LTG's in-domain rule against DOB's CPU base.

- **Deciding (ten):** the eight reads of datasets their split trains on, and the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices.
- **Beside (six):** the four zero-shot reads and WQ-musique's slices.
- **The screen (seed 0).** Each deciding read is called on its own:
  - the floor is max(0.0075, 2 x the base's spread over its three seeds);
  - LOSS: the delta is at or below minus the floor. GAIN: the delta is at or above the floor;
  - NOT_ADOPTED: any LOSS. ADOPT_FOR_FULL_RUN: no LOSS and at least one GAIN. NO_GAIN: every read WITHIN.
- **The full run (seeds 1 and 2).** Each deciding read is called by `zml1.call` on its three seed deltas, with the pair
  floor of both arms' spreads.
  - ADOPT: no LOSS among the ten, and a GAIN on at least one of the eight dataset reads. A slice GAIN alone does not
    adopt, and UNCONFIRMED_GAIN does not count.
  - NOT_ADOPTED otherwise.
- **INCOMPLETE** holds under LTM's conditions (LTM section 6, with LTC's pick), and also when:
  - a fit's CHAIN carves are not its NEXT carves;
  - a chain record changed after the fit read it;
  - a chain record names a scorer other than the split's present LTM fit, or an LTM column record other than the one the
    fit read.

A verdict closes only the form it tested.

## 6. Reported beside, deciding nothing

- The six beside reads, called the same way.
- DEP's sixteen-read rule on the same numbers, as `dep_rule_beside`.
- The chain report per carve (section 2), and LTM's column table.
- zrckpc - zrckpn per read, at each seed where both fits are read.
- Seconds per fit.
- C1R's two reads, marked.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `ltc-gate`, then section 5's full-run rule. **On ADOPT, if LTM's full run is also ADOPT,** a head-to-head decides between them: zrckpc against zrckpn at three seeds, with LTG's full-run call on the ten deciding reads. zrckpc is the MLP candidate when there is no LOSS and a GAIN on at least one of the eight dataset reads; otherwise zrckpn is. **If LTM's full run is not ADOPT,** zrckpc is the candidate. The candidate is measured again for Claim 1 (C1R's intervals). Claim 3's cold timing counts the columns' cost (hop one and both expansions), measured, not assumed. |
| NO_GAIN or NOT_ADOPTED | The chain as declared closes. The next form, in its own file, follows the chain report on WQ-hotpotqa's musique/s1eval carve, where musique is in-domain. **(a)** If depth 2's AUC is more than 0.10 below depth 1's, the scorer does not carry over to an anchor that is not a found gold. The next form trains a hop-2 scorer on its own lists: the anchor is a missed gold one link from a found gold, and the targets are missed golds one link further. **(b)** Otherwise, the next form is the cumulative residual as columns (LTM section 8 (b)): each hop's query takes out the span of every passage on the path. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

- LTM's verdict does not wait on LTC. LTC's columns are built whatever LTM's screen says, since NEXT's columns exist for
  both splits either way.
- **Later (the user, 2 November):** link rules that add nothing (NER's among them) are to be cut at a later stage. LTC
  reads PKO's links as they are, and nothing here prunes them.

## 8. Risks, stated before any number

- **Anchors off the training distribution.** The scorer trained with found golds as anchors. A beam row is usually not a
  gold, and it sits further from the question, so its expansions may score worse. The chain report's AUC per depth shows
  this, and section 7 (a) acts on it.
- **One beam width, one set of depths.** BEAM 3 and depths 2 and 3 are fixed, not tuned.
- **Coverage.** LTM's lists cover 0.045 to 0.047 of musique's rows (0.008 of metaqa's, 0.43 to 0.47 of squad's). The
  chain reaches more rows. If it reaches most of a pool, the columns select less.
- **Seed 0 only in the screen.**
- **The start.** 44 more first-layer inputs, so the fit's random draws are not the base's. The floor is set for a change
  of draw.
- **The columns' cost at serving.** Per question: hop one's five lists, plus up to fifteen expansions at each of depths 2
  and 3. This enters Claim 3 only on adoption, and then as measured.
- **Time.**
  - LTM's column builds took 19 to 104 s per carve. The chain scores about 6 to 7 times as many lists, so it takes
    minutes per carve.
  - zrckpc's fits should take about as long as DOB's CPU fits (1 h 25 min to 2 h 24 min).
  - The user freezes major iterations on 6 to 7 November. The plan: columns on 2 November, the screen's fits by
    3 November, and the full run by 5 November if the screen passes.
- **Memory.**
  - Column items: under 8 GB.
  - Fits: LTM's sizes plus the chain's columns, 36 GB (WQ-musique) and 40 GB (WQ-hotpotqa). The feeder resizes them from
    measured peaks.
- **Disk.** 218 million rows x 14 float16 columns: about 6.1 GB under `outputs/ltc` on the host (86 GB free at this
  declaration).

## 9. How it runs

Feeder items on the host's CPU. None touches the GPU.

1. `ltc-selftest-cols` (`ltccols.py --selftest`, 1 CPU, 3 GB).
2. The twenty column sets, `ltc-chain-<wqm|wqh>-<dataset>-<carve>`, after `ltc-selftest-cols` (2 CPUs, 8 GB). Then
   `ltc-report`.
3. `ltc-selftest` (`zltc.py --selftest`), once zltc.py is committed.
4. The screen's fits, after `pkoh-gate`, `ltc-selftest` and their split's ten column sets (6 CPUs):
   - `ltc-zrckpc-s0`: 36 GB;
   - `ltc-zrckpc-s0-hp`: 40 GB.
5. `ltc-compare` (after the two fits, DOB's six base fits and LTM's two screen fits), then `ltc-calls`, then `ltc-gate`
   (`outputs/ltc/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN).
6. The full run, after `pkoh-gate` and `ltc-gate`: `ltc-zrckpc-s1`, `-s2`, `-s1-hp` and `-s2-hp`, with the screen's
   sizes. Then `ltc-compare-full` and `ltc-calls-full`.
7. The head-to-head's compares and calls, after `ltc-calls-full` and `ltm-calls-full`: `ltc-compare-h2h`, then
   `ltc-calls-h2h`. If LTM's gate stops LTM's full run, the feeder drops them.

```
python outputs/mp_unified/ltccols.py chain --split WQ-hotpotqa --dataset musique --carve s1eval --host
python outputs/mp_unified/ltccols.py report                         (outputs/ltc/report.json and .md)
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zltc.py fit --name scr-ltc-zrckpc-s0
python outputs/mp_unified/zltc.py compare-all
python outputs/mp_unified/zltc.py calls                             (outputs/ltc/calls.json and .md)
```

Only `outputs/ltc/calls.json`, `calls.md`, `report.json` and `report.md` are committed. The columns stay on the host.

`ltccols.py --selftest` passed on the laptop before this commit. It covers:

- a hand-made chain with exact two- and three-link masses: the beam, its ties, a row reached through another, no
  renormalisation, the clip, and the AUC with ties;
- on LTW's toy PKO tree, after LTM's lists, a toy fold fit and LTM's columns, the chain of a training carve and of two
  eval carves:
  - shapes and records;
  - the hop-one identity;
  - an unbatched recomputation of every column by hand (worst difference under 0.02);
  - another fold's fit gives other columns;
- every refusal:
  - a carve outside the split;
  - a split with no fit;
  - an LTM column file whose sha256 is not its record's;
  - hop one that is not LTM's NEXT under matching shas;
  - LTM's columns from another scorer;
  - LTM's lists built with a limit;
- the report.

## 10. zltc.py

Committed after this file, before any LTC fit. Its self-test is listed here when it is committed.
