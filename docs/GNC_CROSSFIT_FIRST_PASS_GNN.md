# GNC: the GNR second pass with the first pass cross-fitted, so that every carve's anchors are out of sample

Declared 7 November 2026, before any GNC number and after the paper freeze (00:00 IST). Stage modules:
`outputs/mp_unified/gnccols.py` (the halves, the half fits' first passes, the lists and the columns) and
`outputs/mp_unified/zgnc.py` (the second-pass fits, compares and calls). They are GNR's two modules under GNC's names
with the first pass replaced (sections 2.1 to 2.4); they import `gnrcols.py`, `zgnr.py`, `zgnx.py`, `ltmcols.py`,
`ltwdiag.py`, `zltm.py`, `zml1s.py`, `ziso.py`, `zisb.py`, `zdep.py`, `readswa.py`, `lean_gpu.py` and `s0stop.py`, and edit
none of them.

## 1. The question

- **GNR closed with in-sample anchors.** GNR (docs/GNR_REANCHORED_NEXT_GNN.md; FILED c33bb44, 4 Nov) gave the GNN of
  record LTM's seven NEXT columns scored against the base GNN's own top five rows. Its screen is NO_GAIN under both rules
  (all ten in-domain reads WITHIN). GNR's section 8 named the risk and section 7 the next form: the base was trained on the
  training carves, so its top five there hold more golds than they do on the s1eval carves the fit is read on (the base's
  p@swa R@5 on WQ-musique's four training carves is 0.82 to 0.95, against 0.66 to 0.93 on the s1eval carves of the same
  four datasets). The second pass learned from anchors better than the ones it reads.
- **GNC removes that one difference.** Every training question's anchors come from a fit that did not train on it. The
  s1eval carves' anchors are GNR's, unchanged (the base never saw those questions). Nothing else in the arm changes.
- **The user ruled on it.** D6, 4 Nov 23:54 IST: "once we are done with the paper freezing we can do" the cross-fitted
  first pass. The record cutoff was 6 Nov 18:00 IST and the freeze 00:00 IST 7 Nov; this file is that declaration.
- **So the question is:** with the training carves' NEXT columns anchored on out-of-sample top fives, does the GNN of record
  gain in-domain R@5 above its floor, above all on musique and its 3- and 4-hop questions?

## 2. What runs

### 2.1 The halves (gnccols.py `halves`, CPU)

For each split and each of its four training carves (`ltmcols.train_carves`: the fit carves of webqsp, squad, hotpotqa and
2wiki for WQ-musique; of webqsp, squad, musique (s1fit) and 2wiki for WQ-hotpotqa):

- the questions' components are LTM's (`Q_comp` of LTM's lists of the carve: questions joined by a shared gold node, a
  question without a gold alone), the components that LTM's folds are dealt from;
- the components are dealt to two halves, largest first (ties: the lower first question), each to the half with fewer
  questions so far (ties: half 0): LTM's rule (`ltmcols.comp_folds`) with two folds;
- `outputs/gnc/halves/<split>.json` records, per carve, the ids' sha256, each half's questions and components and the
  half of every question (as a sha256 of the array and the array's file `outputs/gnc/halves/<split>__<ds>__<carve>.npz`).

A question of half j is trained on by half fit `h<j>` and read by half fit `h<1-j>`.

### 2.2 The half fits (zgnc.py `half-fit`, host GPU)

- **The model.** ISB's zg6r-m16 (docs/ISB_ZG6RM_16_EPOCHS.md): zg6r from its own drawn encoder (ISO's `own_load`, no zspk
  tensor copied), config `2e-3:1e-4:0.1:16:10` (16 epochs, SWA from epoch 10), seed 0; the model, trainer, inputs, PKO's
  roots, the card, the read cap and the device are ML1's zg6r's. **Why not the base.** The base starts from ML1's zspk
  encoder, which trained on every training question; a half fit started from it would carry its held-out half in its
  start. The own start has seen nothing but its half.
- **The questions.** The half's questions of each training carve, none of the other half. `lean_gpu.train_cmd` is run with
  the current `lean_gpu.CacheCarve` subclassed for the call so that the carve's `rows` and `batch` see the half's
  questions only (a question index maps to its original index before the carve's own batch runs, so the cache, the links
  and the chain arrays stay aligned), put back afterwards, also on a failure. A half fit's `train.json` records, per carve,
  the questions it trained on and the sha256 of their ids; the held-out read refuses a fit whose record is not this
  declaration's or whose train holds a question of the half it is asked to read.
- **The fits.** `scr-gnc-h0-s0`, `scr-gnc-h1-s0` (WQ-musique) and `scr-gnc-h0-s0-hp`, `scr-gnc-h1-s0-hp` (WQ-hotpotqa). No
  s1eval read of a half fit is made.
- **Time and card.** About half the data of an ISB fit (ML1's zg6r fits took 1.2 h on WQ-musique and 1.8 h on WQ-hotpotqa
  for 8 epochs): WQ-musique's at a 0.42 share of the card, WQ-hotpotqa's at 0.83, the shares of GNR's first passes.

### 2.3 The held-out first pass (zgnc.py `heldout`, host GPU)

GNR's first pass (gnrcols.py `first`) for each half fit on the four training carves' other half: zml1s's read path for the
fit, p@swa alone (readswa), rows cap 32,768, the same capture of each question's five highest rows (pool positions, in
`lean_gpu.top_hit`'s order) with the golds among them checked against top_hit's for every question. The carve is read
through the same subclass as in 2.2, over the held-out half. A question that appears in both halves' reads, or in neither,
stops the pass.

### 2.4 The assembled first pass (gnccols.py `first`, CPU)

`outputs/gnc/s0/first/<split>.json` in GNR's schema (`anchors_of` reads it unchanged):

- **training carves.** Each question's five rows from the half fit that did not train on it, assembled into the carve's
  read file in the cache's question order. `in_sample` is False. The record names both half fits and their models' sha256.
- **s1eval carves.** GNR's read files, byte-copied (sha256 checked against GNR's record). The record names the base fit
  and its models' sha256 as GNR's does.
- **The check.** Every training question is covered once, by the fit that did not train on it; the five rows of every
  question agree with top_hit's golds; the s1eval files equal GNR's. The record carries, per training carve, the
  out-of-sample R@5 beside GNR's in-sample R@5 and per half fit its held-out R@5. These R@5 values decide nothing.

### 2.5 The lists, the columns, the arm (gnccols.py `lists`, `cols`; zgnc.py `fit`)

GNR's section 2.2 to 2.4 unchanged under GNC's names and root (`outputs/gnc/s0`): LTM's T lists with the assembled anchors,
LTM's questions, components and folds checked against LTM's lists of the carve; LTM's fold fits of the split (read only)
score the columns; GNX's arm (the base's model and start, `gnx_w` on seven NEXT columns, 1,920 parameters, zero-started),
trainer, card, epochs and read. The fits: `scr-gnc-nx-s0` (WQ-musique, 0.42 of the card) and `scr-gnc-nx-s0-hp`
(WQ-hotpotqa, 0.83). **Beside, deciding nothing:** the s1eval carves' lists and columns are rebuilt under GNC's names from
the same anchors and the same scorer as GNR's; the report prints, per s1eval carve, whether GNC's column arrays equal GNR's
and the largest difference when not.

## 3. The reads

DEP's section 3, unchanged: ML1's twelve reads (two splits by six datasets, p@swa on the s1eval carves) and musique's 3-hop
and 4-hop slices on both splits.

## 4. The rule (fixed now)

GNR's section 4, unchanged: LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4) through `zltg.decide_screen` and
`zltg.decide_full`.

- **Deciding (ten):** the eight reads of the datasets each split trains on, and the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices. **Beside (six):** the four zero-shot reads and the WQ-musique fits' musique slices.
- **The screen (seed 0):** floor max(0.0075, 2 x the base's spread over its three seeds). LOSS at or below minus the floor,
  GAIN at or above it. **NOT_ADOPTED:** any LOSS. **ADOPT_FOR_FULL_RUN:** no LOSS and at least one GAIN. **NO_GAIN:** every
  read WITHIN. The base is ML1's zg6r (ISO's KEEP), as in GNR.

**INCOMPLETE** holds on GNR's conditions under GNC's names and when:

- a half fit's record does not name this declaration, or its train holds a question of the half it read;
- the first pass of the fit's split and seed is missing, does not cover every training question once, was not built from
  the half fits (and the base for the s1eval carves) its record names, or its check failed;
- a fit's NEXT block does not name this declaration's first pass.

A verdict closes only the form it tested.

## 5. The seed-0 stop

DEP's seed-0 stop through `s0stop.py` with GNC's entry (as GNR's): `gnc-s0stop` after `scr-gnc-nx-s0`, on the four
in-domain reads of WQ-musique; STOP on a LOSS among them, GO otherwise. `scr-gnc-nx-s0-hp` runs inside
`zgnc.py s0stop-run`.

## 6. Reported beside, deciding nothing

- DEP's sixteen-read rule on the same numbers.
- GNR's filed seed-0 deltas (in-sample anchors) and GNX's (rrf's anchors), read by read beside GNC's: the one question the
  file asks is whether out-of-sample anchors moved a read that in-sample anchors did not.
- Per training carve: the out-of-sample R@5 of the assembled anchors, GNR's in-sample R@5, and the s1eval R@5 of the same
  dataset; the census of the lists (GNR's), and the share of the golds outside the anchors that sit next to one.
- Each fit's parameters, seconds and GPU peak, and the floor beside each delta.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | GNC's input goes to the stack through an STK amendment (a form beside GNX's slot, its columns per seed). Seeds 1 and 2 need their own half fits and first passes, built before any stack fit reads them. No GNC number is a paper number: the paper froze on 7 Nov. |
| NO_GAIN or NOT_ADOPTED | The anchors' form closes for the GNN: GNX (rrf's anchors), GNR (in-sample anchors) and GNC (out-of-sample anchors) all read the same NEXT columns. Nothing follows from this file; a next form is its own file. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

## 8. Risks, stated before any number

- **A different recipe for the anchors.** The half fits are zg6r-m16 on half the data; the s1eval anchors are the base's (ML1's
  zg6r on all of it). The half fits' held-out anchors may hold fewer golds than the base's s1eval ones: the second pass may
  then learn from anchors poorer than the ones it reads, the opposite of GNR's mismatch. Section 2.4 prints both.
- **Shared inputs fit on every question.** PKO's pools, links and pair scores, U2m's types and U2k's scorer are shared by every
  fit and are not refit per half; LTM's scorer folds are cross-fitted by LTM's own five folds, which are not the halves.
  Questions that share a node outside their golds sit in different halves when their golds are different.
- **Seed 0 only.** The screen is one seed; the half fits are seed 0.
- **Components.** The halves use LTM's components (a shared gold node); two questions on the same entity but different gold
  nodes can fall in different halves.
- **Memory.** A half fit holds half of its split's training questions on the card (ML1's fits peaked at 7.85 GB on
  WQ-musique and 18.4 GB on WQ-hotpotqa); a held-out read holds one carve's half.

## 9. How it runs

Feeder items, LF, inserted before the head of the queue's long CPU blocks, GPU fits first:

- `gnc-selftest-cols` and `gnc-selftest` (CPU);
- `gnc-halves-<split>` (two, CPU) after LTM's lists of the training carves;
- `gnc-hfit-<j>-<split>` (four, GPU 0.42 and 0.83) after the halves;
- `gnc-held-<j>-<split>` (four, GPU, reads) after the half fits, then `gnc-first-<split>` (two, CPU);
- `gnc-lists-<split>-<dataset>-<carve>` and `gnc-cols-...` (twenty each, CPU), `gnc-report`, `gnc-check-cols`;
- `gnc-nx-s0` (gpu 0.42), `gnc-s0stop`, `gnc-nx-s0-hp` (gpu 0.83), `gnc-compare`, `gnc-calls`, `gnc-gate`.

Writes: `outputs/gnc` and the fits `scr-gnc-*` only. LTM's lists, fold fits and columns, GNR's first passes and ML1's and
ISB's fit folders are read only.
