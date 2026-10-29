# R2EW: R2e (the MLP half) under WQ on Kaggle's T4

Declared on 29 October 2026, before any R2EW fit. The code is `outputs/mp_unified/zr2ew.py` (the arm, the fits, the
take, the compares and the verdict) and `outputs/mp_unified/zr2ew_session.py` (the Kaggle session). They import WQK's
code (`zwqk.py`, `zwq.py`, `zwqk_session.py`, `zkg_session.py`), R2's (`zr2br.py`), zlink's (`zlink.py`) and R2GW's
(`zr2gw.py`) unchanged, and patch them in process. No committed file is edited. Nothing here is a result.

## 1. Why this, now, and what the links hold

- R2's next form for NOT_ADOPTED with no GAIN is R2e (`zr2br.py` NEXT; docs/R2_BRIDGE_EM.md section 5): "bridge scores
  only for bridges linked to the row by zlink's edges (B1d: the edge is the bridge), as fixed per-row link flags for
  the MLP and inside zg6's messages for the GNN; one declared file". R2GW's section 7 names it as the form after R2g
  if R2g is NOT_ADOPTED. R2GW is running on both T4 kernels (85256e5).
- R2 section 2 lists three flags against the MLP rule, and the second is "graph-free": R2 reads no edge. R2e is the
  answer to that flag. The rule (docs/SCREENS.md, rounds 22 and 23) allows "the graph's fixed structure" among a row's
  inputs and forbids another row's score. A link flag is the first kind and never the second.
- **This file is the MLP half only.** The GNN half (bridge scores inside zg6's messages, for linked bridges only) is a
  separate later declared file. This verdict closes the MLP half only (section 7).
- Under WQ, webqsp trains, metaqa is zero-shot, and one wiki dataset is held out per split.

**What the links arrays hold, and that they are enough.** The flag needs one fact per (bridge, row): does zlink have an
edge between the bridge's node and the row's node? The arrays that hold it exist on Kaggle already:

- Path: `outputs/wqk/p1c2/links/<dataset>/<carve>/part_<i>of<n>/` with `q_ne.npy`, `e_u.npy`, `e_v.npy`, `e_fam.npy` and
  `record.json`. They are zlink's built edges (`zlink.py build`), read by `zlink.load_links`.
- Columns: `q_ne` (int64) is the number of edges of each question. `e_u` and `e_v` (int16) are pool-local row indices,
  `0 <= index < the question's pool size`, in the same pool order as the batch rows and as R2's bridges
  (`bridges_np` offsets). `e_fam` (int8) is the family, 0..2. Each edge is stored in both directions, once per family,
  with no self-loops. `record.json` holds the step-1 part's sha256 and each array's sha256, and `load_links` verifies
  them.
- Source: the look's pool edges (structural, NER and kNN families on the passage graphs; the KB's relation edges on
  metaqa and webqsp). They are the pool graph each question already has. They are built before any row is scored, from
  no model and no label.
- Shipping: webqsp/fit's links are WQK's own W7 build (`outputs/wqk/ship.json`, in mpr-wqk-*). The other ten carves'
  links come from U2d's base build, in KG's datasets, and the overlay links them at WQK's root. `zwq.kinds_of` lists
  links for every carve, and WQK's overlay refuses a carve whose links are not whole (zwqk.py overlay). WQK's GNN fits
  (zspk, zg6) read the same arrays from the same root.
- zrckp's own carve (`KTChainCarveZRC`) does not load them. R2EW's carve loads them with zlink's own `load_links`, as the
  GNN's carve does, so it adds the same host memory per carve for them (about 5 bytes an edge) and the same sha256
  check. Not timed here.

So no new data is needed and none is invented.

## 2. What changes

One thing changes: which bridge slots a row may use.

R2's slot mask M(v, k) is "bridge k is present and is not the row itself". R2e's is M(v, k) AND link(v, k). Everything
downstream reads the new mask and nothing else changes:

- the prior pi_k, renormalised over the row's own unmasked slots, is renormalised over its linked slots;
- the head's inputs [a, m, f] (the prior-weighted log-sum-exp and the max of the unmasked bridge logits, and whether the
  row has one) are taken over the linked slots; a row with no linked bridge has a = m = f = 0, as a row with no bridge
  has in R2;
- R2's EM term L_br uses the same mask. For a gold row it sums over its linked slots, and for each slot the softmax runs
  over the question's rows linked to that slot. A gold row with no linked slot has no term. The term is trained as R2t
  trained it (LAM 1), only on linked slots.

The bridge logits S (the weights r2_U, r2_V, r2_W1, r2_W2, r2_o), the head, the five bridges and zrckp's own score are
as in R2. There is no new parameter.

## 3. The arm

**zrckpbe (R2e, MLP half), deciding.** For row v of question q, with s_v zrckp's score and h_v R2's head on the row:

    score_v = s_v + h_v(a, m, f over the slots with link(v, k) = 1)

The flag, exactly:

    link(v, k) = 1  if the question's links hold an edge (u, v) with u the pool-local row of bridge k, in any family
               = 0  otherwise (including the bridge row itself, which has no self-loop, and absent bridges)

- The bridges are R2's: the question's top five rows by the stored first-stage RRF column (c_rrf), descending, ties by
  pool position (`zr2br.bridges_np`).
- The flag is fixed per carve: a function of the links arrays and the stored c_rrf column only. No learned value, no
  model score and no label enters it. The carve builds it in `batch()`, before the model is called, from the batch's
  edges (`zlink.link_batch`). The MLP stays MP-free: no message is passed and no score is read.
- The self-test shows this (section 9): the flag functions take no score argument; the flags equal a by-hand reference
  computed from the links arrays alone; two models with different weights see the same flags.

Parameters: 59,537 in R2's part, the same as zrckpb (the flags add none). zrckpbe is zrckp plus those 59,537.

The start: R2's last layer is 0, so h_v = 0 and zrckpbe starts as zrckp, bit for bit, exactly as zrckpb does. With every
flag set to 1, zrckpbe is zrckpb bit for bit (self-tested), so the flags only mask.

**zrckpb, beside, taken from R2GW.** R2GW fits zrckpb under WQ on both splits, seed 0, on the same T4 image. R2EW trains
no zrckpb fit. It uses R2GW's two (`scr-r2gw-zrckpb`, `scr-r2gw-zrckpb-hp`) as the second base. They are the same arm,
seed, split, roots and device as zrckpbe would have for a fit of its own, so a new one would repeat them.

**Naming.** zrckpbe is called "zrckp + R2's bridge head on linked bridges" until the user rules whether reading the five
bridges' vectors keeps it inside the MLP track (R2, section 2, flags 1 and 3). The edges answer flag 2 only. No claim
sets zrckpbe against the GNN as the MLP.

## 4. The fits and the device

| fit | arm | split | trains on | zero-shot |
|---|---|---|---|---|
| scr-r2ew-zrckpbe | zrckpbe | WQ-musique | webqsp, squad, hotpotqa, 2wiki | metaqa, musique |
| scr-r2ew-zrckpbe-hp | zrckpbe | WQ-hotpotqa | webqsp, squad, musique, 2wiki | metaqa, hotpotqa |

- Two deciding fits, no beside fit (see section 3). Seed 0. zp1c3p's train under WQ, as zwq trains zrckp, with the arm
  swapped in; WQK's roots and mixed KT input; R2's EM term on with LAM 1. A fit with no training batch carrying R2's term
  stops unstamped.
- Every read: p@swa on the six s1eval carves, `--rows-cap 65536` (WQK's), on the device the fit was trained on. The read
  also keeps, per carve, the link statistics: rows, rows with at least one linked bridge, flags, questions, questions
  with at least one linked row. They are in `read.json`.
- Device: KG's image on a Tesla T4 (WQK's). The two R2EW fits, R2GW's two zrckpb fits and WQK's seed-0 zrckp fit of each
  split must share one device record.
- Cost if a beside fit were ever wanted: two more zrckpb fits, about 0.4-0.55 h on a second kernel. Not queued.

## 5. Compares, null and rule

Compares: lean_screen's compare of p@swa on s1eval, on the laptop, stamped with R2EW's record and the device.

- `scr-r2ew-zrckpbe-vs-zrckp[-hp]`: zrckpbe against WQK's seed-0 zrckp of its split (the first base; it decides), then
  against R2GW's zrckpb of the same split (beside).

Null: WQK's zrckp seed null (`outputs/wqk/floors-zrckp.json`). Per read, seeds 1 and 2 against seed 0; the floor is
max(0.0075, 2 x their root mean square). The floors file must equal its compares' floors, recomputed.

Rule, on zrckpbe's re-call against WQK's seed-0 zrckp under that null, over its twelve reads (two splits x six datasets),
as R2GW's:

- ADOPT_FOR_FULL_RUN: no read is LOSS and at least one read is GAIN.
- NOT_ADOPTED: otherwise.
- INCOMPLETE, when any of these holds:
  - WQK's base verdict is not BASE_READY;
  - a fit or a compare is missing, or is not R2EW's (for R2GW's zrckpb, not R2GW's);
  - R2EW's fits, R2GW's zrckpb fits and WQK's seed-0 zrckp fits are not on one device;
  - a fit or a read names a gate that is not a filed PASS gate of an R2EW session (for R2GW's zrckpb, of an R2GW session);
  - a fit's R2 constants or EM term are not as declared (zr2br's constants, LAM 1, at least one question with the term);
  - zrckpbe's flag is not as declared in section 3, or its read holds no link statistics on the six carves;
  - WQK's floors are not their compares';
  - a re-call does not hold the twelve reads.

## 6. Beside, deciding nothing

- zrckpbe against R2GW's zrckpb under the same floors, the null's seed-0 R@5 replaced by R2GW's zrckpb read: what the
  flags themselves do.
- The link statistics per read: the share of rows with a linked bridge, flags per row, and the share of questions with
  at least one linked row. They say how often R2e's head is silent.
- R@5 per read for rrf, zrckp, zrckpb, zrckpbe and WQK's seed-0 zspk.
- R2's term per fit: the share of training questions with the term, and L_br at the first and the last epoch.
- Seconds and memory per fit, beside WQK's seed-0 zrckp and R2GW's zrckpb of the same split.

## 7. Outcomes, fixed now, and risks

| verdict | next form |
|---|---|
| ADOPT_FOR_FULL_RUN | zrckpbe fitted again on PKO's pools (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, outputs/pko; PKO builds the links on them too), then its full run under WQ beside zrckp's, each in its own declared file |
| NOT_ADOPTED | R2e's GNN half: bridge scores only for linked bridges inside zg6's messages (R2's text for R2e, GNN side), under WQ on the T4, in its own declared file. The MLP half closes with this verdict, and a verdict closes only its form |
| INCOMPLETE | what is missing is supplied (WQK's base set, R2GW's zrckpb fits, a fit, a compare, a session's gate or a re-call); nothing else changes |

The GNN arm is not in this file. It comes later in its own file, whatever this verdict is, because R2's text for R2e
names both halves and this file declares one.

Risks:

- The flags can be rare. R2's bridges are the first stage's top five by RRF, and a gold row has to be linked to one of
  them in the pool graph. D1 (docs/DIAG_BRIDGE.md) found golds sit next to a found gold within two hops, but not that
  the first stage's top five are linked to them. Where few rows are linked, the head is silent and zrckpbe stays near
  zrckp. The share of rows with a linked bridge is reported per read (section 6).
- A gold row that is not in the pool, or is not linked to a bridge, cannot be flagged and has no EM term.
- The links are per pool. PKO's pools need their own links before a refit (PKO builds them; section 7, ADOPT).
- Loading the links adds their host memory and the sha256 check at each carve load, as it does for the GNN's carve with
  the same arrays. Not timed here; the run's seconds and peak RAM are in the verdict's beside table.
- Pools above 32,767 rows do not fit zlink's int16 rows. zlink's loader refuses them, as it does for the GNN.
- The naming ruling is pending (section 3).
- R2GW's zrckpb fits are an input. R2EW's verdict is INCOMPLETE until they are taken and share the device.

## 8. How it runs

The stub refuses unless the tree is clean, HEAD is main, origin/main holds HEAD, WQK's verdict is BASE_READY, and HEAD
tracks every module the stamps hash (including zlink and zr2gw), this document, R2's, and WQK's records, verdict and
floors.

Datasets: KG's nine private datasets, WQK's groups, and the code bundle (a git bundle of main). No new dataset.

The session (`zr2ew_session.py run`) is R2GW's, with R2EW's names: it finds and checks the datasets and links them, runs
WQK's overlay and gate (each in its own process; if either does not exit 0, no fit runs), then the fits, one queue per
T4. No fit starts after 8 hours.

One kernel, one fit per T4 (either kernel, once R2GW's session on it has ended):

    python outputs/mp_unified/zr2ew_session.py stub --out DIR --gpu0 scr-r2ew-zrckpbe-hp --gpu1 scr-r2ew-zrckpbe --rows-cap 65536

Time: about 290 s of setup (WQK's). R2t's zrckpb fits took 1,411-1,485 s on KG's T4 and WQK's fits train in 0.70-0.95 of
KG's seconds, so one fit takes about 1,000-1,450 s, plus the links' load. With the two fits side by side, the session
takes about 0.4-0.55 h. Not measured for R2EW.

On the laptop, after R2GW's two zrckpb fits are taken (`zr2gw.py take`) and R2EW's session output is downloaded into
its own new empty folder:

    python outputs/mp_unified/zr2ew.py take --from DIR
    python outputs/mp_unified/zr2ew.py compare-all
    python outputs/mp_unified/zr2ew.py verdict

Never committed: the per-fit folders under outputs/screen/fits, outputs/r2ew/sessions, and the session's logs and
session.json.

## 9. Self-tests

    PYTHONIOENCODING=utf-8 python outputs/mp_unified/zr2ew.py --selftest
    python outputs/mp_unified/zr2ew_session.py --selftest

Both run on toys in temporary folders and write nothing under the real folders. `zr2ew.py --selftest` covers:

- the names: the two fits, the compare, R2GW's zrckpb fits as the second base, the modules the stamps hash, R2's NEXT
  text for R2e and R2's constants;
- the flags by hand from the links arrays alone (an edge in one direction, two families, an edge between two bridges, an
  absent slot, a question with no edge, a pool of one, an edge outside its pool refused), and the proof that they are
  inputs available before scoring: the flag functions take no score argument, the carve builds them without a model, and
  two models with different weights see the same flags;
- the carve on a toy built by zlink's own `build` and read by its own `load_links`: R2's carve with the flags beside the
  bridges, the batch's flags equal to the by-hand reference for any question order, a links record that does not match
  its part refused;
- the arm: 59,537 numbers in R2's part, the same state keys as zrckpb; at the start zrckpbe equals zrckp bit for bit, in
  eval and in training; with every flag 1 it equals zrckpb; with every flag 0 the head's inputs are 0; the head's
  inputs by hand over the linked slots, and the score is the base's plus the head;
- R2's EM term on the linked slots: the kept mask is M and the flags, a gold row with no linked slot has no term, a batch
  kept twice is refused, and a fit with no term is refused;
- the link statistics of a read: counts equal the flags', reset per carve, the patch removed on an error too;
- zwq.on_wq: zr2br trains on zwq's modules and the arm swap is restored;
- the guards: where, the read cap, WQK's verdict and gate, the device, R2's constants, and the kept fit;
- the stamps of both fits, and the take, compare-all and verdict on a toy of WQK's base set and R2GW's two zrckpb fits:
  ADOPT_FOR_FULL_RUN, NOT_ADOPTED with a LOSS, a GAIN within a raised floor, and every INCOMPLETE case of section 5.

`zr2ew_session.py --selftest` covers the session's run on a toy (the steps first, then each queue, every fit with
`--rows-cap 65536`), a failing overlay or gate, the refusals, the stub's refusals and its files, and the tracked paths.

## 10. Results

**Verdict: zrckpbe is NOT_ADOPTED** (0 GAIN, 1 LOSS), called by `zr2ew.py verdict` and filed in
`outputs/r2ew/screen-verdict.json` and `.md`, with the pairs, the re-calls and `outputs/screen/scr-r2ew-*-vs-zrckp[-hp]`.
The MLP half of R2e closes with it.

Session: mpr-kg-g6b2 version 7 (gate 7196c841e9c8, overlay 187.1 s, gate 83.8 s, 2,082.0 s in all) on 2x Tesla T4, image
kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3, the device R2GW's zrckpb fits and WQK's zrckp fits ran on. Both fits
exited 0 (scr-r2ew-zrckpbe 1,472.5 s, scr-r2ew-zrckpbe-hp 1,605.3 s).

Deciding, zrckpbe against WQK's seed-0 zrckp under WQK's zrckp floors:

- LOSS: WQ-hotpotqa musique (0.5743 -> 0.5639, -0.0105, floor 0.0075).
- No GAIN. The two metaqa zero-shot reads rise (0.4393 -> 0.4488, +0.0095, and 0.3927 -> 0.4403, +0.0476) and are GAIN on
  the pair's own intervals; under the zrckp floors (0.0621 and 0.0489) they are WITHIN. The other ten reads are WITHIN.
- Beside, deciding nothing: zrckpbe against R2GW's zrckpb, 0 GAIN and 0 LOSS. The link flags give back most of what the
  ungated head read at seed 0 on webqsp (WQ-musique webqsp 0.6577 -> 0.6506, WQ-hotpotqa webqsp 0.6488 -> 0.6425) and
  keep most of its metaqa rise (0.4578 -> 0.4488; 0.4459 -> 0.4403).
- R2's EM term reached 0.839 and 0.792 of the training questions on the two fits (zrckpb's: 0.991 and 0.992), so the
  flags removed the term from about a sixth to a fifth of the questions.
- Seconds on the T4: train 1,111 and 1,267 (zrckpb 991 and 985, WQK's zrckp 831 and 837), read 341 and 322 (300 and
  299, 297 and 283); peak GPU 1.87 and 3.99 GB train, 4.43 GB read, as the others; peak RAM 4.51 and 4.61 GB.
- The link statistics per read are in the verdict file (rows linked, flags per row, questions linked).

Next (section 7, fixed before any number): R2e's GNN half under WQ on the T4, its own declared file (bridge scores only for
linked bridges inside zg6's messages). It is being written, with a confirmation of the ungated zrckpb at the null seeds
(R2WS) before it. Nothing here adopts R2.
