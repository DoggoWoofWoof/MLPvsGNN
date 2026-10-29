# R2EG: R2e (the GNN half) under WQ on Kaggle's T4

Declared on 29 October 2026, before any R2EG fit. The code is `outputs/mp_unified/zr2eg.py` (the arm, the fits, the
take, the compares and the verdict) and `outputs/mp_unified/zr2eg_session.py` (the Kaggle session). They import WQK's
code (`zwqk.py`, `zwq.py`, `zwqk_session.py`, `zkg_session.py`), G6's (`zg6.py`), R2's (`zr2br.py`), zlink's
(`zlink.py`), R2GW's (`zr2gw.py`) and R2EW's (`zr2ew.py`) unchanged, and patch them in process. No committed file is
edited. Nothing here is a result.

## 1. Why this, now

- R2EW's section 7 fixed it: NOT_ADOPTED for zrckpbe (0 GAIN, 1 LOSS: WQ-hotpotqa musique, -0.0105) closes the MLP half
  of R2e, and "R2e's GNN half under WQ on the T4, its own declared file: bridge scores only for linked bridges inside
  zg6's messages" is the next form. R2's text for R2e (`zr2br.py` NEXT, docs/R2_BRIDGE_EM.md section 5) names both
  halves; R2EW declared the first and this file declares the second.
- R2t put R2's bridge head on zg6's final score (zg6m) under KG: NOT_ADOPTED with no GAIN and 2 LOSS, both webqsp
  zero-shot (docs/R2T_BRIDGE_EM_ON_T4.md section 9: L-musique -0.0216, L-hotpotqa -0.0605). The head sits after all of
  the GNN's messages and reads no neighbour. R2e's GNN half is the other place R2 names: the bridge score inside the
  messages, for the bridges that the row is linked to.
- The GNN, unlike the MLP, may pass messages (docs/SCREENS.md, rounds 22 and 23; the MLP-only rule does not apply to it).
  What R2e adds to it is not a message from another row's learned state. It is the bridge's score, and only where the
  pool graph has an edge between the bridge's node and the row's node, the same fixed link flag R2EW built.
- Under WQ, webqsp trains, metaqa is zero-shot, and one wiki dataset is held out per split. Both T4 kernels are free.
- **Naming.** zg6e is called "zg6 + R2's linked-bridge scores in its hidden update". The user's open ruling on zrckpb
  (whether reading the five bridges' vectors keeps an arm inside the MLP track, R2 section 2) is about the MLP track and
  does not apply to a GNN arm.

## 2. What changes

One term is added to zg6's hidden update, in each of its three layers. Everything else in zg6 is as G6 filed it: the
encoder, the boundary and context weights, the family dropout over the pool edges, the three layers' messages,
attention and self terms, the readout and the card settings.

zg6's layer, for row v with hidden state h_v:

    pre_v = W_self h_v + W_mean mean_f(msg) + W_agg att_v + W_x x_v + b
    h_v  <- h_v + gelu(layer_norm(pre_v))

zg6e's layer, with the bridge message m(v) of section 3:

    pre_v = W_self h_v + W_mean mean_f(msg) + W_agg att_v + W_x x_v + b + m(v) * e_l
    h_v  <- h_v + gelu(layer_norm(pre_v))

- e_l is one learned direction of 64 numbers per layer (3 layers), starting at 0. A scalar m(v) added to every channel
  alike would be removed by layer_norm (it subtracts the mean over the 64 channels), so the term has to be a direction,
  not a constant.
- At the start e_l = 0, so m(v) * e_l = 0 and zg6e is zg6 bit for bit (self-tested, in eval and in training, with the
  family dropout's draws unchanged). With every link flag 0, m(v) = 0 and zg6e is zg6 for any weights.
- Only the pre-activation of the three layers changes. The readout reads the final h as zg6 does.

## 3. The arm

**zg6e (R2e, GNN half), deciding.** For row v of question q, with R2's bridges (the question's top five rows by the
stored first-stage RRF column c_rrf, descending, ties by pool position: `zr2br.bridges_np`):

    s_vk  = ((hq + hk W1' + (hq * hk) W2' + o) . hv) / sqrt(32)       R2's bridge logit of row v and bridge k
    m(v)  = sum over k of  M(v, k) * link(v, k) * s_vk

- hq, hv and hk are R2's: hq = e_q U (the question's vector, 1536 to 32), hv = p V (the row's 257 inputs to 32), hk = hv
  of the bridge's row. They are fixed inputs of the batch (the SEMB block) through R2's learned maps r2_U, r2_V, r2_W1,
  r2_W2 and r2_o.
- M(v, k) is R2's mask: bridge k is present and is not the row itself.
- link(v, k) is R2EW's flag, exactly: 1 if the question's zlink edges hold an edge (u, v), in any family, with u the
  pool-local row of bridge k; else 0. The bridge row itself has no self-loop and absent bridges are 0.
- The flag is fixed per carve: a function of the links arrays and the stored c_rrf column only. No learned value, no
  model score and no label enters it. The carve builds it in `batch()` from all of the question's edges (the pool edges
  before the family dropout), by R2EW's own `link_flags`. The family dropout drops message edges in training and does not
  change the flags.
- m(v) is a sum over the row's linked bridges, once per bridge, however many edge families join them. A row with no linked
  bridge has m(v) = 0, and sends and receives exactly what zg6's rows do.
- The message is the bridge's score on the row, not the bridge's hidden state. No other row's learned state enters m(v).

R2's head (r2_w1, r2_b1, r2_w2, r2_b2) is deleted from zg6e, as zg6 deletes the MLP's output layer: zg6e has no head on
its final score. R2's EM term is kept: it trains the logits s_vk on the linked slots (M and the flags), as R2EW trains
them, LAM 1, so the logits carry R2's own signal into the messages and not only whatever the listwise loss gives them.

**Parameters, exactly, over zg6's own:**

| part | numbers |
|---|---|
| r2_U (1536 x 32) | 49,152 |
| r2_V (257 x 32) | 8,224 |
| r2_W1 (32 x 32) | 1,024 |
| r2_W2 (32 x 32) | 1,024 |
| r2_o (32) | 32 |
| R2's bridge logits | **59,456** |
| g6e_dir (3 layers x 64) | **192** |
| **new in zg6e** | **59,648** |

R2's head (81 numbers: 16 x 3 + 16 + 16 + 1) is not carried: 59,456 + 81 = 59,537, R2's whole part in zrckpb and zg6m.
The self-test checks these counts against the model's state dict.

R2's weights come from R2's own generator (seed + R2's offset), so the global generator's draws, zg6's own weights (the
card's seed + 3601) and the family dropout's draws are as in zg6.

**zg6, the base (WQK's).** WQK fits zg6 under WQ on both splits, seed 0, on the same T4 image (`scr-wqk-zg6-s0`,
`scr-wqk-zg6-s0-hp`). R2EG trains no zg6 fit. It uses WQK's two as the deciding base: the same arm, seed, split, roots,
KT input and device that zg6e has for a fit of its own, so a new one would repeat them.

**zspk, beside (WQK's).** WQK's seed-0 zspk fits (`scr-wqk-zspk-s0`, `-hp`) are the second base of each compare. G6's own
call was zg6 against zspk, and zspk's is the null that exists. zg6e against zspk is beside and decides nothing.

## 4. The fits and the device

| fit | arm | split | trains on | zero-shot |
|---|---|---|---|---|
| scr-r2eg-zg6 | zg6e | WQ-musique | webqsp, squad, hotpotqa, 2wiki | metaqa, musique |
| scr-r2eg-zg6-hp | zg6e | WQ-hotpotqa | webqsp, squad, musique, 2wiki | metaqa, hotpotqa |

- Two deciding fits. Seed 0. zg6's own `train` under WQ (the arm and the fit names added in process; zg6's card settings,
  basis, gate and KT checks run unchanged), WQK's roots and mixed KT input, as WQK trains zg6; R2's EM term on with
  LAM 1. A fit with no training batch carrying R2's term stops unstamped.
- Every read: p@swa on the six s1eval carves, `--rows-cap 65536` (WQK's), on the device the fit was trained on, by zg6's
  own `read`. The read also keeps, per carve, the link statistics: rows, rows with at least one linked bridge, flags,
  questions, questions with at least one linked row. They are in `read.json`.
- Device: KG's image on a Tesla T4 (WQK's). The two R2EG fits and WQK's seed-0 zg6 and zspk fits of each split must share
  one device record.
- Memory: WQK's zg6 peaked at 9.13 GB of GPU in training on WQ-hotpotqa and 5.24 GB in the read. R2's logits add a few
  N x 32 tensors per batch (computed once, not per layer) and 59,648 weights; the 15 GB of a T4 holds it. Not measured for
  zg6e.

## 5. Compares, null and rule

Compares: lean_screen's compare of p@swa on s1eval, on the laptop, stamped with R2EG's record and the device.

- `scr-r2eg-zg6-vs-zg6[-hp]`: zg6e against WQK's seed-0 zg6 of its split (the first base; it decides), then against WQK's
  seed-0 zspk of the same split (beside).

**Null: the stated fallback.** WQK files seed nulls for zrckp and zspk (`outputs/wqk/floors-zrckp.json`,
`floors-zspk.json`) and none for zg6. R2EG reads zg6e against zg6 under WQK's zspk seed null with its seed-0 R@5 replaced
by zg6's read, and zspk's floors. This is G6's own form, WQK's `g6_beside`, G6RW's call on zg6r and R2t's call on zg6m
(docs/R2T_BRIDGE_EM_ON_T4.md section 4: "zspk's null moved onto zg6"). Per read, the floor is max(0.0075, 2 x the root
mean square of zspk's seeds 1 and 2 against seed 0). The floors file must equal its compares' floors, recomputed.

The limit is R2's section 4 limit: the floor measures zspk's seed spread, not zg6's or zg6e's. zg6 has more parameters
than zspk and may move more between seeds, so a GAIN near its floor is weaker than the same GAIN for zspk. A seed
confirmation of zg6 would answer that, and it is not in this file (section 7).

Rule, on zg6e's re-call against WQK's seed-0 zg6 under that null, over its twelve reads (two splits x six datasets), as
R2GW's:

- ADOPT_FOR_FULL_RUN: no read is LOSS and at least one read is GAIN.
- NOT_ADOPTED: otherwise.
- INCOMPLETE, when any of these holds:
  - WQK's base verdict is not BASE_READY;
  - a fit or a compare is missing, or is not R2EG's;
  - R2EG's fits and WQK's seed-0 zg6 and zspk fits are not on one device;
  - a fit or a read names a gate that is not a filed PASS gate of an R2EG session;
  - a fit's R2 constants, parameters or EM term are not as declared (zr2br's constants, the 59,648 new numbers' names,
    LAM 1, at least one question with the term);
  - zg6e's message is not as declared in section 3, or its read holds no link statistics on the six carves;
  - WQK's floors are not their compares';
  - a re-call does not hold the twelve reads.

## 6. Beside, deciding nothing

- zg6e against WQK's seed-0 zspk under WQK's zspk seed null as filed (G6's call): its GAINs and LOSSes.
- The link statistics per read: the share of rows with a linked bridge, flags per row, and the share of questions with at
  least one linked row. They say how often the message is silent.
- R@5 per read for rrf, zspk, zg6 and zg6e.
- R2's term per fit: the share of training questions with the term, and L_br at the first and the last epoch.
- Seconds and memory per fit, beside WQK's seed-0 zg6 and zspk of the same split.

## 7. Outcomes, fixed now, and risks

| verdict | next form |
|---|---|
| ADOPT_FOR_FULL_RUN | zg6e fitted again on PKO's pools (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, outputs/pko; PKO builds the links on them too), then its full run under WQ beside zg6's, and zg6's own seed null (zg6 at seeds 1 and 2) declared before the full run's calls, each in its own declared file |
| NOT_ADOPTED | R2's GNN line closes (zg6m and zg6e are NOT_ADOPTED; R2WS's confirmation of zrckpb at the null seeds is the MLP line's, in its own file). The next declared forms are those waiting on PKO's pools (U2n's zrckpt on U2s's pick, G6RW's zg6r on PKO's pools) and the typed channels of docs/PROGRAM_2026_10_24.md section 4.3 item 5 (G6 with EM-typed messages over EL, R and kNN), each its own declared file. A verdict closes only its form |
| INCOMPLETE | what is missing is supplied (WQK's base set, a fit, a compare, a session's gate or a re-call); nothing else changes |

Risks:

- The flags can be rare. R2's bridges are the first stage's top five by RRF, and the pool graph has to link them to the
  row. Where few rows are linked, m(v) is zero for most rows and zg6e stays near zg6. R2EW reported the share per read;
  R2EG reports it too (section 6).
- m(v) enters each layer unnormalised before layer_norm and is a sum of up to five logits. R2's logits start at the scale
  of one (r2_U and r2_V are unit normal draws; r2_W2 and r2_o start at 0), so m(v) is a sum of up to five such numbers,
  and e_l starts at 0, so the term is silent at first. A large logit could swamp the other terms in a row's
  pre-activation; layer_norm bounds the layer's output but not the direction it picks. The self-test shows the term is
  added before layer_norm; the fit shows whether it matters.
- R2's logits are trained by two losses once e_l moves: R2's EM term (on the linked slots) and the listwise loss through
  the messages. They can pull against each other. Neither is gated; the beside table reports L_br first and last.
- zg6's 3,356 s train on WQ-hotpotqa is the long pole. zg6e adds R2's logits per batch and a few operations per layer,
  and is expected to cost a small fraction more. The session is two fits on two T4s; see section 8.
- No zg6 seed null is filed (section 5). A NOT_ADOPTED is the safer call under a too-small floor and an ADOPT is the
  weaker, which is why section 7 puts zg6's own seed null before the full run's calls.
- The links are per pool. PKO's pools need their own links before a refit (PKO builds them).
- Pools above 32,767 rows do not fit zlink's int16 rows. zlink's loader refuses them, as it does for the GNN already.
- WQK's seed-0 zg6 and zspk fits are inputs. R2EG's verdict is INCOMPLETE until they are taken and share the device.

## 8. How it runs

The stub refuses unless the tree is clean, HEAD is main, origin/main holds HEAD, WQK's verdict is BASE_READY, and HEAD
tracks every module the stamps hash (including zlink, zg6, zr2gw and zr2ew), this document, R2's and R2EW's, and WQK's
records, verdict and floors on zspk.

Datasets: KG's nine private datasets, WQK's groups, and the code bundle (a git bundle of main). No new dataset.

The session (`zr2eg_session.py run`) is R2EW's, with R2EG's names: it finds and checks the datasets and links them, runs
WQK's overlay and gate (each in its own process; if either does not exit 0, no fit runs), then the fits, one queue per
T4. No fit starts after 8 hours.

One kernel, one fit per T4 (either kernel; both are free):

    python outputs/mp_unified/zr2eg_session.py stub --out DIR --gpu0 scr-r2eg-zg6-hp --gpu1 scr-r2eg-zg6 --rows-cap 65536

Time: about 290 s of setup (WQK's). WQK's zg6 took 1,365 s to train and 706 s to read on WQ-musique, and 3,356 s and 715 s
on WQ-hotpotqa. With zg6e's extra work at 10-25 percent, the hp fit takes about 1.2-1.4 h and the musique fit about
0.6-0.7 h on their own T4s side by side, so the session takes about 1.3-1.5 h with the setup. Not measured for R2EG.

On the laptop, after WQK's seed-0 zg6 and zspk fits are taken (`zwqk.py take`) and R2EG's session output is downloaded
into its own new empty folder:

    python outputs/mp_unified/zr2eg.py take --from DIR
    python outputs/mp_unified/zr2eg.py compare-all
    python outputs/mp_unified/zr2eg.py verdict

Never committed: the per-fit folders under outputs/screen/fits, outputs/r2eg/sessions, and the session's logs and
session.json.

## 9. Self-tests

    PYTHONIOENCODING=utf-8 python outputs/mp_unified/zr2eg.py --selftest
    python outputs/mp_unified/zr2eg_session.py --selftest

Both run on toys in temporary folders and write nothing under the real folders. `zr2eg.py --selftest` covers:

- the names: the two fits, the two compares each with WQK's seed-0 zg6 and zspk, the modules the stamps hash, R2's NEXT
  text for R2e, R2's constants and the parameter counts (59,456, 192, 59,648; R2's whole part 59,537);
- the model on a toy built by zlink's own `build`, read by its own `load_links`: the state dict is zg6's plus R2's five
  bridge tensors and g6e_dir and nothing else (no head), the new numbers are exactly 59,648; at the start zg6e equals
  zg6 bit for bit, in eval and in training, and the global generator's draws are zg6's; with random logit weights and
  e_l = 0 it still equals zg6; with every link flag 0 it equals zg6 for any e_l; the message is the by-hand sum over the
  present, non-self, linked slots (float64); layer 0's pre-activation is zg6's plus m(v) * e_0, caught at layer_norm; the
  stack with m = 0 equals zg6's stack bit for bit, with the family dropout's draws too;
- the flags: equal to R2EW's by-hand loop over the edge files for any question order, never changed by the weights or the
  dropout (three forwards, two models, the same flags);
- the gradients: at the start the loss reaches e_l and not R2's logits; with e_l nonzero it reaches both; R2's EM term on
  the linked slots reaches R2's logits from the start, its kept mask is M and the flags, a batch kept twice is refused,
  evaluation keeps nothing, and a batch with no flags or no bridges is refused;
- the link statistics of a read, reset per carve, the patch removed on an error too;
- zwq.on_wq: zg6's tables get zg6e for the block (its arm and R2EG's fit names), zg6's own `train` checks them and refuses
  what is not R2EG's, and they are restored after;
- the guards: where, the read cap, WQK's verdict and gate, the device, R2's constants, and the kept fit;
- the stamps of both fits, and the take, compare-all and verdict on a toy of WQK's base set: ADOPT_FOR_FULL_RUN, NOT_ADOPTED
  with a LOSS, a GAIN within a raised floor, the zg6 column as the null's seed 0 (and zspk's as is for the beside call),
  and every INCOMPLETE case of section 5.

`zr2eg_session.py --selftest` covers the session's run on a toy (the steps first, then each queue, every fit with
`--rows-cap 65536`), a failing overlay or gate, the refusals, the stub's refusals and its files, and the tracked paths.

## 10. Results

None yet. The fits are not queued. Nothing here is a result.
