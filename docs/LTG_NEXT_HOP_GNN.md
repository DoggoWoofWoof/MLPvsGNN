# LTG: the GNN of record with LTW's learned next-hop term inside its message passing, on PKO's pools

Declared 2 November 2026, before any LTG fit. Stage module: `outputs/mp_unified/zltg.py`. It imports `zml1s.py`,
`ziso.py`, `zdep.py`, `zg6r.py` and `lean_gpu.py` and edits none of them.

## 1. The question

- **The cause, as filed.** musique in-domain is the main target (the user, 2 November). Three reads place its gap:
  - the knee, A3 (docs/PKO_DEPTH_KNEE.md section 11): zg6r reads musique flat at every depth while the pool's reach rises;
    the golds a deeper pool adds are never ranked into the top five;
  - CHN (docs/CHN_CHAIN_COMPLEMENT.md section 9, NO_SIGNAL): no fixed score of the frozen vectors singles out the next hop
    among a found passage's neighbours;
  - LTW (docs/LTW_LEARNED_NEXT_HOP.md sections 10 and 12): a **learned** three-way score of (question, found passage,
    their residual) against the row does. On the store's 256 int8 axes (A1, STORE_KEEPS) it puts musique's missed next
    hop first or second for 0.548 of s1eval's targets, against 0.360 for `pair`; 2wiki 0.852, hotpotqa 0.795. Reranking
    rrf's top five with it (H1) lifts R@5 by +0.041, +0.153 and +0.106 on musique, 2wiki and hotpotqa.
- **What the GNN lacks.** zg6r's edge logit is a fixed form: the sender's state against the question's context, a family
  bias, and on typed edges the relation code. Nothing in it asks which direction of the question the sender's passage
  leaves unanswered, which is what LTW's term reads.
- **What LTW named next** (its section 7, SIGNAL): the GNN, built on its own and not on the MLP, gets the three-way term
  on each message, trained end to end with the GNN's own weights; seed 0 on both WQ splits on PKO's pools; the main reads
  musique in-domain (WQ-hotpotqa) and its 3- and 4-hop slices. This file is that screen.
- **So the question is:** does LTW's next-hop term, inside zg6r's three layers, raise in-domain R@5, above all on musique
  and its 3- and 4-hop questions?
- LTW's H1 gains are over rrf's top five, not over the GNN (musique in-domain R@5 0.5912 against rrf's 0.4734). The GNN's
  own attention may already hold part of them; this screen measures what is left.

## 2. What runs

- **The arm: the GNN of record after ISO, with the term in each of its three layers.** Everything else is the base's: the
  trainer and its loss, the card and its epochs, PKO's roots and typed links, the KT input, the 64-wide state, the seed
  offsets, the read cap of 32,768, and the candidate p@swa.
- **The term's inputs, from the store block every batch already holds** (`SEMB`: the question's 1,536-D vector, each row's
  257 decoded store columns; basis_2wiki, npz sha256 `0ca4797b...`, pinned: a fit refuses another basis):
  - qk = V'q, the question on the store's 256 axes; x_u, a row's decoded axes;
  - ĉ_u, the store's own estimate of cos(q, row u) (lean_mlp3.Store's P . Q);
  - the residual r_u = (qk − ĉ_u x_u) / |qk − ĉ_u x_u|, the question with the sender's passage taken out.
  - LTW A1's residual used the exact 1,536-D cosine; LTG uses ĉ, since a batch holds only the store's rows. The self-test
    shows ĉ is lean_mlp3.Store's estimate.
- **The score of edge u → v** (u sends, v receives), rank 32, LTW's `bil` on the store's axes:
  - the sender's query Q_u = [qk V0, x_u V1, r_u V2] (3 x 32), the receiver's key P_v = [x_v U0, x_v U1, x_v U2];
  - in layer k, s_k = P_v . (Q_u + W_k h_u). W_k h_u lets the sender's state after the layers before (what it has gathered)
    refine its query.
- **How it enters each layer.** ℓ = the log-softmax of s_k over the sender's out-edges, plus log out-degree (the log-ratio
  to uniform), clipped at ±8:
  - the edge's attention logit gains g_k[f] ℓ (one gain per family per layer), before zg6r's clamp;
  - the receiver's update gains am_k E_k + aa_k F_k, where am_k and aa_k are its per-family mean of sigmoid(logit) ℓ and
    attention-weighted ℓ over its in-edges.
- **The start is zg6r's, bit for bit.** U, W, g, E and F start at 0; V is drawn from the term's own generator (seed + 4101),
  so zg6r's own draws and the training's random stream are the base fit's.
- **The next-hop loss, and why it is needed.** At the start P = 0, so every s_k is 0 and every ℓ is exactly 0: the listwise
  loss gives every term parameter exactly zero gradient, and the term would stay inert for the whole fit (a saddle; the
  self-test shows U's gradient is 0.0 under the listwise loss alone). The training loss is therefore listwise + λ x the
  next-hop loss, λ = 1:
  - for each gold row u with a gold neighbour v ≠ u, minus the log of the softmax mass that P_v . Q_u puts on u's gold
    neighbours, among all of u's out-edges; the mean over such senders;
  - it reads only V and U (no GNN state), so it trains the scorer and nothing else. Without it the fit is the base fit.
  - Each fit's loss curve (`fit_variant`'s per-epoch loss) includes λ x the next-hop loss; the two parts are stamped
    apart every 500 steps under `aux`.
- **NER-link next hops** (LTW's weakest family: the store's term 0.401 against `pair`'s 0.381, structural 0.681):
  - each family has its own gain per layer, so a weak family can be weighted down without muting a strong one;
  - the sender's softmax runs over all its out-edges, of every family, so an NER edge competes with the structural ones
    for the sender's mass, as LTW's T2 measured;
  - the next-hop loss treats the families alike.
- **3- and 4-hop questions** (H1's gain was all on 2-hop questions: one hop from rrf's top row is not enough):
  - the term runs in every layer, from every sender, so layer k carries it k hops out;
  - W_k h_u refines each sender's query by what it has gathered;
  - the WQ-hotpotqa fits' musique 3-hop and 4-hop slices are deciding reads (section 4).
- **Parameters.** The term adds 68,745: V and U 2 x 3 x 256 x 32; per layer W 96 x 64, g 3, E and F 3 x 64 each. The
  message-passing stack and readout go from 180,218 to 248,963 at hidden 128.
- **How it is set.** For the length of an LTG fit zltg patches, in-process:
  - zg6r's `g6r_stack` (zg6r's forward calls it by name) with the term's stack, op for op zg6r's;
  - ZG6R's `__init__`, which adds the term after zg6r's own, and ZG6R's `forward`, which hands the store block to the stack;
  - lean_gpu's `listwiseD`, which adds the next-hop loss to the training step whose scores the stack made.

  The basis rides as buffers that are not saved. Every patched name is put back afterwards, also on a failure. Any later
  read of these fits must run under the same patch; a plain zg6r refuses the state (the strict load fails on the term).
- **The base, fixed by ISO's filed verdict, as DEP's** (`zdep.iso_call`, `zdep.base_fit`):
  - **KEEP:** zg6r, ML1's fits `scr-ml1-zg6r-s<k>` (with `-hp` on WQ-hotpotqa), from the same zspk seed-0 encoder;
  - **REPLACE:** zg6r-m, ISO's fits `scr-iso-zg6rm-s<k>`, with its own draw through ISO's own-start patches;
  - **anything else:** no LTG fit, compare or call runs.
- **It does not wait for DEP.** LTG is a change to the three-layer GNN of record, measured against it. Its fits go ahead of
  DEP's on the GPU: LTW's SIGNAL makes it the form aimed at musique's diagnosed cause, and the knee read musique as flat in
  depth.
- **The fits.** Screen: seed 0 on both splits, `scr-ltg-nh-s0` (WQ-musique) and `scr-ltg-nh-s0-hp` (WQ-hotpotqa). Full
  run: seeds 1 and 2 on both, queued now behind the screen's gate. Each fit is stamped under the key `ltg` with the term's
  record (rank, axes, basis sha, λ, clip, parameter counts), its base fit, ISO's verdict, the next-hop curve, and zml1s's
  stamp (device, environment, gate, code).
- **The compares.** Each LTG fit against the base's fit of its seed and split, through zwq's `run_compare`, as DEP's.

## 3. The reads

DEP's section 3, unchanged: ML1's twelve reads (two splits by six datasets, p@swa on the s1eval carves) and musique's
3-hop (760 of 2,417) and 4-hop (405) slices on both splits, each slice from the fit's musique read file (`zdep.slice_r5`),
whose R@5 over all questions must equal its compare's within 0.00006.

## 4. The rule (fixed now)

**In-domain, per the user's "indomain for now" (2 November).** LTW section 7 named DEP's screen rule. This file narrows
the deciding reads to the in-domain ones before any LTG number, and computes DEP's sixteen-read verdicts beside (section 5).

- **Deciding (ten):** the eight reads of datasets their split trains on (zwq: WQ-musique trains on webqsp, squad,
  hotpotqa and 2wiki; WQ-hotpotqa on webqsp, squad, musique and 2wiki), and the WQ-hotpotqa fits' musique 3-hop and 4-hop
  slices.
- **Beside (six):** the four zero-shot reads (zwq's `ZERO_SHOT`: musique and metaqa on WQ-musique, hotpotqa and metaqa on
  WQ-hotpotqa) and the WQ-musique fits' musique slices.

**The screen (seed 0)**, DEP's per-read call on each deciding read:

- the floor is max(0.0075, 2 x the base's spread over its three seeds);
- **LOSS:** the delta is at or below minus the floor; **GAIN:** at or above the floor;
- **NOT_ADOPTED:** any LOSS. **ADOPT_FOR_FULL_RUN:** no LOSS and at least one GAIN. **NO_GAIN:** every read WITHIN.

**The full run** (seeds 1 and 2): each deciding read is called by `zml1.call` on its three seed deltas, with the pair
floor of both arms' spreads.

- **ADOPT:** no LOSS among the ten, and a GAIN (on all three seeds' terms: d0 at or above the floor, d1 and d2 above 0) on
  at least one of the eight dataset reads. A slice GAIN alone does not adopt; UNCONFIRMED_GAIN does not count.
- **NOT_ADOPTED** otherwise.

**INCOMPLETE** holds when:

- a fit, compare or read file is missing, or ISO's verdict is not REPLACE or KEEP;
- a fit's train or read stamp does not name the term (68,745 parameters, rank 32, 256 axes, basis sha `0ca4797b...`,
  λ = 1, three layers), its base fit and ISO's verdict;
- its train ran no next-hop step;
- its encoder's start is not its base's, or it ran on another device than its base;
- its trainer's code differs from its base's (stage modules aside);
- the base's seed-0 R@5 in LTG's compare differs from its own stage's, or a read file's R@5 differs from its compare's.

A verdict closes only the form it tested.

## 5. Reported beside, deciding nothing

- The six beside reads, called the same way.
- DEP's rule on the same numbers (`zdep.decide_screen` on all sixteen reads; `zdep.decide_full` when the full run is
  read), as `dep_rule_beside` in `outputs/ltg/calls.json`.
- Each fit's next-hop curve (listwise and next-hop loss, gold senders a step), its parameter counts, seconds and GPU peak.
- The floor, printed beside each delta.

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `ltg-gate`, then section 4's full-run rule. On ADOPT, zg6r with the term is the GNN of record: Claim 1 (C1R's intervals) and Claim 3 (cold timing) are measured again on it, and a DEP or RTR form adopted on the three-layer base is carried onto it in its own file. |
| NO_GAIN or NOT_ADOPTED | The term as declared closes. The next form, in its own file: the cumulative residual, where each sender's query takes out the span of every row the question's context already weighs high, not the sender's passage alone (aimed at the 3- and 4-hop slices). A warm start from LTW's fitted scorer is not a form here: that scorer saw musique's fit carve, which the WQ-musique split must not. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

The MLP's screen (LTM: the same term as input columns from the first stage's anchors, refit on the MLP's own carves) is
declared in its own file and does not wait on LTG.

## 7. Risks, stated before any number

- **Seed 0 only in the screen.** It is a one-fit screen; the full run decides any adoption.
- **The GNN may already hold the gain.** LTW's gains are over rrf's top five; zg6r's attention starts from a stronger
  ranking. A NO_GAIN would say the GNN's own weights already found what the term reads.
- **The gains start at 0.** The main loss must move them within the base's epochs, while the next-hop loss trains the
  scorer from its first step. The card is not changed in this stage.
- **ĉ is an estimate.** The residual uses the store's cosine, not LTW A1's exact one. A1 showed the store keeps the term's
  T2 with the exact cosine; this one difference is not measured on its own.
- **The next-hop loss reads gold senders only.** In the layers, every row sends; what the scorer does from non-gold
  senders is whatever generalises.
- **Zero-shot reads may fall.** The rule does not stop on them; they are reported beside, and so is DEP's rule.
- **Memory.** The base's hp fits peaked at 17.17 GiB against the 17.27 GiB cap of `--frac 0.72`. The term adds per-edge
  scalars per layer and two N x 96 tensors, so the hp fits take `--frac 0.80` (GPU share 0.83) and run alone on the card.
  The musique-split fits keep 0.40 (share 0.42). A fit that runs out of memory stops INCOMPLETE, and the caps change in
  an amendment before it runs again.
- **Time.** The term's edge products run in chunks of 2^18 edges, three layers plus the loss, in the checkpointed
  forward and the backward: about 10 to 30% over the base's 4,300 s (WQ-musique) and 6,500 s (WQ-hotpotqa).
- **Determinism.** The term uses only ops lean_gpu's deterministic mode already runs (index_add_, scatter_reduce amax,
  gathers), so a repeat is the same fit.
- **The self-test's model is a toy.** It shows that the term builds, starts as zg6r, trains and restores. It says nothing
  about R@5.

## 8. How it runs

Feeder items on the host, ahead of DEP's in the items file (GPU items keep the file's order):

1. `ltg-selftest`, on the CPU.
2. The screen's fits, both after `pkoh-gate`, `iso-verdict` and `ltg-selftest`:
   - `ltg-nh-s0`: GPU share 0.42, cap 0.40, 7 GB;
   - `ltg-nh-s0-hp`: GPU share 0.83, cap 0.80, 12 GB.
3. `ltg-compare`, then `ltg-calls`, then `ltg-gate` (`outputs/ltg/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN).
4. The full run, after `pkoh-gate` and `ltg-gate`: `ltg-nh-s1`, `ltg-nh-s2`, `ltg-nh-s1-hp`, `ltg-nh-s2-hp` with the
   screen's sizes; then `ltg-compare-full` and `ltg-calls-full`.

```
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zltg.py fit --name scr-ltg-nh-s0
python outputs/mp_unified/zltg.py compare-all
python outputs/mp_unified/zltg.py calls          (outputs/ltg/calls.json and .md)
```

Only `outputs/ltg/calls.json` and `calls.md` are committed. `ltg-calls-full` rewrites them with both blocks, so the
screen's are committed before the full run ends.

The self-test, `python outputs/mp_unified/zltg.py --selftest`, passed on the laptop before this commit. It covers:

- the names and pairs, disjoint from ML1's, ISO's, DEP's and RTR's; the ten deciding and six beside reads;
- the edge product against the direct one and by gradcheck; the segment log-softmax against each segment's own; the
  next-hop loss by hand, and 0 without a gold pair;
- zg6r's toy carve (106 typed of 164 edges) with a toy basis of 256 axes, zg6r's parameters drawn at random:
  - at the term's start, the eval forward and the training forward (the same family-dropout draw) are zg6r's bit for bit;
  - ĉ is lean_mlp3.Store's estimate (relative 4.5e-7) and the residual's product taken apart is the direct one (9.2e-7);
  - the listwise loss alone gives U a gradient of exactly 0, the next-hop loss a gradient above 0;
  - with the term's parameters drawn, the score moves and every term parameter gets a gradient; checkpointing on and off
    give the same score and gradients;
  - 68,745 term parameters; the basis pin refuses another sha; the state loads strictly under the patch and a plain zg6r
    refuses it;
  - one epoch of lean_gpu's own `fit_variant` with the next-hop loss firing and U moving;
- the patches (names, stamp with the term's record and curve, ISO's own start on REPLACE), and that they are put back,
  also on a failure;
- section 4's rule and DEP's beside on made-up reads (zero-shot losses beside, a slice GAIN alone, UNCONFIRMED_GAIN).

## 9. Amendment A1, 2 November 2026, before any LTG fit trained: ML1's start read under ML1's names

**What happened.** ISO's verdict is KEEP (docs/ISO_ZG6R_OWN_START.md, section 11), so LTG's fits start from ML1's seed-0 zspk
encoder (section 2). `ltg-nh-s0` stopped 2 s after it started on the host (2 November, 23:20), before any train, with "zml1s: scr-ml1-zspk-s0 is not fitted and read by ML1".

**Why.** `zml1s.load_encoder` takes the start only after `zml1s.stamped` finds ML1's read stamp in the zspk fit: the block of
its read.json under `zml1s.KEY`, declared in `zml1s.DECLARED`. LTG's `on_names` sets those two names to LTG's own for the whole
fit. So the loader looked for an `ltg` block in ML1's fit and refused it. The self-test checked that KEEP leaves ML1's loader in
place; it did not run the loader under LTG's names. DEP, RTR, IRA and zg6-z take the same route and would have stopped the same
way.

**The change.**

- `outputs/mp_unified/ml1start.py` (new). `ml1_load(split)` runs ML1's own loader with ML1's declaration, stage and key in
  force for the length of the load only, and LTG's are back after it, also on a refusal. The tensors and the record are the
  loader's, bit for bit. A zspk fit without ML1's read stamp is still refused, and so is one stamped by LTG alone.
- `zltg.py`'s fit context installs `ml1_load` in zml1s on KEEP, where it left ML1's loader. REPLACE is unchanged (ISO's
  `own_load`).
- Each fit's code block also names `ml1start_sha256`. DEP's CODE_SKIP, which LTG uses, skips it when a fit's code is compared
  with its base's, as it skips zml1s's and ziso's: these are stage modules, not the trainer's code.

**What does not change.** Sections 1 to 8: the term, the next-hop loss, the fits, the base, the reads, the rule and what
follows. The check that a fit's start is its base's start is unchanged.

**The queue.**

- `ltg-nh-s0` is renamed `ltg-nh-s0-b`, and `ltg-compare` waits on it. The feeder's one retry of the old item would have
  stopped the same way and dropped `ltg-compare` and everything behind it.
- A new CPU item, `ml1start-check`, runs `python outputs/mp_unified/ml1start.py --check-host` on the host:
  - ml1start's self-test;
  - the self-tests of the five stages (zltg, zdep, zrtr, zira, zisoz), each in its own process;
  - the real zspk starts of both WQ splits, loaded under each stage's names: the loader alone refuses them, and the wrapper
    must give ML1's tensors and record.

  It writes `outputs/ml1start/check.json`. Every GNN fit of LTG, DEP, zg6-z, RTR and IRA waits on it.

**Self-tests.** `python outputs/mp_unified/ml1start.py --selftest` and `python outputs/mp_unified/zltg.py --selftest` passed on
the laptop's CPU before this commit. ml1start's runs toy zspk fits of both splits, in ML1's form, under each of the five stages'
`on_names`:

- the loader alone refuses ML1's stamp, as `ltg-nh-s0` did, and takes a fit stamped by the stage alone;
- the wrapper loads ML1's fit and refuses the stage's, a missing fit and a models.pt of another fit;
- the stage's names are back after each.

zltg's now checks that the KEEP context installs `ml1_load` and that the code block names ml1start.

## 10. Amendment A2, 3 November 2026, before `ltg-nh-s0-hp` and LTG's full run: a fit's read keeps p@swa only

**What happened.** A GNN fit's read takes about as long as its train: ML1's zg6r at seed 0 trained in 4,320 s and read in
4,618 s. lean_gpu's read scores every candidate a fit saved, that is each epoch's state and the SWA state (nine at 8 epochs).
LTG's rule reads p@swa alone, and so do the compares, DEP's slices, C1R's intervals and the knee. Each finds p@swa by its name.

**The change.**

- `outputs/mp_unified/readswa.py` (new). `on_read()` wraps `lean_gpu.load_models` so that it cuts the fit's candidates to
  p@swa before it builds the models. lean_gpu's read then writes read.json and each carve's npz from that list. The wrap is
  undone after, also on a failure, and entered twice it wraps once. A fit without p@swa, or without its state, is refused.
- `zltg.py`'s fit context enters `on_read()` for the whole fit, on REPLACE and on KEEP. The train never
  calls `load_models`, so only the read changes.
- Each fit's code block also names `readswa_sha256`. A fit's code is compared with its base's over the base's keys only, so
  the new key is not compared, as for the stage modules' own.

**Why no number changes.** `lean_gpu.read_carve` runs each candidate's model on the same batch, one forward per candidate, and
no candidate's forward depends on another's. In eval zg6r draws nothing (its generator draws only while training). So p@swa's
rows (the top five, the hits, the first questions' scores) are the rows a full read gives, up to the GPU's own read noise (ML1's
amendment A1). readswa's self-test runs the real `read_carve` on a toy carve over several batches: p@swa's rows and the
references are the same read alone and among nine.

**The records.** read.json and each carve's npz list p@swa alone, so their `candidates` say which read a fit had. models.pt keeps
every state, so a full read can still be made from it.

**Which fits.** Every LTG fit that starts after `readswa-check` passes: `ltg-nh-s0-hp` and the full run. `ltg-nh-s0-b`
started at 23:46 on 2 November with the old code and reads all nine candidates; its read.json lists them.

**The queue.** A new CPU item, `readswa-check`, runs `python outputs/mp_unified/readswa.py --check-host` on the host:

- readswa's self-test;
- the self-tests of the five stages (zltg, zdep, zrtr, zira, zisoz), each in its own process;
- ML1's seed-0 zg6r fits of both WQ splits, loaded as the read loads them, under `on_read()` and lean_screen's zg6r arm. Each
  must give one ZG6R model, p@swa, holding p@swa's state, with lean_gpu's own `load_models` back after. The check also
  confirms that `on_read()` patches the lean_gpu module that zg6r's read calls.

It writes `outputs/readswa/check.json`. Every GNN fit of LTG, DEP, zg6-z, RTR and IRA not yet started waits on it.

**Self-tests.** `python outputs/mp_unified/readswa.py --selftest` and `python outputs/mp_unified/zltg.py --selftest` passed on
the laptop's CPU before this commit. The stage's self-test now checks that its fit context installs
`on_read()` on REPLACE and on KEEP, that the code block names readswa, and that lean_gpu's own `load_models` is back after.

**What does not change.** Sections 1 to 9: the term, the next-hop loss, the fits, the base, the reads, the rule and what follows.
