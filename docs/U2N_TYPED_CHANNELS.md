# U2n: typed channels for both models, on Kaggle's T4

Declared 27 October 2026. Code: `outputs/mp_unified/zu2n.py` (the models, the typed links, tau0, the fits, the
compares and the verdict) and `outputs/mp_unified/zu2n_session.py` (the Kaggle session). No U2n fit has run, and the
typed links and tau0 are built after this commit. Nothing in this file reads a U2n number.

## 1. Why this, now

- **The program fixed these forms.** The program of 24 October (docs/PROGRAM_2026_10_24.md, section 4.3, step 5)
  names U2n's two forms on the three families:
  - the GNN: G6 with EM-typed messages;
  - the MLP: zrckp with per-type relation columns.

  Section 4.1 gives the user's logic for the GNN. An edge's type on a passage graph is latent. Its posterior can read
  both ends' hidden states and the question's context, which only message passing provides. On a KB the observed
  relation is the prior over the type, because its text vector lives in the same space as the type vectors. So the
  passage edges and the KB's triples train one relation channel.
- **U2m (a) gave the types a start.** U2m (a) (docs/U2M_POOLS_THREE_FAMILIES.md, section 14) fitted 64 latent types per
  dataset on the graph alone (RWe: spherical k-means of the structural arrivals' unit difference vectors, no question,
  no gold), and weighting the walk by the arrival's type was carried (RELATIONAL_CHEAPER, 5404f89). Its section 14
  sent form (i), the family, the link kind and the direction, to U2n "as an edge-type input, where the models learn
  the weight". U2n's channels carry the family and the direction (section 8 says what they do not carry), and their
  directions start at RWe's centres (section 2); the GNN learns them inside its fit.
- **The device is KG's T4.** KG (docs/KG_KAGGLE_T4_LINE.md) fits U2d's base set again on Kaggle's T4: zrckp, zspk and
  zg6 at seed 0, and the seed nulls of zrckp and zspk. Session 1 finished; session 2 runs the seed-0 fits. U2n's fits
  run in the next session of the same kernel, against KG's fits on the same device. R2 waits for workspace 3's renewal
  on 1 November (GL part 2), so the T4 is free for U2n.
- **The pools.** U2n's screens run on U2d's base build, as KG's and G6b's do. No model change is adopted on today's
  pools (the user's rule of 13 October): a form that passes is fitted again on the pools U2s carries before its full
  run (section 6).
- **The risks, stated before any number.**
  - zg6e: G6 found zg6 above zspk on 7 of its 8 in-domain reads and below it on 3 of its 4 zero-shot reads (GL,
    section 7). Typed channels add capacity on the edges, which may fit the training graphs better still and transfer
    worse. The shared geometric prior, one channel for every graph, is the part meant to transfer. The types table
    (section 5) shows whether the learned directions stay near tau0.
  - zrckpt: its columns are fixed functions of the inputs (section 4). The program warns that fitted offsets fed in as
    fixed columns failed (P5b, P5c). These columns are counts by type, and nothing in them is fitted to a gold, but
    they are fixed. On a passage graph they carry only the geometric types of the edges from base rows, which KT's
    input partly sees already.

## 2. The types: tau0, the prior and the typed links

**K_T = 16 directions in the frozen embedding space.**
- **tau0**, per split: 16 spherical centres (seed 0, at most 25 rounds) of U2m (a)'s 64 unit type centres of each of
  the split's four training datasets (256 rows). Each centre file is checked against its record (U2m (a)'s stage, its
  declaration and K = 64, the file's and the centres' sha256). webqsp never trains, and the held-out dataset's centres
  are never read. Written as float32 rows with a record (`zu2n.py tau SPLIT`, outputs/u2n/tau0/).
- **The store.** P = E V holds each row's first 256 store columns (lean_mlp3.Store): the frozen embedding E on the
  basis's 256 axes V. A direction tau (1,536-d) enters the store as tau V.

**The prior** of an edge u → v over the K_T types:

    pi_k = kappa_d cos(P[v] - P[u], tau_k V) * keep_q  (+ on a typed KB edge, its relation term)

- The geometric term is on every edge of every family, on all six datasets. A difference shorter than 1e-6 has none.
  keep_q is the question's SEMB keep (the card's block keep), since the term reads the SEMB block.
- **The relation term**, on a KB's structural edge with a relation set: the log-mean-exp, over the set's relations r
  and the signs its direction bits allow, of sign · kappa_r cos(r, tau_k). r is rmatch's centred relation vector,
  normalised again. A relation vector has no direction, so the sign carries it: + when a stored triple runs from u to
  v, − when one runs back, both when the edge has no bit.
- kappa_d and kappa_r start at 10.

**The typed links** (`zu2n.py typed DS CARVE`), for the three KB carves the fits read: metaqa fit, metaqa s1eval and
webqsp s1eval.
- For each link edge of zlink (in zlink's order), from the look's chunks:
  - t_set: the edge's relation set, the union of its look messages' valid relation slots, as an id into the carve's
    sets (at most 8 relations; the build stops on more);
  - t_dir: two bits, 1 when a stored triple runs from the edge's source to its target, 2 when one runs back.

  Only the structural family carries sets. A structural edge whose messages carry no relation has none (t_set −1).
- **The alignment check.** Each part's link arrays are recomputed from its chunks and must equal its link record's
  sha256 bit for bit, and its relations must be its rmatch chain record's. The sets get ids by first appearance over
  the carve's chunks in order.
- **The chunks.** Only workspace 3's mpr-u2d volume holds the look's chunks, so they were copied read only into the
  laptop's scratch folder. Each read was checked by its size against the volume's listing and by zip's CRC-32 on
  every member; the alignment check then catches a chunk with other content.

## 3. zg6e: zg6 with typed channels

zg6e is zg6's network (docs/G6_FREE_GNN.md, section 2), unchanged: its encoder, boundary, context, three layers,
readout and edge-family dropout, trained as zspk's and zg6's fits are (P1c2's card, P1c3's KT input, p@swa, hidden
128).

What is new, in each of the three layers k:
- **The posterior** of each edge u → v:
  q = softmax(pi + a · h_u + b · h_v + c · ctx_q), with h the layer's input states and ctx_q zg6's question context
  (the weighted sum of the question's states). a, b and c (16 × 64 each) are shared by the layers.
- **The typed message and logit:**
  - msg = (W_f h_u) ⊙ (1 + q δ_k) + q β_k, in place of zg6's W_f h_u;
  - the attention logit gains q α_k.

  δ_k and β_k are 16 × 64 and α_k is 16, per layer.
- **The start.** tau at tau0, both kappas at 10, and everything else new at 0. The messages are then zg6's, so zg6e's
  score is zg6's bit for bit.
- **The EM term.** The prior is trained toward the layers' mean posterior (detached): per question, the mean over its
  edges of the cross entropy of softmax(pi) against it, averaged over the batch's questions with edges, weight 1 (R2's
  lambda). The listwise loss trains every parameter, the prior's included through the posterior. A fit where no batch
  carried the term stops unstamped.
- **The family dropout** drops the same edges from the channels as from the layers (one draw per question and family
  from zg6's generator).
- **Parameters.** 155,644 at hidden 128: zg6's 121,802 and U2n's 33,842 (tau 24,576, the kappas 2, a, b and c 3,072,
  and the layers' δ, β and α 6,192).
- **Memory.** The three layers are checkpointed in training: their activations are recomputed in the backward pass,
  so the per-edge type tensors do not stay on the T4 for the whole stack.

## 4. zrckpt: zrckp with typed columns

zrckpt is zrckp (docs/P1C3P_POINTWISE_HEAD.md) plus a pointwise head on 96 columns per row.
- **The columns.** For each family f and type k, T[f, k] is tau0's prior (both kappas 10) through a softmax over the
  types, summed over the row's incoming edges of family f from base rows. A base row is a row with KT's fill column at
  0 (the base and seed rows). The columns are log1p(T) (48) and log1p(T) times the question's cosine to tau0_k (48).
  They are multiplied by the question's SEMB keep.
- **Why fixed.** The columns use tau0 and fixed kappas, so they are inputs computed before any row is scored, from
  the row's own edges and frozen vectors. Learning the directions inside the MLP would make each column a learned
  function of the neighbours' vectors, a message-passing step, which the MLP rule excludes (docs/SCREENS.md).
- **The head:** a GELU layer of 16 and a linear output, the output layer at 0 at the start (zrckp's score bit for
  bit), the first layer drawn from the seed plus 3701. 1,569 parameters beside zrckp's.

## 5. The fits, the device and the compares

Four fits on U2d's base build, seed 0:

| U2n fit | split | its bases, KG's fits on the same T4 |
| --- | --- | --- |
| `scr-u2n-zg6e` | L-musique | `scr-kg-base-zspk` (decides), `scr-kg-base-zg6` (beside) |
| `scr-u2n-zg6e-hp` | L-hotpotqa | `scr-kg-base-zspk-hp` (decides), `scr-kg-base-zg6-hp` (beside) |
| `scr-u2n-zrckpt` | L-musique | `scr-kg-base-zrckp` (decides) |
| `scr-u2n-zrckpt-hp` | L-hotpotqa | `scr-kg-base-zrckp-hp` (decides) |

- **The same path as KG's.** zu2n.py trains and reads through zu2d.py's train and read bodies as zkg.py runs them:
  U2d's base build's roots and gate, P1c2's card, P1c3's KT input, the seed from the name, and tau0 of the fit's
  split. Each stamp records the device, the build, the pools, the KT, typed, tau0, rmatch, zrc and zlink records, the
  model's settings and zg6e's EM per epoch.
- **The reads.** p@swa on the six s1eval carves, with the arm's rows a batch: 32,768 for zg6e (half of G6's 65,536,
  for its type tensors) and 65,536 for zrckpt. A batch holds whole questions, so each question is scored as in a
  larger batch. Any other cap stops.
- **The device is KG's.** Train and read run only in a Kaggle kernel on a T4 from KG's image
  (kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3), checked before any data is read. A read must run on its fit's
  device.
- **Two compares** for each split: lean_screen's compare (p@swa on s1eval) of each fit against its bases, all on one
  device, each stamped with U2n's record:
  - `scr-u2n-zg6e-vs-zspk` and `-hp`: zg6e against KG's zspk (first, deciding) and KG's zg6 (beside);
  - `scr-u2n-zrckpt-vs-zrckp` and `-hp`: zrckpt against KG's zrckp.
- **The re-calls.** screen_pair over each arm's two compares against its deciding base, then screen_recall under
  KG's seed null of that base (zg6e: zspk's; zrckpt: zrckp's), with KG's floors max(0.0075,
  2·sqrt((D1² + D2²)/2)) from KG's seed-1 and seed-2 fits: `outputs/u2n/u2n-zg6e-pair-recall` and
  `outputs/u2n/u2n-zrckpt-pair-recall`.

**Beside, deciding nothing:**
- zg6e against KG's zg6 under zspk's floors (`u2n-zg6e-vs-zg6-pair-recall`). The null's seed-0 R@5 is zg6's read as
  KG filed it (`scr-kg-zg6-vs-zspk`), which each U2n compare's zg6 column must equal.
- R@5 per read on the T4 for rrf, zspk, zg6, zg6e, zrckp and zrckpt, with the MLP shares (zrckp − rrf)/(zg6 − rrf)
  and (zrckpt − rrf)/(zg6e − rrf) where the GNN's gain over rrf exceeds 0.002.
- **The types** of each fit's p@swa (u2n_types.json, written in the kernel): both kappas, each direction's cosine to
  its tau0, and per KB per direction the softmax over its signed relations of sign · kappa_r cos(r, tau_k) (all 18 on
  metaqa, the top five of webqsp's 14,116). zrckpt's table is tau0's with kappas 10.
- zg6e's EM per epoch: the share of questions with the term, its mean cross entropy and both kappas.
- Seconds and memory of each fit, beside its KG base of the same split on the T4.

## 6. The rule and what each outcome leads to (fixed now)

**The rule**, per arm, on its re-call against its deciding base (twelve reads, six per split):
- **ADOPT_FOR_FULL_RUN:** no LOSS on any read and at least one GAIN.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** any of these:
  - KG's base verdict (outputs/kg/base-verdict.json) is not BASE_READY;
  - a fit or compare of the arm is missing or not U2n's;
  - the arm's fits are not on KG's device;
  - the re-call does not hold the twelve reads.

  The verdict names what is missing.

**What follows:**

| arm | outcome | next form |
| --- | --- | --- |
| zg6e | ADOPT_FOR_FULL_RUN | zg6e fitted again on U2s's carried pools, then its full run on all six splits beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency) measured again; each its own declared file |
| zg6e | NOT_ADOPTED, a GAIN and a LOSS | zg6e under G6b's gate: its edge pass and its no-edge pass mixed per question by the question's edge-family shares (one declared file) |
| zg6e | NOT_ADOPTED, no GAIN | U2n2: the posterior split per edge family, one type posterior per family with the prior shared (one declared file) |
| zrckpt | ADOPT_FOR_FULL_RUN | zrckpt fitted again on U2s's carried pools, then its full run on all six splits beside zrckp's, then Claims 1 and 3 measured again; each its own declared file |
| zrckpt | NOT_ADOPTED, a GAIN and a LOSS | zrckpt-g: the head times 2·sigmoid(a · sh + b) on the question's edge-family shares, the gate at 1 at the start (one declared file) |
| zrckpt | NOT_ADOPTED, no GAIN | R2's EM inside the MLP's fit, after GL's R2 verdict and the user's ruling on zrckpb (one declared file) |
| either | INCOMPLETE | what is missing is supplied (KG's base set, a fit, a compare or the re-call); nothing else changes |

A GAIN that a raised floor turns WITHIN counts as no GAIN. No verdict here closes typed channels (docs/SCREENS.md,
section 3b): each next form above is the idea's next form in each model.

## 7. How it runs

- **Before the kernel, on the laptop:**
  1. `zu2n.py typed metaqa fit`, `typed metaqa s1eval` and `typed webqsp s1eval`, each with `--chunk-root` at the
     copied chunks;
  2. `zu2n.py tau L-musique` and `tau L-hotpotqa`;
  3. the records committed (the arrays never are), then the arrays uploaded as one private dataset, mpr-u2n, with a
     manifest of each file's path, size and sha256;
  4. `zu2n_session.py stub` at that commit once origin holds it: the kernel's code file and a git bundle of main.
- **The kernel.** U2n runs as version 3 of KG's kernel, mpr-kg, not as a new kernel. Kaggle pins a kernel's image at
  its first version ("original"), and a new kernel would take the current image, which KG's device check refuses.
  Kaggle keeps only the latest version's output, so KG's session 2 output is saved before version 3 is pushed. The
  kernel attaches KG's nine datasets, mpr-u2n and the code bundle.
- **The session** (`zu2n_session.py run`, built on zkg_session.py's helpers):
  - it finds and checks KG's nine datasets as KG's session does;
  - it finds mpr-u2n by its manifest and refuses anything but U2n's arrays (.npy under typed/ or tau0/);
  - each array's sha256 must be the one U2n's records in the commit name: the carve's record for sets.npy, the part's
    record (whose own sha256 the carve's record names) for t_set and t_dir, and the split's record for tau0. Every
    array the records name must be there;
  - every file's size and sha256 is checked, and each is linked at its repository path (U2n's under outputs/u2n/);
  - each fit runs as `zu2n.py fit --name N` in its own process, with U2d's image environment and no `--rows-cap`.
- **Order.** The longest fits start first, one queue per T4: `scr-u2n-zg6e-hp` then `scr-u2n-zrckpt` on GPU 0,
  `scr-u2n-zg6e` then `scr-u2n-zrckpt-hp` on GPU 1. No fit starts after 8 hours.
- **Time.** On the T4, zg6b took 2,239 s and 4,347 s to train and 875 s and 786 s to read (L-musique, L-hotpotqa);
  zrckp and zspk took 960 to 1,200 s to train and 285 to 330 s to read. With the type tensors and the layers'
  recomputation, zg6e should take 1.3 to 2 times zg6b. The session should end within 3.5 hours. It costs nothing
  (Kaggle's weekly GPU hours).
- **Memory.** zg6b peaked at 10.29 GB on the T4's 15 GB. If a fit runs out of memory, the session records its exit
  code; a smaller batch is then an amendment declared before its rerun.
- **On the laptop, after the session:**

      python outputs/mp_unified/zu2n.py take --from <the kernel's output folder>
      python outputs/mp_unified/zu2n.py compare-all
      python outputs/mp_unified/zu2n.py verdict

- **What is committed.**
  - Committed: the typed and tau0 records, U2n's compares (outputs/screen/scr-u2n-*), and outputs/u2n: the pairs,
    the re-calls and the verdict, which carries the types, the EM and the timing.
  - Never committed: the arrays (t_set, t_dir, sets, tau0), the fits' folders, weights, reads, logs and session
    files.

## 8. What this file chose, against the program's text

1. **The families as the base build holds them:** structural (on the KBs, the triples), the package's named-entity
   family, and kNN. The program's U2n reads EL, R and kNN. U2m's E1 (a2a2627) filed NO_GAIN for EL in the named-entity
   slot, and the pools keep the package's NER.
   - U2m's section 20 sends EL's kinds (IS, CO) to U2n as edge types. This first form does not carry them: the base
     build's pool edges hold families, not EL kinds, so they need their own arrays aligned to zlink's edges, as the
     typed links are. They are the next typed input after this screen, whatever its verdict, in one declared file.
   - Likewise U2m (a)'s link kind (form (i)): on the base build it is the family.
2. **Per-type modulation, not per-type maps.** The program's M-step trains "one message map W_z and one type vector
   per latent type". A full 64 × 64 map per type and layer would add 196,608 parameters, 1.6 times zg6. zg6e gives
   each type a per-channel scale δ, a vector β (the type vector) and a logit offset α in each layer. A low-rank W_z is
   a later form.
3. **R2's EM is not in the MLP arm.** The program's MLP form is zrckp with the per-type relation columns and R2's EM.
   R2's MLP arm (zrckpb) runs in GL part 2, and whether zrckpb stays in the MLP track is the user's ruling. R2's EM
   inside the MLP is zrckpt's next form when it finds no GAIN (section 6).
4. **zrckpt's columns are fixed** at tau0 (section 4, "Why fixed").
5. **"Edges to B"** (program, section 4.2) is read as the edges from base rows, the rows with KT's fill column at 0.
6. **One pair of direction bits per edge.** An edge's bits are the union over its stored triples, so an edge whose
   set mixes a forward fact of one relation and a backward fact of another allows both signs for both.
7. **A structural KB edge without a relation** keeps only the geometric prior.
8. **The prior reads the store** (P = E V, 256 axes), not the 1,536-d vectors, so a pool's edges cost 256 per edge
   and both models read the same rows they already hold. U2m (a)'s centres came from the full vectors; tau0 lives in
   the full space and is projected by V.
9. **The relation vectors are centred as rmatch centres them** for its chain features, then normalised again.
10. **The SEMB keep** multiplies both the prior and the columns, since both read the SEMB block.
11. **Checkpointing and zg6e's read batch** (sections 3 and 5) are set for the T4's memory.
12. **Kernel mpr-kg, version 3** (section 7), for the image.

## 9. Self-tests, passed before this commit

`python outputs/mp_unified/zu2n.py --selftest` runs on toy carves and fake files in a temporary folder:
1. **The arms.** zg6e on P1c3's link carve and zrckpt on P1c3's chain carve, each under U2n's carve, which sets no
   root of its own; the earlier arms, the names, bases and sizes unchanged.
2. **Where.** train and read refuse anything but a Kaggle T4 on KG's image before touching data; another read cap
   stops.
3. **The typed links.** zlink's edges again from the toy chunks, each part's arrays equal to its link record's; the
   sets and bits equal a brute loop's; a carve that is not U2n's and a changed chunk stop.
4. **tau0.** K_T unit centres of the split's type centres, a zero row skipped and counted, the same bytes again; a
   centre file changed or not U2m (a)'s stops.
5. **The carves.** The channels' edges are zlink's in link_batch's order; the typed edges, sets and bits are the
   build's.
6. **The construction.** From seed 0, zg6e moves the global generator as zg6 does; its state is zg6's plus U2n's
   (tau at tau0, both kappas 10, the rest 0). zrckpt's is zrckp's plus the head.
7. **The start.** zg6e's score is zg6's and zrckpt's is zrckp's, bit for bit, in eval and in training, with the
   layers checkpointed or not.
8. **By hand.** The prior (the store's cosine and the relation term, on a crafted case with every direction code and
   empty slots), every parameter set at random, the layers with the posterior, and zrckpt's columns and score equal
   python loops over the questions and edges.
9. **A question without edges.** zg6e's rows of it are zg6's; zrckpt's columns are zero there. A batch without U2n's,
   the edges' or KT's key stops.
10. **The family dropout.** One draw per question and family from zg6's generator; the channels lose the edges the
    layers lose; the global generator moves as zg6's.
11. **EM.** One kept prior per training forward (a second stops); the term equals loops; the counts and kappas per
    epoch; everything restored after.
12. **Gradients** reach every zg6e parameter, the same with the layers checkpointed or not, and zrckpt's head.
13. **Toy fits** through lean_gpu's loop, zg6e under the EM term: a repeat is bit for bit; the encoder, the stack,
    the directions, the kappas, the posterior's weights, the readout and zrckpt's head move.
14. **The types table** equals one by hand; a vocabulary that is not its record's stops.
15. **Stamps and take.** A fit that is not U2n's, or a read on another device, is refused. take copies records, reads
    and the types file, never models.pt, and leaves KG's fits to KG.
16. **One device** across U2n's fit and KG's bases.
17. **The verdict** per arm, every outcome and its next form: ADOPT_FOR_FULL_RUN; NOT_ADOPTED with a GAIN and a LOSS;
    NOT_ADOPTED without a GAIN, a GAIN under a raised floor included; INCOMPLETE without KG's BASE_READY, without a
    compare, on another device, or with a zg6 column that is not KG's zg6 read; and the beside rows.
18. Nothing is written under U2n's real folders.

`python outputs/mp_unified/zu2n_session.py --selftest` runs on a toy repository and ten toy datasets:
1. find, plan, verify and link: KG's nine groups as KG's session links them, U2n's arrays at outputs/u2n/, the
   records git's own; a second pass changes nothing.
2. The refusals: a changed array, an array the records do not name, a missing one, other bytes than its record's, a
   record or another path in the manifest, a part record that is not the one its carve's record names, a record
   missing or not U2n's, the group missing or attached twice, a record git does not hold.
3. The run: each fit through `zu2n.py fit --name N` with U2d's image environment and no `--rows-cap`, the exit codes
   and seconds in session.json, one queue without two GPUs, nothing started after the start limit.
4. The stub: the commit, the bundle and the queues filled in; it runs U2n's session.
5. Only U2n's four fits, each once; run and stub refuse anything else first.
6. Its constants (the carves, splits, fits, declaration and folders) are zu2n.py's.

The text encoder stays frozen. webqsp never trains. Test splits are never read. There is no LLM or transformer, no new
graph, and no gold in any type, prior or column.

## 10. Results

### 27 October 2026: zg6e NOT_ADOPTED, zrckpt ADOPT_FOR_FULL_RUN

**Where it ran.** Kernel mpr-kg, version 3 (section 7), at edcff68 (zu2n.py and zu2n_session.py as at 3a5da10).
- The session's Python, platform, GPUs, driver and fit environment equal KG's session 2, and every fit's device stamp
  is KG's device.
- It first checked 1,586 files (26.1 GB) against their manifests (119 s): KG's nine datasets and mpr-u2n. It linked
  1,443 at their repository paths; 143 were the same in git.
- GPU 0 ran scr-u2n-zg6e-hp (7,135 s), then scr-u2n-zrckpt (1,390 s). GPU 1 ran scr-u2n-zg6e (4,067 s), then
  scr-u2n-zrckpt-hp (1,640 s). All four exited 0; the session took 8,646 s.

On the laptop at 8c1b477 (zu2n.py as at 3a5da10): `zu2n.py take`, `compare-all` and `verdict`. Records:
- outputs/screen: scr-u2n-zg6e-vs-zspk and scr-u2n-zrckpt-vs-zrckp, each with its -hp twin (.json and .md);
- outputs/u2n: u2n-zg6e-pair, u2n-zg6e-vs-zg6-pair and u2n-zrckpt-pair, each with its -recall, and screen-verdict
  (.json and .md; 2026-10-27T08:42:34Z).

**zg6e against KG's zspk**, re-called under KG's zspk null: **NOT_ADOPTED**.
- **3 GAIN, all in domain:**
  - L-musique metaqa: +0.0117 [+0.0100, +0.0135].
  - L-hotpotqa metaqa: +0.0076 [+0.0060, +0.0091].
  - L-hotpotqa musique: +0.0231 [+0.0169, +0.0291], floor 0.0105.
- **3 LOSS, all zero-shot:**
  - L-musique webqsp: -0.0988 [-0.1198, -0.0773].
  - L-hotpotqa hotpotqa: -0.0311 [-0.0352, -0.0267].
  - L-hotpotqa webqsp: -0.0386 [-0.0603, -0.0170], floor 0.0242.
- **One call changed** with the re-call: L-hotpotqa 2wiki (+0.0172 [+0.0150, +0.0193], floor 0.0219) went from GAIN
  to WITHIN.
- **Next form**, fixed in section 6 for a GAIN and a LOSS: **zg6e under G6b's gate**, its edge pass and its no-edge
  pass mixed per question by the question's edge-family shares, in one declared file. It goes on the T4 after G6r,
  which G6b2 fixed first (docs/G6B2_ROW_BACKOFF.md, section 9).

**zrckpt against KG's zrckp**, re-called under KG's zrckp null: **ADOPT_FOR_FULL_RUN**.
- **1 GAIN:** L-musique musique (zero-shot): +0.0098 [+0.0055, +0.0144], floor 0.0075.
- **0 LOSS.**
- **Two calls changed** with the re-call:
  - L-hotpotqa hotpotqa (zero-shot): +0.0080 [+0.0051, +0.0108], floor 0.0084, went from GAIN to WITHIN.
  - L-hotpotqa webqsp (zero-shot): -0.0108 [-0.0221, -0.0006], floor 0.0237, went from LOSS to WITHIN.
- **Next form**, fixed in section 6: zrckpt fitted again on U2s's carried pools, then its full run on all six splits
  beside zrckp's, then Claims 1 and 3 measured again, each in its own declared file. The refit needs U2s's verdict and
  the model inputs of its carried pools.
- Section 8 also fixed **EL's kinds (IS, CO) as edge types** aligned to zlink's edges as the next typed input, whatever
  this verdict. It gets its own declared file.

Beside, deciding nothing:
- **zrckpt, read by read.** The one GAIN clears its floor by 0.0023.
  - Both passage zero-shot reads went up: musique +0.0098 on L-musique and hotpotqa +0.0080 on L-hotpotqa.
  - Both webqsp reads went down, within their floors: -0.0053 on L-musique and -0.0108 on L-hotpotqa.
  - Every in-domain read is within 0.0042 of zrckp.

  The full run will show whether the webqsp reads stay within their floors.
- **zg6e against KG's zg6**, under zspk's floors: 0 GAIN and 2 LOSS, both zero-shot:
  - L-musique webqsp: -0.0235 [-0.0372, -0.0108].
  - L-hotpotqa hotpotqa: -0.0153 [-0.0184, -0.0120].

  L-hotpotqa webqsp (-0.0176 [-0.0324, -0.0044], floor 0.0242) went from LOSS to WITHIN. Every in-domain read is within
  0.0032 of zg6, so zg6e's in-domain GAINs are zg6's. The typed channels added nothing in domain and lost on two
  zero-shot reads: the transfer risk section 1 named, without the better fit it expected.
- **The MLP share** of KG's zg6's gain over rrf, (MLP - rrf) / (zg6 - rrf), on the T4:

  | read | zrckp | zrckpt |
  |---|---:|---:|
  | L-musique metaqa, squad, hotpotqa, 2wiki (in domain) | 0.9837, 0.9394, 0.9623, 0.9617 | 0.9834, 0.9394, 0.9733, 0.9742 |
  | L-musique musique, webqsp (zero-shot) | 0.9777, 1.6291 | 1.2228, 1.6006 |
  | L-hotpotqa metaqa, squad, musique, 2wiki (in domain) | 0.9905, 0.9265, 0.8277, 0.9678 | 0.9906, 1.0882, 0.8433, 0.9663 |
  | L-hotpotqa hotpotqa, webqsp (zero-shot) | 1.0731, 1.4533 | 1.1243, 1.3963 |

  The one in-domain share below 0.90 is still L-hotpotqa musique's (0.8433). The verdict's table gives the shares
  against zg6e as well.
- **The types.** zrckpt's columns use tau0 with both kappas at 10, as section 4 fixed: cosine 1 to tau0.
  - zg6e's learned directions moved far from tau0: median cosine -0.021 (L-musique) and +0.015 (L-hotpotqa), with
    kappa_d 11.0 and 22.0 at p@swa.
  - On metaqa each direction weighs a few of the nine relations, for example +has imdb votes 0.30 and -has genre 0.27
    on L-musique's first.
  - On webqsp, no relation reaches 0.02 in the first four directions of any fit (the verdict prints those four).
- **zg6e's EM term** was on every question in every epoch. Its mean cross entropy was 2.08 to 2.68 on L-musique and
  1.55 to 1.79 on L-hotpotqa.
- **Seconds and memory**, against KG's fits of the same split on the same T4:

  | fit | train s | read s | peak GPU GB, train / read |
  |---|---|---|---|
  | scr-u2n-zg6e | 2,872, 1.48x zg6's 1,942 | 1,180 (773) | 4.11 / 4.67 (zg6 4.66 / 5.24) |
  | scr-u2n-zg6e-hp | 6,055, 1.51x zg6's 3,998 | 1,067 (687) | 8.21 / 8.21 (zg6 10.29 / 10.29) |
  | scr-u2n-zrckpt | 1,060, 1.10x zrckp's 960 | 323 (291) | 3.71 / 4.77 (zrckp 3.70 / 4.43) |
  | scr-u2n-zrckpt-hp | 1,280, 1.33x zrckp's 963 | 353 (279) | 5.56 / 5.56 (zrckp 5.55 / 5.55) |

  - zg6e's layers were checkpointed in training (section 3), so its peak is below zg6's.
  - Peak RAM was 4.5 to 4.6 GB for every fit.
  - zrckpt's cold cost, the typed links and the prior per edge, is measured in its Claim 3 step, not here.
