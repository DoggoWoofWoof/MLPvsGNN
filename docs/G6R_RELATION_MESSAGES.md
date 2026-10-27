# G6r: relation-aware messages from text, on Kaggle's T4

Declared 27 October 2026. Code: `outputs/mp_unified/zg6r.py` and its session, `outputs/mp_unified/zg6r_session.py`.
Its fits run on KG's device. No G6r fit has run. Nothing in this file reads a G6r number.

## 1. Why this, now

- **G6b2's call fixes this form.** G6b2 was filed NOT_ADOPTED on 27 October (docs/G6B2_ROW_BACKOFF.md, section 9): 3 GAIN,
  all in domain, and 3 LOSS, all zero-shot, against KG's zspk, re-called under KG's zspk null. Before any number, its
  section 6 fixed the next form for NOT_ADOPTED, with or without a GAIN: G6's own NO_GAIN form (G6, section 5), "G6r,
  relation-aware messages from text: the relation embeddings (`outputs/m3b/relations/metaqa_rel_embeddings.npy`,
  `webqsp_rel_embeddings.npy`) through one shared low-rank map in place of the per-family maps on the KB datasets, the
  encoder started from zspk's trained encoder, still unbounded with its own readout (one declared file)". This is that
  file.
- **Why the messages, not the mix.** G6b mixed zg6's score with its twin's per question, and G6b2 per row. Both kept
  zg6's in-domain GAINs and both lost on zero-shot reads, so the next change is to what the edges send (G6b2, section 6).
- **What G6r changes, in two parts**, both from G6's text:
  1. **The messages on a KB's typed edges.** In zg6, every structural edge sends W_0 h_u, one map whatever relation the
     edge carries. On metaqa's training carve, 73.5M of the pools' 84.3M link edges carry relations (U2n's typed links,
     docs/U2N_TYPED_CHANNELS.md, section 2), and all of them are structural. G6r sends each such edge's message through
     one low-rank map that the edge's relations condition, and adds the relations' match to the question's context to
     the edge's attention logit. The map is shared by every relation and KB. A relation enters only through its text
     vector (rmatch's relation embeddings, from the frozen text encoder), so a relation no fit has seen still gets a
     code.
  2. **The encoder's start.** zg6's encoder (zrm's block encoder and chain match) trains from a random start, with the
     layers. G6r starts it from KG's zspk fit of the same split (its p@swa), the base that decides every G6 call.
- **Beside U2n.** U2n's zg6e also typed the KB edges, through learned directions in relation space that weigh its
  channels (U2n, section 10): 3 GAIN in domain and 3 LOSS zero-shot against KG's zspk, and on webqsp no relation reached
  0.02 in its first four directions. G6r puts the relations into each message and its logit instead of into channel
  weights, and changes the encoder's start.
- **What it reads.** It reads no dataset, split, gold or row label. The relation vectors are each KB's own. The typed
  links are U2n's, over each question's own pool edges, checked against U2n's records. The form stays G6's: a free GNN
  with its own encoder, messages between hidden states and an unbounded readout of its own.
- **The device is KG's T4**, as for G6b and G6b2.
- **The risks, stated before any number.**
  - **P learns from metaqa's nine relations alone**, since webqsp never trains. Their centred vectors span 8 of 1,536
    dimensions, and P's gradient lies in that span; only Adam's per-coordinate steps move P outside it. From the vectors
    alone (no fit), webqsp's 7,058 relations keep on average 0.043 of their squared norm inside metaqa's span (median
    0.031, maximum 0.633). So webqsp's codes are likely small, and a typed message there is then close to
    A_k (δ_k ⊙ B_k h_u): one learned map for every typed edge, whatever its relations.
  - **Both directions.** 4.02M of webqsp's 12.70M typed s1eval edges carry both direction bits. No metaqa edge does: each
    is forward or backward, in its training carve as in its s1eval one. The code of a both-way edge (section 2) is never
    seen in training.
  - **The encoder's start changes every dataset**, the passage graphs included, where no edge is typed. A change on a
    passage graph's read comes from the start, not the messages. Only the next form, if no read gains, separates the two
    (section 6).
  - **Not at equal training.** zg6r's encoder has trained on the split's questions twice: zspk's eight epochs, then
    zg6r's own. Its compares with zspk and zg6 are not at equal training.
  - **Memory.** Each typed edge carries a 16-wide code and its own message. The layers are checkpointed in training, as
    zg6e's were, and each read batch holds 32,768 rows (zg6e's), not 65,536.
  - **The start.** zg6r starts as zg6 would with the same encoder weights only where no edge is typed. A typed message
    starts as A_k B_k h_u, a rank-16 map, not zg6's W_0 h_u.

## 2. The model

zg6r is zg6 (G6, section 2), unchanged except on typed edges and in its encoder's start:
- zg6's encoder, boundary, context, three layers and readout;
- its edge-family dropout in training, with the same draws from the module's own generator (a typed edge goes with
  its family, the structural one);
- its training: rmatch.py's train on P1c2's card, as zspk's and zg6's fits.

What is new:
- **The relations.** On a KB carve, rmatch's relation vectors of that KB, centred over its relations and normalised
  again, as zg6e's carve holds them (rel_centered). A typed edge u -> v names a set of up to 8 relations and two
  direction bits: 1 when a stored triple runs from u to v, 2 when one runs back (U2n's t_set and t_dir).
- **The code.** r_e is the mean over the edge's relations r of:
  - P_f r when only bit 1 is set;
  - P_b r when only bit 2 is set;
  - (P_f r + P_b r) / 2 when both bits are set, or neither.

  P_f and P_b are 16 x 1536 (K_R = 16), shared by every relation and KB.
- **The typed message**, in layer k: m_e = A_k ((r_e + δ_k) ⊙ (B_k h_u)), in place of zg6's W_0 h_u. A_k is 64 x 16,
  B_k 16 x 64 and δ_k has 16 entries (64 is zg6's state width).
- **The typed logit**, in layer k: zg6's attention logit plus r_e · (C_k c_q) / sqrt(16), with c_q the question's
  context in that layer and C_k 16 x 64. zg6's clip at ±8 comes after.
- **Every other edge** is zg6's: its family's map and its logit. That covers the passage graphs' edges, and a KB's NER
  and kNN edges and structural edges without relations.
- **The start.** P and C are 0, δ is 1, and A and B are uniform (fan 16 and 64), drawn from the seed plus 3801. So every
  logit starts as zg6's, and a typed message starts as A_k B_k h_u. With no typed edge, the score is zg6's, bit for bit,
  for the same encoder weights.
- **The encoder's start.** When zg6r is built for a fit, its 15 encoder tensors are copied from KG's zspk fit of the
  same split (its p@swa), then trained with everything else. They are zrm's block encoder (U, V), its chain match (the
  nine cm_ tensors) and its two encoder layers (l1, l2): 224,737 numbers at hidden 128. The file is that fit's
  models.pt, pinned by sha256 (section 7). The copy refuses:
  - a file that is not KG's zspk fit of the split at seed 0;
  - a missing tensor, or one of another shape.

  A fit where no zg6r was built from the encoder, or whose basis or width is not the encoder's, stops unstamped.
- **Parameters.** 180,218 at hidden 128: zg6's 121,802 and G6r's 58,416. P_f and P_b hold 49,152; each layer's A, B, δ
  and C hold 3,088.
- **Checkpointing.** The layers are recomputed in the backward pass in training, as zg6e's were. This changes no
  number.

## 3. The fits and the device

Two fits on U2d's base build, seed 0:

| G6r fit | split | its bases, all on the same T4 |
| --- | --- | --- |
| `scr-g6r-zg6r` | L-musique | KG's `scr-kg-base-zspk` (decides); KG's `scr-kg-base-zg6` (beside) |
| `scr-g6r-zg6r-hp` | L-hotpotqa | KG's `scr-kg-base-zspk-hp` (decides); KG's `scr-kg-base-zg6-hp` (beside) |

- **The same path as KG's.** zg6r.py trains and reads through zu2d.py's train and read bodies, as zkg.py runs them:
  - U2d's base build's roots and gate, P1c2's card, and seed 0;
  - reads of p@swa on the six s1eval carves, 32,768 rows a batch (whole questions, so the scores do not change). fit
    refuses any other batch.

  Its stamp records the device, the build, the pools, the KT, typed-link, rmatch, zrc and zlink records, the model's
  settings, its bases, and the encoder's record: its file, sha256, fit, seed and size, and how many zg6r were built
  from it.
- **The device is KG's.** Train and read run only in a Kaggle kernel on a T4 from KG's image
  (kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3), checked before any data is read. A read must run on its fit's
  device.
- **Before G6r is called:**
  - KG's base verdict (outputs/kg/base-verdict.json) must be BASE_READY.
  - G6b2's verdict (outputs/g6b2/screen-verdict.json) must be NOT_ADOPTED with G6r as its next form.
  - G6r's two fits must be on KG's device.
- **Memory.** On the T4, zg6 peaked at 4.66 GB training and 5.24 GB reading on L-musique, and 10.29 GB on L-hotpotqa,
  reading 65,536 rows a batch. zg6e, checkpointed and reading 32,768 rows, peaked at 4.11 / 4.67 GB and 8.21 GB. The
  T4 has 15 GB. If a fit runs out of memory, the session records its exit code; a smaller batch is then an amendment
  declared before its rerun. Each fit's g6r_timing.json records its peaks.

## 4. The compares, the null and the rule

- **Two compares**, `scr-g6r-zg6r-vs-zspk` and `-hp`. Each is lean_screen's compare (p@swa on s1eval) of zg6r's fit
  against two bases:
  - KG's zspk, which comes first and decides;
  - KG's zg6, beside.

  All three fits must be on one device. Each compare is stamped with G6r's record.
- **The call.**
  1. screen_pair over the two compares against zspk: `outputs/g6r/g6r-zg6r-pair`.
  2. A re-call under KG's zspk seed null: `outputs/g6r/g6r-zg6r-pair-recall`. KG's floors are screen_recall's,
     max(0.0075, 2·sqrt((D1² + D2²)/2)), from KG's seed-1 and seed-2 zspk fits.
- **The rule** is G6's (G6, section 4), on the twelve reads:
  - **ADOPT_FOR_FULL_RUN:** no LOSS on any read and at least one GAIN.
  - **NOT_ADOPTED:** otherwise.
  - **INCOMPLETE:** any of these:
    - KG's base verdict is not BASE_READY;
    - G6b2's verdict is not NOT_ADOPTED with G6r next;
    - a fit or compare is missing, or is not G6r's;
    - the fits are not on KG's device;
    - a compare's zg6 column is not zg6's read as KG filed it (section 5);
    - a re-call does not hold the twelve reads.

    The verdict names what is missing.

## 5. Beside, deciding nothing

- **zg6r against zg6**, the compares' second base, through the same pair and re-call
  (`outputs/g6r/g6r-zg6r-vs-zg6-pair` and `-recall`).
  - KG fits no zg6 seed null, so the floors are zspk's, as a proxy.
  - zg6's seed-0 R@5 comes from KG's own `scr-kg-zg6-vs-zspk` compares. Each G6r compare's zg6 column must equal it.
  - This read says what the two changes together do over zg6.
- **R@5 per read on the T4** for rrf, zspk, zg6, zg6r and zrckp (KG's seed-0 fit, the MLP). The MLP share,
  (zrckp - rrf) / (zg6r - rrf), is shown where zg6r's gain over rrf exceeds 0.002.
- **The relation map** of each fit's p@swa:
  - P_f's and P_b's norms and top singular values;
  - the share of each one's norm inside metaqa's centred relation span, the only relations a fit trains on;
  - each metaqa relation's |P_f r| and |P_b r|, and webqsp's top five by |P_f r|, with the mean share of webqsp's
    relations inside metaqa's span;
  - per layer, δ's mean, minimum and maximum, and the norms of A, B and C;
  - the encoder's drift from zspk's, |θ - θ_zspk| / |θ_zspk| over the 15 tensors.

  The kernel computes these from the fit's models.pt and the relation vectors, each KB's vocabulary checked against
  its record, and writes g6r_rel.json. The laptop never holds the weights.
- **Seconds and memory** for each fit, beside KG's zg6 and zspk of the same split on the same T4.

## 6. What each outcome leads to (fixed now)

| zg6r's re-call under KG's zspk null | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zg6r fitted again on U2s's carried pools, then its full run on all six splits beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) measured again; each its own declared file |
| NOT_ADOPTED, with a GAIN and a LOSS | zg6r under G6b's gate (its edge pass and its no-edge pass mixed per question by the question's edge-family shares, as zg6b mixes zg6's; the twin's encoder now starts from zspk's); one declared file |
| NOT_ADOPTED, with no GAIN | the two changes apart: the relation messages with zg6's own encoder (zg6r-m) and zg6 with zspk's encoder start (zg6-z), each one fit pair; one declared file |
| INCOMPLETE | what is missing is supplied (KG's base set, G6b2's verdict, a fit, a compare or a re-call); nothing else changes |

- **Why G6b's gate after a GAIN and a LOSS.** G6b's per-question gate recovered about half of zg6's loss on L-musique's
  webqsp read (G6b2, section 1). With the relation messages in the edge pass and zspk's encoder under both passes, it
  is the mix to try next.
- **Why the two changes apart after no GAIN.** G6r makes two changes at once. If neither helps together, the next file
  measures each alone.
- **Why U2s's pools.** U2s (the walk-mass budget, one construction for all six datasets) is the pools stage the program
  carries, as U2n's and G6b2's sections 6 name it.

No verdict here closes the free GNN (docs/SCREENS.md, section 3b). No model change is adopted on today's pools (the
user's rule of 13 October).

## 7. How it runs

- **Its own session.** zg6r_session.py, standard library only, is built on zkg_session.py's and zu2n_session.py's
  helpers. It finds three sets of private datasets on the account:
  - KG's nine, checked and linked as KG's session does it;
  - U2n's tenth (mpr-u2n): the typed links' arrays, which G6r's carve reads, and tau0's, which G6r does not read. Each
    array must be the one U2n's git-held records name;
  - G6r's eleventh, **mpr-g6r-enc**, a new private dataset on the user's Kaggle account. It holds the two models.pt files
    of KG's session-2 fits `scr-kg-base-zspk` and `scr-kg-base-zspk-hp` (at e0278d1; 8,182,795 bytes each; kept on the
    laptop from that session's output, never committed), as enc/L-musique.pt and enc/L-hotpotqa.pt, with a
    manifest.json (group g6r-enc). Their sha256 are pinned in zg6r.py and in the session:
    - L-musique: 7998083f15fb57a4333f943881ac8a68433ed7c48591427f89f63b196bc305ab;
    - L-hotpotqa: 52bca67c58fdbd91676493d4ecccb64c62fbeb63d3b34cc19040d1ba02593602.

  Every file's size and sha256 is checked. Each file is linked at its repository path (G6r's under outputs/g6r/enc/),
  and outputs/screen/fits is linked to /kaggle/working/fits. Each fit runs as `zg6r.py fit --name N` in its own
  process, with U2d's image environment and no `--rows-cap`. A name that is not G6r's, or a fit queued twice, is refused
  before any data is read.
- **Nothing else changes.** zkg.py, zkg_session.py, zg6b2.py, zu2n.py and zu2n_session.py are not changed.
  `zg6r.py take` takes only scr-g6r- folders. KG's, G6b2's and U2n's takes are never run on a G6r session's output.
- **The session.** One session of whichever of the account's two kernels (mpr-kg, mpr-kg-g6b2) is free, at this
  commit:
  - GPU 0: `scr-g6r-zg6r-hp`;
  - GPU 1: `scr-g6r-zg6r`.

  No fit starts after 8 hours. Where a fit runs decides nothing.
- **Time.** zg6e, the nearest form (typed messages, checkpointed layers, 32,768-row reads), took 2,872 s and 6,055 s to
  train on the T4 (1.48 and 1.51 times zg6's) and 1,180 s and 1,067 s to read (U2n, section 10). zg6r should take about
  as long, so the session should end within 2.5 hours after the datasets' check. It costs nothing (Kaggle's weekly GPU
  hours).
- **Commands.** In the kernel, the stub runs `zg6r_session.py run --gpu0 scr-g6r-zg6r-hp --gpu1 scr-g6r-zg6r`. On the
  laptop, after the session:

      python outputs/mp_unified/zg6r.py take --from <the kernel's output folder>
      python outputs/mp_unified/zg6r.py compare-all
      python outputs/mp_unified/zg6r.py verdict

- **What is committed.**
  - Committed: G6r's compares (outputs/screen/scr-g6r-*) and outputs/g6r: the pairs, the re-calls and the verdict,
    which carries the relation maps and the timing.
  - Never committed: the fits' folders, weights, reads, logs and session files, and the encoder files
    (outputs/g6r/enc).

## 8. What this file chose, against G6's text

1. **"One shared low-rank map" is A_k diag(r_e + δ_k) B_k**, conditioned by the relations' codes through P_f and P_b
   (16 wide), one per layer. The relations scale the map's 16 directions; they do not choose among maps.
2. **Only edges with relations change.** "In place of the per-family maps on the KB datasets" is read as: on a KB's
   structural edges that carry relations. A KB's NER and kNN edges, and its structural edges without relations, carry
   no relation to condition on, so they keep zg6's maps.
3. **A logit term.** "Relation-aware messages" is read to include each typed edge's attention logit: the relations'
   code against the question's context. At the start (C = 0) the logits are zg6's.
4. **Direction by two maps.** A relation vector has no direction, so P_f and P_b carry it (U2n carries it with a sign).
5. **The centred vectors**, rel_centered normalised again, as zg6e's carve and U2n's relation term read them, not the
   raw unit vectors. Each KB's mean direction is common to all its relations.
6. **"The encoder started from zspk's trained encoder"** is the 15 tensors zspk and zg6 share, from zspk's p@swa.
   zspk's KT and rank heads and its readout are not zg6's, so they are not copied.
7. **32,768 rows a read batch**, zg6e's, for the per-edge codes.
8. **Its own session.** G6r needs U2n's dataset and its own beside KG's nine, so it runs through zg6r_session.py, not
   through KG's session.

## 9. Self-tests, passed before this commit

`python outputs/mp_unified/zg6r.py --selftest` runs on a toy carve and fake files in a temporary folder:
1. **The arm.** zg6r sits on P1c3's link carve under G6r's carve, which has no root of its own, with zg6's classes
   below it. zg6, zspk, zg6b2 and zg6e are unchanged.
2. **Where.** train and read refuse anything but a Kaggle T4 on KG's image, and any name that is not G6r's, before
   reading data. fit refuses another read batch. A missing encoder file stops.
3. **The carve.** The typed links are U2n's, aligned with zlink's edges in their batch order. Each typed edge's
   combination holds its relation set and direction bits as built. Typed edges are structural, and the relation
   vectors are unit. The carve grows by the relation vectors alone.
4. **The start.** From seed 0, building zg6r moves the global generator as building zg6 does. Its state is zg6's plus
   G6r's: P and C at 0, δ at 1, A and B drawn from the seed plus 3801. The dropout's generator starts where zg6's does.
5. **No typed edge.** The score is zg6's bit for bit, in eval and in training (checkpointed, with the same draws). On
   the KB batch at the start the scores differ, and the score equals python loops over the questions and edges.
6. **By hand.** The codes, on a crafted case with every direction code and empty slots, equal loops. With every
   parameter set at random, the layers with their typed messages and logits equal loops.
7. **A question without edges** scores as zg6 on its rows. A batch without G6r's typed links, the edges or the KT
   input stops, and so does one whose typed edges come without their relation vectors.
8. **The dropout.** In training there is one B x 3 draw from zg6's generator. A typed edge goes with its family: with
   some typed edges kept and some dropped, the score equals the by-hand model over zg2.family_keep's edges. The global
   generator moves as zg6's does.
9. **Gradients** reach every parameter, checkpointed or not, and are the same both ways.
10. **zspk's encoder.**
    - Its file is loaded with its sha256, fit, seed and size checked.
    - It is copied into each zg6r built while it is active, and into nothing else (not into a zg6). The draws and
      every other tensor are as without it.
    - A state that does not fit stops. After a fit, no copy, or another basis, stops.
11. **Toy fits** through lean_gpu's loop, with the encoder active, repeat bit for bit. The encoder, the stack, the
    readout and every G6r parameter move.
12. **The relation map.** From a p@swa state, P's norms, span shares and codes, the layers' δ and norms, and the
    encoder's drift equal a computation by hand. A vocabulary that is not its record's stops.
13. **Stamps and take.**
    - It refuses a fit that is not G6r's, and a read on another device.
    - take copies records, reads, the relation file and the timing file, never models.pt, and leaves KG's fits to KG.
    - take refuses an unread fit, a name that is not G6r's, and a file with other bytes.
14. **One device** across G6r's fit and KG's two bases.
15. **The verdict**, each outcome with its beside rows (the zg6 re-call, R@5, the MLP share, the relation map and the
    timing):
    - ADOPT_FOR_FULL_RUN;
    - NOT_ADOPTED with G6b's gate next: a GAIN and a LOSS;
    - NOT_ADOPTED with the two changes apart next: no GAIN; a GAIN that a raised floor turns WITHIN; a LOSS alone.

    INCOMPLETE in eight cases:
    - without KG's verdict file, or with KG's verdict not BASE_READY;
    - without G6b2's verdict file, with G6b2's verdict ADOPT_FOR_FULL_RUN, or NOT_ADOPTED with another next form;
    - without a compare;
    - with the fits on another device than KG's;
    - with a zg6 column that is not KG's zg6 read.
16. Nothing is written under G6r's real folders.

`python outputs/mp_unified/zg6r_session.py --selftest` runs on a toy repository and toy datasets:
1. find, plan, verify and link: KG's nine groups and U2n's as their sessions link them, and G6r's two encoder files
   at outputs/g6r/enc/. A second pass changes nothing.
2. **Refusals:**
   - the real pins refuse the toy's files;
   - an encoder file changed, missing, with other bytes than its pin, at another path, or listed twice;
   - an empty manifest;
   - a missing U2n array;
   - G6r's group missing or attached twice.
3. **The run on the toy.** Each fit runs through `zg6r.py fit` with U2d's image environment and no `--rows-cap`, and
   its exit code and seconds go into session.json. Without two GPUs there is one queue. Nothing starts after the start
   limit. The real pins refuse before any fit.
4. **The stub's text.** A HEAD that lacks the session file is named as untracked, so the stub refuses it.
5. **The names.** Only G6r's two fits, each once.
6. **The constants** equal zg6r.py's.

The text encoder stays frozen. webqsp never trains. Test splits are never read. There is no LLM or transformer, and no
new graph.

## 10. Results

Not run yet. **Held on 27 October, before any G6r fit ran.** The user ruled that day that webqsp trains and metaqa is
read zero-shot. Under that protocol, P learns from webqsp's 7,058 relations, not metaqa's nine, and metaqa's nine are
the unseen ones. That is the question this file's first risk (section 1) says the KG splits cannot ask:
- every metaqa relation has a webqsp relation within cosine 0.60 to 0.90 of it;
- only 1.6% of webqsp's relations have a metaqa relation within 0.6.

So G6r does not run on KG's splits. It runs on the new protocol's base set, by an amendment declared before its
numbers. The code and the self-tests above are this commit's, unchanged.
