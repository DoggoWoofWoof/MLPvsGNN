# LTW: does a learned score of (question, found passage, row) single out the next hop?

Declared 2 November 2026, before any number. Script: `outputs/mp_unified/ltwdiag.py`. Host CPU, two small fits, no
GPU.

## 1. The question

The user's direction of 2 November: musique in-domain is the main target. Train only once the cause is known, and only
in rounds aimed at it. Try whatever can fix it: relations, iterative refinement of the vectors, EM- or ELBO-style
objectives.

Three filed reads narrow the cause:

- **The knee, A3** (docs/PKO_DEPTH_KNEE.md section 11). zg6r reads musique 0.5086 to 0.5091 at every depth while reach
  rises from 0.922 to 0.976; zrckp is the same. The golds the deeper pool adds are never ranked into the top five.
- **CHN** (docs/CHN_CHAIN_COMPLEMENT.md section 9, NO_SIGNAL). With an oracle anchor (a found gold), no fixed score of
  the frozen vectors puts the missed next hop first or second among the anchor's neighbours for more than 0.354 of
  musique's cases (cosine 0.327). The anchor has a median of 32 neighbours on musique, against 8 on the other two.
- **D1** (docs/DIAG_BRIDGE.md, REACHABLE_NOT_RANKED). zrm's own score does no better (0.22 to 0.28).

CHN's scores are fixed functions of three cosines: the row's, the anchor's and the anchor-row cosine. They cannot weigh
*which* directions of the question the found passage leaves unanswered. A learned score can. It reads:
- the question vector;
- the found passage's vector;
- their residual.

From these it forms a next-hop query, which it compares with each candidate's frozen row. This is one step of iterative
refinement of the question vector, learned rather than fixed. CHN's `res` is the fixed step, r = q − c_a x_a.

- **If it singles out the next hop,** both models get a change aimed at the cause (section 7).
- **If it does not,** the next hop is not in the frozen vectors in a form a small scorer can find. The paper then states
  musique's gap as the first stage's.

## 2. What it reads

- **PKO's pools, read only:**
  - `outputs/pko/cache/<dataset>/<carve>/part_*`: n, gold_total, q_emb, node, gold, and from `xc` the five rank columns
    and `dense_cos`. They are located by `lean_cache.xc_columns` and checked by name.
  - `outputs/pko/links/<dataset>/<carve>/part_*`: q_ne, e_u, e_v, e_fam. Links are undirected; the families are 0
    structural, 1 NER, 2 kNN.
  - Each links part must name its cache part's record sha (`step1_record_sha256`) and match its query and row counts,
    or the read exits.
- **These are the pools the models of record are read on** (ML1: zg6r on PKO's pools). CHN read the look's pools. LTW
  re-reads CHN's fixed scores on its own lists, so every comparison is within LTW.
- **Carves** (all on the host, with cache and links):

  | role | musique | 2wiki | hotpotqa |
  | --- | --- | --- | --- |
  | fit (section 5) | s1fit, 4,601 questions | fit, 5,928 | fit, 5,930 |
  | decide and report (section 6) | s1eval, 2,417 | s1eval, 12,576 | s1eval, 7,405 |

- **The frozen package's dense node rows** (`lean_mlp3.open_nodes`). Nothing is re-encoded and the encoder stays
  frozen.
- **Identity check.** For every list, each candidate's and the anchor's cosine with the question are computed here.
  Each must equal the cache's `dense_cos` within 0.002 (CHN's tolerance). If more than 1% of checked rows differ, the
  fit or read exits and its numbers are not used.
- No test split and no B1 carve is read in this stage.

## 3. The lists

- **A (every carve): CHN's A on PKO's pools.**
  - Questions: those whose rrf top five holds a gold and misses an in-pool gold.
  - The anchor is the best-ranked gold in rrf's top five.
  - The candidates are the anchor's neighbours (a link of any family, either direction) outside rrf's top five.
  - The targets are the missed golds among them.
  - A list without a target is counted and not kept.
- **H (s1eval only): CHN's H1, label-free.**
  - Every question with a gold.
  - The anchor is rrf's top-1, and the candidates are all its neighbours.
  - `H1_s` is rrf's top three, then the two best candidates by score s that are not yet chosen, then rrf's order.
  - Measured by R@5 and FC@5 against rrf's own top five.
- rrf's order is descending `rrf`, with ties broken by pool position (CHN's `order_of`).
- musique's questions are split by hop count (2, 3, 4), taken from their ids.

## 4. The scores

Notation: a candidate row v, the question q and the anchor a, all unit vectors; c = q·x.

**Fixed scores** (CHN's, re-read on the same lists):

| score | definition |
| --- | --- |
| rrf | the cache's `rrf` column |
| cos | c_v |
| anc | x_a·x_v |
| res | c_v − c_a (x_a·x_v) |
| pair | (c_a + c_v) / ‖x_a + x_v‖ |

**lin** is a two-layer MLP: 14 → 32 → 32 → 1 with GELU, 1,569 parameters. It reads 14 scalars per candidate:
- cos, anc, res, pair, and the anchor's c_a;
- the cache's five rank columns: `dense_rr`, `splade_rr`, `rrf`, `agreement`, `is_seed`;
- log(1 + v's rank in rrf's order);
- three bits for the families of v's links to a.

The scalars are standardised by the training lists' means and standard deviations.

**bil** is lin plus a learned three-way term:

    s_bil(v) = s_lin(v) + x_v · w,   w = Σ_t U_t V_t^T y_t,   y = (q, x_a, r),   r = (q − c_a x_a) / ‖q − c_a x_a‖

- U_t and V_t are 1536 × 32 matrices, three of each.
- U starts at zero, so bil starts as lin.
- This adds 294,912 parameters (296,481 in all).
- w is a learned next-hop query built from the question, the found passage and the residual. It is the term that no
  input of today's scorers forms; they compare a row with the question, or with the first stage's top rows, one at a
  time.

The two forms ask different things:
- **lin:** is the next hop in a learned mix of the scalars the models already have, or could have?
- **bil:** do the frozen vectors hold more than those scalars?

## 5. How the scores are fit

- **Two fits,** each fitting both forms:
  - `joint`: the A lists of the three fit carves. This is the decision fit: one universal scorer, with no per-dataset
    setting.
  - `musique`: musique s1fit's A lists alone. It is reported beside, to show whether the joint fit gives up musique to
    the other two.
- **Held-out set:** 10% of each carve's questions (seed [0, dataset]). Their lists form the held-out set and the rest
  train.
- **The cap:** fit-carve lists keep every target and at most 256 candidates. Negatives are drawn without replacement
  (seed [0, dataset, 1]). s1eval lists are never capped.
- **Loss:** listwise multi-positive softmax cross-entropy, the negative log of the softmax mass on a list's targets.
  - This is the marginal likelihood that the next hop is one of the targets. The latent choice among them is summed out
    exactly, so no E-step is needed.
  - Each dataset's lists are weighted to carry equal weight.
- **Optimiser:**
  - Adam (learning rate 1e-3, weight decay 1e-4), 32 lists per batch, up to 60 epochs.
  - Early stopping on held-out T2, the mean over the fit's datasets, with patience 6. The model kept is the best
    epoch's.
  - Seed 0, one fit per form.
- **The decision form:** bil if its held-out musique T2 at its best epoch is above lin's, otherwise lin. It is chosen on
  the fit carves' held-out questions only; s1eval chooses nothing.
- CPU with 2 threads. The learned term reads float16 rows at fit and at read alike.

## 6. The rule (on musique's s1eval read)

T2(s) is the share of A's targets that score s ranks first or second among their list's candidates, with ties broken by
pool position.

- The **base** is the largest T2 of the five fixed scores (rrf, cos, anc, res, pair) on the same lists. This is
  stricter than CHN's base, max(rrf, cos).
- The **learned** T2 is that of `joint.<decision form>`.

| verdict | when |
| --- | --- |
| SIGNAL | learned ≥ base + 0.10, and ≥ 0.40 |
| WEAK | learned ≥ base + 0.05, not SIGNAL |
| NO_SIGNAL | otherwise |

Beside the verdict, not part of it: **LEARNED_PAYS** holds when H1 with the decision score is:
- at least +0.01 R@5 over rrf's top five on musique's s1eval;
- and no more than 0.0075 under it on 2wiki's and hotpotqa's s1eval.

Reported, choosing nothing:
- both forms of both fits;
- T1, T2, median rank and MRR, with T2 by link family and by hop;
- H1's R@5 and FC@5 for every score, by hop on musique;
- the held-out T2s and the training curves.

## 7. What follows

- **SIGNAL:** two screens, each declared in its own file before any fit.
  - Common settings: seed 0 on both WQ splits, on PKO's pools, under DEP's screen rule. The main reads are musique
    in-domain (WQ-hotpotqa) and its 3- and 4-hop slices.
  - **The MLP (zrckp)** gets the learned next-hop score of each row against the first stage's top rows, as input
    columns.
    - Those anchors are fixed before any row is scored. By the user's ruling of 2 November, reading their frozen
      vectors is not message passing.
    - The scorer is refit on the MLP's own training carves.
  - **The GNN (zg6r)** is built on its own, not on the MLP. Each message carries the three-way term: the receiver's row
    against the sender's learned next-hop query, formed from the question, the sender's row and their residual. The
    term is trained end to end with the GNN's own weights.
- **WEAK:** the MLP's screen first (CPU), then the GNN's if the MLP gains.
- **NO_SIGNAL:**
  - A scorer that reads the three frozen vectors does not single out musique's next hop at this capacity. The learned
    three-way score of these vectors closes.
  - musique's gap is stated as the first stage's, and the GPU stays on depth (DEP, RTR).
  - The named next form depends on the joint fit's record (`fit_joint.json`):
    - if bil's best epoch is its 60th (the fit stopped at the epoch limit, not on patience), it is the same form at
      rank 128, declared in an amendment before its read;
    - otherwise there is none on these vectors.
- **LEARNED_PAYS:** H1 with the joint decision score is read once on B1, as a row beside the link rules.
  - Its scorer is fit on the fit carves and never on B1.
  - The B1 read is declared in an amendment and chooses nothing.

## 8. Risks

- **A is an oracle.** It uses the gold flags to pick the anchor, as CHN and D1 did, so it asks whether the signal
  exists. H is the label-free half.
- **The pools differ from CHN's.** PKO's pools are not the look's, so CHN's numbers are not compared with LTW's. The
  fixed scores are re-read on LTW's own lists.
- **Rank shortcuts.** The learned score could lean on the rank columns. lin has them too and the base includes rrf, so
  bil must beat both.
- **Capacity and data.** The number of musique training lists is known only after the cases are built; it is reported.
  Early stopping on held-out T2 guards against over-fitting, and the curves are recorded.
- **No leakage.** The fit carves and s1eval are disjoint question sets. The scorer never sees s1eval's labels, and the
  decision form is chosen on the fit carves' held-out questions.
- **One seed.** A SIGNAL leads to screens with their own seeds and nulls; it is not itself a model result.
- **Hop order is not used.** The anchor is the best-ranked found gold, usually sub-question 1's paragraph.
- **Nothing else changes.** No model, no encoder, no LLM and no test split.

## 9. How it runs

- **Host CPU items,** after the CHN block and ahead of DOB's held fits:
  - `ltw-selftest`: 1 CPU, 2 GB;
  - `ltw-cases-<dataset>-<carve>` for the six carves: 1 CPU, 3 GB each, no vectors;
  - `ltw-fit-joint`, behind the three fit carves' cases, and `ltw-fit-musique`, behind musique s1fit's: 2 CPUs and
    10 GB each;
  - `ltw-report`, behind both fits and the three s1eval cases: 2 CPUs, 10 GB.
- **Records** are committed with `git add -f` after their host paths are masked:
  - `outputs/ltw/report.{json,md}`;
  - `fit_<name>.json`;
  - `<dataset>__<carve>.cases.json` and `.read.json`.
- `outputs/ltw/work` (the cases and the weights) is never committed.
- **The self-test** (`python outputs/mp_unified/ltwdiag.py --selftest`) passed on the laptop before this commit. It
  checks:
  - the pair scores against CHN's;
  - the family bits and lists on toy questions, the feature rows and the cap;
  - that bil equals lin at the start, and the parameter counts;
  - the loss against a numpy reference;
  - a planted three-way signal at d = 16, which bil must find (synthetic data only);
  - an end-to-end run on toy PKO trees: cases, both fits and the report.
- **Disclosure:** before this commit, only the PKO carves' sizes were listed on the host (parts, questions, rows, and
  the largest parts' file sizes). No list, score or fit was read.

## 10. Result (filed 2 November 2026)

`ltw-report` ran on the host at 13:45 IST, in 153 seconds, after the six cases and both fits. The identity check passed
on all three reads: 0 bad of 205,977, 815,883 and 306,413 rows checked, with a largest difference of 0.00096 against the
0.002 tolerance. Records: `outputs/ltw/report.{json,md}`, `fit_joint.json`, `fit_musique.json`, and the cases and reads
of every carve. Their host paths are masked.

**SIGNAL.** On musique's s1eval A lists, `joint.bil` puts the missed next hop first or second for 0.530 of 923 targets.
The best fixed score, `pair`, does so for 0.360. The gain is +0.170, against bars of +0.10 and 0.40.

**LEARNED_PAYS** holds. H1 with `joint.bil` against rrf's top five, in R@5:

| read | questions | rrf top 5 | H1 pair | H1 joint.lin | H1 joint.bil | dR@5 (bil) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| musique | 2,417 | 0.4734 | 0.4849 | 0.5111 | 0.5125 | +0.0391 |
| 2wiki | 12,576 | 0.6079 | 0.6094 | 0.7089 | 0.7627 | +0.1548 |
| hotpotqa | 7,405 | 0.6846 | 0.7258 | 0.7770 | 0.7930 | +0.1084 |

A, T2 on each s1eval read (lists / targets: musique 915 / 923, 2wiki 7,619, hotpotqa 2,799):

| read | rrf | cos | anc | res | pair | joint.lin | joint.bil | musique.lin | musique.bil |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| musique | 0.298 | 0.326 | 0.181 | 0.314 | 0.360 | 0.495 | 0.530 | 0.481 | 0.484 |
| 2wiki | 0.213 | 0.220 | 0.175 | 0.111 | 0.179 | 0.610 | 0.860 | 0.451 | 0.424 |
| hotpotqa | 0.399 | 0.372 | 0.176 | 0.420 | 0.446 | 0.723 | 0.804 | 0.653 | 0.616 |

**The fits.** Both stopped on patience, not at the epoch limit.

| fit | form | best epoch | epochs run | held-out T2 (musique / 2wiki / hotpotqa) |
| --- | --- | ---: | ---: | --- |
| joint | lin | 8 | 15 | 0.574 / 0.592 / 0.738 |
| joint | bil | 7 | 14 | 0.835 / 0.887 / 0.805 |
| musique | lin | 3 | 10 | 0.617 |
| musique | bil | 4 | 11 | 0.835 |

**Disclosure.**
- The musique-only fit's record (`fit_musique.json`) was fetched and read before `ltw-report` ran. Its held-out T2s
  were seen: lin 0.617, bil 0.835.
- A check of LTW's own case files followed, also before the read. It used a scratch script and no labels beyond the
  cases':
  - inside s1fit, 67.8% of the held-out lists' targets (115) are training targets of other questions, and 19.1% of
    their (anchor, target) pairs;
  - s1eval shares no target, no pair and no anchor with s1fit (0.0 each).
- So the held-out T2s are inflated by targets the scorer has seen. The s1eval read is the honest test.
  - On s1eval, `musique.bil` falls from 0.835 to 0.484.
  - `joint.bil` reads 0.530: the 2wiki and hotpotqa lists carry over to musique's unseen next hops.

**Where it works and where it does not** (reported, choosing nothing):
- **By hop, H1 on musique** (R@5, rrf top 5 to H1 joint.bil):
  - 2-hop: 0.5619 to 0.6502, +0.088;
  - 3-hop: 0.4368 to 0.4219, -0.015;
  - 4-hop: 0.2685 to 0.2568, -0.012.

  One hop from rrf's first row reaches a 2-hop question's second paragraph. On 3- and 4-hop questions it gives up rrf's
  fourth and fifth rows for a step that is not enough.
- **By hop, A on musique** (T2, pair to joint.bil):
  - 2-hop: 0.392 to 0.596;
  - 3-hop: 0.306 to 0.449;
  - 4-hop: 0.323 to 0.395.

  The scorer still finds the next hop on longer questions, when the anchor is a found gold.
- **By link family, A on musique** (T2, pair to joint.bil):

  | family | share of targets | pair | joint.bil |
  | --- | ---: | ---: | ---: |
  | structural | 0.546 | 0.349 | 0.696 |
  | NER | 0.427 | 0.381 | 0.338 |
  | kNN | 0.027 | 0.240 | 0.200 |

  On NER links the joint scorer is below `pair` (`musique.bil` reads 0.416 there). NER links carry 43% of musique's
  targets but only 5% of 2wiki's and hotpotqa's, so the joint loss is set by structural links.
- **The two forms.** Most of the gain on musique is lin's, a learned mix of the 14 scalars: 0.495 against `pair`'s 0.360.
  bil adds +0.035 on musique, +0.250 on 2wiki and +0.080 on hotpotqa.
- **Against the models of record.** C1R's in-domain reads of the same s1eval questions:

  | dataset | GNN | MLP | H1 joint.bil (from rrf) |
  | --- | ---: | ---: | ---: |
  | musique | 0.5912 | 0.5777 | 0.5125 |
  | hotpotqa | 0.8593 | 0.8423 | 0.7930 |
  | 2wiki | 0.9269 | 0.9045 | 0.7627 |

  H1 is not a model, and it reads below both models on every dataset. Whether the next-hop score adds to a model is
  what section 7's screens ask.

**What follows** (section 7):
- **SIGNAL:** the two screens, each declared in its own file before any fit:
  - the MLP (zrckp), with the learned next-hop score of each row against the first stage's top rows as input columns,
    and the scorer refit on the MLP's own training carves;
  - the GNN (zg6r), with its own three-way message term, trained end to end, and not built on the MLP.

  Each declaration says how it treats the two weak spots found here: NER-link next hops, and questions of three and
  four hops.
- **LEARNED_PAYS:** H1 with `joint.bil` is read once on B1, as a row beside the link rules. That read is declared in an
  amendment and chooses nothing.

## 11. Amendment A1: the store read (2 November, after section 10, before any store number)

**Why.** The models read each row through the store, not the full row. The store is lean_mlp3's int8 codes on the 256
axes of `basis_2wiki`, which both WQ splits' fits read. LTW's term read the full 1536-d rows. The GNN screen (section
7) gives each message the three-way term, and it can read the store's coordinates, which every batch already holds,
or the full rows, which would need a new side key in the carve. One read decides which.

**The form.** `store.bil` is bil with every vector on the store's axes:
- a candidate's or the anchor's row x becomes s · codes(x) + c: lean_mlp3's int8 codes, decoded, the first 256
  entries;
- the question q becomes V'q;
- the residual becomes unit(V'q − c_a x_a'), with c_a the anchor's full cosine and x_a' its store coordinates.

Everything else is section 4's and 5's:
- the 14 scalars, from the full rows;
- the lists (LTW's case files, read only), the seeds, the settings and the joint fit's three fit carves.

`store.lin` is fit too. It reads no vector, so it must repeat `joint.lin`.

**The rule, fixed now.** On the three s1eval A reads:
- K is the mean of (T2 store.bil − T2 joint.lin), divided by the mean of (T2 joint.bil − T2 joint.lin), with the
  joint fit's T2s from section 10. The divisor is +0.122.
- **STORE_KEEPS** when K ≥ 0.5 and musique's `store.bil` T2 is at least its `joint.lin` T2 (0.495).
- **STORE_LOSES** otherwise.

**What follows.**
- **STORE_KEEPS:** the GNN screen's term reads the store's coordinates.
- **STORE_LOSES:** the GNN screen's term reads the full rows, through a side key of the carve.
- The MLP screen's columns are computed from the full rows either way, by the scorer outside the MLP.

**How it runs.**
- One host CPU item, `ltw-store` (2 CPUs, 10 GB): `python outputs/mp_unified/ltwstore.py run --host`. It goes
  behind `ltw-report`, ahead of DOB's held fits.
- Records: `outputs/ltw/store/report.{json,md}`, `fit_store.json` and the three `.read.json` files. Their host paths
  are masked before commit. `outputs/ltw/store/work` is never committed.
- The self-test (`--selftest`) passed on the laptop before this commit. It checks that:
  - the store map equals `lean_mlp3.Store`'s codes, decode and query on a full-size random basis;
  - the residual is as defined, and the scalars are untouched;
  - LTW's functions are restored;
  - a toy end-to-end run gives the records, with `store.lin` repeating `joint.lin`.

**The B1 read** (section 7, LEARNED_PAYS) is not part of this amendment.
- B1's looks hold the same retrieval columns as PKO's caches (`look_x_six.RETRIEVAL`), but in the look's format.
- LTW's case builder reads PKO's parts. The B1 read needs a case builder on looks, as CHN's has.
- It is declared in its own amendment, before it runs, and chooses nothing.
