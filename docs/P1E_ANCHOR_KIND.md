# P1e: what the links land on decides how they are used

Declared 13 October 2026, before any s1sel or s1eval number of the stage. P1's fifth stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), under the user's rule of about 10:40
([P1D_FREE_POOL.md](P1D_FREE_POOL.md) section 1):
- the pool is one fixed rule for all six datasets;
- nothing is fit to golds, and nothing is set per dataset;
- the rule is judged by golds per node, against P1a's bar.

It reads the pools only: no model trains or reads here, and nothing is fit.

## 1. What P1d's records ask

P1d chose RRF5 on s1sel and called NOT_ALL_SIX on s1eval (8d6bdc7). Every P1d rule against P1a's bar on s1eval
(points below and the largest gap), from P1d section 8:

| rule | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| A1st | BELOW (30, -0.316) | BELOW (14, -0.019) | BELOW (30, -0.084) | AT_OR_ABOVE | AT_OR_ABOVE | AT_OR_ABOVE |
| A1both | BELOW (30, -0.335) | BELOW (10, -0.015) | BELOW (21, -0.025) | BELOW (5, -0.011) | BELOW (4, -0.018) | AT_OR_ABOVE |
| ROUTEst | BELOW (4, -0.005) | BELOW (14, -0.019) | BELOW (30, -0.132) | AT_OR_ABOVE | AT_OR_ABOVE | AT_OR_ABOVE |
| ROUTEboth | BELOW (27, -0.068) | BELOW (10, -0.015) | BELOW (30, -0.057) | BELOW (4, -0.010) | BELOW (4, -0.018) | BELOW (1, -0.005) |
| SUM5 | BELOW (30, -0.179) | BELOW (4, -0.003) | BELOW (1, -0.003) | BELOW (6, -0.008) | AT_OR_ABOVE | BELOW (1, -0.003) |
| RRF5 | BELOW (29, -0.114) | AT_OR_ABOVE | AT_OR_ABOVE | BELOW (4, -0.008) | AT_OR_ABOVE | BELOW (3, -0.058) |

Two rules split the six between them:
- **ROUTEst holds the KBs.** It is AT_OR_ABOVE on webqsp and 4 points short on metaqa, where RRF5 is 29 points
  short. On a KB the corroborated links name the question's entities, and the walk from them must lead.
- **RRF5 holds the passage corpora, except hotpotqa.** It is AT_OR_ABOVE on squad, musique and 2wiki. ROUTEst loses
  musique by 30 points: there a linked passage's walk floods the paragraphs that share its title (P1d section 8), so
  the lists must lead.
- **hotpotqa sits between.** A1 on the structural family leads it from N = 50 to 100, and RRF5 trails A1 there by up
  to 0.008.

So what the links land on decides how they should be used. A question's anchors are either entity records (KB
nodes whose text is only a name) or passages. That is a per-question fact, read from the node records, the same way
on every dataset. P1e branches on it. On the passage side it also tries one fixed change aimed at hotpotqa and at
musique's flood: walks steered by the first stage's lists.

## 2. What this stage can and cannot claim

- **The candidates were shaped on the fit carves.** We compared the kind test, the cut, the steer, and the sources
  the passage fusion holds on the models' training questions. These were the first 600 questions of each training
  dataset's fit carve (musique's s1fit), against P1a's bar recomputed there. The choice and the call never read the
  fit carves. The five rules there, as points more than 0.002 below the bar (15 sizes N, two measures) with the
  largest gap in brackets:

  | rule | metaqa | squad | musique | hotpotqa | 2wiki | points |
  | --- | --- | --- | --- | --- | --- | ---: |
  | RRF5 | 2 (-0.067) | 2 (-0.005) | 0 | 2 (-0.017) | 0 | 6 |
  | KINDcut | 0 | 2 (-0.005) | 0 | 2 (-0.017) | 0 | 4 |
  | KIND_T | 0 | 2 (-0.005) | 0 | 2 (-0.013) | 0 | 4 |
  | KINDcut_T | 0 | 2 (-0.005) | 0 | 2 (-0.013) | 0 | 4 |
  | KINDcut_7T | 0 | 4 (-0.008) | 0 | 1 (-0.003) | 0 | 5 |

  **metaqa's first 600 fit questions are easy.** The bar holds every gold for 0.977 of them at N = 25, against
  0.546 on s1eval. So the fit evidence on the entity side is thin.
- **P1e was designed after P1d's call read s1eval,** so P1e's call is a second read of the same carve. The choice is
  still made on s1sel alone, by P1d's procedure, unchanged.
- **webqsp has no fit carve.** Only label-free counts were read on its s1eval questions; no gold was read:
  - of 1,503 questions, 1,453 (96.7%) are entity questions and 50 are passage questions;
  - 440 are on ROUTE's K branch, all of them entity questions;
  - 69 have no corroborated link, and 19 of these are entity questions by their seeds.
- **On these six datasets the kind acts almost as a KB switch.** The share of name-only nodes:
  - metaqa: 43,234 of 43,234, so every node;
  - webqsp: 1,004,809 of 2,592,894 (38.8%; its CVT mediators are not name-only);
  - hotpotqa: 114,190 of 5,233,329 (2.2%), and 2wiki: 88,614 of 5,989,847 (1.5%), mostly list and stub titles;
  - musique and squad: none.

  On the fit carves every metaqa question is an entity question, and no question of the four passage corpora is.
  The test reads no dataset name and would apply unchanged to a new dataset. On these six, though, it separates the
  two KBs from the four passage corpora.
- **The entity branch keeps P1d's ROUTE condition.** Off the route, an entity question falls back to A1(st), as in
  P1d. P1d's ROUTEst was 4 points below on metaqa's s1eval, at N = 25, 150, 200 and 300, with gaps of -0.002 to
  -0.0045. The cut changes the order only past the nodes K keeps, so it may leave those points where they are.

## 3. The sources

P1d's five are imported unchanged: A1(st), A1(all), K(st), K(all) and R0 (P1D_FREE_POOL.md section 3). P1e adds
five more, built on one steer.

**The steer.** Step 4e's walk splits a node's pushed mass equally over its non-empty families. Within a family it
splits the mass equally over the node's neighbours. The steered walk keeps the family split. Within a family it gives
each neighbour v a share in proportion to

    w(v) = 1/(60 + rank_dense(v)) + 1/(60 + rank_SPLADE(v))

This is the first stage's RRF weight. It uses the same dense and SPLADE top-1,000 lists that A1's restart and P1a's
RRF read.
- **A node a list does not hold** counts at rank depth + 1 in that list. A node in neither list gets the floor
  1/(61 + depth_dense) + 1/(61 + depth_SPLADE), so every neighbour keeps a share.
- **No new constant.** The constant is the first stage's (60). The restart probability (0.5), the push threshold
  (1e-7) and the top 7,000 are step 4e's.
- **With every w equal, the steered walk is step 4e's walk, bit for bit** (checked, section 5).

The five new sources:
- **KT(g):** K's restart (uniform over L*) walked on g with the steer.
- **A1T(g):** A1's restart walked on g with the steer.
- **KCUT(st):** K(st)'s nodes whose mass (scaled to sum to 1) is at least 1/n, where n is the structural graph's
  node count. These are the nodes the walk holds above the level an uninformed walker would give every node. The
  level is the graph's own, not a chosen threshold.

## 4. The kind, the branches and the five rules, fixed here

- **A name-only node** has stripped text that is non-empty and equals one of its names: its title, display name or
  disambiguated display name. Its text says nothing beyond its name. It is a KB entity, or a list or stub title in a
  passage corpus.
- **A question's kind.** The question starts from L*, its corroborated links, or from its seeds (the dense and SPLADE
  top five) when L* is empty. It is an entity question when every node it starts from is name-only, and a passage
  question otherwise.
- **ROUTE's condition** is P1d's: the question has linked nodes, and L* holds every one of them.
- **ROUTEcut:** on the route, KCUT(st), then A1(st)'s other nodes in A1(st)'s order; A1(st) otherwise. P1d's ROUTEst
  is the same with K(st) uncut.
- **RRF5T:** RRF over A1(st), A1(all), KT(st), KT(all) and R0. This is P1d's RRF5 with K steered.
- **RRF7T:** RRF over A1(st), A1(all), A1T(st), A1T(all), KT(st), KT(all) and R0. This is RRF5T with the steered A1
  walks added.

| rule | entity question | passage question |
| --- | --- | --- |
| RRF5 | RRF5 | RRF5 (P1d's chosen rule, unchanged) |
| KINDcut | ROUTEcut | RRF5 |
| KIND_T | ROUTEst | RRF5T |
| KINDcut_T | ROUTEcut | RRF5T |
| KINDcut_7T | ROUTEcut | RRF7T |

RRF and the order's ties are P1d's (section 4): ranks start at 1, the constant is 60, an empty source is skipped, ties
go by node position, and every order is cut at 7,000.

Each pair of rules isolates one change:
- RRF5 to KINDcut: the kind branch;
- KIND_T to KINDcut_T: the cut;
- KINDcut to KINDcut_T: the steered K;
- KINDcut_T to KINDcut_7T: the steered A1.

**Why these, from the fit carves** (ALL, the share of questions with every gold in the pool):
- **The cut.** K's walk from a few entities spreads its last mass thinly over the whole KB. Past the uniform level
  its order is noise, and A1(st) fills the rest better. On metaqa's fit carve, ROUTEcut reaches 1.000 at N = 1,000,
  where ROUTEst stays at 0.997.
- **Steering K.** The steer sends a linked passage's mass to the neighbours the first stage also ranks, not to every
  paragraph sharing its title. On musique's fit carve, K(st) holds every gold for 0.418 of questions at N = 25
  steered, against 0.173 unsteered. RRF5T holds 0.728 there, against RRF5's 0.680 and the bar's 0.610.
- **Steering A1** helps where the lists are strong and hurts where they are weak. At N = 25 on hotpotqa's fit
  carve, A1T(st) holds 0.857 against A1(st)'s 0.812. At N = 50 on 2wiki's, it holds 0.773 against 0.812.
  - **Added beside the unsteered walks (RRF7T),** it brings hotpotqa to one point at -0.003, from RRF5T's two at
    -0.013.
  - **It costs squad** (four points at -0.008, against two at -0.005).
  - **It costs 2wiki's golds per node:** 0.917 at |I_q|, against RRF5T's 0.943 (the bar is 0.888).

  Both passage branches are offered, and the choice decides between them.
- **Looked at on the fit carves and not offered:**
  - steering the entity branch's backfill (ROUTEcut over A1T(st)), which gained nothing on metaqa;
  - RRF over the steered walks alone (A1T, KT and R0);
  - RRF over A1T with the unsteered K.

## 5. The choice and the call, unchanged from P1d

**The checks come first.** Any failure stops the stage.
- P1d's ten orders, recomputed here from P1d's script imported unchanged, put every gold at the position P1d filed.
  This holds question for question, on every carve.
- With every weight equal, the steered walk is step 4e's walk on K's and A1's restarts, on both unions (the
  self-test and the development runs).
- The verdict refuses:
  - a part whose RRF5 differs from P1d's filed RRF5;
  - parts from different scripts or freezes, and a part that imported another P1d script;
  - a part run with `--limit`;
  - parts whose carves or questions differ from P1d's.

**The choice, on s1sel.** One rule is chosen among the five, on the five training datasets' s1sel carves, by P1d's
order:
1. the fewest points more than 0.002 below P1a's bar;
2. then the smaller largest gap;
3. then the higher mean ALL at |I_q|;
4. then the table's order.

RRF5 heads the table, so a tie keeps P1d's rule. webqsp has no s1sel carve, so the choice is zero-shot there.

**The call, on s1eval.** The chosen rule is compared with P1a's bar on each dataset, as in P1d section 5. A dataset
is AT_OR_ABOVE or BELOW, and the stage is ALL_SIX or NOT_ALL_SIX.

**Reported beside, deciding nothing:**
- every rule and every new order on both carves, with the like-for-like bar;
- at |I_q| and |P4e|: today's pools, every rule (P1d's RRF5 among them), the best SOTA order, and the learned gates
  MX, RC and MXU;
- golds per node for every rule;
- the shares of entity questions, passage questions and route questions, and KCUT's mean length.

## 6. What follows

- **ALL_SIX:** the chosen rule is the pool. The steps that follow:
  1. P2c part 2 reads it with the LLM-free family added to the all graph, in its own file;
  2. P1c is re-declared on it, and the models are retrained on its top |I_q|;
  3. node-space EM and the learned offsets follow, as rerankers.
- **NOT_ALL_SIX:** the chosen rule is still the parameter-free pool.
  - The claim that our pools are at or above the SOTA pools on all six is not made, and the record names where.
  - P2c part 2 is read under the chosen rule.
  - Any further pool change, a source or a rule with nothing fit, is declared in its own file.

Flagged, as for every pool so far: the walks are propagation over the graph. The pool is built before either model
runs, and both models of a family read the same pool, so the MLP's own scoring stays free of message passing. The
steer and the kind read only the first stage's lists and the node records, which both models already have. They
read no gold.

## 7. How it runs

- **Code:** `scripts/p1e_anchor_kind.py`. It imports P1d's script unchanged, and through it:
  - P1b's walk with masses;
  - P2b's unions over P2a's family stores;
  - P1a's restarts, curves and matched sizes;
  - step 4e's carve runner.

  It adds the steered walk (step 4e's push with the within-family split weighted), the name-only test, and the
  orders above.
- **The self-test** covers:
  - the name-only test on toy records, and a question's kind from L* or from its seeds;
  - the cut at the uniform level;
  - on a toy graph, the steered walk with every weight 1 equals step 4e's walk, for a uniform and a weighted
    restart, and an empty restart gives an empty walk;
  - the steer's weights, checked by hand; steered, the walk moves a listed node up and keeps the same node set;
  - P1d's sources and checks on toy questions, and the four steered sources;
  - the orders and branches on toy sources: KCUT and ROUTEcut; KIND_T's entity branch is ROUTEst; the passage
    branches are RRF5, RRF5T and RRF7T, with RRF7T's order checked by hand; off the route, A1(st);
  - the verdict: the choice's order and tie-breaks, the call at the tie and just past it, and the refusals (an RRF5
    that differs from P1d's, a `--limit` part, a part that imported another P1d script);
  - no order is listed twice.
- **Development runs, no number.** `--limit Q` runs the stage on the first Q questions of each carve and writes
  nothing. It logs only that every step ran and that the checks held. Before this file was committed, the
  self-test passed. `--limit 64` runs on musique, metaqa, squad and webqsp, on the laptop, each logged on every carve
  that:
  - every step ran;
  - P1d's orders put every gold at P1d's filed positions;
  - P1d's sources equal step 4e's walks;
  - the steered walk with every weight 1 equals step 4e's walk on K's and A1's restarts, on both unions.
- **Host jobs** (`p1e-pools-<dataset>`, light lane). Each runs P1d's four walks per question, four steered walks and
  one pass over the node records. They are sized from P1d's measured peaks and mean CPU use (rx status). They add
  room for the steered walk's extra array (one float per node per thread):

  | dataset | P1d's peak | P1d's mean CPUs | CPUs | memory |
  | --- | ---: | ---: | ---: | ---: |
  | 2wiki | 5.3 GB | 4.9 | 8 | 7 GB |
  | hotpotqa | 4.1 GB | 3.8 | 6 | 5 GB |
  | metaqa | 2.8 GB | 2.0 | 3 | 4 GB |
  | webqsp | 2.3 GB | 2.0 | 3 | 3 GB |
  | musique | 1.7 GB | 1.9 | 2 | 3 GB |
  | squad | 1.3 GB | 1.8 | 2 | 2 GB |

  Each writes `outputs/p1e/pools_<dataset>.json` and each carve's gold positions under every order in
  `pos_<dataset>__<carve>.npz`, well under a megabyte each. Then `verdict`, run locally, writes
  `outputs/p1e/verdict.{json,md}`.
- **ETA:** about twice P1d's walks. 2wiki is the longest, at about 45 minutes on 8 CPUs, so the call should land
  about an hour after the items start.
- The encoder stays frozen. No model trains or reads, nothing is fit, and test splits are never read.

## 8. Results (filed 13 October 2026)

The six host items ran from 14:02 to 14:35. The verdict ran locally at 14:36 and wrote `outputs/p1e/verdict.json` and
`verdict.md`, with each dataset's curves, shares and gold positions beside them in `outputs/p1e/`. The position files
are larger than section 7 said: metaqa's s1eval file is 3.9 MB, and the folder holds 9.2 MB.

**The checks held on all eleven carves.** P1d's ten orders, recomputed here, put every gold at the position P1d
filed, question for question. The development runs had checked the steered walk against step 4e's (section 7).

### The choice (s1sel): KIND_T

| rule | points below the bar | largest gap | mean ALL at \|I_q\| | metaqa | squad | musique | hotpotqa | 2wiki |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| RRF5 | 38 | -0.112 | 0.938 | BELOW (29) | BELOW (2) | AT_OR_ABOVE | BELOW (7) | AT_OR_ABOVE |
| KINDcut | 27 | -0.045 | 0.942 | BELOW (18) | BELOW (2) | AT_OR_ABOVE | BELOW (7) | AT_OR_ABOVE |
| **KIND_T** | **14** | -0.017 | **0.946** | BELOW (7) | BELOW (2) | AT_OR_ABOVE | BELOW (5) | AT_OR_ABOVE |
| KINDcut_T | 25 | -0.045 | 0.941 | BELOW (18) | BELOW (2) | AT_OR_ABOVE | BELOW (5) | AT_OR_ABOVE |
| KINDcut_7T | 28 | -0.045 | 0.938 | BELOW (18) | BELOW (8) | AT_OR_ABOVE | BELOW (2) | AT_OR_ABOVE |

### The call (s1eval): NOT_ALL_SIX

KIND_T is AT_OR_ABOVE on squad, musique, 2wiki and webqsp, and BELOW on two datasets:
- **metaqa:** 4 points, all against LINKED: ALL at N = 25 (-0.004), 150 (-0.002), 200 (-0.004) and 300 (-0.005).
  They are ROUTEst's four points in P1d; KIND_T's entity branch is ROUTEst.
- **hotpotqa:** 3 points against A1 on the structural family: ALL at N = 50 (-0.005) and N = 100 (-0.007), and gold
  recall at N = 100 (-0.003).

Against P1d's RRF5, metaqa goes from 29 points (-0.114) to 4 (-0.005), and webqsp from 3 points (-0.058) to none.
hotpotqa goes from 4 points (-0.008) to 3 (-0.007).

Every rule against the bar on s1eval (points below and the largest gap), deciding nothing:

| rule | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| RRF5 | BELOW (29, -0.114) | AT_OR_ABOVE | AT_OR_ABOVE | BELOW (4, -0.008) | AT_OR_ABOVE | BELOW (3, -0.058) |
| KINDcut | BELOW (19, -0.048) | AT_OR_ABOVE | AT_OR_ABOVE | BELOW (4, -0.008) | AT_OR_ABOVE | AT_OR_ABOVE |
| **KIND_T** | BELOW (4, -0.004) | AT_OR_ABOVE | AT_OR_ABOVE | BELOW (3, -0.007) | AT_OR_ABOVE | AT_OR_ABOVE |
| KINDcut_T | BELOW (19, -0.048) | AT_OR_ABOVE | AT_OR_ABOVE | BELOW (3, -0.007) | AT_OR_ABOVE | AT_OR_ABOVE |
| KINDcut_7T | BELOW (19, -0.048) | BELOW (8, -0.004) | AT_OR_ABOVE | BELOW (1, -0.004) | AT_OR_ABOVE | AT_OR_ABOVE |

The cut costs metaqa's tail: every rule with ROUTEcut on the entity side stops at 0.944 at N = 7,000, against
KIND_T's 0.995 and LINKED's 0.992.

### At the models' sizes (ALL, s1eval)

At each question's size of today's pools, |I_q|:

| dataset (mean rows) | today's pools | RRF5 | KIND_T | the best SOTA order | MX (learned) | RC (learned) | MXU (learned) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| metaqa (2,018) | 0.895 | 0.910 | **0.962** | LINKED 0.961 | 0.923 | 0.963 | 0.920 |
| squad (50) | 0.980 | 0.979 | 0.979 | A1 0.981 | 0.981 | 0.981 | 0.978 |
| musique (2,092) | 0.803 | 0.899 | **0.900** | A2 0.883 | 0.892 | 0.844 | 0.887 |
| hotpotqa (94) | 0.968 | 0.963 | 0.965 | A1 0.973 | 0.969 | 0.971 | 0.969 |
| 2wiki (106) | 0.906 | 0.961 | **0.960** | A1 0.911 | 0.914 | 0.913 | 0.917 |
| webqsp (2,120) | 0.688 | 0.876 | **0.832** | A1 0.790 | 0.806 | 0.782 | 0.857 |

At step 4e's pool sizes, |P4e|:

| dataset (mean rows) | today's pools | RRF5 | KIND_T | the best SOTA order | MX (learned) | RC (learned) | MXU (learned) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| metaqa (7,082) | 0.953 | 0.971 | 0.995 | LINKED 0.992 | 0.979 | 0.994 | 0.976 |
| squad (150) | 0.992 | 0.991 | 0.991 | A1 0.992 | 0.992 | 0.992 | 0.991 |
| musique (6,899) | 0.937 | 0.944 | 0.941 | A2 0.939 | 0.947 | 0.905 | 0.945 |
| hotpotqa (188) | 0.979 | 0.981 | 0.982 | A1 0.982 | 0.981 | 0.980 | 0.980 |
| 2wiki (221) | 0.945 | 0.970 | 0.970 | A1 0.926 | 0.928 | 0.927 | 0.932 |
| webqsp (7,213) | 0.909 | 0.925 | 0.882 | A1 0.862 | 0.890 | 0.864 | 0.927 |

- **Against today's pools at |I_q|,** KIND_T gains on four datasets: metaqa +0.067, musique +0.097, 2wiki +0.054
  and webqsp +0.144. It is level on squad (-0.001) and hotpotqa (-0.003).
- **Against RRF5,** it gains on metaqa (+0.052 at |I_q|, +0.024 at |P4e|) and loses webqsp's tail (-0.044 at
  |I_q|, -0.043 at |P4e|). At |P4e| on webqsp it is below today's pools too (0.882 against 0.909). On the other
  four it is within 0.003 of RRF5.
- **Against the learned gates,** KIND_T, with nothing fit, is above all three on musique and 2wiki at |I_q|. It is
  level with RC on metaqa (0.962 against 0.963) and above MX and MXU there. On webqsp it is above MX and RC and below
  MXU (0.857). On hotpotqa it is below all three, by 0.004 to 0.006.

**Golds per node:** the smallest N at which an order reaches today's pools' ALL at |I_q|.

| dataset | today's rows | RRF5 | KIND_T | the bar |
| --- | ---: | ---: | ---: | ---: |
| metaqa | 2,018 | 2,000 | **750** | 750 |
| squad | 50 | 50 | 50 | 50 |
| musique | 2,092 | 500 | **500** | 750 |
| hotpotqa | 94 | 150 | 150 | 100 |
| 2wiki | 106 | 50 | **50** | 100 |
| webqsp | 2,120 | 500 | **500** | 1,000 |

So KIND_T holds today's golds in about 37% of the nodes on metaqa, a quarter on musique and webqsp, and half on
2wiki. On hotpotqa it needs 150 rows where today's pools use 94.

**The kind and the branch** (s1eval). The name-only test read every node record. All of metaqa's 43,234 nodes are
name-only, as are 1,004,809 of webqsp's 2,592,894, 114,190 of hotpotqa's 5,233,329 and 88,614 of 2wiki's 5,989,847.
squad and musique have none.
- metaqa: every question is an entity question, and 0.949 take the K branch.
- webqsp: 0.967 are entity questions, and 0.293 take the K branch.
- squad, musique, hotpotqa and 2wiki: every question is a passage question, so none takes the K branch.

**Against the like-for-like bar** (beside), KIND_T is BELOW on metaqa and hotpotqa as above. It is also BELOW on
webqsp, at all 30 points: A1 on the structural family leads there, and the largest gap is -0.051 (gold recall at
N = 25).

### Where KIND_T falls short (diagnostics, deciding nothing)

- **metaqa: the questions off the route.** 499 questions (0.051) have no corroborated link, so KIND_T follows A1(st)
  on them. LINKED, the walk from all their linked entities, holds every gold more often there: 0.371 against 0.283
  at N = 25, and 0.709 against 0.633 at N = 300. These questions make the four points.
- **webqsp: the K branch.** On 484 questions (0.322), KIND_T's gold positions differ from A1(st)'s. Most are the K
  branch (0.293); the rest are passage questions (0.033). On them, A1(st) holds every gold for 0.355 at N = 25,
  against KIND_T's 0.227, and for 0.909 at N = 7,000, against 0.829. On the other 1,019 questions the two orders
  agree. So the walk from the corroborated links is right on metaqa and behind the first stage's walk on webqsp. That
  is the like-for-like gap and the shortfall at |P4e|.
- **The two KBs pull the entity branch in opposite directions.** metaqa wants the walk from its links on every
  question; webqsp wants the first stage's walk on a third of them.
- **hotpotqa: A1's middle.** KIND_T leads A1(st) at N = 25 (0.885 against 0.830) and trails it at N = 50 (0.935
  against 0.940) and N = 100 (0.966 against 0.973). KINDcut_7T, with the two steered A1 walks, is one point short on
  hotpotqa. It loses squad (8 points) and 2wiki's head (0.839 against KIND_T's 0.880 at N = 25).

### What follows

By section 6, NOT_ALL_SIX:
- **KIND_T is the parameter-free pool.** The claim that our pools are at or above the SOTA pools on all six is not
  made. Two datasets fall short: metaqa (4 points, at most -0.005, against LINKED) and hotpotqa (3 points, at most
  -0.007, against A1). Like-for-like, webqsp falls short too.
- **P2c part 2 is read under KIND_T,** in its own file: our LLM-free triples as a fourth link family.
- **Any further pool change, with nothing fit, is declared in its own file.** The records point at three places:
  - metaqa's questions off the route;
  - webqsp's K branch;
  - hotpotqa from N = 50 to 100.
- The learned gates stay beside and are never the pool. P1c stays held, and the models are retrained once the pool
  is final.
