# CHN: does a row's complement to a found passage single out the next hop?

Declared 2 November 2026, before any number. Script: `outputs/mp_unified/chndiag.py`. Training-free, host CPU.

## 1. The question

The user's direction of 2 November: musique is the main target, in-domain; no training round until we know why better
pools do not give better-ranked candidates, and then only rounds aimed at that cause. The cause is already narrowed by
three filed diagnostics:

- **D1** (docs/DIAG_BRIDGE.md, REACHABLE_NOT_RANKED): musique's missed golds sit next to found ones (lift about 14 over
  non-golds), but zrm's score puts the missed gold first or second among the found golds' neighbours for only 0.22 to
  0.28 of them (median rank 5 to 9 of about 45). Even anchored on a known gold, filling ranks 4 and 5 with the anchor's
  best-scored neighbours loses R@5.
- **M1** (docs/M1_MUSIQUE_HOPS.md): the first sub-question's paragraph is found (0.70 to 0.84); the chain breaks at the
  bridge; 40% of musique's missed golds sit beyond rank 50 of the pool.
- **The knee** (docs/PKO_DEPTH_KNEE.md, DEPTH_FLAT): reach rises from 0.922 to 0.976 with depth while R@5 stays at
  0.5051. Better pools do not become better rankings because nothing in the scorer picks the next hop out of them.

Every input of today's scorers compares a row with the question, or with the first stage's top rows, one at a time
(SEM, SEED, DISTS). None says how well a row covers what the question asks *beyond* a found passage. CHN asks whether
that pairwise signal is in the frozen vectors at all. If it is, the next screens add it to both models (each in its own
way). If it is not, no feature drawn from these vectors will fix musique's ranking, and we stop looking there.

## 2. What it reads

- The look pools (`outputs/mp_unified/look/<dataset>/<carve>`, the pools D1 and M1 read): pool rows, pool-local edges
  with their families (0 structural, 1 NER, 2 kNN), gold flags, gold totals, the `rrf` and `dense_cos` columns, and the
  question's embedding.
- The frozen package's dense node rows (`lean_mlp3.open_nodes`). Nothing is re-encoded.
- Reads: musique, 2wiki and hotpotqa on s1eval (decide or report, section 4); the same three on s1fit, the first 3,000
  questions (they pick λ, section 3); B1's looks for musique, 2wiki and hotpotqa with the hyperlinks (`b1`), and 2wiki
  and hotpotqa with the title rule (`b1u`); musique's `b1` is its title-rule graph. B1 is reported only and never picks
  anything.
- **Identity check:** on every question, the cosine of the question with each of rrf's top five rows, computed here,
  must equal the look's `dense_cos` within 0.002. If more than 1% of the checked rows differ, the read exits 1 and its
  numbers are not used.

## 3. What it measures

Scores of a candidate row v given an anchor row a (unit vectors, c = q·x):

| score | definition | what it is |
| --- | --- | --- |
| rrf | the look's `rrf` column | the first stage |
| cos | c_v | the dense first stage |
| anc | x_a·x_v | closeness to the anchor |
| res | c_v − c_a (x_a·x_v) | the residual question q − c_a x_a against x_v (up to a factor fixed per anchor) |
| pair | (c_a + c_v) / ‖x_a + x_v‖ | the question against the pair's mean direction |

`res` and `pair` are the complement scores. Both reward a row that answers the part of the question the anchor does not.

**A: next-hop rank, with an oracle anchor (as D1's O1).** Over questions whose rrf top five holds a gold and misses an
in-pool gold. The anchor is the highest-ranked gold in rrf's top five. Each missed gold next to it (any family) is ranked
among the anchor's neighbours outside the top five, by each score. Reported: T1 and T2 (the share ranked first, and
first or second), the median rank, the median number of neighbours, and the same split by the family of the link.

**B: label-free re-ranks**, R@5 and FC@5 over every question with a gold, against rrf's own top five:

- `H1_s`: rrf's top three, then the two best neighbours of rrf's top-1 by score s that are not yet chosen, then rrf's
  order.
- `H2_s`: rrf's top three, the best neighbour of the top-1, then the best neighbour of the top-2, then rrf's order.
- `S_s_λ`: rrf's z-score within the pool, plus λ times the positive part of s's z-score over the rows next to rrf's top
  two. At each row, s is its best over the anchors it is next to. λ comes from (0.25, 0.5, 1, 2). One λ per score,
  picked by the mean R@5 gain over rrf across the three s1fit reads (ties to the smaller λ), then read on s1eval.

s ranges over rrf (adjacency alone, the control), res and pair. musique is also split by hop count (2, 3, 4).

## 4. The rule (on musique's s1eval read)

T2(s) is A's share of adjacent missed golds that score s ranks first or second. The base is max(T2(rrf), T2(cos)).

| verdict | when |
| --- | --- |
| SIGNAL | max(T2(res), T2(pair)) ≥ base + 0.10, and ≥ 0.40 |
| WEAK | max(T2(res), T2(pair)) ≥ base + 0.05, not SIGNAL |
| NO_SIGNAL | otherwise |

Beside the verdict, not part of it: **PARAM_FREE_PAYS** when a label-free form with a complement score (H1 or H2 with res
or pair, or S with the s1fit λ) is at least +0.01 R@5 over rrf on musique's s1eval and no more than 0.0075 under rrf on
2wiki's or hotpotqa's s1eval.

## 5. What follows

- **SIGNAL:** two screens, each declared in its own file before any fit, seed 0 on both WQ splits on PKO's pools,
  under DEP's screen rule, with musique in-domain (WQ-hotpotqa) and its 3- and 4-hop slices as the main reads.
  - The MLP (zrckp): complement columns, `res` and `pair` against the first stage's top rows, split by whether a link
    joins them. These rows are fixed before any row is scored, and the vectors are frozen. By the user's ruling of 2
    November, reading them is not message passing.
  - The GNN (zg6r), built on its own and not on the MLP: each message carries the complement of its sender and receiver
    to the question, against the GNN's own round-one top rows.
- **WEAK:** the MLP screen first (CPU), then the GNN's if the MLP's gains.
- **NO_SIGNAL:** these vectors do not carry the next hop, and no feature drawn from them will fix musique's ranking. The
  musique gap is then the first stage's, and the paper reports it as such. The GPU stays on depth (DEP, RTR).
- **PARAM_FREE_PAYS:** the form, with its s1fit λ, is read once on B1 as a training-free row beside the link rules.

## 6. Risks

- These are the look's pools, not PKO's. They hold at least 0.95 of the golds, and D1 and M1 read the same pools, so A
  compares directly with D1. PKO's pools are deeper but no richer at the top (the knee).
- A uses the gold flags to pick the anchor, as D1's O1 did. It asks whether the signal exists, not whether a label-free
  rule finds it. B is the label-free half.
- B is measured against rrf, not against the models. The models already read similarity to the first stage's top rows
  one at a time (the SEED lines). The complement is a three-way term (question, anchor and row) that none of their
  inputs forms. Whether it adds over the models is the next screens' question.
- Hop order is not used. The anchor is the best-ranked found gold, usually sub-question 1's paragraph (M1: found 0.70
  to 0.84 of the time).
- B1 is read beside the result, never to choose.

## 7. How it runs

- Host CPU items:
  - `chn-selftest`;
  - `chn-<dataset>-<carve>` for the reads (10 after amendment A1, section 8), 1 CPU and 5 GB each;
  - `chn-report`, which writes `outputs/chn/report.{json,md}` with the verdict.
- Commit `outputs/chn/report.*` and the per-read summaries with `git add -f`. The per-question rows in
  `outputs/chn/work` are never committed.
- The self-test (`python outputs/mp_unified/chndiag.py --selftest`) checks the scores against their definitions, A's
  ranks and B's picks on a toy question built so that the next hop has a low cosine and a high residual.
- Disclosure: before this commit, a 20-question probe on musique's s1eval (`--limit 20`, into `outputs/chn/probe` on the
  host) checked the identity (PASS, max 0.00066) and the speed. Its numbers are not used.

## 8. Amendment A1 (2 November, before any read but the probe)

The host's looks were listed before queueing. musique's s1fit, s1sel, b1 and b1u looks have no chunks left on the host
(cleaned). 2wiki's and hotpotqa's s1fit looks do not exist. These looks do have chunks: musique's s1eval and b1w;
2wiki's and hotpotqa's s1eval, s1sel, b1, b1t, b1u and b1w. So:

- **λ is picked on 2wiki's and hotpotqa's s1sel** (all their questions, 1,496 and 1,508). It is one λ per score, by the
  mean R@5 gain over rrf across the two. musique plays no part in choosing it, so the λ is a universal rule rather than a
  musique-tuned one. musique's S rows are reported at every λ beside, so nothing hidden turns on the choice.
- **musique's B1 read is b1w** (the W2_S graph of B1f, read 65.3 / 65.7 by the J5 fits against the title rule's 66.1 /
  66.3). The title rule's own B1 look for musique is gone. B1 stays a report, never a choice.
- Sections 4 and 5 are unchanged.
