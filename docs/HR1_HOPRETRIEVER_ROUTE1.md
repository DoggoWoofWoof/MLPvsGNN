# HR1: HopRetriever's knowledge fusion from vectors we already have (the after-paper plan's item 4b, route 1)

Declared 17 November 2026, before any HR1 number. Code: `outputs/mp_unified/zhr1.py`. HR1 writes only under `outputs/hr1`.
It trains nothing and changes no fit, graph, pool, link, encoder, cache or feature. It edits no other script. It is a
graph-level scouting stage: it checks a new input on the training questions before any refit would carry it, as UL1
checked links against the oracle before D83's one refit (the user, 16 Nov: verify first, then train once). Status: filed 17 Nov (section 10), NO_SIGNAL on both families.

## 1. The question

- **HopRetriever's eq. 5** (Li et al., AAAI 2021, arXiv 2012.15534, p. 4, read from the PDF on 17 Nov). A hop from
  passage i to entity j has two encodings:
  - m_ij, the mention: BERT's output at an entity marker around j's anchor text in d_i, read with the question;
  - u_j, the document: BERT's [CLS] output on j's own text with the question.

  They are fused with the retrieval history h: a_m = h W_k m_ij, a_u = h W_k u_j, {w_m, w_u} = softmax(a_m, a_u), and
  hop_ij = w_m W_v m_ij + w_u W_v u_j. The hop is scored as sigmoid(h_t · hop_ij) (eq. 6). When j is not mentioned in
  d_i, a trained vector m_P stands in for m_ij.
- **Route 1** uses only vectors we already have: the frozen encoder's passage vectors and the question vector. There is no
  query-conditioned mention encoding. The stand-in for "how the corpus mentions j" is j's **entity vector**: the
  normalised sum of the unit vectors of the passages linked to j. W_k = W_v = I, and h is the question vector.
- **Why corpus level.** m_j depends only on the corpus and its link rules. It is fixed at index time, before any
  question, and is read at query time as one more cosine, like dense_cos. So the MLP may read it (no edge is read while
  rows are scored). The same mean over a pool's own links would be message passing: the GNN already does that, and the
  MLP may not.
- **What exists already.**
  - lean_mlp's NBR block (pool-level structural edges, the 128-wide projection) holds "the cosine of the neighbour mean".
    The cost-aware selection left it out. The record MLP's nine blocks (rank, dense_cos, topo_STRUCT, depth_STRUCT, SEMB,
    SEED, WALK, DISTS, WALKF) hold no neighbour-mean cosine.
  - The GNN's hidden-state message passing aggregates pool neighbours.
  - So HR1 adds three things: neighbours outside the pool, the full 1536 width, and eq. 5's fusion.

## 2. What it reads

- **The questions.** D83F's training carves of the four passage datasets: 2wiki `fit`, hotpotqa `fit`, musique `s1fit`
  and squad `fit` (`outputs/abl/D83F/cache/<d>/<carve>`, lean_cache.part_dirs).
  - Per row: the pool node (package row), the gold flag and the cache's dense_cos column.
  - Per question: q_emb (float16) and gold_total.
  - No s1eval or B1 carve is read, so the reads that would judge a refit stay unseen.
- **The links.** zd91.Stores("D83F", d) rebuilds D83's stores over the package, and its own gate (D91's G2d) refuses
  unless they equal D83's labelled key sets:
  - family `ment`: the slot store, R.mention (W2_S; stored symmetric, so in- and out-links are one set);
  - family `co`: the NER store, R.name_comention plus R.title_comention (F0T).

  K.knn is left out: a mean of j's nearest neighbours in vector space smooths x_j itself and carries no relation.
- **The vectors.** The frozen package's dense node rows (lean_mlp3.open_nodes; its RECORD_SHA256 is filed). The encoder
  rule is unchanged.
- **Not covered:** the KBs (metaqa, webqsp). They have no R.mention. Section 7 names an S-based analog.

## 3. The forms (parameter-free; no label is read)

For each pool row j of question q, with q̂ the unit question vector and x̂ the unit node vectors:

- **s_u** = q̂ · x̂_j (the plain cosine; G2 checks it against the cache's dense_cos).
- **m_j^f** = Σ_{i ∈ N_f(j)} x̂_i, the sum over j's neighbours in family f's package-level store (each neighbour once, j
  itself never), with deg_f(j) = |N_f(j)|. The mean has the same direction, so no division is needed.
- **s_m^f** = q̂ · m_j^f / |m_j^f|, undefined when deg_f(j) = 0 (or the sum is the zero vector).
- **F^f**, eq. 5 with W_k = W_v = I and h = q̂: a_m = s_m^f, a_u = s_u, {w_m, w_u} = softmax(a_m, a_u), and the hop's
  score h · hop = w_m s_m^f + w_u s_u. Where s_m^f is undefined, F^f = s_u (the parameter-free stand-in for m_P).

## 4. Measures

Per question with at least one gold and one non-gold row in its pool:

- **A^f (deciding).** The matched win rate of s_m^f.
  - Each in-pool gold g is paired with the nearest non-gold row above it and the nearest non-gold row below it in the
    s_u order (descending, ties by pool position).
  - A pair counts when both rows have deg_f > 0 and a defined s_m^f. It scores 1 when s_m^f(g) > s_m^f(v), 0.5 on a tie
    and 0 otherwise.
  - The question's value is the mean over its counted pairs. A question with no counted pair is left out and counted.
  - So A^f asks: does the entity vector separate golds from the non-golds that the plain cosine ranks right next to them?
    0.5 is no information.
- **Beside, deciding nothing:**
  - A_miss^f: A^f over the golds outside s_u's top five only (the next-hop golds the question does not describe);
  - coverage: the share of pool rows and of golds with deg_f > 0, and the median degree of golds and non-golds;
  - per-question AUC (mid-rank ties) of s_u, twin0 (cache score2), F^ment and F^co, and the paired differences F^f − s_u;
  - R@5 (golds in the top five over gold_total) of s_u, F^ment and F^co within the pool;
  - A^f split by the gold's deg_f tercile.
- **Intervals.** zd83rep.interval: the paired bootstrap over questions, 2,000 resamples, seed 0 (A^f against 0.5, F^f
  against s_u).

## 5. Gates

- **G1, the stores.** zd91.Stores' own check: the rebuilt R.mention and R.co stores must equal D83's key sets, or it exits.
- **G2, alignment.** The parts must tile the carve (lean_cache.part_dirs). s_u recomputed from q_emb and the node rows
  must equal the cache's dense_cos column within 2e-3 on every row (the largest difference is filed). Otherwise it exits 1.
- **G3, the columns.** Each columns file holds one value per cache row, in the parts' order. The report checks the row
  counts and the cache records' sha256 named in the columns record.

## 6. The rule

- **One dataset's call** for family f: A^f − 0.5 with its interval. ABOVE when the lower end is above 0, BELOW when the
  upper end is below 0, LEVEL otherwise.
- **A dataset carries f** when the call is ABOVE, A^f is at least 0.55, and the questions counted are at least half of the
  questions with an in-pool gold and non-gold.
- **Each family's verdict:**
  - CARRIED: at least three of the four datasets carry it and none is BELOW;
  - NO_SIGNAL: otherwise.
- The 0.55 bar and the two families are fixed here, before any number.

## 7. What follows

- **CARRIED (either family).** This raises decision **D96** for the user (default: yes). It would add s_m^f and F^f as
  two input columns, for the MLP and the GNN alike, in the one next refit, not in a refit of their own. Before that refit:
  - the same columns for the s1eval and B1 carves (the same code, declared in the refit's file);
  - a KB analog (m_j over j's S neighbours, the triples' other ends);
  - the cold cost: m̂_j is precomputed at index time, so a query pays one more 1536-wide gather and dot per row.
- **NO_SIGNAL.** Route 1 closes for these forms; the verdict closes only them. Named, not declared: route 2, the mention's
  own sentence through the frozen encoder (HopRetriever's m_ij without BERT). It needs new encoder calls, which is the
  user's call under the encoder rule.
- **Either way,** the coverage and degree tables go to the journal's analysis beside AP4's precision ledger.

## 8. Runs and order

- **`zhr1.py selftest`** must pass before any row is queued:
  - on the laptop, on synthetic stores, vectors and pools: the entity sums, s_u, s_m, F, the matched pairs, the AUC, R@5
    and the rule against brute-force references, including ties, isolated nodes and pools with no non-gold;
  - on the host, `selftest --host`: G1 and G2 on squad's first cache part, printing no measure.
- **CPU rows** `hr1-cols-<d>`: `zhr1.py cols --ds <d>` writes `outputs/hr1/cols/<d>/` (s_u, s_m and degree per family,
  float32 and int32, one value per cache row) and `record.json`. Squad and musique go first (smallest), then hotpotqa
  and 2wiki.
- **`hr1-rep`** after the four: `zhr1.py report` writes `outputs/hr1/report.json` and `report.md` with section 4 and the
  calls. It runs on the host, which holds the cache.
- **Cost.** One stream of the package's node rows per entity-row group (2wiki and hotpotqa: about 16-18 GB of float16
  per pass), and one sparse product per family. CPU only; it runs beside another host user's GPU job at below-normal
  priority on the E-cores (gpu_guard).

## 9. Risks

- **Hubs.** W2_S's largest degree is 371,977 on HotpotQA and 621,403 on 2Wiki. A hub's entity vector is close to the
  corpus mean, so its s_m says little. The degree split shows it.
- **Symmetric links.** W2_S stores each pair both ways, so "the passages that mention j" also takes in the passages j
  mentions.
- **Float16 question vectors.** G2 bounds the error against the cache's own cosine.
- **Not the models' margin.** A^f measures information beyond the plain cosine, not beyond the record models' other
  inputs (WALK, DISTS, SEED). Only a refit measures that, which is why a carried family goes into the one next refit and
  not into a verdict on the models.

## 10. Results, filed 17 November 2026

**Provenance.**
- `hr1-rep` ran on the host with `zhr1.py report`: sha256 e29184f187189c45 (the file of f9b98ae), written
  2026-11-17T14:59:57Z.
- The four `hr1-cols-<d>` rows ran first (4 CPUs each; 2wiki 17 min at 14 GB, hotpotqa 11 min).
- **Gates.** G1 passed on every dataset (the rows ran). G2's largest |s_u - dense_cos| is 0.000996, over 292,994
  (squad) to 25,112,502 (musique) rows, with no non-finite value. G3 passed (the report was written).
- **Files.** `outputs/hr1/report.json` and `report.md` are filed with this section. The columns stay on the host.

**Verdict: NO_SIGNAL for both families.** Route 1 closes for these forms. D96 is not raised.

| dataset | A^ment | call | carries | A^co | call | carries |
|---|---|---|---|---|---|---|
| squad | 0.393 | BELOW | no | 0.494 | LEVEL | no |
| musique | 0.550 (0.54972) | ABOVE | no: under the 0.55 bar | 0.522 | ABOVE | no: under the bar |
| hotpotqa | 0.693 | ABOVE | yes | 0.580 | ABOVE | yes |
| 2wiki | 0.593 | ABOVE | yes | 0.486 | BELOW | no |

- **ment:** two datasets carry it and one is BELOW (squad); the rule needs three and none BELOW. Had MuSiQue carried,
  squad's BELOW would still stop it.
- **co:** one carries it (hotpotqa) and one is BELOW (2wiki).

**Beside (deciding nothing):**
- **Next-hop golds.** On the three multi-hop datasets, A over the golds outside s_u's top five (A_miss) is above A in
  every cell:
  - ment: 2wiki 0.679, hotpotqa 0.777, musique 0.605;
  - co: 0.522, 0.650 and 0.567.

  On squad it stays under 0.5 (ment 0.457).
- **Degree.** On the multi-hop datasets, A rises with the gold's degree tercile. The lowest tercile is under 0.5 on
  2wiki and musique.

  | dataset | ment A by tercile (low, mid, high) |
  |---|---|
  | 2wiki | 0.44, 0.56, 0.81 |
  | hotpotqa | 0.54, 0.66, 0.88 |
  | musique | 0.42, 0.59, 0.66 |
  | squad | 0.39, 0.39, 0.40 |

  So what the entity vector separates, it separates on well-linked golds.
- **As a score.** Eq. 5's parameter-free fusion is not a usable score on its own:
  - **AUC.** F^ment's AUC is above s_u's on hotpotqa (0.936 against 0.902) and 2wiki (0.820 against 0.774), and below
    it on musique and squad.
  - **R@5.** F's R@5 is below s_u's on every dataset and both families, by 5.6 to 20.9 points.
  - **The trained twin.** twin0's AUC is 0.988 to 0.993 everywhere.
- **Coverage.** R.mention reaches 94 to 100% of pool rows and co 82 to 90%. Gold and non-gold median degrees are
  close.

**What follows (section 7, NO_SIGNAL).**
- Route 1 closes for s_m^ment, s_m^co and their eq. 5 fusion, as declared. The verdict closes only these forms.
- **Route 2 is named, not declared.** It reads the mention's own sentence through the frozen encoder, which needs new
  encoder calls. That is the user's call under the encoder rule, and it waits for the encoder comparison (ENC) to
  settle the encoder.
- **A new form would come after this report.** The tercile pattern was seen here, so a degree-gated or
  next-hop-only form would be chosen on these training carves. It is not named here. Any such form would need its own
  file and unseen carves.
- **The journal.** The coverage and degree tables go to its analysis beside AP4's precision ledger.
