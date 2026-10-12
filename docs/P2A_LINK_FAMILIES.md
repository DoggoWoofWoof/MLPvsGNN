# P2a: what each link family is worth, on all six full corpora (a diagnostic)

Declared 13 October 2026, before any number of the stage. P2's first stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md). Nothing here is adopted, and no model trains.

Two parts, each declared before it runs:
- **Part 1, reach (sections 2 to 5):** what each family adds to the pools, walked under the same kernel as P1a.
  Committed at 03:38 (c866653).
- **Part 2, ranking (sections 6 to 10):** what today's fits lose when one family is masked at read time. Declared 13
  October, before any number of part 2.

## 1. The question

The user, 13 October: how much do the links help, can they retain or add more information (as on musique), and what
can be discarded. Our graphs have three link families, M3B's stores:
- **structural (S):** the corpus's own links: KB triples on metaqa and webqsp, the documents' hyperlinks on 2wiki and
  hotpotqa, title mentions on musique.
- **ner (N):** links through shared named entities.
- **knn (K):** nearest neighbours in the frozen encoder's space.

Each dataset walks a frozen subset of them, its M3A regime:

| dataset | regime | families walked today |
| --- | --- | --- |
| metaqa, hotpotqa, 2wiki | STRUCT | S |
| musique, webqsp | FULL | S, N, K |
| squad | RETRIEVAL | none (the pools are the first-stage lists) |

The regimes were chosen per dataset (M3A's headroom). The rule since 9 October is one construction for all datasets,
with per-dataset picks only an upper bound. P2a measures each family in and out of every dataset's graph, so that the
next graph change can be one rule for all six.

## 2. Part 1: the walks

- **Graphs:** every subset of the three families: none, S, N, K, SN, SK, NK and SNK. All three stores exist for every
  dataset (M3B's CSR cache); the stage reads them and refuses to build any.
- **The kernel:** step 4e's walk (its alpha and push threshold, top 7,000 rows, nothing excluded), unchanged. From a
  node, the walk picks a family uniformly among those in which the node has neighbours (step 4's rule). Degrees count
  only the subset's entries, so A3's specificity weights follow the subset.
- **Restarts:**
  - A3: ours, step 4e's chosen restart (seeds and linked nodes, half; the RRF lists, half; specificity weighted).
  - A1: HippoRAG 2's rule (seeds, half; the RRF lists, half), which leads P1a on the passage graphs and webqsp.
  - LINKED: uniform over our linked entities, on metaqa and webqsp only (it leads P1a on metaqa).
- **none:** with no graph, an order is its restart's own nodes by weight, ties by node position.
- **Carves:** s1eval on all six, and s1sel on the five training datasets, every question.
- **Measures:** as P1a's. At N = 25 to 7,000:
  - ALL: the share of questions with every gold in the top N;
  - recall: the mean share of golds in the top N.

  Also at each question's own size of today's pools (|I_q|) and of step 4e's (|P4e|).
- **The check first:** the regime subset must reproduce P1a's A3, A1 and LINKED records exactly (every N, both
  measures, both matched sizes) on the five graph datasets. Otherwise the stage stops. squad's regime walks nothing,
  so P1a's squad orders were the RRF lists and there is no check there.

## 3. Part 1: the reading, fixed before any number

All on s1eval, as P1a's call; s1sel is reported beside.

- **A family's value, against the dataset's regime:**
  - for a family in the regime, leave one out (LOO): the regime minus the regime without the family;
  - for a family outside it, add one (ADD): the regime with the family minus the regime;
  - on squad (regime none), ADD is the family alone minus none.
- **Labels,** over every N and both measures, with step 4e's tie of 0.002:

  | label | the value is |
  | --- | --- |
  | CARRIES | above +0.002 at some point, never below -0.002 |
  | HARMS | below -0.002 at some point, never above +0.002 |
  | MIXED | both |
  | NOTHING | neither |

- **Candidates:**
  - DROP_CANDIDATE: a regime family whose LOO label is NOTHING or HARMS under every restart, on s1eval and s1sel.
  - ADD_CANDIDATE: a family outside the regime whose ADD label is CARRIES under some restart on s1eval, and not HARMS
    or MIXED under that restart on s1sel (where the dataset has one).
- **One subset for all six:** each subset against each dataset's regime under A3 on s1eval. A subset is a universal
  candidate when it is never below a regime by more than 0.002 on any dataset and above one somewhere.
- **Reported beside, deciding nothing:**
  - the best subset at every N and at |I_q| (a per-dataset upper bound);
  - each subset's entries and the nodes it touches;
  - the values at the matched sizes.

## 4. What follows part 1

- **A drop candidate** is dropped only if part 2 also finds the models do not rely on it. Then a one-fit drop screen,
  declared in its own file, makes the change on every dataset where the family is in the regime.
- **An add candidate, or a universal subset candidate,** is a graph change. It is declared in its own file and graded
  on all six, pools first (as P1) and then a one-fit screen.
- **Anything else stays as it is.** Families that are MIXED name the N where they help and where they hurt, which is
  where a better link rule (P2b) starts: keep what adds golds near the head, drop what pushes them out.

Flagged, as every pool so far: the walks are propagation over the graph. The pool is built before either model runs,
and both models of a family read the same pool, so the MLP's own scoring stays free of message passing.

## 5. Part 1: how it runs

- **Code:** `scripts/p2a_reach.py` (stages `reach` and `verdict`). Its self-test passes:
  - a subset's union equals a union built from those stores alone, and its degrees count the subset only;
  - the regime subset is the union step 4e opens;
  - a walk leaves the seeds' component only through the subset's families;
  - zero-step orders, LOO and ADD against the regime, the labels, the candidates and the universal call.
- **Development check, no new number:** with `--only-regime` on the laptop, the regime subset reproduced P1a's records
  on musique (FULL) and metaqa (STRUCT, with LINKED) on both carves.
- **Host jobs** (`p2a-pools-<dataset>`, step 4f's pattern, light lane), sized from P1a's measured peaks plus the two
  extra stores:

  | dataset | threads | memory |
  | --- | ---: | ---: |
  | 2wiki | 8 | 6 GB |
  | hotpotqa | 6 | 5 GB |
  | metaqa | 4 | 3 GB |
  | webqsp | 3 | 3 GB |
  | musique | 3 | 2 GB |
  | squad | 1 | 2 GB |

  Each writes `outputs/p2a/reach_<dataset>.json`. Then `verdict`, run locally, writes
  `outputs/p2a/reach_verdict.{json,md}`. It refuses parts from different scripts or freezes, parts run with
  `--limit` or `--only-regime`, and parts whose regime subset differs from P1a.
- **ETA:** 7 subsets × 2 or 3 restarts per carve, about 3 times P1a's walks. 2wiki is the longest, at about 35 minutes
  on 8 threads. The call should land about 45 minutes after the start.
- The encoder stays frozen. No model trains or reads, nothing is adopted, and test splits are never read.

## 6. Part 2: the question

Part 1 asks what each family adds to the pools. Part 2 asks what today's lean fits read from each family once the pool
is fixed. The pool is the same in every arm; only the inputs computed from the pool's edges change:

| input | the families it reads |
| --- | --- |
| rank, dense_cos, SEMB, SEED | none |
| the compile's STRUCT topology and depth columns, WALK, DISTS | structural |
| the chain entries (metaqa and webqsp) | structural (typed) |
| WALKF | all three |
| zsp's links, its one propagation step | all three, one slot per family |

So masking N or K changes only WALKF and zsp's N or K slot. Masking S changes the rest of the graph inputs.

Read-time masking measures reliance, not value. A family the fits rely on might be replaced by a refit without it, and
masked inputs are out of the fits' training distribution. A family they do not rely on carries nothing through these
fits. A drop needs both parts (section 4).

## 7. Part 2: the read

- **Fits:** today's lean bases as filed, zrc (the MLP) and zsp (the GNN: zrc's inputs plus one propagation step), J5,
  candidate p@swa, basis 2wiki (`outputs/b1b/fits`, the fits C11 timed). webqsp is zero-shot.
- **Forms:** C11's serial forms (zrc11 and zsp10), which C11 checked against the modules within 1e-4 with the same top
  5 on 200 questions of every dataset.
- **Carve:** s1eval, every question, all six datasets.
- **Pools:** step 1's, from C10's pool builder, the same in every arm.
- **Arms:** none (unmasked), S, N and K; SNK (all three) is reported beside.
- **Masking:** for one question's compile only, the masked family's arrays in the compiler are emptied: no entries, all
  offsets zero. Its pool edges and its global degrees are gone, as if the graph did not hold the family. The other
  families' edges are unchanged, since the 64-neighbour cap is per family. One compile per arm is read by both fits.
- **Measures:** R@5, FC@5 and hit@1 per question, by the grades' own code (lean_gpu's top_hit and metrics_of), with step
  1's golds. A question's gold total counts golds outside the pool, as the filed grades do.
- **Checks:**
  - Exact, every question, fatal: the pool equals step 1's cached pool, and each masked arm's edges are the unmasked
    arm's without the masked families' edges.
  - The unmasked rows against step 1's cached rows. They must be bit for bit equal, or within 1e-3 + 2^-8 of the
    cached value (a few float16 steps). A dataset with more than 1% of its questions beyond that is not read.
  - C1 and C5 measured laptop-against-host differences of at most 7.3e-4, in 10 to 20 of 200 questions on musique,
    metaqa and webqsp. The run uses the host, where step 1's cache was built.

## 8. Part 2: the reading, fixed before any number

- **The loss:** per question, the fit's measure unmasked minus masked, which is what it loses without the family. Its
  mean over the carve gets a paired bootstrap 95% interval: 2,000 resamples of the questions, seed 0.
- **A difference counts** when the mean loss is beyond 0.002 (step 4e's tie) and its interval excludes 0.
- **Labels,** per dataset, fit and family, over R@5, FC@5 and hit@1:

  | label | counted differences |
  | --- | --- |
  | RELIES | a loss on some measure, a gain on none |
  | BETTER_WITHOUT | a gain on some measure, a loss on none |
  | MIXED | both |
  | NOTHING | neither |

- **Feature drop candidate:** a family labelled NOTHING or BETTER_WITHOUT for both fits on all six datasets.
- **Graph drop candidate:** a feature drop candidate that part 1 also labels DROP_CANDIDATE on every dataset whose
  regime walks it.
- **Reported beside, deciding nothing:**
  - the SNK arm;
  - the unmasked R@5 beside the filed J5 reads, which were read on a GPU from step 1's rows;
  - the share of questions whose top 5 changes in each arm;
  - edges per question per family, and the share of questions with any;
  - the rows against step 1.

## 9. What follows part 2

- **A graph drop candidate:** a one-fit drop screen, declared in its own file, on all six.
- **RELIES where part 1 says DROP_CANDIDATE:** the features keep the family, and the pools may drop it under P1's rules.
  The pool and the features are two separate reads of the graph.
- **BETTER_WITHOUT:** goes to P2b (a better link rule), naming the measure where it shows.
- **The MLP's reliance on a family is not message passing.** zrc's graph inputs are counts and walks computed from the
  pool's edges before any row is scored (the 8 October rule). zsp's links are its one propagation step.

## 10. Part 2: how it runs

- **Code:** `scripts/p2a_rank.py` (stages `read` and `verdict`). Its self-test passes:
  - part ranges;
  - emptied family arrays keep their dtypes and leave the live ones untouched;
  - kept edges;
  - row differences against the tolerance;
  - top_hit's top-5 order;
  - the paired bootstrap, counted differences and labels.
- **Development check, no number:** with `--limit 16` on the laptop, on musique (FULL, untyped) and metaqa (STRUCT,
  typed, chains):
  - the pools equal step 1's;
  - each arm drops exactly its family's edges;
  - the unmasked rows are bit for bit step 1's on 15 of 16 and 16 of 16 questions (the other within 1.2e-4);
  - 0.39 to 0.49 s per question for five arms and two fits, with peaks of 3.7 and 5.7 GB.
- **Host jobs:** light lane, one CPU each.
  - First, two development items (`p2a-read-dev-musique` and `p2a-read-dev-metaqa`, `--limit 16 --host`). They write
    nothing.
  - Then 37 parts that depend on both, `p2a-read-<dataset>-<k>of<N>`, each a contiguous range of about 1,250
    questions (`--part k/N`). Each writes `outputs/p2a/rank_<dataset>_<k>of<N>.{json,npz}`.

  | dataset | questions | parts | memory per part |
  | --- | ---: | ---: | ---: |
  | metaqa | 9,785 | 8 | 6.5 GB |
  | squad | 11,873 | 9 | 4.5 GB |
  | musique | 2,417 | 2 | 4.5 GB |
  | hotpotqa | 7,405 | 6 | 5 GB |
  | 2wiki | 12,576 | 10 | 4.5 GB |
  | webqsp | 1,503 | 2 | 6.5 GB |

- **The verdict,** run locally, writes `outputs/p2a/rank_verdict.{json,md}`. It refuses parts that run on different
  scripts, fits or frozen modules, and parts that do not cover the carve once.
- **ETA:** about 0.45 s per question, 45,559 questions, so about 5.7 CPU hours, in parts of about 10 minutes. The call
  should land about 40 minutes after the development items pass.
- The encoder stays frozen. No model trains, nothing is adopted, and test splits are never read.

## 11. Part 1: results

Filed 13 October 2026, about 04:45. The six host jobs ran on the full corpora, and `verdict` ran locally
(`outputs/p2a/reach_verdict.{json,md}`, script sha256 `ba89856f…`). Check passed: on every carve, the regime subsets
equal P1a's records.

**Each family against the dataset's regime** (s1eval, A3, ALL at |I_q|; the label covers every N, both measures and
every restart):

| dataset | regime | structural | ner | knn |
| --- | --- | --- | --- | --- |
| metaqa | S | LOO CARRIES, +0.540 | ADD HARMS, −0.014 | ADD HARMS, −0.063 |
| squad | none | ADD HARMS, −0.025 | ADD HARMS, −0.020 | ADD HARMS, −0.019 |
| musique | SNK | LOO CARRIES, +0.015 | LOO CARRIES, +0.032 | LOO CARRIES (A1 MIXED), +0.008 |
| hotpotqa | S | LOO MIXED (A1 CARRIES), +0.165 | ADD HARMS, −0.044 | ADD MIXED, −0.035 |
| 2wiki | S | LOO CARRIES, +0.310 | ADD MIXED, −0.114 | ADD MIXED, −0.111 |
| webqsp | SNK | LOO CARRIES, +0.265 | LOO HARMS, −0.018 | LOO HARMS, −0.049 |

**Candidates (section 3's rule):**
- **DROP_CANDIDATE:** webqsp's ner and knn. Both are HARMS under A3, A1 and LINKED. Under A3, S alone holds 0.829 of
  webqsp's questions whole at |I_q|, against the regime's 0.759 (+0.070).
- **ADD_CANDIDATE:** none.
- **One subset for all six:** none. Every subset is below some dataset's regime by more than the tie. The
  per-dataset best at |I_q| (an upper bound, deciding nothing) is the regime on five datasets and S on webqsp.

**What it says:**
- **The structural family carries the reach** wherever a dataset has one: the KB triples (metaqa +0.540, webqsp
  +0.265), the hyperlinks (2wiki +0.310, hotpotqa +0.165) and musique's own (+0.015). The one exception is hotpotqa's
  head under A3: at N = 25 the walk pushes golds out (ALL 0.512 against 0.554 without it). A1's restart has no such
  point.
- **ner and knn add reach only on musique**, the one passage graph without corpus hyperlinks. There ner is the
  largest family (+0.032 at |I_q|). On every other graph they push golds out at today's sizes. On 2wiki they help only
  far down the order (N ≥ 1,500 under A3; N ≥ 300 for ner under A1) and hurt the head. Step 4's union walk showed the
  same pattern: a looser family dilutes the structural hop.
- **The MIXED families name where a better rule starts** (P2b), as section 4 says: on 2wiki, ner and knn add golds
  below N = 1,500 and push them out above it. A rule that walks them only after the structural family keeps the head
  and adds the tail. It would be one rule for all six, not a per-dataset subset.

**What follows (section 4):**
- webqsp's ner and knn are dropped only if part 2 also finds the models do not rely on them.
- A drop screen makes the change on every dataset where the family is in the regime. Those are musique and webqsp,
  and on musique both families carry the reach. So a plain drop of either family is not universal, and the
  universal form of what part 1 found is a rule over the families (P2b), declared in its own file.
- No graph changes in this part.

## 12. Part 2: results

Filed 13 October 2026, about 05:15.
- **Runs:** the 37 host parts, after the two development items, ran on the host. `verdict` ran locally
  (`outputs/p2a/rank_verdict.{json,md}`, script sha256 `b3fbf46b…`).
- **Exact checks** passed on every question: the pools equal step 1's, and each masked arm drops exactly its family's
  edges.
- **Rows:** the unmasked rows are within the tolerance of step 1's on every question of all six datasets. They match
  bit for bit on 88% (musique) to 99.9% (squad) of the questions, and the largest difference is 0.0017 (hotpotqa).
- **Unmasked R@5** equals the filed J5 reads to four places on all six datasets, for both fits.

**What each fit loses without each family** (s1eval). Each cell gives the label for zrc / zsp, then the losses that
count: the fit's measure unmasked minus masked, beyond 0.002 with its 95% interval clear of 0. A negative loss is a
gain without the family.

| dataset | structural | ner | knn |
| --- | --- | --- | --- |
| metaqa | RELIES / RELIES: R@5 0.772 / 0.773, FC@5 0.664 / 0.664, hit@1 0.889 / 0.887 | NOTHING / NOTHING | RELIES / RELIES: hit@1 0.006 / 0.003 |
| squad | NOTHING / NOTHING | NOTHING / NOTHING | NOTHING / RELIES: hit@1 0.003 (zsp) |
| musique | RELIES / RELIES: R@5 0.029 / 0.026, FC@5 0.055 / 0.044, hit@1 0.028 / 0.015 | RELIES / RELIES: R@5 0.009 / 0.014, FC@5 0.015 / 0.019 | NOTHING / RELIES: hit@1 0.016 (zsp) |
| hotpotqa | RELIES / RELIES: R@5 0.169 / 0.168, FC@5 0.333 / 0.330 | RELIES / NOTHING: hit@1 0.005 (zrc) | RELIES / RELIES: hit@1 0.006 / 0.008 |
| 2wiki | MIXED / MIXED: R@5 0.169 / 0.165, FC@5 0.309 / 0.311, hit@1 −0.016 / −0.010 | RELIES / NOTHING: hit@1 0.005 (zrc) | RELIES / RELIES: hit@1 0.004 (zrc), FC@5 0.003 (zsp) |
| webqsp | RELIES / RELIES: R@5 0.223 / 0.187, FC@5 0.174 / 0.144, hit@1 0.049 / 0.080 | NOTHING / NOTHING | NOTHING / RELIES: R@5 0.013, FC@5 0.009 (zsp) |

Beside:
- **SNK:** masking all three families costs more than masking S alone on musique (R@5 0.053 against 0.026 to
  0.029), hotpotqa (0.190 to 0.200 against 0.168 to 0.169) and 2wiki (0.224 against 0.165 to 0.169). With the
  structural family gone, ner and knn carry part of what it carried. One label is new: zsp on squad is BETTER_WITHOUT
  (R@5 and FC@5 −0.0025).
- **Questions whose top 5 changes:** 3% to 50% without ner, and 21% to 52% without knn.

**Families (section 8's rule):**
- **No feature drop candidate.** Each family is RELIES for some fit on some dataset:
  - structural on five datasets, for both fits (RELIES on four, MIXED on 2wiki);
  - ner on musique for both fits, and on hotpotqa's and 2wiki's hit@1 for zrc;
  - knn on all six for zsp, and on metaqa, hotpotqa and 2wiki for zrc.
- **So no graph drop candidate.**
- **webqsp, against part 1.** Part 1 made webqsp's ner and knn drop candidates for the pools. Neither fit relies on
  webqsp's ner. zsp relies on its knn, zero-shot (R@5 0.013).
  - Section 9 applies: the features keep the family, and the pools may drop it under P1's rules.
  - A pool drop of either family is not universal, because musique's pools need both (part 1). It waits for P2b's
    family rule.

**What it says:**
- **The structural family is what both fits read from the graph.** Masking it costs R@5 0.17 to 0.77 on the four
  datasets with corpus or KB links (hotpotqa, 2wiki, webqsp, metaqa), and 0.026 to 0.029 on musique.
- **On 2wiki it also costs the top 1.** Without it, hit@1 rises by 0.016 (zrc) and 0.010 (zsp), while R@5 falls by
  0.17. Rows the graph favours displace the row the text scores would put first.
  - This is the one counted gain from masking a single family.
  - Part 1 found the same shape in 2wiki's pools: ner and knn add golds far down the order and push them out at the
    head.
- **ner and knn act mostly at the top 1.** Three exceptions:
  - on musique, ner carries R@5 and FC@5 for both fits, as it carries musique's pools (part 1);
  - zsp's knn on webqsp (R@5 0.013);
  - zsp's knn on 2wiki (FC@5 0.003).

  Every other counted reliance on them is hit@1, by 0.003 to 0.016. They change the top 5 on up to half the
  questions, mostly within the tie on R@5.
- **The GNN reads knn more than the MLP does.** zsp relies on knn on all six datasets, zrc on three. zsp's propagation
  step has a knn slot. zrc reads knn only through WALKF, which is computed from the pool's edges before any row is
  scored, so its reliance is not message passing (section 9).

**What follows (section 9):**
- **No drop screen,** since there is no graph drop candidate. No graph changes in P2a.
- **P2b,** declared in its own file before any number: one link rule for all six that orders the families, the
  structural family first. Its starting points:
  - from part 1, the families' pool shapes: ner and knn add golds on musique, and on 2wiki only far down the order;
  - webqsp's ner and knn in the pools;
  - from part 2, the top 1: 2wiki's hit@1 without the structural family, and ner's and knn's hit@1 reliance.

  It is graded on all six, the pools first and then a one-fit screen.
