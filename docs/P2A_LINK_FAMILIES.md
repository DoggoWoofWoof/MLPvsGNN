# P2a: what each link family is worth, on all six full corpora (a diagnostic)

Declared 13 October 2026, before any number of the stage. P2's first stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md). Nothing here is adopted, and no model trains.

Two parts, each declared before it runs:
- **Part 1, reach (this declaration):** what each family adds to the pools, walked under the same kernel as P1a.
- **Part 2, ranking:** what today's fits lose when one family is masked at read time. It is declared in this file
  before it runs; until then only part 1 runs.

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
