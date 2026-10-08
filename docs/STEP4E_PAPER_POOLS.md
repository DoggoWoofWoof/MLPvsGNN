# Step 4e: pools built the way the papers build them, toward every gold in the pool

Declared 8 October 2026 at about 22:10, before any number of its pools stage, looks, fits or reads. It follows two
messages from the user:
- 21:55: "we need to improve every gold in the pool as close to 100 because that is the entire point, check how the
  papers do it see if we can take something from them".
- 22:05: "take whatever works from the papers and declare it, dont wait".

## 1. Why

A gold outside a question's pool is lost to every model. Step 4d's union pool U_q (today's frozen pool I_q plus the
walk's first B_q nodes it does not hold) raises the share of questions with every gold in the pool. The figures below
are filed in `outputs/step4d/pools/pools.json` on the s1eval carves, before this file:

| dataset | I_q | U_q |
| --- | --- | --- |
| metaqa | 0.895 | 0.941 |
| squad | 0.980 | 0.980 |
| musique | 0.803 | 0.844 |
| hotpotqa | 0.968 | 0.972 |
| 2wiki | 0.906 | 0.909 |
| webqsp | 0.688 | 0.838 |

That is still far from 1 on four datasets. Twice step 4d's walk budget adds at most 0.007, and almost every gold U_q
misses lies outside the walk's top 2,000 nodes (step 4b's arrays). So a longer walk from the same seeds does not close
the gap. The walk starts from about ten seeds: the dense and SPLADE top five. The papers start elsewhere.

## 2. What the papers do, and what is taken

Each item is a fixed rule over the existing graph, retrieval lists and node names. Nothing here uses a transformer, an
LLM, a new graph, a new embedding or a finer text unit. The encoder stays frozen.

1. **Seed the walk from every retrieved row, weighted by its retrieval score (HippoRAG 2).** HippoRAG 2 seeds its
   personalized PageRank with every passage node, at its dense similarity times 0.05, beside its matched phrase nodes.
   - Taken: the walk's restart distribution covers every row of the question's dense and SPLADE top-1000 lists (the
     frozen caches `prepare` already slices), each weighted by its equal-RRF score (the constant of
     `retrieval_pools.equal_rrf`).
   - Today's seeds (dense and SPLADE top five) keep their place: they carry half the restart mass, and the RRF-weighted
     lists carry the other half. HippoRAG 2's split is not carried over: its 0.05 weights passage nodes against phrase
     nodes, and our graphs have no phrase nodes.
2. **Weight the restart by node specificity (HippoRAG).** HippoRAG's node specificity is the inverse of how many
   passages a node appears in, a graph-native IDF; removing it costs MuSiQue R@2 3.3 points in their ablation.
   - Taken: each restart weight is multiplied by 1 / log(e + degree), the degree over the regime's families. Hubs then
     cannot draw the walk's mass from rare bridge nodes.
3. **Link the question's entities by exact surface match (PullNet and GraftNet).** PullNet links MetaQA's questions by
   "simple exact match on surface forms", and WebQSP's by exact match against FACC1's surface forms; the longer of two
   overlapping matches wins. SR and NSM take topic entities as given. Those entities are an oracle we do not use.
   - Taken: every node whose name equals a question n-gram (n up to 8) becomes a linked seed. The name is the node
     record's `title`, `display_name_disambiguated` or `display_name`; a Wikipedia slug's underscores read as spaces.
     Names and n-grams are lowercased and stripped of punctuation.
   - The longest non-overlapping matches win. A name of fewer than three characters, or made only of stopwords
     (sklearn's English list), never links. A name shared by more than 50 nodes never links (ambiguous).
   - Linked seeds join the seeds' half of the restart mass, with equal weight. On the passage datasets the names are
     titles, which is title matching (HotpotQA's and 2Wiki's bridge entities are titled pages). squad links nothing:
     it has no graph and its titles are articles, not passages.
4. **Keep the whole neighbourhood the frozen construction reaches (GraftNet and PullNet).** The papers' subgraphs are
   the topic entities' multi-hop neighbourhood, cut by PPR. Taken: U_q stays whole, so I_q's hops stay, and the new
   walk only adds.
5. **Retrieval depth on the graph-free dataset.** squad's pool is retrieval's top 50 rows; it has no graph to walk.
   Taken: its expansion is the equal-RRF list's next rows in order, the walk's degenerate case with no edges.

**Not taken now:** SR's trained path retriever (relations scored against the question, trained on the shortest paths
from topic entities to answers) and PullNet's trained expansion classifier. Both are learned. They are the next step
if this fixed rule stops short, declared in their own file.

## 3. Arms and budgets

Each arm ranks the nodes outside U_q by a walk (step 4b's kernel, the regime's families, restart alpha 0.5, the same
EPS, top 2,000 outside U_q). The pool is U_q plus the arm's first k × B_q nodes (B_q is step 4c's budget).

| arm | restart |
| --- | --- |
| A0 | today's seeds (step 4d's walk, again past U_q) |
| A1 | item 1: seeds plus the RRF-weighted top-1000 lists |
| A2 | A1 with item 2's specificity weights |
| A3 | A2 with item 3's linked seeds |

The budgets are k ∈ {0, 0.5, 1, 2, 3}; k = 0 is step 4d's U_q. On squad every arm is item 5's RRF order.

## 4. The rule, fixed before any number

Pools are built for every carve. Coverage ("every gold in the pool", ALL) is read on the five training datasets'
s1sel carves; webqsp has none and is read only on s1eval, beside the choice.

1. **The arm.** The arm with the highest mean ALL over the five s1sel carves at k = 1 is chosen. An arm within 0.002
   of a lower-numbered (simpler) arm loses to it.
2. **The budget.** The smallest k whose mean ALL is within 0.005 of k = 3's is chosen, for every dataset (one rule, one
   k).
3. **No pool may lose a gold** that U_q holds: U_q is inside every pool. The stage stops if one does.
4. The s1eval coverage of the chosen pools, on all six datasets, is filed with the choice. A choice whose s1eval ALL
   falls below step 4d's U_q on any dataset stops the step.

## 5. What follows the choice

Step 4d's stages 2 to 5 (looks, caches, identity gate, the zrm screen) move here and run once, on the chosen pools:
- This is an amendment to docs/STEP4D_UNION_POOLS.md made before any number of those stages.
- Step 4d's U_q is k = 0 of every arm here, so its pools are not looked at separately.

The screen and verdict are step 4d's section 4:
- zrm refit on the chosen pools for L-musique and L-hotpotqa, against zrm's screen fits on today's pools, under the
  seed null;
- R@5 over the question's golds;
- PROMISING opens a full run in its own file; an ADOPT moves both models, the MLP and the GNN track, to the chosen
  pools.

## 6. Cost and speed

- Pools grow by up to 3 × B_q past U_q, about 2 to 4 times today's size on metaqa, musique and webqsp. The stage files
  the exact sizes.
- Every model's per-question compile and forward grow with the pool, and so does the walk's cost per question. The
  restart covers about 2,000 rows instead of ten.
- Latency is cold (8216ffe) and reported beside the pool sizes.
- Exact-match linking is a dictionary lookup per question n-gram, built once per dataset from node names.

## 7. Order

The pools stage runs on the laptop (numba). The miss diagnostic, which measures each missed gold's hop distance from the
seeds and its retrieval rank, runs beside it as context only; it decides nothing here. The looks queue on the host's
CPU behind rounds twenty-four and twenty-five's CPU items.

*Amended 8 October about 22:25, before any number of the step (three points the code met):*
1. **The walk's length.** k = 3 needs up to 3 × B_q nodes past U_q, more than 2,000 on metaqa, musique and webqsp. Each
   arm's walk keeps the first 3 × B_q nodes outside U_q, not 2,000.
2. **squad's budget.** squad's B_q is 0: its pool is its base. On squad, B_q is read as |I_q| (50), so k × 50 rows of
   the equal-RRF lists past its pool.
3. **Linking reads the plain question.** MetaQA's question text marks its topic entity in brackets, and the records
   carry topic-entity fields. Linking reads the question with brackets removed (`question_plain` where the record
   has it) and never reads a topic-entity, context or gold field.
4. **A title's qualifier.** Wikipedia-style titles carry a qualifier in parentheses ("Gasera (woreda)"). Each name is
   indexed twice: as written, and with a trailing parenthetical removed. A KB name's surrounding whitespace is
   stripped. The ambiguity guard (more than 50 nodes) counts both forms.
5. **The RRF weight.** A row's restart weight is 1/(c + its dense rank) + 1/(c + its SPLADE rank) over the two top-1000
   lists, with c = `retrieval_pools.equal_rrf.constant`. A list the row is absent from adds 0.

The stage runs in three parts:
- `coverage`: every arm and budget on the five s1sel carves and the six s1eval carves;
- `choose`: section 4's rule;
- `file`: the chosen arm's pools for all 21 carves, in step 4d's layout.

*Amended 8 October about 22:40, before the choice and before any webqsp coverage number (musique's and hotpotqa's
coverage lines had printed):* the user (22:38): "not all 5, all 6".
- **The choice reads all six datasets.** Section 4's means are over six carves: the five s1sel carves and webqsp's
  s1eval carve. webqsp has no select carve: it never trains, and step 1 carves only its eval questions.
- **What that costs.** webqsp's pool coverage is then in-sample for the choice. The choice is one of 20 settings (four
  arms, five k), made on gold coverage alone. Its model reads stay zero-shot: no model trains on webqsp.
- The rest of section 4 is unchanged, including the stop when a chosen pool falls below U_q on any s1eval carve.

*Amended 9 October about 00:55, before any number of the looks, caches, gate or screen (section 5's host stages):*
- **Code.** `scripts/step4e_host.py` (numba-free: the host has no numba) holds the pool replacement and the gate;
  `outputs/mp_unified/look_step4e.py` is step 4c's look binding on the chosen pools; `outputs/mp_unified/zrm4e.py` runs
  zrm's own train, read and compare and rmatch's chain build with step 4e's roots (`outputs/step4e/cache`,
  `outputs/step4e/chains`), and zlink's pair and re-call, decided against zrm's screen fit of each split.
- **What the replacement checks.** Base and seeds and the frozen pool are the filed ones (sha256). The expansion repeats
  no node, holds no node of base and seeds and is at least its new nodes long. The pool is base and seeds plus the
  expansion and holds every node of the frozen pool. Each look shard records its question count and row total, and
  both must be the file's.
- **The gate has two scopes.** `gate --screen` covers the eleven carves the screen reads: the five fit carves it trains
  on and the six s1eval carves. The plain gate covers all 21, for full runs. The checks are step 4c's:
  - every look shard ran on the manifest's pools file;
  - every cache row count is the chosen pool's size;
  - step 1's cache rows are the frozen pools' sizes.
  The screen waits only for its own gate.
- **The look cost is about quadratic in the pool.** On two musique s1sel questions on the laptop, a look took 31.5 s
  per question on the chosen pools against 3.7 s on today's (3.35 times the rows). Step 4c's shard counts are scaled
  by (pool ratio)^1.8 to shards of about 30 minutes, and its cache parts by the row ratio.
- **R@5's denominator is the question's gold count** (`look_x_six`: `gold_total = pop.golds[j].size`). Reads on the
  chosen pools and on today's pools therefore compare question by question; zrm's compare checks the question ids and
  gold totals are the same.
- **GPU share.** Each fit and read of the screen takes 0.62 of the card (cap 0.6). That is zrm's measured torch peak of
  about 4 GB times the chosen pools' 3.4 times the rows on metaqa and musique; the whole carve sits on the device.

## Results

**Coverage and choice (9 October 00:21).** `outputs/step4e/coverage.json`, `outputs/step4e/choice.json`. Every gold in
the pool (ALL) at k = 1 / k = 3 per arm; k = 0 is step 4d's U_q; the last columns are the chosen A3 at k = 2 and the
mean pool size from U_q to it.

| carve | U_q | A0 | A1 | A2 | A3 | chosen (A3, k = 2) | pool |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa s1sel | 0.947 | 0.953 / 0.971 | 0.955 / 0.960 | 0.955 / 0.961 | 0.956 / 0.962 | **0.960** | 3,150 -> 7,093 |
| metaqa s1eval | 0.941 | 0.954 / 0.973 | 0.947 / 0.952 | 0.946 / 0.952 | 0.949 / 0.954 | **0.953** | 3,147 -> 7,082 |
| squad s1sel | 0.979 | 0.989 / 0.993 | 0.989 / 0.993 | 0.989 / 0.993 | 0.989 / 0.993 | **0.992** | 50 -> 150 |
| squad s1eval | 0.980 | 0.989 / 0.994 | 0.989 / 0.994 | 0.989 / 0.994 | 0.989 / 0.994 | **0.992** | 50 -> 150 |
| musique s1sel | 0.819 | 0.861 / 0.882 | 0.926 / 0.949 | 0.925 / 0.947 | 0.934 / 0.958 | **0.950** | 3,134 -> 6,921 |
| musique s1eval | 0.844 | 0.876 / 0.906 | 0.918 / 0.946 | 0.918 / 0.946 | 0.921 / 0.948 | **0.937** | 3,114 -> 6,899 |
| hotpotqa s1sel | 0.970 | 0.974 / 0.978 | 0.979 / 0.985 | 0.977 / 0.982 | 0.975 / 0.979 | **0.978** | 100 -> 188 |
| hotpotqa s1eval | 0.972 | 0.974 / 0.978 | 0.978 / 0.984 | 0.977 / 0.983 | 0.976 / 0.980 | **0.979** | 100 -> 188 |
| 2wiki s1sel | 0.897 | 0.906 / 0.908 | 0.914 / 0.921 | 0.915 / 0.920 | 0.938 / 0.951 | **0.947** | 109 -> 220 |
| 2wiki s1eval | 0.909 | 0.916 / 0.920 | 0.921 / 0.929 | 0.919 / 0.928 | 0.936 / 0.949 | **0.945** | 110 -> 221 |
| webqsp s1eval | 0.838 | 0.862 / 0.876 | 0.897 / 0.916 | 0.894 / 0.912 | 0.893 / 0.915 | **0.909** | 3,374 -> 7,213 |

Mean ALL over the six choice carves (five s1sel, webqsp's s1eval):

| arm | k = 0 | 0.5 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- |
| A0 | 0.9084 | 0.9182 | 0.9242 | 0.9308 | 0.9348 |
| A1 | 0.9084 | 0.9343 | 0.9432 | 0.9497 | 0.9541 |
| A2 | 0.9084 | 0.9326 | 0.9424 | 0.9482 | 0.9524 |
| A3 | 0.9084 | 0.9374 | 0.9474 | 0.9558 | 0.9597 |

- **The choice: A3 at k = 2, CHOSEN.** A3 leads at k = 1 (0.9474 against A1's 0.9432, beyond the 0.002 tie). Its k = 3
  mean is 0.9597; k = 2's 0.9558 is within 0.005 of it, and k = 1's 0.9474 is not.
- **No s1eval carve falls below U_q.** From today's frozen pools I_q through U_q to the chosen pools, every gold in
  the pool on s1eval:

| dataset | I_q | U_q | chosen |
| --- | --- | --- | --- |
| metaqa | 0.895 | 0.941 | **0.953** |
| squad | 0.980 | 0.980 | **0.992** |
| musique | 0.803 | 0.844 | **0.937** |
| hotpotqa | 0.968 | 0.972 | **0.979** |
| 2wiki | 0.906 | 0.909 | **0.945** |
| webqsp | 0.688 | 0.838 | **0.909** |

- **What worked, by paper.**
  - HippoRAG 2's retrieval-weighted restart (A1) carries most of the gain on musique, webqsp and hotpotqa.
  - PullNet's exact title and name linking (A3) adds most on 2wiki (s1sel 0.915 to 0.938 at k = 1) and on musique.
  - HippoRAG's node specificity (A2) adds nothing over A1 here.
  - On metaqa, today's seeds (A0) reach further at k = 3 (s1eval 0.973 against A3's 0.954). Its three-hop answer
    sets sit deep around one topic entity, and spreading the restart moves mass off it. One rule serves all six
    datasets, so metaqa takes A3: it still rises from 0.941 to 0.953 over U_q.
- **The cost.** Pools grow about 2.2 times past U_q: metaqa 3,147 to 7,083 rows, musique 3,114 to 6,899, webqsp 3,374
  to 7,213, hotpotqa 100 to 188, 2wiki 110 to 221, squad 50 to 150. Against today's I_q, that is about 3.5 times on
  the three large datasets. Looks, caches, fits and cold latency grow with it.
- **Next.** The file stage builds the chosen pools for all 21 carves on the laptop. Then step 4d's stages 2 to 5 run
  on them, in the declared order (looks, caches, identity gate, zrm screen).
