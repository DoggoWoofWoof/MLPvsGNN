# Project timeline: what we tried, why, what went wrong, and where we stand (25 Aug to 14 Oct 2026)

Written 14 October 2026 from the stage docs, their filed records and git history (876 commits, 236 docs). Every
number is a development number; the paper's numbers come from one confirmation run later. Each point names the doc it
comes from, so any number can be checked.

**How to read it.** Section 1 is the whole project on one page. Section 2 explains the words. Section 3 is the timeline,
era by era: what we tried, why, the problem we hit, how we fixed it, the result, and what made the difference.
Section 4 sums up each topic (pools, links, models, generalization, speed). Section 5 is an honest audit: do we
hardcode anything per dataset, would it work on a new dataset, and do we copy anything. Section 6 lists open issues
and the fixes we propose. Section 7 is where we stand tonight.

**Numbers from different eras are not directly comparable:**
- 25 Aug to 2 Sep: an older, smaller copy of the datasets; scores in percentage points (for example 68.40).
- From 8 Sep: the full corpora (the CRAG substrate); scores as fractions (for example 0.684).
- From 7 Oct: fixed evaluation sets per dataset (called s1eval), the same for every later model.
- B1 (HippoRAG 2's own benchmark) reports recall@5 in percent, as HippoRAG 2 does (for example 97.2).

## 1. The project on one page

**The question.** Six retrieval datasets: two knowledge bases (metaqa, webqsp: the answers are entities linked by
typed facts) and four passage corpora (squad, musique, hotpotqa, 2wiki: the answers are Wikipedia-style passages).
For each question, a first stage (a frozen text encoder plus SPLADE, a word-matching retriever) proposes candidates.
A graph links the candidates (KB facts, hyperlinks, shared names). Then a model reranks a pool of candidates.

- A **GNN** reranks with message passing: at query time, a candidate's representation or score is updated from its
  neighbours'.
- An **MLP** scores each candidate from that candidate's own row of features. Graph structure can still enter, but
  only as features computed before any row is scored.

**The three claims** (fixed 9 October, docs/PAPER_CLAIMS_2026_10_09.md):
1. The MLP, with no message passing, keeps at least 90% of the GNN's R@5 gain over the first stage on each of the six
   datasets in-domain; zero-shot, it is not further below the GNN than the in-domain gap.
2. A GNN with no LLM call anywhere (only frozen text encoders) is at the state of the art (SOTA).
3. The MLP is faster, measured cold (each question from scratch, nothing cached).

**Where they stand on 14 October:**
- **Claim 1 holds in development for the lean pair** (P3a). On our full corpora the lean MLP keeps at least 0.968 of
  the lean GNN's gain on R@5 and FC@5 on all six datasets; on hit@1 it keeps 0.73 to 0.84 on hotpotqa, 2wiki and
  webqsp. On the two leave-out fits read for this (zero-shot reads included), the two are within 0.015 R@5 on every
  read but one, and there the MLP is higher (webqsp zero-shot, 0.287 against 0.226; docs/ZN_LEAN_NO_MP.md). On
  HippoRAG 2's own benchmark the MLP keeps 0.97 to 0.99 of the GNN's gain. The new lean pair (zrckp, zspk, adopted
  the evening of 14 October) keeps 0.93 to 1.29 on R@5 and 0.98 to 1.35 on FC@5 on the J5 reads: point estimates,
  with intervals still to come. musique is the lowest.
- **The universal pair falls short of claim 1** on two datasets: twin0 keeps at least 0.9 of gnn0's R@5 gain on
  metaqa, hotpotqa and webqsp, but 0.852 on 2wiki and 0.596 on musique (on squad gnn0 has no gain to share). On FC@5
  it is also just short on metaqa (0.893) and hotpotqa (0.899).
- **Claim 2 holds on our corpora.** Both GNNs are at or above the best of eight LLM-free rankers on all six datasets
  (P3a); gnn0 ties it on squad. The eight are five published ranking rules rebuilt on our graphs, plus the dense,
  SPLADE and RRF lists; the bar is not HippoRAG 2 as released. On HippoRAG 2's own benchmark ours is above HippoRAG 2
  on 2wiki (97.2 against 90.4) and hotpotqa (98.2 against 96.3), and below on musique (66.3 against 74.7).
- **Claim 3 is not met as the user set it** ("overall, mlp should be 2-3x faster at the minimum from the fastest
  gnn"). No MLP is 2x faster than the fastest GNN (zsp) on any dataset. Across families, the lean MLP is 2.4x to 4x
  faster than the universal GNN on five datasets for the model stages (1.3x on webqsp), but the pool build now
  dominates hotpotqa and 2wiki, where every end-to-end ratio is near 1.
- **The pools are final.** KIND_T is one parameter-free rule for all six datasets (squad's question links aside,
  section 5). At today's pool sizes it holds every gold for more questions on four datasets, and is level on squad and
  hotpotqa. Against the declared SOTA pool bar it is at or above on four datasets and just below on metaqa and
  hotpotqa, so its filed verdict is NOT_ALL_SIX; against the like-for-like bar, webqsp is below too.
- **The models on the final pool (14 October).** Reading where KIND_T's walks put each candidate lifts both lean
  models: P1c3's full run is ADOPT, with no LOSS in 36 reads for either. The MLP's version broke the
  no-message-passing rule, so a pointwise fix (zrckp) replaced it. Its screen passed and its full run is ADOPT (no
  LOSS in 36 reads), so the lean pair is now (zrckp, zspk). Serving the new input costs 4 to 39 ms per question,
  mostly walks (P4c).

## 2. Words used throughout

**Data and measures**
- **Gold:** a correct candidate. **Pool:** the shortlist a model reranks; a gold outside it is never found.
- **R@5:** the share of a question's golds in the top 5. **FC@5:** 1 only if every gold is in the top 5 (GraphER's
  PR@5 is this measure). **hit@1:** the top result is a gold.
- **ALL:** the share of questions with every gold inside the pool. **Gold recall:** the share of golds in the pool.
- **First stage (RRF):** the dense and SPLADE lists fused by reciprocal rank (a node scores the sum of 1/(60 + rank)).
- **Seeds:** the dense top 5 and the SPLADE top 5.

**Graph**
- **Structural links:** KB facts (metaqa, webqsp), Wikipedia hyperlinks (2wiki, hotpotqa), title mentions (musique,
  squad). **ner links:** passages sharing a named entity. **kNN links:** nearest neighbours by embedding.
- **Walk:** personalised PageRank, a random walk that keeps jumping back to its start nodes. HippoRAG 2 starts it from
  the seeds plus the first stage's top 1,000 (called A1 here).

**Models**
- **Universal pair:** twin0 (MLP) and gnn0 (GNN). Same 197 input columns; the GNN adds three relation-aware
  message-passing steps plus an evidence-flow step.
- **Lean pair:** zrc (MLP) and zsp (GNN). Both grow from zrm (zret's normalisation plus the learned relation match):
  zrc is zrm keeping each row's 64 best chain entries; zsp is zrm plus one propagation step over the pool's links
  (385 weights).
- **Twin:** an MLP trained on exactly the GNN's inputs, rows and loss. GNN minus twin is what message passing adds.
- **MLP share:** (MLP − first stage) / (GNN − first stage). 0.9 means the MLP keeps 90% of the GNN's gain.

**Training and decisions**
- **J5:** a fit trained on five datasets (webqsp never trains). **L-x:** a fit that leaves dataset x out; reading x on
  it is **zero-shot**.
- **Screen:** a quick test of one idea: two leave-out fits, read on all six datasets (12 reads). A **full run** (36
  reads) follows only if the screen passes.
- **Seed null:** the unchanged model retrained with two other seeds. Each read's floor is max(0.0075, twice what
  reseeding alone moves it). **GAIN** or **LOSS** means a change past the floor with its 95% interval clear of 0;
  **WITHIN** otherwise. One LOSS anywhere blocks adoption.
- **Declared:** every stage's rule is written and committed before any of its numbers exist.

## 3. Timeline

### Era 1, 25 Aug to 2 Sep: the first question, on a small copy of the data

**What we tried.**
- Offset MLPs (score a candidate by how close it is to "top hit + a learned shift"), against GNN operators
  (docs/OFFSET_SCREEN_RESULTS.md).
- A six-dataset study of what message passing is worth (docs/SIX_DATASET_RESULTS.md).
- An MLP given fixed structure summaries (distances, path counts, PageRank) and the retrieval seeds, first called
  SA-MLP and renamed QLS-MLP (docs/SA_MLP_CONFIRMATION_RESULTS.md, docs/TERMINOLOGY_AND_POSITIONING.md).
- Packages A to E: rank controls, edge types, pool budgets, timing, all frozen before their numbers.

**Problems and fixes.**
- The QLS-MLP screen passed 3 of 3, but only the MLP had seen the seeds. **Fix:** a GNN that also sees the seeds. Once
  both had the same inputs the two were about level (2wiki 68.40 against 69.85; webqsp 33.37 against 33.09). The
  preregistered non-inferiority gate passed 2 of 3, and the claim that the MLP beats the GNN stayed prohibited.
- Package D timed both models "uncached" (everything rebuilt per question). The MLP was slower in 10 of 12
  dataset-batch cells (1.13 to 1.39x at one question at a time): its whole build prefix (candidates, seeds, topology
  and its structure summaries) cost 2.2 to 3.0 ms against the GNN's 0.8 to 1.6 (docs/EXPERIMENT_EXECUTION_STATUS.md).
  Warm, its operator was 4 to 8x cheaper.
- Package B's sealed graph had duplicated edges. A de-duplicated control changed little (docs/EDGE_PROVENANCE_RESULTS.md).

**Results.**
- Offsets sharpen rank 1 (2wiki R@1 +12.5) but lose coverage when a question has several answers (R@20 −10.1).
- No model wins everywhere: the MLP wins 2wiki (+2.08) and musique (+1.10); the GNN wins webqsp, hotpotqa and metaqa
  (−4.05, −4.64, −7.71).
- Edges that encode real relations beat kNN edges on five of six datasets (only narrowly on musique), and clearly on
  the knowledge bases (metaqa 30.4 against 23.7). On squad kNN was best, by less than 0.1 point (89.33).
- Longer first-stage lists do not fix the knowledge bases: at a budget of 400 candidates (372.5 and 343.2 on average),
  the pool holds only 33.5% (metaqa) and 49.1% (webqsp) of a question's golds (PoolCov,
  docs/CANDIDATE_BUDGET_RESULTS.md).

**What made the difference:** giving both models the same inputs. It turned "MLP against GNN" into a fair question.

### Era 2, 2 to 8 Sep: what the graph can reach, and one frozen feature model

**What we tried.**
- A graph audit (docs/GRAPH_SUBSTRATE_AUDIT_RESULTS.md): from the seeds, which edge types reach the golds?
- M0: can expanding the pool along the graph add golds (docs/M0A_PROBE_RESULTS.md to docs/M0C_BOUNDED_R3_RESULTS.md)?
- M1 and M2: a feature screen across datasets, then one frozen 14-column model for all six (3,585 parameters,
  semantic rung S3 chosen over a 449-parameter S2 and a 205,217-parameter S4).

**Results.**
- **The founding fact:** "kNN edges supply local density, structural edges supply reach." A path to the gold exists
  through kNN for 0.515 of metaqa's golds and 0.338 of webqsp's; through structural edges for all of them.
- Structural expansion raised the reachable R@5 by +15 points on 2wiki and +35 on metaqa; kNN expansion recovered no
  gold. A capped expansion recovered the same golds at about 1x the context, against up to 56x uncapped.
- The frozen model advanced (+1.78 R@5 macro), but webqsp carried most of it (+9.88; without it +0.155).
- The bigger S4 head lost to S3 on musique at every repair (M2C, M2D: STOP).

**What made the difference:** knowing that reach comes from structural links, and that a cap on expansion is free.

### Era 3, 8 to 19 Sep: the full corpora and "today's pools"

**What we tried.**
- **M3A:** move from subsets to the whole corpora: about 20 thousand to 6 million nodes per dataset (2wiki 5.99
  million passages, hotpotqa 5.23 million, webqsp 2.59 million entities). No pool ships with them; the dense list is
  exact over the whole corpus (docs/M3A_RETRIEVAL_BUDGET.md, docs/M3A_HEADROOM_REPORT.md).
- **M3B:** freeze one pool per dataset ("today's pools"), then run the same inputs through first stage → MLP → GNN
  without message passing → GNN. The last step minus the one before is message passing's own effect, δ_MP
  (docs/M3B_RESULTS.md).

**Problems and fixes.**
- Only 0.5255 of webqsp's gold references exist in its corpus. The 46 questions with no resolvable gold are excluded
  (1,503 scored). This caveat stays on every webqsp number.
- One training epoch took 8.9 hours on a paging machine and tripped the 8-hour stall detector; the run resumed from its
  epoch-4 checkpoint.
- KB nodes changed from "name plus facts" to "name only" text, so KB numbers before and after are not comparable.

**Results.**
- Today's pools: the knee rule takes, per dataset, the smallest pool reaching 0.90 of the attainable top-5 recall
  ceiling (recall_ceiling@5). On metaqa and webqsp it was read within the link family the plan named. So metaqa =
  RRF top 50 plus 3 structural hops (2,017 rows; the rule as filed picked RRF top 200 plus 2 hops on all families,
  0.902 at 1,290 rows), hotpotqa = RRF top 50 plus 1 structural hop (94 rows), webqsp = dense top 200 plus 3 hops on
  all families (2,120 rows; no setting reached the knee, best 0.863).
- δ_MP on R@5: 2wiki +0.052, hotpotqa +0.030, musique +0.018, read against published systems. metaqa (+0.147) and
  webqsp (+0.028) are not read: our GNN sits far below the published KB systems (NuTrea, ReaRev) on hit@1, by more than
  the measured exposure difference explains (metaqa 0.086 to 0.252 below, against a 0.002 shortfall; webqsp 0.169
  below, against 0.027). Message passing helps all the golds of multi-gold passage questions get in (FC@5 +0.126 on
  2wiki), not the first gold (hit@1 −0.021).

**What made the difference:** isolating message passing's own effect, on the real corpus sizes.

### Era 4, 19 Sep to 3 Oct: the universal pair, fast compile, and where the GNN's gain lives

**What we tried.**
- **Universal-v2:** one GNN for all six datasets, trained jointly with no dataset identity (420,932 parameters,
  docs/UNIVERSAL_GNN_SIX.md), and its twin MLP. Their seed 0 became **gnn0** and **twin0**, the universal pair.
- **MP-Approx** (levels L0 to L15): non-message-passing approximations of the GNN's gain (docs/MP_APPROX_L15.md).
- **The fast feature compile:** the per-question feature computation rewritten in numba, bit for bit identical.
- **Deploy CK:** compressed one-hop message-passing kernels for serving on 2wiki.
- Exploratory tracks: a lean, cost-aware MLP; a bounded query-dependent message-passing residual (QD); relation
  generalization.

**Problems and fixes.**
- The universal pilot's gate failed for both families at seed 0 (docs/UNIVERSAL_V2_PILOT.md): the GNN by a hair (squad
  0.8988 against 0.8996), the twin on 2wiki (0.8545 against 0.8578).
- A guard compared names case-sensitively and refused every fit; a restart killed a seed (resumed exactly); an old
  driver would have overwritten seed 0. All fixed before numbers.
- Bit-exact fast compiles must reproduce OpenBLAS's float32 summation order (numpy calls it), which changes with the
  BLAS thread count. Fix: build the references at the host's count (8 threads).
- A message-passing residual trained on one graph harmed unseen graphs (docs/DIAG_ZGF_HARM.md). Fix: bound it
  (0.2·tanh) and drop edge families in training. That was the first one-graph message-passing arm with no unseen graph
  below the twin on R@5 (docs/STATUS_2026_10_05.md).
- Unpinned timings swung widely on the laptop's hybrid CPU. Fix: the timing process pins itself to one fast core (its
  own process only, no system setting).

**Results.**
- The universal GNN beat M3B's universal GAT (three seeds) on metaqa (hit@1 +0.068), webqsp (hit@1 +0.037) and 2wiki
  (R@5 +0.005, FC@5 +0.011); squad, hotpotqa and musique were level.
- On passages, the GNN's gain over its twin is a one-hop, query-conditioned reweighting of neighbours: a probe (L3)
  recovers 0.799 of it on 2wiki. On metaqa it is typed chains of 2 to 3 hops: a typed-walk MLP recovered 0.929 of the
  gain over the twin (on metaqa's train-split rows), still 0.006 below the GNN.
- The fast compile is 3.9 to 16.2x faster (metaqa 877 → 54 ms per question).

**What made the difference:** seeing where the gain lives (one hop on passages, typed chains on KBs). That became
the learned relation match and zsp a week later.

### Era 5, 4 to 7 Oct: transfer without labels, and the musique diagnosis

**What we tried.**
- Train on metaqa's KB, read webqsp zero-shot, with no webqsp labels (stages S1 to S6).
- A diagnosis of why musique lags (docs/MUSIQUE_DIAGNOSIS.md).

**Results.**
- Label-free shift rules failed. Only text, or which relations exist, can tie a learned cluster to a relation.
- An EM-trained chain scorer with granularity transforms gave +6.43 R@5 over the untyped walk on webqsp zero-shot. The
  transforms are what transfers (+10.7), at a cost of 4.8 points in-domain.
- musique: message passing helps 2-hop questions only (δ_MP R@5 +0.046, FC@5 +0.101); on 3 and 4 hops FC@5 is 0.034
  and 0. The selection carve shared golds with training (0.945) while the dev carve shared none, so selection rewarded
  memory.
- **Correction:** GraphER's PR@5 is our FC@5 (0.224), not our R@5.

**Rules set from here to 9 October, and kept:**
- every fix changes training on all six datasets and is graded on all six, including zero-shot;
- the encoder stays frozen for good (no bigger encoder; from 8 October, no sentence-level embeddings either);
- no LLMs or transformers inside our models (7 to 9 October).

### Era 6, 7 to 9 Oct: fast screens, the seed null, and the lean pair

**What we tried.**
- **Step 1:** which carve to select on (D against M), NOT_ADOPTED; its fits (J5 and five leave-outs) became the
  screens' base (docs/STEP1_MATCHED_SELECTION.md).
- **One-fit screens:** the user's rule, one training run per idea (about 25 minutes instead of a day; docs/SCREENS.md).
- About 30 screen rounds, mostly on the MLP (rounds 22, 23 and 30 were on the GNN track). Among them: zret (z-score
  each feature block against the pool's retrieved rows), gsurg (gradient surgery across datasets), rmatch (a learned
  match between the question and the relations along typed chains), zrm (rmatch trained with zret's model), zrc (zrm
  keeping each row's 64 best chain entries), zsp (zrm plus one propagation step).
- **Pools:** step 4 (walks for reach), 4b (each dataset's own graph), 4d (keep today's pool, add walk nodes), 4e (the
  papers' way: HippoRAG 2's restart and exact name linking), 4f (can the pool shrink?), 4g (where the misses are).

**Problems and fixes.**
- **Screens passed and full runs failed.** Both PROMISING one-fit screens (bdrop20, zret) failed their full runs, on
  reads a single screen split cannot see. Fix: two fits per screen.
- **Every full run died on a LOSS of 0.008 to 0.014**, and four different ideas all "lost" hotpotqa zero-shot. Fix:
  the seed null. 17 of 36 floors turned out to be above 0.0075; musique zero-shot's is 0.072. A change smaller than
  reseeding is not counted.
- **Message passing crept into an MLP screen.** zlk and zsp read their pool neighbours' scores. Fix (8 Oct): they are
  named "zrm + one propagation step (message passing)", a GNN track; the MLP's inputs are computed before any row is
  scored and never from another row's score.
- Step 4's one walk on a union of all link families diluted the structural hop (2wiki ALL −0.082). Fix: walk each
  dataset's own graph (4b).

**Results.**
- **zret:** musique zero-shot 0.270 → 0.390. **rmatch:** the largest gains on both typed graphs of any arm screened
  so far: metaqa R@5 0.644 → 0.779, webqsp zero-shot 0.117 → 0.280 (learned on metaqa's relations, carried to webqsp's
  unseen ones through frozen relation-name embeddings). It was blocked by a 2wiki zero-shot LOSS (−0.0136), then
  adopted as zrm, trained jointly.
- **zrc** became the MLP base (webqsp zero-shot +0.026 to +0.051, no LOSS); **zsp** the GNN base (metaqa zero-shot
  +0.030, 2wiki zero-shot +0.014, no LOSS).
- **Pools:** step 4e's pools held every gold for 0.909 to 0.992 of questions (webqsp 0.688 → 0.909, musique 0.803 →
  0.937), at about 3.5x the size on metaqa, musique and webqsp, about 2x on hotpotqa and 2wiki, and 3x on squad.
  Step 4f: they cannot shrink except on squad. Step 4g: 78 to 89% of missed golds sit one hop outside the pool on
  metaqa, musique and webqsp, and 30 to 42% on 2wiki; on hotpotqa and squad the misses are the first stage's. On 2wiki
  and hotpotqa every first hop into a missed gold is a hyperlink.
- **Diagnoses:** missed golds sit next to found golds (D1). On musique such a gold ranks 5th to 9th among about 45
  neighbours at the median; on 2wiki and hotpotqa it ranks 1st or 2nd 0.57 to 0.71 of the time. A stronger scorer on
  the same inputs adds at most +0.006 (D2).

**What made the difference:** fast, honest decisions (screens with seed-null floors, rules before numbers), and the
learned relation match along typed chains, the largest lever on KB zero-shot.

### Era 7, 9 to 13 Oct: the claims, HippoRAG 2's benchmark, hyperlinks, and cold speed

**What we tried.**
- **B1:** HippoRAG 2's own three settings (1,000 questions each, no re-encoding), to calibrate against a published
  system (docs/B1_HIPPORAG2_ALIGNMENT.md, docs/B1B_MODELS_ON_HIPPORAG2.md).
- **U1a to U1f:** can a universal, LLM-free link rule replace the Wikipedia hyperlinks? (A passage links to any
  passage whose title it mentions; then precise and damped versions.)
- **C1 to C12:** cold timing, then the fastest exact form of every model (same weights, same scores, often bit for
  bit), one rule for both families (docs/C1_COLD_COST.md to docs/C12_SHARED_COMPILE.md).
- **F1:** which input families each model really uses, to time both on one shared input set (docs/F1_ONE_INPUT_SET.md).

**Problems and fixes.**
- The "6-9x faster" MLP figure was warm and timed the forward only. C1 timed the whole path cold: "Claim 3, as the
  models are served today, does not hold cold." At batch 16 the forward alone was 7 to 20x cheaper; the shared feature
  compile and zrc's walk inputs ate the lead.
- C2 sped up only the MLP. Fix (C3): one exactness rule for both families. musique's ratio fell from 4.89 to 3.39 once
  the GNN got the same treatment.
- Hub nodes: one 104-node pool scanned about 84,600 stored entries. Fix: a galloping merge, for both families (C4).
- webqsp's chain inputs (87,299 entries per question) made the MLP slower than the GNN. Fix: an exact numba chain walk
  (C6), webqsp 0.84x → 1.33x against gnn0.
- The universal link rule linked common words that are also titles ("From", "County"): precision 0.10 to 0.14, 330
  million edges on webqsp. Fix: case-sensitive titles with a capitalization test, then removing mention links into
  nodes with more than 1,024 of them.
- The laptop's load was read at 74% CPU after C9 ran (C9's ratios stood), and C10's checks page-faulted while timing.
  Fix: check the load first; time and check in separate passes.

**Results.**
- **B1** (recall@5 in percent): zsp is above HippoRAG 2 on 2wiki (97.2 against 90.4) and hotpotqa (98.2 against
  96.3), below on musique (66.3 against 74.7). zrc keeps 0.97 to 0.99 of zsp's gain. musique's pools hold 98.2% of its
  golds, so its gap is ranking.
- **Hyperlinks matter.** Without them zsp falls to 77.3 (2wiki) and 90.0 (hotpotqa). Our title rule wins back 95.9
  and 95.0: 94% of the hyperlinks' gain on 2wiki, 61% on hotpotqa. Retraining on the universal link graphs was
  NOT_ADOPTED twice (U1c, U1f): dropping the hyperlinks costs hotpotqa 0.035 to 0.065 R@5 (as a fraction, on our
  corpora). Both models keep them.
- **Speed (C12, cold, model stages):** gnn0/twin0 1.46 to 1.89; zsp/zrc 0.96 to 1.18; gnn0/zrc 2.38 to 3.99 on five
  datasets, 1.30 on webqsp. F1: both models use every input family, so no input can be cut.
- Later rounds (deeper message passing, longer training, feature selection) did not beat their bases.

**What made the difference:** honest timing (cold, pinned, one thread, both families at their fastest exact form),
and calibrating against a published system on its own data.

### Era 8, 13 to 14 Oct: the full corpora, the final pool, and the gap ledger

**The direction (the user, 13 October):** goals on our own full corpora; on all six datasets beat the SOTA pool first,
then the SOTA models; diagnose the gaps. At about 10:40 the user set the rule for pools and links: "dataset agnostic
and parameter free and should have the most golds in least nodes". No learning until the pools are final.

**What we tried.**
- **P3a:** both GNNs against the best of eight LLM-free rankers: five published ranking rules rebuilt on our graphs,
  plus the dense, SPLADE and RRF lists (docs/P3A_SOTA_RANKERS.md).
- **P3b:** a gap ledger: where the bar beats the GNN, and where the GNN beats the MLP (docs/P3B_GAP_LEDGER.md).
- **P1a to P1e:** pool orders against a SOTA bar (the best LLM-free reproduction of a published pool order at each
  size): learned gates (MX, RC), then fixed fusions (RRF5, then KIND_T) (docs/P1A_POOL_SCOREBOARD.md to
  docs/P1E_ANCHOR_KIND.md).
- **P2a to P2c:** links by family; our parser's triples against LLM-built graphs (docs/P2C_LLM_FREE_TRIPLES.md).
- **P5a to P5b3:** learned node spaces and learned offsets g(question, seed, graph features) for reranking.

**Problems and fixes.**
- Our linked names diluted the head of A3's walk on passages (on hotpotqa, the walk from the linked names alone holds
  0.617 at 100 nodes, the walk from the seeds alone 0.947), and the first-stage lists wasted half the walk's restart
  mass on metaqa (P1a). Fix: trust a link only when the first stage also retrieved it ("corroborated").
- The learned gates were fit to golds: MX to where the golds' walk mass sat, RC to their ranks. Under the 10:40 rule
  they became reference lines only, and the models' retrain on MX's pools was held before any read.
- A diagnostic named two keys twice, halving some of its values. This was found after filing and corrected in the doc.
- The learned offsets memorized their training questions (P5b). With a held-out stop the gains held on new golds
  (P5b2), but musique stayed near chance, and a back-off gate still lost musique (P5b3). The track closed.

**Results.**
- **P3a: both GNNs at or above the bar on all six (SOTA_GNN)**; gnn0 ties it on squad (R@5 −0.0015). R@5, bar → zrc /
  zsp: metaqa 0.241 → 0.780 / 0.779; squad 0.905 → 0.911 / 0.910; musique 0.473 → 0.567 / 0.566; hotpotqa 0.685 →
  0.905 / 0.905; 2wiki 0.608 → 0.870 / 0.873; webqsp (zero-shot) 0.110 → 0.307 / 0.268.
- **P3b:** where the bar beats the GNN, the gold was in our pool and ranked out. Where the GNN beats the MLP (hotpotqa,
  2wiki), the gold is one link from a found gold: worth 0.009 to 0.016 R@5 for the lean pair, and 0.034 to 0.051 for
  the universal pair.
- **KIND_T, the final pool** (P1e). Entity questions (all start nodes are bare names) walk from their corroborated
  links on structural edges, backfilled by HippoRAG 2's walk. Passage questions take RRF5T, a fixed fusion of the
  first-stage order and four walks (HippoRAG 2's A1 walk and our KT walk, each on the structural links and on all
  link families); only the KT walks are steered by first-stage rank. Nothing is fit. ALL at today's sizes:

  | dataset | today's pools | KIND_T |
  | --- | ---: | ---: |
  | metaqa | 0.895 | 0.962 |
  | squad | 0.980 | 0.979 |
  | musique | 0.803 | 0.900 |
  | hotpotqa | 0.968 | 0.965 |
  | 2wiki | 0.906 | 0.960 |
  | webqsp | 0.688 | 0.832 |

  It reaches today's coverage with fewer nodes: about 37% of them on metaqa, a quarter on musique and webqsp, and half
  on 2wiki. squad's pool cannot change (its fill is zero), and on hotpotqa it needs 150 rows where today's pools use 94.
  Its filed verdict against the SOTA pool orders is NOT_ALL_SIX: at or above on squad, musique, 2wiki and webqsp, just
  below on metaqa (up to 0.005) and hotpotqa (up to 0.007). Against the like-for-like bar (beside), webqsp is below
  too, at all 30 points (by up to 0.051).
- **Links:** the structural family carries the reach (metaqa +0.540 ALL, 2wiki +0.310); no family can be dropped. Our
  rule-based triples beat LLM-built graphs on B1's musique (+8.4 R@5 against +3.7 to +4.8 as a pool walk) but a
  fourth link family gave NO_GAIN on our corpora. The links stay three families.

**What made the difference:** fixed fusion with nothing fit, trusting links only when the first stage corroborates
them, and treating entity and passage questions differently by a data test, not by dataset name.

### Era 9, 14 Oct: the models on the final pool, and its cost

**What we tried.**
- **P1c2:** retrain both lean models on KIND_T's pools (docs/P1C2_POOLS_KIND_T.md).
- **P1c3:** give both models eight columns per candidate, read by a small head: six positions (where KIND_T's order
  and its walks put the candidate) and two flags (whether the walk's fill added it; whether it is an entity)
  (docs/P1C3_KT_INPUT.md). Then a full run (docs/P1C3_FULL_RUN.md).
- **P4a, P4b, P4c:** time the pool build cold, make its walks faster, and time the new input's serving cost
  (docs/P4A_POOL_COLD.md, docs/P4B_FAST_WALKS.md, docs/P4C_KT_SERVING.md).
- **P1c3p:** a pointwise version of the MLP's head (docs/P1C3P_POINTWISE_HEAD.md), then its full run
  (docs/P1C3P_FULL_RUN.md).

**Problems and fixes.**
- P1c2 was NOT_ADOPTED on one LOSS (zsp, webqsp zero-shot −0.0167), although the pools gave real gains (metaqa +0.020
  to +0.021, 2wiki +0.039 to +0.045 on both models). On musique and webqsp the extra golds were ranked out. Fix: show
  the models the walk positions (P1c3).
- The pool build is slow: 0.19 ms (squad) to 286 ms (hotpotqa) per question at the median. Where a walk runs (all but
  squad), walks are 88 to 97% of it, bound by memory latency. On hotpotqa and 2wiki the build is 97 to 99% of every
  model's time, so every speed ratio there falls to about 1. Fix (P4b): the same walks with a memory-friendly data
  layout, 1.14x (metaqa) to 3.75x (2wiki) faster, every walk equal.
- **The MLP's new head broke the MLP rule.** zrck's head also read the row's score z-scored over its question's rows:
  the mean and spread of the other rows' scores. No edges, but other rows' scores, which the 8 October rule forbids.
  Fix: zrck serves no claim; zrckp's head reads only the row's own eight columns (161 parameters, no score input).

**Results.**
- **P1c3's screen:** 6 GAINs and no LOSS for each model (metaqa +0.021 to +0.026, 2wiki +0.040 to +0.066, webqsp
  zero-shot up to +0.082). musique zero-shot wins back 0.6 to 0.7 of what the pools cost. Beside: hotpotqa hit@1 falls
  0.008 to 0.043.
- **P1c3's full run: ADOPT.** Neither model has a LOSS in its 36 reads, and each has 5 of 11 primary GAINs (zrck 18
  GAINs in all, zspk 16). 2wiki rises +0.040 to +0.066 on every split, and metaqa on every split; webqsp zero-shot
  rises on six splits for zrck and four for zspk. Beside: hotpotqa's held-out hit@1 falls about 0.04 for both, and
  musique's hit@1 rises on all six splits.
- **P1c3p's screen: ADOPT_FOR_FULL_RUN** (7 GAINs, no LOSS). zrckp is level with zrck on ten of twelve reads, above
  it on 2wiki in one fit (+0.022), and keeps 0.65 and 0.74 of zrck's webqsp zero-shot gain.
- **P1c3p's full run: ADOPT.** zrckp against zrc: no LOSS in 36 reads, 5 of 11 primary GAINs, 20 GAINs in all.
  2wiki +0.060 to +0.064 on every split, metaqa +0.108 when held out (0.142 to 0.249), webqsp zero-shot up on all six
  splits (+0.014 to +0.066). Beside: hotpotqa's hit@1 falls on all six splits (−0.002 to −0.034). The lean pair is now
  (zrckp, zspk). On the J5 reads zrckp keeps 0.93 to 1.29 of zspk's R@5 gain over RRF (point estimates).
- **P4b:** hotpotqa's build 171 → 48 ms, 2wiki's 188 → 50 ms (the old form re-timed in P4b's session; P4a filed 286
  and 199 ms). Still about 20x zrc's whole time on those two datasets.
- **P4c: TIMED.** The KT input adds 4 to 7 ms per question where KIND_T already walks (musique, hotpotqa, 2wiki), and
  22 to 39 ms where it does not (squad, webqsp, metaqa). Both lean models pay it, so their ratio stays 0.98 to 1.08.
  The first timing ran while this session was busy, so all six were re-timed idle, by a rule committed first; the two
  agree within 3.3%.

## 4. By topic

### Pools: from fixed hops to one parameter-free rule

| when | pool | what changed | ALL (questions with every gold in the pool) |
| --- | --- | --- | --- |
| Aug | dense 200 + SPLADE 200 | inherited | not measured as ALL; PoolCov (golds per question in the pool, test split, filed 1 Sep): metaqa 0.335, webqsp 0.491 |
| 13 Sep | today's pools (M3B) | per-dataset hops on each dataset's own regime, knee rule | 0.688 (webqsp) to 0.980 (squad) |
| 9 Oct | step 4e | HippoRAG 2's restart walk + exact name linking | 0.909 to 0.992, at about 3.5x the size on metaqa, musique and webqsp (2x to 3x on the others) |
| 13 Oct | KIND_T (P1e) | one fixed rule: corroborated links, entity/passage split, steered walks | metaqa 0.962, squad 0.979, musique 0.900, hotpotqa 0.965, 2wiki 0.960, webqsp 0.832, at today's sizes |

What made the difference: structural links for reach; walks instead of fixed hops; two ideas from the papers
(HippoRAG 2's restart, PullNet-style name linking); and a fixed fusion fit to nothing. Against the learned gates
(P1e, at today's sizes) it is ahead of every one only on musique and 2wiki; it is level with RC on metaqa, within
0.002 of them on squad, below MXU on webqsp (0.832 against 0.857) and below all of them on hotpotqa. On today's pools, golds outside the pool cost 0.02 to 0.13 of R@5 (0.198 on musique's 4-hop
questions); ranking is the larger gap.

### Links: keep the structural family, and the hyperlinks

- Structural links carry the reach (Era 2, P2a); kNN links add density, not reach.
- Mixing link families blindly diluted the structural hop three times (step 4, G1a, P2a).
- An LLM-free title rule rebuilds most hyperlinks (P 0.76 / R 0.81 on 2wiki) and keeps 94% (2wiki) and 61%
  (hotpotqa) of their gain on B1. Retrained models did not keep their value, so the hyperlinks stay.
- Our rule-based triples (spaCy plus fixed rules) match or beat LLM-built graphs on B1 (7 of 8 cells MATCHED), but
  add nothing on our full corpora.
- The words around a link do not say which neighbour a musique question follows (D4, NO_SIGNAL).

### Models: what each one is, and why

| model | family | what it is | status |
| --- | --- | --- | --- |
| twin0 / gnn0 | universal | 197 shared columns; the GNN adds three relation-aware message-passing steps plus an evidence-flow step | gnn0 trained 28 Sep, twin0 1 Oct; named a pair 12 Oct (C9) |
| zrc | lean MLP | zret's normalisation + the learned relation match + 64 chain entries per typed row | 9 Oct |
| zsp | lean GNN | zrm's inputs + one propagation step over the pool's links (385 weights) | 9 Oct |
| zrck / zspk | lean, with KT | + KIND_T's eight walk-position columns through a small head | full run ADOPT 14 Oct; zrck serves no claim (its head reads other rows' scores); zspk is the lean GNN since then |
| zrckp | lean MLP | zrc + a pointwise KT head (the row's own 8 columns only, 161 parameters) | full run ADOPT 14 Oct; the lean MLP since then, paired with zspk |

What paid: fair inputs, structure computed before scoring (without edge-derived inputs the MLP keeps only 13 to 73%
of the gain, ZN), the learned relation match, normalising against retrieved rows, one small dose of message passing,
and the final pool's walk positions. What did not pay: bigger heads, longer training, feature selection, learned
spaces and offsets.

### Generalization: zero-shot reads

R@5 of zrc on today's pools (outputs/zbase2/base-zrc-*.md), trained on the dataset (J5) against held out (L-x):

| dataset | trained on it | held out | first stage alone |
| --- | ---: | ---: | ---: |
| metaqa | 0.780 | 0.142 | 0.005 |
| squad | 0.911 | 0.891 | 0.905 |
| musique | 0.567 | 0.524 | 0.473 |
| hotpotqa | 0.905 | 0.845 | 0.685 |
| 2wiki | 0.870 | 0.794 | 0.608 |
| webqsp | never trains | 0.100 to 0.338 | 0.054 |

- The three Wikipedia-style corpora keep most of their lead held out.
- **metaqa held out collapses** (0.780 → 0.142): with no KB in training, no relation match is learned. webqsp shows
  the same: 0.29 to 0.34 on five fits, but 0.100 on L-metaqa's, which has no KB in training.
- **squad held out lands just below the first stage** (0.891 against 0.905).
- **webqsp** stays below the universal pair, which trains on it.
- Unbounded message passing over-fits its training graph (Era 4).

### Speed: cold, pinned, one thread, both families at their fastest exact form

Ratios are GNN time ÷ MLP time at the median (above 1 = the MLP is faster). Model stages on today's pools (C12) →
end to end with KIND_T's build (P4b; the end-to-end column mixes two timing sessions):

| dataset | gnn0 / twin0 | zsp / zrc | gnn0 / zrc |
| --- | --- | --- | --- |
| squad | 1.59 → 1.59 | 1.05 → 1.06 | 2.57 → 2.55 |
| metaqa | 1.57 → 1.50 | 1.06 → 1.05 | 2.93 → 2.58 |
| musique | 1.89 → 1.62 | 1.18 → 1.09 | 3.99 → 2.51 |
| webqsp | 1.54 → 1.45 | 0.96 → 0.96 | 1.30 → 1.24 |
| hotpotqa | 1.50 → 1.04 | 1.05 → 1.00 | 2.49 → 1.08 |
| 2wiki | 1.46 → 1.03 | 1.04 → 1.00 | 2.38 → 1.06 |

- **The lean pair cannot reach 2x on five datasets, by design:** zsp is zrc plus 3 to 17% of its time. Even with
  every shared stage free it would be 1.16 to 1.34 (2.12 on musique).
- **The universal pair is bounded by the shared feature compile** (58 to 71% of twin0's time). 2x would need a compile
  3 to 3.7x faster on squad, metaqa and webqsp, 1.18x faster on musique, and about 8 to 9x faster on 2wiki and
  hotpotqa, where even a free compile reaches only about 2.17. End to end, those two sit near 1.03 to 1.04.
- **KIND_T's build dominates hotpotqa and 2wiki** (about 50 ms against 2 to 6 ms for the models).
- Cutting inputs was rejected (it loses the wins). C12's compile kernels helped a little (1.02 to 1.10x) and are
  inside the timed forms.
- The KT input adds walks the build skips (P4c): 4 to 7 ms per question on musique, hotpotqa and 2wiki, 22 to 39 ms on
  squad, webqsp and metaqa. Both lean models pay it, so their ratio stays 0.98 to 1.08, and the KT MLP is slower than
  gnn0 on four datasets.

## 5. Honest audit

### Are we hardcoding anything per dataset?

**The ranking rule and the models: no.** KIND_T's rule is the same for all six datasets and reads no gold. The models
take no dataset ID, and one training config serves all. The lean models are read at their averaged final weights
(p@swa), with no checkpoint pick; gnn0 stopped on the macro average over all six datasets. Nothing is picked per
dataset.

**But three real per-dataset choices remain:**
1. **The pool's size and base (HIGH).** KIND_T keeps each question's size in today's pools: its fill is B_q =
   |I_q| − |base ∪ seeds|. Today's pools were set per dataset (50 rows on squad to 2,017 on metaqa, 2,092 on musique
   and 2,120 on webqsp, from configs/m3b_controlled_comparison.yaml) by a knee rule measured on the evaluation
   questions' golds. On metaqa and webqsp it was read within named link families, and webqsp's pool is the rule's
   fallback below the knee. So "nothing is chosen per dataset" is true of the ranking rule, not of the pool's size and
   base. On squad the fill is zero, so KIND_T cannot change its pool.
2. **squad's question links are switched off by its name (MEDIUM)** (scripts/step4e_paper_pools.py), because its
   titles are article titles, not passage titles. KIND_T and the KT input inherit it.
3. **"Is this a typed KB" is a list of two names (MEDIUM):** TYPED = ("metaqa", "webqsp") in
   outputs/mp_unified/rmatch.py and the serving forms. The feature compile already works it out from the data
   (outputs/mp_unified/look_x_six.py). A seventh typed KB would silently get no chain inputs.

Smaller ones: the lean MLP's input set was first chosen by a greedy search on 2wiki (INFO; F1 later showed both models
use every family); the evaluation carves differ per dataset (musique's fit carve is group-disjoint).

### Would it work zero-shot on a new dataset?

**The models would; the pipeline needs edits first.** A seventh dataset needs: its package (nodes, queries, frozen
dense vectors, SPLADE, the three link families); a pool contract entry, which today needs gold labels to size; edits
to closed six-name lists and the carve rules; and a TYPED entry if it is a typed KB. Then KIND_T, the features and the
models run unchanged.

**Measured zero-shot quality** is in section 4: strong on the Wikipedia-style corpora, weak on KB data held out
(metaqa 0.142), just below the first stage on squad, and on webqsp below the universal pair, which trains on it.
**The universal pair has never been tested on a held-out dataset.**

### Are we copying anything?

**No.** There are no third-party licence headers or "adapted from" notes in our code; every "copied from" is
internal. Published methods (HippoRAG 2's restart, PullNet/GraftNet-style linking) are reimplemented and named in our
docs; any paper must cite them. numba, llvmlite and spaCy are runtime libraries, unmodified and not committed.
Third-party data (HippoRAG 2's benchmark files, HippoRAG's OpenIE output, SiReRAG's graphs) is used only to compare,
and never committed. No LLM-built graph is in the system.

**Leakage:** the pool contract was chosen on the evaluation population's golds (MEDIUM, item 1 above). Step 4e's
choice included webqsp's eval carve, and KIND_T's selection carve was matched to the eval mix by aggregate counts
(both LOW). **Test splits:** the current compile (m3b_compile) reads only the evaluation splits, and the older tracks
refuse a test split explicitly. One older headroom tool (m3a_headroom's lane rows) loads every legacy split, so it must
never feed a test read.

## 6. Open issues and the fixes we propose

All fixes are universal: the same change on all six datasets, graded on all six including zero-shot, declared before
their numbers.

1. **One pool size rule for all datasets, chosen without eval golds.** Replace today's per-dataset sizes with one rule
   set on training carves only (for example, one node budget per question kind, or a stop on KIND_T's own order). It
   needs a pool rebuild and retrains, so it is its own declared stage.
2. **Replace squad's name switch with a data test**, for example link a title only when it names exactly one node.
3. **Derive "typed KB" from the data** everywhere, as the feature compile already does.
4. **A zero-shot stage for the universal pair** (leave one dataset out), as the lean pair has.
5. **The MLP's input rule (done):** zrck serves no claim. zrckp's full run is ADOPT, so the lean pair is
   (zrckp, zspk), with its serving cost timed (P4c). Still open: Claim 1's share for the new pair with intervals.
6. **P1c3's side cost:** hotpotqa hit@1 falls with the KT input. A small term scoring each gold against non-golds is
   the lever we have: zsep raised hit@1 by 0.012 to 0.022 in an earlier round, but its re-call was NO_GAIN with a
   musique LOSS (−0.0101), so it needs a fresh screen.
7. **Stages not adopted that still had gains** (the ledger, docs listed per item): zck trained alone on a frozen base,
   so the datasets it never acts on stay unchanged (webqsp zero-shot +0.016 and +0.040; its only LOSS came through
   shared weights); RRF5's position as one more KT column (KIND_T trails RRF5 on webqsp by 0.044).
8. **musique's ranking gap** (3 and 4 hops; the first stage misses far hops; the encoder stays frozen).
9. **KB zero-shot:** metaqa held out collapses (0.142), and webqsp stays below the universal pair, which trains on it.
10. **Claim 3:** within-family 2x is out of reach by design for the lean pair on five datasets, and bounded by the
    shared compile for the universal pair; KIND_T's build dominates two datasets. Multi-threaded walks and the claim's
    wording are the user's calls.
11. **One confirmation run** of the final models, pools and timings in one session.

## 7. Where we stand, 14 October

- **Pools and links: done.** KIND_T is final, parameter-free, and the same rule on all six datasets except squad's
  question-link switch; three link families; hyperlinks kept.
- **Models:** the lean pair is (zrckp, zspk) on KIND_T's pools with the KT input, since zrckp's full run adopted
  (14 October, 20:56). The universal pair is twin0/gnn0 on today's pools.
- **Claims:** 1 holds in development for the lean pair, on R@5 and FC@5 (the new pair's J5 point estimates are 0.93
  to 1.35; the universal pair falls short on musique and 2wiki); 2 holds on our corpora and on B1's 2wiki and
  hotpotqa, not on B1's musique; 3 is not met as set.
- **Filed tonight:** P1c3's full run (ADOPT), P1c3p's screen (ADOPT_FOR_FULL_RUN) and full run (ADOPT), and P4c
  (TIMED, re-timed with the session idle).
- **Next:** Claim 1's share for the new pair with intervals, then the ledger's screens on it (hotpotqa's hit@1,
  RRF5's position as a ninth column, zck on a frozen base). Their results go into their own docs and into
  docs/WHERE_WE_STAND.md.
