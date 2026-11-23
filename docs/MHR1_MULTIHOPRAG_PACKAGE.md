# MHR1: MultiHop-RAG as the seventh dataset, packaged like the six and read on e3v (first stage and links; no training)

Declared 21 November 2026 (04:4x IST), before any MHR1 vector, score, link or grade exists. The user, 21 Nov 2026 (verbatim):
"we can add the last dataset also into the mix too, improve links, the vectors and try to push more". D84 (16 Nov) downloaded
the data and graded the rule-built NER store at article level with no encoder (FILED 9b97f96: CARRIED F0T OFF WIKIPEDIA); its
open items (K, W2_S, any retrieval number) waited on the encoder, which D101 (21 Nov) settled: e3v is the encoder, and the frozen
gte-Qwen2-1.5B stays as the comparison.

**Nothing trains, no model of record reads anything, no filed graph, pool, fit, record or CRAG file changes.** MHR1 builds a
package that the existing code reads the way it reads the six, encodes it, and reads it at the first stage and at the graph
level. The pools and the models' zero-shot reads follow in a later stage (section 8).

## 1. The data (D84's, pinned)

- MultiHop-RAG (Tang and Yang 2024), `yixuantt/MultiHopRAG` revision `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`, ODC-By, in
  `data/raw/multihoprag/` on the laptop (gitignored, never committed). The two files' sizes and sha256 are D84's (section 1 there)
  and `zd84.FILES`; the code refuses other bytes (D84's G1, reused).
- 609 news articles; 2,556 queries, of which the 301 null queries have no evidence and are dropped (D84). Kept: 2,255.

## 2. The package (a new folder; the CRAG package is not touched)

Root `outputs/mhr1/package/` (gitignored, never committed), dataset folder `multihoprag/`, in the CRAG datasets' layout:

- `nodes.jsonl`: one node per article, in the corpus file's order (so position i is D84's node i). Fields:
  - `node_id` = `multihoprag:` + the first 24 hex of sha256(text);
  - `text` = the article's `body` verbatim (the record's passage nodes keep the title out of `text`, too);
  - `title` verbatim;
  - `content_hash` (sha256 of text), `source`, `category`, `published_at`, `url`, `author` verbatim.
  - Node ids must be unique; a repeat stops the build.
- `queries/<split>.jsonl` and `queries/query_ids.json`: the kept queries. Fields:
  - `query_id` = `mhr_q` + the file position, 4 digits; `question` = `query` verbatim; `answer`; `question_type`;
  - `gold_node_ids` = the distinct articles the evidence items name, in their first order (D84's G2: by url, the title agreeing);
  - `evidence` = the items with their `node_id` and `fact`; `n_golds`.
- **Splits (the rule):** by question, stratified by `question_type` (types in sorted order). Within a type, the questions in file
  order are permuted by `numpy.random.default_rng(0)` (one generator, drawn type after type). The first floor(40%) go to
  `train`, the next floor(10%) to `dev`, the rest to `test`. Rows are `train`, `dev`, `test` concatenated; each split keeps
  file order.
  - Every read in MHR1 is zero-shot: no model of record has seen any MultiHop-RAG question. The headline reads use **all 2,255
    kept questions** (the benchmark reads them all); `test` is reported beside.
  - `train` and `dev` exist only so that a later in-domain fit, if one is declared, has a fixed carve that `test` never meets.
    None is declared here.
- `DATASET.json`: `RECORD` = `MHR1_PACKAGE` (not the CRAG record's name), the counts, every file's bytes and sha256, the layout,
  the split rule and the raw files' sha256. `outputs/mhr1/package/PACKAGE_FREEZE.json` names DATASET.json's sha256. The code
  refuses a package whose DATASET.json does not hash to it. It never carries or compares the CRAG `RECORD_SHA256`.
- `embeddings/dense/{docs,queries}`: the frozen encoder's vectors, as in the six (zenc's `vec_dir` reads the frozen arm from
  the package). `embeddings/splade/{docs,queries}`: the record's SPLADE. One shard each (fewer than 40,000 rows).
- e3v and its two parts live where zenc puts every encoder's vectors: `outputs/enc/{e3v,gtemodern,granitesm}/multihoprag/`.
- `retrieval_cache/{dense,splade}_top1000.npz`: every question's ranking of all 609 articles (609 < 1,000, so a row holds all
  609), dense from the frozen vectors.
- `graph/knn.npz`: K on the frozen vectors (the six's knn family is the frozen one); the rule families are in
  `outputs/mhr1/links/` (section 5).

**Gates of the build (each stops it):**
- **P1**: D84's G1 and G2 on the raw files, unchanged code (`zd84.load`, `zd84.golds_of`).
- **P2**: the package reproduces D84. F0T and CN_all built from the package's (title, text) through D84's code have D84's
  filed pair counts exactly (25,080 and 25,005, outputs/d84/grade.json), and the kept questions and gold sets are D84's.
- **P3**: every evidence item's `fact` (spaces and newlines removed) is inside its named article's text (the same removal).
  This is the benchmark's relevance test, so P3 says the article-level gold and the benchmark's test agree on the gold article.
- **P4**: the encode and package readers accept it. zenc's `query_texts` returns every row and its id, and `doc_chunks` yields
  609 texts in order (in-process, before any encode).

## 3. The encodes

| arm | what | where it runs | path |
|---|---|---|---|
| e3v | gte-modernbert-base and granite-embedding-small-english-r2 at section 17.2's pins, each through zenc3's encode path (GP, then zenc's `cmd_encode`, max length 8,192, no prompts), fused by zenc3v's `fuse` (w 0.5), FC exactly on **every** row (zenc4's `fc_ds`) | host GPU (parts), host CPU (fuse, FC) | outputs/enc/e3v/multihoprag/ |
| frozen | gte-Qwen2-1.5B-instruct, revision a9af15a6, through zenc's `load_model` and `encode_texts` (the G0-checked path: float16 on the GPU, its query instruction, max length 32,768, unit rows) | host GPU | the package's embeddings/dense |
| SPLADE | the record's contract (CRAG's host encoder): naver/splade-cocondenser-ensembledistil revision 49cf4c7b, AutoModelForMaskedLM float32, batches of 64 in input order, truncation at 256 tokens, log1p(relu(logits)) max-pooled over every position (padding included, as the record pools), CSR float32 | laptop CPU (the model is on the laptop only; the host has no copy and no download is made) | the package's embeddings/splade |

- The wrapper (`outputs/mp_unified/zmhr.py`) points zenc's `served_root` at this package in its own process only, and sends
  zenc3's GP record to `outputs/mhr1/gp/`, so E0N's GP records are not overwritten. No filed module is edited.
- **Windows.** An article longer than an encoder's window is cut at the window, as the record cuts. Each encode records how many
  were cut; the read reports the golds' share on cut articles. e3v's window is 8,192 tokens, frozen's 32,768, and SPLADE's 256
  (so SPLADE reads an article's lead only).
- The parts' shards stay (a few MB in all; FC can be rerun).

### 3.1 Amendment (21 Nov, 14:40 IST, before any vector): the three encodes move to the host CPU

- **Why.** None of the three GPU rows had been sent by the time the host restarted (10:33 IST). Since 13:03, another
  user's virtual machine has held 20.7 GB of the card's 24.6 GB and about 97% of its engines, and the GPU guard has
  yielded since then. E4's and E6's GPU rows are queued ahead of these rows. The host CPU is idle. These are small
  encodes (609 articles and 2,255 questions), so they run there.
- **What changes:**
  - the device is the CPU, with the weights in float32 (zenc's `load_model` on the CPU), for both e3v parts and the frozen
    arm. Rows are still stored as float16 unit rows;
  - the token budget per batch is 8,192 for all three encoders, so that one long article goes alone and attention memory
    stays bounded on the CPU;
  - GP is not run. It compares float16 against float32 on the GPU, and float32 is its reference. Its record in
    outputs/mhr1/gp/ says "not run (CPU float32)";
  - each run records its seconds and its peak working set, so later CPU asks are measured.
- **What does not change:** the models, revisions, windows (8,192 and 32,768), prompts, normalization, the fusion, FC on
  every row, the cut check, and every rule in sections 4 and 5.
- **The feed:** the three GPU rows are withdrawn, unsent. CPU rows go in under new names (`mhr1-cenc-gtemodern`,
  `mhr1-cenc-granitesm`, `mhr1-cfrozen`), and the fuse, FC and cut rows follow them (`mhr1-cvec`, `mhr1-cfc`,
  `mhr1-ccuts`).
- **Beside, not a rule:** the six datasets' e3v rows were encoded in float16 on the GPU. These are float32 on the CPU. GP
  bounds that difference on the six (minimum cosine at or above its bar).

## 4. The first stage (zero-shot; each arm alone and fused with SPLADE)

Exact float32 inner products of the stored float16 rows against all 609 articles, ranked by score and then position (zenc's
`search` with k = 609); SPLADE's ranking the same way over its sparse products; the record's RRF (`zenc.rrf_top`, 1/(60 + rank)
summed over the two lists).

**Measures, per arm (e3v, frozen) and for SPLADE alone:**
- **The record's:**
  - gold coverage at K: the mean over questions of (golds in the top K) / (golds), at K = 2, 4, 5, 10 and 20 (d5 is the
    record's headline);
  - `rrf5`: the same at 5 on the RRF list.
- **The benchmark's (its `retrieval_evaluate.py`, read 21 Nov; applied to articles in the place of its 256-token chunks):**
  - Hits@4 and Hits@10: the share of questions with at least one retrieved item among the top 4 (10) that contains a gold
    `fact` (spaces and newlines removed from both);
  - MAP@10 (its form): at each rank up to 10, the facts newly matched there divided by the rank, summed, divided by
    min(facts, 10);
  - MRR@10: 1 / the rank of the first item holding a fact.
  - These are **not** comparable with the paper's chunk-level numbers: 609 whole articles are a smaller and coarser haystack
    than its chunks.
- **e3v minus frozen** on each measure: the point gap and its paired 95% interval (2,000 resamples of the questions, seed 0),
  called ABOVE / LEVEL / BELOW by zap1.call.
  - Headline cells: d5 and rrf5 over all 2,255 kept questions.
  - Beside: every other measure; `test` alone; each question type; each gold count (2, 3, 4); the golds on cut articles.

**The rule (the first stage on this corpus):** **e3v HOLDS ON MULTIHOP-RAG** if e3v minus frozen is not BELOW on d5 and not
BELOW on rrf5 (all kept questions). Otherwise **e3v LOSES ON MULTIHOP-RAG**, the failed cells named. Either call closes only the
first stage on this corpus at article level. It does not reopen D101 (taken on the six by the user).

## 5. The links (graph level; the families D83 and D84 name, each labelled by its rule)

MultiHop-RAG's domain is R + K: news articles ship no structure, so S is absent (A1; the user's 10 Nov ruling).

| family | label (D83's vocabulary) | rule | code |
|---|---|---|---|
| F0 | `R.name_comention` | CN_all: two articles share a rule name (df 2 to 25) | zul1 via zd84.rule_forms (P2) |
| F0T minus F0 | `R.title_comention` | an article's own title run shared | the same |
| W2_S | `R.mention` | U2b's rule `new`, variant W2 (exact titles), form S (mention links plus same-title siblings); a choice among same-named targets by the arm's dense vectors | zu2b's `Surfaces`, `stat_pass`, `link_pass`, `sibling_pairs`, unedited |
| K | `K.knn` | the record's K: text groups, top 4 with self, self dropped, inherited by every position, undirected, the larger score kept | zenc1's `nn_top`, `out_positions`, `tie_rows` and zenc1b2's `scores_at`, `k_edges`, unedited |

- W2_S and K are built on each arm's vectors (e3v; frozen beside).
- **Stores, deduplicated R > K:** R = W2_S ∪ F0T. K's store holds K's pairs that R lacks. Every pair keeps every label whose rule
  made it (a bit set), read before the dedup. Files go to `outputs/mhr1/links/<arm>/`: one npz per family, the labelled store,
  and a json of the counts. Written: pairs per family, pairs per label, K's pairs already in R, and each store's size.
- **Forms graded:** F0T, W2_S, R, K, RK = R ∪ K, each arm's own.
- **Measures:** D84's, unchanged. GP share with its 95% interval, density, lift with POINTS, GS share, price (pairs per linked
  slot), by question type and gold count.
- **Differences** (zap1.boot_diff over questions, 2,000 resamples, seed 0, zap1.call):
  - R minus F0T (what W2_S adds);
  - RK minus R (what K adds), each arm;
  - RK_e3v minus RK_frozen and K_e3v minus K_frozen (the encoder at graph level).
- **The rule (links on this corpus):** **e3v's LINKS CARRY ON MULTIHOP-RAG** if RK_e3v minus RK_frozen is not BELOW on GP.
  Otherwise **NOT_CARRIED**, naming the cell. K_e3v minus K_frozen stands beside.
- These files are the inputs the unified link stage (UL2) reads for this corpus. It owns any further family (H's hub names,
  typed links) and the comparison across datasets.

### 5.1 Amendment (21 Nov, 14:55 IST, before any link number): the universal rule set

The user's ruling ("we need to have universal rules"): MultiHop-RAG's links are the same rule set as the other six, with the
same code and constants. Section 5 differed from that in three places, and this amendment brings it in line. Nothing in it
is tuned or switched for this corpus.

- **The layout of record on MultiHop-RAG** is the universal one (UL2, docs/UL2_LAYOUT_ON_E3V.md, sections 2 and 9.3): RK,
  with R = W2_S + F0T + H_5 (R.hub) + whichever Part B forms UL2's record carries, and K = e3v's kNN-3. Dedup R > K;
  every pair labelled by its rule.
- **Difference 1: W2_S is built once, on the frozen rows.** The six's W2_S were built that way (U2b's recorded builds: the
  choice among same-named targets by the record's frozen dense vectors), and UL2's L2e reuses them under e3v. So W2_S here
  is built once with the frozen rows, and the same pairs serve both arms. The record states its ambiguous count (the
  choices the vectors made). Section 5's "W2_S built on each arm's vectors" is withdrawn; K is still built on each arm.
- **Difference 2: H_5 and Part B's forms.** Section 5 had none. They are built by zul2's code (UL2 sections 4 and 7),
  unedited, from this package and MHR1's files: N = W2_S ∪ F0T ∪ K_arm, k = 5, 1e-6, the same key band (df 2 to
  100,000) and the same Part B constants. That build is UL2's (its section 9.3: "RK, with H_5 built by this code"); MHR1
  builds no H or Part B form with its own code. MHR1's files are its inputs (section 9 names the paths).
- **Difference 3: the call.** Section 5's rule is narrowed to **before H**: RK_e3v − RK_frozen, where the two arms now differ
  only through K. In UL2's terms that is L2e without H minus REC (on this corpus REC = D83 = W2_S + F0T + K_frozen). Its
  name becomes **e3v's LINKS CARRY ON MULTIHOP-RAG (before H)**, or NOT_CARRIED (before H), naming the cell. The call on
  the full layout is UL2's contrast, L2e − REC, once H_5 is built here.
- **Unchanged:** F0T (D84's filed rule, zul1 via zd84, CN_all df 2 to 25 plus the title runs), K (the record's kNN-3 code),
  the dedup, the labels, the measures, the differences (R − F0T now reads the same on both arms) and sections 6 to 8.

## 6. Expectations, written before any number

1. The first stage is easier than the benchmark's chunk task, because there are 609 whole articles. Both arms' Hits@10 are
   above the paper's best chunk-level Hits@10 (about 0.75).
2. e3v is LEVEL with frozen on d5: news is edited English like Wikipedia, and e3v was level on MuSiQue. A BELOW call, if it
   comes, is more likely on the 47 articles over about 5,000 tokens, where frozen's longer window reads more.
3. SPLADE alone is far below either dense arm (it reads only an article's first 256 tokens). RRF5 is LEVEL or BELOW each arm's
   d5: the record's fusion gains less when one list is weak.
4. Comparison questions are the hardest at the first stage. They name two sources on one event, and the two articles share an
   event but few words.
5. W2_S is nearly empty (fewer than 1% of F0T's pairs): news headlines are seldom quoted verbatim inside other articles, and W2
   reads exact titles only. The tie-break never decides a W2 link (headlines are distinct). (Under 5.1 one W2_S serves both arms.)
6. K adds little reach over R: F0T already links about 85% of the gold pairs at 13.5% density. RK minus R is under +3 GP points
   for each arm, and most of K's pairs are already in R.
7. K_e3v is LEVEL with K_frozen on GP.

## 7. Code and compute

`outputs/mp_unified/zmhr.py` (new; it imports and never edits zenc, zenc3, zenc3v, zenc4, zenc1, zenc1b2, zd84, zul1, zap1 and zu2b):

| command | where | what |
|---|---|---|
| `selftest` | laptop | the split rule, the benchmark's measures against a brute-force copy of its loop, the K and W2_S calls on a toy |
| `package` | laptop | section 2 with P1 to P4; `outputs/mhr1/package.json` |
| `splade` | laptop CPU | section 3's SPLADE |
| `encode --enc gtemodern\|granitesm [--cpu]` | host GPU; host CPU with `--cpu` (3.1) | GP to outputs/mhr1/gp/ (not run on the CPU), then zenc's encode |
| `frozen [--cpu]` | host GPU; host CPU with `--cpu` (3.1) | the frozen arm into the package |
| `vectors` | host CPU | zenc3v's fusion; `fc` (host CPU): FC on every row, outputs/mhr1/fc.json |
| `cuts` | host CPU | the articles each window cut (each encoder's tokenizer, as zenc counts), checked against the encodes' counts; outputs/mhr1/cuts.json |
| `first` | laptop | section 4 (reads the fetched vectors); outputs/mhr1/first/ and the package's retrieval_cache |
| `links` | laptop | section 5's families and stores; outputs/mhr1/links/ and the package's graph/knn.npz |
| `grade` | laptop | section 5's measures and call; outputs/mhr1/grade.json |
| `report` | laptop | outputs/mhr1/report.md, filed in section 9 |

Compute and space: about 10 GPU minutes on the host for the three encodes (on the CPU under 3.1: minutes for the parts, one to three hours for the frozen arm), and seconds to minutes on the laptop for the rest.
Under 50 MB in all.

## 8. What follows (named here, not run)

- **Pools:** PKO-style pools on e3v (frozen beside) from this package's first stage and the RK store, the gold share of the
  pools graded as COV graded the six.
- **The zero-shot reads** of the models of record (zrckpn, the MLP; GNX, the GNN) on the `test` carve and on all kept
  questions, beside frozen.
- **Named options for later forms:**
  - chunk nodes at e3v's window (the benchmark's granularity);
  - the title written into `text`;
  - SPLADE max-pooled over 256-token windows of the whole article;
  - H's hub names: no longer an option. H_5 and Part B's carried forms are part of the layout of record (5.1), built by UL2.

## 9. Result

Filed 23 November 2026, 11:35 IST. Records: outputs/mhr1/package.json, fc.json, cuts.json, first.json, links.json,
grade.json, gp/, run-*.json and report.md (json and the report only; first/lists.npz, links/*.npz and the package's vectors
and caches stay uncommitted).

**Runs.** The three encodes ran on the host CPU under 3.1. The frozen arm took 142,371 s for the documents (10 threads, no
article cut, the longest 16,447 tokens of 32,768, no OOM) and 1,198 s for the questions. Section 7's estimate was one to
three hours: most of the run fell in the hours the host sat in Modern Standby (21 Nov 16:55 to 23 Nov 10:16 IST). FC passed
on every row (609 articles, 2,255 questions). e3v's windows cut 13 articles (169 gold slots, 2.86%; 164 questions have a cut
gold); frozen cut none; each count equals its encode's record.

**First stage (section 4): e3v LOSES ON MULTIHOP-RAG.**

- All 2,255 questions: d5 e3v 64.29, frozen 66.91, **-2.62 [-3.70, -1.56] BELOW**; rrf5 -1.56 [-2.30, -0.84] BELOW.
- Other cells: d2 -0.61 LEVEL, d4 -2.18 BELOW, d10 -2.18 BELOW, d20 -1.00 BELOW; hits@4 -2.26 BELOW, hits@10 98.31 against
  98.54 (-0.22 LEVEL), MAP@10 -0.67 BELOW, MRR@10 -0.64 LEVEL.
- The test carve (1,130): d5 -2.19 [-3.79, -0.61] BELOW.
- By type: comparison -2.88 BELOW, inference -3.75 BELOW, temporal -0.66 LEVEL.
- The cut articles do not carry the gap. The 164 questions with a gold article e3v cut read d5 -0.30 [-4.52, +4.07] LEVEL
  (rrf5 +3.61 ABOVE); the other 2,091 read d5 -2.80 [-3.87, -1.65] BELOW.
- SPLADE alone: d5 57.46.

**Links (sections 5 and 5.1): NOT_CARRIED (before H).** The deciding cell, GP over all questions: RK_e3v minus RK_frozen
**-1.88 [-2.27, -1.49] BELOW** (GS -1.77 BELOW).

- K_e3v minus K_frozen: -0.75 [-1.64, +0.13] LEVEL.
- F0T holds 25,080 pairs (13.55% density) and links 84.67% of the gold pairs. W2_S holds 2 pairs (one build, on the frozen
  rows), so R equals F0T (R minus F0T +0.00).
- K over R: +0.43 [+0.26, +0.61] on e3v and +2.31 [+1.93, +2.73] on frozen. e3v's K has 1,366 pairs (263 outside R) and
  frozen's 1,339 (224 outside R): frozen's fewer new pairs reach more gold pairs.
- Labels (both arms): R.name_comention 25,005, R.title_comention 75, R.mention 2, S 0, K.knn 1,366 (e3v) and 1,339 (frozen).

**Expectations (section 6).**

1. Held: both arms' hits@10 are about 98, against about 75 at chunk level in the benchmark's paper.
2. Failed: e3v is BELOW, and not on the long articles. The questions whose gold e3v cut read LEVEL; the gap is on the rest.
3. Held: SPLADE d5 57.46; RRF5 is below each arm's d5.
4. Failed: inference questions are the hardest at the first stage (d5 57.0 e3v, 60.8 frozen), then comparison (67.0, 69.9),
   then temporal (70.5, 71.1).
5. Held: W2_S has 2 pairs.
6. Held: RK minus R is +0.43 and +2.31, both under +3.
7. Held: K_e3v is LEVEL with K_frozen.

**What this changes.** Nothing trains, and no filed graph, pool, fit or record changes. MultiHop-RAG now has its package,
its e3v vectors and its links, so it joins UL2's chain as the seventh dataset (UL2 section 9.3: RK, with H_5 built by UL2's
code), and its pools (section 8) follow the unified layout. Beside E4's reads, it is the fourth dataset where e3v's first
stage reads BELOW frozen (with SQuAD, WebQSP and MetaQA; ABOVE on 2Wiki and MuSiQue, LEVEL on HotpotQA). D101 stands: e3v is
the encoder and frozen the comparison, and the gap is for the links, the preprocessing and the semantic-space work to close.
