# D84: MultiHop-RAG, the rule-built NER store graded against the oracle pairs off Wikipedia (graph level, no encoder)

Declared 16 November 2026 (18:5x IST), before `outputs/mp_unified/zd84.py` reads the downloaded files beyond their field names.
The user, 16 November 2026 (verbatim): "yes download but we have to first fix encoder before encoding this, only thing we can
verify is improved links with oracle if possible". Earlier the same day: "if we created better links we can train a model after
verifying those links improved connectivity etc by comparing with the oracle links we have and across datasets".

**Nothing is encoded, nothing trains, no model reads anything, no filed graph, pool, fit or record changes.** D84 asks one
question of the after-paper program's point (a), going beyond Wikipedia: does the link change D83 carries (R's NER store, spaCy's
N_sp replaced by the rule-built F0T, UL1 FILED c662a6b) hold on a corpus that is not Wikipedia, judged against that corpus's own
gold evidence pairs, by UL1's rule?

## 1. The data (pinned)

- MultiHop-RAG (Tang and Yang 2024), Hugging Face `yixuantt/MultiHopRAG`, revision `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`,
  licence ODC-By. Downloaded 16 Nov 2026 18:51 IST into `data/raw/multihoprag/` (gitignored; never committed).
  - `corpus.json`: 6,785,567 bytes, sha256 `20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f`.
  - `MultiHopRAG.json`: 5,171,312 bytes, sha256 `03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff`.
  - The code refuses any other bytes (gate G1).
- Nodes: the corpus's news articles, one node each, in file order. A node's title is `title`; its text is `body`.
- Questions: every query whose `evidence_list` is not empty (null queries have no gold and are dropped). A query's golds are the
  distinct articles its evidence items name (section 3, G2). Queries keep their file order; a query's id is its file position.
- Granularity: the ARTICLE. The gold labels are article-level, so the oracle pairs need no construction of ours. Chunk-level
  nodes wait for the encoder decision (the chunk length is the encoder's window); nothing here depends on it.

## 2. What is built (the families the encoder does not touch)

The forms of R's NER slot, as in UL1 (docs/UL1_UNIFIED_LINKS_ORACLE_CHECK.md), each a set of unordered article pairs:

| form | rule | code |
|---|---|---|
| none | no NER store | (empty) |
| N_sp | spaCy `en_core_web_sm` 3.8.0 entities of the labels {PERSON, GPE, ORG, LOC, FAC, WORK_OF_ART, EVENT, PRODUCT, NORP}, longer than 2 characters, lower-cased; co-mention pairs of an entity mentioned by 2 to 25 articles | `scripts/b1_build.py` `ner_edges`, unchanged (the package's N_sp) |
| CN_all | U2r's rule names (capitalized runs, the corpus's own capitalization statistic), co-mention pairs at df [2, 25] | `zul1.build_entries` + `zul1.forms_from_entries` (F0) |
| F0T | CN_all united with the co-mention pairs of the names plus each article's own title runs (AP1b's level-0 title form): D83's R.name_comention + R.title_comention | the same (F0 united with LT) |

Not built, and why:
- **K** (the 3 nearest neighbours) needs the encoder. The user: encode only after the encoder is fixed.
- **R's rule part W2_S** (U2b's title mentions, D83's R.mention) breaks a tie between same-named targets with the frozen dense
  vectors (`zu2b.link_pass`), so it also needs the encoder. D83 does not change W2_S; D84 grades only the store D83 changes.
- **S**: news articles ship no structure (A1: S only where the data has structure). MultiHop-RAG's domain is R + K; here, R's
  NER slot alone.

So "none" is the empty set, and every form's union is the form itself.

## 3. Gates (each stops the run, naming what failed)

- **G1** the two files' sha256 and sizes are section 1's; the corpus is a list of objects with `title` and `body`; the queries a
  list with `query`, `question_type`, `evidence_list`.
- **G2** every evidence item of a kept query names exactly one article: its `url` is the `url` of exactly one article, and that
  article's `title` equals the item's `title` after whitespace normalization. Corpus urls are unique.
- **G3** F0 (CN_all) is a subset of F0T, every pair is a < b inside the nodes, and N_sp is `ner_edges`' output on the bodies in
  node order (the pair count equals its returned arrays).

## 4. The oracle and the measures

- **GP slots**: for every kept question with at least two distinct golds, every unordered pair of its golds (zul1.gp_slots).
- **GP share** of a form: the share of GP slots the form links, by the ratio estimator over questions; the difference of two forms
  with its 95% interval from zap1.boot_diff (2,000 resamples of the questions, seed 0), called ABOVE / LEVEL / BELOW (zap1.call).
- **Size**: the form's pair count; ratio to N_sp's.
- Beside (reported, never deciding):
  - **density** = pairs / (n (n - 1) / 2) and **lift** = GP share / density, with the call POINTS (the GP share's 95% interval,
    the same bootstrap over questions, lies above the density), BELOW_BASE (below it) or NO_LIFT;
  - **GS**: the share of questions with 2 or more golds whose golds form ONE connected component in the form's pairs among them
    (the chain-connectivity analogue: MultiHop-RAG ships no decomposition);
  - GP and GS per question type (comparison, inference, temporal) and per gold count (2, 3, 4);
  - **price**: pairs per linked GP slot;
  - the title runs: how many articles' titles add a run that is not already among their names.

## 5. The rule (UL1's section 6, off Wikipedia)

**CARRIED F0T OFF WIKIPEDIA** if F0T's GP share minus N_sp's is not BELOW, and F0T's pair count is at most 1.30 times N_sp's.
Else **NOT_CARRIED**, every failed condition named. CN_all's row stands beside (D83 carries F0T, not CN_all).

A NOT_CARRIED call closes only this form on this corpus at article level; it does not reopen D83 (which is decided by its own
reads, docs/D83_UNIFIED_R_REFIT.md section 7). The next form named for that case: the chunk-level grade after the encoder
decision, and a df cap read from the corpus size (25 of 609 articles is 4%, of a Wikipedia corpus far less).

## 6. Expectations, written before any number

1. F0T is LEVEL or ABOVE N_sp on GP: news bodies are written in edited English, where capitalized runs and spaCy's entities agree
   on most names.
2. Both forms POINT (lift above 1 with the interval above the density): the queries ask about the same people, firms and events
   across sources, and co-mention is the evidence.
3. The df cap of 25 removes the most-mentioned names (large firms, countries), so GP shares sit well below 100% for both forms.
4. Title runs add little: news headlines are mostly sentence case, so few titles add a run beyond the article's names.
5. Comparison queries (two sources on one event) are linked more often than temporal ones.

## 7. Code

`outputs/mp_unified/zd84.py`:
- `selftest`: a toy corpus and queries; brute-force CN_all / F0T pairs, GP and GS shares, G2's refusals, the lift call; passes
  before any real read.
- `grade`: G1-G3, the four forms, the measures; writes `outputs/d84/grade.json` (counts and shares only; pair arrays stay in
  memory) on the laptop (609 articles: seconds to minutes; spaCy dominates).
- `report`: section 5's call and the beside rows into `outputs/d84/report.md`, filed in section 9 below.

## 8. What D84 does not do

- No encoding, no K, no W2_S, no pool, no fit, no read, no host job.
- No change to D83 or its arms; D84's numbers never enter a D83 cell.
- No chunking and no retrieval number: those follow the encoder decision.

## 9. Result

(Filed after the run.)
