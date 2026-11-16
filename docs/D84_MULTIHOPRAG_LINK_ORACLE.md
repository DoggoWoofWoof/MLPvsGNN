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

## 9. Result (filed 16 Nov 2026, 18:59 IST; laptop, 245 s; declared at 9d1433c)

**Call: CARRIED F0T OFF WIKIPEDIA.** Records: `outputs/d84/grade.json` and `outputs/d84/report.md` (script sha256 dac9b0035300).

G1 and G2 passed: 609 articles, 2,556 queries, 2,255 kept (301 null dropped); every evidence item names exactly one article by url
and title. Gold counts: 2 golds 1,169 questions, 3 golds 774, 4 golds 312. GP: 5,363 slots.

| form | pairs | x N_sp | density % | GP linked | GP share % [95%] | lift | lift call | GS share % | pairs per linked slot |
|---|---|---|---|---|---|---|---|---|---|
| N_sp | 21,817 | 1.00 | 11.78 | 4,334 | 80.81 [79.35, 82.21] | 6.86 | POINTS | 77.07 | 5.0 |
| CN_all | 25,005 | 1.146 | 13.51 | 4,525 | 84.37 [83.11, 85.61] | 6.25 | POINTS | 79.91 | 5.5 |
| F0T | 25,080 | 1.150 | 13.55 | 4,541 | 84.67 [83.42, 85.91] | 6.25 | POINTS | 80.58 | 5.5 |

| minus N_sp | GP points [95%] | GS points [95%] |
|---|---|---|
| F0T | +3.86 [+3.10, +4.59] ABOVE | +3.50 [+2.30, +4.66] ABOVE |
| CN_all | +3.56 [+2.85, +4.27] ABOVE | +2.84 [+1.69, +3.90] ABOVE |

Beside (F0T minus N_sp on GP): comparison +1.90 [+0.33, +3.49] ABOVE (75.33% against 73.43%); inference +4.20 ABOVE (87.88%
against 83.68%); temporal +5.11 ABOVE (86.00% against 80.89%); 2 golds +1.37 [-0.17, +2.99] LEVEL; 3 golds +6.29 ABOVE;
4 golds +2.40 ABOVE. Golds all in one component (GS), N_sp / F0T: comparison 68.46 / 70.09%, inference 82.72 / 88.73%,
temporal 81.82 / 84.56%.

Rule build: 26,532 name keys, 44,495 name entries; title runs added 233 entries on 163 of the 609 articles; F0T holds 75 pairs
beyond CN_all. spaCy found 21,213 distinct entities.

**Reading.**
- The rule-built store links more gold pairs than spaCy's off Wikipedia too: the third corpus kind after Wikipedia passages
  (UL1's four) and KB names (UL1's two). It does so at 1.15 times spaCy's pairs, inside the 1.30 bound.
- Both stores point: a linked pair is about 6 to 7 times as likely to be a gold pair as a random pair. Per pair, spaCy's store
  is the more precise (lift 6.86 against 6.25; 5.0 pairs per linked slot against 5.5). F0T's gain is in reach, bought with
  about 15% more pairs. The model reads the pairs with their labels (D83), not the count.
- The dense article graph (13.6% of all pairs) is an article-level fact: 609 long articles share many names. Chunk-level nodes
  will be sparser, and that grade comes with the encoder decision.

**The expectations (section 6):**
1. F0T LEVEL or ABOVE N_sp on GP: **held** (ABOVE).
2. Both forms point: **held** (both POINTS; lift 6.25 and 6.86).
3. GP shares well below 100%: **did not hold**. 81% and 85%: the df cap of 25 removed fewer gold links than expected, because
   most of a query's shared names are rarer than the cap.
4. Title runs add little: **held**. 163 articles gained a run; F0T is 75 pairs and +0.30 GP points beyond CN_all.
5. Comparison linked more than temporal: **did not hold**. Comparison is the LOWEST group on both forms (73-75%); temporal sits
   at 81-86%. A comparison query pairs two sources reporting one event in their own words, and such pairs share fewer names.

**What this closes and what stays open.** It closes the graph-level question for the store D83 carries, on news articles at
article level. Still open, all waiting on the encoder decision (the user, 16 Nov: encode only after the encoder is fixed):
- K;
- W2_S (R.mention) on this corpus;
- the chunk-level grade;
- any retrieval number on MultiHop-RAG.
