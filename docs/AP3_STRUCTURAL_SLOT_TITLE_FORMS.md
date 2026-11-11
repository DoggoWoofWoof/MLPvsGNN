# AP3: rare base-title links in the structural slot of the rule-built carve (b1y)

Declared 12 November 2026, before any AP3 number (the ladder or a read). Follows AP1 ([AP1_RULE_NAMES_AND_MUSIQUE_REACH.md](AP1_RULE_NAMES_AND_MUSIQUE_REACH.md), filed in
981dab4), AP1b ([AP1B_SUBSPAN_NAMES.md](AP1B_SUBSPAN_NAMES.md), 3bb8c25) and AP2 ([AP2_RULE_NER_SLOT_ON_B1.md](AP2_RULE_NER_SLOT_ON_B1.md), Part A 4d648f9, Part B
0c82757). Code: `outputs/mp_unified/zap3.py` (new; it imports `zb1l.py`, which is `zb1h.py` rebound, and edits neither). Outputs only under `outputs/ap3/` and
`outputs/b1y/`. Decision D73 (AP3's scope) is answered here; D79 to D81 are new (section 9).

Nothing trains. ML1's and LTM's fits, PKO's chain code, and the reads of b1t, b1w and b1n are read and never written. AP3 decides nothing about a model, a pool or a
column of record. It asks what a second rule family in the structural slot adds, at the graph level first and then, where a rung is carried, to the models of record.

## 1. Why

The user's list for the after-paper program (11 November): "more rules to improve our results based on the datasets, hyperlinks, triplets etc". Three steps are filed:
the rule's names replace spaCy's NER slot at read time (AP2: REPLACE, carve b1n = R + F0T + K), and the structural slot holds R (U2b's title links and sibling links) in
the models' training carves. The structural slot is where the shipped structure S (2Wiki's and HotpotQA's hyperlinks, MuSiQue's package title-mention graph) differs
from R, and the filed reads of the models of record on S against R (B1H, in-domain, R@5 points with 95% intervals) are the size of what that slot can matter: S is above R on
HotpotQA (+2.2 (+1.5, +2.9) for both models) and level on 2Wiki (+0.4 and +0.3) and MuSiQue (-0.4 and -0.1), every interval there holding 0. A rule that links more of what S links, at
a price, is the one place the program can still add structure without an LLM.

U2b's own rule has a second form that R does not use: a title's **base** form (the title without its parenthetical, or the text before its first comma) as a surface
that a passage mentions, linked exact-first (variant W5e against W2). On a corpus of disambiguated titles ("Name (film)") that is the natural mention of the title. Its
cost is hubs: a base form that many passages mention links all of them to the owner. AP3 therefore reads the form on a **ladder of caps on the surface's document
frequency** and carries a rung only by a price rule declared here (S1).

## 2. What is built and read

**K1** = U2b's link rule (`zu2b.NEW_RULE`: the look boundary, underscores as spaces, prefixes, descriptive sources only, the setting's capitalization statistic as the
common-word gate), run on each B1 setting's own passages, with the pairs that W5e adds over W2: a surface that is a base form (kind 1) of exactly one title group (after
the passage's own title is dropped) and no exact title (kind 0) of any other. An ambiguous surface (several title groups) links nothing: the context choice U2b makes by
dense cosine needs vectors the laptop does not hold. df of a surface = the number of the setting's passages whose text holds it (case-checked), its owner's own among
them; a pair's df = the smallest df of the surfaces that give it. Prefix forms and the last-token form are not built (a scouting read, section 10, found them noise).

**The ladder**: caps c in 1, 2, 3, 4, 6, 8, 12, 16, 25 and no cap. Rung c = R plus the K1 pairs with df <= c that R lacks. The carve's structural family is **Ry** = R
plus the carried rung's pairs; R is always a subset.

| carve | structural | NER slot | K | settings | role |
| --- | --- | --- | --- | --- | --- |
| b1t | none | spaCy's N | kNN | musique, 2wiki, hotpotqa | B1P's and B1M's reads; reference |
| b1w | R | spaCy's N | kNN | the same | B1P's and B1M's reads |
| b1n | R | F0T | kNN | the same | AP2 Part A's reads; **the base of this stage** (the spaCy-free pipeline) |
| **b1y** | **Ry** | F0T | kNN | the settings where S1 carried a rung | new: Part B |

- **The folder.** b1n's six files (`outputs/b1n/bench/hipporag2_n/<setting>/`) byte for byte except `graph_structural.npz`, which holds Ry (each pair once each way, ascending
  by (src, dst), int32, `rel` zeros int16: b1w's layout, and the writer reproduces b1w's file from R's pairs byte for byte, G3b). `build.json` is b1n's with the structural file's
  sha replaced, its `u1a` block relabelled (the carve, the variant, the edge count, this script's sha256; `b1a_structural_sha256` kept as the block holds it), a note in the `b1f`
  block, b1n's `ap2` block kept (it holds the NER sha the lists check needs) and an `ap3` block (the cap and its ladder row, the pair counts, the shas of `ladder.json`,
  `k1.npz`, b1w's structural file and b1n's `build.json`).
- **The models.** zg6r and zrckpn, seeds 0 to 2, fitted on each of the two training splits: 12 reads per setting and carve, as AP2's. In-domain = the mean over the fits whose
  training split is the setting's partner (`zb1h.IN_DOMAIN`); held-out = the others. The questions are the setting's 1,000.
- **The lists.** B1a's, through `zap3.lists_of`, whose check (`zap3.check_lists`) pins the two replaced files: the structural one to `u1a.b1a_structural_sha256`, the NER one to
  `ap2.b1a_ner_sha256`.

## 3. Part A: the ladder (the laptop, graph level, no model) and S1

For each setting, on the slots of its **basis**: GP (every pair of gold passages of a question) on 2Wiki (2,175 slots) and HotpotQA (1,000), MuSiQue's chain slots CH (1,648)
on MuSiQue (its GP, 2,462 slots, is reported beside). Per rung: the pairs added over R, the slots newly linked (and so the gain in points of the setting's slots, with a
paired 95% interval over questions from `zap1.boot_diff`'s estimator), the price (pairs added per slot added), the share of the added pairs that are S's, and the same
pair and slot counts over b1n's union R + F0T + K. Everything is written to `outputs/ap3/ladder.json` and `ladder.md`; the K1 pairs
with their df to `outputs/ap3/<setting>/k1.npz`.

**S1, the carry rule (counts decide, never the rounded fields).** p0 = R's own price: its pairs per slot of the basis it links. A rung's price = (pairs added) / (slots
added) over R; a rung that adds pairs and no slot has no price and fails; a rung that adds nothing passes. The walk starts at the first rung and stops at the first rung
whose price is above p0 (the prefix rule: every rung up to the carried one passes). The carried rung is the last rung reached; it is **carried only if it adds at least
2.0 points of the setting's basis slots**, otherwise the setting gets **no carve**. The constants (the caps, the floor of 2.0 points, p0 as the price) are fixed here; the
ladder's numbers do not change them.

**Gates (a failed gate ends the stage; nothing is filed past one).**

| gate | the ladder (laptop) | the carve (`bench`, laptop) | the host, the report |
| --- | --- | --- | --- |
| G1 | the setting-level rebuild of R's rule (K0 and sibling pairs) and the filed R have Jaccard >= 0.95 | `k1.npz`, R's pair count and the carve's pair count equal the ladder record's | |
| G2 | every K1 pair is a < b inside the setting's passages | b1n's structural file is b1w's | |
| G3 | per passage, the single-hit choices (K0, K1) equal `zu2b.choose_targets` on all the passage's hits | K1 pairs a < b; the written file holds exactly Ry, once each way, sorted | |
| G3b | | the writer reproduces b1w's structural file from R, byte for byte | |
| G4 | R is b1w's structural file (pairs and sha); the node count and the slot counts equal build.json's and AP1b's report's | b1n's folder, build.json and the ladder agree on the node count | |
| G2b | | the folder as written carries its own build.json's shas, the five copied files equal b1n's, and B1a's lists check accepts it | |
| G5 | AP1b's `levels.npz` and its F0T pair count equal AP1b's report's | | PKO's gate on the carve (`zap3.py gate`) passes |
| G6 | `boot_ratio` equals `zap1.boot_diff` | | the rows recomputed from B1P's and B1M's reads equal B1M's filed rows |

## 4. Part B: the models of record on the carve (the host reads; only where S1 carried a rung)

All differences are paired by question, 95% bootstrap intervals over the 1,000 questions (2,000 resamples, seed 0: `b1b.boot_ci`), no threshold.

- **A model's word** on b1y - b1n (in-domain): GREATER when the bottom of the interval is above 0, LESS when the top is below 0, SAME otherwise (SAME is not an equivalence).
  b1y - b1w and b1y - b1t are reported with the same words (VS_W and UP / DOWN / FLAT).
- **A setting's call**: GREATER, SAME or LESS where the two models agree; NOT_LESS where they differ and neither is LESS; MIXED where one is LESS and the other is not;
  INCOMPLETE where a read is missing (`zb1l.stage_call`).
- **The verdict: ADOPT** when no carved setting's call is LESS or MIXED and at least one is GREATER or NOT_LESS; **NO_GAIN** when every carved setting's call is SAME;
  **HOLD** when a setting's call is LESS or MIXED (decided even while another setting's read is missing); **INCOMPLETE** while a read is missing and none is LESS or MIXED.
  ADOPT says Ry may stand in the place of R in the structural slot of the after-paper program's read path (on the settings it was carved for); NO_GAIN and HOLD keep R and
  name what fell. If S1 carries no rung on any setting, Part B does not run and AP3's result is "no carve" with the ladder as the record.
- **Reported, not deciding:** the held-out reads; b1y - b1w and b1y - b1t; MuSiQue by hop depth (where carved); the gap GNN - MLP at b1y.

## 5. What it can and cannot show

- A **read-time shift**: the models were fitted with R in the structural slot, so a SAME or LESS here does not say Ry is a worse or an equal training input. A refit on carves
  whose structural slot holds Ry is D79 (default: not now, with D78).
- **Selection on the evaluation questions.** The rung is chosen by the gold pairs of the same 1,000 questions the reads grade (S1 reads the gold pairs and chain slots), so a
  GREATER word carries that selection; the rule has one free choice per setting (the rung) inside constants fixed here, and B1's settings hold no other questions to choose on.
  The ladder's graph-level numbers are not held-out either (they are the grade of a family chosen on them).
- **Dev-read disclosure.** The family (K1), the ladder's shape and S1's constants were fixed after scratch reads of the three settings (section 10); those reads are not filed
  and their numbers are not used here, except as the expectation stated below.
- **The expectation, stated before the run** (from the scratch reads): no rung on 2Wiki, a shallow rung on HotpotQA (a handful of passages per surface) and a deeper one on
  MuSiQue (up to the cap of 25); a carve changes R@5 by at most a few tenths of a point (the structural slot's whole range on the models, S against R, is +2.2 points on
  HotpotQA and level elsewhere, B1H), so NO_GAIN is the likely word on a 1,000-question read; a GREATER word would be a small effect.
- Nothing here changes a number of the paper or of any record: b1w's rows remain the record.

## 6. What AP3 does not do

- **Triples, typed rules.** Not built here. AP1b filed that F0T leaves 475 MuSiQue chain slots unlinked, 264 of them at hub names. A scouting read (section 10, not filed; its numbers
  are not cited) found that links to hub names cost far more new pairs per added slot than N_sp's 128.6, so S1's own price rule refuses them, and that an entity layer which does reach
  most of those slots leaves the linked partner in the top 10 of the family's own ordering for fewer than half of them: reach is not the gap, ranking among reached passages is, and that
  is a query-side question (D80).
- **Prefix and last-token title forms** (U2b's kinds 2 and 3): noise in the scouting read; not built.
- **An S-in-the-structural-slot ceiling with F0T** (the b1s idea): B1H (the models of record) and J5S (the J5 fits) already read S against R with spaCy's slot, and Part B's carve answers
  the AP question with b1n as base (D81, default no).
- **Any change to R, to the pools or to a model.**

## 7. Rows (queued only after Part A is filed)

Part B per carved setting, AP2's `ap2x-` rows mirrored as `ap3y-` (selftest, settings, fork, chain at 2 cpus, PKO's gate, lists, columns for both splits, 12 reads: GPU reads 1.1 cpus
and GPU share 0.15, MLP reads 6 cpus). Memory is reserved at the measured peak of AP2's same step scaled by the union's size (Ry adds a few percent of pairs), and the host's measured
peak replaces it in the record. The report (`zap3.py report`, on the laptop) after the reads; it reads b1n's reads (AP2's) from `outputs/b1n/`.

## 8. Files

`outputs/ap3/ladder.json`, `ladder.md`, `<setting>/k1.npz` (not committed: it is a rebuildable array); Part B adds `outputs/ap3/report.json`, `report.md`, `calls.json`;
the carve folders `outputs/b1y/bench/hipporag2_y/<setting>/` stay local until the host's copy is the one read.

## 9. Decisions (the user's; my default first)

- **D73** (AP3's scope): the structural slot's base-title family on a df ladder, as declared. Default: this scope; triples and typed rules wait for D80.
- **D79:** a refit whose training carves hold Ry (or any new structural family) is not part of AP3. Default: not now, after Part B and with D78.
- **D80:** a later stage (AP4) on query-side typed rules and a triple family, starting with a scouting stage with its own declaration. Default: yes after AP3's Part A is filed, as a scouting stage only.
- **D81:** a ceiling carve with S in the structural slot beside F0T. Default: no (section 6).

## 10. Development log (disclosed before any AP3 number)

- Scratch reads on the laptop, 12 November (not filed; scripts in the session's scratch folder; `ladder.py`, `link_variants.py`, `link_variants2.py`, `k1_precision.py`, `r_vs_s.py`,
  `rel_probe.py`, `probe_graphs.py`): U2b's variants (W2, W5e, with prefixes and last-token forms) rebuilt on each setting's own passages and graded against S, the gold pairs and
  MuSiQue's chain slots; the df ladder and its price per added slot; the precision of the K1 pairs against S by df; R against S; a relation probe and graph probes for the triple and typed
  scouting of section 6. They showed that the base form's gain sits at very small df (the gold pairs it links are rare-surface pairs), that
  uncapped it adds thousands of pairs for a handful of slots, and that the prefix and last-token forms add pairs and no slots. The constants of S1 and the caps were fixed after them.
- Scratch dev read of `zap3.k1_links` on the three settings (12 November, 03:2x IST, not filed): runs in 26 to 63 seconds per setting; G3 holds on every passage; the setting-level rebuild
  of R's rule has Jaccard 0.974 (2Wiki), 0.981 (HotpotQA) and 0.989 (MuSiQue) with the filed R (the gate was written as 0.95 before this read).
- `zap3.py selftest` (host-safe: the pair layout, the ladder on a toy, the bootstrap copy, S1 on synthetic ladders including the floor and the prefix rule, the lists check, the carve's
  build.json, bench's refusal of a setting with no carried rung, the settings label, `as_variant`, the pair, word, call and verdict code on fake reads, the markdown) and
  `zap3.py selftest-ladder` (the link rule on a toy corpus: exact-title and base-title links, the df of a surface, an ambiguous surface dropped and counted once, each pair once) pass.
  The writer was checked on the three real b1w structural files (rebuilt from R's pairs, sha equal to build.json's): the check is now gate G3b.
- Corrected before any number was filed (the commit after the declaration's): section 6's first bullet cited "the pools already hold nearly all the gold passages" for the triple
  and typed scouting; that was not what the scouting read found (it found that reach is not the gap and ranking is), and the bullet now says so and cites only AP1b's filed counts.
  No rule, constant or gate changed.
- A failing step is fixed and the fix disclosed here with its commit before any number is filed.
