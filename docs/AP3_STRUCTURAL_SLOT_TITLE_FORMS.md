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
- **D82** (added when Part B was filed, section 12): use Ry at cap 3 for HotpotQA alone in the after-paper read path, against the declared verdict (HOLD). Default: no.

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

## 11. Part A filed (12 November 2026)

`python -I outputs/mp_unified/zap3.py ladder` on the laptop, 03:35 to 03:38 IST on 12 November (the record's `utc` stamp is 2026-11-11T22:07:56Z), started after the declaration's commit
(b4fd7be, 03:35:06 IST). The correction recorded at the end of section 10 was committed after the run began (e5449e7, 03:39 IST); it changed no rule, constant, gate or script, and the
declaration and the correction were pushed before this filing. The run ended with exit 0, so every gate of the ladder column of section 3 held (each is a hard stop): G1 the
setting-level rebuild of R's rule has Jaccard 0.9741 (2Wiki), 0.9806 (HotpotQA) and 0.9891 (MuSiQue) with the filed R (gate 0.95);
G2 and G3 on every passage U2b's rule scans (its descriptive sources: 6,119 of 6,119 passages on 2Wiki, 9,809 of 9,811 on HotpotQA,
11,656 of 11,656 on MuSiQue); G4 R is b1w's structural file, the node and slot counts equal build.json's and AP1b's; G5 AP1b's `levels.npz` and its F0T pair
count equal its report's; G6 `boot_ratio` equals `zap1.boot_diff`. The record names its inputs by sha256: the script (91f81139, equal to the committed `zap3.py`'s), `zap1.py`
(48412755), `zu2b.py` (4bfbf747) and AP1b's report (bbc5134c).

**The ladder.** A rung is R plus the K1 pairs at or under the cap that R lacks. Gain = the slots the rung newly links over R, in points of the setting's basis slots, with a paired 95%
interval over the 1,000 questions (ABOVE: the bottom of the interval is above 0); price = pairs added per slot added; the union columns count the same over b1n's union R + F0T + K
(pairs of the rung that the union lacks, and slots they newly link). The record is `outputs/ap3/ladder.json`; the tables are `outputs/ap3/ladder.md`'s.

**2Wiki** (basis GP, 2,175 slots; 6,119 passages). R's 2,030 pairs link 872 slots (40.09%), so p0 = 2.328 pairs per slot; S precision of R 0.8246, S recall 0.8438.
K1 holds 2,481 pairs (2,281 not in R; 2,443 ambiguous surface hits dropped).

| cap | pairs added | slots added | gain (points) [95%] | share linked | price | S share of added | union: pairs added | union: slots added |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 30 | 13 | +0.60 [+0.28, +0.94] ABOVE | 40.69 | 2.31 | 15/30 | 2 | 0 |
| 2 | 191 | 50 | +2.30 [+1.69, +2.96] ABOVE | 42.39 | 3.82 | 72/191 | 25 | 0 |
| 3 | 300 | 64 | +2.94 [+2.25, +3.68] ABOVE | 43.03 | 4.69 | 95/300 | 55 | 0 |
| 4 | 396 | 69 | +3.17 [+2.45, +3.93] ABOVE | 43.26 | 5.74 | 111/396 | 90 | 0 |
| 6 | 533 | 73 | +3.36 [+2.61, +4.12] ABOVE | 43.45 | 7.3 | 129/533 | 128 | 1 |
| 8 | 679 | 76 | +3.49 [+2.75, +4.29] ABOVE | 43.59 | 8.93 | 134/679 | 201 | 1 |
| 12 | 786 | 80 | +3.68 [+2.92, +4.49] ABOVE | 43.77 | 9.82 | 142/786 | 253 | 1 |
| 16 | 829 | 80 | +3.68 [+2.92, +4.49] ABOVE | 43.77 | 10.36 | 142/829 | 279 | 1 |
| 25 | 1005 | 81 | +3.72 [+2.97, +4.55] ABOVE | 43.82 | 12.41 | 145/1005 | 385 | 1 |
| inf | 2281 | 82 | +3.77 [+3.00, +4.60] ABOVE | 43.86 | 27.82 | 148/2281 | 1503 | 1 |

**HotpotQA** (basis GP, 1,000 slots; 9,811 passages). R's 4,315 pairs link 591 slots (59.1%), p0 = 7.301; S precision of R 0.8197, S recall 0.8039.
K1 holds 5,450 pairs (5,244 not in R; 281 ambiguous surface hits dropped).

| cap | pairs added | slots added | gain (points) [95%] | share linked | price | S share of added | union: pairs added | union: slots added |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 15 | 5 | +0.50 [+0.10, +1.00] ABOVE | 59.6 | 3.0 | 7/15 | 4 | 2 |
| 2 | 215 | 43 | +4.30 [+3.10, +5.50] ABOVE | 63.4 | 5.0 | 90/215 | 45 | 3 |
| 3 | 357 | 54 | +5.40 [+4.00, +6.70] ABOVE | 64.5 | 6.61 | 129/357 | 79 | 3 |
| 4 | 495 | 57 | +5.70 [+4.30, +7.00] ABOVE | 64.8 | 8.68 | 158/495 | 122 | 3 |
| 6 | 683 | 61 | +6.10 [+4.70, +7.50] ABOVE | 65.2 | 11.2 | 189/683 | 202 | 3 |
| 8 | 895 | 64 | +6.40 [+4.90, +7.80] ABOVE | 65.5 | 13.98 | 213/895 | 297 | 3 |
| 12 | 1311 | 69 | +6.90 [+5.40, +8.40] ABOVE | 66.0 | 19.0 | 263/1311 | 449 | 3 |
| 16 | 1510 | 73 | +7.30 [+5.70, +8.80] ABOVE | 66.4 | 20.68 | 269/1510 | 582 | 3 |
| 25 | 1707 | 75 | +7.50 [+5.90, +9.10] ABOVE | 66.6 | 22.76 | 274/1707 | 715 | 3 |
| inf | 5244 | 77 | +7.70 [+6.10, +9.30] ABOVE | 66.8 | 68.1 | 283/5244 | 3937 | 4 |

**MuSiQue** (basis CH, 1,648 chain slots; 11,656 passages). R's 41,429 pairs link 475 slots (28.82%), p0 = 87.219; S precision of R 0.9234, S recall 0.8933.
K1 holds 6,426 pairs (5,844 not in R; 1,691 ambiguous surface hits dropped). On MuSiQue's GP view (2,462 slots, beside, deciding
nothing) the same rungs add the same slots (cap 25: 54) and the gain at cap 25 is +2.19 [+1.62, +2.81] points.

| cap | pairs added | slots added | gain (points) [95%] | share linked | price | S share of added | union: pairs added | union: slots added |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 29 | 1 | +0.06 [+0.00, +0.18] LEVEL | 28.88 | 29.0 | 0/29 | 11 | 0 |
| 2 | 154 | 16 | +0.97 [+0.54, +1.45] ABOVE | 29.79 | 9.62 | 0/154 | 43 | 0 |
| 3 | 272 | 24 | +1.46 [+0.91, +2.05] ABOVE | 30.28 | 11.33 | 1/272 | 88 | 0 |
| 4 | 348 | 33 | +2.00 [+1.38, +2.68] ABOVE | 30.83 | 10.55 | 1/348 | 124 | 0 |
| 6 | 545 | 35 | +2.12 [+1.48, +2.81] ABOVE | 30.95 | 15.57 | 1/545 | 211 | 0 |
| 8 | 682 | 37 | +2.25 [+1.58, +2.95] ABOVE | 31.07 | 18.43 | 1/682 | 271 | 0 |
| 12 | 951 | 45 | +2.73 [+2.02, +3.48] ABOVE | 31.55 | 21.13 | 1/951 | 418 | 0 |
| 16 | 1300 | 50 | +3.03 [+2.22, +3.94] ABOVE | 31.86 | 26.0 | 1/1300 | 618 | 0 |
| 25 | 1864 | 54 | +3.28 [+2.41, +4.18] ABOVE | 32.1 | 34.52 | 2/1864 | 897 | 0 |
| inf | 5844 | 58 | +3.52 [+2.62, +4.49] ABOVE | 32.34 | 100.76 | 2/5844 | 4576 | 0 |

**S1 by the pinned code** (the walk re-derived from the counts, not the rounded fields, equals the record's):

- **2Wiki: no carve.** p0 = 2,030 / 872 = 2.3280. Cap 1 passes (30 pairs for 13 slots = 2.3077) and cap 2 stops the walk (191 pairs for 50 slots =
  3.82). The deepest passing rung adds 13 slots, 0.60 points of 2,175, below the floor of 2.0.
- **HotpotQA: cap 3.** p0 = 4,315 / 591 = 7.3012. Caps 1 to 3 pass (prices 3.00, 5.00, 6.61), cap 4 stops the walk (8.68). Cap 3 adds
  357 pairs and 54 slots, +5.40 [+4.00, +6.70] points of 1,000 (R links 59.1%, Ry 64.5%); 129 of the 357 added pairs are S's.
- **MuSiQue: cap 25.** p0 = 41,429 / 475 = 87.219. Caps 1 to 25 pass (prices 9.62 to 34.52), no cap stops the walk (100.76). Cap 25 adds
  1,864 pairs and 54 chain slots, +3.28 [+2.41, +4.18] points of 1,648 (R links 28.82%, Ry 32.1%); 2 of the 1,864 added pairs are S's.

**What the ladder says.**

- The expectation stated before the run (section 5) held for the rungs: no rung on 2Wiki, a shallow rung on HotpotQA (cap 3), the deepest capped rung on MuSiQue (cap 25). Its effect-size part
  (an effect of a few tenths of a point at most, NO_GAIN likely) is Part B's to test and this filing does not change it.
- The base form's gain over R is mostly reached at low df on all three settings: HotpotQA +4.30 [+3.10, +5.50] points by cap 2 and +5.40 [+4.00, +6.70] by cap 3 against +7.70 [+6.10, +9.30] uncapped; 2Wiki +2.30 [+1.69, +2.96] by cap 2 against
  +3.77 [+3.00, +4.60]; MuSiQue +3.28 [+2.41, +4.18] by cap 25 against +3.52 [+2.62, +4.49]. Past the carried rungs each added slot costs many more pairs (uncapped: 68.1 pairs per slot on HotpotQA,
  100.8 on MuSiQue, 27.8 on 2Wiki).
- 2Wiki's K1 gain over R is real at the graph level (+2.30 [+1.69, +2.96] points at cap 2) but costs 3.82 pairs per slot against R's own 2.33, and over b1n's union it adds
  0 slots up to cap 4 and 1 at any higher rung: no carve is the declared rule's answer (price and floor), not a statement about the family's reach over R.
- The carried rungs' pairs are mostly not S's: 129 of HotpotQA cap 3's 357 added pairs are S pairs (R's own S precision is 0.8197) and 2 of MuSiQue cap 25's 1,864 (R's 0.9234).
- Over b1n's union (R + F0T + K) the carried rungs add little reach. HotpotQA cap 3: of its 357 pairs over R, 79 are new to the union (278 are already in F0T's or the kNN file, by
  arithmetic) and they link 3 slots (0.3 points; the union already links 796 of the 1,000 gold pairs, 79.6%). MuSiQue cap 25: of 1,864, 897 are new
  to the union (967 already there) and they link 0 chain slots (the union links 1,173 of 1,648, 71.18%). The ladder's gain is a gain in the structural slot over R; whether the
  models' reads move at all, with 278 and 967 pairs changing slot and 79 and 897 entering the graph, only the reads can show.
- Selection (section 5): each rung is graded on the gold pairs and chain slots of the questions S1 reads, so the ladder's gains describe the chosen family on the questions it was chosen on; they are not held-out.

Nothing here decides a model, a pool or a column of record.

**Next: Part B** (section 4) for HotpotQA (cap 3) and MuSiQue (cap 25) only: `zap3.py bench --dataset D` builds each carve folder on the laptop, then the `ap3y-` rows of section 7 (12 reads per
setting) run on the host, and the report is filed as section 12. 2Wiki has no carve and no Part B.

Files: `outputs/ap3/ladder.json` and `ladder.md` (committed); `outputs/ap3/<setting>/k1.npz` (not committed, rebuildable; sha256 8a3a6eef, a3d1cec4, 0beb2902 for 2Wiki, HotpotQA, MuSiQue).

**Part B set up (12 November, 03:45 to 03:54 IST), before any read.** `zap3.py bench --dataset D` built the two carve folders (`outputs/b1y/bench/hipporag2_y/hotpotqa/` at cap 3: R's 4,315 pairs plus 357 =
4,672 pairs, 9,344 directed edges; `.../musique/` at cap 25: 41,429 plus 1,864 = 43,293 pairs, 86,586 directed edges), 1 s per setting, with gates G1, G2, G3, G3b, G4 and G2b held (each a hard stop); the
structural file and `build.json` are the only files that differ from b1n's. 38 `ap3y-` rows were queued behind AP2's (the self-test, the fork, and per setting the settings record, the chain, PKO's gate, the lists check, two
column sets and 12 reads); the chains reserve AP2's measured peaks on b1n (hotpotqa 4.76 GB, musique 6.56 GB, from the host's job records) plus about 15%: 5.5 and 7.5 GB. The queue (sha256 30fd07e0) and `zap3.py`
(91f81139) are equal on the host. A fixture run of `zap3.py report` on the laptop (b1y's reads copies of b1n's real reads, stub settings records, the fixture folders removed afterwards) passed before any read: rc 0, no read
missing, every b1y - b1n difference exactly 0 (SAME, verdict NO_GAIN by construction), 64 rows recomputed from B1P's and B1M's reads equal to B1M's; nothing in it is a result. The report's b1n level column reads "--"
(b1n's levels are AP2's, section 9 of its document); the difference columns decide.

## 12. Part B filed (12 November 2026)

**The run.** The 38 `ap3y-` rows queued behind AP2's at 03:50 IST (section 7; queue sha256 30fd07e0) all finished rc 0 on the host, all 38 done by 04:37 IST on 12 November. The chains' measured peaks (the host's job
records) were 4.99 GB for HotpotQA (4 min 33 s, 71 output files; 5.5 GB reserved; AP2's b1n chain peaked at 4.76) and 6.59 GB for MuSiQue (26 min 50 s, 379 output files; 7.5 GB reserved; b1n's chain peaked
at 6.56); the job record of MuSiQue's chain lists one leftover process, as that of AP2's b1n chain does. The 24 read files and the two settings records were fetched to the laptop (HotpotQA's at 04:09 IST,
MuSiQue's at 04:38 to 04:39); four of them (three reads and a settings record) were compared with the host's by sha256 and are equal.

`python -I outputs/mp_unified/zap3.py report` ran on the laptop from 04:39:29 to 04:39:37 IST (the record's `utc` stamp is 2026-11-11T23:09:37Z), after every rule, constant and gate of this stage had been committed
and pushed (b4fd7be, e5449e7, ec19758, 771abf6) and after the last read file arrived. It ended with exit 0 and "Missing reads: 0"; G6 held (64 rows recomputed from B1P's and B1M's reads equal B1M's filed rows). The
record names its inputs by sha256: the script (91f81139, equal to the committed `zap3.py`'s), `zb1h.py` (33b0e495), B1P's and B1M's records (76f76cf0, b523126c), AP2's report (93a0f089) and the ladder (a1d2f99f);
each read file's sha256 is in `report.json`. Three further checks ran on the laptop before this section was written (a scratch script, not filed): each carve's pair counts equal the ladder's rung; each setting's
call recomputed by `zb1l.stage_call` from the two models' words, and the verdict recomputed by `zap3.verdict` from the two calls, equal the record's; and b1y's level minus b1n's level (AP2's, section 9 of its
document) equals the paired difference within 0.03 points on all 8 pairs (two settings, two splits, two models).

Levels are R@5 in points (means over the three seeds' fits, with the 95% interval over the 1,000 questions); differences are paired by question with their 95% intervals (2,000 resamples, seed 0); the words are
section 4's. b1n's levels are AP2's, filled into these copies of the report's tables (the committed `report.md` prints "--" in that column, as the fixture run did; the difference columns decide).
In-domain = the fits whose training split is the setting's partner; held-out = the others.

### In-domain

| setting | model | b1t | b1w | b1n | b1y | b1y - b1n | b1y - b1w | b1y - b1t | word (b1y - b1n) |
|---|---|---:|---:|---:|---:|---|---|---|---|
| hotpotqa | zg6r | 91.1 (90.0-92.3) | 94.3 (93.3-95.3) | 94.3 (93.3-95.4) | 95.1 (94.1-96.0) | +0.8 (+0.4, +1.2) | +0.8 (+0.3, +1.3) | +4.0 (+3.1, +4.8) | GREATER |
| hotpotqa | zrckpn | 90.4 (89.2-91.6) | 93.4 (92.4-94.5) | 93.5 (92.4-94.5) | 94.2 (93.2-95.2) | +0.8 (+0.4, +1.1) | +0.8 (+0.4, +1.2) | +3.9 (+3.0, +4.8) | GREATER |
| musique | zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | 68.7 (67.0-70.4) | 68.6 (66.9-70.3) | -0.1 (-0.5, +0.3) | -0.4 (-0.9, +0.2) | +2.6 (+1.5, +3.8) | SAME |
| musique | zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | 67.2 (65.5-68.9) | 66.7 (65.0-68.4) | -0.5 (-1.0, -0.1) | -0.7 (-1.3, -0.2) | +2.2 (+1.0, +3.3) | LESS |

### Held-out

| setting | model | b1t | b1w | b1n | b1y | b1y - b1n | b1y - b1w | b1y - b1t | word (b1y - b1n) |
|---|---|---:|---:|---:|---:|---|---|---|---|
| hotpotqa | zg6r | 89.4 (88.2-90.6) | 92.9 (91.9-94.0) | 92.9 (91.8-94.0) | 93.5 (92.5-94.5) | +0.6 (+0.3, +1.0) | +0.6 (+0.1, +1.0) | +4.1 (+3.2, +5.0) | GREATER |
| hotpotqa | zrckpn | 88.6 (87.4-89.9) | 92.2 (91.2-93.4) | 92.1 (91.0-93.2) | 92.7 (91.6-93.8) | +0.7 (+0.3, +1.0) | +0.5 (+0.1, +0.9) | +4.1 (+3.2, +5.0) | GREATER |
| musique | zg6r | 62.1 (60.4-63.7) | 60.8 (59.1-62.5) | 61.1 (59.4-62.8) | 61.2 (59.5-62.9) | +0.1 (-0.4, +0.7) | +0.4 (-0.2, +1.1) | -0.8 (-2.2, +0.5) | SAME |
| musique | zrckpn | 60.0 (58.4-61.7) | 61.8 (60.1-63.6) | 62.2 (60.5-64.0) | 62.0 (60.3-63.7) | -0.1 (-0.7, +0.4) | +0.2 (-0.4, +0.8) | +2.0 (+0.7, +3.3) | SAME |

### The setting's call (in-domain, both models)

| setting | call |
|---|---|
| hotpotqa | GREATER |
| musique | MIXED |

### zg6r - zrckpn at b1y, paired by question

| setting | read | zg6r - zrckpn (CI) | label |
|---|---|---|---|
| hotpotqa | in-domain | +0.9 (+0.4, +1.3) | GNN_ABOVE |
| hotpotqa | zero-shot | +0.7 (+0.2, +1.3) | GNN_ABOVE |
| musique | in-domain | +1.9 (+1.1, +2.8) | GNN_ABOVE |
| musique | zero-shot | -0.8 (-1.7, +0.1) | LEVEL |

### MuSiQue by hop depth (in-domain, paired by question, descriptive)

| hops | questions | model | b1y - b1n |
|---|---:|---|---|
| 2hop | 518 | zg6r | +0.1 (-0.6, +0.8) |
| 2hop | 518 | zrckpn | -0.4 (-1.1, +0.3) |
| 3hop | 316 | zg6r | -0.6 (-1.2, +0.1) |
| 3hop | 316 | zrckpn | -1.0 (-1.9, -0.2) |
| 4hop | 166 | zg6r | +0.3 (-0.4, +1.1) |
| 4hop | 166 | zrckpn | +0.1 (-0.5, +0.9) |

### The structural files

| setting | df cap | R pairs | pairs added | pairs in the carve |
|---|---|---:|---:|---:|
| hotpotqa | 3 | 4315 | 357 | 4672 |
| musique | 25 | 41429 | 1864 | 43293 |

Missing reads: 0.

**What Part B says.**

- **The verdict is HOLD**, by section 4's rule: HotpotQA's call is GREATER and MuSiQue's is MIXED (one model is LESS and the other is not). HOLD keeps R in the structural slot of the after-paper read path on both
  settings and names what fell: zrckpn on MuSiQue in-domain, b1y - b1n -0.5 [-1.0, -0.1] (LESS; the top of the interval is 0.1 points below zero, a thin margin), with b1y - b1w -0.7 [-1.3, -0.2]; zg6r on the
  same reads is SAME (-0.1 [-0.5, +0.3]).
- **HotpotQA (cap 3) is GREATER on both models, in-domain and held-out.** In-domain zg6r goes from 94.3 to 95.1 (+0.8 [+0.4, +1.2]) and zrckpn from 93.5 to 94.2 (+0.8 [+0.4, +1.1]); held-out +0.6 [+0.3, +1.0]
  and +0.7 [+0.3, +1.0]. Against b1w (R with spaCy's N) the in-domain differences are +0.8 [+0.3, +1.3] and +0.8 [+0.4, +1.2], both GREATER. Part A counted the carve's reach over b1n's union at 79 new pairs and
  3 gold-pair slots (+0.3 points), so the reads move by more than that reach: 278 of the 357 added pairs were already in the union (through F0T's or the kNN file, whose files are b1n's byte for byte), and nothing
  here separates the 79 new pairs from the move of the other 278 into the structural slot (no run for it was declared).
- **MuSiQue (cap 25) is MIXED.** In-domain zg6r is SAME (68.7 to 68.6) and zrckpn LESS (67.2 to 66.7); held-out both are SAME (+0.1 [-0.4, +0.7] and -0.1 [-0.7, +0.4]). By hop depth (descriptive): zrckpn's fall
  sits in the 316 three-hop questions (b1y - b1n -1.0 [-1.9, -0.2], LESS) and is SAME on the 518 two-hop (-0.4 [-1.1, +0.3]) and the 166 four-hop (+0.1 [-0.5, +0.9]) questions; zg6r is SAME at every depth.
  Part A found that the 897 added pairs new to the union link no chain slot, so on MuSiQue the carve adds edges without gold reach; why one model's read falls by half a point is not examined here.
- **The expectation of section 5 did not hold.** It said an effect of a few tenths of a point at most and NO_GAIN likely; found: HOLD, with +0.8 points on both models on HotpotQA and -0.5 on one model on MuSiQue.
  What held is the placement of the rungs (Part A: none on 2Wiki, a shallow one on HotpotQA, the deepest capped one on MuSiQue).
- **The gap GNN - MLP at b1y** (reported): HotpotQA +0.9 [+0.4, +1.3] in-domain and +0.7 [+0.2, +1.3] held-out (GNN_ABOVE both; at b1n +0.9 and +0.8); MuSiQue +1.9 [+1.1, +2.8] in-domain (GNN_ABOVE; at b1n +1.5)
  and -0.8 [-1.7, +0.1] held-out (LEVEL; at b1n -1.1 [-2.0, -0.1], MLP_ABOVE). Of the four labels only MuSiQue's held-out one changes.

**What it can and cannot show** (section 5 applies in full).

- A read-time shift: the models were fitted with R in the structural slot, so the HotpotQA gain and the MuSiQue fall say what these fitted models do when the slot holds Ry; neither says how a model fitted on Ry
  would read (D79).
- Selection: S1 chose each rung on the gold pairs of the same 1,000 questions the reads grade, so the HotpotQA word carries that selection (one cap out of ten, inside a family fixed in advance); B1's settings hold
  no other questions to read on.
- One setting carries the GREATER (HotpotQA), and MuSiQue's LESS rests on one model's in-domain read with a thin margin (its held-out reads and the GNN's reads are SAME).

**What it decides.** The declared verdict, HOLD: Ry is not carried into the read path, and b1w's rows remain the record; no number of the paper, no model, pool or column of record changes. HotpotQA's GREATER is
filed as a finding of this form (a read-time substitution of R by Ry at S1's caps), not as an adoption.

**Open after Part B** (the user's; my default first): **D79** (a refit with Ry in the training carves; default: not now, with D78) is where HotpotQA's carve (cap 3) would be the first candidate. **D82** (added with
this filing): use Ry at cap 3 for HotpotQA alone in the read path, against the declared verdict. Default: no, since the verdict rule is declared and a per-setting adoption would be a rule made after the numbers, on
the questions the rung was chosen on. D80 and D81 are unchanged.

Files: `outputs/ap3/report.json`, `report.md` and `calls.json` (committed); the 24 read files (`outputs/b1y/fits/h/<fit>/reads/<setting>__b1y.npz`) and the two settings records (`outputs/b1y/settings/`) are
the host's files fetched and stay local; the carve folders are rebuilt by `zap3.py bench`.
