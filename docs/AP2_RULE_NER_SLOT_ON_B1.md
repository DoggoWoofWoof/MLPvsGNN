# AP2: the models of record read with the rule-built NER slot in the place of spaCy's (b1n, b1x)

Declared 12 November 2026, before any AP2 read. Follows AP1 ([AP1_RULE_NAMES_AND_MUSIQUE_REACH.md](AP1_RULE_NAMES_AND_MUSIQUE_REACH.md), filed in
981dab4) and AP1b ([AP1B_SUBSPAN_NAMES.md](AP1B_SUBSPAN_NAMES.md), filed in 3bb8c25; D76 and D77), and B1L's design
([B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md](B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md): `zb1h.py` rebound by import, B1G's `zb1g.py` as the model for a variant
carve). Code: `outputs/mp_unified/zap2.py` (new). Outputs only under `outputs/b1n/`, `outputs/b1x/` and `outputs/ap2/`.

Nothing trains. ML1's and LTM's fits, PKO's chain code, B1P's and B1M's reads of b1w and b1t are read and never written. AP2 decides nothing about
a model, a pool or a column of record. It measures what the models of record get when the NER slot holds a rule-built family.

## 1. Why

The after-paper program (the user, 11 November) starts with "removing spaCy". AP1 and AP1b showed at the graph level that the rule's names, F0T
(U2r's capitalized-run names at the df cap [2, 25], plus each passage's own title as a further name of the passage), link as many of the gold
pairs as spaCy's co-mention family does, and more on MuSiQue and HotpotQA (section 9 of AP1b). A graph-level reach is not a model's recall. The
models of record, zg6r (ML1's GNN) and zrckpn (LTM's MLP), were fitted with spaCy's `ner` store in the NER slot of their training carves. The
question, on HippoRAG 2's three 1,000-question settings (B1): **what do they get when the NER slot of b1w (R + N + K) holds F0T instead of N, with no
new fit?**

## 2. What is read

| carve | structural (R) | NER slot | K | settings | role |
| --- | --- | --- | --- | --- | --- |
| b1t | none | spaCy's N | kNN | musique, 2wiki, hotpotqa | B1P's and B1M's reads; reference |
| b1w | W2_S | spaCy's N | kNN | the same | B1P's and B1M's reads; the base |
| **b1n** | W2_S | **F0T** | kNN | the same | new: Part A |
| **b1x** | W2_S | **F3T** | kNN | musique | new: Part B (D77's default) |

- **The folder.** b1w's six files (`outputs/bench/hipporag2_w/<setting>/`) byte for byte except `graph_ner.npz`, which holds the form's pairs (`src` < `dst`,
  int32, ascending by (src, dst), as b1w's NER file is) and their `weight` (float32). `build.json` is b1w's with the NER file's sha replaced, its `u1a` block
  relabelled (the carve, the variant, this script's sha256; `b1a_structural_sha256` kept as b1w's block holds it) and an `ap2` block (the form and its level tables,
  the pair counts, the AP1b report's sha256 and its `levels.npz` sha256, the sha of b1w's NER file that B1a's lists name).
- **The forms** are AP1b's (`outputs/ap1b/<setting>/levels.npz`; AP1b's report holds each form's pair count). F0T = F0 (U2r's CN_all names) united with
  the level-0 pairs of the names with each passage's own title as a name; F3T = the deepest form (end-anchored and infix sub-spans of two or more
  words, one-word sub-spans, and the titles). A pair's weight is the co-mention weight, the sum of 1/df over the names the two passages share (df
  in [2, 25]), as spaCy's file carries (its weights run from 0.04 = 1/25 up, as the levels' do); where several of a form's level tables hold a pair its weight is
  the deepest table's (at a level, the table with the title before the one without).
- **The models.** zg6r and zrckpn, seeds 0 to 2, fitted on each of the two training splits (WQ-musique, WQ-hotpotqa): 12 reads per setting and carve,
  as B1L's. In-domain = the mean over the fits whose training split is the setting's partner (`zb1h.IN_DOMAIN`: musique reads the WQ-hotpotqa fits, hotpotqa
  the WQ-musique fits, 2wiki both); held-out = the others. The questions are the setting's 1,000.
- **The lists.** B1a's dense and SPLADE lists do not depend on any graph family; `look_b1u.load_lists` checks the setting's other files against the shas
  they name and the replaced structural file against `u1a.b1a_structural_sha256`. AP2's loader (`zap2.lists_of`, its check `zap2.check_lists`) does the same and
  pins the NER file the same way: the lists name the NER file that b1w holds (B1a's own), recorded in `ap2.b1a_ner_sha256`, so a read cannot run on lists built
  for other files. The host's mirror has no SPLADE document vectors, so every host step reads B1a's lists through this loader.

## 3. Gates (a failed gate ends the stage; no read is filed past one)

G1: `levels.npz` loads for the setting, and the built pair count of each form equals AP1b's report (F0T: 98,845 MuSiQue, 42,247 2Wiki, 86,518 HotpotQA;
F3T: 231,258 MuSiQue). G2: the five files other than the NER file equal b1w's `build.json` shas, and b1w's NER file equals B1a's (the lists name it). G2b: the
folder as written carries its own `build.json`'s shas, and B1a's lists check accepts it (the structural and the NER file as the two it replaced). G3: the
NER file has `src` < `dst`, ascending distinct pairs, finite weights at least 0.04 and no pair outside the setting's passages. G4: the node count equals
`build.json`'s. G5: PKO's gate on the carve (`zap2.py gate`, on the host) passes on every setting. G6: the rows recomputed from B1P's and B1M's reads equal
B1M's filed rows, as B1L's check. The self-test passes before any row runs.

## 4. Calls, and what the stage decides

All differences are paired by question, 95% bootstrap intervals over the 1,000 questions (2,000 resamples, seed 0: `b1b.boot_ci`), no threshold.

- **A model's word** on b1n - b1w (in-domain): **GREATER** when the bottom of the interval is above 0, **LESS** when the top is below 0, **SAME** otherwise.
  SAME means only that the interval holds 0; it is not an equivalence.
- **A setting's call**: GREATER, SAME or LESS where the two models agree; **NOT_LESS** where they differ and neither is LESS; **MIXED** where one is LESS
  and the other is not; INCOMPLETE where a read is missing (B1L's `stage_call`).
- **The verdict, Part A: REPLACE** when no setting's call is LESS or MIXED (six in-domain reads, none below), **HOLD** when a setting's call is LESS or MIXED
  (decided even while another setting's read is missing), **INCOMPLETE** while a read is missing and none is LESS or MIXED. REPLACE says the rule-built
  NER slot may stand in the place of spaCy's in the read path of the after-paper program (AP3 builds on it); HOLD keeps spaCy there and names the
  setting and model that fell. This is the WWW rule's shape ("no LOSS among the six deciding reads"), written for paired reads.
- **Reported, not deciding:** the held-out reads; b1n - b1t and b1w - b1t; MuSiQue by hop depth (2, 3, 4 hops; the setting where chain slots matter);
  the gap GNN - MLP at b1n, as B1L reports it beside.
- **Part B (b1x), descriptive:** b1x - b1w and b1x - b1n on MuSiQue, the same words, **no verdict**: S1 refused F3T's price (AP1b section 9) and
  AP2 does not reinterpret it; the read shows what the extra reach (F3T links 1,272 of 1,648 MuSiQue chain slots, F0T 1,173) is worth to a model.

## 5. What it can and cannot show

- It is a **read-time shift**: the models were fitted on spaCy's NER store, so a LESS or SAME here does not say F0T is a worse or an equal training input.
  A refit on carves whose NER slot is rule-built (training carves for all five training datasets rebuilt without spaCy, then the fits) is the user's call (D78;
  default: not now).
- The intervals are over questions, not over seeds: the seeds enter as the mean over fits, as B1L's.
- **Dev-read disclosure.** AP1b chose its forms after a scratch read of MuSiQue's chain labels (D75), so the MuSiQue reads are a dev read of the form. 2Wiki and
  HotpotQA were not looked at for the design; they are the held-out check of the rule.
- Nothing here changes a number of the paper or of any record: b1w's rows remain the record.

## 6. Rows

One block per carve, B1L's rows mirrored (`zb1l`'s names with `b1n` / `b1x`): selftest, settings, fork, chain (per setting, 2 cpus), gate, lists, columns (both
splits), 12 reads per setting (GPU reads 1.1 cpus, GPU share 0.15; MLP reads 6 cpus). Memory is reserved at the measured peak of b1w's same step (B1P's and
B1M's records) scaled by the union's size, B1L's rule (b1n's union holds 1.12 to 1.19 times b1w's pairs, b1x's 2.18 times); the host's measured peak replaces it in
the record. Part B's rows queue after Part A's reads. The report (`zap2.py report`, on the laptop) after the reads.

## 7. Decisions (the user's; my default first)

- **D77** (opened in AP1b section 9): Part B reads F3T on MuSiQue as a disclosed second carve. Default yes, as declared here; the alternative is Part A alone.
- **D78:** a refit with a rule-built NER slot is not part of AP2. Default: not now; after AP2's reads and the user's say-so.

## 8. Development log (disclosed before any read number)

- `zap2.py selftest` (laptop, 12 November, `--variant n` and `--variant x`): the form's pair builder (the union of the level tables, the deepest table's weight,
  `a < b`), the lists check (accepts the replaced structural and NER shas, refuses any other difference), the carve's `build.json`, the settings record's label, the
  paired differences, words, setting calls and the verdict on synthetic reads, and the report's markdown. Passes.
- `zap2.py bench` (laptop, 12 November 01:04 IST, and again at 01:10 after the last code change, so that each folder's recorded script sha is the committed
  file's; the NER files' shas are the same in both runs): the four folders (b1n on MuSiQue, 2Wiki and HotpotQA; b1x on MuSiQue). G1 to G4 and G2b passed. No model, no fit.
- The lists loader (`lists_of`) was run on the four folders against B1a's real lists record (a stand-in for the look modules, its own path only): it loads B1a's
  lists for each and refuses a changed NER sha.
- `zap2.py report` was run once on a temporary fixture (b1n's and b1x's reads made as copies of b1w's reads, so every difference is 0; stub settings records): the
  plumbing, the 92 rows equal to B1M's filed rows, and the report's files. Nothing there is a result; the fixture folders were removed.
- No laptop pass of `settings`, pools or look (B1L's practice: they run on the host; the laptop keeps 24 GB of disk free). `selftest` and `settings` are the block's
  first host items and the chain waits for `settings`. A failing item is fixed, and the fix is disclosed here with its commit, before any read number.
