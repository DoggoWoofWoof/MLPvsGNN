# U1a: B1 without the corpus hyperlinks, and with one LLM-free link rule

Declared 9 October 2026, about 19:40, before any number. This is the first of the three U1 stages the user asked for
(the hyperlink caveat of docs/B1B_MODELS_ON_HIPPORAG2.md):
1. **U1a**, this file: read the B1b fits with the hyperlinks removed, and with one universal rule in their place;
2. **U1b**: the same rule on all six full corpora, scored first on how well it recovers the hyperlinks;
3. **U1c**: retrain and grade on all six, zero-shot included.

U1a trains nothing and chooses nothing. It reads B1b's eight fits on two new versions of B1a's three settings.

## Why

On 2wiki and hotpotqa, B1b's `structural` family is the corpora's own Wikipedia hyperlinks. HippoRAG 2 does not use
them, and they are curated links. The question this stage answers: how much of B1b's lead over HippoRAG 2 rests on
them, and how much an LLM-free rule built from the passages alone gets back.

The user's direction: the headline must stand without hyperlinks. The hyperlink rows are reported beside it, labelled
as curated links.

## The two variants

Each variant changes only `graph_structural.npz` of each setting. Everything else is B1a's file, byte for byte:
- the passages, the questions and the golds;
- the first stage, and the laptop's lists;
- the ner and knn families.

**b1t (text only).** `structural` is empty on all three settings. The graph is ner + knn, both built from the passages'
text by B1a's rules.

**b1u (universal links).** `structural` is the title-mention rule, on all three settings alike. It is the rule that
built musique's and squad's package graphs (CRAG scratchpad/c3_derived.py):
- an edge A → B iff A's text contains B's title as a phrase, word-bounded and case-insensitive;
- titles shorter than 4 characters never match;
- A ≠ B, and a pair is kept once;
- the rule reads the setting corpus's own titles and texts (B1a's raw corpus files);
- untyped (one relation, 0), no weights, as B1a's structural.

The rule has no parameter fitted to anything; the 4-character floor is c3_derived's, unchanged. On these settings the
rule needs no candidate index. Every title is matched against every text with c3_derived's selective-token index, so the
edges are exactly the ones c3_derived's loop emits on the same documents.

## Checks, before any read

- **musique, the identity check.** musique's B1 structural graph is the package's title-mention graph induced on the
  setting. b1u's musique graph must equal it, up to passages whose text differs from the package's only in
  whitespace (B1a's `exact` flag). A difference outside those passages stops U1a. Exact equality is expected.
- **The variants' other files** are checked against B1a's build.json shas.
- **Lists.** The look's lists (b1_lists.npz) are B1a's. They are read from B1a's folder and checked against B1a's
  build.json for the files they depend on (everything but the structural graph).

## Reported before the reads: does the rule recover the hyperlinks?

On 2wiki and hotpotqa, the title-mention edges against the hyperlink edges, on the setting:
- precision: the share of rule edges that are hyperlinks;
- recall: the share of hyperlinks the rule finds;
- both as directed pairs and as undirected pairs;
- the degree tables (mean, max, isolated) of both graphs.

This is the evidence for "the LLM is not needed to build these links". It is reported whatever it shows.

## What runs

On the host, through the feeder, as B1b's items with new names:
1. **Looks:** `look_b1u.py --variant t|u`. It is look_b1 with the setting folder and the carve name swapped: carves
   b1t and b1u, written to `outputs/mp_unified/look/<dataset>/b1t` and `/b1u`.
2. **Caches:** lean_cache, and zlink's edge build, each with `--carve b1t` and `--carve b1u`.
3. **Fit copies:** `u1a.py fork` copies each of B1b's eight fit sources into `outputs/u1a/fits/<variant>/<arm>/<split>`
   (the arms: zrc and zsp; the splits: J5, L-musique, L-2wiki, L-hotpotqa), so no read writes into B1b's copies.
4. **Reads:** the same commands as B1b's reads, on each copy, with `--read musique=b1<v>,2wiki=b1<v>,hotpotqa=b1<v>`.

## The report

`u1a.py report` writes `outputs/u1a/report.json` and `report.md`. Per setting, per carve (b1, b1t, b1u), arm and read:
- R@5, FC@5 and hit@1 (candidate p@swa, as B1b), with 95% bootstrap intervals (2,000 resamples, seed 0);
- the call against HippoRAG 2's published R@5;
- the rrf reference on each carve.

**Paired differences:** b1t − b1 and b1u − b1 per question, and b1u − b1t, each with its bootstrap interval. b1 is
B1b's filed reads (outputs/b1b/fits).

**How it is read:**
- The headline row for Claim 2 becomes **b1u**: an LLM-free graph built by one rule on every dataset, and no curated
  links.
- b1 stays as the labelled "+ corpus hyperlinks" row.
- b1t is the floor.
- These are the B1b fits, trained on the package graphs, which carry hyperlinks on 2wiki and hotpotqa. On b1t and b1u
  they read a graph unlike their training graph. A drop here is therefore partly a training shift, and U1c's retraining
  is the fair form. U1a states the read as is and does not adjust it.

## Files

    outputs/mp_unified/u1a.py                      settings (laptop), fork, report, --selftest (new)
    outputs/mp_unified/look_b1u.py                 the looks on a variant (new)
    outputs/bench/hipporag2_t/<dataset>/           b1t's setting files (B1a's, structural emptied) + build.json
    outputs/bench/hipporag2_u/<dataset>/           b1u's setting files (B1a's, structural by the rule) + build.json
    outputs/bench/hipporag2_u/recovery.json        the rule against the hyperlinks; the musique identity check
    outputs/u1a/fits/<variant>/<arm>/<split>/      the fit copies and their reads
    outputs/u1a/report.json, report.md             the table

## Results

Run 9-10 October 2026 (the zsp reads 00:10-00:30, the zrc reads 01:08-01:12 once the GPU was free); missing 0.
Records: outputs/u1a/report.json and report.md. R@5 x100 on B1's settings, 95% bootstrap intervals; HippoRAG 2's
published numbers beside them.

**The rule on B1's settings.** It gives musique's graph exactly (0 of 45,568 row pairs differ). It recovers the
hyperlinks at P 0.76 / R 0.81 on 2wiki and P 0.66 / R 0.79 on hotpotqa (directed).

| setting | read | hyperlinks (b1) | no links (b1t) | the rule (b1u) | HippoRAG 2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2wiki | zsp/J5 | 97.2 ABOVE | 77.3 BELOW | **95.9 ABOVE** | 90.4 |
| 2wiki | zrc/J5 | 96.4 ABOVE | 75.6 BELOW | **95.3 ABOVE** | 90.4 |
| 2wiki | zsp/L-2wiki | 93.4 ABOVE | 77.1 BELOW | 92.2 ABOVE | 90.4 |
| 2wiki | zrc/L-2wiki | 93.3 ABOVE | 75.6 BELOW | 92.2 ABOVE | 90.4 |
| hotpotqa | zsp/J5 | 98.2 ABOVE | 90.0 BELOW | **95.0 BELOW** | 96.3 |
| hotpotqa | zrc/J5 | 98.2 ABOVE | 88.7 BELOW | 94.8 BELOW | 96.3 |
| hotpotqa | zsp/L-hotpotqa | 95.9 AT | 89.1 BELOW | 93.8 BELOW | 96.3 |
| hotpotqa | zrc/L-hotpotqa | 96.5 AT | 88.5 BELOW | 94.2 BELOW | 96.3 |
| musique | zsp/J5 | 66.3 BELOW | 62.9 BELOW | 66.3 BELOW (identical) | 74.7 |
| musique | zrc/J5 | 66.1 BELOW | 63.4 BELOW | 66.1 BELOW (identical) | 74.7 |

Paired differences (zsp/J5; the other three reads agree in sign, intervals in report.md):
- the rule over no links (b1u - b1t): +18.5 on 2wiki, +5.0 on hotpotqa, +3.3 on musique;
- the rule against the hyperlinks (b1u - b1): -1.3 on 2wiki and -3.3 on hotpotqa.

What this says:

- **The lead over HippoRAG 2 on 2wiki does not need the hyperlinks.** One LLM-free rule built from the passages alone
  keeps the GNN 5.5 points above HippoRAG 2 (95.9 against 90.4), and the MLP 4.9 points above it (95.3). It keeps
  94% of the gain the hyperlinks give over no links on 2wiki (18.5 of 19.8), and 61% on hotpotqa (5.0 of 8.2).
- **On hotpotqa the rule falls 1.3 points short of HippoRAG 2** (95.0 against 96.3), where the hyperlinks were 1.9
  above it. The gap is the hyperlinks the rule misses or adds, and U1d works on it.
- **Without any links, the models fall well below HippoRAG 2**: 77.3 on 2wiki, 90.0 on hotpotqa. A few precise edges are
  what the models need, not an LLM-built graph.
- **The MLP stays close to the GNN** under the rule: zrc / zsp = 95.3 / 95.9 on 2wiki, 94.8 / 95.0 on hotpotqa, 66.1 /
  66.3 on musique (J5).
- **musique stays below HippoRAG 2** (66.3 against 74.7) whatever its links: its gap is ranking, as B1b found.

These are B1b's fits, unchanged and trained on graphs with hyperlinks. U1c retrains on the universal graphs (U1d's rule)
and reads B1 again.
