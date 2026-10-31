# B1f: U2b's link rule (W2_S) on HippoRAG 2's benchmark, read by the existing fits

Declared 31 October 2026, before any read. Follows [U1A_B1_WITHOUT_HYPERLINKS.md](U1A_B1_WITHOUT_HYPERLINKS.md) (b1: the corpus
hyperlinks, b1t: no links, b1u: the plain title-mention rule) and [U2B_LINK_RULE.md](U2B_LINK_RULE.md) (W2_S, chosen as the one
LLM-free link rule for all six datasets). Code: `outputs/mp_unified/b1f.py` (the graph, the fit copies, the report) and
`outputs/mp_unified/look_b1w.py` (the looks on the new variant). Outputs only under `outputs/bench/hipporag2_w/` and `outputs/b1f/`.

Nothing trains. No filed graph, fit or record changes.

## 1. Why

The paper's rule-based row is b1u, the plain title rule of U1a. It gets back 94% of the hyperlinks' gain on 2wiki and 61% on
hotpotqa, and it leaves hotpotqa 3.3 to 3.4 points below the hyperlink reads. U1d, U2b and U2k since built a better rule, W2_S, for
the full corpora. B1 has not seen it. The question: does the rule that was chosen for the full corpora, with no hyperlinks and
nothing fitted on B1, recreate the hyperlinks' gain on B1 more closely than the plain rule? The user's priority of 31 October is the
in-domain read, the recreated links against the original links, and the comparison with the best published row.

## 2. The variant

**b1w.** B1a's three settings with `graph_structural.npz` replaced by W2_S: U2b's stored graph of the full corpus
(`outputs/u2b/<dataset>/graphs`: W2_S.npz for hotpotqa and musique, W2_S.d.npz for 2wiki, whose neighbour count 142,967,492 is
W2_S's in the build record), cut to the setting's nodes through `nodes.json` `package_row`. Both directions of every neighbour pair
are written. Everything else is B1a's file, byte for byte (first stage, ner and knn graphs, nodes, queries); `build.json` is B1a's
with that one sha replaced, and `look_b1u`'s checks (the lists name B1a's other files, and B1a's structural graph) pass unchanged.

## 3. Reads

The eight fits of U1a's reads, copied by `b1f.py fork` into `outputs/b1f/fits/w/<arm>/<split>` (arms zsp and zrc; splits J5,
L-musique, L-2wiki, L-hotpotqa), each read on `musique=b1w,2wiki=b1w,hotpotqa=b1w` with U1a's commands and placements (zsp on CPU
with 6 threads, zrc on the host's GPU through zrct), so b1w and b1u differ in the graph and nothing else the code controls.
Candidate p@swa, R@5, FC@5 and hit@1, 95% bootstrap intervals (B1b's: 2,000 resamples, seed 0).

## 4. Rule, fixed before the reads

Per setting and arm, the in-domain read (split J5), paired difference **b1w - b1u** on R@5, with its bootstrap interval:
- **GAIN** on the setting: the interval's lower end is above 0 for both arms.
- **LOSS** on the setting: the interval's upper end is below 0 for either arm.
- otherwise **NO_CHANGE**, reported with its interval.

**Verdict.** W2_S replaces the plain rule as the paper's recreated-links row (b1u stays beside it) when the setting hotpotqa is a
GAIN and neither 2wiki nor musique is a LOSS. Otherwise b1u stays the row and b1w is reported beside it. The held-out reads
(L-2wiki, L-hotpotqa, L-musique) are reported and decide nothing.

**Reported, not decided:**
- the share of the hyperlinks' gain each rule recovers, (b1u - b1t) / (b1 - b1t) and (b1w - b1t) / (b1 - b1t), per arm and read;
- the paired differences b1w - b1t and b1 - b1w, so the remaining price of the recreated links is read from the same fits;
- the graph against B1a's structural graph: precision and recall, undirected, beside the plain rule's as filed;
- each read's R@5 against HippoRAG 2 and against the best published row of the setting (named in the report), by the bootstrap
  interval.

## 5. What this does not do (stated so a reader does not credit it)

- **A preview was seen first.** On 31 October, before this file, W2_S's edges cut to B1 were compared with the hyperlinks (no model
  read, no question read): hotpotqa precision 0.820, recall 0.804 (the plain rule: 0.649, 0.817); 2wiki 0.825, 0.844 (0.764, 0.843).
  `b1f.py settings` recomputes them. W2_S itself was chosen by U2b on s1sel's pool reach, on the full corpora, before any B1 read.
- **Direction.** b1u's edges were directed (a passage points to the titles it mentions); b1w's are both ways. A difference between
  them is the graph and its direction together.
- **Full-corpus statistics.** W2_S was built on the full corpus, so its gate for common words and its title forms use that corpus,
  not the setting's alone. Cutting it to B1's nodes is a larger corpus's rule applied to a subset.
- **musique changes.** b1 and b1u are musique's package graph (title mentions); W2_S adds the sibling links between passages of one
  article and its forms differ. The cut has 82,858 directed edges against the package graph's 42,823 pairs, with precision 0.924 and
  recall 0.893 against it, and a hub of degree 1,467. The musique read is therefore a different graph, not an identity check.
- **The fits are trained with hyperlinks.** They are the B1b fits, trained on the package graphs. Reading them on b1w is a shift
  from their training graph; retraining on W2_S is not part of this stage.
- **B1's questions are inside s1eval** (the 14 October finding): no fit trained on them, and the reads are the same kind as b1, b1t
  and b1u.

## 6. Files

    outputs/mp_unified/b1f.py                       settings (laptop), fork, report, --selftest (new)
    outputs/mp_unified/look_b1w.py                  the looks on the variant (new)
    outputs/bench/hipporag2_w/<dataset>/            b1w's setting files + build.json
    outputs/bench/hipporag2_w/recovery.json         the graph against the hyperlinks
    outputs/b1f/fits/w/<arm>/<split>/               the fit copies and their reads
    outputs/b1f/report.json, report.md              the table

## 7. Results

(to be filed after the reads)
