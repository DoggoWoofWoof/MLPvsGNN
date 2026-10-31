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

Filed 31 October 2026, 17:05 IST, from `python -I outputs/mp_unified/b1f.py report` (`outputs/b1f/report.json`, `report.md`) on the
24 reads fetched from the host: the 26 feeder items (fork, look, cache, link build, read) ran in 23 minutes, 0 failed, 0 missing.
Nothing in sections 1 to 6 changed after the reads. R@5 x100, candidate p@swa, 95% bootstrap intervals (2,000 resamples, seed 0).

**Verdict: W2_S_NOT_ADOPTED. The plain title rule (b1u) stays the paper's recreated-links row; b1w is reported beside it.**

Section 4's rule, in-domain (J5), paired difference b1w - b1u on R@5:

| setting | MLP (zrc) | GNN (zsp) | call |
|---|---|---|---|
| 2wiki | -1.1 (-1.6, -0.7) | -0.6 (-1.0, -0.3) | **LOSS** (both upper ends below 0) |
| hotpotqa | +0.0 (-0.4, +0.5) | -0.6 (-1.1, -0.1) | **LOSS** (the GNN's upper end is below 0); not a GAIN either way |
| musique | -0.8 (-1.5, 0.0) | -0.6 (-1.4, +0.2) | NO_CHANGE (the MLP's upper end prints 0.0 at four decimals, not below it) |

The verdict needs hotpotqa to be a GAIN and neither 2wiki nor musique a LOSS; hotpotqa is not a GAIN and 2wiki is a LOSS.

Reported, not decided (J5; MLP / GNN):

| | 2wiki | hotpotqa | musique |
|---|---|---|---|
| b1 (hyperlinks; musique: title links) | 96.4 / 97.2 | 98.2 / 98.2 | 66.1 / 66.3 |
| b1u (plain title rule) | 95.3 / 95.9 | 94.8 / 95.0 | 66.1 / 66.3 |
| b1w (W2_S) | 94.2 / 95.2 | 94.8 / 94.3 | 65.3 / 65.7 |
| b1t (no links) | 75.6 / 77.3 | 88.7 / 90.0 | 63.4 / 62.9 |
| share of the hyperlinks' gain recovered, b1u | 0.95 / 0.93 | 0.65 / 0.60 | 1.00 (identity) |
| share recovered, b1w | 0.89 / 0.90 | 0.65 / 0.53 | 0.72 / 0.82 |
| b1 - b1w, the price left (CI) | +2.2 (+1.4, +3.1) / +1.9 (+1.1, +2.8) | +3.4 (+2.5, +4.2) / +3.9 (+3.0, +4.9) | +0.8 (-0.0, +1.5) / +0.6 (-0.2, +1.4) |

- W2_S still gains a lot over no links (b1w - b1t: 2wiki +18.6 / +17.9, hotpotqa +6.1 / +4.4, musique +1.9 / +2.7, every interval above 0).
  It does not gain over the plain rule.
- Against the best published row of the setting, no b1w read changes a call from b1u's: all six are BELOW both the best row of any kind and the best
  peer-reviewed row (2wiki: EviReform 98.5, G-reasoner 98.2; hotpotqa: BridgeRAG 98.8, PropRAG 97.4; musique: SAG2 80.7, PropRAG 78.3). The hyperlink
  reads are AT BridgeRAG and ABOVE PropRAG on hotpotqa and BELOW every best row on 2wiki and musique.
- Held-out reads (decide nothing), b1w - b1u: 2wiki -2.3 / -1.7, hotpotqa -1.9 / -1.8, musique +0.3 / +1.0 (the GNN's interval, +0.1 to +2.0, is above 0).

**What this says.** On B1, an edge set that matches the hyperlinks better as a set (hotpotqa precision 0.82 against the plain rule's 0.65, recall 0.80
against 0.82; 2wiki 0.83 against 0.76 and 0.84 against 0.84) is not read better by the fits that were trained on the hyperlinks. The remaining price of
recreated links, 1.9 to 2.2 points on 2wiki and 3.4 to 3.9 on hotpotqa, is not the precision or recall of the edge set as a whole. **Untested** reasons,
named so nobody reads them as findings: the fits were trained on the hyperlink graph and W2_S differs from it in degree, direction and in its sibling
links (musique's hub reaches degree 1,467); the lost edges that matter are the few that join a question's bridge passages, not the average edge.

**What this closes and what it does not.** It closes one form: U2b's chosen rule, cut to B1's nodes, read by hyperlink-trained fits. It does not close
W2_S for the full corpora (U2b chose it on pool reach there), a fit trained on W2_S, or the higher-reach forms (W3e_S, W7eg_S: hotpotqa reach 0.925
against 0.909 on s1sel), which are not stored on the laptop. The paper's row stays b1u.
