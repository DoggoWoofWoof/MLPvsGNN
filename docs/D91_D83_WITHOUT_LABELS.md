# D91: D83 without the labels (the same refit, links once per family)

Declared 17 November 2026, 04:55 IST, before any D91 link, fit or read is built. Code: `outputs/mp_unified/zd91.py` (the
links and the arms); the fit, read and report files follow, each committed before its step runs, none changing a rule on
this page.

## 1. Why

D83 (docs/D83_UNIFIED_R_REFIT.md) makes three changes in one refit: F0T in R's NER store, the S > R > K dedup, and edge
labels the models read. Its section 9 names D91, "D83 without the labels", as the form that separates the labels from the
other two, by default only if D83 is NOT_ADOPTED "or on the user's word". The user's word, 17 November 2026, after AP4's E3
did not hold (pairs with several labels were not more precise than single-label pairs on every dataset): "i just wanted the
rule name which got that link like title match etc not each link to have its own label, just the rule which made that link",
and on running D91 beside the encoder comparison: "do both if host has space we have a lot to do anyways".

**What D83's labels are.** Each D83 label is the name of a rule that made a link (`S`, `R.mention`, `R.name_comention`,
`R.title_comention`, `K.knn`; D83 section 3). A link carries only the rules that made its pair; a pair two rules found is
linked once for each of them. No link has a label of its own. That is the form the user describes, so D83 is not changed.
D91 drops the rule names: one link per family whose store holds the pair (0 the slot, 1 the NER store, 2 the knn store), as
the record links its pairs. D83 against D91 then measures what reading the rule names adds, with F0T and the dedup equal on
both sides.

## 2. What D91 is

- **Equal to D83 (the files themselves, hard-linked from D83's arms after a sha256 check, never rewritten):** the pools of
  record, the KT rows, the look's records, the bases, the lean cache, rmatch and zrc, the kt rows. So F0T, the dedup, every
  pool, every cache row and every first-stage input are D83's.
- **D91's own:** the links (one per question, u, v and family, zlink's form: FAMS = 3, the record's settings), the typed
  sets on the KB carves (on the family-0 links, which are D83's S links), the carve records, the gate.
- **The models:** the record's model code with no D83 change: G6's maps, attention biases, family means and family dropout
  over three families; LTM's candidate features with the record's three family bits (ltwdiag unedited); no LAB. zrckpn
  and GNX read NEXT only, as in the record, so the same-input pair holds.
- **Fits:** D83's 20 (zspk at seed 0; zg6r, GNX and zrckpn at seeds 0, 1 and 2; both WQ splits; the same epochs, widths,
  optimiser, seeds, starts and candidate), in a world of its own, `d91world` (outputs/pko = `outputs/abl/D91F`), with LTM's
  lists, fold fits and NEXT columns made again on D91's links.

## 3. The family links, derived from D83's labelled links

D83's look chunks were pruned on 17 Nov at 00:13, with the user's go ("remove whatever is not required"), after every step
that reads them was filed. So D91 does not run zlink on the chunks. It derives the family links from D83's labelled links
(`zd91.derive_part`). D83's links hold every pool edge of the look, once per label the pair carries (d83stores.label_edges:
no edge dropped). The stores after D83's dedup decide each pair's families:

- **KB (MetaQA, WebQSP).** slot = S; NER = F0T minus S; knn = knn minus (S and NER). The three stores are disjoint, so a
  linked pair's family is the one store that holds it: an S label gives family 0; otherwise an R label gives family 1;
  otherwise K.knn gives family 2. Exact.
- **Passages (SQuAD, HotpotQA, 2Wiki, MuSiQue; B1's settings).** slot = W2_S (R.mention); NER = F0T (R.name_comention,
  R.title_comention), with no dedup against the slot (both are R); knn = knn minus (slot and F0T). A pair with the slot
  label and no NER label is in the slot only. A pair with an NER label and no slot label is in the NER store only. A K.knn
  pair with neither is in knn only. All exact.
- **Pairs in both the slot and F0T.** Such a pair has a family-f edge when the look kept it. The look keeps, per pool node
  and family, the node's first 64 in-pool neighbours (m3b_features: the cap of 64; position order for the slot, weight then
  position for NER). A node with 64 or more in-pool neighbours of a family gets 64 edges of it, so it carries 64 or more
  links with that family's labels. So an end with 63 or fewer such links has every in-pool neighbour of the family as an
  edge: the pair has the family-f edge. (Passage stores hold no self-loop, so no slot is taken by the node itself.) Only a
  pair whose two ends both carry 64 or more links of the family's labels is undecided: an **open** pair.
- **Open pairs** are decided by the look's own function (`typed_pool_edges` for the slot, `weighted_pool_edges` for NER, cap
  64, through the look's lookup table) on the question's pool (the cache part's `node` rows, which are the look's pool
  rows in order) and D83's store:
  - the slot store: U2c's store of the W2_S graph on D83F and D83E, `look_b1.setting_store` on b1v's folder for b1x;
  - the NER store: F0T (zd83.f0t_pairs) on D83F and D83E, `setting_store` for b1x.

  Surveyed before this page: no open pair on SQuAD or on b1v's 2Wiki and HotpotQA; 580 to 10,188 directed open pairs on
  the 2Wiki and HotpotQA carves (out of 24.8M to 85.4M directed pairs); 57,992 on b1v's MuSiQue (of 65.7M). D83F's and D83E's MuSiQue carves
  are counted by their link rows.

## 4. Arms and builds (CPU)

- **D91F** (from D83F: squad, hotpotqa, 2wiki `fit`; musique `s1fit`; webqsp `fit`), **D91E** (from D83E: the six s1eval
  carves), under `outputs/abl/`. Each is built on ABL's machinery, zabl unedited, with its record step and PKO's gate.
- **b1x** (from b1v: B1's three settings), under `outputs/b1x`. The carve keeps b1v's name and files, so its pools,
  manifests and cache records are b1v's; zb1h's record step and gate are rebound to b1x's root, as zd83b rebinds them for b1v.
- **Per carve, one row** (`zd91.py arm --arm A --dataset D`):
  1. hard-link D83's files listed in section 2 (a file already present must be the same file or the same sha256);
  2. the family links per cache part (derivation, open pairs decided), each record tied to its cache part's record, with
     a `d91` block (D83's links record, the counts, the open pairs and how each was decided);
  3. on a KB carve, the typed sets moved from D83's S links to the family-0 links (the sets file the same bytes);
  4. the carve record (zabl's or zb1h's record step).

  Then the gate per arm.
- **LTM** (in the world): lists, fold fits and NEXT columns on D91F (fits) and on D91E and b1x (reads), with the record's
  features.

## 5. Gates (a failure stops the carve's or the fit's rows; none of its numbers is read)

- **G1d (the derivation reproduces the look).** On every passage part, the family links reproduce the look's filed
  message-edge count (D83's links record, `look_edges.raw`, no self-loop): the sum over families and pool nodes of
  min(64, the node's family links). Checked before this page on first parts: equal on SQuAD (7,615,340), b1v's 2Wiki and
  D83F's 2Wiki (its 96 open pairs did not move the count); 3 short on D83F HotpotQA's first part before its 62 open pairs
  are decided, so the count also checks the decisions. KB parts are
  exact by the disjoint stores (their S stores hold self-loops, so the count does not apply there). Also on every part:
  each question's pairs are D83's pairs; every link's family is 0 to 2; on a KB the family-0 links are D83's S links.
- **G2d (the stores the open pairs are decided on).** Each rebuilt store's pairs equal D83's label sets: the slot's equal
  R.mention, NER's equal R.name_comention plus R.title_comention. Per question with an open pair, the family's recomputed
  edges must contain every pair already decided for the family, and every recomputed edge must be a D83 link with the
  family's label.
- **G3d (equal to D83).** Every hard-linked file is D83's (same file, or sha256 equal). D83's arms are read only and
  nothing in them is written.
- **G4 world, G6 reads:** as D83's section 8 (G5, G6), for d91world.

## 6. Reads, and the rule (fixed now)

- **Reads.** The 18 read fits (zg6r, GNX, zrckpn at three seeds, both splits) on two arms: `B1X` (b1x, B1's three settings;
  deciding) and `D91E` (the six s1eval carves; reported).
- **Cells.** D83's 15 B1 cells (section 7 there): in-domain MuSiQue from WQ-hotpotqa, HotpotQA from WQ-musique, 2Wiki from
  both; held out MuSiQue from WQ-musique and HotpotQA from WQ-hotpotqa; for zg6r, GNX and zrckpn. Per question, R@5 at
  p@swa averaged over the three seeds. Interval: `zap1.boot_diff` (2,000 resamples of the questions, seed 0), 95%:
  **ABOVE**, **LEVEL**, **BELOW**. Seed 0's difference is reported beside.
- **Call L (the rule names).** D83 minus D91 on the 15 cells:
  - **HELP**: at least one ABOVE and none BELOW;
  - **HURT**: at least one BELOW and none ABOVE;
  - **MIXED**: both;
  - **LEVEL**: neither.
- **Call A (D91 itself).** D91 minus the record on the 15 cells (b1x against b1w, the record's filed reads; D83's rule):
  **ADOPT** when no cell is BELOW, **NOT_ADOPTED** otherwise. INCOMPLETE while a cell is unread and none is BELOW.
- **What carries** (the after-paper record's input set for the journal program):
  1. Only D83 ADOPT: D83.
  2. Only D91 ADOPT: D91.
  3. Both ADOPT: D83 when L is HELP or LEVEL (the user asked for the rule names on the edges: they stay unless they cost);
     D91 when L is HURT; the user's call when L is MIXED (**D95**).
  4. Neither: the record stays, and the report names the next form.
- **Reported, not deciding:** the s1eval carves (D91E against D83E and against the record), FC@5 and hit@1, MuSiQue by hops
  and 2Wiki/HotpotQA by kind, the gaps GNX - zrckpn and zg6r - zrckpn, and the per-family link counts beside D83's
  per-label counts.

## 7. Rows, order and cost (estimates; the host's measured peaks replace them)

1. The arms (CPU, now; one row per carve, 14 rows): hard links are free; derivation and open pairs a few minutes per
   carve, MuSiQue's longest. Then the record steps and the gates.
2. The world (`d91world`), then LTM's lists, fold fits and NEXT columns on D91F (CPU).
3. 20 fits. zrckpn's 6 run on the CPU beside the rest. The 14 GPU fits (about 16 to 20 h) take the GPU after D83's last fit
   and ENC's two MPNet encodes. ENC's Qwen3 encode follows them.
4. Reads (CPU, as D83's), then the report.

About 2 days of host time. ETA for the report: about 20 November 2026. Disk: D91's links and typed sets about 10 GB; the
rest is hard-linked.

## 8. What it can and cannot show

D83 against D91 shows whether the models of record read B1's settings better, the same, or worse when each link names the
rule that made it rather than only its family, with F0T and the dedup equal on both sides, at three seeds. It cannot split
F0T from the dedup. It is not an encoder or an LLM comparison. No number of it goes into a paper sentence before section 9
files it.

## 9. Results

(Not yet filed.)
