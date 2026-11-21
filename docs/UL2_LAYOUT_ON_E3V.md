# UL2: the canonical layout on D101's encoder (e3v), graded against the oracle links before the pools and the refit

Declared 21 November 2026 at 05:04 IST, before any UL2 number exists. Graph level only: nothing trains, no model reads
anything, and no filed graph, pool, fit or record changes. Gold positions and MuSiQue's decompositions are read by the
grading only; no rule reads them.

## 0. The user's words this page follows

- 21 Nov: "we can add the last dataset also into the mix too, improve links, the vectors and try to push more, identify
  the gaps we have, see where we are according to the plan, i do not think we were finished with the links and
  deduplication too".
- 21 Nov, on S: "yeah the S hyperlink should have 2 versions it is fine so keep both dedupes, need to target musique hub
  and find better links".
- 21 Nov, the same ruling's scope: "S ruling was for hyperlink based datasets since we have to show both versions".
- 16 Nov: "i do not think we need to unnecessarily train models after every improvement, like if we created better links
  we can train a model after verifying those links improved connectivity etc by comparing with the oracle links we have
  and across datasets etc and unifying everything, properly check the after paper stuff, spacy removal, musique links
  etc".
- D101 (21 Nov): "yeah this sunds nice, with better links and preprcessing we can cse the gap and then imprve the vectrs
  itsef in semantic space", and "keep the frzen as cmparisn rest yu can recaim back space frm rest encders".

So UL2 is the links-and-dedup step on e3v, graded against the oracle links on all six datasets, with the frozen encoder
beside it as the comparison. One refit follows only after the layout is verified (section 9).

## 1. What is open, and what UL2 closes

- **The encoder moved.** D101 made e3v the encoder. Two families depend on it: K (the record's kNN-3) and R.hub (H's
  scores). S, W2_S and F0T are text and structure rules and do not change.
- **H was never built in full.** HFO (845f52b) adopted H at k = 5 on frozen, graded only at the slot ends. UL2 builds
  H_5 as a full static family on the four passage corpora, on both encoders.
- **Both S versions.** On 2Wiki and HotpotQA, RK (R > K) and SRK (S > R > K) are both first-class layouts, graded and
  carried. The user's ruling above.
- **Dedup and labels.** Every layout's dedup is counted per family, and its edge labels' overlaps are counted.
- **MuSiQue's hubs.** MuSiQue's hub links are a named target (Part B, section 7).
- **The pools and the refit wait.** Both wait on this stage (section 9).

## 2. The layouts (the family rulings; docs/CANONICAL_LINK_AND_POOL_LAYOUT.md, ruling of 10 Nov, the 21 Nov rulings)

| dataset | versions | S | R | K |
|---|---|---|---|---|
| 2Wiki, HotpotQA | RK and SRK, both first-class | hyperlinks (SRK only) | W2_S + F0T + H_5 | kNN-3 |
| MuSiQue, SQuAD | RK | none | W2_S + F0T + H_5 | kNN-3 |
| MetaQA, WebQSP | one layout, SRK | the KB's triples | ABL-U's W2_S (D89) + F0T | kNN-3 |
| MultiHop-RAG | RK (MHR1's track) | none | MHR1's rule, then H_5 by this code (section 9) | kNN-3 |

- R's labels:
  - W2_S is R.mention.
  - F0T is R.name_comention (U2r's CN_all) plus R.title_comention (F0T \ CN_all).
  - H_5 is R.hub.
- K is K.knn. S's pairs carry S.<rel>.
- **Dedup.** S > R > K, or R > K. Each pair is held once, by its highest family. Inside R a pair keeps every R label
  its rules give it, and the overlaps are counted (section 8).
- **R is built without reading S.** H's exclusion set is N(a) = a's neighbours in W2_S ∪ F0T ∪ K, using the arm's own
  K. So one R serves both versions on 2Wiki and HotpotQA, and the version only decides which family holds a pair.
  - This differs from HFO, which also excluded S on those two datasets. A pair that H picks and S holds goes to S under
    SRK; it is counted in "R removed in S".
  - Gate G-H checks the machinery with HFO's own exclusion (section 5).
- **No H on the KBs.** H is a passage-name rule (HFO section 3), so the KBs carry no H.

## 3. Arms and contrasts

| arm | 2Wiki, HotpotQA, MuSiQue, SQuAD | MetaQA, WebQSP |
|---|---|---|
| D83 | W2_S + F0T + K_frozen (RK; there is no D83 SRK on the passages) | S + F0T + K_frozen |
| REC | (S +) W2_S + F0T + K_frozen: UL1's carried layout (c662a6b), gate G-U | S + W2_S + F0T + K_frozen |
| L2f | REC + H_5 on frozen: the layout under the frozen encoder | = REC |
| L2e | (S +) W2_S + F0T + H_5 on e3v + K_e3v: the candidate | S + W2_S + F0T + K_e3v |

The contrasts:
- **L2e − REC decides.**
- The others are read:
  - L2f − REC (H alone, on frozen);
  - L2e − L2f (the encoder, through K and H together);
  - L2e − D83;
  - REC − D83 (ABL-U's rule graph on the KBs; zero on the passages, where REC equals D83 in RK).

## 4. H_5 as a full family

- **The candidates and the score.** C(a) is zmhb.Names.cand: the passages that share with a a name key of df 2 to
  100,000, from F0T's entries. The eligible set is E(a) = C(a) \ N(a) \ {a}. The score is the inner product of the
  stored float16 document rows, read as float32.
- **The pick.** b is picked by a when fewer than k passages of E(a) score more than 1e-6 above b. This is HFO's rule,
  and it is the same as s(b) ≥ s_(k) − 1e-6, where s_(k) is the k-th best score in E(a). A pair is linked when either
  end picks the other.
- **The exact build: one pass over the keys.**
  - **Each key κ with 26 ≤ df ≤ 100,000:** the scores of κ's passages against each other, with each passage itself and
    its N removed. Each passage's best 24 are offered to a running table, which keeps each passage's best 24 over all
    its keys (score descending, then id ascending). The table also keeps tmax, the highest score the passage ever
    dropped.
  - **Keys of df 2 to 25 are skipped.** Every pair they hold is in F0T, which is part of N. Gate G-C checks every such
    pair.
- **When the table cannot decide.** A passage's picks come from the table unless tmax ≥ s_(k) − 1e-6 at any k of 3, 5
  or 10. Then its whole eligible list is recomputed exactly and stored. Fewer than k entries left after a variant's
  filter counts the same way.
- **HK.** For each hub key (df > 25), each passage's two best eligible passages within that key are also kept (ties go
  to the lower id; Part B's HK).
- **Where it runs.** On the host CPU, with no GPU. The cost (Σ df² over the keys read) and the seconds are recorded.
- **Builds.** Frozen and e3v, on MuSiQue, SQuAD, HotpotQA and 2Wiki. The e3v builds use e3v's K in N:
  - SQuAD: E2's K;
  - MuSiQue: E1b part 2's K;
  - the others: E4's K.

  Each K is checked against its record (enc, dataset, nodes, nodes sha256, npz sha256 and edge count).

## 5. Gates (a failure stops that dataset; none of its numbers is read)

- **G1:** zul1.families' own gates hold, and the name keys equal UL1's build count.
- **G-C:** every pair of every skipped key (df 2 to 25) is in F0T.
- **G-U:** REC links exactly UL1's filed F0T counts, for every carve, version and kind.
- **G-H (frozen; MuSiQue, HotpotQA, 2Wiki):**
  - **What is rebuilt.** Frozen H_k at k = 3, 5 and 10 is rebuilt with HFO's own exclusion: RK on MuSiQue; on 2Wiki
    and HotpotQA, SRK, which is the table with S's pairs filtered and exact rows at every slot end.
  - **What it must match.** Read at HFO's unlinked slots, it must reproduce HFO's filed H_new for every carve, kind and
    k. HFO's union and its linked counts must match first.
  - **The allowance.** Per cell, the difference is at most the cell's near ties. A near tie is an end whose score to
    the other end is within 1e-5 of its own k-th score, where the other end is eligible.
  - **On a FAIL.** The grade refuses every e3v number on that dataset.
- **Self-test.** The self-test (zul2.py selftest; it reads no data) passed before this page was committed. It covers:
  - the pass against brute force on toy data, with exact ties, many merges, skipped keys, threads, the SRK filter on a
    subset of rows, and HK's top two;
  - the dedup;
  - Part B's chain;
  - both calls.

## 6. Part A: the call (ADOPT_L2), fixed now

The measures:
- **Slots.** UL1's slots and carves, unchanged:
  - GP: every pair of a question's golds;
  - CH: MuSiQue's chain hops by the ORC rule;
  - carves s1sel and s1eval (WebQSP: s1eval only).

  A slot is linked when its pair is in the arm's union.
- **Differences.** Each difference is paired over questions (the ratio estimator, 2,000 seeded resamples:
  zap1.boot_diff), with its 95% interval and its call (ABOVE, LEVEL or BELOW: zap1.call).

**ADOPT_L2** needs both conditions:
- On MuSiQue, HotpotQA and 2Wiki, for both carves and every version, L2e − REC is not BELOW on GP, nor on MuSiQue's CH.
- |L2e| ≤ 1.30 × |REC| on all six datasets and every version.

Read beside, never deciding:
- the KBs' GP (questions with 2 to 10 golds);
- MuSiQue's chain connectivity by hops (2, 3 and 4), per arm, with L2e − REC over the connected questions;
- the price, as union pairs per slot gained over REC;
- on SQuAD, which has no slots: the sizes, the dedup, the labels and H_5's pairs (the size bound applies).

**NOT_ADOPTED** names the failures. The next form is fixed now and graded the same way before any pool:
- A BELOW: e3v's K with four neighbours per passage.
- The size bound: H at k = 3.

## 7. Part B: better hub links on MuSiQue (the user's target), fixed now

**Where the gap is.**
- MHB (f060cc1): of the unlinked s1eval chain slots, 63% are B1 and 22% are B4.
  - B1: the answer's key is in both passages' entries, with df > 25.
  - B4: the answer is written in the later passage, but its entries lack the key.
- AP3's scouting: 475 unlinked chain slots, 394 of them with the answer string in both passages behind df > 25 names.

**The forms.** All are static and label-free, use e3v's scores, and exclude N(a) as in section 4.
- **H10 (R.hub):** H at k = 10.
- **HM (R.hub):** mutual H10. a and b each pick the other at k = 10.
- **HK1, HK2 (R.hub_key):** for each hub key κ (25 < df ≤ 100,000) and each passage a naming κ, a's best one or two
  eligible passages among κ's other passages (ties go to the lower id). Aimed at B1: the best partners within each hub,
  not only in the merged list.
- **NTc (R.written).** Aimed at B4.
  - **When y writes κ.** κ's normalized tokens appear in y's text as a run of adjacent tokens: at most 8 tokens,
    case-insensitive, with a leading "the" dropped and the last token's possessive stripped. y's own entries must also
    lack κ.
  - **The pairs.** For each κ with |P_κ| + |W_κ| ≤ 25 (P are the passages naming κ, W those writing it), every pair
    (x ∈ P_κ, y ∈ W_κ).
- **NT1 (R.written):** for each κ with 25 < |P_κ| + |W_κ| ≤ 100,000, each namer x's best writer y (by e3v score, with
  N(x) and x excluded).

**Not repeated (closed forms):**
- D4, U2i, NMD, LT2 and L2D;
- the static title-anchored hub links at S1's price;
- CN_w;
- the F1-F3 sub-span cliques;
- the K1 ladder.

**The chain (MuSiQue, s1eval, CH, RK).**
- The order: HM, HK1, NTc, NT1, HK2, H10. The most targeted and cheapest come first.
- Each form is judged over L2e plus the forms carried before it, and is carried when all three hold:
  - its gain is ABOVE;
  - it adds at least 20 new chain slots (0.5 points of 3,987);
  - its price (new union pairs per new chain slot) is at most P*. P* is H_5e's own price: the union pairs H_5e adds
    per chain slot it adds, over L2e without H.

**On the other datasets.**
- The carried chain is built and graded on SQuAD, HotpotQA and 2Wiki (both versions). The KBs take no passage-name form,
  and their layout is unchanged.
- The verdict comes from the longest prefix of the carried chain whose union stays within 1.30 × |REC| on every passage
  dataset and version, MuSiQue included:

| verdict | meaning |
|---|---|
| CARRIED | the whole carried chain stays within the bound |
| CARRIED_PART | only a shorter prefix does; that prefix is carried |
| NOT_CARRIED_SIZE | even the first carried form breaks the bound |
| NONE_CARRIED | no form passes on MuSiQue |
| PENDING | a carried written-name form is not yet built on a passage dataset |

- PENDING leads to round 2: the same code builds the missing form there, and the grades are rerun.
- Read beside: the GP gains and prices there, MuSiQue's s1sel, and each form alone over L2e.

## 8. Dedup and labels (read for every arm and version)

- **Dedup.** S > R > K: the pairs held per family, R's parts removed in S, and K removed in S and in R.
- **Labels:**
  - the labels: S, R.mention, R.name_comention, R.title_comention, R.hub and K.knn;
  - their pair counts and pairwise overlaps;
  - for each slot kind, the slots each label holds, and the slots it holds alone.

## 9. What follows (fixed now)

1. **The pools.** The pools are built on the adopted layout (both versions on 2Wiki and HotpotQA, plus Part B's carried
   forms) and graded on all six with COV's measures, before any fit. That stage has its own declaration.
2. **One refit.** Then exactly one refit: D83's code, D89, H and Part B's carried forms, on e3v. It has its own file and
   runs after D101. One MLP and one GNN.
3. **MultiHop-RAG.** It joins when MHR1 files its package, with the same rule: RK, with H_5 built by this code.
4. **If NOT_ADOPTED.** The named next form (section 6) comes first, and no pool is built until a layout is adopted.

## 10. Code, rows and outputs

- **Code.** outputs/mp_unified/zul2.py (new). It imports zmhb, zul1, zap1, zenc, u1b and zu2r, and edits none of them.
  Subcommands: selftest | hbuild --ds --enc frozen|e3v | nbuild --ds | grade --ds | report.
- **Host rows** (prefix ul2-):
  - the frozen builds first, since they carry G-H;
  - the e3v builds (HotpotQA and 2Wiki after E4's encodes and K);
  - nbuild on MuSiQue;
  - the grades: MuSiQue first, since the others read its chain; the KBs after E4's K;
  - the report.
- **Outputs.** outputs/ul2/<d>/h_<enc>.{npz,json}, nt_e3v.{npz,json}, grade.json, and outputs/ul2/report.{json,md}.
  The arrays (npz) stay uncommitted.

## 11. Results

### 11.1 Part filed 21 November 2026 (MuSiQue, SQuAD, MetaQA); the calls wait on HotpotQA, 2Wiki and WebQSP

Code: zul2.py at f809831. Host rows, prefix ul2-. Records: outputs/ul2/<d>/h_<enc>.json, nt_e3v.json and grade.json.
Graph level, no model. Differences are in points, paired over questions, with 95% intervals (2,000 resamples, seed 0).
Neither call (sections 6 and 7) is read yet: Part A needs HotpotQA and 2Wiki, and Part B's size bound needs both.

**Gates.**
- **G-H PASS on all three slot datasets.** The frozen H reproduces HFO's filed H_new in every cell:
  - MuSiQue: all 12 cells (k = 3, 5 and 10; GP and CH; s1sel and s1eval), e.g. s1eval CH at k = 5: 83 = 83;
  - HotpotQA: all 6 GP cells, e.g. s1eval at k = 5: 11 = 11;
  - 2Wiki: all 6 GP cells, e.g. s1eval at k = 5: 51 = 51.
- **G-C PASS.** No pair of a skipped key (df ≤ 25) falls outside F0T:
  - MuSiQue: 83,957 keys, 1,056,651 pairs checked;
  - SQuAD: 15,750 keys, 189,832 pairs;
  - HotpotQA: 1,721,802 keys, 25,606,892 pairs;
  - 2Wiki: 2,678,351 keys, 42,423,204 pairs.
- **Near ties, recomputed exactly.**
  - MuSiQue and SQuAD flagged no rows, under either encoder.
  - HotpotQA's frozen build flagged 2,694 rows (RK) and 17 of the 3,091 SRK slot ends.
  - 2Wiki's flagged 2,532 (RK) and 26 of 15,301.
- **H_5 on frozen.** HotpotQA 19,369,321 pairs; 2Wiki 22,176,462. The builds took 72 and 110 min on 8 host CPUs, with
  measured peaks of 24.8 and 31.9 GB.
- **G-U PASS** on MuSiQue and MetaQA: REC links exactly UL1's filed count in every cell.
- **Fix to a row (no number changed).** The first SQuAD grade row failed because SQuAD's s1eval questions file was
  missing on the host (the laptop's copy matches its record). It ran again as ul2-grade-squad-b, with the file pushed.

**MuSiQue (RK).** Shares are linked slots over slots, in %.

| arm | pairs | s1eval GP (5,962) | s1eval CH (3,987) | s1sel GP (3,783) | s1sel CH (2,530) |
|---|---:|---:|---:|---:|---:|
| D83 = REC | 3,387,597 | 33.63 | 49.71 | 29.61 | 42.81 |
| L2f (frozen H and K) | 3,821,163 | 35.16 | 51.79 | 30.48 | 43.79 |
| L2e (e3v H and K) | 3,818,331 | 35.32 | 51.92 | 30.80 | 44.31 |

- **L2e − REC:** ABOVE in all four cells.
  - s1eval: GP +1.69 [+1.34, +2.05], CH +2.21 [+1.72, +2.71].
  - s1sel: GP +1.19 [+0.85, +1.57], CH +1.50 [+1.03, +2.03].
- **L2e − L2f** (e3v against frozen, for H and K):
  - s1eval: LEVEL (GP +0.17 [−0.15, +0.46], CH +0.13 [−0.29, +0.51]);
  - s1sel: ABOVE (GP +0.32 [+0.03, +0.64], CH +0.51 [+0.04, +0.99]).
- **Size and price.** |L2e| / |REC| = 1.1272. On s1eval CH, L2e costs 4,894.7 union pairs per slot gained over REC.
- **Chain connectivity (s1eval).** REC 42.28% → L2e 44.35%, +2.07 [+1.49, +2.69] ABOVE (s1sel +1.24 [+0.72, +1.83]).
  - 2 hops: 70.37 → 72.20;
  - 3 hops: 14.08 → 17.63;
  - 4 hops: 8.40 → 8.40.
- **Dedup (L2e).** R holds 3,639,401 pairs: W2_S 2,343,920, F0T 973,013 and R.hub 429,747. R.hub meets no other
  family, by construction. K holds 178,930: 84,192 of e3v's 263,122 kNN pairs are already in R.
- **Labels on s1eval CH** (slots linked, and in brackets those linked by that label alone):
  R.name_comention 1,315 (702), R.mention 1,082 (483), K.knn 416 (33), R.title_comention 112 (34), R.hub 86 (86).

**Part B's chain (MuSiQue, s1eval CH, RK).** P\* = 429,747 / 86 = 4,997.1 union pairs per chain slot.

| form | new slots | pairs added | price | gain over the running base | carried |
|---|---:|---:|---:|---|---|
| HM | 6 | 46,404 | 7,734.0 | +0.15 [+0.05, +0.28] ABOVE | no (under 20 slots) |
| HK1 | 30 | 101,748 | 3,391.6 | +0.75 [+0.50, +1.04] ABOVE | yes |
| NTc | 138 | 482,638 | 3,497.4 | +3.46 [+2.93, +4.03] ABOVE | yes |
| NT1 | 19 | 363,832 | 19,149.1 | +0.48 [+0.27, +0.71] ABOVE | no |
| HK2 | 14 | 144,072 | 10,290.9 | +0.35 [+0.18, +0.55] ABOVE | no |
| H10 | 49 | 358,675 | 7,319.9 | +1.23 [+0.91, +1.58] ABOVE | no (price) |

- **The carried forms {HK1, NTc} over L2e:**
  - s1eval CH: +4.21 [+3.64, +4.85] ABOVE. That is 168 slots, so 51.92% → 56.13%; REC was 49.71%.
  - s1eval GP: +3.00 [+2.59, +3.46] (179 slots).
  - s1sel, read beside: CH +0.99 [+0.64, +1.41], GP +0.90 [+0.63, +1.22].
- **Size.** The forms add 584,386 pairs on MuSiQue.
  - Union over REC: MuSiQue 1.2997 (the bound is 1.30); SQuAD 1.143.
  - HotpotQA and 2Wiki are not built yet, so Part B's verdict is open.
- **Part B's gap.** NTc, the written-name form aimed at B4, does most of the gain. HK1 (each hub's best partner,
  aimed at B1) adds 30 slots.

**SQuAD (RK; no slots, sizes only).**
- REC has 997,652 pairs and L2e 1,056,550 (1.059 × REC); with HK1 and NTc, 1.143.
- 36,852 of e3v's 43,350 kNN pairs (85%) are already in R.
- NTc has 100,108 pairs and NT1 55,061.

**MetaQA (SRK, one layout; its GP is read beside and decides nothing).**
- REC has 351,892 pairs and L2e 349,818 (0.9941).
- s1eval GP: 13.81 → 13.12, L2e − REC −0.70 [−0.86, −0.55] BELOW. s1sel: −1.22 [−1.74, −0.75] BELOW.
- On the KBs, L2e differs from REC only in K. e3v's kNN links fewer KB gold pairs than the frozen kNN does, and the
  refit's KB layout should know this.

**Still to come (queued).**
- **HotpotQA and 2Wiki.** First the e3v builds, then NT on both, then their grades. The rows are now
  ul2-h-<d>-e3v-s, ul2-nt-<d>-s and ul2-grade-<d>-s. They wait for E4's K as rebuilt in ENC section 20.10 (34b5bf5),
  which runs as e4s-knn-<d>. The first e3v K on HotpotQA failed its gate, and 2Wiki's was cancelled.
- **WebQSP's grade**, after e4g-knn-webqsp (running again since the host restarted).
- **ul2-report-s**, which reads all six and writes both calls.

## 12. Universal rules (declared 21 November 2026, 15:05 IST, 09:35 UTC)

The user ruled on 21 November, verbatim: "we need to have universal rules".

This section is declared before any e3v number on HotpotQA, 2Wiki or WebQSP, and before any number of the KBs' new
families (F0T on the rule text, R.hub and Part B's forms on a KB). It does not change sections 3-11 or their filed
numbers. Sections 6 and 7 stay as the record of what was declared first. Section 12's call is the one the refit's
layout takes.

**What had been seen when it was written.**
- MuSiQue's whole grade (section 11.1), including Part B's chain on MuSiQue. So this section comes after MuSiQue's
  Part B numbers were seen.
- SQuAD's sizes and MetaQA's REC and L2e arms (section 11.1).
- On HotpotQA and 2Wiki: the frozen H_5 sizes and the gate G-H cells. Those cells reproduce HFO's filed numbers.
- Two cost readings taken from the KB degree counts while this section was written; no rule text was built.
  - Pairs of entities that share one KB neighbour of degree at most 25, counted with repeats: at most 7.6e5 on MetaQA
    and 1.63e7 on WebQSP. MetaQA's REC has 351,892 pairs.
  - WebQSP's key pass would score about 3.1e10 pairs (this sized the rows).
- Nothing else of HotpotQA's, 2Wiki's or WebQSP's, and nothing of e3v there.

### 12.1 The rule: one rule set, seven datasets

1. **Same code, same constants on every dataset.** These are the R rules:
   - W2_S and F0T;
   - H_5 (section 4: k = 5, table depth 24, keys of df 26 to 100,000, EPS 1e-6);
   - Part B's forms (section 7): HM, HK1, HK2, NTc, NT1 and H10 (window 8, cap 25, df at most 100,000).

   This removes "R.hub is a passage-name family" (section 4) and "The KBs take no passage-name form" (section 7).
   MultiHop-RAG takes the same set when MHR1 files its package.
2. **Each rule reads that dataset's rule text.**
   - On a passage dataset, the rule text is the passage's own text, as filed.
   - On a KB, it is ABL-U's rule text (docs/ABL_LINK_FAMILY_ABLATION.md, section 8; zabl_u.texts_of):
     - the names of the entity's distinct KB neighbours, both directions, in node-id order, joined by "; ";
     - its own name is left out;
     - an entity with no neighbour keeps its own name;
     - names are as ABL-U reads them (u1b.read_nodes), so a WebQSP CVT mediator's name is its package text.
   - On the KBs every R rule reads that text:
     - W2_S already does (ABL-U, filed).
     - F0T is rebuilt on it, as **F0T_u**. This is UL1's build unedited (zul1.build_entries and forms_from_entries).
     - R.hub's names and the written names read it (zmhb.build_entries_keep, zul2.written_entries).
3. **Unchanged.**
   - S follows the 10 November ruling and the 21 November rulings: 2Wiki and HotpotQA have RK and SRK; the KBs have
     SRK; MuSiQue, SQuAD and MultiHop-RAG have RK.
   - K is e3v's kNN-3 everywhere.
   - The dedup is S > R > K. Labels are by rule (section 8); Part B's labels are as in section 7, and H_5 is R.hub.
   - REC is still the record (section 3), checked by gate G-U. The bound reads REC.

### 12.2 What is judged

- **The base, B0** (per dataset and version) = S (SRK only) ∪ W2_S ∪ F0T ∪ K_e3v.
  - On the KBs, F0T is F0T_u.
  - On the passage datasets, B0 is section 7's "L2e without H".
- **The forms, in the chain's order:** H5, HM, HK1, NTc, NT1, HK2, H10.
  - H5, HM, HK1, HK2 and H10 come from e3v's R.hub table on that dataset's rule text. Its exclusion is
    N = W2_S ∪ F0T ∪ K_e3v, as section 4 (on the KBs, F0T_u). S is never read, as on the passage datasets.
  - NTc and NT1 are built on e3v's rows.
  - The passage tables are zul2's, already declared and partly built, read by zul2's build identity.
- **The judged cells.** s1eval decides; s1sel is read beside. The cells are:
  - MuSiQue: CH and GP (RK);
  - HotpotQA and 2Wiki: GP, both RK and SRK;
  - WebQSP and MetaQA: GP (SRK). KB GP counts questions with 2 to 10 golds only, as UL1 does.
  - SQuAD has no slot. It enters through the size condition only.

### 12.3 The decision: one decision, all datasets

The forms are judged one at a time, in the order above. Each is judged over the running base: B0 and the forms carried
before it.

**A form is carried on every dataset, or on none.** It is carried when all three conditions hold:
- **(a) Gain.** The paired difference over questions is ABOVE in at least one judged cell and BELOW in none. This uses
  zap1.boot_diff with 2,000 resamples, seed 0. At graph level a form only adds links, so BELOW cannot occur; the
  condition is stated as given.
- **(b) Price.** In every judged cell where the form adds new slots, the pairs it adds to that version's union, divided
  by the new slots, are at most P* = 4,997.1.
  - P* is H_5e's price on MuSiQue s1eval CH, from section 7, filed in 61fca78. The report checks it against
    outputs/ul2/musique/grade.json.
  - The coordinator's reading carries no minimum count of new slots, so section 7's 20 is not part of this rule.
- **(c) Size.** The running base with the form is at most 1.30 × |REC| on every dataset and version, SQuAD included.

**H_5 is the chain's first form.** If it fails, H is carried nowhere, and every later form is judged over B0 without it.
If HotpotQA's roughly 19 million R.hub pairs break (b) or (c) there, H is carried nowhere.

**If the base itself is over the bound.** If B0 alone is over 1.30 × |REC| on some dataset or version, no form can meet
(c), and the verdict is BASE_OVER_BOUND.
- The chain still runs, and every form's (a) and (b) are recorded.
- The call goes back to the user.
- The MetaQA cost reading above makes this possible there, through F0T_u.

**The verdicts:** CARRIED (the forms), NONE_CARRIED, BASE_OVER_BOUND, or REFUSED (a grade missing, or not this code's).

**Read beside, never deciding:**
1. MuSiQue's own chain verdict as filed in section 11.1 ({HK1, NTc}), kept as it is.
2. **U-name:** the same chain with the KBs' F0T as filed by UL1 (the name text) in place of F0T_u. One run then shows
   what the rule text does to F0T.
3. Every step's s1sel cells, at the s1eval chain's running base.
4. **A scale-free price per cell:** (pairs added / |REC|) ÷ (new slots / the cell's slots), set beside H_5e's on
   MuSiQue CH.
   - It is shown because P* counts pairs per slot.
   - HotpotQA's and 2Wiki's graphs are many times MuSiQue's: frozen H_5 adds 19.4M and 22.2M pairs there
     (section 11.1).
5. L2f, frozen H_5 with frozen K, on every dataset.

**MultiHop-RAG** joins when MHR1 files its package. The chain then runs again over the seven datasets, and that call
replaces this one.

### 12.4 Gates (a failure stops that dataset)

- **G-T, the KB rule text.** zabl_u's recipe identity (the sha256 ABL-U's build computes) and the nodes' sha256 must
  equal ABL-U's filed build record. ABL-U's W2_S is read as UL1 reads it (zul1.rule_part).
- **G-F, F0T's code on the KBs.** Run on the KB's filed name text (zu2r.corpus), the same build path must reproduce
  UL1's filed F0T exactly: the same pairs, and the same key count as UL1's build record. Only then is F0T_u built on
  the rule text.
- **G-C on the rule text.** The keys the key pass skips (df 2 to 25) must hold only F0T_u's pairs (zul2.gate_gc).
- **G-U and G-H** are unchanged: REC must link exactly UL1's filed counts in every cell, and G-H must pass on the frozen
  passage tables.
- **Builds are read by their build identity** (functions and constants), as in zul2.

### 12.5 Code, rows and outputs

**Code.** outputs/mp_unified/zul2u.py is new. It imports zul2, zul1, zmhb, zabl_u, zu2r and zap1, and edits none of
them. Its commands:
- `kbtext --ds metaqa|webqsp`: G-T, G-F, then F0T_u. Writes outputs/ul2u/<d>/f0t_u.npz and kbtext.json.
- `hbuild --ds metaqa|webqsp --enc frozen|e3v`: R.hub's table on the rule text, using zul2's key pass, exact rows and
  picks. Writes outputs/ul2u/<d>/h_<enc>.npz and .json.
- `nbuild --ds metaqa|webqsp`: the written-name forms on the rule text. Writes outputs/ul2u/<d>/nt_e3v.npz and .json.
- `grade --ds D` (all six): REC, B0, U-name's base and L2f; each form's pairs over each base (a histogram of form
  masks, so that any carried set's union size is exact); each judged cell's slot masks. Writes
  outputs/ul2u/<d>/grade.json and masks.npz.
- `report`: the chain of section 12.3. Writes outputs/ul2u/report.json and report.md.

**Rows.** The host rows are named ul2u-*. The KB builds and the MuSiQue, SQuAD and MetaQA grades run now on the host
CPU. WebQSP's e3v builds wait for e4g-knn-webqsp. HotpotQA's and 2Wiki's grades wait for zul2's -s rows.

**Outputs.** The json records are committed. The arrays (npz) are not.

### 12.6 Carrying by budget (declared 21 November 2026, 17:43 IST, 12:13 UTC)

The user ruled on section 12's BASE_OVER_BOUND, 21 November, verbatim: "what? somehow make it carry".

This subsection replaces section 12.3's base on the KBs and its size condition (c). Everything else in sections 12.1 to
12.5 stands. It is numbered 12.6 because 12.4 already holds the gates. It is declared before any judged form number on
HotpotQA, 2Wiki or WebQSP.

**What had been seen when it was written.**
- The user's "make it carry".
- MuSiQue's Part B numbers (section 11.1).
- Section 12's records filed in 3a0f788 (MuSiQue, SQuAD, MetaQA). Only their gates, sizes and form pair counts were
  read: MetaQA's F0T_u holds 706,154 pairs against the filed F0T's 12,605, so B0 = 2.599 × |REC|. U-name's base is
  0.994 × |REC| on MetaQA, 1.0003 on MuSiQue and 0.998 on SQuAD. The per-form gains in those records were not read.
- Nothing of e3v on HotpotQA, 2Wiki or WebQSP.

#### 12.6.1 The base: F0T reads the node's own text

- **F0T reads a node's own text.** That is a passage's body, and a KB entity's name.
- So on the KBs the deciding base takes UL1's filed F0T (section 12's U-name reading):
  B0 = S ∪ W2_S ∪ F0T ∪ K_e3v on every dataset, with F0T as UL1 filed it. On the passage datasets nothing changes.
- **Why F0T_u moves beside.** ABL-U's rule text appends the names of the entity's KB neighbours. Those are its facts,
  and S already links them. F0T on that text pairs entities that share a neighbour's name; that is what made MetaQA's
  F0T_u 56 times the filed F0T.
- **The KBs' other R rules stay as section 12 built them.**
  - W2_S is ABL-U's, as filed.
  - R.hub's table and the written names read the rule text.
  - Their exclusion stays N = W2_S ∪ F0T_u ∪ K_e3v. The key pass skips the keys of df 2 to 25, whose pairs F0T on the
    same text holds (gate G-C), so its exclusion must hold F0T on that text.
  - So F0T_u's pairs enter the layout neither through the base nor through H: H never picks them.
- **Beside:** section 12's U (F0T_u) readings, from zul2u's grades and report.

#### 12.6.2 The budget: a form keeps its highest-ranked pairs

**The room.** On each dataset and version, the room is ⌊1.30 × |REC|⌋ minus the running union (B0 and the kept parts of
the forms carried so far). It is read in integers: a union U fits when 100 × U ≤ 130 × |REC|.

**The cut.** A form is cut, on each dataset, to the longest prefix of its ranked list whose new pairs fit the room in
every version of that dataset.
- One cut per dataset, so R is the same family in both versions (2Wiki, HotpotQA).
- A pair already in the running union takes no room. In SRK, a pair in S takes no room there.
- On a dataset where the room is enough, the whole form is kept.
- On a dataset whose base alone is over the bound, the forms keep nothing, and the record names it.
- So condition (c) holds by construction. The report checks it.

**Each form's ranked list.** The order is the build's own score. No gold is read, and no constant or weighting is added.
- **H5, H10** (R.hub's table on e3v): the table's score. That is the exact float32 dot product of the two nodes' stored
  e3v rows (tab_s, and the exact list's ov_s on flagged rows). A pair picked from both ends keeps the higher of its
  stored scores.
- **HM** (H10's pairs picked from both ends): the same score.
- **HK1, HK2** (a hub key's best one and two): the hub key's score (hk_s1, hk_s2). A pair from several hub keys keeps its
  highest.
- **NT1** (a namer's best writer): the build's own score, the namer-writer dot product of the stored e3v rows. NT1 stores
  only its pairs, so the ntrank row recomputes the score in float32 from the same rows.
- **NTc has no score in its build.** It takes every namer-writer pair of a key whose namers plus writers are at most 25.
  - Its order is that count, the group size its cap reads, smallest first.
  - A pair from several keys keeps its smallest group.
  - The ntrank row recomputes the groups from the same names and written entries.
- **Ties** in every list go to the lower pair key (lo × n + hi) first, as the builds break ties to the lower id.

#### 12.6.3 The decision

- It is section 12.3's chain, with conditions (a) and (b) read on what each form keeps.
- The order is H5, HM, HK1, NTc, NT1, HK2, H10. Each form is judged over the running base.
- (a) and (b) are unchanged:
  - (a) ABOVE in at least one judged s1eval cell, and BELOW in none;
  - (b) a price of at most P* = 4,997.1 in every judged cell where the form adds slots.
- A form that is not carried takes no room.
- If a form fails (b), the record names each cell and its price over P*. (b) is not loosened.
- **Verdicts:**
  - CARRIED, with each carried form's cut on each dataset: the rank kept, the pairs kept of its new pairs, and the score
    at the cut;
  - NONE_CARRIED;
  - REFUSED, when a grade is missing or is not this code's.
- **Read beside, never deciding:**
  1. Each step's uncut reading: the whole form over the same running base, as section 12.3 read it.
  2. s1sel at each step.
  3. The scale-free price.
  4. Section 12's U and U-name verdicts (zul2u's report, once filed).
  5. MuSiQue's chain as filed in section 11.1.
- **MultiHop-RAG** joins when MHR1 files its package. The chain then runs again over seven datasets, and that call
  replaces this one.

#### 12.6.4 Gates (a failure stops that dataset)

- **G-R:** each ranked list holds exactly the pairs zul2 derives for that form (h_pairs, mutual_pairs, hk_pairs).
- **G-N:** the recomputed NTc pairs are exactly the filed NTc, and the scored NT1 pairs exactly the filed NT1.
- **G-U and G-H** are unchanged. The frozen passage tables are read from their records.
- **Sizes against the filed grades,** where they exist: REC and U-name's base must equal zul2u's REC and N, and zul2's
  REC (and, on a KB, zul2's L2e).
- **Builds are read by their build identity,** so a change to the report alone does not stale a grade.

#### 12.6.5 Code, rows and outputs

**Code.** outputs/mp_unified/zul2c.py is new. It imports zul2u, zul2, zul1, zmhb, zap1 and zenc, and edits none of them.
Its commands:
- `ntrank --ds D`: NTc's groups and NT1's scores; gate G-N. Writes outputs/ul2c/<d>/ntrank.npz and .json.
- `grade --ds D`: each form's ranked pairs beyond B0, their ranks and scores; S membership; each judged cell's slot
  ranks and B0 cover; the sizes. No union is built: sizes are counted pairwise. Writes outputs/ul2c/<d>/ranks.npz and
  grade.json.
- `report`: the chain of 12.6.3. Writes outputs/ul2c/report.json and report.md, and outputs/ul2c/<d>/carried.npz (the
  kept pairs beyond B0, each with the carried forms that keep it, for the labels).

Each record holds its own process's peak working set and peak commit. The next rows' memory asks are read from those.

**Rows.** The host rows are named ul2c-*. Each waits for its tables: zul2's on the passage datasets (HotpotQA and 2Wiki:
the -s rows), zul2u's on the KBs.

**Outputs.** The json records are committed. The arrays (npz) are not.
