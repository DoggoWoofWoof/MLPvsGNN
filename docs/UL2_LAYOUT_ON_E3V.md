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
