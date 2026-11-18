# HFO: the hub family H against the gold links, before any training

Declared 18 November 2026 at 22:50 IST, before any HFO number exists. Graph level only: nothing trains, no model reads
anything, and no filed graph, pool, fit or record changes.

## 1. Question

The user (18 Nov): "then we have to judge wrt to the oracle or gold labels, how much of the links there are here and can
we improve that number, then we have to check about transferability to models".

MHB (f060cc1, section 8) found the following on MuSiQue's s1eval chain slots:
- The RK union links 1,982 of 3,987 (49.7%).
- Of the 2,005 unlinked slots, 76% share a name key in both passages, and 63% share a name that is not a rule link.
- When the question ranks the anchor's hub-name passages, the partner is in the top 10 for 57.7% of them.

A family that a model reads must be static: it is built without the question. HFO asks two things:
1. How many of the gold links does the record's graph hold now, and through which family?
2. Does a static hub family H raise that number by more than the same budget of plain semantic neighbours?

Transfer to the models is the step after HFO (section 7). It runs only on HFO's outcome.

## 2. The forms

All forms are static and label-free, and none reads a question.
- **C(a):** MHB's candidates (zmhb.Names.cand). These are the passages that share with a any name key of df 2 to
  100,000, without a itself. The keys come from F0T's entries: UL1's names plus each passage's own title runs.
- **N_U(a):** a's neighbours in the record's union. On MuSiQue the union is RK: W2_S ∪ F0T ∪ K. On 2Wiki and
  HotpotQA it is SRK, the same plus S.
- **score(a, b):** the inner product of the stored frozen document rows (float16 read as float32), as in MHB. A
  passage counts as above b only when it scores more than 1e-6 above b; ties are not above.
- **H_k (hub family, label R.hub):** each passage a links to the k highest-scoring passages of C(a) \ N_U(a). These
  are its k best new partners among the passages that share a name with it.
- **D_k (the control):** each passage a links to the k highest-scoring passages of all passages \ ({a} ∪ N_U(a)).
  These are its k best new plain neighbours.
- **Budget:** both forms add k new partners per passage. H adds fewer when C(a) \ N_U(a) has fewer than k
  passages.
- **k:** 3, 5 or 10. K itself is the record's kNN with 3 neighbours.
- A pair is linked by a form when either end picks the other.

## 3. The gold links (the oracle)

UL1's slots are used unchanged (docs/UL1_UNIFIED_LINKS_ORACLE_CHECK.md, zul1):
- **GP:** every unordered pair of a question's distinct golds, on MuSiQue, 2Wiki and HotpotQA.
- **CH:** MuSiQue's chain hops by the ORC rule (zul1.chain_slots).

Carves s1sel and s1eval are both read; s1eval decides the call.

Out of this check:
- MetaQA and WebQSP: S is the KB itself, and H is a passage-name rule.
- SQuAD: one gold per question, so there are no slots.

## 4. What is measured

For each dataset, carve and kind:
- **The union's linked share.** It must equal UL1's filed count (G3). Each family's linked share alone is also
  reported (S, W2_S, F0T, K).
- **For each k:**
  - the share linked by U ∪ H_k and by U ∪ D_k;
  - each one's gain over U;
  - the diff (U ∪ H_k) − (U ∪ D_k), by the ratio estimator over questions with 2,000 seeded bootstrap resamples
    of the questions (zap1.boot_diff, seed 0), with its 95% interval;
  - the call on the diff, ABOVE, LEVEL or BELOW (zap1.call).
- **MuSiQue CH by hop count:** the gains are also shown separately for 2-, 3- and 4-hop questions.
- **Budget.** It is estimated on 20,000 passages drawn with seed 0 (all of them when there are fewer). For H_k it
  is the mean of min(k, |C(a) \ N_U(a)|); for D_k, the mean of min(k, n − 1 − |N_U(a)|). Each mean times n gives
  the new directed edges, an upper bound on the new undirected pairs.

How it is computed:
- Only at the slot ends. For each unlinked slot (x, y) and each end, HFO finds the other end's rank among the
  anchor's candidates.
- There is one exact pass over the document rows per block of anchors: zmhb.ranks, with the anchors' rows as the
  queries.
- The anchor itself, and its union neighbours that score above the other end, are then removed from both counts.

## 5. Gates

A failed gate stops the dataset, and no number is read from it.
- **G1:** zul1.families' own gates hold, and the name keys equal UL1's build count.
- **G2:** every rank is at least 1, and the removed passages never exceed the raw count above.
- **G3:** the union links exactly UL1's filed count, for each carve and kind.
- **G4:** 64 cases drawn with seed 0 are recomputed by a direct scan with explicit exclusion masks. Every
  linked-at-k decision for H and D must agree, and the ranks must agree within 2 + 0.1%. GEMM blocks of other
  shapes round differently near a tie.

## 6. The call (s1eval)

For each k, three conditions:
- **P1:** the diff is ABOVE on MuSiQue CH (2,005 unlinked slots of 3,987; at least 200 are needed).
- **P2:** H_k's gain over U on MuSiQue CH is at least 0.02 (two points of the chain slots).
- **P3:** the diff is not BELOW on MuSiQue GP, 2Wiki GP or HotpotQA GP.

Outcome:
- **ADOPT_H at the smallest k that meets P1-P3.** The transfer step (section 7) follows.
- **SPLIT if P1 and P2 hold at some k but P3 fails at every such k.** The hub restriction helps MuSiQue, while the
  plain neighbours do better elsewhere. A mixed form (half the budget from each) is named and is graded the same way
  before any training. Nothing trains.
- **NO_H if P1 or P2 fails at every k.** If D_k's gain on MuSiQue CH is at least 0.02 at some k, a larger K is named
  as the next family and is graded the same way. If not, static link growth closes at these budgets.

MHB's query-time form (the question picks among the pool seeds' hub-name passages) is named only. It belongs to pool
construction, not to a family that a model reads.

## 7. Transfer to the models

This step is named here and gets its own declared section before it runs. It runs only on ADOPT_H, or later on a
graded form that is adopted the same way. Its steps:
1. Build H_k over the passage corpora (MuSiQue, 2Wiki, HotpotQA, SQuAD) as one static family labelled R.hub.
2. Grade its pools on all six datasets.
3. Refit the GNN once in D83's world with R.hub added.
4. Read the six datasets by the record's rules.

Which models read R.hub, and how the MLP stays MP-free, is decided in that declaration.

## 8. Code and rows

- **Script:** outputs/mp_unified/zhfo.py (new). Subcommands: `selftest`; `run --dataset D [--host] [--threads T]
  [--limit Q] [--out DIR]`; `report`.
- **Imports, none edited:** zmhb, zul1, zap1, zenc, u1b and zu2r.
- **Outputs:** outputs/hfo/<d>/result.json and outputs/hfo/report.{json,md}.
- **Host CPU rows, no GPU:** one run per dataset, then the report.

## 9. Results (filed 19 Nov 2026, 00:00 IST)

Files: outputs/hfo/report.json and report.md, and outputs/hfo/<dataset>/result.json for MuSiQue, 2Wiki and HotpotQA.
None is a dev result (no --limit).

**Gates.**
- G1 to G3 held on all three datasets. The code stops at a failed gate, and every result was written.
- G4: 64 cases per carve on all six carves (s1sel and s1eval of each dataset), 0 disagreements. The largest rank
  difference was 2 (2Wiki and HotpotQA s1sel), within 2 + 0.1%.

**The call: ADOPT_H at k = 5.**
- P1: on MuSiQue CH (s1eval), H_5 minus D_5 is +0.60 points [+0.25, +0.96]: ABOVE.
- P2: H_5's gain over U on MuSiQue CH is 0.0208 [0.0163, 0.0251]. The rule reads the point estimate (at least 0.02);
  the interval's lower end is below 0.02.
- P3: MuSiQue GP, 2Wiki GP and HotpotQA GP are LEVEL.
- k = 3 fails P1 and P2. k = 10 also meets all three (diff +2.21 [+1.64, +2.76], gain 0.0424, MuSiQue GP ABOVE), but
  the rule takes the smallest k.

**Size.**
- MuSiQue s1eval CH (3,987 slots): U links 49.7%, U + H_5 51.8%, U + D_5 51.2%. H_5 links 83 of the 2,005 unlinked
  slots, D_5 links 59.
- MuSiQue s1eval GP (5,962 slots): U 33.6%, U + H_5 35.2%, U + D_5 34.9% (LEVEL at k = 5, ABOVE at k = 10).
- 2Wiki and HotpotQA: neither H nor D changes the union's share by more than 0.1 points at any k. The union already
  links 45.8% (2Wiki s1eval) and 80.0% (HotpotQA s1eval) of the gold slots.

**What this says.** At the same number of new partners per passage, restricting the neighbours to hub-name passages
finds more of MuSiQue's gold chain partners than plain nearest neighbours do. At k = 5 the difference is small (0.6
points of the chain slots). It does not move 2Wiki or HotpotQA.

**Next.** The transfer step (section 7) gets its own declared section before it runs. Under the user's 16 Nov rule
(check links against the oracle on all datasets, unify, then train once):
- Its steps 1 and 2 need no training: build H_5 as the static family R.hub on the four passage corpora, then grade
  the pools on all six datasets.
- Its step 3, the GNN refit, is not a refit of its own. It joins the one unified refit after the encoder decision
  (ENC, D101).
