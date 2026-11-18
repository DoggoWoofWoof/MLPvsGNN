# MHB: the multi-hop bridges our links miss (hub names, unwritten bridges, how the chains are built)

**Declared 18 November 2026, 20:45 IST (stamp: `date` at writing), before any MHB number exists.** Script:
`outputs/mp_unified/zmhb.py` (new). Graph level only: nothing trains, no model reads anything, no filed graph, pool, fit
or record changes. Outputs only under `outputs/mhb/`.

## 1. Why

The user, 18 November:
- "make sure we do have better links regardless, if we can recreate multi hops from musique etc, and maybe even see how
  triplets are made"
- "make sure we train only when we know it could work"
- They chose all three link checks: hub bridges, unwritten bridges, and a census of how the datasets build their
  chains and triplets.

**What is on file:**
- **UL1** (16 November, F0T carried into D83): D83's union links 49.71% of MuSiQue's s1eval chain slots and 33.63% of its
  gold pairs (RK). 2Wiki's union links 45.82% of gold pairs and HotpotQA's 80.01% (SRK).
- **AP1** (12 November, on B1's MuSiQue setting):
  - The bridge answer is written in the earlier passage in 99.6% of chain slots, and in the later passage in 85.7%.
  - About 223 of the roughly 502 unlinked slots have the bridge name in both passages, at df > 25. These are hubs, which
    the cap of 25 refuses.
- **NMD** (3 November): NO_SIGNAL (RARE). The capped names reach 2% of the unlinked next-hop golds from the anchor. Its
  open forms are prefix-merged titled names and **names as entity nodes**, which is Part A here.
- **L2D** (3 November): SIGNAL on MuSiQue. 57% of the unlinked next-hop golds are two links from the anchor, but they
  sit among about 1,000 two-link rows a case.

So the question is whether a hub name, which joins too many passages to be a link, becomes selective once the question
picks among its passages.

## 2. What it reads

- **The links:** D83's union, through `zul1.families`, with every gate that function runs:
  - RK on MuSiQue: W2_S ∪ F0T ∪ K.
  - SRK on 2Wiki and HotpotQA: S ∪ W2_S ∪ F0T ∪ K.
- **The slots:** UL1's on the s1sel and s1eval carves (`zul1.carve_questions`, `gp_slots`):
  - GP, every pair of gold passages;
  - on MuSiQue also CH, the chain slots. They are kept directed here (earlier hop, later hop, the earlier
    sub-question's answer).
  - A slot is **unlinked** when its pair is not in the union.
- **The names:** F0T's own entries, rebuilt with `zul1.build_entries`: each passage's rule names plus its title runs,
  over one vocabulary. df is the number of passages holding a key.
- **The vectors:** the package's frozen document and question vectors (`zenc.vec_shards`, encoder `frozen`). These are
  float32 inner products, as in E0.
- **Labels:** the gold pairs, MuSiQue's decompositions, and 2Wiki's evidence triplets and question types. They are the
  grading and the census only. No rule reads one.

## 3. Part A: hub names as entity nodes, with the question choosing (the call)

For a case, meaning an unlinked slot read from one end (anchor *a*, target *b*, question *q*; both ends are read):
- **C(a)** is the passages that share any key of *a* with df from 2 to DFMAX = 100,000, without *a* itself. This is a
  hop through *a*'s names as entity nodes, with no cap of 25. Keys above DFMAX are counted, not walked.
- **rank_hub** is 1 plus the number of passages in C(a) that score above *b* on cos(*q*, ·). It applies only when *b*
  is in C(a).
- **rank_dense** is 1 plus the number of passages in the whole corpus that score above *b*: the question alone.
- "Above" means by more than 1e-6. Blocked matrix products round differently from a single dot product, and this keeps a passage from
  counting itself.
- **The anchor is found** when its own rank_dense is at most 5.

**Read per dataset, carve and slot kind:**
- the share of unlinked slots whose two passages share a key (any df, and by df band);
- HUB@k for k = 5, 10 and 20: the share of unlinked slots with rank_hub ≤ k from either end;
- NEW@k: the same, but only where that end's rank_dense of the target is > k;
- each of these over all cases, and over the cases whose anchor is found;
- |C(a)| (median and p90), the keys above DFMAX, and the share of slots whose only shared keys are above DFMAX.

**The call, on MuSiQue's s1eval chain slots:**
- **SIGNAL** iff NEW@10 ≥ 0.10 of the unlinked slots, with at least 200 unlinked slots.
- Otherwise **NO_SIGNAL**: RARE if fewer than 0.10 of the unlinked slots share any key, else NOT_SELECTIVE.
- 2Wiki's and HotpotQA's GP and MuSiQue's GP are read beside. Any form built later must not lose there.

**What follows:**
- **SIGNAL:**
  - A query-time hub family H is declared in its own file: from each pool seed, the top k of its hub names' passages by
    the question, labelled R.hub.
  - H is graded first by pool gold coverage on all six datasets. That is the same oracle check E1b uses, with no
    training.
  - It is trained only if coverage rises with no dataset BELOW.
- **NO_SIGNAL:** hub names close as links, and the census below names the next form.

## 4. Part B: the unwritten bridges (MuSiQue's chain slots, a census)

Each directed chain slot is placed in the first class that fits. It runs over the unlinked slots, with every slot read
beside. *ans* is the earlier sub-question's answer, and key() is U2r's `norm_key`.

| class | test |
| --- | --- |
| B1 named both, hub | key(ans) is among both passages' entries, with df > 25 |
| B2 named both, capped | the same with df ≤ 25. This is a gate: it must be 0 on unlinked slots, because F0T links such pairs |
| B3 not a name | ans gives no rule name (dates, numbers, lowercase phrases); split by whether the later passage's text holds ans |
| B4 written, not a name there | the later passage's text holds ans (case-insensitive), but its entries do not hold key(ans) |
| B5 the later passage's title | the later passage's stripped title shares a word of 3 or more letters with ans, or one holds the other |
| B6 partial | a key of the later passage shares a word of 3 or more letters with ans (an alias or a part of the name) |
| B7 unwritten | none of these (a pronoun, a paraphrase, or nothing) |

Per class, the read also gives Part A's HUB@10 from the earlier end, and whether ans is written in the earlier passage.

## 5. Part C: how the chains and triplets are built (a census, no call)

- **2Wiki GP** (both carves), by question type (compositional, inference, comparison, bridge_comparison):
  - the share of slots each family links: S, W2_S, F0T, K and the union;
  - the same by the first evidence triplet's relation (its 25 most common);
  - for compositional and inference questions, the triplet's object (the bridge): is it written in the subject's
    passage, and is a gold passage titled as it?
- **HotpotQA GP** by type (bridge or comparison): the share each family links.
- **MuSiQue CH** by hop count, and by the earlier sub-question's relation (the text after ">>", its 25 most common):
  the share each family links.

This says which relations and question kinds our links miss. It also names the labels SEM2 would read.

## 6. Gates (any failure stops the run with no verdict)

| gate | test |
| --- | --- |
| G1 | `zul1.families`' own gates: the package's stores, W2_S and F0T against their build records |
| G2 | the directed chain slots' pair keys equal `zul1.chain_slots`' keys, slot for slot |
| G3 | the union's linked counts per carve, kind and domain equal UL1's filed F0T counts in `outputs/ul1/<d>/grade.json` |
| G4 | for each case question, our dense scores of the top ten ids in the record's dense_top1000 list equal the record's scores (max absolute difference ≤ 1e-3) |
| G5 | B2 = 0 on unlinked slots |

## 7. Cost and rows

A dev run on MuSiQue on the laptop (written to the scratchpad, not filed) passed G1 to G5 in 58 s before this page was committed.
Its numbers were not read.


The run uses host CPU only, one row per dataset, with no GPU.
- **Name entries:** the stream UL1's build took (MuSiQue 48 s, HotpotQA 431 s, 2Wiki 756 s).
- **Dense pass:** one pass over the document vectors per block of 1,024 case questions, for the questions with an
  unlinked slot only.
- **Memory:** about 16 GB on 2Wiki.

`zmhb.py report` gathers the three into `outputs/mhb/report.json` and `report.md`.

## 8. Results

None yet.
