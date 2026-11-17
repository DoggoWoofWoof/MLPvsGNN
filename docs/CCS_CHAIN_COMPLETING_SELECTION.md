# CCS: chain-completing selection over the GNN's top five (the after-paper plan's item 4a)

Declared 17 November 2026, about 15:25 IST, before any CCS number. Code: `outputs/mp_unified/zccs.py`. CCS writes only
under `outputs/ccs`. It trains nothing, changes no fit, graph, pool, link, encoder or feature, and edits no other script:
it re-reads the GNN fits of record with their per-row scores kept, then picks a different top five from those scores and
the links the fits already read. Status: declared.

## 1. The question

- **Where the gap is.** PH1 (docs/PH1_PER_HOP_READS.md, descriptive) splits B1's reads by hops. On MuSiQue the GNN (zg6r,
  b1w, in-domain) has every gold in its top five for 61.7% of the 2-hop questions, 22.8% of the 3-hop and 2.6% of the 4-hop.
  On 2Wiki the four-gold bridge_comparison kind is the weak one (FC@5 70.0, against 87.6-100 for the others). PH1 names
  chain-completing selection as what would act there; it tested nothing.
- **What is closed.** Every filed form so far anchored on one found row and chose among its neighbours:
  - DIAG_BRIDGE's C1, C2 and O1 (docs/DIAG_BRIDGE.md, zrm, REACHABLE_NOT_RANKED): fixed swaps of ranks 4 and 5 for the
    top-1's best-scored neighbours. They lost R@5 on MuSiQue and 2Wiki. Ranks 4 and 5 often held golds already, and
    MuSiQue's missed gold ranked a median 5th to 9th of about 45 neighbours.
  - CHN's H1, H2 and S (docs/CHN_CHAIN_COMPLEMENT.md, rrf, NO_SIGNAL): the top-1's neighbours by complement scores.
    H1_pair helped 2-hop questions only.
  - LTM, LTC, LTG and LTX: learned next-hop scores one to three links out, as inputs or inside message passing.
- **What CCS asks instead (a set question).** Does the GNN's top five hold a row linked to nothing else in it (off the
  chain)? And is there a row outside it linked to two or more of its rows (the chain's missing middle)? Two anchors narrow
  MuSiQue's 32-45 neighbours of one found row to the few rows that touch two. Dropping only an unlinked row keeps the
  linked golds that DIAG_BRIDGE's swaps threw out.
- **Which models.** The GNN of record only: zg6r (B1P's b1w reads) and GNX (IDPA's b1w reads). The MLP (zrckpn) is left
  out: a link-based choice after scoring is message passing, and the MLP stays free of it.

## 2. What it reads

- **The fits.** The twelve GNN fits of record: `scr-ml1-zg6r-s{0,1,2}` and `scr-gnx-nx-s{0,1,2}`, each with and without
  `-hp` (WQ-musique and WQ-hotpotqa). Nothing is refitted.
- **The re-reads.**
  - Each fit is forked from `outputs/screen/fits/<fit>` into `outputs/ccs/fits/w/<fit>`, as B1P and IDPA forked it. Its
    reads/ and read.json are left out. Its models.pt must equal the record fork's (`models_sha256` in B1P's or IDPA's
    fork.json).
  - Each fork is read on B1's three b1w carves on the host GPU, because the fits are cuda and the reads refuse another
    device. The read is the record's own code: zb1p.read for zg6r and zidpa.read for GNX, with their roots, gate and
    carves, and only their output folder moved to `outputs/ccs`.
  - **The one change** is lean_gpu.read_carve's `first_q` default, raised to cover every question, so the read keeps
    every question's per-row scores. The filed reads keep the first 64.
  - The read must then hold `n64` equal to its question count, and the rows of `scores64` equal to the carve's rows, or it
    exits 1.
- **The candidate.** p@swa, the read candidate of record (zd83rep.CAND).
- **The pools and links.**
  - B1P's cache (`outputs/b1p/cache/<d>/b1w`) gives pool sizes, gold flags, gold totals and ids.
  - B1P's links (`outputs/b1p/links/<d>/b1w`) give the pool-local links with their families, as the fits read them:
    - 0, the slot (S on these passage graphs);
    - 1, NER (spaCy);
    - 2, K (kNN).
  - ltwdiag.part_arrays ties each links part to its cache part by the links record.

## 3. The forms (parameter-free; no label is read)

Per question:

- **The base order.** The pool's rows by p@swa's score, descending, NaN last, ties by pool position: lean_gpu.top_hit's
  order.
- **T** is the first five rows of that order (all rows when the pool has fewer).
- **Linked** means joined by a link of family 0 or 1 (S or NER), in either direction.
  - K is left out of the forms. The top rows lie near the question in the vector space, and so near each other, so K would
    join most of T and say nothing about a chain.
  - The same counts over all three families are reported beside (section 4).

**BF (bridge fill), at most one swap:**
1. **The bridge b.** Among the rows outside T linked to at least two distinct rows of T, b is the first in the base order.
   With no such row, T stands.
2. **The drop d.** Among T's rows at base ranks 2 to 5 that are linked to no other row of T and not to b, d is the last in
   the base order. With no such row, T stands.
3. **The new top five** is T without d, with b.

**BJ (bridge join), at most one swap:** as BF, but the bridge must join two parts of T. b must be linked to two rows of T
that lie in different connected parts of T's own link graph (the links among T's rows, families 0 and 1). So it completes
a chain between two found pieces, not one already joined.

**What the forms hold fixed:**
- Base rank 1 is never dropped, so hit@1 is the base's by construction (checked).
- R@5 and FC@5 are measures of the set, so the order inside the new five does not matter.
- A pool of five rows or fewer is never changed.

## 4. Beside, deciding nothing (reported before the calls)

All per question, then by cell and by PH1's groups: MuSiQue by hops (2, 3, 4); 2Wiki and HotpotQA by kind.

- **Reach.** Over the questions with at least one in-pool gold outside T:
  - adj1 and adj2: the share of those golds linked to at least one, or at least two, rows of T;
  - the same two shares among the non-gold rows outside T, and the lift (gold share over non-gold share);
  - each over families 0 and 1 (the forms' links) and over all three.
- **Reach bounds (oracle; they read the gold flags).**
  - O2: T's golds plus every missed in-pool gold linked to a row of T, at most five.
  - O2b: the same with "linked to at least two rows of T".
  - O1s: one swap of a non-gold row of T for one such two-anchor gold, when both exist. This is BF's own ceiling.
- **Isolation precision.** Among T's rows at base ranks 2 to 5: the gold share of the rows linked to no other row of T,
  and of the linked ones.
- **Firing and outcome.** Per form:
  - the share of questions it changes;
  - among those, the counts of gold in / non-gold out, non-gold in / gold out, both gold, and neither gold.
- **FC@5** with its interval in every cell, and R@5 and FC@5 in every group.

## 5. Gates

- **G1, identity.** Checked in the report, on the laptop, which holds every record read.
  - Each re-read's p@swa top and hit must equal the record's read of the same fit and setting, for every question. The
    record reads are `outputs/b1p/fits/w/<fit>/reads` for zg6r and `outputs/idpa/fits/w/<fit>/reads` for GNX.
  - If they differ, the re-read is accepted when its mean R@5 lies within 0.002 of the record's (DIAG_BRIDGE's
    tolerance), and the report says so.
  - Beyond that, the fit's cells are INCOMPLETE.
- **G2, alignment.** Checked in each analysis, on the host. The top and hit recomputed from the re-read's full scores and
  the cache's gold flags must equal the re-read's own top and hit, for every candidate it holds. Its ids, gold totals and
  row count must equal the cache's. Otherwise it exits 1.
- **G3, the links.** Each links part's record must name its cache part (ltwdiag.part_arrays).
- **G4, the models.** The fork's models.pt must equal the record fork's, by sha256.
- **G5, the patch.** n64 must equal Q in every re-read (section 2).

## 6. The rule

- **The cells.** D83's B1 cells (zd83rep.B1_CELLS) for each of the two GNN kinds, ten in all:
  - MuSiQue, HotpotQA and 2Wiki in-domain;
  - MuSiQue and HotpotQA held-out.

  Each cell holds its fits at seeds 0, 1 and 2 (2Wiki in-domain holds both splits, six fits).
- **One cell's call.** Per question, the mean over the cell's fits of the form's R@5 minus the base's R@5. The base is the
  same re-read's own top five.
  - The interval is zd83rep.interval: the paired bootstrap over questions, 2,000 resamples, seed 0, over the questions
    with a gold.
  - ABOVE when its lower end is above 0, BELOW when its upper end is below 0, LEVEL otherwise.
- **Each form's verdict:**
  - NOT_ADOPTED: any cell BELOW;
  - ADOPT: no cell BELOW and at least one ABOVE;
  - LEVEL: every cell LEVEL;
  - INCOMPLETE: a cell missing or refused by G1.
- **If both forms ADOPT,** the one carried has the larger mean difference over the ten cells (ties go to BF).
- **The three-seed rule holds:** seed 0's reads come first and are reported as development numbers, but only the
  three-seed cells call.

## 7. What follows

- **ADOPT.**
  - The form is read next on the six s1eval carves (squad, hotpotqa, 2wiki, musique, metaqa, webqsp). Each needs a
    full-score re-read of the same fits, declared in its own file. Only then is it carried as a universal second stage.
  - In the journal's tables it is a row "GNN + CCS" beside the GNN, never in its place.
  - Once D83 files its fits, the form is read on them (their reads keep 64 questions' scores, so that also needs re-reads).
- **NOT_ADOPTED or LEVEL.** The hard set swap closes for these forms. The verdict closes only them. Two next forms are
  named here and not declared:
  - a soft two-anchor term inside the GNN's readout (DIAG_BRIDGE's "soft signal inside the scorer");
  - the learned three-way score of (question, found passage, row) that CHN named.
- **In every case,** section 4's reach and bounds go to the journal's per-hop analysis (the guide's point (d)), beside
  PH1.

## 8. Runs and order

- **`zccs.py fork`** ran on the host CPU at 15:24 on 17 November, before this file was committed. It copied the twelve
  fits into `outputs/ccs/fits/w` (178 MB; G4 held on all twelve), so the cleaner cannot remove their source before the
  reads.
- **GPU rows** `ccs-rd-<fit>` (twelve), seed 0 first. Each runs `zccs.py read --name <fit>` under cuda_alloc at 0.27 of
  the card, B1P's and IDPA's read size.
  - They sit after D83's GPU rows in the feeder.
  - Like every GPU row, they wait while another host user's job holds the card and our paused trainer keeps its share.
- **CPU rows** `ccs-an-<fit>-<d>` (36). Each runs `zccs.py analyse --name <fit> --dataset <d>` after its fit's read and
  writes `outputs/ccs/sel/<fit>/<d>.npz` and `.json`.
- **The report runs on the laptop:** `zccs.py files` prints the fetch commands, then `zccs.py report` writes
  `outputs/ccs/report.json` and `report.md` with G1, section 4 and the calls.
- **`zccs.py selftest`** must pass before any row is queued:
  - On the laptop:
    - both forms and section 4's counts against a brute-force reference on 3,000 random pools (ties, NaN, -0.0, all
      three families, pools of 0 to 40 rows);
    - the top-five order against lean_gpu.top_hit itself, on the CPU;
    - the report's G1, calls, verdicts and carry rule on synthetic analyses built over the record's own base reads.
  - On the host, G2 and G3 on the record reads' first 64 questions, the rows whose scores are filed. No form's measure is
    computed or printed there.

## 9. Risks

- **Deep pools.** MuSiQue's b1w pools hold about 3,460 rows per question, so many non-gold rows may touch two rows of T by
  chance. adj2's lift says how selective two anchors are, and the forms take only the best-scored such row.
- **Comparison questions.** Their two golds need no link between them (PH1), so a gold may be unlinked in T. BF and BJ
  drop the lowest-ranked unlinked row first, so a comparison gold at rank 2 goes only when ranks 3 to 5 are all linked. The
  isolation precision and the per-kind groups show it, and any BELOW cell stops adoption.
- **Hop mix.** About half of MuSiQue's questions (518 of 1,000) are 2-hop. A 3- and 4-hop gain can be offset in the
  dataset cell by a 2-hop loss. The cells decide; the hop groups are read beside.
- **Device.** The re-reads run on the host's RTX 4500 Ada, as the record's reads did, and G1 checks them against those
  reads.
