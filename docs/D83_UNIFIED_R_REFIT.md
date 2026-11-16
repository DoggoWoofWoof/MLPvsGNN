# D83: one refit of the models of record on the unified R (F0T in R's NER store, the S > R > K dedup, edge labels the models read), under A2

Declared 16 November 2026, before any D83 store, carve, fit or read is built. UL1 (docs/UL1_UNIFIED_LINKS_ORACLE_CHECK.md,
filed c662a6b) CARRIED F0T at graph level. The user, 16 November 2026, answering "Declare D83 with pools kept, the S>R>K dedup
and edge labels in one refit?" (verbatim): "Yes, declare D83". The rulings this page carries out, in the user's words:

- "i do not think we need to unnecessarily train models after every improvement, like if we created better links we can train a
  model after verifying those links improved connectivity etc by comparing with the oracle links we have and across datasets etc
  and unifying everything" (UL1 was the verification; D83 is the one training it names).
- "make sure the SRK and the related family rulings are properly followed" (section 0).
- "the edges should also have labels based on how it was created so that we can unify with KB labels and make the model train
  and understand better rather than just giving family names" (section 3).

**What D83 is:** ONE refit of zg6r, GNX and zrckpn (zspk first, as zg6r's start), three seeds, both WQ splits, with three changes
made together: (1) R's NER store is F0T on all six training corpora (spaCy's store leaves every model input); (2) a pair is held by
one family only, S > R > K; (3) every link carries the labels of the rules that made it, and the models read them. The three are
not separated: D83 reports what the change as a whole does to the models of record. Nothing of the record is written; the paper's
numbers do not change.

## 0. The family rulings this page follows (docs/CANONICAL_LINK_AND_POOL_LAYOUT.md)

- **S, R, K** (8 Nov): S = the original structural edges; R = our rule links (U2b's W2_S) plus the NER store, one family; K = the
  3 nearest neighbours. Priority S > R > K.
- **A1** (10 Nov): S only where the data has structure. A with-S fit (S on 2Wiki, HotpotQA and the KBs, ABL-U's rule part in R
  on the KBs) is a new stage: **D89**, not D83 (section 9).
- **A2** (11 Nov): the pools of record stay; a change to a family reuses pools, caches, looks and fits by sha256 and recomputes
  only what reads the changed edges, named before it runs (section 4). Only a ruling of the user's that names A2 changes the
  pool standard. The user's question above named the pools as kept.
- **Layout section 6, items 2, 4 and 5**: the rule-based NER store replaces spaCy's inside R on all six datasets ("R's definition
  does not change when the store does"); the dedup; the refit. D83 is items 2, 4 and 5 in one training.
- **The model rulings**: the MLP stays free of message passing (its inputs are known before any row is scored); the GNN is its
  own model; the same-input pair is GNX and zrckpn (every input zrckpn reads, GNX reads); three-seed records (mean and interval,
  seed 0 beside); judged on unseen data as well as in-domain.

## 1. The domain: the record's own (where the pools of record exist)

D83 keeps the record's slot on every dataset, because the pools of record (A2) were built on it:

| dataset | structural slot (the record's) | NER store | knn store | R in D83 |
|---|---|---|---|---|
| MuSiQue, SQuAD, HotpotQA, 2Wiki | W2_S (U2b's rule links; R) | **F0T** (was N_sp) | the package's K | W2_S + F0T |
| MetaQA, WebQSP | S (the KB's typed triples) | **F0T** (was N_sp) | the package's K | F0T |

So D83 is the record's RK on the passage datasets and the record's S + R + K on the KBs, with R's NER store replaced. The A1
layout (S in the slot on 2Wiki and HotpotQA; ABL-U's rule graph inside R on the KBs) is D89.

## 2. The stores (outputs/mp_unified/d83stores.py, zd83.py)

- **F0T** on a training corpus = U2r's CN_all (`outputs/u2r/<D>/names.npz`) united with UL1's title table LT
  (`outputs/ul1/<D>/f0t.npz`), a < b, ascending; where both hold a pair, LT's weight (AP2's rule: the deepest table's weight).
  Weight = the sum of 1/df over the shared names (U2r's and AP1b's). Built as `FamilyStore.from_graph` builds the package's NER
  store. Its pair count must equal UL1's filed `f0t_pairs` (G2).
- **The dedup (S > R > K).** The knn store loses every pair that the slot or F0T holds; on a KB, F0T also loses every pair S
  holds. Inside R nothing is removed (W2_S and F0T are one family; a pair in both stays in both stores, as in the record). A
  store's kept entries keep their order, so the deduplicated store equals `from_graph`'s store of its kept pairs, entry for
  entry (self-test). Every dataset's counts (pairs in, removed, out, per store) are filed.
- **The labels' key sets** (section 3) are read from the stores before the dedup, so a pair keeps every label it has.

## 3. Edge labels, and how each model reads them

**The vocabulary, one for all six datasets** (UL1's, less `R.ner_spacy`, which no D83 store holds): `S` (a KB triple; its relation
stays on the edge as the record's relation-text code, `zg6r.rel_codes`, KB only), `R.mention` (a W2_S pair), `R.name_comention`
(an F0 pair: two passages share a rule name), `R.title_comention` (an F0T pair that F0 lacks: a passage's own title run shared), `K.knn`
(a pair among the package's 3 nearest neighbours, before the dedup). Five labels, ids 0 to 4 in that order. A pair carries every
label whose rule made it (a bit set); the dedup decides only which family's store holds it.

**Where the labels go.** The look's pool edges stay per holding family (the cache blocks read them as in the record: WALK and
DISTS the slot, WALKF the three families). The links step (zlink.chunk_edges on PKO's chain) writes one link per (question, u, v,
label) instead of per (question, u, v, family): each family edge's pair is looked up in the five key sets, and one link is
written per label it carries, unique. Every reader of the links then reads labels in the place of families:

- **zg6r and GNX** (the GNNs): G6's message maps, attention biases, family means and family dropout run per label (FAMS = 5 in
  place of 3): per-label message passing. The KB relation-text channel (rmatch, typed sets) is unchanged.
- **LTM** (the NEXT columns zrckpn and GNX read): its candidate features hold the five label bits in place of the three family
  bits (`ltwdiag.feature_rows`); its fold fits are fitted again on D83's links.
- **LAB** (new, ten columns per pool row, zrckpn and GNX): for each label, log1p of the row's links with that label to any pool
  row, and to rrf's five leading rows (the first stage's, known before any row is scored). It enters each model the way NEXT
  does (a block of zrckpn's encoder; beside NEXT in GNX's hook), so the same-input pair holds. zg6r reads neither, as in the
  record (it has no NEXT).

The MLP passes no message: LAB and NEXT are fixed columns of the row and its first-stage neighbourhood.

## 4. A2: what is reused, what is recomputed, what is never touched

- **Reused by sha256:** the pools of record, PKO's pools files (`outputs/pko/pools/<D>__<C>.npz`) on the five fit carves and the
  six s1eval carves, and B1P's pools on B1's three settings, hard-linked into D83's roots after a sha256 check (zwq.place); the
  chain's pools and merge steps never run (refused in code). The step-1 bases. The encoder (frozen forever) and every embedding.
- **Recomputed (each reads a changed store or the links):** the KT rows (P1c3's columns read the union walks over all three
  stores; recomputed on the record's pools, `KT3.columns` on each question's filed pool and expansion, the walks on D83's
  stores); the look; the lean cache; rmatch and zrc (chained carves); the links (with labels); typed (chained carves); kt (the
  KT rows in the new cache's row order); the carve records; LTM's lists, fold fits and NEXT columns; LAB; every fit.
- **Never touched:** `outputs/pko` (the record's roots), the record's fits, columns, reads and reports, `outputs/b1n`, the
  bench folders of b1w and b1n, and the J5S hold's folders.

## 5. Builds (CPU)

- **Arms:** `D83I` (G1's identity build: squad's fit carve and metaqa's s1eval carve, one passage and one KB carve), `D83F` (the five fit carves: squad, hotpotqa, 2wiki `fit`; musique
  `s1fit`; webqsp `fit`), `D83E` (the six s1eval carves), under `outputs/abl/<arm>/`, on ABL's chain (zabl.py, unedited) driven
  by zd83.py: link the pools, KT pass, look, cache, rmatch and zrc, links (labels), typed, kt, record, gate. `zd83.py build`
  runs the first part (link the pools, look, bases, cache); the rest runs as `zd83.py chain` once its code is committed.
- **B1's read carve `b1u`:** b1n's bench folder (W2_S, F0T, K) with K deduplicated against W2_S and F0T, plus the setting's five
  key sets (F0 = AP1b's L0, LT = L0T, K = b1w's knn before the dedup); zb1h's chain on B1P's pools (linked, not rebuilt), KT
  recomputed, then lists and columns per split; outputs under `outputs/b1u/`.

## 6. Training (GPU and CPU), in a world of its own

A tree `d83world` (the SWP-E pattern, `outputs/host_ops/nworld.py` → `d83world.py`): its `outputs/pko` is a junction to
`outputs/abl/D83F`, its fit, LTM and screen folders are its own, FAMS = 5 is set in its processes, and every D83 train step
refuses to run elsewhere, so no fit named like a record fit is written outside it. Fits, as ML1 and LTM made the record's:

| fit | names (both splits, `-hp` = WQ-hotpotqa) | seeds | start |
|---|---|---|---|
| zspk | scr-ml1-zspk-s0 | 0 | none (zg6r's start) |
| zg6r | scr-ml1-zg6r-s{0,1,2} | 0, 1, 2 | the split's seed-0 zspk encoder (ML1's rule) |
| GNX | scr-gnx-nx-s{0,1,2} | 0, 1, 2 | the same seed's zg6r (ml1start.LOAD) |
| zrckpn | scr-ltm-zrckpn-s{0,1,2} | 0, 1, 2 | none (as LTM's) |

20 fits (14 on the GPU). Each D83 fit differs from the record's fit of the same name in its inputs and, for the GNNs, in the
label axis (FAMS = 5) only: same epochs, widths, optimiser, seeds and candidate (p@swa). A changed encoder input width (LAB; a
block dead on the training rows) is recorded per fit and is not a gate failure, as SWP-E's amendment A2 recorded it.

## 7. Reads, and the rule (fixed now)

- **Deciding:** B1's three settings on `b1u` (the D83 fits) against `b1w` (the record's fits; filed reads, by sha256), R@5 per
  question, the three-seed mean, for zg6r, GNX and zrckpn; in-domain (MuSiQue ← WQ-hotpotqa, HotpotQA ← WQ-musique, 2Wiki ← both)
  and held-out (MuSiQue ← WQ-musique, HotpotQA ← WQ-hotpotqa): 15 cells. Each cell: D83 minus record on the per-question
  three-seed means, the ratio estimator's paired bootstrap (2,000 resamples, seed 0), 95% interval: **ABOVE** (bottom above 0),
  **BELOW** (top below 0), **LEVEL** otherwise. Seed 0's difference beside.
- **The call:** **ADOPT** when no deciding cell is BELOW (the unified R becomes the after-paper record's input set for the
  journal program); **NOT_ADOPTED** otherwise, which closes this form only and names the next (D91: the same refit without the
  labels, to separate the labels from F0T and the dedup).
- **Reported, not deciding:** the six s1eval carves (SQuAD and WebQSP in-domain, MetaQA zero-shot, and the three B1 datasets),
  D83 at three seeds against every record seed filed; FC@5 and hit@1; MuSiQue by hops (2, 3, 4) and 2Wiki/HotpotQA by kind
  (PH1's groups); the gap GNX minus zrckpn and zg6r minus zrckpn at D83 beside the record's; per-label counts of links and of
  gold-touching links.

## 8. Gates (a failure stops the dataset's or the fit's rows; none of its numbers is read)

- **G1 identity:** `D83I` = zd83's chain with the record's stores (N_sp, no dedup, families as link labels, FAMS = 3) on
  squad's fit carve and metaqa's s1eval carve reproduces the record's look, cache parts, links, typed arrays and KT rows array
  for array (each record's array sha256). It proves the plumbing (pools linked, KT pass, on_arm, the labelled links build)
  changes nothing by itself. G1's rows go first; the D83F and D83E look and cache builds run beside them (they read no
  result), and no labelled link, KT pass, fit or read of D83 runs before G1 passes.
- **G2 inputs:** F0T's pair count equals UL1's filed count; U2r's and UL1's files equal their records' sha256; the node counts
  agree.
- **G3 pools:** every linked pools file and its manifest equal the record's by sha256; KT rows = pool rows per question.
- **G4 dedup and labels:** no knn pair inside R or S; no NER pair inside S on a KB; every link carries at least one label; the
  per-label link counts are filed; on a passage dataset no `S` label, on a KB no `R.mention` label.
- **G5 world:** every D83 train step refuses outside `d83world`; after the fits, the record's fit folders' sha256 are unchanged.
- **G6 reads:** question order and gold equal the record's reads; the record's reads match their filed sha256.

## 9. What follows, and the open numbers

- ADOPT: the journal program's runs (D84 onward) read the D83 fits. **D89** (open; default after D83 files): the A1 with-S layout fit.
- **D90** (open; default none): a pool rebuild on D83's stores; only on a ruling of the user's that names A2.
- **D91** (open; default only if D83 is NOT_ADOPTED, or on the user's word): D83 without the labels.
- MuSiQue's fourth hop: UL1 found chain connectivity at four hops unchanged (about 7 to 9%); D83 cannot fix a link the graph
  lacks. The by-hop report says where the change lands.

## 10. Rows, order and cost (estimates; the host's measured peaks replace them)

0. The code goes in by parts, each self-tested and committed before its step runs, none changing a rule on this page: this
   page's commit carries the stores, the pools link and the look and cache builds (`zd83.py build`; the later steps refuse);
   the labelled links, the KT pass, typed, kt and record, then the world tree, LTM, LAB, the fits and the reads follow.
1. G1 on squad's fit carve and metaqa's s1eval carve (`D83I`): a few hours CPU.
2. `D83F` five chains, queued with G1 (look and cache dominate; MuSiQue's s1fit is the longest, about 6-8 h without pools), the KT pass per carve
   (a fraction of PKO's pools pass), then the gate: about 12 h wall on 3 lanes.
3. In the world: LTM's lists, fold fits and NEXT on D83F; LAB; zspk (2 GPU fits), zg6r (6), GNX (6) on the GPU, about 16-20 h;
   zrckpn (6) on the CPU beside them.
4. `D83E` six s1eval chains and `b1u` three settings beside the fits (CPU), then lists, columns and reads (12 reads per setting
   per model family, as AP2's), then the report.

About 2.5 to 3 days of host time in all, with the GPU busy about 20 h. ETA for the report: about 19 November 2026.

## 11. What it can and cannot show

It can show whether the models of record, trained and read on the unified R with labelled links, read B1's three settings
better, the same, or worse, at three seeds and with intervals over questions. It cannot attribute a change to F0T, the dedup or
the labels separately (D91 would split the labels off). It is not an LLM comparison and not an encoder change. Its numbers go
into no paper sentence before section 12 files them.

## 12. Results

(Not yet filed.)
