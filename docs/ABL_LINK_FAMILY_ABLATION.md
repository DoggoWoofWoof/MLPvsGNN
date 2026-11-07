# ABL: what each link family carries, with the models of record fixed, on every dataset

Declared 6 November 2026, before any ABL number. Redefined three times the same evening on the user's corrections
(first declaration 590f7a4; second bd1db44; third 2b4c9cd, amended 4a61bf1; this is the fourth). Follows
[U2C_POOLS_ON_CHOSEN.md](U2C_POOLS_ON_CHOSEN.md), [U2B_LINK_RULE.md](U2B_LINK_RULE.md),
[B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md) (the b1t arm) and [P2A_LINK_FAMILIES.md](P2A_LINK_FAMILIES.md). Code:
`outputs/mp_unified/zabl.py` (the arms, reads and report), `zabl_u.py` (the rule on the two KB datasets) and `zabl_g.py`
(the slot graphs and their types), each committed before it runs. Outputs: only under `outputs/abl/`. No filed pool,
fit, read or record changes.

## 1. Definitions (the user's rulings of 6 November, applied to all six datasets)

Three families, the same three on every dataset. A family is in or out as a whole.

| letter | what it is | stores it fills |
| --- | --- | --- |
| **S** | the original structural edges the dataset's graph store comes with, whatever they are | the `structural` slot, built from the store's own file (graph "today") |
| **R** | the rules as one family: the rule-built links (U2b's W2_S, the recreated hyperlinks) together with the NER store | the rule graph joins the `structural` slot; the NER store is the `ner` store |
| **K** | nearest-neighbour links in the frozen encoder's space | the `knn` store |

What each is, per dataset (edge counts are adjacency entries of the files the stage reads):

| dataset | S as shipped | R's rule graph | R's NER store | K |
| --- | --- | --- | --- | --- |
| 2Wiki | the corpus hyperlinks (5,989,847 nodes, 56.4M entries) | W2_S on the passages | the package's | the package's |
| HotpotQA | the corpus hyperlinks (5,233,329 nodes, 30.2M) | W2_S on the passages | the package's | the package's |
| MuSiQue | title-mention links (117,534 nodes, 5.3M) | W2_S on the passages | the package's | the package's |
| SQuAD | title-mention links (20,233 nodes, 1.49M; the store does ship one) | W2_S on the passages | the package's | the package's |
| MetaQA | the KB triples | **built here (section 8)**: the same rule on each entity's name and KB facts | the package's | the package's |
| WebQSP | the KB triples | **built here (section 8)** | the package's | the package's |

The record is not a cell on every dataset. The pools of record (U2c) run **R + K** on the four passage datasets (the rule
graph replaced S in the slot) and **S + NER + K** on the two KB datasets (no rule graph existed there). So SRK, with S and
the rule graph both in, is a new build on all six datasets. The passage record RK is ML1's reads and is not rebuilt; the
KB records are ML1's reads of a different pool and are kept beside the grid as a reference, not as a cell.

## 2. Arms

The eight subsets of {S, R, K}. A family that is out is an empty store (same node count, no edge) in every process that
opens it. B, the seeds, the first stage, the rule, its budgets and cuts are the record's. What fills the `structural`
slot: S alone is the original file; R alone is W2_S; S with R is the **union** of the two (undirected neighbours, rows
sorted by node id); neither is empty.

| arm | slot | ner | knn | note |
| --- | --- | --- | --- | --- |
| SRK | S ∪ R's graph | in | in | the full set; new build on all six |
| SR | S ∪ R's graph | in | out | K removed |
| SK | S | out | in | R removed |
| RK | R's graph | in | in | S removed; passage = the record (ML1's reads), KB = new build |
| S | S | out | out | |
| R | R's graph | in | out | |
| K | empty | out | in | |
| 0 | empty | out | out | the first stage only |

Build order (the user's asks first: take R out, take K out, take both out, from the full set): SRK, SK, SR, S, then RK
(KB), R, K, 0. The comparison base for every cell is SRK, on all six datasets.

## 3. Carves, fits, reads

- **Carves:** the six development carves PKO builds (`(ds, "s1eval")`: webqsp 1,503; musique 2,417; hotpotqa 7,405;
  metaqa 9,785; squad 11,873; 2wiki 12,576; 45,559 questions). Nothing is refitted.
- **Fits:** ML1's twelve pair-1 fits (zg6r, zrckp; seeds 0-2; WQ-musique and WQ-hotpotqa), forked without their reads to
  `outputs/abl/fits/<arm>/<fit>`. Each model reads the arm's pools, look, cache, links and KT in place of the record's.
  Both models read the same inputs.
- **Read:** ML1's read with `ML.ROOTS` and `ML.GATE` moved to `outputs/abl/<arm>/`; carves, regimes (in-domain, held
  out) and metric are the record's. R@5 on all questions. MetaQA has no in-domain fit (held out in both splits), WebQSP
  no held-out one, as in ML1.
- **Not in this stage:** the second pair (zrckpn, GNX) needs the next-hop columns rebuilt per arm (amendment A1, after
  IDP files). A refit on each arm's own pools is amendment A2 and the user's call. ABL measures what the families carry
  for models fitted on the record's pools (B1P's "carry-over").

## 4. What is checked

1. B, the seeds and the first-stage lists of every question equal the record's (none reads a graph).
2. A family that is out holds zero edges wherever it is loaded (asserted at load).
3. PKO's section 5 gate on the arm's carves as B1P ran it, the checks against U2s's arrays (graph-dependent by design)
   replaced by check 1; the replaced checks are named in the arm's gate record.
4. Rows per carve and arm are filed beside (costs only).
5. Before any number is used, the passage record re-read through ABL's own read path on musique/s1eval, seed 0, both
   models, equals ML1's read to 1e-4 (the GOK check; it moves from webqsp, whose record pool is no longer a cell).
6. The union slot graph: every row is the union of the two sources' rows (all rows checked), with its source files' sha256.
7. The KB rule graph (section 8): its build record names the node file's sha256, the text recipe's sha256 and the rule's
   code; beside it the share of its edges that are also KB triples (filed, not a gate).
8. The types of each slot graph are fit on that graph (never carried over from another), by the record's script
   unchanged (`zu2m_a_types.py`'s types stage, called by `zabl_g.py` with the slot graph in place of the record's): the
   EM centers on 200,000 pairs of the slot graph and every entry's class; the record's own files serve where the slot
   is the record's graph (R on the passage datasets, S on the KB). On the KB a slot that holds S's typed rows keeps
   them, and an entry only R's graph supplies has an empty block (the carried walk, RWe, weights an entry by its class,
   so no block is read). The passage score classes (`zu2p.py`) exist on the rule graph's pairs only: U2k scores a
   mention link or a sibling pair and refuses any other neighbour. So a slot's score class for a pair is the record's
   class where the rule graph holds the pair and the null class where it does not (S alone: every class null); the
   equal-count cuts over the scored entries are R's in every slot that holds R. Records name the graph file's sha256
   and the types files' sha256.

## 5. Calls (declared before any number)

For each dataset, model, regime (in-domain, held out) and arm: R@5 difference in points, arm minus SRK, of the seed mean
(three seeds, both splits where the dataset is read under both), with a 95% interval from 2,000 paired bootstrap resamples
over questions (the seeds' per-question scores averaged). Call per cell: BELOW if the whole interval is below zero, ABOVE
if above, LEVEL otherwise; beside it, the number of the six fits where the arm is below SRK. Derived columns, as
differences of two arm cells with the interval from the paired per-question differences:

- what S adds = SRK - RK; what R adds = SRK - SK; what K adds = SRK - SR; and each family's effect alone (S - 0, R - 0,
  K - 0);
- the GNN - MLP difference per arm (does message passing matter more when fewer families are in).

The stage decides nothing about adoption; it files the grid and the calls.

## 6. Cost and order

PKOH measured 80 to 90 lane-hours for the whole 11-carve PKO build; the six development carves are 60% of its disk and
66% of its rows, so about 50 lane-hours for a record-sized arm; arms with fewer edges cost less. Seven arms per
dataset (RK on passage is free): about 350 lane-hours in all, 30 hours at 12 lanes, plus the slot graphs and types (minutes
each, section 3 of the records) and the KB rule builds. Twelve GPU reads per arm beside the IDP fits. Disk is up to
86 GiB per full arm (look 32 GiB of it, regenerated per arm because the rows differ): arms run dataset by dataset
(chain, gate, twelve reads, prune of its look and cache), so a dataset's footprint is the peak. Inside an arm the carves
go in the order musique, metaqa, hotpotqa, squad, 2wiki, webqsp (the small carves first, and webqsp last because
its rule graph, ABL-U's build on 2.59M nodes, is the longest wait; the order changes no number).

## 7. Not claimed

- No refit: the numbers are the models of record on pools and edges with a family added or missing.
- Development-split reads only; no test split is touched.
- On the KB datasets the rule can read only what the package holds there, the entity name and its KB facts, so its links
  are expected to be mostly KB neighbours already in S; the extras are the alias and sibling links and the NER store.
  That is what the rule can see there, not a finding about the datasets.

## 8. ABL-U: the rule on MetaQA and WebQSP

The user's ruling: R is the NER store and the recreated hyperlinks, applied exactly the same way to every dataset, on
whatever text exists. U2b's rule links a text to the nodes whose titles it names; a KB node's own text is its name, which
names nothing (U2b switch 4), so the rule has nothing to read until a node has text beyond its name. What exists in the
package for a KB entity is its name and its KB facts. So, as the declared recipe (changeable by the user, in which case
the KB arms are rebuilt):

- **Node text for the rule only** (the encoder's node vectors stay the name-only vectors of the substrate): the names of
  the entity's distinct KB neighbours, both directions, in node-id order, joined by "; " (its own name is left out, so no
  name is read as a mention of a shorter name inside it; an entity with no neighbour keeps its name alone and names
  nothing). The rule's boundary, forms, prefix and common-form switches, its AMBIG guard (50), its context choice by the frozen
  dense vectors and its variant W2 with the sibling links (the chosen graph, W2_S) are U2b's code, unchanged
  (`zu2b.Surfaces`, `stat_pass`, `link_pass`, `sibling_pairs`, `undirected_csr`).
- The capitalization statistic that gates common forms is computed over these texts, where every name is capitalized;
  the build record files how many forms it marks common (MetaQA: 149 of its surfaces).
- Output: `outputs/abl/u/<dataset>/W2_S.npz` with a record (node file sha256, the recipe's sha256, the code's sha256,
  edge counts, the share of edges that are KB triples).
- Then the slot graphs (S ∪ R's graph, R's graph) and their EM types (section 4, check 8). On the KB the slot graph keeps
  the KB's typed rows for S's entries; an entry only R's graph supplies has an empty typed run.

## 9. Amendment A2 (7 Nov 2026, 06:51 IST, before any ABL number): the chained carves' steps and the arm's typed links

**What happened.** Section 6's chain ended its steps at links, KT and the record. On the chained carves (WebQSP and MetaQA;
PKO's `CHAINED`) PKO's record also names the rmatch and zrc chain records and the typed arrays, which PKO builds between the
cache and the KT. The first chained chain (arm SRK, MetaQA) built its pools, look shards, cache parts, links and KT in 2 h 38 min
and then its record step wrote a record that named `no rmatch records, no zrc records, typed/metaqa/s1eval/record.json is
missing` and exited 1. No read had run and no number exists.

**The correction (code only; calls, arms, reads and rules are unchanged).**
1. For the chained carves the chain builds rmatch and zrc (PKO's `chain_build`), then links, then typed (PKO's `typed_carve`),
   each on the arm's own roots and slot graph, in the order of PKO's `post_steps`. The passage datasets (MuSiQue, HotpotQA,
   SQuAD, 2Wiki) are not chained: their chains are exactly the ones section 6 describes, and MuSiQue's filed SRK chain is
   unchanged.
2. PKO's record step writes the record and then exits 1 when it names problems, and the chain took that file for a filed record.
   A record that names problems is now removed (its problems are logged) and the record step runs again. It is ABL's own file
   under ABL's roots.
3. ML1's read takes the chained carves' typed arrays from PKO's root (`zml1s.TYPED`). Sections 4 and 8 make the typed rows
   part of the arm (the slot graph keeps S's typed rows; an arm without S has none), so ABL's read now takes the arm's own
   typed arrays (`outputs/abl/<arm>/typed`). Check 5 (arm REC, MuSiQue) is a passage carve and does not use them.

**What it costs.** One more rmatch, zrc and typed step per chained chain (measured on SRK, MetaQA: rmatch 79 s, zrc 78 s, typed 372 s; the
retry of that chain took 13 min in all, its look, cache, links and KT being filed; WebQSP is not measured yet); every MetaQA and WebQSP chain of every arm carries them. **Not checkable against PKO:** on the KB, SRK adds R's rule graph
(section 8) to the KB's triples, so its rmatch, zrc and typed arrays are not PKO's; the MuSiQue check 5 stays the only
equality check against a record.

## 10. Amendment A3 (7 Nov 2026, 09:50 IST, before any ABL number): the reads use seed 0 only

**What happened.** Section 3 reads ML1's twelve pair-1 fits (zg6r, zrckp; seeds 0-2; both WQ splits) on every arm and dataset: 530
reads in all with check 5, on a GPU that GNC's fits (D6) hold first. The user was told on 7 Nov that this could be cut to seed 0
(about 177 reads) and answered "fine we are short of time anyways finish fast". No ABL number exists: the only reads filed are
check 5's two (musique, seed 0, in-domain), which stay as they are.

**The change.** Each arm and dataset is read by **four fits**: zg6r and zrckp, seed 0, WQ-musique (plain) and WQ-hotpotqa
(`-hp`). The twelve fits are still forked (the fork copies are small and some already exist); only the four are read. In
`zabl.py`: `SEEDS = (0,)` and `FITS` is the four seed-0 fits of B's twelve (the self-test asserts the subset). The prune of a
dataset's look and cache waits for the four reads. In the host queue the reads of seeds 1 and 2 are removed and the prunes
wait on the four. About 176 reads plus check 5's two remain. Nothing else changes: carves, regimes, arms, build, order, the read path, the
bootstrap, the resamples, the intervals and the call rule of section 5.

**What the calls then mean.** A cell is the difference of the seed-0 fit's per-question R@5 (arm minus SRK); the interval is the
paired bootstrap over questions, as in section 5, so it carries the sampling variation of the questions and **not the variation
of the fit's seed**. 'Below' counts the fits of those read where the arm is below SRK: at most two (the two splits) for a dataset
read under both, one otherwise (a one-fit count is a sign, not a vote). A call says what the pools and edges do to **this fit**;
it does not say that another seed would agree. The report and anything written from it name seed 0 only. The seeds 1 and 2 reads
are not queued: the prune removes each dataset's look and cache once its four reads are filed, so adding them later means
regenerating those (lane-hours, section 6), and would be a further amendment.

## 11. Amendment A4 (7 Nov 2026, 10:25 IST, before any ABL number): ABL-U's scan and CSR made linear (same arrays, speed only)

**What happened.** Section 8's WebQSP build (`abl-u-webqsp`, host job 261106-233355) has run 10 h 46 min (38,285 CPU s) in U2b's
first pass (`stat_pass`) without a progress line. The cause is U2b's `scan`: for every surface a node's text matches, `occ_look`
runs `tl.find` over the whole text and `u1d.sent_start` copies `text[:p]`, so a node costs about degree^2.6. WebQSP's KB graph has
31 nodes above 10,000 neighbours and two of 130,075 and 121,511; the texts of those hubs are 1 to 2 million characters. Measured on
the biggest hub's own neighbour names: 5,000 neighbours 0.3 s, 20,000 neighbours 11.0 s. The second pass (`link_pass`) repeats the
scan, so the old job could not finish soon even if the first pass ended now. Its sibling step also holds about 324 million ordered
pairs (the largest title groups are 9,837, 7,793 and 7,384 unnamed nodes) several times over, which is why the row asks 28 GB.

**The change.** In `zabl_u.py` only; `zu2b.py`, `zu2b_fast.py`, `u1b.py` and `u1d.py` are untouched.
- `scan_fast` returns exactly what `zu2b.scan` returns (the hit list in order, the mid-sentence counts, the capmid set). It
  finds a text's bounded occurrences in one pass over the text's word runs (a bounded occurrence of a title is a run of whole word
  runs, so it starts at a run whose tuple the table indexes), applies `occ_look`'s own boundary test, and replaces
  `sent_start`'s copy by a scan back over whitespace. It is installed as `zu2b.scan` for the process, so `stat_pass`, `link_pass`
  and `zu2b_fast.link_pass` all use it. U2b's own is kept (`SCAN_SLOW`) for the checks and for the u1d boundary.
- `lean_csr` returns exactly `zu2b.undirected_csr` of the mention edges and `zu2b.sibling_pairs` (same arrays, dtypes and
  sibling-pair count) from sorted keys, without the pair lists or the second copy.
- `stat_pass_logged` is `zu2b.stat_pass` with a progress line every 100,000 nodes. `--out DIR` writes a build under DIR and, when a
  filed build of the same dataset exists, compares the arrays and the record's counts with it.

**What does not change.** The rule (NEW_RULE, variant W2, form S), the text recipe, the surfaces, every count, the calls, the arms,
the reads and the order of section 3 to 6. No ABL number depends on the speed.

**How it was checked (before it runs on WebQSP).** `zabl_u.py --selftest` and `check`: `scan_fast` equals `zu2b.scan` on 308
random texts with punctuation, parentheses, commas, digits, underscores, non-ASCII letters, newlines, no-break spaces and mixed
case (3,740 hits, 341 capmid, 1,340 mid-sentence counts), under both rules; `lean_csr` equals `sibling_pairs` + `undirected_csr`
on 40 random graphs; on MetaQA all 43,234 node texts agree and `stat_pass`'s `cap` and `low` are identical over 68,447 lowercase
forms (U2b 6 s, fast 3 s); on WebQSP's biggest hub the first 5,000, 20,000 and 60,000 neighbours agree (11.0 s against 0.2 s at 20,000; 182.5 s against 1.4 s at 60,000). The host
rebuilt MetaQA with `--out` (rx job 261107-102609, 22 s) and the arrays (`indptr`, `col`, dtypes) and the record's counts are
identical to the filed `outputs/abl/u/metaqa` built with the old code. Only then does WebQSP start again.

**On the host.** The running WebQSP job is stopped by killing its python process; `retry_cmd.py` (3 tries) re-runs the same command
in the same job with the new file, so the job, its reservation and every row that waits on it are unchanged. The old job's
partial work is thrown away (it kept none: no file, no progress line). The new build's record (`build.json`) carries the new
`code_sha256`; the MetaQA build filed earlier carries the old one, and the rebuild check shows they give the same arrays.

**A4, second part (7 Nov 2026, 11:03 IST, before any ABL number).** The first run of the fast scan on the host (rx job
261107-103311, `abl5-u-webqsp`) had not logged its first 100,000 nodes after 21 minutes, 8 times slower than the same code
on the laptop with a table a quarter of the size. The cause is in the candidate lookup that `scan_fast` took from U2b's table:
at each word run it tries every title length listed under that run as a first word, and WebQSP's titles list 382 lengths
under 'award' (92,000 tuple elements per occurrence), 188 under 'government', 174 under 'regular' and about 100 under 'the',
'i', 'in', 'tv'; the cost grows with the table, and the full table has 2.6 million titles. `pair_index` lists the lengths under
the first **two** runs instead (a title found at run k starts with runs k and k+1, so the pair lists a superset of the lengths
that can match, and the tuple lookup that follows is unchanged). Nothing else changes and the results are the same by
construction; it is checked as before: 308 random texts, 40 random graphs, MetaQA's 43,234 texts with `stat_pass` identical,
WebQSP's biggest hub identical at 20,000 neighbours; on the laptop with a table of 800,000 titles the scan costs 0.48 ms a node
against 0.75 ms before. The running job is replaced by a new row (`abl6-u-webqsp`) with the same command; the five rows that
wait on the build wait on it.

**A4, third part (7 Nov 2026, 11:40 IST, before any ABL number).** WebQSP's 2,592,894 node texts are only 1,109,157 distinct
texts (MetaQA's 43,234 are 36,974), and the link pass scans the same texts as the first pass. A scan depends on the table, the
rule and the text alone, so `scan_memo` remembers `scan_fast`'s three outputs by text (the hits as integers, the capmid set, the
lowercase-form counts) and each distinct text is scanned once for both passes; every call returns fresh objects equal to
`scan_fast`'s, and the memo is dropped after the link pass. Nothing else changes: the rule, the arrays, the calls and the arms
stay. Checked as before, with the memo in the path: the 308 random texts and the 40 random graphs; MetaQA's 43,234 texts
through `scan_fast`, the memo's first call and its second call against U2b's scan, with `stat_pass` identical; and a rebuild of
MetaQA's graph that is identical to the filed one (arrays and counts; 36,974 texts scanned for 86,468 scans). On the laptop,
over WebQSP's first 100,000 nodes with a table of 800,000 titles (74,096 distinct texts among them), the two passes take 99.2 s
with `scan_fast` alone and 60.7 s with the memo, with identical counts and edges. The second part's rebuild check also had its
`for` loop one level too far out (the comparison printed once per key and would have failed on a missing filed build); it is
re-indented and only runs with `--out`. The running job `abl6-u-webqsp` (sent 11:39, two minutes in, with the second part's code) is replaced by `abl7-u-webqsp` with the
same command; the five rows that wait on the build wait on it.
