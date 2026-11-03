# L2D: are the unlinked next-hop golds two links from their anchor, and do two links single them out?

Declared 3 November 2026, before any number. Script: `outputs/mp_unified/l2ddiag.py`. Training-free, host CPU, fit
carves only.

- **Module:** `l2ddiag.py` (new). It imports `lthcols` (question, a_case, xc_take, toy_carve), `ltwdiag` (fam_bits,
  part_arrays, write_json, the toy tree), `chndiag` (through lthcols), `zwq` (sha_src) and `lean_cache` (part_dirs,
  sha_file). None of them is edited.
- **Reads:** PKO's cache and links parts of musique/s1fit, hotpotqa/fit and 2wiki/fit, read only. Also
  `outputs/lta/report.json`, LTA's census of the same A cases, read only and checked.
- **Writes:** `outputs/l2d` only: one census per carve (`<dataset>__<carve>.json`), plus `report.json` and `report.md`.

## 1. The question

LTH's section 7 names the next MLP form under every outcome: the term on two-link rows (LTM's lists widened by one link,
so each anchor's term covers the rows two links from it). Building it means lists, a scorer fit per split, columns and
a zrckp fit; LTA's took four to six hours of host CPU. L2D asks first whether the links two steps out carry anything
for that term.

The filed facts behind the question:

- **LTA's census.** On musique/s1fit, 1,262 of the 3,798 A targets (0.332) are linked to their anchor. On 2wiki/fit
  the share is 0.612, and on hotpotqa/fit 0.785. So two thirds of musique's next-hop golds have no link to the found
  gold.
- **LTA's screen (NOT_ADOPTED: 1 LOSS, 9 WITHIN).** Its learned term over every row ranks the unlinked targets no
  better than rrf does. On musique/s1eval the unlinked targets' T2 is 0.067; rrf's T2 on all its targets is also 0.067.
- **So for musique's unlinked two thirds, nothing lifts them:** not the vectors, not the ranks, not one link.

A two-link term adds one thing: the shape of the links two steps out. That is whether a row is two links from the
anchor, through how many middle rows, and by which families.

- If two links reach the unlinked targets, and single them out from the rows of the same rrf rank, the term has
  something to learn and is built.
- If they do not, it is not built, and we know why.

## 2. What it reads

- **The A cases** are LTA's and LTH's (`lthcols.question` and `a_case`). A case is a question whose rrf top five holds
  a gold and whose rows outside the top five hold one.
  - The anchor is the first gold among the top five.
  - The candidates are every row outside the top five.
  - The targets are the golds among the candidates.
- **The links** are the question's PKO links (pool-local). They are read in either direction, without repeats or
  self-links, over the families 0 structural, 1 NER and 2 kNN.
- **Two checks stop the census (exit 1) on any difference:**
  - Per carve, the questions, rows, A cases, targets and linked targets must equal LTA's census in
    `outputs/lta/report.json`.
  - On every case, a row's depth 1 from the anchor must be LTW's link to it (`fam_bits > 0`).
- **Carves:** musique/s1fit decides; hotpotqa/fit and 2wiki/fit are read beside. Fit carves only, never s1eval or test.
- **Nothing else:** no vectors, no fits, and no labels beyond the gold flags the A cases already use.

## 3. What it measures

**Per case:**

- each candidate's depth from the anchor: 1, 2, 3, or far (more than three links, or not linked at all);
- its depth from the nearest top-five row;
- its depth on each family's links alone;
- for a row two links from the anchor:
  - its middle rows: the anchor's neighbours it is linked to;
  - whether a middle row is a gold, or one of the top five.

A row is **unlinked** when it has no link to the anchor (depth 2 or more).

**The reads, pooled over the carve's A cases:**

- **share2:** the share of the unlinked targets that are two links from the anchor.
- **lift2:** share2 over the unlinked candidates' share at depth 2.
- **Matched lift2:** the observed count of unlinked targets at depth 2, over the expected count.
  - The expected count comes from each case's rrf deciles (of its unlinked candidates). Within a decile, a target is
    expected at depth 2 as often as that decile's unlinked candidates are.
  - 1.0 means two links reach the targets no more often than they reach rows of the same rrf rank.
- **Matched AUC of the middle count:** the chance that an unlinked target has more middle rows than an unlinked
  non-gold of the same case and rrf decile (ties count half), pooled over such pairs.

**Read beside:**

- **Unmatched AUCs** of the middle count and of rrf, among the unlinked candidates.
- **Middle-count bins** (1, 2 to 3, 4 or more), with their lifts.
- **Two-link targets** with a gold middle row, or a top-five middle row.
- **Depth from the top five:** targets and candidates.
- **Each family on its own**, and **each musique hop**.
- **Per-case sizes:** candidates, unlinked rows, two-link rows and mean degree.
- **Three fixed orders** of each case's unlinked candidates:
  - rrf's order;
  - two-link rows first, then rrf;
  - by middle count, then rrf.

  For each order: the unlinked targets' T2 (LTW's: the share ranked first or second), T10, median rank and MRR.

## 4. The rule (fixed now)

The call is made on musique/s1fit, from the anchor:

- **INCOMPLETE** if any of these holds:
  - there is no census made by this script;
  - LTA's check is not "equal";
  - there are fewer than 200 A cases.
- **SIGNAL** iff share2 >= 0.30, and either matched lift2 >= 1.5 or matched AUC of the middle count >= 0.65.
- **NO_SIGNAL** otherwise. The reading is FAR if share2 < 0.30, else NOT_SELECTIVE.

hotpotqa/fit and 2wiki/fit are reported beside and decide nothing.

**Why these bars:**

- With share2 >= 0.30, the term reaches at least 0.30 × 0.668, about 0.20, of musique's A targets beyond what one link
  reaches.
- **The matched bars ask for information beyond rrf.** That means 1.5 times the rate of rows of the same rrf rank, or a
  0.65 chance of placing a target above a same-rank negative.
- **The unmatched reads are reported but not used,** because rrf alone can produce them. Rows near the top of rrf link
  to each other more often: kNN links join similar rows.

## 5. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| SIGNAL | The two-link term (LTH section 7's next form) is declared in its own file now: each anchor's term on the rows two links from it, with the middle count and the two links' families as inputs. Its lists and columns are built on the host's CPU, so its screen can start when LTH's verdict lands. On LTH's ADOPT it uses LTH's draw; otherwise LTA's uniform draw (LTH section 7). |
| NO_SIGNAL, FAR | The two-link term is not built, and LTH section 7's next form closes on this read. Most unlinked targets are three or more links from their anchor, or not linked to it at all, so the MLP's next-hop line stops at the links' reach. That reach goes to the links track as its open question: which rule reaches musique's next hop. No MLP term follows from it. |
| NO_SIGNAL, NOT_SELECTIVE | The two-link term is not built, and LTH section 7's next form closes on this read. Two links reach the unlinked targets, but no more often, and through no more middle rows, than they reach rows of the same rrf rank. No column drawn from the links' shape two steps out separates them, so the MLP's next-hop line ends at one link. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

- No GNN form waits on L2D, and no other stage reads its outputs.

## 6. Risks, stated before any number

- **Pooled shares weight large pools.** LTA's census puts musique's candidates per case at a median of 5,068.5 and a
  p90 of 7,032.6. The per-case sizes are reported.
- **Deciles are coarse.** Rank still varies within an rrf decile, so the matched reads may keep a little of rrf's
  signal. A case with few unlinked candidates puts one or two rows in each decile.
- **One anchor.** The anchor is the first gold among the top five (LTA's choice). A target two links from another
  top-five row, but not from the anchor, counts as far from the anchor. Depth from the top five is reported beside.
- **Depth cap.** Depth stops at three, and rows beyond it count as far. That is enough to tell two links from more.
- **Either direction.** PKO's links are read in either direction, as LTW's `fam_bits` reads them.
- **A fixed order is not a learned term.** The fixed orders' T2 illustrates what the structure alone does; it is not a
  forecast of a screen.
- **Time.** LTA's lists read the same parts the same way, plus the A lists, and took under 25 minutes per carve. L2D
  adds five breadth-first searches per case: the anchor, the top five, and each family alone.

## 7. Items and commands

| item | CPU, memory | command |
| --- | --- | --- |
| `l2d-selftest` | 1, 2 GB | `python outputs/mp_unified/l2ddiag.py --selftest` |
| `l2d-census-<dataset>-<carve>` (three) | 1, 3 GB | `python outputs/mp_unified/l2ddiag.py census --dataset D --carve C` |
| `l2d-report` | 1, 1 GB | `python outputs/mp_unified/l2ddiag.py report` |

**The self-test** checks:

- random graphs against brute-force BFS, middle counts and AUC;
- PKO-shaped toy carves (LTH's `toy_carve`, with links thinned so every depth occurs) against a by-hand recount of
  every count and sum;
- LTA's check: equal passes, and any difference stops the census;
- the report's INCOMPLETE paths;
- the rule on fixed reads.

**Committed after the report:** `outputs/l2d/report.json` and `report.md`, masked. The per-carve censuses stay on the
host and the laptop.
