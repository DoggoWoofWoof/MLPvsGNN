# P3b: the gap ledger of the bar, the GNNs and the MLPs on our full corpora (a diagnostic)

Declared 13 October 2026, about 09:20, before any number of the stage. P3's second stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), after [P3A_SOTA_RANKERS.md](P3A_SOTA_RANKERS.md) (cdf85d5). It reads
rankings only: no model trains, and nothing is adopted.

## 1. Question

P3a found both GNNs (zsp, gnn0) at or above the best LLM-free SOTA row on all six full corpora. That says where we
stand, not why each ranking misses what it misses. Per dataset and hop slice:
- where does each miss of the bar, each GNN and each MLP go;
- which part of each gap (the bar over a GNN, the bar over an MLP, a GNN over its MLP) is the pool's, a link's, or the
  ranking's?

This is item 4 of the program ("the gaps, three ways"). The buckets choose the next changes (section 6).

## 2. The rankings

| ranking | what | its pool |
| --- | --- | --- |
| the bar | per dataset and measure, P3a's best SOTA row (its filed `best_sota`), read from P3a's records: each gold's position in the order | the order's first \|I_q\| nodes, where \|I_q\| is the models' pool size for the question |
| zrc, zsp | the lean pair (J5 fits), P2a part 2's unmasked read, rerun with its reader to keep each gold's rank and the first ten nodes | today's pool (step 1's s1eval looks) |
| twin0, gnn0 | the universal pair, step 1's s1eval cache, score columns 0 and 1 | the same pool |

- **The order.** The models rank in lean_gpu's order: descending, NaN last, ties by pool position. A bar row ranks in its
  own order.
- **Why the bar is cut at \|I_q\|.** The four models read one pool per question. Cutting a bar row at the same size makes
  "not in the pool" mean the same number of nodes for every ranking.

## 3. The ledger

Per ranking, each gold of each question gets one code:

| code | meaning |
| --- | --- |
| FOUND | in the first five |
| POOL | outside the ranking's pool (a model: not a pool row; a bar row: not among its first \|I_q\| nodes) |
| BRIDGE | in the pool, and one link from a gold the ranking has in its first five (any family, either direction) |
| RANKED | the rest: in the pool, ranked out, no link to a found gold. Split by rank: 6-10, 11-50, 51+ |

RANKED is the program's "beyond the features" bucket, by elimination: this stage does not test the features.

- **Links.** M3B's three stores per dataset (structural, ner, knn), read as filed. Two golds are one link apart when any
  family's store holds the edge, in either direction.
- **R@5, per question.** FOUND is the R@5. Each missed gold's share (1/gt) goes to its bucket. With more than five golds,
  only min(gt, 5) − found of the misses could have been in the first five: that much is shared equally among the missed
  golds, and the rest is CAPACITY. Each row sums to 1.
- **FC@5, per question,** the first that fits:
  - FOUND (every gold in the first five);
  - CAPACITY (more than five golds);
  - POOL (a missed gold outside the pool);
  - RANKED (a missed gold ranked out, with no link to a found gold);
  - BRIDGE (every missed gold is one link from a found gold).
- **hit@1, per question:** HIT, TOP_RANK (a gold in the first five, none first: the program's "top rank only"), or
  NO_TOP5.
- **Also per ranking:**
  - for its BRIDGE golds, the share whose link to a found gold is on each family (S, N, K; a gold can count on more
    than one);
  - the bar's found golds outside our pools (a share of R@5): the part of the bar's lead that no ranking of our pool can
    recover. It is the same against all four models.

## 4. The gaps

For each family (lean: zrc, zsp; universal: twin0, gnn0) there are three pairs (A over B): bar>GNN, bar>MLP and GNN>MLP.
For the bar, A is the bar row of that measure. Each pair is read in both directions:

| measure | lost (A has it, B does not), by B's bucket | won back: the reverse, by A's bucket |
| --- | --- | --- |
| R@5 | per question, the golds in A's first five and not in B's, each 1/gt | the same with A and B swapped |
| FC@5 | per question, 1 where A has every gold in its first five and B does not | as left |
| hit@1 | per question, 1 where A's first is a gold and B's is not, by B's TOP_RANK or NO_TOP5 | as left |

Per question, R@5's lost minus won back equals A's R@5 minus B's (checked). GNN>MLP has no POOL part, since both read
one pool (checked).

## 5. Slices

`all`, then step 1's strata, as the population carries them:

| dataset | strata |
| --- | --- |
| metaqa | hop1, hop2, hop3 |
| squad | answerable=True, answerable=False |
| musique | hop2, hop3, hop4 |
| hotpotqa | hard/bridge, hard/comparison |
| 2wiki | bridge_comparison, comparison, compositional, inference |
| webqsp | one stratum (all) |

## 6. The reading, fixed before any number

For bar>GNN and GNN>MLP, per family, dataset and slice, the reading uses the lost parts of the R@5 gap (POOL, BRIDGE,
RANKED):
- **NONE:** the total lost is within the tie (0.002).
- **LEAD \<bucket\>:** the largest bucket is above the second by more than the tie, and the paired bootstrap interval of
  their per-question difference is above 0 (2,000 resamples, seed 0, 95%).
- **SPLIT \<a\>+\<b\>:** otherwise.

**Where each lead goes.** No stage is declared here; each gets its own file.

| lead | where it goes |
| --- | --- |
| POOL | the pool stages (P1c and after): golds no ranking of our pool can reach |
| BRIDGE in bar>GNN | a second step from the found golds |
| BRIDGE in GNN>MLP | what message passing buys. P5's offsets are the MP-free route to it: reach a gold one link from a found one without walking (a learned g(query, seed, graph features), the user's 13 October direction) |
| RANKED | P5's representation: today's features do not place the gold |
| TOP_RANK (hit@1) | the head of the ranking. It is reported, not called |

## 7. Beside the bar on metaqa and webqsp: the seeds removed

P3a's results (section 7) promised this read. On the KB datasets the walk rows put the seeds first, and the seeds are
rarely the answers.

**What is removed.**
- From every ranking (the eight SOTA rows and the four models): each question's seeds (step 4e's: the dense and SPLADE
  top five) and exact-match linked nodes, which are the nodes P3a's walks restart from.
- From the gold set: the same nodes. The KB-QA line never answers with the topic entity.

The sets come from step 4e's run_carves on s1eval, as P3a's read took them.

**How it is read.**
- Each ranking is read from its first ten (P3a's records keep ten, and so do the models' reads): the first five of what
  is left.
- A question is read when a gold is left, and every ranking's first ten decide its five. That holds when five or more
  are left, or the ranking ends within its ten.
- The rest are counted, per ranking: questions with no gold left, and questions a ranking's first ten leave undecided.

**The bar and the scores.** The bar is the best of the eight SOTA rows per measure; ties go to the earlier row in P3a's
list. Each model is scored against it as P3a scores:
- +1 or −1 beyond the tie, with the paired interval clear of 0;
- AT_OR_ABOVE when no measure scores −1.

**What it decides.** Nothing in P3a's file changes.
- If a GNN is BELOW this bar on metaqa or webqsp, Claim 2 is cited with that exception named, from this read, and the
  shortfall joins P5's bar.
- If both GNNs are AT_OR_ABOVE, Claim 2 on the KB datasets is cited with both reads.

## 8. The checks, first

The verdict refuses, and nothing is read, unless all of these hold:
- **Each read part:**
  - reproduces P2a part 2's measures exactly on every question (zrc and zsp; R@5, FC@5, hit@1);
  - has measures equal to those of its own ranks;
  - has pools equal to step 1's.
- **The gather:**
  - its questions are step 1's cache's, in order, with the same gold counts;
  - its gold rows are exactly the pool rows that hold a gold;
  - twin0's and gnn0's pools hold the same golds.
- **Against P3a:**
  - P3a's questions and golds equal the gather's;
  - twin0's and gnn0's top-five and top-one counts equal P3a's;
  - each bar row's means, from its positions, equal P3a's filed means;
  - the bar rows are among the eight SOTA rows.
- **The four models' pools** hold the same golds.
- **The ledgers and gaps:**
  - each ledger sums to 1, and its FOUND equals the ranking's R@5;
  - each R@5 gap adds up to the difference;
  - GNN>MLP has no POOL part;
  - the bar's golds outside our pools are the same against all four models.
- **On metaqa and webqsp:** every ranking's first ten agree with its golds' positions and its length.
- **Provenance:** every read and gather ran on one script and on P3a's freeze, and none with `--limit`.

## 9. How it runs

**Code:** `scripts/p3b_gap_ledger.py`. It imports, unchanged:
- P2a part 2's reader (`p2a_rank`);
- P3a's module;
- step 4e's carve runner (the seeds and linked nodes);
- M3B's stores.

**The self-test** covers:
- the order and ranks against lean_gpu.top_hit, on 200 made-up questions with ties, NaN and signed zeros;
- links in either direction, with the family bits;
- the codes, the R@5 ledger (CAPACITY included), the FC@5 and hit@1 codes, on hand-worked questions;
- the gap identity on random rankings, and no POOL part between two models that read one pool;
- the reading's three calls;
- the verdict's checks on made-up records. It refuses golds, lean measures, universal counts, P3a means or pools that
  differ;
- the seeds-removed read on hand-worked questions:
  - a gold lost to the seeds;
  - a question with no gold left;
  - a ranking whose first ten leave it undecided;
  - the bar's tie rule;
  - a BELOW call on 100 questions;
  - refusals of a first ten, or a length, that disagrees;
- the report, written from the made-up ledger and seeds-removed read;
- a read part written and joined back.

**Development runs, no number.** `--limit Q` reads the first Q questions and writes nothing. It logs only that every
step ran and the checks held. Before this file was committed:
- the self-test passed;
- `read --part 0/N --limit 50` ran on musique and metaqa, and reproduced P2a part 2's measures on every question;
- `gather --limit 200` ran on musique and squad, and on metaqa with the seed sets (52 s, 2.9 GB peak on the laptop). Its
  checks held.

**Host jobs** (light lane), 43 items:

| item | what | size |
| --- | --- | --- |
| `p3b-read-lean-<dataset>-<k>of<n>` (37) | P2a part 2's parts: metaqa 8, squad 9, musique 2, hotpotqa 6, 2wiki 10, webqsp 2 | 1 CPU each. Memory from P2a part 2's measured peaks (5.75, 4.25, 3.79, 4.59, 4.58 and 6.92 GB): metaqa 6.5 GB, squad 4.5, musique 4.5, hotpotqa 5, 2wiki 5, webqsp 7.5 |
| `p3b-read-gather-<dataset>` (6) | golds, slices, links and the universal pair's ranks; the seed sets on metaqa and webqsp | 2 CPUs each: 2wiki 6 GB, hotpotqa 5, webqsp 5, metaqa 3.5, musique 2, squad 2 |
| `verdict` | the checks, the ledger, the gaps and the reading | locally, on the fetched records and P3a's |

Records go to `outputs/p3b/`:
- `lean_<dataset>_<k>of<n>.json` and `.npz` (the reads);
- `gather_<dataset>.json` and `.npz`;
- `ledger_<dataset>.npz` (every ranking's gold ranks and codes);
- `verdict.json` and `verdict.md`.

**ETA.**
- P2a part 2's parts took 1 to 9 minutes each to read, after a setup of 17 to 107 seconds.
- The gathers took about a minute each on the laptop.
- So the verdict should land about 20 minutes after the items start, if the host's CPUs are free.

The encoder stays frozen. No model trains, nothing is adopted, and test splits are never read.

## 10. Results (filed 13 October 2026, about 10:05)

`python scripts/p3b_gap_ledger.py verdict` on the 43 fetched records and P3a's (`outputs/p3b/verdict.json`,
`verdict.md` and `ledger_<dataset>.npz`). Every check in section 8 held on all six:
- the 37 read parts reproduce P2a part 2's measures on every question;
- the gathers match step 1's cache, and P3a's questions, golds, counts and means;
- every ledger sums to 1, with FOUND equal to the ranking's R@5;
- every R@5 gap adds up to its difference, and GNN>MLP has no POOL part;
- on metaqa and webqsp, every ranking's first ten agree with its golds' positions;
- one script, P3a's freeze, and no `--limit`.

**The readings** (section 6, R@5's lost parts, slice `all`; the total lost in brackets):

| dataset (the R@5 bar) | lean bar>GNN | lean GNN>MLP | universal bar>GNN | universal GNN>MLP |
| --- | --- | --- | --- | --- |
| metaqa (LINKED) | LEAD RANKED (0.005) | LEAD RANKED (0.020) | LEAD RANKED (0.004) | LEAD RANKED (0.096) |
| squad (A1) | LEAD RANKED (0.014) | LEAD RANKED (0.006) | LEAD RANKED (0.020) | LEAD RANKED (0.020) |
| musique (RRF) | LEAD RANKED (0.026) | SPLIT RANKED+BRIDGE (0.023) | LEAD RANKED (0.052) | SPLIT BRIDGE+RANKED (0.072) |
| hotpotqa (RRF) | SPLIT RANKED+BRIDGE (0.010) | LEAD BRIDGE (0.011) | LEAD RANKED (0.009) | LEAD BRIDGE (0.041) |
| 2wiki (RRF) | LEAD RANKED (0.016) | LEAD BRIDGE (0.019) | LEAD RANKED (0.012) | LEAD BRIDGE (0.061) |
| webqsp (DENSE) | LEAD RANKED (0.063) | LEAD RANKED (0.039) | LEAD RANKED (0.007) | LEAD RANKED (0.071) |

- **The bar's leads are small.** What the bar finds and a GNN misses is 0.004 to 0.063 of R@5. Over the same questions,
  the GNN finds 0.02 to 0.55 that the bar misses. Squad is the exception: for gnn0, lost (0.0198) is more than won back
  (0.0183), the 0.0015 within the tie that P3a filed.
- **What the bar wins, the GNNs had in their pools.** The bar's golds outside our pools are 0.0006 of R@5 on metaqa
  and 0 on the other five, so no part of any bar lead is out of our pools' reach. The GNNs ranked those golds out.
- **The lean pair on metaqa** trade golds evenly: lost 0.0202, won back 0.0204. The reading only says where the
  difference lives (ranking, not links).
- **The lean pair on webqsp** (zero-shot): zrc is above its GNN (R@5 0.307 against 0.268; GNN>MLP lost 0.039, won
  back 0.079).

**Slices** (where they differ from `all`):
- **metaqa:** bar>GNN is NONE on hop2 in both families (lost 0.0008 and 0.0009). Lean's hop3 is SPLIT RANKED+POOL.
- **musique:** GNN>MLP leads BRIDGE on hop2 in the universal family (lean: SPLIT BRIDGE+RANKED), and RANKED on hop3
  and hop4 in both.
- **hotpotqa:** GNN>MLP leads BRIDGE on hard/bridge and RANKED on hard/comparison, in both families.
- **2wiki:** GNN>MLP leads BRIDGE on bridge_comparison, compositional and inference, and RANKED on comparison, in both
  families. Universal bar>GNN is SPLIT RANKED+BRIDGE on inference.

So the BRIDGE leads sit where a question needs a second passage reached through a link. The comparison slices are
RANKED.

**Where each ranking misses** (R@5 shares, slice `all`; the range over zrc, zsp, twin0 and gnn0):

| dataset | POOL | BRIDGE | RANKED (of which 51+) | CAPACITY |
| --- | --- | --- | --- | --- |
| metaqa | 0.020-0.021 (hop3 0.047-0.049) | 0.001-0.004 | 0.025-0.084 (0.001-0.004) | 0.170 |
| squad | 0.020 | 0 | 0.069-0.083 (0.000) | 0 |
| musique | 0.079 (hop4 0.198) | 0.107-0.135 | 0.233-0.276 (0.091-0.126) | 0 |
| hotpotqa | 0.020 | 0.045-0.072 | 0.028-0.030 (0.002) | 0 |
| 2wiki | 0.036 (bridge_comparison 0.066) | 0.044-0.083 | 0.029-0.037 (0.001) | 0 |
| webqsp | 0.126-0.135 | 0.010-0.020 | universal 0.122-0.145, lean 0.425-0.466 (0.023-0.154) | 0.114 |

- **Which family carries a BRIDGE gold's link** (zsp and gnn0; a gold can count on more than one):
  - hotpotqa: structural 1.00, ner 0.24-0.25;
  - 2wiki: structural 0.92-0.93, ner 0.40-0.43;
  - musique: ner 0.68-0.70, structural 0.44-0.46;
  - metaqa: knn 0.90-0.92;
  - webqsp: structural 0.53-0.62, knn 0.48-0.59.
- **hit@1's top rank only** (a gold in the first five, none first; reported, not called):
  - webqsp: lean 0.27-0.35, universal 0.18-0.23;
  - musique: 0.20-0.22;
  - squad: 0.16;
  - hotpotqa: 0.10-0.13;
  - 2wiki: 0.07-0.12;
  - metaqa: 0.07 (twin0 0.17).

**Beside the bar: the seeds removed (section 7).** zrc, zsp, twin0 and gnn0 are all AT_OR_ABOVE, on metaqa and on
webqsp. The read is thin:
- It decides 17 of 9,785 metaqa questions and 12 of 1,503 webqsp questions.
- The walk rows (A1, A2, A1_all, A2_all) spend their first ten on the seeds and linked nodes. The read removes 9.08
  nodes per metaqa question and 10.84 per webqsp question, so these rows leave their five undecided on 9,714 and 1,353
  questions.
- 51 and 136 questions have no gold left.

On the questions it reads (R@5, FC@5, hit@1):

| dataset | the bar | zsp | gnn0 |
| --- | --- | --- | --- |
| metaqa | LINKED 0.510, 0.471, 0.176 | 0.861, 0.824, 0.765 | 0.861, 0.824, 0.882 |
| webqsp | A1 0.333, 0.333, 0.250 | 0.583, 0.500, 0.167 | 0.847, 0.833, 0.750 |

**What it decides.**
- **Claim 2 on metaqa and webqsp.** Both GNNs are AT_OR_ABOVE, so the claim is cited with both reads (section 7), and
  the seeds-removed read is cited with its size. On 17 and 12 questions it finds nothing against the claim; it cannot
  show more than that. A fuller read needs P3a's walk rows kept past ten nodes. That would be its own file.
- Nothing in P3a's file changes.

**Where the leads go** (section 6's table):
- **RANKED → P5's representation.** RANKED leads every bar>GNN gap but one (lean's hotpotqa is SPLIT). It also leads
  GNN>MLP on metaqa, squad and webqsp, and on musique's 3- and 4-hop questions. The bar's wins over the GNNs are golds
  the GNN had in its pool and ranked out, with no link to anything it found.
- **BRIDGE in GNN>MLP → P5's offsets.** BRIDGE leads GNN>MLP on hotpotqa and 2wiki in both families, and on musique's
  2-hop questions in the universal family. What message passing buys there is the gold one link from a found gold. P5's
  offsets are the MP-free route to it. P5a's fixed map STOPS ([P5A_NODE_SPACE.md](P5A_NODE_SPACE.md) section 8), so
  the route is the learned offset g(question, seed, graph features), declared in its own file.
- **POOL leads no gap.** Each ranking's own POOL share stays the pool stages' target (P1c, held for host disk):
  - musique's hop4: 0.198;
  - webqsp: 0.13;
  - 2wiki's bridge_comparison: 0.066;
  - metaqa's hop3: 0.048.
- **BRIDGE in bar>GNN leads nowhere.** It is only ever half of a SPLIT: lean's hotpotqa, hotpotqa's hard/bridge in
  both families and 2wiki's inference in the universal family. A second step from the found golds is not a stage of its
  own.
- **The zero-shot lean pair on webqsp has the largest RANKED share in the ledger:** 0.43 to 0.47 of R@5, against 0.12
  to 0.15 for the universal pair trained on it. That is the representation gap P5 is for, and zero-shot stays graded on
  it.

Nothing is adopted.
