# HLW: what the corpus hyperlinks add to HippoRAG 2's walk, on HippoRAG 2's own settings

Declared 3 November 2026 (about 23:50 IST), before any HLW number. Module: `outputs/mp_unified/hlwdiag.py`. It imports
`scripts/b1c_walks.py` and does not edit it. Records: `outputs/hlw/report.json` and `report.md`. A diagnostic: nothing
trains, nothing is fit, and no model, pool, input or rule changes, whatever it finds.

## 1. Question

Our trained models gain a lot from the corpus hyperlinks. U1a read them on B1 (HippoRAG 2's 1,000 questions and
corpora). With the hyperlinks removed (b1t against b1), R@5 fell by 19.8 points [18.3, 21.2] for the GNN zsp/J5 and
20.8 for the MLP zrc/J5 on 2wiki, and by 8.3 and 9.5 points on hotpotqa (`outputs/u1a/report.json`,
settings.<ds>.paired).

The question: **does the walk at the heart of HippoRAG 2, personalized PageRank from HippoRAG 2's restart, gain anything
from the same hyperlinks?** Two filed reads point to no:
- B1C: on B1, no training-free walk moved R@5 off the first stage, with or without the hyperlinks. The walks there used
  B1C's two restarts, not HippoRAG 2's.
- P3a: on the full corpora, HippoRAG 2's restart rule walked on the hyperlinks ranked below its own first stage.

HLW measures it with HippoRAG 2's restart rule, with and without the hyperlinks, on the same questions, as paired
differences.

## 2. What runs (part A, B1)

**Settings.** B1's three: `outputs/bench/hipporag2/<setting>/`, all 1,000 questions each. On 2wiki and hotpotqa the
structural family is the corpus's own Wikipedia hyperlinks. On musique it is title mentions, so musique is reported
beside and has no call.

**Graphs.** Four, built from B1a's families with B1C's `family_matrix`: undirected, structural weight 1, ner and knn
their filed weights, rows normalized, a node with no edge keeping its mass:

| graph | families |
| --- | --- |
| none | no edges (the walk returns its restart) |
| S | structural (the hyperlinks) |
| NK | ner + knn (text links, no hyperlinks) |
| SNK | all three |

**Restarts.** Four:

| restart | mass |
| --- | --- |
| **A1** (decides) | HippoRAG 2's rule as step 4e and P3a define it: half the mass uniform on the seeds, half on the RRF weights of the dense and SPLADE top-1,000 lists (1/(60 + rank), summed over the two lists) |
| A2 | A1 with HippoRAG's node specificity, 1/log(e + degree) in the walked graph |
| T5 | B1C's: uniform on the RRF top 5 |
| RW | B1C's: the RRF top 1,000 in proportion to 1/(60 + rank) |

- The seeds are m3b's frozen rule: the dense top 5, then the SPLADE top 5, first occurrence kept.
- The lists are `first_stage.npz`'s dense, SPLADE and RRF arrays, the ones B1C read.
- The walk is B1C's: restart probability 0.5 and 50 power iterations.
- Passages are ranked by walk mass, with ties broken by RRF rank and then by node index.
- R0 is the RRF list itself.

**Measures.** On all 1,000 questions: R@2, R@5 and R@10 (B1C's), and FC@5 (every gold in the top 5).

**Differences**, per restart and setting, are paired by question. Each has a 2,000-resample bootstrap 95% interval
(B1C's `boot_ci`, seed 0):
- **ADD** = X/SNK − X/NK, what the hyperlinks add to the text links;
- **ALONE** = X/S − X/none, what the hyperlinks add to no links;
- X/SNK − R0 and X/S − R0, against the first stage.

**Checks before any report** (the run stops if one fails):
- R0's R@5 equals B1C's filed R0 to four decimals.
- These arms equal B1C's filed arms at R@2, R@5 and R@10, to four decimals:

  | HLW arm | B1C arm |
  | --- | --- |
  | T5/S | T5/structural |
  | T5/SNK | T5/all |
  | RW/S | RW/structural |
  | RW/SNK | RW/all |

- Every question has at least one gold, and every gold is a node of the setting.

## 3. Part B, the full corpora (filed reads only)

From P3a's filed orders (`outputs/p3a/orders_<dataset>.npz`) on every s1eval question of 2wiki and hotpotqa:
- R@5 of RRF, A1 (on the hyperlinks: P3a's regime graph on these two datasets is the structural family alone), A1_all
  (hyperlinks + ner + knn), A2 and A2_all;
- paired differences against RRF, and A1_all − A1, with the same bootstrap.

P3a walked no graph without the hyperlinks at full scale, so part B has no ADD and no ALONE. That is stated, not run.

## 4. The call (fixed now)

On each hyperlinked setting (2wiki, hotpotqa), restart A1 decides. MARGIN is 0.010, one point of R@5.

| call | when |
| --- | --- |
| NO_LIFT | ADD's and ALONE's intervals both end below +MARGIN |
| LIFT | ADD or ALONE has a lower end above 0 and a point value of at least +MARGIN |
| SMALL | otherwise |

**Overall:**

| overall call | when |
| --- | --- |
| PPR_NO_HYPERLINK_LIFT | NO_LIFT on both settings |
| PPR_HYPERLINK_LIFT | LIFT on either setting |
| MIXED | otherwise |

A2, T5 and RW are reported beside under the same rule and decide nothing.

**What each call allows a written claim to say:**
- **PPR_NO_HYPERLINK_LIFT.** "Personalized PageRank from HippoRAG 2's restart gains less than one point of R@5 from the
  corpus hyperlinks on HippoRAG 2's own 2Wiki and HotpotQA settings, while the same hyperlinks add N points to our
  trained models."
- **Any other call.** The measured ADD and ALONE with their intervals, beside the models' lift, and never "no lift".
- **In every case**, it is HippoRAG 2's walk on our LLM-free graph, not HippoRAG 2 as released with its LLM-built graph.
  The sentence says so.

**Beside, deciding nothing (filed records, read as they stand):**
- U1a's b1 − b1t (the J5 fits);
- B1P's b1w − b1t for the models of record (`outputs/b1p/report.json`, settings.<ds>.paired_w_minus_t).

## 5. Running it

```
python outputs/mp_unified/hlwdiag.py selftest
python outputs/mp_unified/hlwdiag.py run        (laptop CPU, about 10-20 minutes)
```

The self-test runs on made-up data:
- A1's restart equals step 4e's `restart("A1", ...)`, and A2's equals `restart("A2", ...)`;
- every restart sums to 1;
- the none graph's walk returns its restart;
- a tie in walk mass goes to the better RRF rank;
- the call table gives each of its three outcomes on inputs built for it.

## 6. Results

(Filed here after the run; nothing above changes.)
