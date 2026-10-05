# Where we stand, 5 October 2026

This page explains what has been compared so far, what the numbers say, and where the project stands on generalization.
It is the place to start. Each phase's own document stays the record, and every number here is copied from one.

Every number after 2 September is a development number: chosen on development splits, often after many arms. The
tracks in sections 3 and 4 are how we build the best universal MLP and the best universal GNN. The paper's numbers
come from one declared confirmation run of the chosen MLP and GNN on held-out data.

## The question

The aim is to build the best universal MLP and the best universal GNN on the same features, make both generalize to
graphs they were not trained on, and then measure what message passing adds.

Every model here reranks the same candidate pool for a question, built from the same features. They differ in one
thing: whether the model can look at a candidate's neighbours in the graph (message passing) before it scores it.

There are three questions, and each is harder than the one before:

1. **In-domain.** Trained and tested on the same dataset, does message passing help?
2. **Universal.** Can one model, trained on all six datasets at once, keep that help on each of them?
3. **Zero-shot.** Does the help carry to a graph the model never saw in training, with relations it has never seen?

## The models

| Name | What it is | Message passing |
| --- | --- | --- |
| `rrf` | Fixed reciprocal-rank fusion of the retrievers. No training. | no |
| QLS-U (`qls_u_sota_v1`) | Our trained MLP scorer over the same features. Its publication name is QLS-MLP. | no |
| GAT-NO-MP (`gat_no_mp_v1`) | The universal GAT with its message edges switched off. Everything else is the same, so it is the causal control. | no |
| GAT (`gat_universal_v1`) | The universal GAT, the M3B model. | yes |
| Universal GNN (`u_gnn_v2_ef`) | The stage 2 GNN: one model trained jointly on all six datasets. | yes |
| Universal MLP (`u_mlp_v2_mix`) | The GNN's twin: the same features and training, with no message passing. | no |

δ_MP = GAT − GAT-NO-MP is the effect of message passing alone.

Three metrics are used:

- **recall@5 (R@5):** the share of gold items in the top 5.
- **full_coverage@5 (FC@5):** the share of questions with every gold item in the top 5. This is the multi-hop metric.
- **hit@1:** whether the first item is gold.

## 1. In-domain: message passing helps (M3B, closed 19 Sep, 0bbb3c4)

Seed 0, with the paired 95% interval in brackets. The band says whether the dataset's δ_MP can be read at all. On
metaqa and webqsp the GAT sits below the published systems (NuTrea, ReaRev) by more than our measured exposure
difference, so their δ_MP is reported but not read. squad is a control whose graph exposes no extra gold.

| dataset | `rrf` | QLS-U | GAT-NO-MP | GAT | δ_MP (R@5) | band |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | 0.005 | 0.612 | 0.617 | **0.763** | +0.147 [+0.144, +0.150] | not read |
| squad | 0.905 | 0.889 | 0.896 | 0.885 | −0.011 [−0.015, −0.008] | control |
| musique | 0.473 | 0.491 | 0.503 | **0.521** | +0.018 [+0.008, +0.027] | **read** |
| hotpotqa | 0.685 | 0.870 | 0.866 | **0.896** | +0.030 [+0.026, +0.035] | **read** |
| 2wiki | 0.608 | 0.838 | 0.834 | **0.885** | +0.052 [+0.048, +0.055] | **read** |
| webqsp | 0.054 | 0.577 | 0.574 | **0.602** | +0.028 [+0.014, +0.044] | not read |

Where the help comes from:

| dataset | δ_MP on FC@5 | δ_MP on hit@1 |
| --- | --- | --- |
| musique | +0.049 [+0.036, +0.063] | +0.010 [−0.007, +0.026] |
| hotpotqa | +0.068 [+0.060, +0.077] | −0.010 [−0.018, −0.003] |
| 2wiki | +0.126 [+0.118, +0.133] | −0.021 [−0.026, −0.015] |

**musique is the weak dataset** ([MUSIQUE_DIAGNOSIS.md](MUSIQUE_DIAGNOSIS.md)).

- **Only its two-hop half gains.** There message passing adds +0.046 R@5 and +0.101 FC@5. On three- and four-hop
  questions it adds nothing, and almost none of those get every passage into the top 5.
- **Its training carve rewards a shortcut.** 94.5% of the select carve shares a gold passage with training questions;
  dev shares none.
- **The published number is a different metric.** GraphER's PR@5 (21.6 to 25.6) is set coverage, our FC@5 (0.224),
  not our R@5. On recall@5, HippoRAG 2 reports 74.7, on a corpus a tenth the size of ours and with a 7B
  encoder.

**Reading.** On the three readable datasets, message passing helps R@5 by 0.018 to 0.052. The help is in
full_coverage@5, which means getting every supporting passage of a multi-hop question into the top 5. It is not in
hit@1, where the GAT is level or slightly below its no-MP control. QLS-U and GAT-NO-MP are close to each other
everywhere: without message passing, the GAT architecture adds little over the MLP.

## 2. Universal: one GNN for all six datasets works; its MLP twin does not pass

**The universal GNN (stage 2, 28 Sep, c2cfda1)** against the per-dataset M3B GAT. Three seeds, mean ± sd, with the
paired 95% interval of the difference.

| dataset | metric | universal GNN | M3B GAT | difference |
| --- | --- | --- | --- | --- |
| metaqa | hit@1 | **0.909** ± 0.013 | 0.841 ± 0.009 | +0.068 [+0.065, +0.070] |
| metaqa | R@5 | **0.786** ± 0.002 | 0.765 ± 0.001 | +0.021 [+0.020, +0.022] |
| 2wiki | R@5 | **0.889** ± 0.003 | 0.884 ± 0.004 | +0.005 [+0.003, +0.007] |
| 2wiki | FC@5 | **0.761** ± 0.004 | 0.750 ± 0.007 | +0.011 [+0.008, +0.014] |
| squad (control) | R@5 | 0.891 ± 0.010 | 0.892 ± 0.006 | −0.001 [−0.004, +0.001] |
| hotpotqa | R@5 | 0.897 ± 0.002 | 0.898 ± 0.004 | −0.001 [−0.003, +0.002] |
| musique | R@5 | 0.524 ± 0.010 | 0.529 ± 0.007 | −0.005 [−0.010, +0.000] |
| webqsp | hit@1 | **0.630** ± 0.005 | 0.593 ± 0.010 | +0.037 [+0.023, +0.051] |
| webqsp | R@5 | **0.619** ± 0.009 | 0.607 ± 0.005 | +0.012 [+0.003, +0.021] |

One jointly trained GNN matches the six per-dataset GATs on the passage datasets and beats them on the two KBs, mostly
in hit@1.

**The universal GNN against its MLP twin** (the replication, 3 seeds; `UNIVERSAL_V2_REPLICATION.md`):

| dataset | model | R@5 | hit@1 | FC@5 |
| --- | --- | --- | --- | --- |
| metaqa | universal GNN | **0.782** | **0.910** | **0.660** |
| metaqa | universal MLP | 0.716 | 0.775 | 0.588 |
| 2wiki | universal GNN | **0.900** | 0.891 | **0.776** |
| 2wiki | universal MLP | 0.852 | 0.898 | 0.672 |
| squad (control) | universal GNN | 0.902 | 0.734 | 0.902 |
| squad (control) | universal MLP | 0.906 | 0.743 | 0.906 |

The GNN family passes its replication gate. The MLP twin fails its gate on 2wiki R@5 (0.852 against a threshold of
0.858), so the universal MLP stands as `REPLICATION_FAIL`. Two repairs of the MLP, UMLP-v2.1 and v2.2A, stopped at
stage A on 29 Sep.

## 3. Can an MLP recover the GNN's in-domain gain? Much of it

These tracks ask how much of the GNN's gain over its MLP twin an MLP, or a cheaper form, can recover in-domain. A
share of 1 would match the GNN.

| attempt | dataset | share of the GNN's gain recovered | record |
| --- | --- | --- | --- |
| MP-Approx, the best arm without message passing (L15, FZ-TW-1x-b1d) | metaqa | 0.929 (0.983 with 4× the labels); still BELOW the GNN on R@5 and FC@5 | `MP_APPROX_L15.md`, 2c246c1 |
| MP-Approx, anchor-phrase walks | 2wiki | about 0.70 (0.75 as an ensemble) | `MP_APPROX_L*.md` |
| Deploy CK, ck_qi | 2wiki | 0.093 | `DEPLOY_CK_2WIKI.md`, d2e51c3 |
| Deploy CK, ck_full (a compressed, query-conditioned one-hop message-passing form) | 2wiki | 0.895 [0.851, 0.944], at 0.656 of the GNN's end-to-end p50 latency | `DEPLOY_CK_FULL_2WIKI.md`, 0c4f370 |
| Lean MLP with the AW edge-label block | 2wiki | 0.865; its R@5 is 0.9964 of the GNN's | `outputs/mp_unified/lean/` |

So in-domain, an MLP recovers much of the GNN's gain, but the best arm without message passing stays below the GNN.
The forms that come closest still use the neighbours in some way: ck_full through one hop of message passing, and the
AW block through fixed edge labels on one- and two-hop walks.

## 4. Zero-shot: this is the open problem

On a graph it was not trained on, every model, MLP or GNN, loses most of its gain. Two kinds of transfer have been
tested.

**Passage graph to passage graph** (for example, train on 2wiki, read hotpotqa). R@5 points (× 100):

| result | number |
| --- | --- |
| Lean MLP trained on 2wiki, read on hotpotqa | −6.6 to −9.3 against the twin |
| QD-GNN trained on one graph | hurts the unseen graphs; a 0.2 tanh bound with edge dropout keeps them within noise |
| QD-GNN with edge-family dropout (`-fd25`) | the first one-graph message-passing arm with no unseen graph below the twin on R@5 |
| Joint four-graph training (W4-T0) | +0.51 (2wiki) and +0.57 (hotpotqa) over the twin, about 0 on metaqa, musique and webqsp |
| The same with hotpotqa held out | keeps about 70% of the gain on hotpotqa (+0.40), never below the twin |

**KB to KB with unseen relations** (train on metaqa, read webqsp; S1 to S6). The baseline is an untyped walk that
ignores relation names. R@5 points:

| result | number |
| --- | --- |
| A model trained on webqsp itself (the ceiling we compare against) | about 60 R@5 |
| The lean MLP zero-shot on webqsp | 20 to 25 R@5 (`rrf` alone: 7.7) |
| A relation the model never saw falls to or below the twin. There is no back-off to "unknown relation". | held-relation tests |
| Part 3: EM chain scorer with granularity transforms | +6.43 over the untyped walk; the transforms carry it (+10.7), at −4.8 in-domain |
| Part 4: the GNN as the scorer, with EM-typed attention | +3.5 over the part 3 MLP scorer on the same read |
| Part 5: typed passages added to training | −3.3 on the KB zero-shot read (hurts) |
| Part 6: per-population map pq | +3.9 [+2.8, +5.1] |
| Part 8: more training distributions | nothing carries |
| Part 14, first interim read (5 Oct) | one extra EM round at read time: +0.2 to +0.3 on metaqa, level on webqsp; re-estimating the level mix on the new graph: −3 to −5 (hurts); the MLP's static neighbour summaries help at one step (+4.4) and hurt beyond it (−3.3 at 3 steps) |

**Reading.** Zero-shot transfer works only partly. Joint training on several graphs keeps most of the gain on a
held-out passage graph. For unseen KB relations, the things that carry are the granularity transforms, EM typing,
and the GNN scorer with EM-typed attention. Nothing yet closes the gap to a model trained on the target graph.

## How we got here

- **3 to 8 September: the regime map and the MLP selection (M0 to M2).** We first measured, with no training, which
  datasets leave room above fixed fusion. Then we froze the MLP scorer (QLS-v2) and selected its smallest semantic
  form, S3 with 3,585 parameters. Two attempts to add structure to it (S4, in M2C and M2D) did not hold up, so S3 stays.
- **8 to 19 September: the controlled comparison (M3A, M3B).** We trained the universal GAT, its no-MP control and
  QLS-U on one substrate. Section 1 of this page is that result: message passing helps multi-hop coverage on the three
  readable passage datasets.
- **19 to 28 September: one model for everything (Universal-v2).** The pilot failed on both families. The replication
  passed for the GNN only. The stage 2 joint GNN then matched or beat the per-dataset GATs (section 2).
- **28 September: compute.** We moved runs to the lab GPU host and checked that its CPU gives the same numbers as the
  laptop.
- **29 September to 1 October: can an MLP get there without message passing? (MP-Approx, Deploy CK).** Sixteen levels
  of approximation, each declared before its run. The best approximation without message passing stays below the
  GNN. A compressed one-hop form (ck_full) keeps most of the gain and is faster (section 3).
- **1 to 3 October: the exploratory MLP and GNN tracks.** The lean MLP comes close to the GNN in-domain but transfers
  poorly. QD-GNN and joint training found the first arms that do not hurt unseen passage graphs (section 4).
- **3 to 5 October: transfer to unseen KB relations without labels (S1 to S6).** We tried label-free rules, EM typing
  of relations, and MLP and GNN scorers over the same features. Section 4 has the parts that carry.
- **5 to 7 October, running on the host: where the zero-shot gap comes from (S6 parts 12 to 15).**
  - Part 12 measures the gap against models trained on the target.
  - Part 13 trains the two models on the same features.
  - Part 14 tests EM rounds as the balancer, and message passing as going back to the neighbours.
  - Part 15 attributes the gap to feature groups with Shapley values.
  - The full grades land about 7 October.

## What is next

1. **Read parts 12 to 15** and act on the largest part of the zero-shot gap. Implement each fix for each model class.
2. **Finish M2 to M9 on the current datasets** before the canonical CRAG substrate, one declared file per phase.
3. **Write the paper.** The core claim available today is section 1: message passing helps multi-hop coverage, with a
   causal control. Its limits are sections 2 to 4: universal training works for the GNN, and zero-shot transfer is
   only partial.
