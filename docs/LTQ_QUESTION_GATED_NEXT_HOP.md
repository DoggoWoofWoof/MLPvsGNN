# LTQ: a question-gated next-hop scorer with relations and IRA's refinement, each gold in its own softmax

Declared 3 Nov 2026 (~22:50 IST), before any LTQ number. Module: `outputs/mp_unified/ltqdiag.py`. LTQ writes only under
`outputs/ltq`. It is a diagnostic: no MLP or GNN is fitted under it, and what follows it is fixed in section 7.

## 1. Why

The user, 3 Nov: "cant we train it to know or understand the vector itself and since we already know the golds train it
to work towards that, is there no learning there", and then: "not just question and hops, add in relation to that,
improve the vectors to actually find the golds using IRA".

LTA's column records (every row outside rrf's top five; split WQ-hotpotqa, where musique trains) give the share of A
targets in each score's top two (T2), for the targets with no link to the anchor:

| unlinked targets, T2 | musique/s1fit (out of fold) | musique/s1eval | 2wiki/fit (out of fold) |
|---|---:|---:|---:|
| rrf | 0.079 | 0.070 | 0.096 |
| LTA's bil, trained on the golds | 0.096 | 0.084 | 0.191 |
| CHN's pair, fixed | 0.144 | 0.113 | 0.155 |
| anc (cosine to the anchor), fixed | 0.000 | 0.000 | 0.000 |

So the frozen vectors hold more about musique's unlinked next hop than LTA's trained scorer extracts. LTH (the same
scorer, negatives drawn with weight 1/rrf rank) lowered it to 0.044 and 0.055. What the build of LTW's scorer explains:

- LTW's loss is one softmax over a list with the mass of every target together (`list_loss`). It is met once any target
  ranks first, so where a linked target is in the list, an unlinked one gets almost no gradient.
- One scorer serves two kinds of next hop: through the found passage (its links and its neighbourhood; linked targets
  reach T2 0.33-0.36) and toward what the question asks beyond it (unlinked targets: far from the anchor, near the
  question's remainder; res and pair rank them best).
- Under that loss, LTH's rrf-ranked negatives taught it that rows near the top of rrf are negatives, and the unlinked
  targets are such rows.

NMD (53210eb) found that among the unlinked candidates rrf already orders the kind of passage well (AUC 0.884 on
musique) but not which one. LTQ tests whether a scorer trained per gold, with the question's relations, finds the one.

## 2. The scorer

For a candidate row v of an A list (question q, anchor row a, unit residual r of q against x_a; ltwdiag's inputs):

- **A** is LTW's bil unchanged: its MLP over all 14 features, plus x_v . (U1 V1^T q + U2 V2^T x_a + U3 V3^T r).
- **B** reads what the question asks beyond the found passage. It is an MLP of LTW's shape over the 10 features that are
  not the row's raw cosine to the anchor or its three link bits (cos, res, pair, anchor_cos, the five rank columns,
  log1p rank), plus x_v . p, the row's match to the next hop's vector p. B cannot see the row's links to the found
  passage, nor x_a except through res, pair, r and p. p is built from the question's hop slots and learned relations:
  - **Hop slots.** c = V_q^T q + V_a^T x_a + V_r^T r (rank 32). Each of 4 slots (one per hop of musique's longest
    questions) starts at c plus its own learned start e_s.
  - **Relations.** 16 learned relations, each a key k_j (rank 32) and an offset o_j in the rows' space (zero at the
    start): "the main person first, then a marriage offset".
  - **IRA's plan** (zira.plan's form, re-implemented here; zira is not imported): Sinkhorn in log space, with a dustbin,
    of the 16 relations over the 4 slots, on the logits b_z h_s . k_j / sqrt(32) and the dustbin's b_0 (b_z = 1 and
    b_0 = 0 at the start, IRA's ira_bz and ira_b0). Each relation's mass over the slots and the dustbin is 1; each
    slot's mass over the relations is 1; the dustbin takes the rest. 20 iterations, each a column step and then a row
    step.
  - **IRA's rounds.** Two rounds (RTR's two): h_s <- h_s + step_t sum_j P_js k_j, step zero at the start. The plan is
    then taken again on the refined slots.
  - **The next hop.** A pointer w = softmax_s(pi . h_s + b_s) (uniform at the start) picks the next hop's slot. Its
    relation mass is m = sum_s w_s P_.s, and p = U^T (sum_s w_s h_s) + sum_j m_j o_j (U zero at the start).
  - So p is zero at the start and B starts as its MLP alone; training moves p toward the golds.
  - The plan runs over the question's relations, not over the list's rows. A fit's lists are 256 sampled rows and a
    read's are every row (about 5,000), so a plan over rows would see different lists in the two.
- **The gate** is gamma = w . [q, x_a, q * x_a] + b, zero at the start.
- **The score** is log(sigmoid(gamma) e^A + sigmoid(-gamma) e^B): a row ranks high when either expert, weighted by the
  question's gate, ranks it high. That way a question with a linked and an unlinked missing gold can have both on top.

The loss gives each target its own softmax against its list's negatives: softplus(logsumexp over the negatives - the
target's score), averaged over a list's targets. Lists are weighted per dataset as ltwdiag weighs them.

## 3. Two forms, the same scorer and loss

| form | A lists | negatives |
|---|---|---|
| u | LTA's (`outputs/lta/work/lists`) | every target, then negatives drawn uniformly, cap 256 |
| h | LTH's (`outputs/lth/work/lists`) | every target, then negatives drawn with weight 1/rrf rank, cap 256 |

Both lists are read only. The A cases, questions, parts and folds in both are LTM's.

## 4. Fits

These are ltmcols' fits unchanged, with LTQ's stage, root, form, scorer and loss patched in for the length of a fit
(every name put back after):

- five fold fits per split (fit j leaves out fold j's questions);
- the early stop on 0.10 of the remaining components, by held-out T2 (macro over the datasets);
- LTW's settings (epochs 60, patience 6, lr 1e-3, weight decay 1e-4, batch 32, seed 0, cap 256, min lists 200).

There are four fits: forms u and h, splits WQ-hotpotqa and WQ-musique. The fit records stamp a hash of the source of the
scorer, the plan and the loss and of their settings (the B features, 16 relations, 4 slots, 2 rounds, 20 iterations); a
read refuses fits stamped otherwise. Lists built with a limit or another cap are refused.

## 5. Reads

For each split and carve read, each question's A case is ranked by LTQ u, LTQ h and the free scores rrf, cos, anc, res
and pair. bil's numbers are LTA's column record's (the same lists and fold fits), not re-scored. The A case is
ltacols' (its first gold in the top five as anchor, every row outside the top five as candidates). Each list is scored
by the fold fit its question draws: Q_fold on a training carve (out of fold), Q_hfold otherwise, as LTA's columns draw.

For each score the read records T2 over all targets, over linked targets (any link bit to the anchor) and over unlinked
ones, plus T10, MRR and the unlinked T2 by hop. Beside, deciding nothing, it records what each form's fit uses: the
gate's mean weight on A (over all lists, lists with an unlinked target, and linked-only lists), each relation's share as
the top relation, the relation entropy, and each slot's share as the pointer's top slot.

**Check:** the free scores' T2, linked T2 and unlinked T2, the A ranked count and the linked share must equal LTA's
column record of the same split and carve, whose stage, split, dataset and carve must match.

| split | reads (decisive first) |
|---|---|
| WQ-hotpotqa | musique/s1fit, musique/s1eval; beside: 2wiki/fit, 2wiki/s1eval, hotpotqa/s1eval |
| WQ-musique | beside: hotpotqa/fit, 2wiki/fit, hotpotqa/s1eval, 2wiki/s1eval, musique/s1eval |

## 6. The rule (fixed before any number)

The decisive reads are WQ-hotpotqa musique/s1fit (out of fold) and musique/s1eval: musique in-domain.

- **The decision form** d is the one of u and h with the higher unlinked T2 on musique/s1fit (a tie goes to u). s1eval
  does not take part in that pick.
- **SIGNAL** iff, on both decisive reads:
  - T2_unlinked(d) >= T2_unlinked(pair) + 0.03, and
  - T2_linked(d) >= T2_linked(bil) - 0.02, bil's from LTA's column record.
- **INCOMPLETE** if a decisive read is missing, fails the check, fails ltwdiag's identity check, or ranks fewer than 200
  A targets.
- **NO_SIGNAL** otherwise, read as:
  - BELOW_PAIR if the unlinked condition fails on either read;
  - TRADE if only the linked condition fails.

The beside reads decide nothing.

## 7. What follows

- **SIGNAL:** LTQ's columns (ltacols' seven columns, from LTQ's decision form, for both splits' 20 carves) and an MLP fit
  on them (zrckpn plus LTQ's columns, as LTA's zrckpa is built), then a GNN form reading them through GNX's machinery
  (as GNA reads LTA's). Each is declared as an amendment here before any fit. The WWW rule (6cb8f39) and each stage's
  own rule judge those fits.
- **NO_SIGNAL:** a next-hop scorer trained on the golds, with relations and IRA's refinement, is not what musique's
  unlinked next hop lacks; it needs a new input. No further next-hop scorer is declared before the freeze; LT2's
  two-link term is in flight, and the names forms are NMD section 5's (filed 53210eb).

## 8. Items

| item | needs | CPU | memory (GB) |
|---|---|---:|---:|
| ltq-selftest | - | 1 | 2 |
| ltq-fit-{u,h}-wqh | ltq-selftest | 2 | 13 |
| ltq-read-wqh-musique-{s1fit,s1eval} | both WQ-hotpotqa fits | 2 | LTA's column item's peak + 1 |
| ltq-report-dec | the two decisive reads | 1 | 2 |
| ltq-fit-{u,h}-wqm | ltq-selftest | 2 | 13 |
| ltq-read-{wqh,wqm}-{ds}-{carve} (beside) | both fits of the split | 2 | LTA's column item's peak + 1 |
| ltq-report | every read | 1 | 2 |

The order is the self-test, the WQ-hotpotqa fits, the two decisive reads and a report, then the WQ-musique fits, the
beside reads and the report again. The fits go through `retry_cmd.py` as LTA's do. What LTQ reads and never writes:

- LTA's lists, fit records, fold fits and column records;
- LTH's lists;
- LTM's lists;
- PKO's cache and links.

Committed: this file, `ltqdiag.py`, and `outputs/ltq/report.{json,md}` once filed. Never committed: fold fits, read
records.

## 9. Results (filed 4 November 2026)

`ltq-report-dec` wrote `outputs/ltq/report.json` and `.md` on the host at about 23:55 IST on 3 November. Nothing in
sections 1 to 8 changed after the reads.

**NO_SIGNAL (BELOW_PAIR).** The decision form is u: its unlinked T2 on musique/s1fit is 0.089, against h's 0.043.

| decisive read | unlinked T2: u | pair | needed (pair + 0.03) | linked T2: u | needed (bil - 0.02) |
| --- | ---: | ---: | ---: | ---: | ---: |
| WQ-hotpotqa musique/s1fit | 0.089 | 0.144 | 0.174 | 0.342 | 0.311 |
| WQ-hotpotqa musique/s1eval | 0.086 | 0.113 | 0.143 | 0.343 | 0.326 |

- **The unlinked condition fails on both reads.** u is below pair itself, not only below pair + 0.03, and below res
  (0.128 and 0.104). It is close to bil (0.096 and 0.084).
- **The linked condition holds on both.** u keeps bil's linked T2: 0.342 against 0.331, and 0.343 against 0.346.
- Every read's bil, pair, res and rrf columns equal LTA's record ("equal" in the report).
- **Beside, deciding nothing:**
  - On 2wiki/fit, u's unlinked T2 is 0.194, against pair's 0.155.
  - On hotpotqa/s1eval it is 0.126, against pair's 0.269.
  - u's question gate puts 0.69 to 0.71 of its weight on expert A in every read.

**What follows (section 7, NO_SIGNAL).** A next-hop scorer trained on the golds, with relations and IRA's refinement,
is not what musique's unlinked next hop lacks; it needs a new input. No further next-hop scorer is declared before the
freeze. The verdict closes only this form (docs/SCREENS.md, section 3b).
