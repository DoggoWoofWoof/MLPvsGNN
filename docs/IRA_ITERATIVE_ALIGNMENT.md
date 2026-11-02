# IRA: the GNN of record with IsoNet++'s refined alignment: four hop slots aligned to the pool's rows between its two rounds, on PKO's pools

Declared 2 November 2026, before any IRA fit, before DEP's verdict and before RTR's. Stage module:
`outputs/mp_unified/zira.py`. It imports `zml1s.py`, `ziso.py`, `zdep.py`, `zrtr.py`, `zg6.py` and `zg6r.py` and edits none
of them.

## 1. The question

- **What the diagnosis says.** Pool size is not the limit: the knee (docs/PKO_DEPTH_KNEE.md, stage 3) found R@5 flat from 500
  rows to the full pool on four datasets for zrckp, and on musique for zg6r. The golds the models miss are in the pool and
  rank below rows already found; most of musique's misses are on its 3- and 4-hop questions (docs/MUSIQUE_DIAGNOSIS.md). LTW
  (docs/LTW_LEARNED_NEXT_HOP.md) found a learned next-hop signal one step from a found gold (SIGNAL), and its label-free re-rank
  of rrf's top five, one step from rrf's top-1, gained on 2-hop questions and lost on 3- and 4-hop ones. So the signal holds for
  one step from a right anchor, and nothing yet chains it.
- **What IsoNet++ does** (Ramachandran, Raj, Roy, Chakrabarti and De, NeurIPS 2024, arXiv 2510.22538; after IsoNet, Roy et al.,
  AAAI 2022). It scores a query graph against a corpus graph through an explicit injective alignment between their nodes: a
  doubly stochastic plan from Sinkhorn normalisation. Three things are new:
  - **Early interaction.** Each node's message passing reads its partner in the other graph through the plan.
  - **Lazy rounds.** Each round runs the GNN's layers afresh with the plan held at the last round's estimate; after the round,
    the plan is re-estimated from the last layer's states.
  - **Node-pair partners.** A node also hears its neighbours' partners.

  The plan is trained only through the ranking loss. In their ablations, lazy rounds beat updating the plan at every layer, and
  node-pair partners beat node partners.
- **The mapping.**

  | IsoNet++ | here |
  |---|---|
  | query graph | the question's hops (latent: nothing decomposes the question) |
  | corpus graph | the question's pool and its typed links |
  | the plan | which row holds which hop |

  RTR's round two (docs/RTR_GO_BACK_ROUNDS.md) is the one-slot case: one softmax over the question's rows from round one's
  score. A softmax puts its mass on the rows round one already ranks first, most often the first hop. An injective plan over K
  slots must put its K slots on K different rows.
- **The form.** RTR's two rounds, with a plan of K = 4 slots over the question's rows between them. The plan enters round two
  in two places:
  - **Each row's start.** The slots a row holds are added as learned vectors, so its neighbours' messages carry them: the
    node-pair partner, through zg6r's own layers.
  - **The question's context.** The slots' mass is mixed with RTR's context by a learned weight that starts at 0.
- **So the question is:** do four aligned slots between the rounds raise R@5 on the twelve reads and on musique's 3-hop and
  4-hop questions, against the GNN of record?

## 2. What runs

**The arm: the GNN of record after ISO, with RTR's two rounds and IRA's plan between them.**

**Round one** is zg6r's stack, op for op (RTR's round one): h0, the context weights from s0, three layers, s1.

**The plan**, from round one's last states h1 and score s1:
- z is s1's z-score over the question's rows, clipped at zg6's ±ZCLIP (RTR's z).
- **Logits.** Slot k's logit for row i is `ira_bz · z_i + ira_a[k] · h1_i / 8`, with 8 = √64. The dustbin's logit is
  `ira_b0`.
- **Sinkhorn**, in log space, in float32, per question. It runs fifty iterations, each a column step followed by a row step:
  - each row's mass over the four slots and the dustbin is 1;
  - each slot's mass over the question's rows is t = min(1, n/5), n being the question's rows;
  - the dustbin's mass is n − 4t.

  No Gumbel noise is drawn: the plan is the same in training and at reading.
- **Why fifty.** A probe before any fit used 2,000-row questions with Gumbel round-one scores and the slots alike:
  - Ten iterations hold the slots' marginals within 0.044 up to `ira_bz` = 4, but miss by up to 0.74 at 8.
  - Fifty hold within 0.009 up to 16.
  - Sharper than that, the plan drifts toward the softmax it replaces.

  Fifty iterations cost a few MB of activations each at the read cap's 32,768 rows.
- **P** is the four slot columns, N × 4.

**Round two:**
- **Start:** `hr = h0 + z · ira_e + P · ira_E`.
- **Context:** `w2 = w_r + λ (w_s − w_r)`.
  - w_r is the softmax of z over the question's rows (RTR's context).
  - w_s is each row's slot mass Σ_k P[i, k], normalised over the question's rows.
  - λ is `ira_lam` clipped to [0, 1].
- The same three layers run with w2 as the question's context. The score is `s2 = W_out h + W_skip hr + b`, in training and at
  reading. The family dropout's one draw holds for both rounds, and the layers are checkpointed in training, as zg6r's are.

**Parameters**, 579 in all:

| name | shape | start |
|---|---|---|
| `ira_e` | 64 | 0 |
| `ira_E` | 4 × 64 | 0 |
| `ira_a` | 4 × 64 | standard normal, from its own generator at the fit's seed + 7919; no draw of zg6r's moves |
| `ira_bz` | 1 | 1 |
| `ira_b0` | 1 | 0 |
| `ira_lam` | 1 | 0 |

The message-passing stack and readout go from 180,218 to 180,797 at hidden 128.

**At the start**, `ira_e`, `ira_E` and `ira_lam` are 0, so IRA's score is RTR's at its start. The self-test checks this bit for
bit against `zrtr.rounds_stack`, with RTR's e set to IRA's.

**Where it departs from IsoNet++:**
- no Gumbel noise;
- two rounds, not their three (RTR's two, so the comparison isolates the plan);
- the plan aligns rows to latent slots, not to a second graph's nodes;
- the loss is the base's listwise loss, not their hinge.

**How it is set.** For the length of an IRA fit, zira patches two names in-process:
- zg6r's `g6r_stack`, which zg6r's forward calls by name;
- ZG6R's `__init__`, which adds IRA's parameters after zg6r's own.

It also patches zml1s's names, stamp key, stage and declaration, as RTR does. Every patched name is put back afterwards, also
on a failure. Any later read of these fits must run under the same patch: a plain zg6r refuses the state, because the strict
load fails on IRA's names.

**Everything else is the base's:** the trainer, the card and its epochs, PKO's roots and typed links, the KT input, the 64-wide
state, the seed offsets, and the read cap of 32,768.

## 3. The base, and when IRA runs

- **The base** comes from ISO's filed verdict, as DEP and RTR read it (`zdep.iso_call`, `zdep.base_fit`):
  - **KEEP:** ML1's fits `scr-ml1-zg6r-s<k>` (with `-hp` on WQ-hotpotqa). IRA starts from the same zspk seed-0 encoder.
  - **REPLACE:** ISO's fits `scr-iso-zg6rm-s<k>`. IRA takes its own draw, through ISO's own-start patches.
- **When it runs:** under RTR's condition, when DEP's filed calls keep three layers (`zrtr.dep_call`).
  - If DEP's screen is ADOPT_FOR_FULL_RUN, IRA's queued items drop with their opening gate. If DEP's full run is then
    NOT_ADOPTED, the same items are queued again, in an amendment that adds items only.
  - IRA does not wait for RTR's verdict. RTR's screen and IRA's run side by side, and RTR's is read beside IRA's (section 5).

## 4. The rule: DEP's, as RTR calls it

- **Screen:** seed 0 on both WQ splits against the base's seed-0 fits. Sixteen reads: the twelve, plus musique's 3-hop and
  4-hop slices under both splits. The floors come from the spread of the base's three seeds, and `zdep.decide_screen` decides.
- **Full run:** on ADOPT_FOR_FULL_RUN, seeds 1 and 2 on both splits, decided by `zdep.decide_full`.
- **The checks that make a read INCOMPLETE** are RTR's, plus the slots:
  - each fit's rounds, slots, layers, base and ISO verdict;
  - DEP's call;
  - the encoder's start;
  - the device;
  - the trainer's code against the base fit's.

## 5. Beside, deciding nothing

- **RTR's screen** against the same base (outputs/rtr/calls.json), when filed: the one-slot form.
- **The fitted plan's numbers** from each IRA fit's `models.pt` (`p@swa`), read on the host: λ, `ira_bz`, `ira_b0`, and the norms
  of `ira_E`, `ira_e` and `ira_a`. These show whether the fit used the slots' context at all.

## 6. What follows

| IRA's calls | what follows |
|---|---|
| screen ADOPT_FOR_FULL_RUN | seeds 1 and 2 on both splits (queued behind `ira-gate`), then the full run's calls |
| full run ADOPT | zg6r with IRA's plan is the GNN of record; the next form is three rounds (IsoNet++'s default) |
| NO_GAIN or NOT_ADOPTED (screen or full run) | The form as declared closes. The next form, in its own file, follows the beside numbers (calls.json `next_form_by_rule`). (a) If every screen fit leaves λ at 0, the slots' context went unused, and the next form adds IsoNet++'s coverage hinge as a second loss, so each slot must hold a gold row. (b) Otherwise, it is three rounds with the plan carried between them. |

## 7. Not in this stage

- **The MLP.** The MLP's form has no message passing: slots refined over the frozen vectors of a fixed rrf beam, each row
  reading its similarity to the slots as columns. It gets its own file.
- **Pools and links.** Pool size makes no difference to the ranking (the knee), and IRA changes neither pools nor links.

## 8. The queue (host)

- **Screen** (CPU self-test, GPU fits):
  - `ira-selftest`
  - `ira-open` (after `dep-calls`: the verdict gate on DEP's screen, wanting NO_GAIN or NOT_ADOPTED)
  - `ira-k4-s0` (gpu 0.42)
  - `ira-k4-s0-hp` (gpu 0.75)
  - `ira-compare`, `ira-calls`, `ira-gate` (wanting ADOPT_FOR_FULL_RUN)
- **Full run:** `ira-k4-s1`, `ira-k4-s2`, `ira-k4-s1-hp`, `ira-k4-s2-hp`, `ira-compare-full`, `ira-calls-full`.

The screen's items sit right after RTR's screen, ahead of RTR's full run. The full run's items come after RTR's.

## 9. The self-test (laptop or host, CPU, no fit)

- **Names and pairs.**
- **The plan on toy questions:**
  - the marginals: rows exactly 1, slots within 0.01 of t;
  - questions independent;
  - rows permuted with the plan;
  - with a sharp z (`ira_bz` = 25) and the four slots alike, each of the four top rows holds at least 0.8 of a slot and the
    other 36 rows under 0.1 in all (the injective plan), where a softmax puts over 0.99 on the top row and under 0.01 on the
    next.
- **The rounds on zg6r's toy carve:**
  - at the start, IRA's score is RTR's two rounds bit for bit, with RTR's e set to IRA's;
  - zg6r's forward runs IRA's stack;
  - the score moves with `ira_E` and with λ;
  - a training forward (checkpointed) and backward gives every IRA parameter a gradient;
  - the parameter count;
  - the strict load under the patch, and its refusal without;
  - the patch's restoring.
- **The patches** and their restoring, also on a failure.
- **DEP's rule** as IRA calls it.

## Results

None yet.
