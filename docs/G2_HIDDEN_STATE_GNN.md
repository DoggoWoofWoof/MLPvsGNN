# G2: the lean GNN passes hidden states, not scores

Declared 15 October 2026, before any G2 number exists. Stage 1 of docs/PROGRAM_2026_10_15.md. Code:
`outputs/mp_unified/zg2.py`. Status: declared; results in section 7.

## 1. Why

zspk, the lean GNN, passes one number per row along an edge: the score the MLP already computed (zsp's propagation
step reads the neighbours' z-scored scores). On P3c's 36 reads it sits within 0.008 R@5 of the MLP, zrckp, in-domain on
every dataset. zg1 (SCREENS.md, the thirty-first round) stacked four layers over those score summaries and gained
nothing. gnn0, which passes hidden states over three relation-aware layers, leads zsp on 2wiki (0.891 against 0.873,
P3a). The hypothesis: **message passing gains when what travels is a representation, not a score.**

## 2. The model

**zg2** is zspk, unchanged, plus a stack whose readout is added to zspk's score.

- **Encoder state.** H is the input of zrm's output layer: the 128-wide hidden state of the network both lean models
  share, read from the same forward pass. The MLP is exactly this encoder with its output layer, so zg2 without its
  stack is zspk, and zspk without its propagation step is zrckp's network.
- **Boundary.** x = [H, z, KT]: z is zspk's score z-scored over the question's rows (detached, clipped to ±8); KT is the
  row's eight KT columns. h0 = GELU(W_in x + b_in), 64 wide.
- **Each of three layers**, over zlink's pool edges (both directions; structural, named-entity and kNN families):
  - the question's context c: its rows' states weighted by the softmax of z;
  - for an edge u → v of family f: a logit l = ⟨W_a h_u, c⟩ / 8 + β_f, clipped to ±8, and a message m = W_f h_u (one
    64 × 64 map per family);
  - two aggregations at v: per family, the mean of sigmoid(l) · m over v's edges of that family; and over all of v's
    edges, the attention average Σ exp(l) m / Σ exp(l), a soft pick of one neighbour. A row without edges gets 0;
  - the update: h ← h + GELU(layer_norm(W_s h + W_mean [the three means] + W_att att + W_x [z, KT] + b)). [z, KT]
    enters at every layer (NBFNet's boundary condition).
- **Readout.** r = w_o h + b_o, zero at the start. zg2's score = zspk's + 0.5 · tanh(r / 0.5). The bound is qd_gnn13's
  tb5: unbounded message passing on one graph hurt zero-shot reads in the MP-unified track.
- **Edge-family dropout**, in training only (qd_gnn13's fd25 rule): each question drops each family it has with
  probability 0.25, keeping its structural family (or, without one, its lowest family) when all would go. The draws
  come from the module's own generator.
- **Parameters:** 121,418 in the stack (at hidden 128), on top of zspk's. No new graph, column, encoder or text.

**zg2t, the twin,** is zg2 with no edges reaching its layers: the same parameters, context and readout. It measures what
the stack's capacity adds without message passing. It is not an MLP-track model: its context pools the question's other
rows.

The whole network trains together (zspk's weights included), with zrm's settings (rmatch.py's train: the listwise loss,
eight epochs, the weight average read as p@swa), seed 0. From the same seed both arms' forward is zspk's bit for bit
until the readout moves, and no draw of zspk's moves.

## 3. What is read

Four card fits on P1c3's carves (P1c2's pools at each question's frozen size, the KT input beside):

| arm | L-musique | L-hotpotqa |
| --- | --- | --- |
| zg2 | scr-g2-zg2 | scr-g2-zg2-hp |
| zg2t | scr-g2-zg2t | scr-g2-zg2t-hp |

Each fit is read on the six s1eval carves (p@swa). **The base, which decides:** zspk's P1c3 card fit of the same split
(scr-p1c3-zspk, scr-p1c3-zspk-hp): the same pools, KT input, card and settings. Beside it, deciding nothing: the twin
(for zg2), zsp's card fit on today's pools (scr-zspg) and step 1's fit. The pair and its re-call under the seed null are
relz's, as in every screen since the tenth round, with the seed null's floors (max(0.0075, twice the seed-only spread)).

## 4. The rule, fixed before any number

- **SCREEN_PASSED:** zg2's re-call has no LOSS on any of its twelve reads (six datasets × two fits, webqsp zero-shot
  included, and each fit's held-out dataset zero-shot) and at least one GAIN.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** the re-call is missing, is not G2's, or does not hold the twelve reads.

**Nothing is adopted on today's pools** (the program's section 2): a passed screen is fitted again on P1f's final pools,
screened there and then run in full (36 reads).

**Beside, deciding nothing:**
- zg2 minus zg2t on each read (R@5, FC@5, hit@1, with the 95% question-bootstrap interval): what message passing adds;
- the twin's own re-call against zspk: what capacity alone adds.

## 5. What it means

- **SCREEN_PASSED, and zg2 above its twin:** state-passing message passing is what the lean GNN lacked. G2 becomes the
  GNN candidate for P1f's pools, and G2r (relation text on edges) and E1 (EM over latent paths) build on its gates.
  Claim 1 then has a GNN with a real δ_MP to measure the MLP against.
- **SCREEN_PASSED, but the twin is as good:** the gain is capacity, not message passing. It is reported as such; the
  MLP may take the same capacity without edges in a declared stage of its own (the twin's context reads other rows, so
  the MLP's version would drop it).
- **NOT_ADOPTED:** reported read by read. A LOSS on webqsp zero-shot would say the stack learns the training graphs'
  structure and does not transfer; G2r's relation text is the next lever there.

## 6. Running it

- **Selftest** (`python outputs/mp_unified/zg2.py --selftest`, passed before this commit):
  - the arms registered on P1c3's link carve, with P1c3's, P1c2's and P1c3p's arms unchanged;
  - the family dropout's rule by hand;
  - at the start both arms equal zspk bit for bit, in eval and in training, and the global generator is unmoved;
  - with the stack set, each arm equals a by-hand loop over the edges (the twin with none), in training with the
    dropout's own draws; a batch without the edges or the KT input is zspk's own;
  - toy fits move the stack and repeat bit for bit;
  - train and read refuse another arm, split, fit name, cache root or device, and a fit that is not G2's;
  - relz's pair and re-call against zspk's card fits, stamped G2's, relz restored after;
  - the verdict's cases (pass, a LOSS, no GAIN, a foreign or missing re-call, the twin's re-call, fewer reads).
- **No development run on real data.** P1c2's caches and P1c3's KT files are on the host only.
- **Host items** (`g2-*`):
  - two compares that make zspk's fits the null's base (zspk's fit against step 1's: `outputs/g2/base-zspk-*`);
  - the four fits on the card (`--frac 0.45`, share 0.47, two at a time; zspk's fits peaked at 5.2 GB, and the stack
    adds per-edge 64-wide messages); the hotpotqa fits first;
  - four reads after their fits; then the compares (the twin's first), two pairs, two re-calls and the verdict.
- **Disk:** under 0.2 GB.
- **ETA:** zspk's hotpotqa fit took 57 minutes; the stack's per-edge work may double that. The verdict is about four
  hours after the fits start.
- The encoder stays frozen. webqsp never trains. Test splits are never read.

## 7. Results

Not run yet.
